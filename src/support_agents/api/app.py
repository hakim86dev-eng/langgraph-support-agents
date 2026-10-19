"""FastAPI application exposing the multi-agent graph."""

import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import date
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from support_agents import __version__
from support_agents.config import Settings, get_settings
from support_agents.graph import build_graph
from support_agents.llm import create_chat_model
from support_agents.service import (
    AgentService,
    CheckpointInfo,
    RunResult,
    ThreadBusyError,
    ThreadState,
)
from support_agents.tools import Toolkit, build_toolkit

ThreadId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_\-]{1,64}$")]


def get_service(request: Request) -> AgentService:
    service: AgentService = request.app.state.service
    return service


Service = Annotated[AgentService, Depends(get_service)]


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    customer_id: str | None = Field(default=None, max_length=64)


class ThreadCreated(BaseModel):
    thread_id: str


class Health(BaseModel):
    status: str
    version: str
    llm_provider: str


def create_app(
    settings: Settings | None = None,
    *,
    llm: BaseChatModel | None = None,
    toolkit: Toolkit | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            saver = checkpointer
            if saver is None:
                settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
                saver = await stack.enter_async_context(
                    AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db))
                )
            fixed_today = settings.today
            graph = build_graph(
                llm or create_chat_model(settings),
                toolkit
                or build_toolkit(today=(lambda: fixed_today) if fixed_today else date.today),
                max_plan_steps=settings.max_plan_steps,
                max_revisions=settings.max_revisions,
                worker_max_iterations=settings.worker_max_iterations,
                checkpointer=saver,
            )
            app.state.service = AgentService(graph)
            yield

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    app = FastAPI(
        title="LangGraph Support Agents",
        version=__version__,
        description="Planner -> tool-using workers -> writer/reviewer, with checkpointed threads.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=Health, tags=["ops"])
    async def health() -> Health:
        return Health(status="ok", version=__version__, llm_provider=settings.llm_provider)

    secured = [Depends(require_api_key)]

    @app.post(
        "/v1/threads",
        response_model=ThreadCreated,
        status_code=status.HTTP_201_CREATED,
        dependencies=secured,
        tags=["threads"],
    )
    async def create_thread() -> ThreadCreated:
        return ThreadCreated(thread_id=uuid.uuid4().hex)

    @app.post(
        "/v1/threads/{thread_id}/runs",
        response_model=RunResult,
        dependencies=secured,
        tags=["runs"],
    )
    async def run(thread_id: ThreadId, body: RunRequest, service: Service) -> RunResult:
        try:
            return await service.run(thread_id, body.question, body.customer_id)
        except ThreadBusyError as exc:
            raise HTTPException(status_code=409, detail="a run is already in progress") from exc

    @app.post(
        "/v1/threads/{thread_id}/runs/stream",
        dependencies=secured,
        tags=["runs"],
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def run_stream(
        thread_id: ThreadId, body: RunRequest, service: Service
    ) -> StreamingResponse:
        try:
            events = service.stream(thread_id, body.question, body.customer_id)
            first = await anext(events)
        except ThreadBusyError as exc:
            raise HTTPException(status_code=409, detail="a run is already in progress") from exc

        async def body_iter() -> AsyncIterator[str]:
            yield first
            async for event in events:
                yield event

        return StreamingResponse(
            body_iter(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(
        "/v1/threads/{thread_id}",
        response_model=ThreadState,
        dependencies=secured,
        tags=["threads"],
    )
    async def get_thread(thread_id: ThreadId, service: Service) -> ThreadState:
        state = await service.get_thread(thread_id)
        if state is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return state

    @app.get(
        "/v1/threads/{thread_id}/history",
        response_model=list[CheckpointInfo],
        dependencies=secured,
        tags=["threads"],
    )
    async def history(
        thread_id: ThreadId, service: Service, limit: int = 50
    ) -> list[CheckpointInfo]:
        items = await service.history(thread_id, limit=min(max(limit, 1), 200))
        if not items:
            raise HTTPException(status_code=404, detail="thread not found")
        return items

    return app
