from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from support_agents.api import create_app
from support_agents.config import Settings
from support_agents.service import AgentService, ThreadBusyError


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        checkpoint_db=tmp_path / "db" / "checkpoints.sqlite",
        today=date(2026, 10, 19),
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(tmp_path))) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "version": "0.1.0",
        "llm_provider": "demo",
    }


def test_full_thread_lifecycle(client: TestClient) -> None:
    thread_id = client.post("/v1/threads").json()["thread_id"]
    r = client.post(
        f"/v1/threads/{thread_id}/runs",
        json={"question": "Can I return my earbuds from order ORD-1001?", "customer_id": "CUST-1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved"
    assert "18 days ago" in body["answer"]
    assert {s["worker"] for s in body["plan"]} == {"account", "knowledge"}
    assert body["findings"][0]["tool_calls"][0]["tool"] == "get_order"

    client.post(f"/v1/threads/{thread_id}/runs", json={"question": "How long is shipping?"})
    state = client.get(f"/v1/threads/{thread_id}").json()
    assert len(state["conversation"]) == 2
    assert "business days" in state["last_run"]["answer"]

    history = client.get(f"/v1/threads/{thread_id}/history", params={"limit": 5}).json()
    assert len(history) == 5
    assert history[0]["step"] > history[-1]["step"]


def test_stream_emits_node_events_then_done(client: TestClient) -> None:
    with client.stream(
        "POST", "/v1/threads/s1/runs/stream", json={"question": "warranty for ORD-1003?"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [block.split("\n") for block in raw.strip().split("\n\n")]
    names = [lines[0].removeprefix("event: ") for lines in events]
    assert names[0] == "planner"
    assert names.count("worker") == 2
    assert names[-3:] == ["reviewer", "finalize", "done"]
    done = json.loads(events[-1][1].removeprefix("data: "))
    assert done["status"] == "approved"
    assert "12-month" in done["answer"]


def test_unknown_thread_returns_404(client: TestClient) -> None:
    assert client.get("/v1/threads/nope").status_code == 404
    assert client.get("/v1/threads/nope/history").status_code == 404


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/threads/t/runs", {"question": ""}),
        ("/v1/threads/t/runs", {}),
        ("/v1/threads/bad%20id/runs", {"question": "hi"}),
    ],
)
def test_validation(client: TestClient, path: str, payload: dict[str, str]) -> None:
    assert client.post(path, json=payload).status_code == 422


def test_api_key_required_when_configured(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, api_key=SecretStr("s3cret")))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # health stays public
        assert c.post("/v1/threads").status_code == 401
        assert c.post("/v1/threads", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/v1/threads", headers={"X-API-Key": "s3cret"}).status_code == 201


def test_openapi_lists_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/threads/{thread_id}/runs/stream" in paths


async def test_concurrent_run_on_same_thread_is_rejected() -> None:
    service = AgentService(graph=None)  # type: ignore[arg-type]
    lock = service._lock("busy")
    async with lock:
        with pytest.raises(ThreadBusyError):
            await service.run("busy", "q", None)
    await asyncio.sleep(0)


def test_busy_thread_maps_to_409(client: TestClient) -> None:
    service: AgentService = client.app.state.service  # type: ignore[attr-defined]

    async def hold() -> None:
        lock = service._lock("held")
        await lock.acquire()

    client.portal.call(hold)  # type: ignore[union-attr]
    r = client.post("/v1/threads/held/runs", json={"question": "hi"})
    assert r.status_code == 409
    r = client.post("/v1/threads/held/runs/stream", json={"question": "hi"})
    assert r.status_code == 409
