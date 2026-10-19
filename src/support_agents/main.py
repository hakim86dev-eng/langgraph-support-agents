"""ASGI entry point: ``uvicorn support_agents.main:app``."""

from support_agents.api import create_app

app = create_app()
