"""Create the chat model for the configured provider (imports are lazy and optional)."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from support_agents.config import Settings
from support_agents.llm.demo import demo_responder
from support_agents.llm.fake import FakeToolCallingChatModel


def create_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            model: BaseChatModel = ChatOpenAI(
                model=settings.model, temperature=settings.temperature
            )
            return model
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(model_name=settings.model, temperature=settings.temperature)
            return model
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            model = ChatOllama(
                model=settings.model,
                temperature=settings.temperature,
                base_url=settings.ollama_base_url,
            )
            return model
    except ImportError as exc:
        raise RuntimeError(
            f"provider '{provider}' needs an optional dependency: "
            f"pip install 'langgraph-support-agents[{provider}]'"
        ) from exc
    return FakeToolCallingChatModel(responder=demo_responder)
