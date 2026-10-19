"""Help-centre knowledge base with keyword search, exposed as a LangChain tool."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from langchain_core.tools import BaseTool, tool

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "be",
        "can",
        "do",
        "does",
        "for",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "me",
    ]
)


def _terms(text: str) -> set[str]:
    # Prefix-truncation is a crude but effective stemmer for a small help centre:
    # "cancel", "cancelled", "cancellation" and "cancellable" all map to "cancel".
    return {w[:6] for w in _WORD.findall(text.lower()) if w not in _STOP}


@dataclass(frozen=True, slots=True)
class Article:
    id: str
    title: str
    body: str


class KnowledgeBase:
    def __init__(self, articles: list[Article]) -> None:
        self.articles = articles
        self._index = [(a, _terms(f"{a.title} {a.title} {a.body}")) for a in articles]

    @classmethod
    def bundled(cls) -> KnowledgeBase:
        raw: list[dict[str, Any]] = json.loads(
            resources.files("support_agents.data").joinpath("kb.json").read_text("utf-8")
        )
        return cls([Article(**item) for item in raw])

    def search(self, query: str, k: int = 2) -> list[Article]:
        q = _terms(query)
        scored = [(len(q & terms), a.id, a) for a, terms in self._index]
        ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))
        return [a for _, _, a in ranked[:k]]


def make_kb_tools(kb: KnowledgeBase) -> list[BaseTool]:
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the Orbit Electronics help centre (returns, shipping, warranty, payments...)."""
        hits = kb.search(query)
        if not hits:
            return "No matching help-centre articles."
        return "\n\n".join(f"{a.id} - {a.title}: {a.body}" for a in hits)

    return [search_knowledge_base]
