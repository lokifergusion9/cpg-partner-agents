"""Scout's graph nodes (HLD §5.2, §6). Dependencies are passed in by ``make_*`` factories so
that nodes stay plain ``state -> update`` functions that tests can call directly.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence

from langchain_core.language_models import BaseChatModel

from partner_agents.graph.state import AgentState
from partner_agents.llm import StructuredOutputError, invoke_structured
from partner_agents.observability import NodeUpdate, redact_contacts, timed_node
from partner_agents.prompts.scout import PROMPT_VERSION, build_extraction_messages
from partner_agents.schemas import (
    DiscoveryIntent,
    ExtractedLead,
    ExtractionResult,
    RetrievedChunk,
    StageError,
)

logger = logging.getLogger(__name__)

Node = Callable[[AgentState], Awaitable[NodeUpdate]]


async def extract_candidates(
    model: BaseChatModel,
    intent: DiscoveryIntent,
    evidence: Mapping[str, Sequence[RetrievedChunk]],
) -> tuple[dict[str, list[ExtractedLead]], list[StageError]]:
    """One extraction call per page, run together (``LLM_MAX_CONCURRENCY`` bounds the calls).

    A page that fails becomes a ``StageError`` and the others carry on. A page with no
    matching company gives an empty list, which is a valid answer.
    """

    async def one(url: str, chunks: Sequence[RetrievedChunk]) -> list[ExtractedLead] | StageError:
        try:
            result = await invoke_structured(
                model,
                ExtractionResult,
                build_extraction_messages(intent, url, chunks),
                role="extractor",
                prompt_version=PROMPT_VERSION,
            )
        except StructuredOutputError as exc:
            return StageError(
                stage="scout_extract", code="invalid_output", message=exc.detail, url=url
            )
        except Exception as exc:
            logger.warning(
                "extraction call failed", extra={"url": url, "error": type(exc).__name__}
            )
            return StageError(
                stage="scout_extract",
                code="model_error",
                message=f"{type(exc).__name__}: {redact_contacts(str(exc))[:200]}",
                url=url,
            )
        return result.leads

    pages = [(url, chunks) for url, chunks in evidence.items() if chunks]
    outcomes = await asyncio.gather(*(one(url, chunks) for url, chunks in pages))
    candidates: dict[str, list[ExtractedLead]] = {}
    errors: list[StageError] = []
    for (url, _), outcome in zip(pages, outcomes, strict=True):
        if isinstance(outcome, StageError):
            errors.append(outcome)
        else:
            candidates[url] = outcome
    return candidates, errors


def make_scout_extract(model: BaseChatModel) -> Node:
    """The ``scout_extract`` node: ``evidence`` + the discovery intent -> ``candidates``."""

    @timed_node("scout_extract")
    async def scout_extract(state: AgentState) -> NodeUpdate:
        route = state.get("route")
        if route is None or route.discovery is None:
            raise ValueError("scout_extract needs state['route'].discovery")
        candidates, errors = await extract_candidates(
            model, route.discovery, state.get("evidence", {})
        )
        return {"candidates": candidates, "errors": errors}

    return scout_extract
