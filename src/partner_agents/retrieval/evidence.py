"""Per-page hybrid evidence selection (HLD §6.6, ADR-007).

For one fetched page: the chunks most similar to the discovery query, plus chunks that look
like they hold an email or phone number even when they score low (a footer is rarely
"similar" to a sales query). The result is capped and returned in reading order.
"""

from partner_agents.retrieval.store import VectorStore
from partner_agents.schemas import DiscoveryIntent, RetrievedChunk

MAX_CONTACT_CHUNKS = 3


def build_query(intent: DiscoveryIntent) -> str:
    """The search text for a page: the user's words plus what we want from the page."""
    parts = [intent.partner_type, intent.product_category, intent.location or ""]
    detail = " ".join(part for part in parts if part)
    return f"{intent.raw_query} {detail} contact email phone"


async def select_evidence(
    store: VectorStore,
    intent: DiscoveryIntent,
    run_id: str,
    url: str,
    *,
    k: int,
    max_total: int,
) -> list[RetrievedChunk]:
    """Choose at most ``max_total`` chunks of ``url`` for the extractor, sorted by ``chunk_index``.

    1. The top ``k`` chunks by vector similarity.
    2. Up to ``MAX_CONTACT_CHUNKS`` more chunks flagged ``has_contact_pattern``, best
       similarity first, not already chosen.
    3. If that is more than ``max_total``, the weakest vector chunks are dropped first: the
       contact chunks are the point of the exercise.
    """
    query = build_query(intent)
    by_similarity = await store.search(query, run_id, url, k)
    chosen = {chunk.chunk_index for chunk in by_similarity}
    contact = await store.search(
        query,
        run_id,
        url,
        min(MAX_CONTACT_CHUNKS, max_total),
        contact_only=True,
        exclude=chosen,
    )
    room = max(max_total - len(contact), 0)
    kept = [*by_similarity[:room], *contact]  # by_similarity is already best first
    return sorted(kept, key=lambda chunk: chunk.chunk_index)
