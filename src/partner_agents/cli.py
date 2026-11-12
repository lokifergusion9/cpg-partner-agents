"""Command line entry point.

- ``ask "<message>"`` sends one message through the full graph: Charlie's router picks the
  branch (knowledge-base answer, discovery, lead selection, clarify or refuse).
- ``scout "<query>"`` runs Scout alone, without a router model: the query becomes a
  ``DiscoveryIntent`` through simple keyword rules, which ``--partner-type`` /
  ``--category`` / ``--location`` override. Handy for checking Scout in isolation.

Both print the reply, the leads, the errors and the time each stage took.
"""

import argparse
import asyncio
import re
import sys
import uuid
from collections.abc import Sequence
from typing import Any, get_args

from langchain_core.messages import AIMessage, HumanMessage

from partner_agents.config import Settings, get_settings
from partner_agents.graph.build import TurnResult, build_app_deps, build_graph, run_turn
from partner_agents.graph.scout import ScoutDeps, build_scout_graph
from partner_agents.graph.state import AgentState
from partner_agents.llm import get_chat_model, track_providers, track_usage
from partner_agents.observability import configure_logging, run_context
from partner_agents.outreach.email import authorize
from partner_agents.retrieval.embeddings import SentenceTransformerEmbedder
from partner_agents.retrieval.store import VectorStore, make_client
from partner_agents.schemas import (
    DiscoveryIntent,
    Lead,
    PartnerType,
    RouteDecision,
    StageError,
)
from partner_agents.storage.repo import InMemoryRepository
from partner_agents.web.fetch import Fetcher
from partner_agents.web.search import SearchService, build_providers

NODE_ORDER = (
    "charlie_route",
    "scout_search",
    "scout_fetch",
    "scout_index",
    "scout_evidence",
    "scout_extract",
    "scout_verify",
    "scout_report",
)

_PARTNER_WORDS: dict[PartnerType, tuple[str, ...]] = {
    "broker": ("broker", "brokers", "brokerage", "brokerages"),
    "distributor": ("distributor", "distributors", "wholesaler", "wholesalers"),
    "retailer": ("retailer", "retailers", "store", "stores", "shop", "shops", "grocer", "grocers"),
    "manufacturer": ("manufacturer", "manufacturers", "co-packer", "co-packers"),
    "other": (),
}
_FILLER = frozenset({"find", "me", "some", "top", "best", "the", "a", "an", "good", "local", "for"})
_LOCATION = re.compile(r"\b(?:in|near|around)\s+(.+)$", re.IGNORECASE)


def parse_query(
    text: str,
    *,
    max_results: int = 10,
    partner_type: PartnerType | None = None,
    category: str | None = None,
    location: str | None = None,
) -> DiscoveryIntent:
    """A keyword-rule stand-in for the router: ``cookie brokers in California``."""
    query = " ".join(text.split())
    where = _LOCATION.search(query)
    rest = query[: where.start()].strip() if where else query
    found: PartnerType = "other"
    category_words: list[str] = []
    seen_type = False
    for word in rest.split():
        bare = word.lower().strip(",.")
        matched = next((t for t, words in _PARTNER_WORDS.items() if bare in words), None)
        if matched and not seen_type:
            found, seen_type = matched, True
        elif not seen_type and bare not in _FILLER:
            category_words.append(word)
    return DiscoveryIntent(
        partner_type=partner_type or found,
        product_category=category or " ".join(category_words) or "food",
        location=location or (where.group(1).strip() if where else None),
        max_results=max_results,
        raw_query=query,
    )


def format_leads(leads: Sequence[Lead]) -> str:
    """A plain-text table of leads, verified ones first."""
    if not leads:
        return "(no leads)"
    ordered = sorted(
        leads, key=lambda lead: (not lead.outreach_eligible, lead.company_name.lower())
    )
    rows = [("#", "Company", "Type", "Email", "Phone", "Source")]
    for number, lead in enumerate(ordered, start=1):
        email = f"{lead.email} [{lead.email_status}]" if lead.email else "-"
        phone = f"{lead.phone} [{lead.phone_status}]" if lead.phone else "-"
        source = str(lead.source_url)
        rows.append(
            (
                str(number),
                lead.company_name,
                lead.partner_type or "-",
                email,
                phone,
                source if len(source) <= 48 else source[:45] + "...",
            )
        )
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def format_errors(errors: Sequence[StageError]) -> str:
    if not errors:
        return "(none)"
    return "\n".join(
        f"{error.stage:<15} {error.code:<16} {error.url or '-'}  {error.message}".rstrip()
        for error in errors
    )


def format_timings(timings: dict[str, int]) -> str:
    ordered = [name for name in NODE_ORDER if name in timings]
    ordered += sorted(set(timings) - set(ordered))
    lines = [f"{name:<15} {timings[name]:>7} ms" for name in ordered]
    lines.append(f"{'total':<15} {sum(timings.values()):>7} ms")
    return "\n".join(lines)


def build_deps(settings: Settings) -> ScoutDeps:
    repo = InMemoryRepository()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return ScoutDeps(
        settings=settings,
        repo=repo,
        search=SearchService(build_providers(settings), repo, settings.effective_search_blocklist),
        fetcher=Fetcher(settings, repo),
        store=VectorStore(make_client(settings), embedder),
        model=get_chat_model("extractor", settings=settings),
    )


async def run_scout(query: str, intent: DiscoveryIntent, deps: ScoutDeps) -> dict[str, Any]:
    """Run the Scout subgraph once and return its final state plus tokens and providers."""
    run_id = f"r_{uuid.uuid4().hex[:8]}"
    graph = build_scout_graph(deps)
    state = AgentState(
        messages=[HumanMessage(content=query)],
        user_id="cli",
        thread_id=f"cli_{run_id}",
        run_id=run_id,
        route=RouteDecision(intent="discovery", confidence=1.0, discovery=intent, reason="cli"),
    )
    with run_context(run_id), track_usage() as tokens, track_providers() as providers:
        final: dict[str, Any] = await graph.ainvoke(state)
    final["tokens"] = tokens
    final["providers"] = providers
    return final


def render(final: dict[str, Any]) -> str:
    reply = next(
        (m.content for m in reversed(final.get("messages", [])) if isinstance(m, AIMessage)), ""
    )
    sections = [
        str(reply),
        format_leads(final.get("leads", [])),
        "Errors:\n" + format_errors(final.get("errors", [])),
        "Timings:\n" + format_timings(final.get("timings_ms", {})),
    ]
    if final.get("tokens"):
        used = ", ".join(f"{role}: {n}" for role, n in sorted(final["tokens"].items()))
        sections.append(f"Tokens: {used}")
    if final.get("providers"):
        sections.append(
            "Models: " + ", ".join(f"{r}={p}" for r, p in sorted(final["providers"].items()))
        )
    return "\n\n".join(sections)


async def _scout(args: argparse.Namespace, settings: Settings) -> str:
    intent = parse_query(
        args.query,
        max_results=args.max_results,
        partner_type=args.partner_type,
        category=args.category,
        location=args.location,
    )
    deps = build_deps(settings)
    try:
        final = await run_scout(args.query, intent, deps)
    finally:
        await deps.fetcher.aclose()
        await deps.store.close()
    return render(final)


def render_turn(result: TurnResult) -> str:
    """One ``ask`` turn: the reply, the leads of a discovery, errors, timings and tokens."""
    run = result.run
    sections = [result.reply]
    if run.intent == "discovery":
        sections.append(format_leads(result.state.get("leads", [])))
    sections += [
        f"Intent: {run.intent}  (run {run.run_id})",
        "Errors:\n" + format_errors(run.errors),
        "Timings:\n" + format_timings(run.timings_ms),
    ]
    if run.tokens:
        sections.append("Tokens: " + ", ".join(f"{r}: {n}" for r, n in sorted(run.tokens.items())))
    if run.providers:
        sections.append(
            "Models: " + ", ".join(f"{r}={p}" for r, p in sorted(run.providers.items()))
        )
    return "\n\n".join(sections)


async def _ask(args: argparse.Namespace, settings: Settings) -> str:
    deps = build_app_deps(settings)
    graph = build_graph(deps)
    try:
        result = await run_turn(graph, deps, thread_id="cli", user_id="cli", text=args.message)
    finally:
        await deps.aclose()
    return render_turn(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="partner_agents.cli", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="one message through Charlie's router and its branch")
    ask.add_argument("message", help='for example "what is a slotting fee?"')
    commands.add_parser("gmail-auth", help="authorise Gmail drafts once (opens a browser)")
    scout = commands.add_parser("scout", help="find partners and verify their contacts")
    scout.add_argument("query", help='for example "cookie brokers in California"')
    scout.add_argument("--max-results", type=int, default=10, choices=range(1, 26), metavar="1-25")
    scout.add_argument("--partner-type", choices=get_args(PartnerType))
    scout.add_argument("--category", help="product category, overrides the parsed one")
    scout.add_argument("--location", help="location, overrides the parsed one")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    if args.command == "gmail-auth":
        print(f"Gmail authorised. Token saved to {authorize(settings)}")
        return 0
    command = _ask if args.command == "ask" else _scout
    print(asyncio.run(command(args, settings)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
