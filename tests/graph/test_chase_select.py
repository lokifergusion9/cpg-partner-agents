"""``resolve_selector`` is deterministic and table-driven (HLD §6.9)."""

from datetime import UTC, datetime

import pytest

from partner_agents.graph.chase import resolve_selector
from partner_agents.schemas import Lead


def lead(name: str, location: str | None, partner_type: str = "broker") -> Lead:
    return Lead(
        id=name.lower().split()[0],
        run_id="r",
        user_id="u",
        company_name=name,
        partner_type=partner_type,  # type: ignore[arg-type]
        location=location,
        website=None,
        email=None,
        email_status="unchecked",
        phone=None,
        phone_status="unchecked",
        source_url="https://a.example/",  # type: ignore[arg-type]
        evidence_quote="q",
        created_at=datetime.now(UTC),
    )


LEADS = [
    lead("Acme Foods", "Los Angeles, CA"),
    lead("Beta Brokerage", "San Diego, CA"),
    lead("Gamma Distribution", "Los Angeles, CA", "distributor"),
    lead("Delta Reps", None),
]


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("top 3", ["acme", "beta", "gamma"]),
        ("the first two", ["acme", "beta"]),
        ("top 10", ["acme", "beta", "gamma", "delta"]),
        ("the first one", ["acme"]),
        ("the second one", ["beta"]),
        ("the last one", ["delta"]),
        ("lead 3", ["gamma"]),
        ("#2", ["beta"]),
        ("lead 9", []),
        ("the LA ones", ["acme", "gamma"]),
        ("only the ones in LA", ["acme", "gamma"]),
        ("the ones in Los Angeles", ["acme", "gamma"]),
        ("the LA broker", ["acme"]),
        ("San Diego", ["beta"]),
        ("Gamma", ["gamma"]),
        ("all of them", ["acme", "beta", "gamma", "delta"]),
        ("the ones in Texas", []),
        ("the ones", None),
        ("", None),
        (None, None),
    ],
)
def test_resolve_selector(selector: str | None, expected: list[str] | None) -> None:
    chosen = resolve_selector(LEADS, selector)
    assert (None if chosen is None else [lead.id for lead in chosen]) == expected
