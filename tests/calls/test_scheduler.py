"""Meeting proposals and partner time parsing with a fixed clock (HLD §6.11, T-56)."""

from datetime import UTC, datetime

import pytest

from partner_agents.calls.scheduler import is_bookable, parse_time, propose_slot, zone

LA = "America/Los_Angeles"  # PST (UTC-8) all through these dates: DST ended on 2026-11-01


def la(day: int, hour: int, minute: int = 0) -> datetime:
    """A November 2026 time in Los Angeles. The 13th is a Friday, the 16th a Monday."""
    return datetime(2026, 11, day, hour, minute, tzinfo=zone(LA))


# --- propose_slot -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        pytest.param(la(13, 11), la(16, 10), id="friday-goes-to-monday"),
        pytest.param(la(16, 16, 30), la(18, 10), id="1630-next-day-17-not-allowed"),
        pytest.param(la(16, 9, 15), la(17, 10), id="rounds-up-to-the-hour"),
        pytest.param(la(16, 10), la(17, 10), id="exactly-on-the-hour"),
        pytest.param(la(16, 14, 20), la(17, 15), id="afternoon"),
        pytest.param(la(19, 16), la(20, 16), id="last-start-of-the-day"),
        pytest.param(la(20, 16, 1), la(23, 10), id="friday-after-four"),
        pytest.param(la(14, 12), la(16, 10), id="saturday-call"),
        pytest.param(la(16, 6), la(17, 10), id="early-morning"),
    ],
)
def test_propose_slot(now: datetime, expected: datetime) -> None:
    slot = propose_slot(now, LA)
    assert slot == expected
    assert (slot - now).total_seconds() >= 24 * 3600
    assert is_bookable(slot, LA)


def test_the_proposal_is_in_brand_time_for_a_utc_clock() -> None:
    now = datetime(2026, 11, 16, 18, 0, tzinfo=UTC)  # 10:00 in Los Angeles
    slot = propose_slot(now, LA)
    assert slot == datetime(2026, 11, 17, 18, 0, tzinfo=UTC)  # Tue 10:00 LA
    assert slot.tzinfo == zone(LA) and slot.hour == 10
    # Friday 15:30 in LA is already Friday 23:30 UTC: still the Friday rule.
    assert propose_slot(datetime(2026, 11, 13, 23, 30, tzinfo=UTC), LA) == la(16, 10)


def test_an_unknown_timezone_falls_back_to_utc() -> None:
    slot = propose_slot(datetime(2026, 11, 16, 9, 0, tzinfo=UTC), "Mars/Olympus_Mons")
    assert slot == datetime(2026, 11, 17, 10, 0, tzinfo=UTC)


# --- parse_time ---------------------------------------------------------------------

NOW = la(16, 11)  # Monday 11:00


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tomorrow 3pm", la(17, 15)),
        ("Thursday at 11", la(19, 11)),
        ("next Monday morning", la(23, 10)),
        ("Monday at 2", la(23, 14)),  # today's weekday means next week
        ("today at 4:30 pm", la(16, 16, 30)),
        ("How about tomorrow at 10.30am?", la(17, 10, 30)),
        ("tomorrow at noon works", la(17, 12)),
        ("Wednesday afternoon", la(18, 14)),
        ("3 p.m. on Friday", la(20, 15)),
        ("Tomorrow at 14:00", la(17, 14)),
    ],
)
def test_parse_time_reads_a_day_and_a_time(text: str, expected: datetime) -> None:
    assert parse_time(text, NOW, LA) == expected


@pytest.mark.parametrize(
    "text",
    [
        "later",
        "sometime",
        "sometime next week",
        "maybe tomorrow",  # a day without a time
        "at 3pm",  # a time without a day
        "today at 9am",  # already past
        "Thursday at 25",
        "tomorrow, we have 15 stores",
        "call me later this week",
        "",
    ],
)
def test_vague_or_impossible_times_are_none(text: str) -> None:
    assert parse_time(text, NOW, LA) is None


def test_parse_time_reads_the_day_in_brand_time() -> None:
    # 05:00 UTC on Tuesday is still Monday evening in Los Angeles, so "tomorrow" is Tuesday.
    now = datetime(2026, 11, 17, 5, 0, tzinfo=UTC)
    slot = parse_time("tomorrow 3pm", now, LA)
    assert slot == la(17, 15) == datetime(2026, 11, 17, 23, 0, tzinfo=UTC)


# --- is_bookable --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slot", "bookable"),
    [
        (la(17, 10), True),
        (la(17, 16), True),
        (la(17, 16, 30), False),
        (la(17, 9, 30), False),
        (la(21, 11), False),  # Saturday
        (datetime(2026, 11, 17, 18, 0, tzinfo=UTC), True),  # 10:00 LA
    ],
)
def test_is_bookable(slot: datetime, bookable: bool) -> None:
    assert is_bookable(slot, LA) is bookable
