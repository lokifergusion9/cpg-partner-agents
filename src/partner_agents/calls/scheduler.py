"""Meeting times for the handoff stage (HLD §6.11). Pure: the call time is always passed in.

``propose_slot`` gives the default proposal: the first full hour at least 24 hours after the
call, on a weekday, starting between 10:00 and 16:00 brand time (so it ends by 17:00).
``parse_time`` reads a partner's "tomorrow 3pm", "Thursday at 11" or "next Monday morning"
with a small rule set and no new dependency. A time it cannot pin down ("later", "sometime",
a day without a time) is ``None``, and the call asks once more.
"""

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

FIRST_HOUR = 10  # earliest meeting start, brand time
LAST_HOUR = 16  # latest meeting start: a one-hour meeting ends by 17:00
MIN_LEAD = timedelta(hours=24)

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_PARTS_OF_DAY = {"morning": 10, "noon": 12, "midday": 12, "lunchtime": 12, "afternoon": 14}

_DAY = re.compile(r"\b(today|tomorrow|" + "|".join(_WEEKDAYS) + r")\b")
_CLOCK = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?(?!\s*(?:%|\d))")
_PART = re.compile(r"\b(" + "|".join(_PARTS_OF_DAY) + r")\b")


def zone(name: str) -> ZoneInfo:
    """The brand's timezone; an unknown name falls back to UTC rather than failing the call."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def is_bookable(slot: datetime, tz: str) -> bool:
    """A weekday start between 10:00 and 16:00 brand time (so the meeting ends by 17:00)."""
    local = slot.astimezone(zone(tz))
    return local.weekday() < 5 and time(FIRST_HOUR) <= local.time() <= time(LAST_HOUR)


def propose_slot(now: datetime, tz: str) -> datetime:
    """The default proposal (HLD §6.11), as an aware datetime in the brand's timezone."""
    earliest = (now + MIN_LEAD).astimezone(zone(tz))
    slot = earliest.replace(minute=0, second=0, microsecond=0)
    if slot < earliest:
        slot += timedelta(hours=1)
    while not (slot.weekday() < 5 and FIRST_HOUR <= slot.hour <= LAST_HOUR):
        if slot.hour < FIRST_HOUR:
            slot = slot.replace(hour=FIRST_HOUR)
        else:
            slot = (slot + timedelta(days=1)).replace(hour=FIRST_HOUR)
    return slot


def parse_time(text: str, now: datetime, tz: str) -> datetime | None:
    """A partner's proposed time, read relative to the call time in brand time, or ``None``.

    It needs a day (today, tomorrow or a weekday name; a bare or "next" weekday means its
    next occurrence after today) and a time of day (a clock time or morning / noon /
    afternoon). A bare hour from 1 to 7 is read as pm. A time in the past is ``None``.
    """
    lowered = text.lower()
    local_now = now.astimezone(zone(tz))
    day_match = _DAY.search(lowered)
    if day_match is None:
        return None
    clock = _clock_time(lowered)
    if clock is None:
        part = _PART.search(lowered)
        if part is None:
            return None
        clock = time(_PARTS_OF_DAY[part.group(1)])

    word = day_match.group(1)
    if word == "today":
        offset = 0
    elif word == "tomorrow":
        offset = 1
    else:
        offset = (_WEEKDAYS.index(word) - local_now.weekday()) % 7 or 7
    day = local_now.date() + timedelta(days=offset)
    slot = datetime.combine(day, clock, tzinfo=zone(tz))
    return slot if slot > local_now else None


def _clock_time(text: str) -> time | None:
    for match in _CLOCK.finditer(text):
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        meridiem = (match.group(3) or "").replace(".", "")
        if minute > 59 or hour > 23 or (meridiem and not 1 <= hour <= 12):
            continue
        if not meridiem and match.group(2) is None and not 1 <= hour <= 12:
            continue  # a bare "15" or "0" is more likely a count than a time
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif not meridiem and 1 <= hour <= 7:
            hour += 12  # "at 3" during business hours means 3 pm
        return time(hour, minute)
    return None
