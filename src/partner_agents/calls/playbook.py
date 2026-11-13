"""Load and validate a call playbook from ``data/playbooks/<id>.toml`` (HLD §6.11, ADR-017).

The file holds stage goals, must / never rules with stable ids, the qualification slots,
banned phrases and one fallback line per stage. Validation is strict, because the stage
machine and the reply checks rely on it: the six stages in order, unique rule ids, and every
``check = "code"`` rule mapped to a known check id.
"""

import re
import string
import tomllib
from pathlib import Path

from pydantic import ValidationError

from partner_agents.schemas import CallPlaybook, CallStage, PlaybookRule

# data/playbooks at the repository root (this file is src/partner_agents/calls/playbook.py).
DEFAULT_PLAYBOOK_DIR = Path(__file__).resolve().parents[3] / "data" / "playbooks"

STAGE_ORDER: tuple[CallStage, ...] = (
    "opening",
    "discovery",
    "qualification",
    "pitch",
    "handoff",
    "closing",
)

# The reply checks in calls/checks.py (HLD §6.11), plus C-FLOW: a rule that the stage machine
# (calls/stages.py) enforces in code rather than a check on the reply text.
CHECK_IDS = frozenset(
    {"C-NUM", "C-BAN", "C-ONEQ", "C-LEN", "C-AI", "C-PII", "C-STAGE", "C-SIGNOFF", "C-FLOW"}
)

# Brand values a stage goal or fallback line may use; calls/ fills them from the BrandProfile.
BRAND_FIELDS = frozenset({"brand_name", "founder_name", "category", "sign_off"})

_PLAYBOOK_ID = re.compile(r"^[a-z0-9_]{1,64}$")


class PlaybookError(ValueError):
    """The playbook file is missing, is not valid TOML or breaks a playbook rule."""


def load_playbook(playbook_id: str, directory: Path = DEFAULT_PLAYBOOK_DIR) -> CallPlaybook:
    if not _PLAYBOOK_ID.match(playbook_id):
        raise PlaybookError(f"playbook id {playbook_id!r} must match [a-z0-9_]{{1,64}}")
    path = directory / f"{playbook_id}.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlaybookError(f"no playbook file {path.name} in {directory}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PlaybookError(f"{path.name} is not valid TOML: {exc}") from exc
    playbook = parse_playbook(data)
    if playbook.id != playbook_id:
        raise PlaybookError(f"{path.name} declares id {playbook.id!r}")
    return playbook


def parse_playbook(data: dict[str, object]) -> CallPlaybook:
    try:
        playbook = CallPlaybook.model_validate(data)
    except ValidationError as exc:
        raise PlaybookError(f"invalid playbook: {exc}") from exc
    problems = playbook_problems(playbook)
    if problems:
        raise PlaybookError("invalid playbook: " + "; ".join(problems))
    return playbook


def playbook_problems(playbook: CallPlaybook) -> list[str]:
    """Every broken rule, as short messages. Empty means the playbook is valid."""
    problems: list[str] = []
    names = tuple(stage.name for stage in playbook.stages)
    if names != STAGE_ORDER:
        problems.append(f"stages must be {', '.join(STAGE_ORDER)} in this order, got {names}")

    rules = list(playbook.universal_rules) + [r for s in playbook.stages for r in s.rules]
    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            problems.append(f"duplicate rule id {rule.id}")
        seen.add(rule.id)
        problems += _rule_problems(rule)

    for stage in playbook.stages:
        if stage.name != "qualification" and (stage.slots or stage.required_slots):
            problems.append(f"only qualification has slots, {stage.name} has some")
        if len(set(stage.slots)) != len(stage.slots):
            problems.append(f"{stage.name} lists a slot twice")
        extra = [slot for slot in stage.required_slots if slot not in stage.slots]
        if extra:
            problems.append(f"{stage.name} requires unknown slots {extra}")
        for label, text in (("goal", stage.goal), ("fallback line", stage.fallback_line)):
            unknown = _placeholders(text) - BRAND_FIELDS
            if unknown:
                problems.append(f"{stage.name} {label} uses unknown fields {sorted(unknown)}")
    qualification = [s for s in playbook.stages if s.name == "qualification"]
    if qualification and not qualification[0].slots:
        problems.append("qualification needs at least one slot")
    if any(not phrase.strip() for phrase in playbook.banned_phrases):
        problems.append("banned phrases must not be blank")
    return problems


def _rule_problems(rule: PlaybookRule) -> list[str]:
    if rule.check == "code":
        if rule.check_id is None:
            return [f"rule {rule.id} has check=code but no check_id"]
        if rule.check_id not in CHECK_IDS:
            return [f"rule {rule.id} has unknown check id {rule.check_id}"]
    elif rule.check_id is not None:
        return [f"rule {rule.id} has a check_id but check={rule.check}"]
    return []


def _placeholders(text: str) -> set[str]:
    try:
        return {field for _, field, _, _ in string.Formatter().parse(text) if field is not None}
    except ValueError:
        return {"<unbalanced braces>"}
