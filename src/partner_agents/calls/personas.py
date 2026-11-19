"""Load the demo personas from ``data/personas/<id>.toml`` (HLD section 6.11).

A persona is a card the ``simulator`` role plays against the call agent. The demo personas are
written from scratch about fictional companies (CLAUDE.md rule 12). The eval personas live in
``eval/datasets/`` and are never read from here (rule 9).
"""

import re
import tomllib
from pathlib import Path

from pydantic import ValidationError

from partner_agents.schemas import Persona

# data/personas at the repository root (this file is src/partner_agents/calls/personas.py).
DEFAULT_PERSONA_DIR = Path(__file__).resolve().parents[3] / "data" / "personas"

_PERSONA_ID = re.compile(r"^[a-z0-9_]{1,64}$")


class PersonaError(ValueError):
    """The persona file is missing, is not valid TOML or breaks the persona schema."""


def load_persona(persona_id: str, directory: Path = DEFAULT_PERSONA_DIR) -> Persona:
    if not _PERSONA_ID.match(persona_id):
        raise PersonaError(f"persona id {persona_id!r} must match [a-z0-9_]{{1,64}}")
    path = directory / f"{persona_id}.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PersonaError(f"no persona file {path.name} in {directory}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PersonaError(f"{path.name} is not valid TOML: {exc}") from exc
    try:
        persona = Persona.model_validate(data)
    except ValidationError as exc:
        raise PersonaError(f"invalid persona {path.name}: {exc}") from exc
    if persona.id != persona_id:
        raise PersonaError(f"{path.name} declares id {persona.id!r}")
    return persona


def list_personas(directory: Path = DEFAULT_PERSONA_DIR) -> list[Persona]:
    """Every persona in ``directory``, by id."""
    return [load_persona(path.stem, directory) for path in sorted(directory.glob("*.toml"))]
