import importlib

import pytest

import partner_agents

SUBPACKAGES = [
    "prompts",
    "graph",
    "web",
    "retrieval",
    "verify",
    "outreach",
    "kb",
    "storage",
    "api",
    "ui",
    "utils",
]


def test_package_has_version() -> None:
    assert partner_agents.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"partner_agents.{name}")
