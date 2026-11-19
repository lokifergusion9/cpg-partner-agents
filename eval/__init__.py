"""Evaluation datasets, schemas, metrics and the runner (docs/EVAL.md).

Nothing in ``src/partner_agents`` imports from here (CLAUDE.md rule 9: test splits never reach
a prompt). The only dev-split rows the app uses are the few-shot copies in
``prompts/charlie.py``, and a test checks that they match.
"""
