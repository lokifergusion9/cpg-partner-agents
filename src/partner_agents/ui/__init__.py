"""The Gradio UI (ADR-010)."""

import os

# Gradio sends usage analytics and a version check to api.gradio.app by default. Off, unless the
# operator turned it on explicitly.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
