"""The FastAPI app (HLD §7). ``uvicorn partner_agents.api.main:app`` serves it at ``/api``.

``create_app`` takes ready dependencies for tests; otherwise the lifespan builds the real ones
from settings on startup, so importing this module stays cheap.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from partner_agents.api.errors import install_error_handlers
from partner_agents.api.ratelimit import RateLimiter
from partner_agents.api.routes import open_router, router
from partner_agents.config import Settings, get_settings
from partner_agents.graph.build import AppDeps, Context, build_app_deps, build_graph
from partner_agents.observability import configure_logging
from partner_agents.ui.app import build_ui


def create_app(
    deps: AppDeps | None = None, settings: Settings | None = None, *, with_ui: bool = True
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = deps
        if owned is None:
            active = settings or get_settings()
            configure_logging(active)
            owned = build_app_deps(active)
        ensure_indexes = getattr(owned.repo, "ensure_indexes", None)
        if ensure_indexes is not None:  # MongoDB: indexes and TTLs (HLD §8.1)
            await ensure_indexes()
        if owned.rate_limiter is None:
            owned.rate_limiter = RateLimiter(owned.settings.rate_limit_runs_per_hour)
        app.state.ctx = Context(deps=owned, graph=build_graph(owned))
        try:
            yield
        finally:
            await owned.aclose()  # the app owns the dependencies it is given

    app = FastAPI(
        title="CPG Partner Agents",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    install_error_handlers(app)
    app.include_router(router)
    app.include_router(open_router)
    if with_ui:

        @app.get("/", include_in_schema=False)
        async def home() -> RedirectResponse:
            return RedirectResponse("/ui/")

        active = settings or (deps.settings if deps else get_settings())
        ui = build_ui(lambda: app.state.ctx, demo_mode=active.demo_mode)
        app = gr.mount_gradio_app(app, ui, path="/ui", ssr_mode=False)
    return app


app = create_app()
