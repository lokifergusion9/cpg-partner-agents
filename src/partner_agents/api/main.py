"""The FastAPI app (HLD §7). ``uvicorn partner_agents.api.main:app`` serves it at ``/api``.

``create_app`` takes ready dependencies for tests; otherwise the lifespan builds the real ones
from settings on startup, so importing this module stays cheap.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from partner_agents.api.errors import install_error_handlers
from partner_agents.api.routes import Context, open_router, router
from partner_agents.config import Settings, get_settings
from partner_agents.graph.build import AppDeps, build_app_deps, build_graph
from partner_agents.observability import configure_logging


def create_app(deps: AppDeps | None = None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = deps
        if owned is None:
            active = settings or get_settings()
            configure_logging(active)
            owned = build_app_deps(active)
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
    return app


app = create_app()
