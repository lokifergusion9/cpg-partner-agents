"""The HTTP routes (HLD §7). All paths are under ``/api``.

``/health`` is open. Every other route needs ``X-API-Key`` when ``APP_API_KEY`` is set (never in
demo mode). The call routes run the call sub-graph (``graph/call.py``); ``/simulate`` plays a
demo persona against it (``calls/simulator.py``).
"""

import logging
import secrets
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request
from pydantic import BaseModel, Field

from partner_agents.api.errors import ApiError
from partner_agents.calls.personas import PersonaError, load_persona
from partner_agents.calls.simulator import simulate_call
from partner_agents.graph.build import (
    Context,
    NothingToApprove,
    TurnResult,
    resume_turn,
    run_turn,
)
from partner_agents.graph.call import CallDeps, CallGraph, CallNotActive, run_call_turn
from partner_agents.schemas import (
    ApprovalDecision,
    BrandProfile,
    CallOutcome,
    CallRecord,
    CallStage,
    Intent,
    Lead,
    OutreachDraft,
    RuleCheck,
    RunRecord,
    StageError,
)

logger = logging.getLogger(__name__)

ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
UserId = Annotated[str, Query(pattern=ID_PATTERN)]


def get_context(request: Request) -> Context:
    ctx: Context = request.app.state.ctx
    return ctx


Ctx = Annotated[Context, Depends(get_context)]


def require_api_key(request: Request, ctx: Ctx) -> None:
    settings = ctx.deps.settings
    if settings.app_api_key is None or settings.demo_mode:
        return
    given = request.headers.get("X-API-Key", "")
    expected = settings.app_api_key.get_secret_value()
    if not secrets.compare_digest(given.encode(), expected.encode()):
        raise ApiError(401, "unauthorized", "A valid X-API-Key header is required.")


router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])
open_router = APIRouter(prefix="/api")


# --- models -------------------------------------------------------------------------


class ChatRequest(BaseModel):
    user_id: str = Field(pattern=ID_PATTERN)
    thread_id: str | None = Field(default=None, pattern=ID_PATTERN)
    message: str = Field(min_length=1, max_length=2000)


class RouteSummary(BaseModel):
    intent: Intent
    confidence: float


class ChatResponse(BaseModel):
    thread_id: str
    run_id: str
    route: RouteSummary
    reply: str
    leads: list[Lead]
    pending_approval: dict[str, Any] | None  # {"drafts": [...], "problems": {...}}
    errors: list[StageError]


class ProfileBody(BrandProfile):
    user_id: str = Field(pattern=ID_PATTERN)


def chat_response(thread_id: str, result: TurnResult) -> ChatResponse:
    route = result.state.get("route")
    return ChatResponse(
        thread_id=thread_id,
        run_id=result.run.run_id,
        route=RouteSummary(
            intent=result.run.intent, confidence=route.confidence if route is not None else 0.0
        ),
        reply=result.reply,
        leads=result.state.get("leads", []),
        pending_approval=result.pending_approval,
        errors=result.run.errors,
    )


# --- chat ---------------------------------------------------------------------------


async def _thread_state(ctx: Context, thread_id: str, user_id: str) -> Any:
    """The thread's snapshot, or 404 if it belongs to someone else."""
    snapshot = await ctx.graph.aget_state({"configurable": {"thread_id": thread_id}})
    owner = snapshot.values.get("user_id")
    if owner is not None and owner != user_id:
        raise ApiError(404, "thread_not_found", "No such conversation.")
    return snapshot


@router.post("/chat")
async def chat(body: ChatRequest, request: Request, ctx: Ctx) -> ChatResponse:
    thread_id = body.thread_id or f"t_{uuid.uuid4().hex[:12]}"
    async with ctx.lock(thread_id):
        snapshot = await _thread_state(ctx, thread_id, body.user_id)
        if "approval_gate" in snapshot.next:
            raise ApiError(
                409,
                "approval_pending",
                "Approve or reject the pending drafts before sending another message.",
            )
        client = request.client.host if request.client else "unknown"
        result = await run_turn(
            ctx.graph,
            ctx.deps,
            thread_id=thread_id,
            user_id=body.user_id,
            text=body.message,
            rate_keys=[f"user:{body.user_id}", f"ip:{client}"],
        )
    retry_after = result.state.get("retry_after_s")
    if retry_after:  # rate_gate refused this discovery run
        raise ApiError(429, "rate_limited", result.reply, retry_after_s=retry_after)
    return chat_response(thread_id, result)


@router.post("/chat/{thread_id}/approve")
async def approve(
    thread_id: Annotated[str, Path(pattern=ID_PATTERN)],
    decision: ApprovalDecision,
    user_id: UserId,
    ctx: Ctx,
) -> ChatResponse:
    async with ctx.lock(thread_id):
        await _thread_state(ctx, thread_id, user_id)
        try:
            result = await resume_turn(
                ctx.graph, ctx.deps, thread_id=thread_id, user_id=user_id, decision=decision
            )
        except NothingToApprove as exc:
            raise ApiError(
                404, "nothing_to_approve", "No drafts are waiting for approval."
            ) from exc
    return chat_response(thread_id, result)


# --- data ---------------------------------------------------------------------------


@router.get("/leads")
async def leads(
    ctx: Ctx, user_id: UserId, run_id: Annotated[str | None, Query(max_length=64)] = None
) -> list[Lead]:
    return await ctx.deps.repo.list_leads(user_id, run_id=run_id)


@router.get("/outreach")
async def outreach(
    ctx: Ctx, user_id: UserId, status: Annotated[str | None, Query(max_length=32)] = None
) -> list[OutreachDraft]:
    return await ctx.deps.repo.list_drafts(user_id, status=status)


@router.get("/runs/{run_id}")
async def run_record(run_id: str, user_id: UserId, ctx: Ctx) -> RunRecord:
    run = await ctx.deps.repo.get_run(run_id)
    if run is None or run.user_id != user_id:
        raise ApiError(404, "run_not_found", "No such run.")
    return run


@router.get("/profile")
async def get_profile(user_id: UserId, ctx: Ctx) -> BrandProfile:
    profile = await ctx.deps.repo.get_profile(user_id)
    if profile is None:
        raise ApiError(404, "profile_not_found", "No brand profile has been saved yet.")
    return profile


@router.put("/profile")
async def put_profile(body: ProfileBody, ctx: Ctx) -> BrandProfile:
    profile = BrandProfile.model_validate(body.model_dump(exclude={"user_id"}))
    await ctx.deps.repo.save_profile(body.user_id, profile)
    return profile


# --- calls (HLD §6.11) ----------------------------------------------------------------

CallId = Annotated[str, Path(pattern=ID_PATTERN)]


class CallTurnBody(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class CallTurnResponse(BaseModel):
    call_id: str
    stage: CallStage
    reply: str
    checks: list[RuleCheck]
    ended: bool
    outcome: CallOutcome | None


def _call_deps(ctx: Context) -> tuple[CallGraph, CallDeps]:
    if ctx.call_graph is None or ctx.deps.calls is None:
        raise ApiError(503, "calls_unavailable", "The call agent is not configured.")
    return ctx.call_graph, ctx.deps.calls


def _turn_response(record: CallRecord) -> CallTurnResponse:
    last = record.turns[-1]
    return CallTurnResponse(
        call_id=record.id,
        stage=record.progress.stage,
        reply=last.text,
        checks=last.checks,
        ended=record.progress.outcome is not None,
        outcome=record.progress.outcome,
    )


async def _call_turn(ctx: Context, user_id: str, call_id: str, text: str | None) -> CallRecord:
    graph, deps = _call_deps(ctx)
    async with ctx.lock(f"call:{call_id}"):
        try:
            record, _ = await run_call_turn(
                graph, deps, user_id=user_id, call_id=call_id, partner_text=text
            )
        except LookupError as exc:
            raise ApiError(404, "call_not_found", "No such call.") from exc
        except CallNotActive as exc:
            raise ApiError(409, exc.code, str(exc)) from exc
    return record


@router.post("/calls/{call_id}/start")
async def start_call(call_id: CallId, user_id: UserId, ctx: Ctx) -> CallTurnResponse:
    """The opening line of an approved call (created by the mock caller at dispatch)."""
    return _turn_response(await _call_turn(ctx, user_id, call_id, None))


@router.post("/calls/{call_id}/turn")
async def call_turn(
    call_id: CallId, body: CallTurnBody, user_id: UserId, ctx: Ctx
) -> CallTurnResponse:
    """One partner turn typed as text (``live_text``); the reply is the agent's next line."""
    return _turn_response(await _call_turn(ctx, user_id, call_id, body.text))


class SimulateBody(BaseModel):
    persona_id: str = Field(pattern=r"^[a-z0-9_]{1,64}$")


@router.post("/calls/{call_id}/simulate")
async def simulate(call_id: CallId, body: SimulateBody, user_id: UserId, ctx: Ctx) -> CallRecord:
    """Play a demo persona against a call that has not started; returns the whole call."""
    graph, deps = _call_deps(ctx)
    if deps.simulator is None:
        raise ApiError(503, "simulator_unavailable", "No simulator model is configured.")
    try:
        persona = load_persona(body.persona_id)
    except PersonaError as exc:
        raise ApiError(404, "persona_not_found", "No such persona.") from exc
    async with ctx.lock(f"call:{call_id}"):
        try:
            return await simulate_call(
                graph, deps, deps.simulator, persona, user_id=user_id, call_id=call_id
            )
        except LookupError as exc:
            raise ApiError(404, "call_not_found", "No such call.") from exc
        except CallNotActive as exc:
            raise ApiError(409, exc.code, str(exc)) from exc
        except Exception as exc:  # the turns played so far stay saved
            logger.warning("simulation failed", extra={"error_type": type(exc).__name__})
            raise ApiError(502, "simulator_error", "The simulator failed mid-call.") from exc


@router.get("/calls")
async def list_calls(
    ctx: Ctx, user_id: UserId, outcome: Annotated[str | None, Query(max_length=32)] = None
) -> list[CallRecord]:
    calls = await ctx.deps.repo.list_calls(user_id, outcome=outcome)
    return [call.model_copy(update={"turns": []}) for call in calls]  # turns omitted


@router.get("/calls/{call_id}")
async def get_call(call_id: CallId, user_id: UserId, ctx: Ctx) -> CallRecord:
    record = await ctx.deps.repo.get_call(user_id, call_id)
    if record is None:
        raise ApiError(404, "call_not_found", "No such call.")
    return record


# --- health -------------------------------------------------------------------------


@open_router.get("/health")
async def health(ctx: Ctx) -> dict[str, str]:
    """Configuration only: no outbound call, no secret and no key."""
    settings = ctx.deps.settings
    return {
        "status": "ok",
        "llm": settings.llm_provider,
        "qdrant": "local" if settings.qdrant_is_local else "remote",
        "mongo": "memory" if settings.mongo_is_in_memory else "configured",
        "search": settings.search_provider,
    }
