"""Email dispatch for approved drafts (HLD §6.9): outbox, Gmail draft, Gmail send.

``dispatch_email`` is the only place a draft leaves the system, and it only gets drafts the
graph has seen a human approve. ``outbox`` (the default) stores the draft and sends nothing.
``gmail_draft`` creates a draft in the user's own Gmail. ``gmail_send`` sends, and only to
recipients on ``OUTREACH_ALLOWLIST`` (checked again here, after the policy check in
``chase_draft``).

The Gmail calls are two plain ``httpx`` POSTs (ADR-027). They are not retried: a send that timed
out may still have gone out, and a second attempt would send it twice. The OAuth token lives in
``GMAIL_TOKEN_PATH``, which must be outside the repository.
"""

import asyncio
import base64
import contextlib
import logging
from collections.abc import Callable, Iterable
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Protocol

import httpx

from partner_agents.config import OutreachMode, Settings
from partner_agents.outreach.policy import is_allowlisted
from partner_agents.schemas import OutreachDraft, StageError

logger = logging.getLogger(__name__)

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]  # drafts and send
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_TIMEOUT_S = 20.0
REPO_ROOT = Path(__file__).resolve().parents[3]


class GmailError(Exception):
    """The Gmail call or its authorisation failed. The message never holds mail content."""


class GmailClient(Protocol):
    async def create_draft(self, raw: str) -> str:
        """Create a draft from a base64url RFC 822 message. Returns the Gmail draft id."""
        ...

    async def send(self, raw: str) -> str:
        """Send a base64url RFC 822 message. Returns the Gmail message id."""
        ...


# --- building the message -----------------------------------------------------------


def build_raw_message(draft: OutreachDraft, sender: str) -> str:
    """The draft as a base64url RFC 822 message. A header with a line break raises ValueError."""
    message = EmailMessage()
    message["From"] = sender
    message["To"] = draft.recipient
    message["Subject"] = draft.subject or ""
    message.set_content(draft.body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


# --- dispatching --------------------------------------------------------------------

_DONE = {"outbox": "outboxed", "gmail_draft": "drafted_in_gmail", "gmail_send": "sent"}


def _failed(
    draft: OutreachDraft, code: str, message: str, now: datetime
) -> tuple[OutreachDraft, StageError]:
    error = StageError(stage="chase_dispatch", code=code, message=message)
    update = {"status": "failed", "decided_at": draft.decided_at or now}
    return draft.model_copy(update=update), error


async def dispatch_email(
    draft: OutreachDraft,
    *,
    mode: OutreachMode,
    allowlist: Iterable[str],
    sender: str,
    client: GmailClient | None,
    now: datetime,
) -> tuple[OutreachDraft, StageError | None]:
    """Deliver one approved draft according to ``mode``. Returns the draft with its new status."""
    if draft.status != "approved" or draft.channel != "email":
        return _failed(draft, "not_approved", "only an approved email draft can be dispatched", now)
    if not is_allowlisted(draft.recipient, mode, allowlist):
        return _failed(draft, "not_allowlisted", "the recipient is not on OUTREACH_ALLOWLIST", now)
    if mode == "outbox":
        return draft.model_copy(update={"status": "outboxed"}), None
    if client is None:
        return _failed(draft, "gmail_not_configured", f"{mode} needs Gmail authorisation", now)
    try:
        raw = build_raw_message(draft, sender)
        if mode == "gmail_draft":
            await client.create_draft(raw)
        else:
            await client.send(raw)
    except ValueError:
        return _failed(draft, "invalid_message", "the subject or an address has a line break", now)
    except Exception as exc:
        logger.warning("gmail dispatch failed", extra={"error_type": type(exc).__name__})
        detail = str(exc) if isinstance(exc, GmailError) else type(exc).__name__
        return _failed(draft, "gmail_error", detail[:200], now)
    return draft.model_copy(update={"status": _DONE[mode]}), None


# --- Gmail over httpx ---------------------------------------------------------------


def ensure_outside_repo(path: Path) -> Path:
    """Refuse an OAuth file inside the repository (CLAUDE.md rule 5)."""
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(REPO_ROOT):
        raise GmailError("GMAIL_CREDENTIALS_PATH and GMAIL_TOKEN_PATH must be outside the repo")
    return resolved


def load_access_token(settings: Settings) -> str:
    """A valid access token from the saved authorisation, refreshed (and saved) if expired."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = ensure_outside_repo(settings.gmail_token_path)
    if not token_path.exists():
        raise GmailError("Gmail is not authorised yet: run `cli gmail-auth`")
    credentials = Credentials.from_authorized_user_file(  # type: ignore[no-untyped-call]
        str(token_path), GMAIL_SCOPES
    )
    if not credentials.valid:
        if not credentials.refresh_token:
            raise GmailError("the Gmail authorisation expired: run `cli gmail-auth` again")
        try:
            credentials.refresh(Request())
        except Exception as exc:
            raise GmailError(f"refreshing the Gmail token failed ({type(exc).__name__})") from exc
        token_path.write_text(credentials.to_json(), encoding="utf-8")
    return str(credentials.token)


def authorize(settings: Settings) -> Path:
    """Run the installed-app OAuth flow in a browser and save the token. Returns its path."""
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]

    secrets = ensure_outside_repo(settings.gmail_credentials_path)
    token_path = ensure_outside_repo(settings.gmail_token_path)
    if not secrets.exists():
        raise GmailError(f"put your OAuth client file at {secrets} first (see the README)")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), GMAIL_SCOPES)
    credentials = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    with contextlib.suppress(OSError):
        token_path.chmod(0o600)  # best effort; Windows ignores the mode bits
    return token_path


class HttpxGmailClient:
    """``GmailClient`` over the Gmail REST API with a bearer token."""

    def __init__(
        self,
        settings: Settings,
        *,
        token: Callable[[], str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = token or (lambda: load_access_token(settings))
        self._transport = transport

    async def _post(self, path: str, body: dict[str, Any], what: str) -> str:
        access_token = await asyncio.to_thread(self._token)
        async with httpx.AsyncClient(timeout=GMAIL_TIMEOUT_S, transport=self._transport) as http:
            response = await http.post(
                f"{GMAIL_API}/{path}",
                json=body,
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if response.status_code >= 400:
            raise GmailError(f"Gmail {what} failed with HTTP {response.status_code}")
        return str(response.json().get("id", ""))

    async def create_draft(self, raw: str) -> str:
        return await self._post("drafts", {"message": {"raw": raw}}, "draft")

    async def send(self, raw: str) -> str:
        return await self._post("messages/send", {"raw": raw}, "send")
