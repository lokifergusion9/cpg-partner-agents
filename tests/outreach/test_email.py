"""Email dispatch: outbox, Gmail draft and send, allowlist, demo mode (HLD §6.9)."""

import asyncio
import base64
from datetime import UTC, datetime
from email import message_from_bytes
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from partner_agents.config import Settings
from partner_agents.outreach.email import (
    GMAIL_API,
    GmailError,
    HttpxGmailClient,
    build_raw_message,
    dispatch_email,
    ensure_outside_repo,
)
from partner_agents.schemas import OutreachDraft

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
SENDER = "priya@crumb.example"


def draft(status: str = "approved", **overrides: Any) -> OutreachDraft:
    fields: dict[str, Any] = {
        "id": "d1", "lead_id": "l1", "user_id": "u", "run_id": "r", "channel": "email",
        "recipient": "buyer@acme.example", "subject": "Gluten-free cookies",
        "body": "A short intro. Priya Shah", "status": status, "prompt_version": "v",
        "created_at": NOW,
    }  # fmt: skip
    return OutreachDraft(**{**fields, **overrides})


class FakeGmail:
    def __init__(self, fail: Exception | None = None) -> None:
        self.drafts: list[str] = []
        self.sent: list[str] = []
        self.fail = fail

    async def create_draft(self, raw: str) -> str:
        if self.fail:
            raise self.fail
        self.drafts.append(raw)
        return "gd_1"

    async def send(self, raw: str) -> str:
        if self.fail:
            raise self.fail
        self.sent.append(raw)
        return "gm_1"


def dispatch(d: OutreachDraft, mode: str, client: FakeGmail | None, allowlist: list[str] = ()):  # type: ignore[no-untyped-def,assignment]
    return asyncio.run(
        dispatch_email(
            d,
            mode=mode,
            allowlist=allowlist,
            sender=SENDER,
            client=client,
            now=NOW,  # type: ignore[arg-type]
        )
    )


def decode(raw: str) -> Any:
    return message_from_bytes(base64.urlsafe_b64decode(raw))


# --- the three modes ----------------------------------------------------------------


def test_outbox_stores_the_draft_and_touches_no_client() -> None:
    gmail = FakeGmail()
    result, error = dispatch(draft(), "outbox", gmail)
    assert (result.status, error) == ("outboxed", None)
    assert gmail.drafts == [] and gmail.sent == []


def test_gmail_draft_creates_a_draft_and_sends_nothing() -> None:
    gmail = FakeGmail()
    result, error = dispatch(draft(), "gmail_draft", gmail)
    assert (result.status, error) == ("drafted_in_gmail", None)
    assert gmail.sent == [] and len(gmail.drafts) == 1
    message = decode(gmail.drafts[0])
    assert (message["From"], message["To"]) == (SENDER, "buyer@acme.example")
    assert message["Subject"] == "Gluten-free cookies"
    assert "A short intro." in message.get_payload()


def test_gmail_send_sends_to_an_allowlisted_recipient() -> None:
    gmail = FakeGmail()
    result, error = dispatch(draft(), "gmail_send", gmail, ["Buyer@Acme.example"])
    assert (result.status, error) == ("sent", None)
    assert len(gmail.sent) == 1 and gmail.drafts == []


def test_gmail_send_refuses_a_recipient_that_is_not_allowlisted() -> None:
    gmail = FakeGmail()
    result, error = dispatch(draft(), "gmail_send", gmail, ["someone@else.example"])
    assert result.status == "failed" and error is not None
    assert (error.stage, error.code) == ("chase_dispatch", "not_allowlisted")
    assert gmail.sent == [] and gmail.drafts == []  # nothing reached the client


# --- refusals and failures ----------------------------------------------------------


@pytest.mark.parametrize("status", ["pending_approval", "rejected", "outboxed", "sent", "failed"])
def test_only_an_approved_draft_is_dispatched(status: str) -> None:
    gmail = FakeGmail()
    result, error = dispatch(draft(status), "gmail_send", gmail, ["buyer@acme.example"])
    assert result.status == "failed" and error is not None and error.code == "not_approved"
    assert gmail.sent == []


def test_a_voice_draft_is_not_dispatched_as_email() -> None:
    result, error = dispatch(draft(channel="voice"), "outbox", None)
    assert result.status == "failed" and error is not None and error.code == "not_approved"


def test_gmail_modes_without_a_client_fail_clearly() -> None:
    result, error = dispatch(draft(), "gmail_draft", None)
    assert result.status == "failed" and error is not None
    assert error.code == "gmail_not_configured"


def test_a_gmail_failure_is_recorded_without_the_mail_content() -> None:
    gmail = FakeGmail(fail=RuntimeError("boom buyer@acme.example A short intro"))
    result, error = dispatch(draft(), "gmail_draft", gmail)
    assert result.status == "failed" and error is not None and error.code == "gmail_error"
    assert error.message == "RuntimeError"  # the type only; the exception text can hold content


def test_a_gmail_error_message_is_kept_when_it_is_ours() -> None:
    gmail = FakeGmail(fail=GmailError("Gmail send failed with HTTP 403"))
    _, error = dispatch(draft(), "gmail_draft", gmail)
    assert error is not None and error.message == "Gmail send failed with HTTP 403"


@pytest.mark.parametrize(
    "overrides",
    [
        {"subject": "Hi\r\nBcc: attacker@evil.example"},
        {"recipient": "a@x.example\nBcc: b@y.example"},
    ],
)
def test_a_header_with_a_line_break_is_refused(overrides: dict[str, str]) -> None:
    gmail = FakeGmail()
    result, error = dispatch(draft(**overrides), "gmail_draft", gmail)
    assert result.status == "failed" and error is not None and error.code == "invalid_message"
    assert gmail.drafts == []
    with pytest.raises(ValueError):
        build_raw_message(draft(**overrides), SENDER)


def test_demo_mode_forces_the_outbox_even_when_gmail_send_is_configured() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        demo_mode=True,
        outreach_mode="gmail_send",
        outreach_allowlist=["buyer@acme.example"],
    )
    assert settings.outreach_mode == "outbox"
    gmail = FakeGmail()
    result, _ = dispatch(draft(), settings.outreach_mode, gmail, settings.outreach_allowlist)
    assert result.status == "outboxed" and gmail.sent == [] and gmail.drafts == []


# --- Gmail over httpx ---------------------------------------------------------------


def settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@respx.mock
def test_the_httpx_client_posts_the_draft_with_a_bearer_token() -> None:
    route = respx.post(f"{GMAIL_API}/drafts").respond(200, json={"id": "gd_9"})
    client = HttpxGmailClient(settings(), token=lambda: "tok_123")
    assert asyncio.run(client.create_draft("RAW")) == "gd_9"
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer tok_123"
    assert request.content == b'{"message":{"raw":"RAW"}}'


@respx.mock
def test_the_httpx_client_sends_through_messages_send() -> None:
    route = respx.post(f"{GMAIL_API}/messages/send").respond(200, json={"id": "gm_9"})
    client = HttpxGmailClient(settings(), token=lambda: "tok")
    assert asyncio.run(client.send("RAW")) == "gm_9"
    assert route.calls.last.request.content == b'{"raw":"RAW"}'


@respx.mock
def test_an_http_error_becomes_a_gmail_error_without_the_response_body() -> None:
    respx.post(f"{GMAIL_API}/messages/send").respond(403, json={"error": "secret detail"})
    client = HttpxGmailClient(settings(), token=lambda: "tok")
    with pytest.raises(GmailError) as raised:
        asyncio.run(client.send("RAW"))
    assert str(raised.value) == "Gmail send failed with HTTP 403"


@respx.mock
def test_a_send_is_not_retried_after_a_server_error() -> None:
    route = respx.post(f"{GMAIL_API}/messages/send").mock(return_value=httpx.Response(503))
    client = HttpxGmailClient(settings(), token=lambda: "tok")
    with pytest.raises(GmailError):
        asyncio.run(client.send("RAW"))
    assert route.call_count == 1  # a retried send could deliver twice


def test_a_missing_token_file_says_how_to_authorise(tmp_path: Path) -> None:
    client = HttpxGmailClient(
        Settings(_env_file=None, gmail_token_path=tmp_path / "token.json")  # type: ignore[call-arg]
    )
    with pytest.raises(GmailError, match="gmail-auth"):
        asyncio.run(client.send("RAW"))


# --- credentials stay out of the repo -----------------------------------------------


def test_oauth_files_must_be_outside_the_repo(tmp_path: Path) -> None:
    assert ensure_outside_repo(tmp_path / "token.json") == (tmp_path / "token.json").resolve()
    inside = Path(__file__).resolve().parents[2] / "gmail_token.json"
    with pytest.raises(GmailError, match="outside the repo"):
        ensure_outside_repo(inside)
