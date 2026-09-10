from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _server_request_id(headers: Mapping[str, Any] | None) -> str | None:
    if not headers:
        return None
    lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
    for name in ("x-request-id", "request-id", "x-trace-id", "trace-id"):
        if lowered.get(name):
            return lowered[name]
    return None


@dataclass
class RequestDiagnostic:
    """Persist small, prompt-free diagnostics for one logical provider request."""

    path: Path
    request_id: str
    role: str
    model: str
    endpoint: str
    timeout_seconds: int
    max_output_tokens: int
    input_characters: int
    _started: float = field(default_factory=time.monotonic)
    _attempt_started: float | None = None
    record: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.record = {
            "schema_version": 1,
            "client_request_id": self.request_id,
            "role": self.role,
            "model": self.model,
            "endpoint": self.endpoint,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
            "input_characters": self.input_characters,
            "input_tokens": None,
            "input_token_source": "unavailable_until_provider_response",
            "started_at": _now(),
            "state": "in_flight",
            "attempts": [],
        }
        _write(self.path, self.record)

    def begin_attempt(self, name: str) -> None:
        self._attempt_started = time.monotonic()
        self.record["attempts"].append({"name": name, "started_at": _now()})
        _write(self.path, self.record)

    def transport_succeeded(
        self, status: int, headers: Mapping[str, Any] | None, response_bytes: int
    ) -> None:
        attempt = self.record["attempts"][-1]
        attempt.update(
            {
                "transport_state": "response_received",
                "http_status": status,
                "server_request_id": _server_request_id(headers),
                "response_bytes": response_bytes,
                "elapsed_seconds": self._attempt_elapsed(),
            }
        )
        _write(self.path, self.record)

    def transport_failed(
        self, exc: BaseException, *, http_status: int | None = None,
        headers: Mapping[str, Any] | None = None,
    ) -> None:
        attempt = self.record["attempts"][-1]
        attempt.update(
            {
                "transport_state": "failed",
                "http_status": http_status,
                "server_request_id": _server_request_id(headers),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:500],
                "failure_category": self._category(exc, http_status),
                "elapsed_seconds": self._attempt_elapsed(),
            }
        )
        self.record.update(
            {"state": "outcome_unknown", "finished_at": _now(),
             "elapsed_seconds": round(time.monotonic() - self._started, 3)}
        )
        _write(self.path, self.record)

    def complete(self, raw: dict[str, Any], usage: dict[str, Any]) -> None:
        self.record.update(
            {
                "state": "completed",
                "finished_at": _now(),
                "elapsed_seconds": round(time.monotonic() - self._started, 3),
                "server_response_id": raw.get("id"),
                "resolved_model": raw.get("model"),
                "usage": usage,
                "input_tokens": usage.get("prompt_tokens", 0),
                "input_token_source": "provider_usage",
            }
        )
        _write(self.path, self.record)

    def fail_after_response(self, exc: BaseException) -> None:
        self.record.update(
            {
                "state": "response_processing_failed",
                "finished_at": _now(),
                "elapsed_seconds": round(time.monotonic() - self._started, 3),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:500],
            }
        )
        _write(self.path, self.record)

    def _attempt_elapsed(self) -> float:
        start = self._attempt_started if self._attempt_started is not None else self._started
        return round(time.monotonic() - start, 3)

    @staticmethod
    def _category(exc: BaseException, status: int | None) -> str:
        if status is not None:
            return "http_timeout" if status == 408 else "http_error"
        if isinstance(exc, TimeoutError):
            return "read_timeout"
        name = type(exc).__name__.casefold()
        reason = getattr(exc, "reason", None)
        detail = f"{exc} {reason}".casefold()
        if "timeout" in name or "timed out" in detail:
            return "transport_timeout"
        if "name or service" in detail or "getaddrinfo" in detail:
            return "dns_error"
        if "connection" in detail:
            return "connection_error"
        if "json" in name:
            return "invalid_json_response"
        return "transport_error"
