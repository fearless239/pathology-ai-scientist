from __future__ import annotations

import argparse
import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRACE_SCHEMA_VERSION = 1
_WRITE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Span:
    trace_id: str
    span_id: str
    name: str
    started_at: str
    attempt: int


class TraceWriter:
    """Small, dependency-free JSONL trace writer for durable research runs.

    Trace attributes must contain metadata and references, not prompts, model
    responses, dataset rows, images, credentials, or other sensitive payloads.
    """

    def __init__(self, path: Path, trace_id: str) -> None:
        self.path = path
        self.trace_id = trace_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_span(
        self, name: str, *, attempt: int = 1, attributes: dict[str, Any] | None = None
    ) -> Span:
        span = Span(self.trace_id, uuid.uuid4().hex, name, _now(), attempt)
        self._append(
            {
                "event": "span.started",
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "name": span.name,
                "attempt": span.attempt,
                "timestamp": span.started_at,
                "attributes": attributes or {},
            }
        )
        return span

    def end_span(
        self,
        span: Span,
        *,
        status: str,
        duration_seconds: float,
        attributes: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "event": "span.ended",
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "name": span.name,
            "attempt": span.attempt,
            "timestamp": _now(),
            "started_at": span.started_at,
            "status": status,
            "duration_seconds": round(max(0.0, duration_seconds), 6),
            "attributes": attributes or {},
        }
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error)[:2000],
            }
        self._append(record)

    def event(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        span_id: str | None = None,
    ) -> None:
        self._append(
            {
                "event": name,
                "trace_id": self.trace_id,
                "span_id": span_id,
                "timestamp": _now(),
                "attributes": attributes or {},
            }
        )

    def _append(self, record: dict[str, Any]) -> None:
        payload = {"schema_version": TRACE_SCHEMA_VERSION, **record}
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with _WRITE_LOCK:
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY)
            try:
                os.write(descriptor, line.encode("utf-8"))
            finally:
                os.close(descriptor)


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid trace JSON at line {number}") from exc
    return records


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ended = [item for item in records if item.get("event") == "span.ended"]
    stages = [item for item in ended if str(item.get("name", "")).startswith("stage.")]
    llm_events = [item for item in records if item.get("event") == "llm.completed"]
    failed = [item for item in stages if item.get("status") != "ok"]
    stage_costs = {
        item["name"].removeprefix("stage."): float(
            item.get("attributes", {}).get("cost_usd", 0.0)
        )
        for item in stages
        if item.get("status") == "ok"
    }
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_ids": sorted({str(item.get("trace_id")) for item in ended}),
        "stage_attempts": len(stages),
        "completed_stages": sum(item.get("status") == "ok" for item in stages),
        "failed_stages": [item["name"].removeprefix("stage.") for item in failed],
        "total_stage_seconds": round(
            sum(float(item.get("duration_seconds", 0.0)) for item in stages), 6
        ),
        "total_cost_usd": round(sum(stage_costs.values()), 10),
        "stage_cost_usd": stage_costs,
        "llm_calls": len(llm_events),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a pathology-agent JSONL trace")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = summarize(read_trace(args.trace))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
