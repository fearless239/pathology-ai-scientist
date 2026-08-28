import json

from pathmnist.trace import TraceWriter, main, read_trace, summarize


def test_trace_records_stage_lifecycle_and_summary(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path, "task-1")
    span = writer.start_span("stage.literature_collected", attributes={"reservation_usd": 0.1})
    writer.event("llm.completed", attributes={"request_id": "req-1", "cost_usd": 0.02})
    writer.end_span(
        span,
        status="ok",
        duration_seconds=1.25,
        attributes={"cost_usd": 0.02, "artifacts": {"literature.json": "abc"}},
    )

    records = read_trace(path)
    report = summarize(records)
    assert len(records) == 3
    assert report["completed_stages"] == 1
    assert report["failed_stages"] == []
    assert report["llm_calls"] == 1
    assert report["total_cost_usd"] == 0.02


def test_trace_summary_reports_failed_attempt(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path, "task-2")
    span = writer.start_span("stage.baseline_completed", attempt=2)
    writer.end_span(
        span,
        status="error",
        duration_seconds=0.5,
        error=RuntimeError("training failed"),
    )
    report = summarize(read_trace(path))
    assert report["failed_stages"] == ["baseline_completed"]
    assert report["stage_attempts"] == 1


def test_trace_cli_writes_json_report(tmp_path):
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "summary.json"
    writer = TraceWriter(trace, "task-3")
    span = writer.start_span("stage.task_created")
    writer.end_span(span, status="ok", duration_seconds=0.01)
    assert main([str(trace), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["completed_stages"] == 1
