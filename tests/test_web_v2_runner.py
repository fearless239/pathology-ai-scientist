import json

from pathmnist.web_v2_runner import log_tail, run_status


def test_web_run_status_defaults_when_not_started(tmp_path):
    assert run_status(tmp_path)["state"] == "not_started"


def test_web_run_status_and_log_are_readable(tmp_path):
    (tmp_path / "web_run.json").write_text(
        json.dumps({"state": "completed", "message": "done"}), encoding="utf-8"
    )
    (tmp_path / "web_run.log").write_text("one\ntwo\nthree\n", encoding="utf-8")

    assert run_status(tmp_path)["message"] == "done"
    assert log_tail(tmp_path, lines=2) == "two\nthree"


def test_agent_progress_does_not_change_stopped_state(tmp_path):
    (tmp_path / "web_run.json").write_text(json.dumps({"state": "interrupted"}))
    (tmp_path / "agent_progress.json").write_text(json.dumps({"stage": "2_baseline_tuning", "node_count": 3}))
    status = run_status(tmp_path)
    assert status["state"] == "interrupted"
    assert status["progress"]["stage"] == "2_baseline_tuning"
    (tmp_path / "agent_progress.json").write_text("{")
    assert run_status(tmp_path)["state"] == "interrupted"
