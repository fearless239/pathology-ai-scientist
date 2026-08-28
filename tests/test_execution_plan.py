import json

import pytest

from pathmnist.execution_plan import (
    ExecutionPlanError,
    StepExecutor,
    require_compatible_execution,
    validate_plan,
)


def plan():
    def train(name, value, fraction=0.2):
        return {
            "id": name,
            "kind": "train",
            "depends_on": [],
            "seed": 0,
            "role": "proposed_method",
            "parameters": {"alpha": value},
            "train_fraction": fraction,
            "epochs": 3,
            "timeout_seconds": 60,
            "max_attempts": 2,
        }

    final = train("final", None, 1.0)
    final.update(parameters={}, parameters_from="select", depends_on=["select"])
    return {
        "schema_version": 1,
        "max_total_epochs": 18,
        "steps": [
            train("a", 0.05),
            train("b", 0.1),
            {
                "id": "select",
                "kind": "select",
                "depends_on": ["a", "b"],
                "metric": "accuracy",
                "split": "validation",
            },
            final,
        ],
    }


def callbacks(calls, fail=None):
    def execute(step, parameters, directory):
        calls.append((step["id"], parameters))
        if step["id"] == fail:
            raise RuntimeError("interrupted")
        value = 0.8 if parameters["alpha"] == 0.1 else 0.7
        (directory / "metrics.json").write_text(json.dumps({"accuracy": value}))
        return {"metrics": {"accuracy": value}}

    def verify(step, evidence, directory):
        assert json.loads((directory / "metrics.json").read_text()) == evidence["metrics"]
        return evidence["metrics"]

    return execute, verify


def test_selection_and_resume_only_pending(tmp_path):
    calls = []
    execute, verify = callbacks(calls, "final")
    runner = StepExecutor(tmp_path, plan(), "contract", "dataset")
    with pytest.raises(RuntimeError):
        runner.run(execute, verify)
    assert [c[0] for c in calls] == ["a", "b", "final"]
    calls.clear()
    execute, verify = callbacks(calls)
    result = StepExecutor(tmp_path, plan(), "contract", "dataset").run(execute, verify)
    assert result["status"] == "completed"
    assert calls == [("final", {"alpha": 0.1})]
    calls.clear()
    assert (
        StepExecutor(tmp_path, plan(), "contract", "dataset").run(execute, verify)["status"]
        == "completed"
    )
    assert not calls


@pytest.mark.parametrize("mutation", ["cycle", "test", "budget", "override", "seed", "path"])
def test_invalid_plans_fail_before_execution(mutation):
    value = plan()
    if mutation == "cycle":
        value["steps"][0]["depends_on"] = ["final"]
    if mutation == "test":
        value["steps"][2]["split"] = "test"
    if mutation == "budget":
        value["max_total_epochs"] = 1
    if mutation == "override":
        value["steps"][3]["parameters"] = {"alpha": 0.2}
    if mutation == "seed":
        value["steps"][1]["seed"] = 1
    if mutation == "path":
        value["steps"][0]["id"] = "../outside"
    with pytest.raises(ExecutionPlanError):
        validate_plan(value)


def test_legacy_multistep_is_blocked_before_paid_execution():
    with pytest.raises(ExecutionPlanError, match="EXECUTION_PLAN_REQUIRED"):
        require_compatible_execution({"research_question": "先使用20%子集选参，最后完整训练"})
    assert require_compatible_execution({"research_question": "Train a baseline"}) is None
    with pytest.raises(ExecutionPlanError, match="EXECUTION_BACKEND_UNSUPPORTED"):
        require_compatible_execution({"execution_plan": plan()})


def test_changed_identity_rejects_resume(tmp_path):
    StepExecutor(tmp_path, plan(), "contract", "dataset").run(*callbacks([]))
    with pytest.raises(ExecutionPlanError, match="identity changed"):
        StepExecutor(tmp_path, plan(), "contract", "changed")


def test_corrupt_completed_evidence_does_not_retrain(tmp_path):
    calls = []
    StepExecutor(tmp_path, plan(), "contract", "dataset").run(*callbacks(calls))
    (tmp_path / "a/1/metrics.json").write_text("{}")
    calls.clear()
    with pytest.raises(AssertionError):
        StepExecutor(tmp_path, plan(), "contract", "dataset").run(*callbacks(calls))
    assert not calls


def test_pause_before_start_has_no_callbacks(tmp_path):
    calls = []
    assert (
        StepExecutor(tmp_path, plan(), "contract", "dataset").run(
            *callbacks(calls), should_stop=lambda: True
        )["status"]
        == "paused"
    )
    assert not calls


def test_retry_budget_is_persistent(tmp_path):
    for _ in range(2):
        with pytest.raises(RuntimeError, match="interrupted"):
            StepExecutor(tmp_path, plan(), "contract", "dataset").run(*callbacks([], "a"))
    calls = []
    with pytest.raises(ExecutionPlanError, match="retry budget exhausted"):
        StepExecutor(tmp_path, plan(), "contract", "dataset").run(*callbacks(calls))
    assert not calls
