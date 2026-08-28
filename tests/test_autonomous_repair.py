import pickle
import sys
from pathlib import Path

from pathmnist.autonomous_repair import REPAIR_STAGE, prepare_repair


def test_prepare_repair_preserves_journals_and_is_idempotent(tmp_path):
    project = Path(__file__).parents[1]
    vendor = project / "vendor/AI-Scientist-v2"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from ai_scientist.treesearch.agent_manager import Stage
    from ai_scientist.treesearch.journal import Journal

    checkpoint = tmp_path / "task/experiment_logs/manager.pkl"
    checkpoint.parent.mkdir(parents=True)
    old = Stage("4_ablation_studies_1_first_attempt", "first_attempt", "old", 2, 0, 4)
    state = {
        "stages": [old],
        "journals": {old.name: Journal()},
        "stage_history": [],
        "current_stage": None,
    }
    with checkpoint.open("wb") as handle:
        pickle.dump(state, handle)

    assert prepare_repair(project, tmp_path, "task") is True
    assert prepare_repair(project, tmp_path, "task") is False
    with checkpoint.open("rb") as handle:
        repaired = pickle.load(handle)
    assert old.name in repaired["journals"]
    assert REPAIR_STAGE in repaired["journals"]
    assert repaired["current_stage"].name == REPAIR_STAGE
    assert repaired["stage_history"][-1].from_stage == old.name
