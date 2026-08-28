import json

import pytest

from pathmnist.dashboard_ui import (
    APP_CSS,
    RESEARCH_PRESETS,
    TaskDeletionError,
    artifact_placeholder,
    delete_v2_task,
    discover_artifacts,
    format_bytes,
    list_v2_task_summaries,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_v2_task_summaries_are_newest_first_and_ignore_other_schemas(tmp_path):
    _write_json(
        tmp_path / "older" / "task.json",
        {
            "schema_version": 2,
            "research_direction": "旧课题",
            "completed_stage": "dataset_validated",
            "updated_at": "2026-08-20T00:00:00+00:00",
        },
    )
    _write_json(
        tmp_path / "newer" / "task.json",
        {
            "schema_version": 2,
            "research_direction": "新课题",
            "completed_stage": "figures_generated",
            "updated_at": "2026-08-21T00:00:00+00:00",
        },
    )
    _write_json(tmp_path / "legacy" / "task.json", {"schema_version": 1})
    (tmp_path / "broken" / "task.json").parent.mkdir(parents=True)
    (tmp_path / "broken" / "task.json").write_text("not-json", encoding="utf-8")

    summaries = list_v2_task_summaries(tmp_path)

    assert [item.task_id for item in summaries] == ["newer", "older"]
    assert summaries[0].research_direction == "新课题"
    assert summaries[0].completed_stage == "figures_generated"


def test_research_presets_are_editable_prompt_sources():
    assert [preset.slug for preset in RESEARCH_PRESETS] == [
        "robustness",
        "interpretability",
        "hard-cases",
    ]
    assert all("PathMNIST" in preset.prompt for preset in RESEARCH_PRESETS)
    assert len({preset.prompt for preset in RESEARCH_PRESETS}) == len(RESEARCH_PRESETS)


def test_artifact_discovery_uses_meaningful_image_order_and_deduplicates_files(tmp_path):
    figure_root = tmp_path / "paper" / "figures_generated" / "figures"
    figure_root.mkdir(parents=True)
    for name in ("dataset_splits.png", "custom_plot.png", "confusion_matrix.png", "test_metrics.png"):
        (figure_root / name).write_bytes(name.encode())
    paper = tmp_path / "paper" / "revision_completed" / "final_paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"pdf")
    manifest = tmp_path / "paper" / "figures_generated" / "figure_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    superseded = tmp_path / "paper" / "superseded" / "old.pdf"
    superseded.parent.mkdir(parents=True)
    superseded.write_bytes(b"old")

    images, files = discover_artifacts(tmp_path)

    assert [item.path.name for item in images] == [
        "test_metrics.png",
        "confusion_matrix.png",
        "dataset_splits.png",
        "custom_plot.png",
    ]
    assert [item.path.name for item in files] == ["final_paper.pdf", "figure_manifest.json"]
    assert images[0].label == "最终测试指标"


def test_artifact_placeholder_never_claims_nonexistent_results():
    assert artifact_placeholder("task_created")[0] == "研究尚在准备"
    assert artifact_placeholder("literature_collected")[0] == "研究方案已在形成"
    assert artifact_placeholder("analysis_completed")[0] == "成果正在生成"


def test_file_sizes_are_human_readable():
    assert format_bytes(12) == "12 B"
    assert format_bytes(1536) == "1.5 KB"


def test_sidebar_can_be_reopened_after_it_is_collapsed():
    assert 'data-testid="stExpandSidebarButton"' in APP_CSS
    assert 'data-testid="stToolbarActions"' in APP_CSS
    assert '[data-testid="stToolbar"] {\n    display: flex;' in APP_CSS


def test_delete_v2_task_removes_only_the_selected_stopped_task(tmp_path):
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    _write_json(selected / "task.json", {"schema_version": 2})
    _write_json(other / "task.json", {"schema_version": 2})
    (selected / "paper.md").write_text("generated", encoding="utf-8")

    delete_v2_task(tmp_path, "selected")

    assert not selected.exists()
    assert other.is_dir()


def test_delete_v2_task_rejects_running_or_unsafe_targets(tmp_path):
    running = tmp_path / "running"
    _write_json(running / "task.json", {"schema_version": 2})
    _write_json(running / "web_run.json", {"state": "running"})

    with pytest.raises(TaskDeletionError, match="running task"):
        delete_v2_task(tmp_path, "running")
    with pytest.raises(TaskDeletionError, match="Invalid task ID"):
        delete_v2_task(tmp_path, "../outside")
    assert running.is_dir()
