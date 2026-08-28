import json
import zipfile

from pathmnist.archive import build_archive


def _make_project(root) -> None:
    (root / "configs").mkdir()
    (root / "configs/pathmnist_m4.yaml").write_text("dataset:\n  classes: 9\n")
    (root / "pathmnist").mkdir()
    (root / "pathmnist/train.py").write_text("print('train')\n")
    (root / "docs/latex").mkdir(parents=True)
    (root / "docs/M5_FORMAL_PAPER.md").write_text("# Title\n")
    (root / "docs/latex/M5_FORMAL_PAPER.tex").write_text("\\documentclass{article}\n")
    (root / "docs/latex/M5_FORMAL_PAPER.pdf").write_bytes(b"%pdf-1.7\n")
    (root / "docs/latex/M5_FORMAL_PAPER.aux").write_text("aux\n")
    (root / "runs/pathmnist-m4/main/baseline/seed_7").mkdir(parents=True)
    (root / "runs/pathmnist-m4/main/baseline/seed_7/run.json").write_text("{}\n")
    (root / "runs/pathmnist-m4/main/baseline/seed_7/checkpoint.pt").write_bytes(b"weights")
    (root / "state/workflow/formal-paper-v2").mkdir(parents=True)
    (root / "state/workflow/formal-paper-v2/budget.json").write_text("{}\n")
    (root / "state/workflow/formal-paper-v2.json").write_text("{}\n")
    (root / "state/workflow/worker-task-1.json").write_text("{}\n")
    (root / "pathmnist_64.npz").write_bytes(b"dataset")
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")


def test_archive_includes_evidence_and_excludes_bulk_artifacts(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _make_project(project)
    output = tmp_path / "out/archive.zip"
    result = build_archive(project, output)
    assert output.is_file()
    assert (tmp_path / "out/archive.zip.manifest.json").is_file()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    prefix = "path-ai-scientist/"
    for expected in [
        "configs/pathmnist_m4.yaml",
        "pathmnist/train.py",
        "docs/M5_FORMAL_PAPER.md",
        "docs/latex/M5_FORMAL_PAPER.tex",
        "docs/latex/M5_FORMAL_PAPER.pdf",
        "runs/pathmnist-m4/main/baseline/seed_7/run.json",
        "state/workflow/formal-paper-v2/budget.json",
        "state/workflow/formal-paper-v2.json",
        "pyproject.toml",
        "manifest.json",
    ]:
        assert prefix + expected in names
    for excluded in [
        "docs/latex/M5_FORMAL_PAPER.aux",
        "runs/pathmnist-m4/main/baseline/seed_7/checkpoint.pt",
        "state/workflow/worker-task-1.json",
        "pathmnist_64.npz",
    ]:
        assert prefix + excluded not in names
    manifest = json.loads((tmp_path / "out/archive.zip.manifest.json").read_text())
    assert manifest["file_count"] == result["file_count"]
    assert manifest["file_count"] == len(manifest["files"])
    assert all("sha256" in entry and len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_archive_is_deterministic(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _make_project(project)
    first = tmp_path / "out/first.zip"
    second = tmp_path / "out/second.zip"
    build_archive(project, first)
    build_archive(project, second)
    assert first.read_bytes() == second.read_bytes()
