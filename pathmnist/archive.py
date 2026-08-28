"""Deterministic lightweight evidence archive for the PathMNIST M4/M5 chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_PRUNED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "vendor",
}
_ROOT_FILES = {"pyproject.toml", "UPSTREAM_MANIFEST.sha256"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included(relative: Path) -> bool:
    parts = relative.parts
    name = relative.as_posix()
    suffix = relative.suffix.lower()
    if parts[0] == "configs":
        return suffix in {".yaml", ".yml", ".json"}
    if parts[0] == "docker":
        return True
    if parts[0] == "pathmnist":
        return suffix == ".py"
    if parts[0] == "scripts":
        return name in {"scripts/gate-a.sh", "scripts/pathmnist.sh"}
    if parts[0] == "docs":
        if len(parts) == 2:
            return suffix == ".md"
        return len(parts) == 3 and parts[1] == "latex" and suffix in {".tex", ".pdf"}
    if parts[0] == "runs":
        return len(parts) > 1 and parts[1] == "pathmnist-m4" and suffix == ".json"
    if parts[0] == "state":
        if len(parts) < 2 or parts[1] != "workflow":
            return False
        if len(parts) == 3:
            return name in {
                "state/workflow/formal-paper-v2.json",
                "state/workflow/formal-paper-paid-smoke.json",
            }
        return len(parts) > 3 and parts[2] == "formal-paper-v2"
    return name in _ROOT_FILES


def _candidate_files(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for current, directories, files in os.walk(project_root):
        directories[:] = sorted(d for d in directories if d not in _PRUNED_DIRECTORIES)
        base = Path(current)
        for filename in sorted(files):
            path = base / filename
            relative = path.relative_to(project_root)
            if relative.as_posix().count("/") == 0 and relative.name not in _ROOT_FILES:
                continue
            if path.is_file() and _included(relative):
                candidates.append(path)
    return sorted(candidates, key=lambda path: path.relative_to(project_root).as_posix())


def build_archive(project_root: Path, output: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")
    candidates = _candidate_files(project_root)
    entries = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in candidates
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "project": "path-ai-scientist",
        "scope": "pathmnist-m4-m5-lightweight",
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, entry in zip(candidates, entries):
            info = zipfile.ZipInfo(
                f"path-ai-scientist/{entry['path']}", date_time=_FIXED_TIMESTAMP
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        manifest_info = zipfile.ZipInfo(
            "path-ai-scientist/manifest.json", date_time=_FIXED_TIMESTAMP
        )
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(manifest_info, json.dumps(manifest, indent=2, sort_keys=True))
    manifest_path = output.with_name(output.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "archive": str(output),
        "archive_sha256": _sha256(output),
        "archive_bytes": output.stat().st_size,
        "file_count": manifest["file_count"],
        "manifest": str(manifest_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/archives/pathmnist-m4-m5-archive.zip")
    )
    arguments = parser.parse_args()
    print(json.dumps(build_archive(arguments.project_root, arguments.output), indent=2))
