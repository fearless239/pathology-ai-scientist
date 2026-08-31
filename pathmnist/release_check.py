from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


FORBIDDEN_TRACKED_PREFIXES = ("runs/", "state/", "build/", "dist/", ".venv/")
FORBIDDEN_TRACKED_NAMES = {"pathmnist_64.npz", ".env"}
FORBIDDEN_TRACKED_SUFFIXES = {
    ".ckpt",
    ".key",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".pem",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".zip",
}
SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Authorization bearer": re.compile(r"Authorization\s*[:=]\s*[\"']?Bearer\s+[A-Za-z0-9._-]{16,}", re.I),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "local Windows path": re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "local WSL path": re.compile(r"/mnt/[a-z]/Users/[^/\s]+/", re.I),
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def check_release(repo: Path, max_bytes: int = 10 * 1024 * 1024) -> dict:
    repo = repo.resolve()
    tracked = [value for value in _git(repo, "ls-files").splitlines() if value]
    errors: list[str] = []
    warnings: list[str] = []
    largest: list[dict] = []
    for relative in tracked:
        normalized = relative.replace("\\", "/")
        path = repo / relative
        if normalized in FORBIDDEN_TRACKED_NAMES or normalized.startswith(FORBIDDEN_TRACKED_PREFIXES):
            errors.append(f"runtime/data file is tracked: {normalized}")
        if path.suffix.casefold() in FORBIDDEN_TRACKED_SUFFIXES:
            errors.append(f"generated/data binary is tracked: {normalized}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        largest.append({"path": normalized, "bytes": size})
        if size > max_bytes:
            errors.append(f"tracked file exceeds {max_bytes} bytes: {normalized} ({size})")
        if size > 2 * 1024 * 1024:
            warnings.append(f"large tracked file: {normalized} ({size})")
        if size <= 2 * 1024 * 1024 and path.suffix.casefold() not in {".pdf", ".png", ".jpg", ".jpeg", ".pt", ".npz"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"possible {label} in tracked file: {normalized}")
    for required in (
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "pyproject.toml",
        ".gitignore",
        ".dockerignore",
        ".github/workflows/ci.yml",
        ".env.example",
        "compose.yaml",
        "docker/demo.Dockerfile",
        "docs/ARCHITECTURE.md",
        "docs/RELEASE_CHECKLIST.md",
    ):
        if required not in tracked:
            errors.append(f"required release file is not tracked: {required}")
    root_license = repo / "LICENSE"
    upstream_license = repo / "vendor/AI-Scientist-v2/LICENSE"
    if root_license.is_file() and upstream_license.is_file():
        root_text = root_license.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip()
        upstream_text = (
            upstream_license.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip()
        )
        if root_text != upstream_text:
            errors.append("root LICENSE does not exactly preserve the vendored upstream license")
    return {
        "schema_version": 1,
        "passed": not errors,
        "tracked_files": len(tracked),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "largest_files": sorted(largest, key=lambda item: item["bytes"], reverse=True)[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GitHub release boundaries")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = check_release(args.repo)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
