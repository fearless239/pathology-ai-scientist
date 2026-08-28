from __future__ import annotations

import ast
from pathlib import Path


class CodePolicyError(ValueError):
    """Raised before generated code is allowed into the experiment sandbox."""


FORBIDDEN_IMPORT_ROOTS = {
    "ctypes",
    "ftplib",
    "http",
    "paramiko",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "telnetlib",
    "urllib",
}

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
}


def validate_generated_code(code: str) -> None:
    if not code.strip():
        raise CodePolicyError("Generated code is empty")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodePolicyError(f"Generated code has invalid syntax: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
            forbidden = roots & FORBIDDEN_IMPORT_ROOTS
            if forbidden:
                raise CodePolicyError(
                    f"Forbidden import: {', '.join(sorted(forbidden))}"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                raise CodePolicyError(f"Forbidden import: {root}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise CodePolicyError(f"Forbidden call: {node.func.id}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr in {
                "system",
                "popen",
                "spawnl",
                "spawnv",
            }:
                raise CodePolicyError(f"Forbidden process call: os.{node.attr}")

    if "OPENROUTER_API_KEY" in code or "Authorization" in code:
        raise CodePolicyError("Generated code references online credentials")


def require_relative_artifact(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise CodePolicyError(f"Artifact escaped run directory: {resolved}")
    return resolved
