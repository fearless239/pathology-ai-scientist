from __future__ import annotations

import hashlib
import ast
import sys
from pathlib import Path
from typing import Any


class UpstreamIntegrationError(RuntimeError):
    pass


class PathologyAIScientistV2Adapter:
    """Thin pathology specialization over imported AI-Scientist-v2 primitives.

    The vendored tree remains read-only. Domain constraints and artifact contracts
    stay in ``pathmnist`` while prompt compilation and the experiment-manager class
    identity come directly from the supplied upstream framework.
    """

    COMPONENTS = (
        "ai_scientist.treesearch.parallel_agent.MinimalAgent",
        "ai_scientist.treesearch.agent_manager.AgentManager",
        "ai_scientist.treesearch.backend.compile_prompt_to_md",
        "ai_scientist.perform_ideation_temp_free.FinalizeIdea contract",
    )

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.vendor_root = self.project_root / "vendor" / "AI-Scientist-v2"
        if not (self.vendor_root / "ai_scientist" / "treesearch" / "parallel_agent.py").is_file():
            raise UpstreamIntegrationError("Vendored AI-Scientist-v2 source is missing")
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))
        self.runtime_imported = False
        self.minimal_agent_module = "ai_scientist.treesearch.parallel_agent"
        self.agent_manager_module = "ai_scientist.treesearch.agent_manager"
        if sys.version_info >= (3, 11):
            from ai_scientist.treesearch.backend import compile_prompt_to_md
            from ai_scientist.treesearch.agent_manager import AgentManager
            from ai_scientist.treesearch.parallel_agent import MinimalAgent

            self.compile_prompt_to_md = compile_prompt_to_md
            self.agent_manager_module = AgentManager.__module__
            self.minimal_agent_module = MinimalAgent.__module__
            self.runtime_imported = True
        else:
            # The project and production image require Python 3.11. Keep local
            # read-only inspection usable on older host Python installations.
            self.compile_prompt_to_md = _compile_prompt_compat

    def compile_pathology_prompt(self, sections: dict[str, Any]) -> str:
        return self.compile_prompt_to_md(
            {
                "AI-Scientist-v2 task": "Finalize one feasible research idea",
                "Computational pathology specialization": sections,
            }
        )

    def framework_record(self) -> dict[str, Any]:
        manifest = self.project_root / "UPSTREAM_MANIFEST.sha256"
        return {
            "framework": "SakanaAI/AI-Scientist_v2",
            "integration": (
                "imported_upstream_components_with_pathology_domain_adapter"
                if self.runtime_imported
                else "source_verified_deferred_to_python311_runtime"
            ),
            "vendor_root": str(self.vendor_root),
            "components": list(self.COMPONENTS),
            "minimal_agent_module": self.minimal_agent_module,
            "agent_manager_module": self.agent_manager_module,
            "runtime_imported": self.runtime_imported,
            "required_python": ">=3.11",
            "upstream_manifest_sha256": _sha256(manifest),
            "specialization_boundary": {
                "upstream_reused": [
                    "prompt compilation",
                    "MinimalAgent/BFTS implementation through gate_a.upstream_bridge",
                    "AgentManager stage model",
                    "FinalizeIdea-compatible research contract",
                ],
                "pathology_extensions": [
                    "PathMNIST dataset adapter and split policy",
                    "pathology-safe research prompts",
                    "supported intervention registry",
                    "experiment artifact validation",
                    "claim-bounded paper and review workflow",
                ],
            },
        }

    def plotting_adapter(self) -> "UpstreamPlottingAdapter":
        return UpstreamPlottingAdapter(self.vendor_root)


class UpstreamPlottingAdapter:
    """Reuse upstream plot prompting without adopting its legacy directory contract."""

    def __init__(self, vendor_root: Path):
        self.vendor_root = vendor_root.resolve()
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))

    def build_prompt(self, evidence: dict[str, Any], idea_text: str) -> str:
        from ai_scientist.perform_plotting import build_aggregator_prompt

        return build_aggregator_prompt(json_dumps(evidence), idea_text)

    def extract_code(self, response: str) -> str:
        from ai_scientist.perform_plotting import extract_code_snippet

        return extract_code_snippet(response)

    def validate_code(self, code: str) -> None:
        """Reject capabilities that do not belong in the isolated plotting runner."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise UpstreamIntegrationError("Upstream plotting code is not valid Python") from exc
        forbidden_modules = {"httpx", "requests", "socket", "subprocess", "urllib"}
        forbidden_calls = {"eval", "exec", "compile", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name.split(".")[0] in forbidden_modules for alias in node.names):
                    raise UpstreamIntegrationError("Plotting code requests a network/process module")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in forbidden_modules:
                    raise UpstreamIntegrationError("Plotting code requests a network/process module")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in forbidden_calls:
                    raise UpstreamIntegrationError("Plotting code requests dynamic code execution")


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compile_prompt_compat(prompt: Any, depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    if isinstance(prompt, list):
        return "\n".join(f"- {item}" for item in prompt) + "\n"
    if isinstance(prompt, dict):
        parts = []
        for key, value in prompt.items():
            parts.append(f"{'#' * depth} {key}")
            parts.append(_compile_prompt_compat(value, depth + 1))
        return "\n".join(parts).strip() + "\n"
    return str(prompt).strip() + "\n"
