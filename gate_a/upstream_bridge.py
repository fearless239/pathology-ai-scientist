from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .provider import ChatProvider, ProviderError


class UpstreamBridgeError(RuntimeError):
    """Raised when the constrained upstream node cannot be produced."""


class UpstreamMinimalBFTS:
    """One-root-node BFTS smoke adapter built on the upstream MinimalAgent.

    The upstream source is imported from the read-only vendor snapshot. A subclass narrows only
    the environment and artifact contract; upstream `_draft` and plotting prompt construction
    remain in use.
    """

    def __init__(self, vendor_root: Path, provider: ChatProvider, timeout_seconds: int):
        self.vendor_root = vendor_root.resolve()
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        if str(self.vendor_root) not in sys.path:
            sys.path.insert(0, str(self.vendor_root))

        from ai_scientist.treesearch.backend import compile_prompt_to_md
        from ai_scientist.treesearch.parallel_agent import MinimalAgent
        from ai_scientist.treesearch.utils.response import (
            extract_code,
            extract_text_up_to_code,
        )

        self._compile_prompt = compile_prompt_to_md
        self._extract_code = extract_code
        self._extract_text = extract_text_up_to_code
        outer = self

        class GateAMinimalAgent(MinimalAgent):
            active_role = "experiment_code"
            active_request_id = "bfts-node-000-code"

            @property
            def _prompt_environment(self):
                return {
                    "Installed Packages": (
                        "Only Python 3.11, numpy, matplotlib, and scikit-learn are installed. "
                        "Do not import or install anything else. The experiment has no network."
                    )
                }

            @property
            def _prompt_impl_guideline(self):
                return {
                    "Implementation guideline": [
                        "Use only a deterministic, CPU-only synthetic experiment.",
                        "Finish within 60 seconds and use a fixed random seed.",
                        "Do not access the network, environment variables, subprocesses, or paths outside the current directory.",
                        "Start with: import os; working_dir = os.path.join(os.getcwd(), 'working'); os.makedirs(working_dir, exist_ok=True).",
                        "Save raw plottable data to working/experiment_data.npy with numpy.save.",
                        "Save a JSON object to working/metrics.json containing primary_metric, value, and n.",
                        "Print the primary metric to standard output.",
                        "Do not use an if __name__ == '__main__' guard.",
                        "This is an engineering smoke test; do not claim novelty or external validity.",
                    ]
                }

            def plan_and_code_query(self, prompt, retries=2):
                compiled = outer._compile_prompt(prompt)
                last = ""
                for attempt in range(retries):
                    request_id = (
                        self.active_request_id
                        if attempt == 0
                        else f"{self.active_request_id}-retry-{attempt}"
                    )
                    last, _ = outer.provider.call_text(
                        self.active_role,
                        request_id,
                        "Follow the requested format exactly. Return executable code only in the single requested Python block.",
                        compiled,
                    )
                    code = outer._extract_code(last)
                    plan = outer._extract_text(last)
                    if code and plan:
                        return plan, code
                raise ProviderError(
                    f"Unable to extract plan and code for {self.active_role}"
                )

            def _generate_plotting_code(
                self, node, working_dir, plot_code_from_prev_stage=None
            ):
                self.active_role = "plotting"
                self.active_request_id = "bfts-node-000-plot"
                try:
                    return super()._generate_plotting_code(
                        node, working_dir, plot_code_from_prev_stage
                    )
                finally:
                    self.active_role = "experiment_code"
                    self.active_request_id = "bfts-node-000-code"

        self._agent_class = GateAMinimalAgent

    def generate(self, idea: dict[str, Any], workspace: Path):
        config = SimpleNamespace(
            agent=SimpleNamespace(k_fold_validation=1, data_preview=False),
            exec=SimpleNamespace(timeout=self.timeout_seconds),
            experiment=SimpleNamespace(num_syn_datasets=1),
        )
        task = json.dumps(idea, indent=2, ensure_ascii=False)
        agent = self._agent_class(
            task_desc=task,
            cfg=config,
            memory_summary="Gate A is configured for exactly one root node.",
            evaluation_metrics=["validation_accuracy"],
            stage_name="1_initial_implementation_1_preliminary",
        )
        node = agent._draft()
        if not node.code or not node.plan:
            raise UpstreamBridgeError("Upstream MinimalAgent returned an empty node")
        plotting_code = agent._generate_plotting_code(node, str(workspace / "working"))
        if not plotting_code:
            raise UpstreamBridgeError("Upstream plotting prompt returned empty code")
        return node, plotting_code
