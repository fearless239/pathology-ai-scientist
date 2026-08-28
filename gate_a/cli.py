from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline, select_live_models
from .provider import probe_zhipu_model_list


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the constrained AI-Scientist-v2 Gate A chain"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/gate_a.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="Validate live model availability and price without inference"
    )
    preflight.add_argument("--show-models", action="store_true")

    run = subparsers.add_parser("run", help="Execute the Gate A chain")
    run.add_argument(
        "--provider",
        choices=["fixture", "openrouter", "zhipu", "openai_compatible"],
        required=True,
    )
    run.add_argument("--output-root", type=Path, default=Path("runs/gate-a"))
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--confirm-paid-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config.resolve())
    project_root = Path(__file__).resolve().parent.parent
    vendor_root = project_root / "vendor" / "AI-Scientist-v2"

    if args.command == "preflight":
        selected = select_live_models(config)
        provider_name = config.provider.name
        if provider_name != "openrouter":
            listed = probe_zhipu_model_list(config)
            if listed is None:
                print(
                    "Note: this provider exposes no usable model list endpoint; "
                    "model availability is proven by the paid smoke run.",
                    file=sys.stderr,
                )
            else:
                missing = [
                    model.model_id
                    for model in selected.values()
                    if model.model_id not in listed
                ]
                if missing:
                    print(
                        f"Warning: configured models absent from the listed catalog: "
                        f"{', '.join(sorted(set(missing)))}",
                        file=sys.stderr,
                    )
        if args.show_models:
            print(
                json.dumps(
                    {role: model.model_id for role, model in selected.items()}, indent=2
                )
            )
        else:
            print(f"Provider '{provider_name}' model and price preflight passed.")
        return 0

    paid_providers = {"openrouter", "zhipu", "openai_compatible"}
    if args.provider in paid_providers and not args.confirm_paid_smoke:
        print(
            "Refusing paid inference: pass --confirm-paid-smoke only after reviewing the offline report.",
            file=sys.stderr,
        )
        return 2
    if args.provider in paid_providers and args.repeat != 1:
        print("Paid smoke may run exactly once.", file=sys.stderr)
        return 2
    if not 1 <= args.repeat <= 2:
        print("--repeat must be 1 or 2.", file=sys.stderr)
        return 2

    completed: list[Path] = []
    for _ in range(args.repeat):
        run_dir = run_pipeline(
            config=config,
            output_root=args.output_root.resolve(),
            provider_mode=args.provider,
            vendor_root=vendor_root,
        )
        completed.append(run_dir)
        print(run_dir)

    if len(completed) == 2:
        reports = [
            json.loads((run_dir / "acceptance.json").read_text(encoding="utf-8"))
            for run_dir in completed
        ]
        repeatable = reports[0]["summary"] == reports[1]["summary"]
        result = {
            "runs": [str(path) for path in completed],
            "same_structural_summary": repeatable,
        }
        destination = args.output_root.resolve() / "offline-repeatability.json"
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not repeatable:
            print("Offline clean-container runs differed.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
