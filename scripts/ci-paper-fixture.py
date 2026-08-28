"""Build a tiny illustrated paper fixture for CI; writes only to the requested directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from pathmnist.paper_disclosure import ensure_disclosure
    from pathmnist.paper_export import markdown_to_latex

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3], [0.4, 0.6, 0.7])
    ax.set(title="Fixture evidence", xlabel="Epoch", ylabel="Macro-F1")
    fig.tight_layout()
    fig.savefig(figures / "fixture.png", dpi=120)
    plt.close(fig)
    markdown = """# Offline illustrated paper fixture

## Results

![Fixture evidence](figures/fixture.png)

## References

* **[1]** Fixture reference. DOI: 10.0000/fixture.
"""
    markdown = ensure_disclosure(markdown, "en")
    (output / "paper.md").write_text(markdown, encoding="utf-8")
    (output / "paper.tex").write_text(markdown_to_latex(markdown), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
