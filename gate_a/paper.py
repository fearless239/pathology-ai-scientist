from __future__ import annotations

import re
from pathlib import Path
from typing import Any


PAPER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        key: {"type": "string", "minLength": 1}
        for key in [
            "title",
            "abstract",
            "introduction",
            "method",
            "results",
            "limitations",
            "conclusion",
        ]
    },
    "required": [
        "title",
        "abstract",
        "introduction",
        "method",
        "results",
        "limitations",
        "conclusion",
    ],
    "additionalProperties": False,
}


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "decision": {"type": "string", "minLength": 1},
        "summary": {"type": "string", "minLength": 1},
        "strengths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "weaknesses": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "required_changes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overall_score",
        "decision",
        "summary",
        "strengths",
        "weaknesses",
        "required_changes",
    ],
    "additionalProperties": False,
}


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_latex(content: dict[str, Any], figure_name: str) -> str:
    clean = {key: _latex_escape(str(value).strip()) for key, value in content.items()}
    figure = ""
    if re.fullmatch(r"[A-Za-z0-9_.-]+\.png", figure_name):
        figure = rf"""
\begin{{figure}}[ht]
\centering
\includegraphics[width=0.78\linewidth]{{{figure_name}}}
\caption{{Validation metric generated exclusively from the saved experiment artifact.}}
\label{{fig:metric}}
\end{{figure}}
"""
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage[T1]{{fontenc}}
\title{{{clean['title']}}}
\author{{AI-Scientist-v2 Gate A automated draft}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{clean['abstract']}

\textbf{{Disclosure:}} This manuscript was machine-generated using The AI Scientist-v2
through OpenRouter. It is a preliminary engineering artifact. Human review is required.
\end{{abstract}}

\section{{Introduction}}
{clean['introduction']}

\section{{Method}}
{clean['method']}

\section{{Results}}
{clean['results']}
{figure}

\section{{Limitations}}
{clean['limitations']}

\section{{Conclusion}}
{clean['conclusion']}

\section*{{AI involvement and responsible use}}
All research planning, code drafting, paper drafting, and automated review in this smoke run
involved language models. The experiment used synthetic data only. The artifact makes no
clinical, diagnostic, or real-world validity claim. Human review is required before reuse.
\end{{document}}
"""


def validate_pdf(path: Path) -> None:
    if not path.exists() or path.stat().st_size < 1024:
        raise ValueError("Compiled PDF is missing or unexpectedly small")
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError("Compiled paper does not have a PDF signature")
