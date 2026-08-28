import json
from pathlib import Path

import pytest

from pathmnist import upstream_publication as pub


TEX = r"""\documentclass{article}
\usepackage{natbib}
\begin{document}
\begin{abstract}Single comparison, descriptive only. Accuracy 0.8.\end{abstract}
\section{Results}No clinical claims.\cite{R1}
\section{AI assistance}AI generated manuscript.
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}"""


class Provider:
    def __init__(self):
        self.calls = []

    def call_text(self, role, request, system, prompt):
        self.calls.append(request)
        if role == "reviewer":
            result = (
                "```json\n"
                + json.dumps(
                    {
                        "Summary": "Valid single comparison",
                        "Weaknesses": [],
                        "Questions": [],
                        "Decision": "Accept",
                    }
                )
                + "\n```"
            )
        elif "Translate the JSON list" in system:
            result = json.dumps(json.loads(prompt))
        else:
            result = "```latex\n" + TEX + "\n```"
        return result, {}


def analysis():
    return {
        "evidence": {
            "literature": {
                "references": [{"title": "Reference", "year": 2023, "doi": "10.1/test"}]
            },
            "validation_metrics": {"accuracy": 0.8},
            "research_contract": {"repeat_count": 1},
            "contract_results": {"hypothesis_supported": False},
            "timing": None,
        },
        "figures": {"figures": []},
    }


def test_real_native_steps_resume_without_new_provider_calls(tmp_path, monkeypatch):
    from pathmnist import autonomous_postprocess, autonomous_pdf

    stages = []
    monkeypatch.setattr(
        autonomous_postprocess, "_commit_stage", lambda path, stage: stages.append(stage)
    )

    def compile(directory, name, **kwargs):
        (directory / name).with_suffix(".pdf").write_bytes(b"%PDF" + b"x" * 12000)
        return "compiled"

    monkeypatch.setattr(autonomous_pdf, "_compile", compile)
    provider = Provider()
    project = Path(__file__).resolve().parents[1]
    pub.run(project, tmp_path, analysis(), provider)
    count = len(provider.calls)
    assert count == 4  # Native draft, native review, native reflection, translation.
    pub.run(project, tmp_path, analysis(), provider)
    assert len(provider.calls) == count
    assert stages[-1] == "translation_completed"
    assert pub.artifacts(tmp_path, "revision_completed")[0]["source"].endswith("template.tex")


def test_unknown_backend_and_path_escape_fail_closed(tmp_path):
    assert pub.backend({}) == "legacy_local"
    with pytest.raises(ValueError):
        pub.backend({"publication_backend": "wrong"})
    with pytest.raises(ValueError):
        pub.safe_path(tmp_path, "../escape")


@pytest.mark.parametrize(
    "replacement",
    [
        r"\cite{unknown}",
        r"\citep[see][p. 2]{unknown}",
        r"\input{/secret}",
        r"\includegraphics{unknown.png}",
        "TODO",
    ],
)
def test_invalid_native_artifacts_are_rejected(replacement):
    with pytest.raises(ValueError):
        pub.validate_tex(TEX + replacement, ["R1"], [])


def test_translation_preserves_environment_and_citation_tokens():
    pieces, indices = pub.translate_segments(r"\begin{abstract}Accuracy 0.8\end{abstract}\cite{R1}")
    assert [pieces[i] for i in indices] == ["Accuracy 0.8"]


@pytest.mark.parametrize("boundary", pub.STAGES)
def test_native_pipeline_recovers_after_artifact_commit(tmp_path, monkeypatch, boundary):
    from pathmnist import autonomous_postprocess, autonomous_pdf

    interrupted = []

    def commit(path, stage):
        if stage == boundary and not interrupted:
            interrupted.append(stage)
            raise RuntimeError("injected crash after artifact receipt")

    def compile(directory, name, **kwargs):
        (directory / name).with_suffix(".pdf").write_bytes(b"%PDF" + b"x" * 12000)
        return "ok"

    monkeypatch.setattr(autonomous_postprocess, "_commit_stage", commit)
    monkeypatch.setattr(autonomous_pdf, "_compile", compile)
    provider = Provider()
    project = Path(__file__).resolve().parents[1]
    with pytest.raises(RuntimeError, match="injected"):
        pub.run(project, tmp_path, analysis(), provider)
    pub.run(project, tmp_path, analysis(), provider)
    assert len(provider.calls) == 4
    assert len(set(provider.calls)) == 4


def test_gateway_response_receipt_survives_missing_projection(tmp_path):
    gateway = pub.Gateway(Provider(), tmp_path, "version", "paper_writer")
    first = gateway("draft")
    next(tmp_path.glob("*.txt")).unlink()
    assert gateway("draft") == first
    assert len(gateway.provider.calls) == 1


def test_numeric_hallucination_is_rejected():
    with pytest.raises(ValueError, match="absent from evidence"):
        pub.validate_evidence_numbers(TEX.replace("0.8", "0.99"), analysis()["evidence"])


def test_compiler_receipt_recovers_without_recompile(tmp_path):
    (tmp_path / "template.tex").write_text(TEX)
    (tmp_path / "references.bib").write_text("reference")
    calls = []

    def compile(directory, name, **kwargs):
        calls.append(name)
        (directory / name).with_suffix(".pdf").write_bytes(b"%PDF")
        return "ok"

    pub.compile_cached(tmp_path, "template.tex", compiler=compile)
    pub.compile_cached(tmp_path, "template.tex", compiler=compile)
    assert calls == ["template.tex"]


def test_budget_failure_never_falls_back_to_native_client(tmp_path):
    class Failed:
        def call_text(self, *args):
            raise RuntimeError("budget exhausted")

    with pytest.raises(RuntimeError, match="budget exhausted"):
        pub.Gateway(Failed(), tmp_path, "v", "paper_writer")("draft")


def test_two_scoped_gateways_do_not_share_messages(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    ga = pub.Gateway(Provider(), a, "v", "paper_writer")
    gb = pub.Gateway(Provider(), b, "v", "paper_writer")
    _, history = ga("first")
    _, other = gb("second")
    assert history[0]["content"] == "first"
    assert other[0]["content"] == "second"


def test_native_to_archive_and_archive_commit_recovery(tmp_path, monkeypatch):
    """Publication integration; pre-publication evidence acceptance is a boundary stub."""
    from pathmnist import autonomous_postprocess, autonomous_pdf, autonomous_acceptance
    from pathmnist.research_stages import RESEARCH_STAGES
    import zipfile

    pub.save(
        tmp_path / "task.json",
        {
            "task_id": "offline",
            "publication_backend": "upstream_v2",
            "completed_stage": "figures_generated",
            "stages": {
                s: "completed"
                if RESEARCH_STAGES.index(s) <= RESEARCH_STAGES.index("figures_generated")
                else "waiting"
                for s in RESEARCH_STAGES
            },
        },
    )

    def accept(root, stage, **kwargs):
        for s in pub.STAGES:
            if RESEARCH_STAGES.index(s) <= RESEARCH_STAGES.index(stage):
                pub.artifacts(root, s)

    monkeypatch.setattr(autonomous_postprocess, "require_task", accept)
    monkeypatch.setattr(autonomous_acceptance, "require_task", accept)
    compiles = []

    def compile(directory, name, **kwargs):
        compiles.append(str(directory / name))
        (directory / name).with_suffix(".pdf").write_bytes(b"%PDF" + b"x" * 12000)
        return "ok"

    monkeypatch.setattr(autonomous_pdf, "_compile", compile)
    provider = Provider()
    project = Path(__file__).resolve().parents[1]
    pub.run(project, tmp_path, analysis(), provider)
    commit = autonomous_postprocess._commit_stage
    failed = []

    def interrupted(path, stage):
        if stage == "archived" and not failed:
            failed.append(True)
            raise RuntimeError("archive commit crash")
        commit(path, stage)

    monkeypatch.setattr(autonomous_postprocess, "_commit_stage", interrupted)
    with pytest.raises(RuntimeError, match="archive commit crash"):
        pub.build_pdfs(project, tmp_path)
    before = list(compiles)
    result = pub.build_pdfs(project, tmp_path)
    assert compiles == before
    assert len(provider.calls) == 4
    assert pub.read(tmp_path / "task.json")["completed_stage"] == "archived"
    with zipfile.ZipFile(result["archive"]) as archive:
        assert json.loads(archive.read("task.json"))["completed_stage"] == "archived"
        assert any(n.endswith("translation.pdf") for n in archive.namelist())


def test_compiler_timeout_cleans_named_container(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from pathmnist import autonomous_pdf

    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 300)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(autonomous_pdf.subprocess, "run", execute)
    with pytest.raises(subprocess.TimeoutExpired):
        autonomous_pdf._compile(tmp_path, "paper.tex", bibtex=True)
    assert calls[1][:3] == ["docker", "rm", "--force"]
    assert calls[1][-1] == calls[0][calls[0].index("--name") + 1]


def test_empty_manifest_stage_rejected(tmp_path):
    pub.save(
        tmp_path / "paper/publication_manifest.json",
        {"backend": "upstream_v2", "stages": {"paper_written": []}},
    )
    with pytest.raises(ValueError, match="exactly one"):
        pub.artifacts(tmp_path, "paper_written")


def test_init_new_backend_and_legacy_resume(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from pathmnist.cli import _autonomous_init

    dataset = tmp_path / "dataset.npz"
    arrays = {}
    for split in ("train", "val", "test"):
        arrays[split + "_images"] = np.zeros((4, 28, 28, 3), dtype=np.uint8)
        arrays[split + "_labels"] = np.array([[0], [1], [0], [1]])
    np.savez(dataset, **arrays)
    args = SimpleNamespace(
        state_root=tmp_path / "state",
        task_id="new",
        resume=False,
        seed=0,
        direction="Offline classification",
        dataset_path=dataset,
    )
    _autonomous_init(args)
    task_path = tmp_path / "state/new/task.json"
    task = pub.read(task_path)
    assert task["publication_backend"] == "upstream_v2"
    del task["publication_backend"]
    pub.save(task_path, task)
    args.resume = True
    _autonomous_init(args)
    assert pub.backend(pub.read(task_path)) == "legacy_local"


def test_original_native_entry_reuses_extracted_steps():
    import ast

    project = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (project / "vendor/AI-Scientist-v2/ai_scientist/perform_writeup.py").read_text(
            encoding="utf-8"
        )
    )
    entry = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "perform_writeup"
    )
    calls = {
        n.func.id
        for n in ast.walk(entry)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert {"writeup_step", "reflection_step"} <= calls


@pytest.mark.parametrize(
    "formula",
    [
        r"\[\mathrm{accuracy}=0.8\]",
        r"\(\mathrm{accuracy}\)",
        r"\begin{equation}\mathrm{accuracy}=0.8\end{equation}",
    ],
)
def test_translation_protects_display_and_inline_math(formula):
    pieces, indices = pub.translate_segments("Results " + formula)
    assert [pieces[i] for i in indices] == ["Results "]


def test_compiler_cache_invalidates_changed_figure(tmp_path):
    (tmp_path / "template.tex").write_text(TEX)
    (tmp_path / "references.bib").write_text("reference")
    (tmp_path / "figures").mkdir()
    figure = tmp_path / "figures/result.png"
    figure.write_bytes(b"first")
    calls = []

    def compile(directory, name, **kwargs):
        calls.append(name)
        (directory / name).with_suffix(".pdf").write_bytes(b"%PDF")
        return "ok"

    pub.compile_cached(tmp_path, "template.tex", compiler=compile)
    figure.write_bytes(b"changed")
    pub.compile_cached(tmp_path, "template.tex", compiler=compile)
    assert len(calls) == 2
