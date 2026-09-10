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
        self.messages = []

    def call_text(self, role, request, system, prompt):
        self.calls.append(request)
        self.messages.append((role, system, prompt))
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
    draft_payload = json.loads(provider.messages[0][2])
    draft_prompt = draft_payload["messages"][0]["content"]
    assert "research_question" in draft_prompt
    assert "figure_hashes" not in draft_prompt
    assert "Organize the manuscript around the research question" in draft_prompt
    assert "Check scientific argument" in provider.messages[1][2]
    assert "Check scientific argument" in provider.messages[2][2]
    # Full scientific evidence remains available without the cache identity envelope.
    assert json.dumps(analysis()["evidence"], ensure_ascii=False) in draft_prompt


def test_initial_manuscript_validation_uses_bounded_cached_repair(tmp_path, monkeypatch):
    from pathmnist import autonomous_postprocess, autonomous_pdf

    monkeypatch.setattr(autonomous_postprocess, "_commit_stage", lambda *_: None)
    monkeypatch.setattr(
        autonomous_pdf,
        "_compile",
        lambda directory, name, **kwargs: (directory / name).with_suffix(".pdf").write_bytes(
            b"%PDF" + b"x" * 12000
        ),
    )

    class RepairingProvider(Provider):
        def call_text(self, role, request, system, prompt):
            result, metadata = super().call_text(role, request, system, prompt)
            paper_calls = sum(message[0] == "paper_writer" for message in self.messages)
            if role == "paper_writer" and paper_calls == 1:
                result = "```latex\n" + TEX.replace("0.8", "0.99") + "\n```"
            return result, metadata

    provider = RepairingProvider()
    project = Path(__file__).resolve().parents[1]
    pub.run(project, tmp_path, analysis(), provider)
    manifest = pub.read(tmp_path / "paper/publication_manifest.json")
    draft = tmp_path / manifest["stages"]["paper_written"][0]["source"]
    assert "0.99" not in draft.read_text(encoding="utf-8")
    assert any("draft-repair" in path.name for path in draft.parent.glob("*.json"))


def test_research_brief_preserves_question_and_does_not_invent_missing_basis():
    value = analysis()
    value["task_id"] = "private-task"
    value["evidence"]["research_contract"] = {
        "research_question": "Does smoothing improve this CNN?",
        "success_criteria": ["accuracy difference >= 0.01"],
        "repeat_plan": {"seeds": [0]},
        "baseline": {"name": "CNN"},
    }
    value["finding"] = "Observed test accuracy difference 0.02; descriptive only."
    brief = pub.research_brief(value)
    assert brief["research_question"] == "Does smoothing improve this CNN?"
    assert brief["conclusion_limits"]["repeat_plan"] == {"seeds": [0]}
    assert brief["observed_finding"] == value["finding"]
    assert brief["literature_basis"][0]["citation_key"] == "R1"
    assert brief["literature_basis"][0]["abstract"].startswith("Not provided")
    assert brief["comparison_design"]["interventions"].startswith("Not provided")
    assert "private-task" not in json.dumps(brief)


@pytest.mark.parametrize("stage", pub.STAGES)
def test_resume_rejects_modified_committed_output_before_cache_repair(tmp_path, monkeypatch, stage):
    from pathmnist import autonomous_postprocess, autonomous_pdf

    monkeypatch.setattr(autonomous_postprocess, "_commit_stage", lambda *_: None)

    def compile(directory, name, **kwargs):
        (directory / name).with_suffix(".pdf").write_bytes(b"%PDF")
        return "ok"

    monkeypatch.setattr(autonomous_pdf, "_compile", compile)
    provider = Provider()
    project = Path(__file__).resolve().parents[1]
    pub.run(project, tmp_path, analysis(), provider)
    row = pub.artifacts(tmp_path, stage)[0]
    path = tmp_path / row["source"]
    path.write_text("modified", encoding="utf-8")
    calls = len(provider.calls)
    with pytest.raises(ValueError, match="hash mismatch") as error:
        pub.run(project, tmp_path, analysis(), provider)
    assert row["source"] in str(error.value)
    assert path.read_text() == "modified"
    assert len(provider.calls) == calls


def test_changed_evidence_requires_version_review_without_new_calls(tmp_path, monkeypatch):
    from pathmnist import autonomous_postprocess, autonomous_pdf

    monkeypatch.setattr(autonomous_postprocess, "_commit_stage", lambda *_: None)

    def compile(directory, name, **kwargs):
        (directory / name).with_suffix(".pdf").write_bytes(b"%PDF")
        return "ok"

    monkeypatch.setattr(autonomous_pdf, "_compile", compile)
    provider = Provider()
    project = Path(__file__).resolve().parents[1]
    pub.run(project, tmp_path, analysis(), provider)
    changed = analysis()
    changed["evidence"]["validation_metrics"]["accuracy"] = 0.5
    with pytest.raises(ValueError, match="new-version review"):
        pub.run(project, tmp_path, changed, provider)
    assert len(provider.calls) == 4


def test_backend_only_change_migrates_existing_review(tmp_path, monkeypatch):
    import hashlib
    from pathmnist import autonomous_postprocess, autonomous_pdf

    monkeypatch.setattr(autonomous_postprocess, "_commit_stage", lambda *_: None)
    monkeypatch.setattr(
        autonomous_pdf,
        "_compile",
        lambda directory, name, **kwargs: (directory / name).with_suffix(".pdf").write_bytes(b"%PDF"),
    )
    provider = Provider()
    project = Path(__file__).resolve().parents[1]
    value = analysis()
    pub.run(project, tmp_path, value, provider)
    manifest_path = tmp_path / "paper/publication_manifest.json"
    manifest = pub.read(manifest_path)
    previous_version = "previous-backend"
    identity = json.dumps(
        {"analysis": value, "version": previous_version, "figure_hashes": {}},
        sort_keys=True,
        ensure_ascii=False,
    )
    manifest["backend_version"] = previous_version
    manifest["input_sha256"] = hashlib.sha256(identity.encode()).hexdigest()
    pub.save(manifest_path, manifest)
    calls = len(provider.calls)
    pub.run(project, tmp_path, value, provider)
    assert len(provider.calls) == calls
    assert pub.read(manifest_path)["backend_version"] != previous_version


def test_changed_source_figure_blocks_publication_resume(tmp_path, monkeypatch):
    from pathmnist import autonomous_postprocess, autonomous_pdf

    monkeypatch.setattr(autonomous_postprocess, "_commit_stage", lambda *_: None)

    def compile(directory, name, **kwargs):
        (directory / name).with_suffix(".pdf").write_bytes(b"%PDF")
        return "ok"

    monkeypatch.setattr(autonomous_pdf, "_compile", compile)
    source = tmp_path / "result.png"
    source.write_bytes(b"original offline figure")
    value = analysis()
    value["figures"]["figures"] = [{"path": "result.png"}]
    provider = Provider()
    project = Path(__file__).resolve().parents[1]
    pub.run(project, tmp_path, value, provider)
    source.write_bytes(b"substituted offline figure")
    with pytest.raises(ValueError, match="new-version review"):
        pub.run(project, tmp_path, value, provider)
    assert len(provider.calls) == 4


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


def test_native_template_figure_paths_are_made_explicit():
    generated = TEX.replace(
        r"\begin{document}",
        "\\begin{filecontents}{references.bib}\nunsafe embedded bib\n\\end{filecontents}\n"
        "\\graphicspath{{../figures/}} % inherited upstream directive\n"
        r"\begin{document}\includegraphics[width=1in]{result.png}",
    )
    normalized = pub.normalize_tex_paths(generated, ["figures/result.png"])
    assert r"\begin{filecontents}" not in normalized
    assert r"\graphicspath" not in normalized
    assert r"\includegraphics[width=1in]{figures/result.png}" in normalized
    pub.validate_tex(normalized, ["R1"], ["figures/result.png"])


@pytest.mark.parametrize(
    "alias",
    ["result.png", "result", "figures/result", "./figures/result.png"],
)
def test_verified_figure_aliases_are_normalized_without_model_retry(alias):
    generated = TEX.replace(
        r"\begin{document}",
        rf"\begin{{document}}\includegraphics{{{alias}}}",
    )
    normalized = pub.normalize_tex_paths(generated, ["figures/result.png"])
    assert r"\includegraphics{figures/result.png}" in normalized
    pub.validate_tex(normalized, ["R1"], ["figures/result.png"])


def test_ambiguous_figure_stem_is_not_silently_selected():
    generated = TEX.replace(
        r"\begin{document}",
        r"\begin{document}\includegraphics{result}",
    )
    normalized = pub.normalize_tex_paths(
        generated, ["figures/result.png", "figures/result.pdf"]
    )
    with pytest.raises(ValueError, match=r"Unknown figure reference.*result"):
        pub.validate_tex(
            normalized,
            ["R1"],
            ["figures/result.png", "figures/result.pdf"],
        )


def test_generated_tables_are_constrained_once_without_changing_cells():
    table = (
        r"\begin{tabular}{lcc}" "\n"
        r"Mean Difference & 0.0422 & 0.0743 \\" "\n"
        r"\end{tabular}"
    )
    generated = TEX.replace(r"\begin{document}", r"\begin{document}" + table)
    normalized = pub.normalize_tex_paths(generated, [])
    assert normalized.count(r"\resizebox{\columnwidth}{!}{%") == 1
    assert table in normalized
    assert pub.normalize_tex_paths(normalized, []) == normalized


def test_literature_authors_are_rendered_as_bibtex_names():
    assert pub.bibtex_authors("Taymaz Akan, Richa Aishwarya, et al.") == (
        "Taymaz Akan and Richa Aishwarya and others"
    )
    assert pub.bibtex_authors(["Ada Lovelace", "Alan Turing"]) == (
        "Ada Lovelace and Alan Turing"
    )


def test_translation_preserves_environment_and_citation_tokens():
    pieces, indices = pub.translate_segments(r"\begin{abstract}Accuracy 0.8\end{abstract}\cite{R1}")
    assert [pieces[i] for i in indices] == ["Accuracy 0.8"]


def test_translation_safety_allows_reordered_numbers_and_preserved_latex():
    assert pub.translation_segment_is_safe(
        r"Counts 1041, 1057 for classes 0--8; subset 20\%.",
        r"类别0--8的计数为1057、1041；子集20\%。",
    )
    assert not pub.translation_segment_is_safe("Accuracy 0.8", "准确率0.9")
    assert not pub.translation_segment_is_safe(r"subset 20\%", "子集20%")
    assert pub.translation_segment_is_safe(r"baseline vs.\ smoothing", "基线与平滑")


def test_compile_source_adds_cleveref_for_preserved_cref():
    source = "\\documentclass{article}\n\\begin{document}See \\cref{fig:a}.\\end{document}"
    normalized = pub.normalize_compile_source(source)
    assert r"\usepackage{cleveref}" in normalized
    assert normalized.count(r"\usepackage{cleveref}") == 1
    assert pub.normalize_compile_source(normalized) == normalized


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


def test_numeric_audit_accepts_magnitude_of_evidenced_decrease():
    manuscript = TEX.replace("0.8", "0.61")
    pub.validate_evidence_numbers(manuscript, {"mean_difference": -0.0061281337})


def test_numeric_audit_ignores_latex_layout_dimensions_but_not_scientific_units():
    layout = TEX.replace(
        r"\begin{document}",
        r"\begin{document}\vskip 0.3in\hspace{0.25\linewidth}",
    )
    pub.validate_evidence_numbers(layout, analysis()["evidence"])
    scientific = TEX.replace("Accuracy 0.8", "Accuracy 0.8 at a measured 0.3 mm")
    with pytest.raises(ValueError, match=r"0\.3"):
        pub.validate_evidence_numbers(scientific, analysis()["evidence"])


def test_numeric_audit_accepts_explicitly_derived_class_ratio():
    from pathmnist.publication import publication_dataset_profile

    evidence = {
        "dataset": publication_dataset_profile(
            {"name": "binary", "class_counts": {"train": {"0": 1214, "1": 3494}}}
        )
    }
    manuscript = TEX.replace("Accuracy 0.8", "Class imbalance was 2.88:1")
    pub.validate_evidence_numbers(manuscript, evidence)


def test_review_schema_checks_list_members_and_normalizes_decision_case():
    review = pub.normalize_review(
        {
            "Summary": "Summary",
            "Weaknesses": ["Limited repeats"],
            "Questions": ["None"],
            "Decision": "reject",
        }
    )
    assert review["Decision"] == "Reject"
    with pytest.raises(ValueError, match="list of strings"):
        pub.normalize_review(
            {
                "Summary": "Summary",
                "Weaknesses": [1],
                "Questions": [],
                "Decision": "Reject",
            }
        )


def test_final_compile_removes_icml_review_ruler_without_claiming_acceptance():
    source = r"""\documentclass{article}
\usepackage{icml2025}
\begin{document}
Report
\end{document}"""

    compiled = pub.normalize_compile_source(source)

    assert r"\ClearShipoutPicture" in compiled
    assert r"\renewcommand{\Notice@String}{}" in compiled
    assert r"\def\isaccepted{1}" not in compiled
    assert r"\usepackage[accepted]{icml2025}" not in compiled
    assert r"\icmlcorrespondingauthor{Anonymous}{anonymous@example.invalid}" in compiled


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
