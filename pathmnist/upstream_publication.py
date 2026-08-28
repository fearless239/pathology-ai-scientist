"""Versioned, evidence-bound adapter around native AI Scientist writing steps."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

from .artifact_cache import cached_artifact

STAGES = ("paper_written", "review_completed", "revision_completed", "translation_completed")


def backend(task):
    value = task.get("publication_backend", "legacy_local")
    if value not in ("legacy_local", "upstream_v2"):
        raise ValueError(f"Unknown publication backend: {value}")
    return value


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Publication path escaped task root")
    return path


def artifacts(root, stage):
    manifest = read(root / "paper/publication_manifest.json")
    if manifest["backend"] != "upstream_v2":
        raise ValueError("Publication backend mismatch")
    rows = manifest["stages"][stage]
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("Publication stage must contain exactly one artifact")
    for row in rows:
        path = safe_path(root, row["source"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Publication artifact hash mismatch")
        if (
            "pdf" in row
            and hashlib.sha256(safe_path(root, row["pdf"]).read_bytes()).hexdigest()
            != row["pdf_sha256"]
        ):
            raise ValueError("Publication PDF hash mismatch")
    return rows


class Gateway:
    """Preserve native message history without bypassing provider accounting."""

    def __init__(self, provider, directory, fingerprint, role):
        self.provider, self.directory, self.fingerprint, self.role = (
            provider,
            directory,
            fingerprint,
            role,
        )

    def __call__(self, msg, *, system_message="", msg_history=None, **kwargs):
        history = list(msg_history or [])
        payload = json.dumps(
            {
                "system": system_message,
                "messages": history + [{"role": "user", "content": msg}],
                "version": self.fingerprint,
                "role": self.role,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        response = cached_artifact(
            self.directory / f"{digest}.txt",
            payload,
            lambda _: self.provider.call_text(
                self.role, f"publication-{digest}", system_message, payload
            )[0],
        )
        return response, history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": response},
        ]


def validate_tex(text, citations, figures):
    if not all(token in text for token in (r"\begin{document}", r"\end{document}")):
        raise ValueError("Incomplete LaTeX document")
    if re.search(r"\b(TODO|FIXME|TITLE HERE|ABSTRACT HERE)\b", text):
        raise ValueError("Unresolved manuscript placeholder")
    if (
        re.search(r"\\(?:write18|input|include|openout|read|graphicspath)\b", text)
        or r"\begin{filecontents" in text
    ):
        raise ValueError("Generated LaTeX may not read external files or execute commands")
    used = set()
    for match in re.findall(r"\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}", text):
        used.update(x.strip() for x in match.split(","))
    if not used <= set(citations):
        raise ValueError("Unverified citation key")
    if citations and not used:
        raise ValueError("Manuscript has no verified citations")
    if any(name != "references" for name in re.findall(r"\\bibliography\{([^}]+)\}", text)):
        raise ValueError("Unknown bibliography file")
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        if name not in figures:
            raise ValueError("Unknown figure reference")


def validate_evidence_numbers(text, evidence):
    """Conservative decimal audit of manuscript body, not a semantic proof."""
    numbers = set()

    def visit(value):
        if isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
        elif type(value) in (float, int):
            for n in (value, value * 100):
                numbers.update(f"{n:.{precision}f}" for precision in range(1, 7))
        elif isinstance(value, str):
            numbers.update(re.findall(r"\d+\.\d+", value))

    visit(evidence)
    body = text.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = re.sub(r"\\includegraphics\[[^]]*\]\{[^}]*\}", "", body)
    body = re.sub(r"(?<!\\)%[^\n]*", "", body)
    unknown = set(re.findall(r"(?<![A-Za-z])\d+\.\d+", body)) - numbers
    if unknown:
        raise ValueError(f"Manuscript decimals absent from evidence: {sorted(unknown)}")


def translate_segments(text):
    """Translate text nodes only; preserve all LaTeX structure and math literally."""
    pattern = r"(\\begin\{(?:equation\*?|align\*?|gather\*?|math|displaymath)\}[\s\S]*?\\end\{(?:equation\*?|align\*?|gather\*?|math|displaymath)\}|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\\begin\{filecontents\*?\}[\s\S]*?\\end\{filecontents\*?\}|https?://[^\s}]+|\\(?:begin|end|cite\w*|label|ref|bibliography|bibliographystyle|includegraphics)(?:\[[^]]*\])?\{[^}]*\}|\$\$[\s\S]*?\$\$|\$[^$]*\$|\\[A-Za-z]+\*?|[{}\[\]\n])"
    pieces = re.split(pattern, text)
    indices = [
        i
        for i, p in enumerate(pieces)
        if re.search("[A-Za-z]{3}", p) and not re.match(r"(\\|https?://|\$)", p)
    ]
    return pieces, indices


def run(project, root, analysis, provider):
    from .autonomous_postprocess import _commit_stage

    vendor = project / "vendor/AI-Scientist-v2"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from ai_scientist import perform_writeup as native
    from ai_scientist.perform_llm_review import perform_review

    evidence = analysis["evidence"]
    template_dir = vendor / "ai_scientist/blank_icml_latex"
    sources = [
        *template_dir.glob("*"),
        vendor / "ai_scientist/perform_writeup.py",
        vendor / "ai_scientist/perform_llm_review.py",
        project / "configs/gate_a_llm.yaml",
        Path(__file__),
    ]
    version = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(sources) if p.is_file())
    ).hexdigest()
    identity = json.dumps(
        {
            "analysis": analysis,
            "version": version,
            "figure_hashes": {
                row["path"]: hashlib.sha256(safe_path(root, row["path"]).read_bytes()).hexdigest()
                for row in analysis["figures"]["figures"]
            },
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    directory = root / "paper/versions" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "paper/publication_manifest.json"
    manifest = (
        read(manifest_path)
        if manifest_path.exists()
        else {
            "schema_version": 1,
            "backend": "upstream_v2",
            "input_sha256": fingerprint,
            "backend_version": version,
            "stages": {},
        }
    )
    if manifest["input_sha256"] != fingerprint:
        raise ValueError("Publication inputs changed; explicit new-version review required")
    for p in template_dir.iterdir():
        if p.is_file() and p.suffix in (".sty", ".bst"):
            shutil.copy2(p, directory / p.name)
    figure_names = []
    for row in analysis["figures"]["figures"]:
        source = safe_path(root, row["path"])
        target = directory / "figures" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        figure_names.append("figures/" + source.name)
    refs = evidence["literature"]["references"]
    from .paper_export import _escape_latex

    bib = "\n".join(
        "@misc{R%d, title={%s}, author={%s}, year={%s}, doi={%s}}"
        % (
            i,
            _escape_latex(r["title"]),
            _escape_latex(str(r.get("authors", "Unknown"))),
            r.get("year", ""),
            r.get("doi", ""),
        )
        for i, r in enumerate(refs, 1)
    )
    (directory / "references.bib").write_text(bib, encoding="utf-8")
    template = (template_dir / "template.tex").read_text(encoding="utf-8")
    template = re.sub(
        r"\\begin\{filecontents\}\{references.bib\}[\s\S]*?\\end\{filecontents\}",
        lambda _: "\\begin{filecontents}{references.bib}\n" + bib + "\n\\end{filecontents}",
        template,
    )
    writer = Gateway(provider, directory / "responses", fingerprint, "paper_writer")
    reviewer = Gateway(provider, directory / "responses", fingerprint, "reviewer")
    (directory / "responses").mkdir(exist_ok=True)

    def commit(stage, path, language):
        manifest["stages"][stage] = [
            {
                "language": language,
                "source": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ]
        save(manifest_path, manifest)
        _commit_stage(root / "task.json", stage)

    def step(name, generate):
        path = directory / (name + ".json")
        return json.loads(
            cached_artifact(
                path, fingerprint + name, lambda _: json.dumps(generate(), ensure_ascii=False)
            )
        )

    restrictions = (
        "Use only supplied evidence. Unmeasured time is unmeasured; negative results are valid. "
        "Single-repeat results are descriptive only, not significant or stable. No clinical claims. "
        "Include AI assistance disclosure. Cite only R1..Rn from supplied references. Do not invent numerical results. "
        "Keep references.bib external; do not emit filecontents or graphicspath. Use exact supplied figure paths. "
        "Complete ICML author metadata without inventing identities or contact details; use anonymous if unknown. "
    )
    draft = step(
        "draft",
        lambda: dict(
            zip(
                ("text", "history"),
                native.writeup_step(
                    idea_text=restrictions + identity,
                    combined_summaries_str=json.dumps(evidence, ensure_ascii=False),
                    aggregator_code="No generated plotting code; all figures are host-verified.",
                    plot_names=figure_names,
                    writeup_text=template,
                    plot_descriptions_str=json.dumps(analysis["figures"], ensure_ascii=False),
                    query=writer,
                ),
            )
        ),
    )

    def write_tex(name, text):
        try:
            validate_tex(text, [f"R{i}" for i in range(1, len(refs) + 1)], figure_names)
            validate_evidence_numbers(text, evidence)
        except ValueError as error:
            rejected = directory / (name + ".rejected.json")
            save(rejected, {"status": "rejected", "text": text, "error": str(error)})
            raise
        path = directory / name
        path.write_text(text, encoding="utf-8")
        return path

    draft_path = write_tex("draft.tex", draft["text"])
    commit("paper_written", draft_path, "en")
    review = step(
        "review",
        lambda: perform_review(
            draft["text"] + "\nTRUSTED EVIDENCE:\n" + identity,
            model="gateway",
            client=None,
            num_reflections=1,
            num_fs_examples=0,
            num_reviews_ensemble=1,
            query=reviewer,
        ),
    )
    if (
        not isinstance(review, dict)
        or not all(
            isinstance(review.get(k), t)
            for k, t in (
                ("Summary", str),
                ("Weaknesses", list),
                ("Questions", list),
                ("Decision", str),
            )
        )
        or review["Decision"] not in ("Accept", "Reject")
    ):
        raise ValueError("Invalid native review JSON; no synthetic defaults")
    review_path = directory / "review.json"
    commit("review_completed", review_path, "en")
    save(
        directory / "review_display.json",
        {
            "summary": review["Summary"],
            "issues": review["Weaknesses"],
            "checklist": review["Questions"],
            "decision": review["Decision"],
        },
    )
    if (directory / "final.json.cache.json").exists():
        current = step("final", lambda: None)
        final_path = write_tex("template.tex", current["text"])
    else:
        current = draft
        from .autonomous_pdf import _compile

        for i in range(2):
            diagnostics = ""
            try:
                path = write_tex("template.tex", current["text"])
                compile_cached(directory, path.name, compiler=_compile)
            except (RuntimeError, ValueError) as e:
                diagnostics = str(e)
            revision = step(
                f"revision-{i + 1}",
                lambda: dict(
                    zip(
                        ("response", "history"),
                        native.reflection_step(
                            unused_figs=[],
                            invalid_figs=[],
                            reflection_page_info="8 pages is a soft target, not a validity criterion.",
                            check_output=diagnostics,
                            big_model_system_message=native.writeup_system_message_template.format(
                                page_limit=8
                            ),
                            msg_history=current["history"],
                            query=writer,
                            review=json.dumps(review, ensure_ascii=False) + "\n" + restrictions,
                        ),
                    )
                ),
            )
            text = (
                current["text"]
                if revision["response"].strip() == "I am done"
                else native.extract_writeup(revision["response"])
            )
            current = {"text": text, "history": revision["history"]}
            try:
                final_path = write_tex("template.tex", text)
                compile_cached(directory, final_path.name, compiler=_compile)
                break
            except (RuntimeError, ValueError):
                if i == 1:
                    raise RuntimeError("Revision limit reached")
        step("final", lambda: current)
    commit("revision_completed", final_path, "en")
    body = current["text"].split(r"\begin{document}", 1)[1].rsplit(r"\end{document}", 1)[0]
    if r"\begin{abstract}" not in body:
        raise ValueError("Native paper lacks an abstract for portable translation")
    body = body[body.index(r"\begin{abstract}") :]
    body = re.sub(r"(?<!\\)%[^\n]*", "", body)
    title = re.search(r"\\(?:icmltitle|title)\{([^{}]+)\}", current["text"])
    body = (r"\section*{" + title.group(1) + "}\n" if title else "") + body
    pieces, indices = translate_segments(body)
    translation = step(
        "translation",
        lambda: json.loads(
            provider.call_text(
                "paper_writer",
                f"publication-{fingerprint}-translation",
                "Translate the JSON list of text segments to Chinese. Return only a JSON list of identical length. Preserve every number and punctuation; add nothing.",
                json.dumps([pieces[i] for i in indices], ensure_ascii=False),
            )[0]
        ),
    )
    if not isinstance(translation, list) or len(translation) != len(indices):
        raise ValueError("Translation structure mismatch")
    for i, value in zip(indices, translation):
        if (
            not isinstance(value, str)
            or re.findall(r"\d+(?:\.\d+)?", value) != re.findall(r"\d+(?:\.\d+)?", pieces[i])
            or any(c in value for c in "{}\\$")
        ):
            raise ValueError("Translation changed numeric content or LaTeX structure")
        pieces[i] = value
    chinese = (
        '\\documentclass{article}\n\\usepackage{fontspec,graphicx,booktabs,amsmath,amssymb,natbib}\n\\setmainfont{Noto Serif CJK SC}\n\\XeTeXlinebreaklocale "zh"\n\\XeTeXlinebreakskip=0pt plus 1pt\n\\emergencystretch=2em\n\\renewcommand{\\abstractname}{摘要}\n\\renewcommand{\\refname}{参考文献}\n\\begin{document}\n'
        + "".join(pieces)
        + "\n\\end{document}\n"
    )
    chinese = chinese.replace(r"\icmltitle", r"\title")
    # ICML's two-column title environment is not portable to the Chinese template.
    if any(token in chinese for token in (r"\icmlauthor", r"\icmlaffiliation", r"\twocolumn[")):
        raise ValueError(
            "Chinese template needs unsupported ICML title structure; retain draft for review"
        )
    translation_path = write_tex("translation.tex", chinese)
    commit("translation_completed", translation_path, "zh")
    return {
        "task_id": root.name,
        "completed_stage": "translation_completed",
        "publication_backend": "upstream_v2",
    }


def compile_cached(directory, name, *, xelatex=False, compiler):
    source = directory / name
    pdf = source.with_suffix(".pdf")
    receipt = source.with_suffix(".compiled.json")

    def input_digest():
        dependencies = [
            source,
            *directory.glob("*.bib"),
            *directory.glob("*.bst"),
            *directory.glob("*.sty"),
            *directory.glob("figures/**/*"),
        ]
        digest = hashlib.sha256(f"publication-runner:0.1:{xelatex}".encode())
        for path in sorted(set(dependencies)):
            if path.is_file():
                digest.update(str(path.relative_to(directory)).encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    digest = input_digest()
    if receipt.exists() and pdf.exists():
        prior = read(receipt)
        if (
            prior["input"] == digest
            and prior["pdf"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
        ):
            return prior["log"]
    log = compiler(
        directory, name, xelatex=xelatex, bibtex=True, image="path-scientist-publication-runner:0.1"
    )
    if not pdf.exists():
        raise RuntimeError("Compiler did not create a PDF")
    digest = input_digest()
    save(
        receipt, {"input": digest, "pdf": hashlib.sha256(pdf.read_bytes()).hexdigest(), "log": log}
    )
    return log


def build_pdfs(project, root, *, allow_paid=False):
    from .autonomous_pdf import _compile, _compile_with_repair
    from .autonomous_acceptance import require_task

    require_task(root, "translation_completed")
    manifest = read(root / "paper/publication_manifest.json")
    provider = None
    if allow_paid:
        from gate_a.config import load_config
        from gate_a.pipeline import select_live_models
        from gate_a.budget import BudgetLedger
        from gate_a.provider import ZhipuProvider

        config = load_config(project / "configs/gate_a_llm.yaml")
        provider = ZhipuProvider(
            config,
            select_live_models(config),
            BudgetLedger(
                root / "budget.json", float(read(root / "task.json").get("budget_limit_usd", 8))
            ),
            root / "research/responses",
        )
    for stage in ("revision_completed", "translation_completed"):
        row = artifacts(root, stage)[0]
        authoritative = safe_path(root, row["source"])
        workspace = authoritative.parent / "compile" / row["sha256"]
        workspace.mkdir(parents=True, exist_ok=True)
        source = workspace / authoritative.name
        if not source.exists():
            shutil.copy2(authoritative, source)
        for dependency in authoritative.parent.iterdir():
            if dependency.is_file() and dependency.suffix in (".bib", ".bst", ".sty"):
                shutil.copy2(dependency, workspace / dependency.name)
        if (authoritative.parent / "figures").exists():
            shutil.copytree(
                authoritative.parent / "figures", workspace / "figures", dirs_exist_ok=True
            )

        def compiler(directory, name, *, xelatex=False, **kwargs):
            def repair(text, error, attempt):
                digest = hashlib.sha256((text + error).encode()).hexdigest()
                response = provider.call_text(
                    "paper_writer",
                    f"publication-compile-{digest}-{attempt}",
                    "Fix LaTeX syntax only. Return complete LaTeX without fences; never change text, numbers or citations.",
                    text + "\nDIAGNOSTIC:\n" + error,
                )[0]

                def tokens(value):
                    return re.findall(r"[\w.]+", re.sub(r"\\[A-Za-z]+\*?", "", value))

                if tokens(response) != tokens(text):
                    rejected = directory / f"{name}.rejected-{attempt}.tex"
                    rejected.write_text(response, encoding="utf-8")
                    raise ValueError(
                        "Compilation repair changed manuscript content; manual review required"
                    )
                return response

            return _compile_with_repair(
                directory,
                name,
                xelatex=xelatex,
                repair=repair if provider else None,
                compiler=lambda d, n, **k: _compile(
                    d, n, bibtex=True, image="path-scientist-publication-runner:0.1", **k
                ),
            )

        compile_cached(
            source.parent, source.name, xelatex=row["language"] == "zh", compiler=compiler
        )
        row["compiled_source"] = str(source.relative_to(root))
        pdf = source.with_suffix(".pdf")
        if not pdf.exists() or pdf.stat().st_size < 10000:
            raise ValueError("Invalid PDF output")
        row["pdf"] = str(pdf.relative_to(root))
        row["pdf_sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        manifest["stages"][stage] = [row]
    save(root / "paper/publication_manifest.json", manifest)
    save(root / "paper/pdf_quality.json", {"passed": True, "backend": "upstream_v2"})
    archive = root / (root.name + "-evidence.zip")
    save(
        root / "paper/archived/archive.json",
        {"archive": archive.name, "pdf_quality": "paper/pdf_quality.json"},
    )
    temp = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob("*"):
            if p.is_file() and p not in (archive, temp):
                if p == root / "task.json":
                    task = read(p)
                    task["completed_stage"] = "archived"
                    task["stages"]["archived"] = "completed"
                    z.writestr("task.json", json.dumps(task, ensure_ascii=False))
                else:
                    z.write(p, p.relative_to(root))
    temp.replace(archive)
    from .autonomous_postprocess import _commit_stage

    _commit_stage(root / "task.json", "archived")
    require_task(root, "archived", require_pdf=True)
    return {"task_id": root.name, "archive": str(archive)}
