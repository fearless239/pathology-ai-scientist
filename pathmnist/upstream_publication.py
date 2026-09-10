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

WRITING_GUIDANCE = (
    "Organize the manuscript around the research question, not the sequence of workflow stages. "
    "Use the native ICML section guidance: motivate the question in Introduction, explain why "
    "the comparison tests it in Methods, report observations in Results, and interpret them "
    "against supplied literature in Discussion. Do not invent novelty, research gaps, mechanisms, "
    "or missing implementation details. Metadata alone does not establish a literature finding. "
    "Distinguish observed effects from possible explanations; an untested explanation is a "
    "hypothesis, not a demonstrated cause. Scope the title, abstract and conclusion to the "
    "actual dataset, model and repeat plan. Do not claim negligible overhead without measurements. "
    "Describe scientific controls in ordinary research language, not contract/guardrail/receipt "
    "terminology. Keep hashes, cache identities and internal paths out of prose. Preserve actual "
    "training controls, seeds, selection rules, split-specific results and uncertainty limitations. "
)

REVIEW_GUIDANCE = (
    "Check scientific argument and manuscript style as well as correctness. Identify specific "
    "sections or passages: Is the question motivated by supplied literature? Does Methods explain "
    "the comparison and retain reproducibility details? Does Discussion interpret rather than "
    "repeat Results? Are proposed mechanisms clearly untested when appropriate? Do the title "
    "and conclusion respect the repeat plan and evidence? Flag internal workflow jargon. "
    "Distinguish issues fixable in prose from missing experiments; resolve the latter by limiting "
    "claims, never inventing results. Compilation success does not resolve scientific weaknesses. "
)


def research_brief(analysis):
    """Select existing scientific facts; no model call or inferred research gap."""
    evidence = analysis["evidence"]
    contract = evidence.get("research_contract", {})
    missing = "Not provided in supplied evidence; do not invent."

    def supplied(value):
        return value if value is not None and value != "" and value != [] and value != {} else missing

    references = [
        {
            "citation_key": f"R{i}",
            "title": reference.get("title"),
            "abstract": supplied(reference.get("abstract")),
            "relevance_status": reference.get("relevance_status"),
            "relevance_reason": supplied(reference.get("relevance_reason")),
        }
        for i, reference in enumerate(evidence.get("literature", {}).get("references", []), 1)
    ]
    return {
        "research_question": supplied(contract.get("research_question")),
        "dataset": supplied(evidence.get("dataset")),
        "literature_basis": supplied(references),
        "pre_specified_hypothesis": {
            "success_criteria": supplied(contract.get("success_criteria")),
            "instruction": "These are pre-specified criteria, not evidence of success.",
        },
        "comparison_design": {
            key: supplied(contract.get(key))
            for key in ("baseline", "interventions", "comparisons", "required_ablations")
        },
        "observed_finding": supplied(analysis.get("finding")),
        "conclusion_limits": {
            "claim_boundary": supplied(contract.get("claim_boundary")),
            "repeat_plan": supplied(contract.get("repeat_plan")),
            "statistical_plan": supplied(contract.get("statistical_plan")),
            "limitations": supplied(analysis.get("limitations")),
        },
    }


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
            raise ValueError(f"Publication artifact hash mismatch: {row['source']}")
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
    referenced_figures = re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text
    )
    unknown_figures = sorted(set(referenced_figures) - set(figures))
    if unknown_figures:
        raise ValueError(
            "Unknown figure reference(s): " + ", ".join(unknown_figures)
        )


def normalize_review(review):
    """Validate the native review schema and normalize only its decision case."""
    if not isinstance(review, dict):
        raise ValueError("Native review must be a JSON object")
    result = dict(review)
    if not isinstance(result.get("Summary"), str):
        raise ValueError("Native review Summary must be text")
    for key in ("Weaknesses", "Questions"):
        value = result.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Native review {key} must be a list of strings")
    decision = result.get("Decision")
    if not isinstance(decision, str) or decision.strip().casefold() not in {
        "accept",
        "reject",
    }:
        raise ValueError("Native review Decision must be Accept or Reject")
    result["Decision"] = decision.strip().title()
    return result


def normalize_tex_paths(text: str, figures: list[str]) -> str:
    """Convert unambiguous verified-figure aliases into explicit local paths.

    Paper models often omit either the ``figures/`` prefix or the file extension.
    A basename or stem is safe to repair only when it identifies exactly one of
    the host-verified figures. Ambiguous and invented names remain unchanged and
    are rejected by :func:`validate_tex`.
    """
    text = re.sub(
        r"\\begin\{filecontents\*?\}\{references\.bib\}[\s\S]*?"
        r"\\end\{filecontents\*?\}\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?m)^[ \t]*\\graphicspath\{[^\r\n]*\}[^\r\n]*(?:\r?\n|$)",
        "",
        text,
    )
    alias_targets: dict[str, set[str]] = {}
    for figure in figures:
        canonical = figure.replace("\\", "/")
        basename = canonical.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0]
        path_without_extension = (
            canonical.rsplit(".", 1)[0] if "." in basename else canonical
        )
        for alias in (canonical, basename, stem, path_without_extension):
            alias_targets.setdefault(alias, set()).add(canonical)
    aliases = {
        alias: next(iter(targets))
        for alias, targets in alias_targets.items()
        if len(targets) == 1
    }

    def explicit_path(match: re.Match[str]) -> str:
        options, name = match.groups()
        lookup = name.replace("\\", "/")
        if lookup.startswith("./"):
            lookup = lookup[2:]
        return rf"\includegraphics{options or ''}{{{aliases.get(lookup, name)}}}"

    text = re.sub(
        r"\\includegraphics(\[[^]]*\])?\{([^}]+)\}",
        explicit_path,
        text,
    )
    return normalize_table_widths(text)


def normalize_table_widths(text: str) -> str:
    """Constrain generated tabular blocks to the current ICML column width.

    The trusted template already loads ``graphicx``. This deterministic wrapper
    changes layout only and preserves all scientific text, numbers, and table cells.
    """
    pattern = re.compile(
        r"\\begin\{tabular\}(?:\{[^{}]*\})[\s\S]*?\\end\{tabular\}"
    )

    def constrain(match: re.Match[str]) -> str:
        prefix = text[max(0, match.start() - 100) : match.start()]
        if re.search(
            r"\\resizebox\{\\(?:columnwidth|linewidth)\}\{!\}\{%?\s*$",
            prefix,
        ):
            return match.group(0)
        return "\\resizebox{\\columnwidth}{!}{%\n" + match.group(0) + "\n}"

    return pattern.sub(constrain, text)


def bibtex_authors(value: object) -> str:
    """Render literature-provider author lists using BibTeX's `and` separator."""
    if isinstance(value, list):
        authors = [str(author).strip() for author in value if str(author).strip()]
    else:
        authors = [author.strip() for author in str(value or "Unknown").split(",")]
        authors = [author for author in authors if author]
    authors = ["others" if author.casefold().rstrip(".") == "et al" else author for author in authors]
    return " and ".join(authors) or "Unknown"


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
            # The manuscript may describe a signed effect as a positive-sized
            # decrease (for example, "a decrease of 0.61 percentage points").
            for n in (value, abs(value), value * 100, abs(value * 100)):
                numbers.update(f"{n:.{precision}f}" for precision in range(1, 7))
        elif isinstance(value, str):
            numbers.update(re.findall(r"\d+\.\d+", value))

    visit(evidence)
    body = text.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = re.sub(
        r"\\includegraphics(?:\[[^]]*\])?\{[^}]*\}", "", body
    )
    body = re.sub(r"(?<!\\)%[^\n]*", "", body)
    # TeX dimensions configure presentation rather than report scientific
    # observations. Keep prose such as ``0.3 mm`` auditable, while excluding
    # attached TeX lengths (``0.3in``) and relative layout widths
    # (``0.95\linewidth``).
    body = re.sub(
        r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
        r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em|mu)\b",
        "",
        body,
    )
    body = re.sub(
        r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
        r"\\(?:textwidth|linewidth|columnwidth|paperwidth|paperheight)\b",
        "",
        body,
    )
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


def translation_segment_is_safe(source: str, translated: object) -> bool:
    if not isinstance(translated, str):
        return False

    def numbers(value):
        return sorted(re.findall(r"\d+(?:\.\d+)?", value))

    def protected(value):
        # A backslash followed by whitespace is only a LaTeX spacing command.
        # Natural Chinese spacing does not need to preserve it.
        value = re.sub(r"\\(?=\s)", "", value)
        return [char for char in value if char in "{}\\$"]

    return numbers(translated) == numbers(source) and protected(translated) == protected(source)


def normalize_compile_source(text: str) -> str:
    """Prepare the archival PDF without changing the authoritative review source."""
    review_style = r"\usepackage{icml2025}"
    if review_style in text and r"\ClearShipoutPicture" not in text:
        # Keep review-mode author handling, but remove the margin ruler and
        # conference-review notice from the archival reader copy.
        release_style = "\n".join(
            (
                review_style,
                r"\ClearShipoutPicture",
                r"\makeatletter",
                r"\renewcommand{\Notice@String}{}",
                r"\makeatother",
            )
        )
        text = text.replace(review_style, release_style, 1)
        if r"\icmlcorrespondingauthor" not in text:
            text = text.replace(
                r"\begin{document}",
                r"\icmlcorrespondingauthor{Anonymous}{anonymous@example.invalid}"
                + "\n"
                + r"\begin{document}",
                1,
            )
    if r"\cref" in text and r"\usepackage{cleveref}" not in text:
        text = text.replace(r"\begin{document}", r"\usepackage{cleveref}" + "\n" + r"\begin{document}", 1)
    return text


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
    figure_hashes = {
        row["path"]: hashlib.sha256(safe_path(root, row["path"]).read_bytes()).hexdigest()
        for row in analysis["figures"]["figures"]
    }

    def fingerprint_for(backend_version: str) -> str:
        identity = json.dumps(
            {"analysis": analysis, "version": backend_version, "figure_hashes": figure_hashes},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    fingerprint = fingerprint_for(version)
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
    source_fingerprint = manifest["input_sha256"]
    if manifest["input_sha256"] != fingerprint:
        previous_version = manifest.get("backend_version")
        if not previous_version or manifest["input_sha256"] != fingerprint_for(previous_version):
            raise ValueError("Publication inputs changed; explicit new-version review required")
        # The evidence and figures are identical; migrate a pipeline-only change
        # while retaining and validating every already committed artifact.
        manifest["input_sha256"] = fingerprint
        manifest["backend_version"] = version
        save(manifest_path, manifest)
    # Validate committed outputs before cached projections can overwrite them.
    # A missing uncommitted response projection remains recoverable from its receipt.
    for stage in manifest["stages"]:
        artifacts(root, stage)
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
            _escape_latex(bibtex_authors(r.get("authors", "Unknown"))),
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
    resumed_revision = None
    resumed_draft = None
    resumed_cached_revision = None
    resumed_revision_count = 0
    resumed_review = None
    if "revision_completed" in manifest["stages"]:
        resumed_revision = safe_path(root, artifacts(root, "revision_completed")[0]["source"])
        resumed_review = read(safe_path(root, artifacts(root, "review_completed")[0]["source"]))
        draft = {"text": resumed_revision.read_text(encoding="utf-8"), "history": []}
    elif "paper_written" in manifest["stages"]:
        # A backend-only repair can change the version fingerprint after the paid
        # draft and review were committed. Reuse those verified artifacts instead
        # of paying to regenerate the manuscript from scratch.
        resumed_draft = safe_path(
            root, artifacts(root, "paper_written")[0]["source"]
        )
        if "review_completed" in manifest["stages"]:
            resumed_review = read(
                safe_path(root, artifacts(root, "review_completed")[0]["source"])
            )
        draft_text = resumed_draft.read_text(encoding="utf-8")
        draft = {
            "text": draft_text,
            "history": [
                {
                    "role": "assistant",
                    "content": "```latex\n" + draft_text + "\n```",
                }
            ],
        }
        # Recover completed, input-bound revision calls from the prior version.
        # cached_artifact validates both the original input fingerprint and value
        # hash, so an uncommitted or edited projection is never trusted directly.
        recovered = draft
        recovered_count = 0
        for revision_index in range(1, 3):
            revision_name = f"revision-{revision_index}"
            revision_path = resumed_draft.parent / f"{revision_name}.json"
            receipt_path = revision_path.with_suffix(".json.cache.json")
            if not receipt_path.exists():
                break

            def missing_revision(_: str, name: str = revision_name) -> str:
                raise RuntimeError(f"Cached {name} receipt could not be recovered")

            revision = json.loads(
                cached_artifact(
                    revision_path,
                    source_fingerprint + revision_name,
                    missing_revision,
                )
            )
            response = revision["response"]
            recovered = {
                "text": (
                    recovered["text"]
                    if response.strip() == "I am done"
                    else native.extract_writeup(response)
                ),
                "history": revision["history"],
            }
            recovered_count = revision_index
        if recovered_count:
            draft = recovered
            resumed_revision_count = recovered_count
        if recovered_count == 2:
            resumed_cached_revision = recovered
    else:
        draft = step(
            "draft",
            lambda: dict(
                zip(
                    ("text", "history"),
                    native.writeup_step(
                        idea_text=restrictions + WRITING_GUIDANCE + json.dumps(
                            research_brief(analysis), ensure_ascii=False
                        ),
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
        text = normalize_tex_paths(text, figure_names)
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

    # Give the initial manuscript the same bounded, durable repair behavior as
    # later revisions. A recoverable validator diagnostic must not abort the
    # whole publication stage before a correction can be requested and cached.
    for draft_attempt in range(3):
        try:
            draft_path = write_tex("draft.tex", draft["text"])
            break
        except ValueError as error:
            if draft_attempt == 2:
                raise RuntimeError(
                    f"Initial manuscript repair limit reached: {error}"
                ) from error
            diagnostic = str(error)
            repaired = step(
                f"draft-repair-{draft_attempt + 1}",
                lambda diagnostic=diagnostic, draft=draft: dict(
                    zip(
                        ("response", "history"),
                        native.reflection_step(
                            unused_figs=[],
                            invalid_figs=[],
                            reflection_page_info=(
                                "Correct only the reported validation problem; preserve "
                                "all evidence, claims, citations, and verified figure paths."
                            ),
                            check_output=diagnostic,
                            big_model_system_message=native.writeup_system_message_template.format(
                                page_limit=8
                            ),
                            msg_history=draft["history"],
                            query=writer,
                            review=restrictions + WRITING_GUIDANCE,
                        ),
                    )
                ),
            )
            repaired_text = (
                draft["text"]
                if repaired["response"].strip() == "I am done"
                else native.extract_writeup(repaired["response"])
            )
            draft = {"text": repaired_text, "history": repaired["history"]}
    commit("paper_written", draft_path, "en")
    review = resumed_review or step(
        "review",
        lambda: perform_review(
            draft["text"] + "\n" + REVIEW_GUIDANCE + "\nTRUSTED EVIDENCE:\n"
            + json.dumps(evidence, ensure_ascii=False),
            model="gateway",
            client=None,
            num_reflections=1,
            num_fs_examples=0,
            num_reviews_ensemble=1,
            query=reviewer,
        ),
    )
    try:
        review = normalize_review(review)
    except ValueError as initial_review_error:
        repair_prompt = (
            "Return only a JSON object with exactly these fields: Summary (string), "
            "Weaknesses (list of strings), Questions (list of strings), and Decision "
            "('Accept' or 'Reject'). Preserve the review's meaning and invent no evidence.\n\n"
            f"SCHEMA ERROR: {initial_review_error}\n\nREVIEW:\n"
            + json.dumps(review, ensure_ascii=False)
        )
        review = step(
            "review-schema-repair",
            lambda: json.loads(
                reviewer(
                    repair_prompt,
                    system_message="Repair review JSON structure only.",
                )[0]
            ),
        )
        review = normalize_review(review)
    review_path = directory / "review.json"
    save(review_path, review)
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
    if resumed_revision is not None:
        current = draft
        final_path = write_tex("template.tex", current["text"])
        from .autonomous_pdf import _compile

        compile_cached(directory, final_path.name, compiler=_compile)
    elif resumed_cached_revision is not None:
        current = resumed_cached_revision
        final_path = write_tex("template.tex", current["text"])
        from .autonomous_pdf import _compile

        compile_cached(directory, final_path.name, compiler=_compile)
    elif (directory / "final.json.cache.json").exists():
        current = step("final", lambda: None)
        final_path = write_tex("template.tex", current["text"])
        from .autonomous_pdf import _compile

        compile_cached(directory, final_path.name, compiler=_compile)
    else:
        current = draft
        from .autonomous_pdf import _compile

        for i in range(resumed_revision_count, 2):
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
                            review=json.dumps(review, ensure_ascii=False) + "\n" + restrictions
                            + WRITING_GUIDANCE + REVIEW_GUIDANCE,
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
            except (RuntimeError, ValueError) as error:
                if i == 1:
                    raise RuntimeError(f"Revision limit reached: {error}") from error
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
    translation = []
    translation_batch_size = 24
    source_segments = [pieces[i] for i in indices]

    def translate_batch(batch, label):
        try:
            translated = step(
                f"translation-{label}",
                lambda: json.loads(
                    provider.call_text(
                        "paper_writer",
                        f"publication-{fingerprint}-translation-{label}",
                        "Translate the JSON list of text segments to Chinese. Return only a JSON list of identical length. Preserve every number and punctuation; add nothing.",
                        json.dumps(batch, ensure_ascii=False),
                    )[0]
                ),
            )
        except json.JSONDecodeError as error:
            if len(batch) == 1:
                raise ValueError("Translation response is not valid JSON") from error
            midpoint = len(batch) // 2
            return translate_batch(batch[:midpoint], label + "a") + translate_batch(
                batch[midpoint:], label + "b"
            )
        if isinstance(translated, list) and len(translated) == len(batch):
            return translated
        if len(batch) == 1:
            raise ValueError("Translation structure mismatch")
        midpoint = len(batch) // 2
        return translate_batch(batch[:midpoint], label + "a") + translate_batch(
            batch[midpoint:], label + "b"
        )

    for offset in range(0, len(source_segments), translation_batch_size):
        batch = source_segments[offset : offset + translation_batch_size]
        translation.extend(translate_batch(batch, f"{offset // translation_batch_size:03d}"))
    if not isinstance(translation, list) or len(translation) != len(indices):
        raise ValueError("Translation structure mismatch")
    for segment_index, (i, value) in enumerate(zip(indices, translation)):
        if not translation_segment_is_safe(pieces[i], value):
            value = step(
                f"translation-repair-{segment_index:03d}",
                lambda source=pieces[i]: json.loads(
                    provider.call_text(
                        "paper_writer",
                        f"publication-{fingerprint}-translation-repair-{segment_index:03d}",
                        "Translate this JSON string to Chinese and return only one JSON string. "
                        "Copy every Arabic numeral, punctuation mark, and LaTeX character exactly; "
                        "do not spell out, remove, reorder, or add any of them.",
                        json.dumps(source, ensure_ascii=False),
                    )[0]
                ),
            )
        if not translation_segment_is_safe(pieces[i], value):
            raise ValueError(
                f"Translation changed numeric content or LaTeX structure in segment {segment_index}"
            )
        pieces[i] = value
    chinese = (
        '\\documentclass{article}\n\\usepackage{fontspec,graphicx,booktabs,amsmath,amssymb,natbib,cleveref}\n\\setmainfont{Noto Serif CJK SC}\n\\XeTeXlinebreaklocale "zh"\n\\XeTeXlinebreakskip=0pt plus 1pt\n\\emergencystretch=2em\n\\renewcommand{\\abstractname}{摘要}\n\\renewcommand{\\refname}{参考文献}\n\\begin{document}\n'
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
        shutil.copy2(authoritative, source)
        source.write_text(
            normalize_compile_source(source.read_text(encoding="utf-8")), encoding="utf-8"
        )
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
