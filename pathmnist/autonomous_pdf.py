from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pathmnist.autonomous_acceptance import require_task
from pathmnist.paper_disclosure import ensure_disclosure, has_disclosure
from pathmnist.paper_export import markdown_to_latex
from pathmnist.execution_control import task_operation


RUNNER_IMAGE = "path-scientist-gate-a-runner:0.2"


def _normalize_figure_references(markdown: str, figure_names: set[str]) -> str:
    """Map manifest figures to the copied directory and remove duplicate uses."""
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        alt, raw_source = match.groups()
        source = raw_source.replace("\\", "/")
        name = Path(source).name
        if name not in figure_names:
            return match.group(0)
        if name in seen:
            return ""
        seen.add(name)
        return f"![{alt}](figures/{name})"

    return re.sub(r"!\[([^]]*)\]\(([^)]+)\)", replace, markdown)


def _normalize_verified_references(path: Path, literature: dict, *, language: str) -> None:
    references = literature.get("references")
    if literature.get("status") != "verified" or not isinstance(references, list) or not references:
        raise RuntimeError("Cannot build a formal reference section without verified literature")
    text = path.read_text(encoding="utf-8").rstrip()
    heading_name = "References" if language == "en" else "参考文献"
    heading_pattern = re.compile(
        r"^#{1,3}\s+(?:\d+\.\s*)?(?:References|参考文献)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = heading_pattern.search(text)
    body = text[: match.start()].rstrip() if match else text
    body = re.sub(r"\[R(\d+)\]", r"[\1]", body)
    entries: list[str] = []
    for index, reference in enumerate(references, start=1):
        if not isinstance(reference, dict) or not reference.get("title"):
            raise RuntimeError(f"Verified reference R{index} is incomplete")
        stable = (
            reference.get("doi")
            or reference.get("pmid")
            or reference.get("corpus_id")
            or reference.get("url")
        )
        if not stable:
            raise RuntimeError(f"Verified reference R{index} has no stable identifier")
        parts = [
            f"**[{index}]** {reference.get('authors') or 'Unknown authors'}.",
            f"\"{reference['title']}\"",
            f"*{reference.get('venue') or 'Unknown venue'}* ({reference.get('year') or 'n.d.'}).",
        ]
        if reference.get("doi"):
            parts.append(f"DOI: {reference['doi']}.")
        if reference.get("pmid"):
            parts.append(f"PMID: {reference['pmid']}.")
        if reference.get("url"):
            parts.append(f"URL: {reference['url']}")
        entries.append("* " + " ".join(parts))
    normalized = body + f"\n\n## {heading_name}\n\n" + "\n".join(entries) + "\n"
    cited = {int(value) for value in re.findall(r"\[(\d+)\]", body)}
    missing = sorted(value for value in cited if value < 1 or value > len(references))
    if missing:
        raise RuntimeError(f"Paper cites keys absent from verified literature: {missing}")
    path.write_text(normalized, encoding="utf-8")


def _compile(directory: Path, tex_name: str, *, xelatex: bool = False, bibtex: bool = False, image: str = RUNNER_IMAGE) -> str:
    engine = "xelatex" if xelatex else "pdflatex"
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.tex',tex_name):
        raise ValueError('Unsafe TeX filename')
    bibliography = f'bibtex {Path(tex_name).stem} && ' if bibtex else ''
    container_name = 'path-scientist-pdf-' + uuid.uuid4().hex
    command = [
        "docker", "run", "--rm", "--network", "none",
        "--name", container_name, "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--mount", f"type=bind,src={directory},dst=/workspace",
        "--workdir", "/workspace",
        image,
        "sh", "-lc",
        f"{engine} -no-shell-escape -interaction=nonstopmode -halt-on-error {tex_name} && "
        + bibliography + f"{engine} -no-shell-escape -interaction=nonstopmode -halt-on-error {tex_name} && "
        f"{engine} -no-shell-escape -interaction=nonstopmode -halt-on-error {tex_name}",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=300)
    except subprocess.TimeoutExpired:
        cleanup = subprocess.run(['docker','rm','--force',container_name],text=True,capture_output=True,timeout=30)
        if cleanup.returncode and 'No such container' not in cleanup.stderr:
            raise RuntimeError('PDF compiler timeout cleanup could not be confirmed')
        raise
    if result.returncode:
        raise RuntimeError(
            f"{engine} failed for {tex_name}: {result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    log = (directory / Path(tex_name).with_suffix(".log")).read_text(
        encoding="utf-8", errors="replace"
    )
    fatal_patterns = (
        r"LaTeX Warning: Citation .* undefined",
        r"LaTeX Warning: There were undefined references",
        r"Overfull \\hbox \((?:[3-9]|\d{2,})\.",
    )
    problems = [pattern for pattern in fatal_patterns if re.search(pattern, log)]
    if problems:
        raise RuntimeError(f"PDF quality gate failed for {tex_name}: {problems}")
    return log


def _compile_with_repair(directory, tex_name, *, xelatex=False, repair=None, compiler=None):
    path = directory / tex_name
    state_path = path.with_suffix('.compile.json')
    source = path.read_text(encoding='utf-8')
    fingerprint = hashlib.sha256(source.encode('utf-8')).hexdigest()
    state = {'input_sha256': fingerprint, 'repairs': 0, 'source': source, 'phase': 'ready'}
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding='utf-8'))
        if fingerprint in (previous['input_sha256'], hashlib.sha256(previous['source'].encode('utf-8')).hexdigest()):
            state = previous
        elif previous.get('phase') in ('repairing', 'rejected'):
            # A deterministic exporter fix may compile without another model call.
            # Preserve the external-request barrier and consumed attempts.
            state.update(repairs=previous['repairs'], phase=previous['phase'])
    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
        temporary.replace(state_path)
    path.write_text(state['source'], encoding='utf-8')
    while True:
        try:
            result = (compiler or _compile)(directory, tex_name, xelatex=xelatex)
            state['phase'] = 'compiled'
            save()
            return result
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            (directory / 'compile_failure.json').write_text(json.dumps({
                'attempt': state['repairs'], 'error': str(error), 'source': tex_name,
            }, ensure_ascii=False), encoding='utf-8')
            if state['phase'] in ('repairing', 'rejected'):
                save()
                raise RuntimeError('Prior LaTeX repair is unresolved or rejected; local compilation still failed; no external request resent') from error
            if state['repairs'] >= 2 or repair is None:
                save()
                raise
            source = path.read_text(encoding='utf-8')
            state['repairs'] += 1
            state['phase'] = 'repairing'
            save()  # Reserve the attempt before an external request; restart cannot reset the cap.
            try:
                revised = repair(source, str(error), state['repairs'])
            except ValueError as rejection:
                state.update(phase='rejected', rejection=str(rejection))
                save()
                raise
            # Preserve the returned response even if content validation rejects it.
            path.with_suffix(f'.repair-{state["repairs"]}.tex').write_text(revised, encoding='utf-8')
            if set(re.findall(r'\d+(?:\.\d+)?', revised)) != set(re.findall(r'\d+(?:\.\d+)?', source)):
                state.update(phase='rejected', rejection='numeric content changed')
                save()
                raise RuntimeError('LaTeX repair changed numeric content; manual review required')
            state.update(source=revised, phase='ready')
            save()
            path.write_text(revised, encoding='utf-8')


def _validate_source(path: Path, *, language: str) -> None:
    text = path.read_text(encoding="utf-8")
    forbidden = ("TODO", "FIXME", "not exhaustively detailed", "dependent on the frozen source code")
    found = [value for value in forbidden if value.casefold() in text.casefold()]
    if found:
        raise RuntimeError(f"Paper source contains unresolved placeholders: {found}")
    audit_leaks = []
    # Stable scholarly URLs can legitimately contain long hexadecimal IDs
    # (notably Semantic Scholar corpus identifiers). Audit hashes remain
    # forbidden in manuscript prose, while the normalized References section
    # is validated separately for stable identifiers and citation mapping.
    references_match = re.search(
        r"^#{1,3}\s+(?:\d+\.\s*)?(?:References|参考文献)\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    manuscript_body = text[: references_match.start()] if references_match else text
    body_without_urls = re.sub(r"https?://\S+", "", manuscript_body)
    if re.search(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])", body_without_urls):
        audit_leaks.append("long hexadecimal identifier")
    for token in ("candidate_frozen/", "final_evaluation/", "dataset/dataset_profile.json", "runner_identity", "evaluator_version"):
        if token.casefold() in text.casefold():
            audit_leaks.append(token)
    if audit_leaks:
        raise RuntimeError(f"Formal {language} paper leaks audit-only details: {sorted(set(audit_leaks))}")
    if not has_disclosure(text, language):
        raise RuntimeError(f"Formal {language} paper has no required AI-generation disclosure")
    heading = r"References" if language == "en" else r"(?:参考文献|References)"
    if not re.search(
        rf"^#{{1,3}}\s+(?:\d+\.\s*)?{heading}\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    ):
        raise RuntimeError(f"Formal {language} paper has no References section")
    citations = {int(value) for value in re.findall(r"\[(\d+)\]", text)}
    definitions = {
        int(value)
        for value in re.findall(
            r"^\s*\*\s+\*\*\[(\d+)\]\*\*", text, re.MULTILINE
        )
    }
    if not citations:
        raise RuntimeError(f"Formal {language} paper has no structured citations")
    if missing := sorted(citations - definitions):
        raise RuntimeError(f"Formal {language} paper has unmapped citation keys: {missing}")


@task_operation
def build_pdfs(project_root: Path, state_root: Path, task_id: str, *, allow_paid: bool = False) -> dict[str, object]:
    task_root = state_root.resolve() / task_id
    from .upstream_publication import backend, build_pdfs as build_upstream
    if backend(json.loads((task_root/'task.json').read_text(encoding='utf-8'))) == 'upstream_v2':
        return build_upstream(project_root.resolve(),task_root,allow_paid=allow_paid)
    revision = task_root / "paper" / "revision_completed"
    translation = task_root / "paper" / "translation_completed"
    if not (revision / "final_paper.md").is_file():
        raise RuntimeError("final_paper.md is missing; run autonomous-postprocess first")
    if not (translation / "translation.md").is_file():
        raise RuntimeError("translation.md is missing; run autonomous-postprocess first")
    require_task(task_root, "translation_completed")
    figure_manifest = json.loads(
        (task_root / "paper/figures_generated/figure_manifest.json").read_text(encoding="utf-8")
    )
    figures = figure_manifest.get("figures", [])
    if not figures:
        raise RuntimeError("Formal paper requires at least one evidence-backed figure")
    figure_names = {Path(str(figure.get("path", ""))).name for figure in figures}
    for destination in (revision / "figures", translation / "figures"):
        destination.mkdir(parents=True, exist_ok=True)
        for figure in figures:
            source_figure = task_root / str(figure.get("path", ""))
            if not source_figure.is_file() or not figure.get("source_artifacts"):
                raise RuntimeError(f"Invalid figure manifest entry: {figure}")
            shutil.copy2(source_figure, destination / source_figure.name)
    literature = json.loads(
        (task_root / "research/literature.json").read_text(encoding="utf-8")
    )
    for source, language in (
        (revision / "final_paper.md", "en"),
        (translation / "translation.md", "zh"),
    ):
        markdown = re.sub(
            r"\((?:paper/figures_generated/)?figures/([^)/]+)\)",
            r"(figures/\1)",
            source.read_text(encoding="utf-8"),
        )
        markdown = _normalize_figure_references(markdown, figure_names)
        # Markdown image alt text becomes the authoritative LaTeX caption.
        # Remove an immediately following model-written italic caption to avoid
        # duplicated figure numbers and detached captions after float placement.
        markdown = re.sub(
            r"(!\[[^]]*\]\([^)]+\))\s*\n\s*\*(?:Figure\s+\d+|图\s*\d+)[^\n]*\*\s*",
            r"\1\n\n",
            markdown,
            flags=re.IGNORECASE,
        )
        if not re.search(r"!\[[^]]*\]\(figures/[^)]+\)", markdown):
            heading = "Evidence figures" if language == "en" else "证据图"
            caption = figures[0].get("title") or heading
            figure_name = Path(str(figures[0]["path"])).name
            insertion = f"\n\n## {heading}\n\n![{caption}](figures/{figure_name})\n"
            reference_heading = re.search(
                r"^#{1,3}\s+(?:References|参考文献)\s*$", markdown, re.MULTILINE | re.IGNORECASE
            )
            if reference_heading:
                markdown = markdown[: reference_heading.start()].rstrip() + insertion + "\n" + markdown[reference_heading.start():]
            else:
                markdown = markdown.rstrip() + insertion
        source.write_text(ensure_disclosure(markdown, language), encoding="utf-8")
    _normalize_verified_references(
        revision / "final_paper.md", literature, language="en"
    )
    _normalize_verified_references(
        translation / "translation.md", literature, language="zh"
    )
    _validate_source(revision / "final_paper.md", language="en")
    _validate_source(translation / "translation.md", language="zh")

    revision_tex = revision / "final_paper.tex"
    translation_tex = translation / "translation.tex"
    revision_tex.write_text(
        markdown_to_latex((revision / "final_paper.md").read_text(encoding="utf-8"), "en"),
        encoding="utf-8",
    )
    translation_tex.write_text(
        markdown_to_latex((translation / "translation.md").read_text(encoding="utf-8"), "zh"),
        encoding="utf-8",
    )
    repair = None
    if allow_paid:
        from gate_a.config import load_config
        from gate_a.budget import BudgetLedger
        from gate_a.pipeline import select_live_models
        from gate_a.provider import ZhipuProvider
        config = load_config(project_root / 'configs/gate_a_llm.yaml')
        provider = ZhipuProvider(config, select_live_models(config), BudgetLedger(task_root / 'budget.json', 8.0), task_root / 'research/responses')
        def repair(source, error, attempt):
            import hashlib
            fingerprint = hashlib.sha256((source + error).encode('utf-8')).hexdigest()[:16]
            return provider.call_text('paper_writer', f'{task_id}-latex-repair-{fingerprint}-{attempt}',
                'Repair LaTeX syntax, references and layout only. Return complete LaTeX without Markdown fences. Never change scientific numbers, claims or data.',
                source + '\n\nCOMPILER DIAGNOSTIC:\n' + error[-4000:])[0]
    english_log = _compile_with_repair(revision, revision_tex.name, repair=repair)
    chinese_log = _compile_with_repair(translation, translation_tex.name, xelatex=True, repair=repair)
    for pdf in (revision / "final_paper.pdf", translation / "translation.pdf"):
        if not pdf.is_file() or pdf.stat().st_size < 10_000:
            raise RuntimeError(f"Compiled PDF is missing or implausibly small: {pdf}")

    archive_path = task_root / f"{task_id}-evidence.zip"
    if archive_path.exists():
        archive_path.unlink()
    archive_base = task_root / f"{task_id}-evidence"
    archive = archive_path
    qa = {
        "schema_version": 1,
        "passed": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": [
            "both PDFs compiled twice",
            "no undefined citations or references",
            "no overfull hbox above 3pt",
            "no unresolved placeholder language",
            "both sources contain a References section",
            "all paper figures are manifest-backed and present",
        ],
        "english_log_bytes": len(english_log.encode("utf-8")),
        "chinese_log_bytes": len(chinese_log.encode("utf-8")),
    }
    (task_root / "paper/pdf_quality.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    archive_root = task_root / "paper/archived"
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "archive.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": task_id,
                "archive": archive.name,
                "pdf_quality": "paper/pdf_quality.json",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    task_path = task_root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["stages"]["archived"] = "completed"
    task["completed_stage"] = "archived"
    task["control"] = "paused"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=task_root))
    acceptance = require_task(task_root, "archived", require_pdf=True)
    return {
        "task_id": task_id,
        "english_pdf": str(revision / "final_paper.pdf"),
        "chinese_pdf": str(translation / "translation.pdf"),
        "archive": str(archive),
        "acceptance": acceptance.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build_pdfs(args.project_root, args.state_root, args.task_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
