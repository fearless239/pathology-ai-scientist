from pathlib import Path

import pytest

from pathmnist.autonomous_pdf import _normalize_figure_references, _normalize_verified_references, _validate_source


def test_unicode_math_export_in_prose_and_equations():
    from pathmnist.paper_export import markdown_to_latex
    text = markdown_to_latex('# Test\nα=0.15, 28×28; $α ∈ {0.05, 0.15}$')
    assert 'α' not in text and '×' not in text and '∈' not in text
    assert r'\ensuremath{\alpha}=0.15' in text
    assert '0.05' in text


@pytest.mark.parametrize('changed', [False, True])
def test_unresolved_repair_can_compile_locally_without_resending(tmp_path, monkeypatch, changed):
    import json
    import hashlib
    from pathmnist import autonomous_pdf as module
    path = tmp_path/'paper.tex'
    path.write_text('fixed 42' if changed else 'original 42',encoding='utf-8')
    path.with_suffix('.compile.json').write_text(json.dumps({'source':'original 42',
        'input_sha256':hashlib.sha256(b'original 42').hexdigest(),'repairs':1,'phase':'repairing'}))
    def no_request(*args):
        pytest.fail('Do not resend external repair')
    monkeypatch.setattr(module,'_compile',lambda *a,**kw:'ok')
    assert module._compile_with_repair(tmp_path,path.name,repair=no_request)=='ok'
    state=json.loads(path.with_suffix('.compile.json').read_text())
    assert state['phase']=='compiled' and state['repairs']==1


def test_rejected_repair_is_durable_and_never_retried(tmp_path, monkeypatch):
    import json
    from pathmnist import autonomous_pdf as module
    path=tmp_path/'paper.tex'
    path.write_text('original 42')
    def fail(*a,**kw):
        raise RuntimeError('compile failure')
    calls=[]
    def repair(*a):
        calls.append(1)
        return 'changed 43'
    monkeypatch.setattr(module,'_compile',fail)
    with pytest.raises(RuntimeError,match='numeric'):
        module._compile_with_repair(tmp_path,path.name,repair=repair)
    assert json.loads(path.with_suffix('.compile.json').read_text())['phase']=='rejected'
    with pytest.raises(RuntimeError,match='no external request resent'):
        module._compile_with_repair(tmp_path,path.name,repair=repair)
    assert calls==[1]


def test_compile_repair_cap_survives_restart(tmp_path, monkeypatch):
    from pathmnist import autonomous_pdf as module
    source = tmp_path / 'paper.tex'
    source.write_text('original 42', encoding='utf-8')
    calls = []
    def fail(*args, **kwargs):
        raise RuntimeError('bad layout')
    def repair(text, error, attempt):
        calls.append(attempt)
        return text + ' repaired'
    monkeypatch.setattr(module, '_compile', fail)
    with pytest.raises(RuntimeError, match='bad layout'):
        module._compile_with_repair(tmp_path, source.name, repair=repair)
    # build_pdfs regenerates the original source on entry.
    source.write_text('original 42', encoding='utf-8')
    with pytest.raises(RuntimeError, match='bad layout'):
        module._compile_with_repair(tmp_path, source.name, repair=repair)
    assert calls == [1, 2]


def test_publication_cache_recovers_missing_projection(tmp_path):
    from pathmnist.artifact_cache import cached_artifact
    output = tmp_path / 'paper.md'
    calls = []
    def generate(fingerprint):
        calls.append(fingerprint)
        return 'paper'
    assert cached_artifact(output, 'evidence-v1', generate) == 'paper'
    output.unlink()  # Simulate a committed receipt but missing projection.
    assert cached_artifact(output, 'evidence-v1', generate) == 'paper'
    assert len(calls) == 1
    cached_artifact(output, 'evidence-v2', generate)
    assert len(calls) == 2


def test_reference_export_uses_standard_numeric_labels(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Study\n\nPrior work [R1].\n\n## References\n\nold\n", encoding="utf-8")
    literature = {"status": "verified", "references": [{"title": "Relevant paper", "authors": "A. Author", "year": 2025, "venue": "Journal", "doi": "10.1/example"}]}
    _normalize_verified_references(paper, literature, language="en")
    result = paper.read_text(encoding="utf-8")
    assert "Prior work [1]." in result
    assert "**[1]**" in result
    assert "[R1]" not in result


def test_source_validation_rejects_audit_hash(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Study\n\nHash: " + "a" * 64 + "\n\n## AI Assistance Disclosure\n\n**AI-generation disclosure:** reviewed\n\n## References\n\n* **[1]** Paper\n\nCited [1].\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audit-only"):
        _validate_source(paper, language="en")


def test_source_validation_allows_scholarly_hex_identifier_in_references(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    semantic_id = "d8fc8df8c30943994badbf305243971b27b2702b"
    paper.write_text(
        "# Study\n\nPrior work [1].\n\n## AI Assistance Disclosure\n\n"
        "**AI-generation disclosure:** reviewed\n\n## References\n\n"
        f"* **[1]** Paper. URL: https://www.semanticscholar.org/paper/{semantic_id}\n",
        encoding="utf-8",
    )
    _validate_source(paper, language="en")


def test_source_validation_allows_scholarly_url_in_manuscript_prose(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    semantic_id = "d8fc8df8c30943994badbf305243971b27b2702b"
    paper.write_text(
        f"# Study\n\nSource: https://www.semanticscholar.org/paper/{semantic_id} [1].\n\n"
        "## AI Assistance Disclosure\n\n**AI-generation disclosure:** reviewed\n\n"
        "## References\n\n* **[1]** Paper. DOI: 10.1/example\n",
        encoding="utf-8",
    )
    _validate_source(paper, language="en")


def test_figure_references_are_normalized_and_deduplicated() -> None:
    markdown = "![First](dataset_splits.png)\n\n![Duplicate](paper/figures_generated/figures/dataset_splits.png)\n\n![Other](figures/metrics.png)"
    result = _normalize_figure_references(markdown, {"dataset_splits.png", "metrics.png"})
    assert result.count("figures/dataset_splits.png") == 1
    assert "![First](figures/dataset_splits.png)" in result
    assert "![Other](figures/metrics.png)" in result
