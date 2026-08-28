from pathmnist.paper_export import markdown_to_latex


def test_table_column_spec_matches_column_count() -> None:
    markdown = "\n".join(
        [
            "# Title",
            "",
            "| A | B |",
            "| --- | --- |",
            "| 1 | 2 |",
            "| 3 | 4 |",
        ]
    )
    latex = markdown_to_latex(markdown)
    expected = (
        "\\begin{longtable}"
        "{p{\\dimexpr\\linewidth/2-2\\tabcolsep}p{\\dimexpr\\linewidth/2-2\\tabcolsep}}"
    )
    assert expected in latex
    assert latex.count("\\begin{longtable}") == 1
    assert latex.count("\\end{longtable}") == 1
    assert "\\toprule" in latex
    assert "\\bottomrule" in latex


def test_title_heading_is_not_duplicated_as_a_section() -> None:
    latex = markdown_to_latex("# Paper Title\n\n## 1. Introduction\nBody")
    assert "\\title{Paper Title}" in latex
    assert "\\section{Introduction}" in latex
    assert latex.count("\\section{") == 1


def test_zh_preamble_uses_xelatex_native_cjk_font() -> None:
    latex = markdown_to_latex("# 标题\n\n正文", "zh")
    assert "\\usepackage{fontspec}" in latex
    assert "Noto Serif CJK SC" in latex
    assert "CJKutf8" not in latex
    assert "\\begin{CJK}" not in latex


def test_long_hash_has_safe_line_breaks() -> None:
    digest = "a" * 64
    latex = markdown_to_latex(f"# Evidence\n\nSHA-256: `{digest}`")
    assert "\\allowbreak{}" in latex
    assert "\\setlength{\\emergencystretch}{3em}" in latex


def test_inline_math_is_preserved_as_latex() -> None:
    latex = markdown_to_latex("# Metrics\n\n$\\text{Macro-F1} = \\frac{1}{C}\\sum_{c=1}^{C}\\text{F1}_c$")
    assert "$\\text{Macro-F1} = \\frac{1}{C}\\sum_{c=1}^{C}\\text{F1}_c$" in latex
    assert "\\textbackslash{}text" not in latex
    assert "\\usepackage{amsmath}" in latex


def test_blockquote_marker_is_not_emitted_and_references_flush_floats() -> None:
    latex = markdown_to_latex("# Study\n\n> Disclosure\n\n## References\n\n* **[1]** Paper")
    assert "> Disclosure" not in latex
    assert "\\FloatBarrier\n\\section{References}" in latex
