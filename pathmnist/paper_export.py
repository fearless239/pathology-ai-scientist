from __future__ import annotations

import re
from pathlib import Path


MATH_SYMBOLS = dict(zip('αβγδεθλμσφωΔΣΩ×±≤≥≠∈−',
                       ('alpha','beta','gamma','delta','epsilon','theta','lambda','mu',
                        'sigma','phi','omega','Delta','Sigma','Omega','times','pm',
                        'leq','geq','neq','in','mathord{-}')))


def _unicode_math(value: str) -> str:
    return ''.join('\\ensuremath{\\' + MATH_SYMBOLS[c] + '}' if c in MATH_SYMBOLS else c for c in value)


def _escape_latex(value: str) -> str:
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
    return "".join(replacements.get(char, _unicode_math(char)) for char in value)


def _inline(value: str) -> str:
    math: list[str] = []

    def protect(match: re.Match[str]) -> str:
        math.append(_unicode_math(match.group(0)))
        return f"MATHPLACEHOLDER{len(math) - 1}END"

    value = re.sub(r"\$\$(.+?)\$\$|\$(?!\s)(.+?)(?<!\s)\$", protect, value)
    value = _escape_latex(value)
    value = re.sub(
        r"(?<![0-9a-fA-F])([0-9a-fA-F]{32,})(?![0-9a-fA-F])",
        lambda match: r"\allowbreak{}".join(
            match.group(1)[index : index + 8]
            for index in range(0, len(match.group(1)), 8)
        ),
        value,
    )
    value = re.sub(r"\`\`(.+?)\'\'", r"``\1''", value)
    value = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", value)
    value = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\\emph{\1}", value)
    value = re.sub(r"`([^`]+)`", r"\\texttt{\1}", value)
    for index, expression in enumerate(math):
        value = value.replace(f"MATHPLACEHOLDER{index}END", expression)
    return value


def _separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _heading_title(raw: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", raw).strip()


def _table(lines: list[str]) -> str:
    rows = [line.strip() for line in lines if line.strip()]
    if len(rows) < 2:
        return ""
    cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
    body = [row for row in cells if not _separator_row(row)]
    if len(body) < 2:
        return ""
    columns = max(len(row) for row in body)
    for row in body:
        row.extend([""] * (columns - len(row)))
    column = "p{\\dimexpr\\linewidth/" + str(columns) + "-2\\tabcolsep}"
    result = ["\\begin{longtable}{" + column * columns + "}"]
    result.append("\\toprule")
    result.append(" & ".join(_inline(cell) for cell in body[0]) + " \\\\")
    result.append("\\midrule \\endhead")
    for row in body[1:]:
        result.append(" & ".join(_inline(cell) for cell in row) + " \\\\")
    result.append("\\bottomrule\\end{longtable}")
    return "\n".join(result)


def markdown_to_latex(markdown: str, language: str = "en") -> str:
    lines = markdown.splitlines()
    result = []
    index = 0
    in_list = False
    title = "Untitled"
    title_seen = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            if in_list:
                result.append("\\end{itemize}")
                in_list = False
            result.append(_table(table_lines))
            continue
        if stripped.startswith("```"):
            index += 1
            code = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(_escape_latex(lines[index]))
                index += 1
            index += 1
            result.extend(["\\begin{verbatim}", *code, "\\end{verbatim}"])
            continue
        if not stripped:
            if in_list:
                result.append("\\end{itemize}")
                in_list = False
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            if in_list:
                result.append("\\end{itemize}")
                in_list = False
            text = _heading_title(heading.group(2))
            if not title_seen:
                title = text
                title_seen = True
                index += 1
                continue
            level = len(heading.group(1))
            command = "section" if level <= 2 else ("subsection" if level == 3 else "subsubsection")
            if re.fullmatch(r"References|参考文献", text, re.IGNORECASE):
                result.append("\\FloatBarrier")
            result.append(f"\\{command}{{{_inline(text)}}}")
            index += 1
            continue
        image = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image:
            if in_list:
                result.append("\\end{itemize}")
                in_list = False
            caption, source = image.groups()
            if source.startswith(("http://", "https://")) or ".." in Path(source).parts:
                raise ValueError(f"Unsafe or remote figure path: {source}")
            result.extend([
                "\\begin{figure}[htbp]",
                "\\centering",
                f"\\includegraphics[width=0.92\\linewidth]{{\\detokenize{{{source}}}}}",
                f"\\caption{{{_inline(caption or Path(source).stem.replace('_', ' '))}}}",
                "\\end{figure}",
            ])
            index += 1
            continue
        if stripped.startswith(("- ", "* ")):
            if not in_list:
                result.append("\\begin{itemize}")
                in_list = True
            result.append(f"\\item {_inline(stripped[2:])}")
            index += 1
            continue
        if stripped.startswith("---"):
            index += 1
            continue
        if stripped.startswith(">"):
            result.append(_inline(stripped.lstrip("> ")))
            index += 1
            continue
        result.append(_inline(stripped))
        index += 1
    if in_list:
        result.append("\\end{itemize}")
    if language == "zh":
        packages = (
            "\\usepackage{fontspec}\n"
            '\\XeTeXlinebreaklocale "zh"\n'
            "\\XeTeXlinebreakskip = 0pt plus 1pt\n"
            "\\IfFontExistsTF{Noto Serif CJK SC}{\\setmainfont{Noto Serif CJK SC}}"
            "{\\setmainfont{Microsoft YaHei}}"
        )
        begin = ""
        end = ""
        link_package = ""
    else:
        packages = ""
        begin = ""
        end = ""
        link_package = "\\usepackage[hidelinks]{hyperref}"
    return f"""\\documentclass[10pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{placeins}}
{link_package}
\\setlength{{\\emergencystretch}}{{3em}}
\\sloppy
{packages}
\\title{{{_inline(title)}}}
\\author{{Path-AI Scientist}}
\\date{{}}
\\begin{{document}}
\\maketitle
{begin}
{chr(10).join(result)}
{end}
\\end{{document}}
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render paper Markdown to LaTeX")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        markdown_to_latex(args.input.read_text(encoding="utf-8"), args.language),
        encoding="utf-8",
    )
