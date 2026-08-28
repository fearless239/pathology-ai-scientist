from __future__ import annotations

import re


EN_DISCLOSURE = (
    "> **AI-generation disclosure:** This manuscript was generated with substantial assistance "
    "from Path-AI Scientist, a derivative workflow built on AI-Scientist-v2. All claims and "
    "artifacts require human review."
)

ZH_DISCLOSURE = (
    "> **AI生成披露：** 本文由 Path-AI Scientist 在 AI-Scientist-v2 衍生工作流基础上提供"
    "实质性自动生成协助。所有论断与实验产物均须经过人工审核。"
)


def ensure_disclosure(markdown: str, language: str) -> str:
    disclosure = EN_DISCLOSURE if language == "en" else ZH_DISCLOSURE
    markdown = re.sub(
        r"^#{1,3}\s+(?:AI Assistance Disclosure|AI 辅助声明)\s*$.*?(?=^#{1,3}\s|\Z)",
        "", markdown, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"^\s*>?\s*\*\*AI[- ]generation disclosure:\*\*[^\n]*(?:\n(?!\s*#)[^\n]*)*\n?", "", markdown, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*>?\s*\*\*AI生成披露：\*\*[^\n]*(?:\n(?!\s*#)[^\n]*)*\n?", "", text, flags=re.MULTILINE).strip()
    heading = "AI Assistance Disclosure" if language == "en" else "AI 辅助声明"
    block = f"## {heading}\n\n{disclosure.lstrip('> ')}"
    references = re.search(r"^#{1,3}\s+(?:\d+\.\s*)?(?:References|参考文献)\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if references:
        text = text[:references.start()].rstrip() + "\n\n" + block + "\n\n" + text[references.start():].lstrip()
    else:
        text = text.rstrip() + "\n\n" + block
    return text.rstrip() + "\n"


def has_disclosure(markdown: str, language: str) -> bool:
    marker = "AI-generation disclosure" if language == "en" else "AI生成披露"
    return marker.casefold() in markdown.casefold()
