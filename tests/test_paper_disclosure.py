from pathmnist.paper_disclosure import ensure_disclosure, has_disclosure


def test_english_disclosure_is_prominent_and_idempotent() -> None:
    original = "# Study\n\n> **AI-Generation Disclosure:** Old text.\n\n## Abstract\n\nEvidence.\n\n## References\n"
    disclosed = ensure_disclosure(original, "en")
    assert disclosed.index("AI-generation disclosure") > disclosed.index("## Abstract")
    assert disclosed.index("AI-generation disclosure") < disclosed.index("## References")
    assert "AI-Generation Disclosure" not in disclosed
    assert has_disclosure(disclosed, "en")
    assert ensure_disclosure(disclosed, "en") == disclosed


def test_chinese_disclosure_is_prominent_and_idempotent() -> None:
    original = "# 研究\n\n## 摘要\n\n证据。\n\n## 参考文献\n"
    disclosed = ensure_disclosure(original, "zh")
    assert disclosed.index("AI生成披露") > disclosed.index("## 摘要")
    assert disclosed.index("AI生成披露") < disclosed.index("## 参考文献")
    assert has_disclosure(disclosed, "zh")
    assert ensure_disclosure(disclosed, "zh") == disclosed
