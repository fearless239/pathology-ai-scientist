import pytest

from pathmnist import literature


def test_verified_search_prefers_semantic_scholar(monkeypatch):
    monkeypatch.setattr(literature, "search_semantic_scholar", lambda query, max_results=8: [{"title": "Histopathology image classification", "authors": "Ada", "year": 2025, "venue": "Journal", "doi": "10.1/s2"}])
    monkeypatch.setattr(literature, "search_crossref", lambda query, max_results=8: pytest.fail("fallback should not run"))
    result = literature.search_verified_literature("pathology")
    assert result[0]["verification"] == "semantic_scholar_api"


def test_verified_search_falls_back_to_crossref(monkeypatch):
    def unavailable(query, max_results=8):
        raise literature.LiteratureError("rate limited")

    monkeypatch.setattr(literature, "search_semantic_scholar", unavailable)
    monkeypatch.setattr(literature, "search_crossref", lambda query, max_results=8: [{"title": "Computational pathology with dynamic convolution", "authors": "Ada", "year": 2025, "venue": "Journal", "doi": "10.1/crossref"}])
    result = literature.search_verified_literature("pathology")
    assert result[0]["verification"] == "crossref_api"


def test_crossref_normalization_requires_stable_doi():
    paper = literature._normalize_crossref_paper(
        {
            "DOI": "10.1000/example",
            "title": ["A pathology study"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "published": {"date-parts": [[2025, 1, 2]]},
            "container-title": ["Journal"],
            "URL": "https://doi.org/10.1000/example",
        }
    )
    assert paper["doi"] == "10.1000/example"
    assert paper["authors"] == "Ada Lovelace"
    assert paper["year"] == 2025


def test_relevance_filter_rejects_non_articles_off_topic_results_and_versions():
    papers = [
        {"title": "Review for a pathology classifier", "authors": "Reviewer", "year": 2025, "doi": "10.1/x/review1"},
        {"title": "Dynamic routing for remote sensing images", "authors": "A", "year": 2024, "doi": "10.1/remote"},
        {"title": "PathMNIST histopathology image classification", "authors": "A", "year": 2024, "venue": "SPIE", "doi": "10.1/path"},
        {"title": "PathMNIST histopathology image classification v2", "authors": "A", "year": 2024, "venue": "SPIE", "doi": "10.1/path.v2"},
        {"title": "Conditional computation for image classification", "authors": "B", "year": 2023, "venue": "NeurIPS", "doi": "10.1/method"},
    ]
    accepted, rejected = literature.filter_relevant_literature(papers)
    assert [item["relevance_status"] for item in accepted] == ["directly_relevant", "background_only"]
    assert {item["rejection_reason"] for item in rejected} == {
        "non_article_record", "off_topic_domain", "duplicate_or_version"
    }
