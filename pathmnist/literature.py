"""Anonymous Semantic Scholar access mirroring the vendored upstream tool protocol.

The request endpoint, query parameters, optional ``X-API-KEY`` header, and
exponential backoff match ``ai_scientist/tools/semantic_scholar.py`` from the
read-only upstream snapshot. This module keeps the protocol but uses only the
standard library so it runs in every pinned container without new dependencies.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import re

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_SEARCH_URL = "https://api.crossref.org/works"
FIELDS = "paperId,externalIds,title,authors,venue,year,abstract,citationCount,url"
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)
MAX_AUTHORS_SHOWN = 4
MAX_ABSTRACT_CHARS = 400

NON_ARTICLE_PATTERNS = (
    r"^review for\b", r"^decision letter\b", r"^table\s+\d+:",
    r"^figure\s+\d+:", r"^peer review\b", r"^author response\b",
)
OFF_TOPIC_TERMS = (
    "document image", "remote sensing", "vehicle routing", "traffic routing",
    "large reasoning model", "large language model", "internal thinking",
    "mixture of experts", "moe load balance",
)
PATHOLOGY_TERMS = (
    "pathology", "pathological", "histopathology", "histopathological", "pathmnist",
    "medical image", "tissue classification", "staining", "whole slide",
)
METHOD_TERMS = (
    "image classification", "computer vision", "convolution", "early exit",
    "conditional computation", "dynamic routing", "adaptive resolution",
    "shortcut learning", "domain shift", "robust classification",
)


class LiteratureError(RuntimeError):
    """Raised when the Semantic Scholar API stays unavailable within policy."""


def search_semantic_scholar(
    query: str,
    max_results: int = 8,
    timeout_seconds: float = 15.0,
):
    """Search papers for one query, retrying transient failures with backoff."""
    query = query.strip()
    if not query:
        return []
    parameters = urllib.parse.urlencode(
        {"query": query, "limit": max_results, "fields": FIELDS}
    )
    headers = {"Accept": "application/json"}
    api_key = os.getenv("S2_API_KEY", "").strip()
    if api_key:
        headers["X-API-KEY"] = api_key
    request = urllib.request.Request(f"{SEARCH_URL}?{parameters}", headers=headers)
    last_error = None
    for attempt, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            papers = payload.get("data") or []
            normalized = [_normalize_paper(paper) for paper in papers]
            return [paper for paper in normalized if paper["title"]]
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
    raise LiteratureError(
        f"Semantic Scholar search failed after {attempt + 1} attempts "
        f"for {query!r}: {last_error}"
    )


def search_crossref(
    query: str,
    max_results: int = 8,
    timeout_seconds: float = 15.0,
):
    """Search Crossref as a keyless metadata fallback with stable DOI evidence."""
    query = query.strip()
    if not query:
        return []
    parameters = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "rows": max_results,
            "select": "DOI,title,author,published,container-title,URL,abstract,is-referenced-by-count",
        }
    )
    request = urllib.request.Request(
        f"{CROSSREF_SEARCH_URL}?{parameters}",
        headers={
            "Accept": "application/json",
            "User-Agent": "Path-Scientist-Agent/0.1 (public-beta literature verification)",
        },
    )
    last_error = None
    for attempt, delay in enumerate((0.0, *RETRY_DELAYS_SECONDS)):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            items = ((payload.get("message") or {}).get("items") or [])
            normalized = [_normalize_crossref_paper(item) for item in items]
            return [paper for paper in normalized if paper["title"] and paper["doi"]]
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
    raise LiteratureError(
        f"Crossref search failed after {attempt + 1} attempts for {query!r}: {last_error}"
    )


def search_verified_literature(query: str, max_results: int = 8):
    """Use Semantic Scholar first and fall back to Crossref when unavailable."""
    failures = []
    for provider, search in (
        ("semantic_scholar_api", search_semantic_scholar),
        ("crossref_api", search_crossref),
    ):
        try:
            papers = search(query, max_results=max_results)
        except LiteratureError as exc:
            failures.append(str(exc))
            continue
        accepted, _ = filter_relevant_literature(papers)
        if accepted:
            return [{"verification": provider, **paper} for paper in accepted]
    raise LiteratureError("; ".join(failures) or f"No verified literature found for {query!r}")


def canonical_reference_key(paper: dict) -> str:
    doi = str(paper.get("doi") or "").strip().casefold()
    doi = re.sub(r"\.v\d+$", "", doi)
    if doi:
        return f"doi:{doi}"
    corpus_id = str(paper.get("corpus_id") or "").strip().casefold()
    if corpus_id:
        return f"corpus:{corpus_id}"
    return "title:" + re.sub(r"\W+", " ", str(paper.get("title") or "").casefold()).strip()


def assess_reference(paper: dict) -> tuple[str, str]:
    title = str(paper.get("title") or "").strip()
    text = " ".join(
        str(paper.get(field) or "") for field in ("title", "abstract", "venue")
    ).casefold()
    if not title:
        return "rejected", "missing_title"
    if any(re.search(pattern, title, re.I) for pattern in NON_ARTICLE_PATTERNS):
        return "rejected", "non_article_record"
    doi = str(paper.get("doi") or "").casefold()
    if any(token in doi for token in ("/review", "/decision", "/table-", "/figure-")):
        return "rejected", "non_article_doi"
    if not paper.get("authors") or not paper.get("year"):
        return "rejected", "incomplete_bibliographic_metadata"
    if not any(paper.get(field) for field in ("doi", "corpus_id", "pmid", "url")):
        return "rejected", "missing_stable_identifier"
    has_pathology = any(term in text for term in PATHOLOGY_TERMS)
    if any(term in text for term in OFF_TOPIC_TERMS) and not has_pathology:
        return "rejected", "off_topic_domain"
    if has_pathology:
        return "directly_relevant", "pathology_or_medical_imaging"
    if any(term in text for term in METHOD_TERMS):
        return "background_only", "relevant_general_method"
    return "rejected", "no_pathology_or_method_relevance"


def filter_relevant_literature(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for paper in papers:
        status, reason = assess_reference(paper)
        key = canonical_reference_key(paper)
        if not key or key in seen:
            rejected.append({**paper, "relevance_status": "rejected", "rejection_reason": "duplicate_or_version"})
            continue
        seen.add(key)
        if status == "rejected":
            rejected.append({**paper, "relevance_status": status, "rejection_reason": reason})
        else:
            accepted.append({**paper, "relevance_status": status, "relevance_reason": reason})
    return accepted, rejected


def _normalize_paper(paper):
    authors = [
        author.get("name", "").strip()
        for author in paper.get("authors") or []
        if author.get("name")
    ]
    if len(authors) > MAX_AUTHORS_SHOWN:
        authors = authors[:MAX_AUTHORS_SHOWN] + ["et al."]
    abstract = (paper.get("abstract") or "").strip()
    return {
        "corpus_id": (paper.get("paperId") or "").strip(),
        "doi": str((paper.get("externalIds") or {}).get("DOI") or "").strip(),
        "pmid": str((paper.get("externalIds") or {}).get("PubMed") or "").strip(),
        "title": (paper.get("title") or "").strip(),
        "authors": ", ".join(authors),
        "venue": (paper.get("venue") or "").strip(),
        "year": paper.get("year"),
        "citation_count": paper.get("citationCount"),
        "url": (paper.get("url") or "").strip(),
        "abstract": abstract[:MAX_ABSTRACT_CHARS],
    }


def _normalize_crossref_paper(paper):
    authors = []
    for author in paper.get("author") or []:
        name = " ".join(
            part for part in (author.get("given", "").strip(), author.get("family", "").strip()) if part
        )
        if name:
            authors.append(name)
    if len(authors) > MAX_AUTHORS_SHOWN:
        authors = authors[:MAX_AUTHORS_SHOWN] + ["et al."]
    date_parts = ((paper.get("published") or {}).get("date-parts") or [[]])
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    titles = paper.get("title") or []
    venues = paper.get("container-title") or []
    abstract = (paper.get("abstract") or "").strip()
    return {
        "corpus_id": "",
        "doi": str(paper.get("DOI") or "").strip(),
        "pmid": "",
        "title": str(titles[0] if titles else "").strip(),
        "authors": ", ".join(authors),
        "venue": str(venues[0] if venues else "").strip(),
        "year": year,
        "citation_count": paper.get("is-referenced-by-count"),
        "url": str(paper.get("URL") or "").strip(),
        "abstract": abstract[:MAX_ABSTRACT_CHARS],
    }
