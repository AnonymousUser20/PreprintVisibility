from __future__ import annotations

import ast
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import TypedDict, cast
from urllib.parse import urlencode

import feedparser
import pandas as pd
import requests
from tqdm import tqdm


class ArxivCandidate(TypedDict):
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    url: str


class ScoredCandidate(TypedDict):
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    url: str
    title_similarity: float
    author_overlap: float | None
    combined_score: float


class ClassificationResult(TypedDict):
    status: str
    best: ScoredCandidate | None
    strong_candidates: list[ScoredCandidate]


class PaperRecord(TypedDict):
    source_csv: str
    csv_row: int
    title: str
    authors: object
    file_name: str

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OLD_OUTPUTS_DIR = PROJECT_ROOT / "OldOutputs"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"

PAPERS_PER_CSV = 10
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "arxiv_match_summary.csv"
OUTPUT_CANDIDATES_CSV = PROJECT_ROOT / "arxiv_match_candidates.csv"
CACHE_FILE = PROJECT_ROOT / "arxiv_search_cache.json"

SKIP_DIR_NAMES = {"institute_batches"}
SKIP_FILE_NAMES = {"rank_fill_summary.csv"}

# Number of arXiv candidates to retrieve per title
MAX_RESULTS = 20

# Delay between uncached arXiv API requests (skipped on cache hits)
REQUEST_DELAY = 3.0
API_RETRY_DELAY = 10.0
API_MAX_RETRIES = 2
CHECKPOINT_EVERY = 50

# Similarity thresholds
EXACT_THRESHOLD = 0.995
STRONG_THRESHOLD = 0.95
POSSIBLE_THRESHOLD = 0.85

ARXIV_API = "https://export.arxiv.org/api/query"


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "null", "na"}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: object) -> str:
    """
    Normalize text for matching.

    Example:
        "(QA)^2 Question Answering with Questionable Assumptions"

    becomes approximately:
        "qa 2 question answering with questionable assumptions"
    """

    if is_missing(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Lowercase
    text = text.lower()

    # Replace common LaTeX-like constructs
    text = text.replace("\\", " ")

    # Convert ^2 to 2
    text = re.sub(r"\^\s*2", " 2 ", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_author(author: object) -> str:
    """
    Normalize author name for comparison.
    """

    if is_missing(author):
        return ""

    author = unicodedata.normalize(
        "NFKD",
        str(author)
    )

    author = author.lower()

    author = re.sub(
        r"[^a-z0-9\s]",
        " ",
        author
    )

    author = re.sub(
        r"\s+",
        " ",
        author
    )

    return author.strip()


# ============================================================
# AUTHORS
# ============================================================

def parse_input_authors(authors: object) -> list[str]:
    """
    Convert an Authors field into a list.

    Supports examples such as:

        John Smith; Jane Doe
        ['John Smith', 'Jane Doe']
        John Smith, Jane Doe

    Adjust this function if your CSV uses another format.
    """

    if is_missing(authors):
        return []

    authors = str(authors).strip()

    if not authors:
        return []

    # Handle Python-list-like strings
    if authors.startswith("[") and authors.endswith("]"):

        try:
            result = ast.literal_eval(authors)

            if isinstance(result, list):
                return [
                    normalize_author(x)
                    for x in result
                ]

        except (ValueError, SyntaxError, TypeError):
            pass

    # Prefer semicolon separation
    if ";" in authors:

        parts = authors.split(";")

    else:

        parts = authors.split(",")

    return [
        normalize_author(x)
        for x in parts
        if x.strip()
    ]


def author_overlap(
    input_authors: list[str],
    arxiv_authors: list[str],
) -> float | None:
    """
    Fraction of input authors that approximately appear
    in the arXiv author list.
    """

    if not input_authors:
        return None

    if not arxiv_authors:
        return 0.0

    matched = 0

    for input_author in input_authors:

        best_score = 0

        for arxiv_author in arxiv_authors:

            score = SequenceMatcher(
                None,
                input_author,
                arxiv_author
            ).ratio()

            best_score = max(
                best_score,
                score
            )

        if best_score >= 0.80:
            matched += 1

    return matched / len(input_authors)


# ============================================================
# TITLE SIMILARITY
# ============================================================

def title_similarity(title1: object, title2: object) -> float:
    """
    Calculate normalized title similarity from 0 to 1.
    """

    a = normalize_text(title1)
    b = normalize_text(title2)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# CACHE
# ============================================================

def load_cache() -> dict[str, list[ArxivCandidate]]:

    path = Path(CACHE_FILE)

    if not path.exists():
        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return cast(dict[str, list[ArxivCandidate]], json.load(f))

    except (json.JSONDecodeError, OSError):

        return {}


def save_cache(cache: dict[str, list[ArxivCandidate]]) -> None:

    with open(
        CACHE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# ARXIV SEARCH
# ============================================================

def search_arxiv(
    title: str,
    cache: dict[str, list[ArxivCandidate]],
) -> list[ArxivCandidate]:
    """
    Search arXiv using a title query.
    """

    cache_key = normalize_text(title)

    # -------------------------------------------
    # Use cache if already searched
    # -------------------------------------------

    if cache_key in cache:

        return cache[cache_key]

    query = f'ti:"{title}"'

    params = {
        "search_query": query,
        "start": 0,
        "max_results": MAX_RESULTS
    }

    url = (
        ARXIV_API
        + "?"
        + urlencode(params)
    )

    headers = {
        "User-Agent": "ResearchPaperMatcher/1.0 (academic research)",
    }

    response = None
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            break
        except Exception as e:
            if attempt >= API_MAX_RETRIES:
                tqdm.write(
                    f"API error for title:\n{title}\n{e}"
                )
                return []
            tqdm.write(
                f"API error (retry {attempt + 1}/{API_MAX_RETRIES}): {e}"
            )
            time.sleep(API_RETRY_DELAY * (attempt + 1))

    if response is None:
        return []

    feed = feedparser.parse(
        response.text
    )

    results: list[ArxivCandidate] = []

    for entry in feed.entries:
        entry_id = str(getattr(entry, "id", ""))
        entry_title = str(getattr(entry, "title", ""))
        arxiv_id = entry_id.split("/abs/")[-1]
        arxiv_title = " ".join(entry_title.split())

        authors = [
            normalize_author(a.name)
            for a in getattr(entry, "authors", [])
        ]

        result: ArxivCandidate = {
            "arxiv_id": arxiv_id,
            "title": arxiv_title,
            "authors": authors,
            "published": str(getattr(entry, "published", "")),
            "updated": str(getattr(entry, "updated", "")),
            "url": entry_id,
        }

        results.append(result)

    # Save in cache

    cache[cache_key] = results

    save_cache(cache)

    # Be polite to API

    time.sleep(
        REQUEST_DELAY
    )

    return results


# ============================================================
# SCORE CANDIDATES
# ============================================================

def score_candidates(
    title: str,
    authors: object,
    candidates: list[ArxivCandidate],
) -> list[ScoredCandidate]:

    input_authors = parse_input_authors(
        authors
    )

    scored: list[ScoredCandidate] = []

    for candidate in candidates:

        title_score = title_similarity(
            title,
            candidate["title"]
        )

        overlap = author_overlap(
            input_authors,
            candidate["authors"]
        )

        if overlap is None:

            combined_score = title_score

        else:

            combined_score = (
                0.85 * title_score
                +
                0.15 * overlap
            )

        scored_candidate: ScoredCandidate = {
            "arxiv_id": candidate["arxiv_id"],
            "title": candidate["title"],
            "authors": candidate["authors"],
            "published": candidate["published"],
            "updated": candidate["updated"],
            "url": candidate["url"],
            "title_similarity": title_score,
            "author_overlap": overlap,
            "combined_score": combined_score,
        }

        scored.append(
            scored_candidate
        )

    scored.sort(
        key=lambda x:
            x["combined_score"],
        reverse=True
    )

    return scored


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_match(
    scored_candidates: list[ScoredCandidate],
) -> ClassificationResult:
    """
    Determine whether the title has:

        NO_MATCH
        UNIQUE_MATCH
        MULTIPLE_MATCHES
        AMBIGUOUS
    """

    if not scored_candidates:
        return ClassificationResult(
            status="NO_MATCH",
            best=None,
            strong_candidates=[],
        )

    strong = [
        x
        for x in scored_candidates
        if x["title_similarity"]
        >= STRONG_THRESHOLD
    ]

    exact = [
        x
        for x in scored_candidates
        if x["title_similarity"]
        >= EXACT_THRESHOLD
    ]

    # ------------------------------------------
    # Multiple essentially exact titles
    # ------------------------------------------

    if len(exact) > 1:
        return ClassificationResult(
            status="MULTIPLE_EXACT_MATCHES",
            best=exact[0],
            strong_candidates=exact,
        )

    # ------------------------------------------
    # Exactly one very strong match
    # ------------------------------------------

    if len(exact) == 1:
        return ClassificationResult(
            status="UNIQUE_EXACT_MATCH",
            best=exact[0],
            strong_candidates=exact,
        )

    # ------------------------------------------
    # Multiple strong fuzzy candidates
    # ------------------------------------------

    if len(strong) > 1:

        # Check whether best candidate clearly wins
        best = strong[0]
        second = strong[1]

        margin = (
            best["combined_score"]
            -
            second["combined_score"]
        )

        if margin >= 0.05:
            return ClassificationResult(
                status="UNIQUE_STRONG_MATCH",
                best=best,
                strong_candidates=strong,
            )

        return ClassificationResult(
            status="MULTIPLE_MATCHES",
            best=best,
            strong_candidates=strong,
        )

    # ------------------------------------------
    # One strong fuzzy candidate
    # ------------------------------------------

    if len(strong) == 1:
        return ClassificationResult(
            status="UNIQUE_STRONG_MATCH",
            best=strong[0],
            strong_candidates=strong,
        )

    # ------------------------------------------
    # Possible weaker match
    # ------------------------------------------

    best = scored_candidates[0]

    if best["title_similarity"] >= POSSIBLE_THRESHOLD:
        return ClassificationResult(
            status="AMBIGUOUS",
            best=best,
            strong_candidates=scored_candidates,
        )

    return ClassificationResult(
        status="NO_MATCH",
        best=None,
        strong_candidates=[],
    )


# ============================================================
# PROCESS ONE PAPER
# ============================================================

def process_paper(
    row_number: int,
    title: str,
    authors: object,
    cache: dict[str, list[ArxivCandidate]],
    source_csv: str = "",
) -> tuple[dict[str, object], list[dict[str, object]]]:

    candidates = search_arxiv(
        title,
        cache
    )

    scored = score_candidates(
        title,
        authors,
        candidates
    )

    classification = classify_match(
        scored
    )

    best: ScoredCandidate | None = classification["best"]

    summary: dict[str, object] = {

        "row_number":
            row_number,

        "source_csv":
            source_csv,

        "input_title":
            title,

        "input_authors":
            authors,

        "api_candidates":
            len(candidates),

        "strong_candidates":
            len(
                classification[
                    "strong_candidates"
                ]
            ),

        "status":
            classification[
                "status"
            ],

        "best_arxiv_id":
            best["arxiv_id"]
            if best
            else "",

        "best_arxiv_title":
            best["title"]
            if best
            else "",

        "title_similarity":
            round(
                best[
                    "title_similarity"
                ],
                4
            )
            if best
            else "",

        "author_overlap":
            round(
                best[
                    "author_overlap"
                ],
                4
            )
            if (
                best
                and
                best[
                    "author_overlap"
                ]
                is not None
            )
            else "",

        "combined_score":
            round(
                best[
                    "combined_score"
                ],
                4
            )
            if best
            else "",

        "published":
            best["published"]
            if best
            else "",

        "arxiv_url":
            best["url"]
            if best
            else ""
    }

    detailed: list[dict[str, object]] = []

    for rank, candidate in enumerate(
        scored,
        start=1
    ):

        detailed.append({

            "row_number":
                row_number,

            "source_csv":
                source_csv,

            "input_title":
                title,

            "candidate_rank":
                rank,

            "arxiv_id":
                candidate[
                    "arxiv_id"
                ],

            "arxiv_title":
                candidate[
                    "title"
                ],

            "title_similarity":
                round(
                    candidate[
                        "title_similarity"
                    ],
                    4
                ),

            "author_overlap":
                round(
                    candidate[
                        "author_overlap"
                    ],
                    4
                )
                if (
                    candidate[
                        "author_overlap"
                    ]
                    is not None
                )
                else "",

            "combined_score":
                round(
                    candidate[
                        "combined_score"
                    ],
                    4
                ),

            "published":
                candidate[
                    "published"
                ],

            "url":
                candidate[
                    "url"
                ]
        })

    return summary, detailed


# ============================================================
# CSV DISCOVERY / LOADING
# ============================================================

def discover_csv_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.glob("**/*.csv")):
            if csv_path.name in SKIP_FILE_NAMES:
                continue
            if any(part.lower() in SKIP_DIR_NAMES for part in csv_path.parts):
                continue
            files.append(csv_path)
    return files


def load_papers_from_oldoutputs_csv(csv_path: Path, limit: int) -> list[PaperRecord]:
    df = pd.read_csv(csv_path, nrows=limit)
    if "Title" not in df.columns:
        return []

    papers: list[PaperRecord] = []
    for row_num, (_, row) in enumerate(df.iterrows(), start=1):
        title = str(row["Title"]).strip()
        if not title or title.lower() in {"nan", "none"}:
            continue
        authors = row.get("Authors", "")
        papers.append(
            PaperRecord(
                source_csv=str(csv_path.relative_to(PROJECT_ROOT)),
                csv_row=row_num,
                title=title,
                authors=authors,
                file_name="",
            )
        )
    return papers


def load_papers_from_outputs_csv(csv_path: Path, limit: int) -> list[PaperRecord]:
    df = pd.read_csv(csv_path)
    if "paper_title" not in df.columns:
        return []

    title_col = "paper_title"
    author_col = "author" if "author" in df.columns else None
    file_col = "file_name" if "file_name" in df.columns else None

    grouped: dict[tuple[str, str], list[str]] = {}
    order: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        title = str(row.get(title_col, "")).strip()
        if not title or title.lower() in {"nan", "none"}:
            continue
        file_name = str(row.get(file_col, "")).strip() if file_col else ""
        key = (file_name, title)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        if author_col:
            author = str(row.get(author_col, "")).strip()
            if author and author not in grouped[key]:
                grouped[key].append(author)

    papers: list[PaperRecord] = []
    for csv_row, key in enumerate(order[:limit], start=1):
        file_name, title = key
        author_list = grouped[key]
        authors = str(author_list) if author_list else ""
        papers.append(
            PaperRecord(
                source_csv=str(csv_path.relative_to(PROJECT_ROOT)),
                csv_row=csv_row,
                title=title,
                authors=authors,
                file_name=file_name,
            )
        )
    return papers


def load_papers_from_csv(csv_path: Path, limit: int = PAPERS_PER_CSV) -> list[PaperRecord]:
    try:
        _ = csv_path.relative_to(OUTPUTS_DIR)
        return load_papers_from_outputs_csv(csv_path, limit)
    except ValueError:
        return load_papers_from_oldoutputs_csv(csv_path, limit)


def load_all_papers(csv_files: list[Path], limit: int = PAPERS_PER_CSV) -> list[PaperRecord]:
    papers: list[PaperRecord] = []
    for csv_path in tqdm(csv_files, desc="Loading CSVs", unit="file"):
        loaded = load_papers_from_csv(csv_path, limit=limit)
        papers.extend(loaded)
        tqdm.write(f"{csv_path.relative_to(PROJECT_ROOT)}: {len(loaded)} papers")
    return papers


def save_checkpoint(
    summary_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> None:
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY_CSV, index=False)
    pd.DataFrame(candidate_rows).to_csv(OUTPUT_CANDIDATES_CSV, index=False)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    csv_files = discover_csv_files(OLD_OUTPUTS_DIR, OUTPUTS_DIR)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {OLD_OUTPUTS_DIR} or {OUTPUTS_DIR}"
        )

    print(f"\nDiscovered {len(csv_files)} CSV files")
    print(f"Reading up to {PAPERS_PER_CSV} papers from each file\n")

    papers = load_all_papers(csv_files, limit=PAPERS_PER_CSV)
    if not papers:
        raise ValueError("No papers loaded from the selected CSV files.")

    print(f"\nTotal papers to match: {len(papers)}")

    cache = load_cache()
    summary_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for paper_number, paper in enumerate(
        tqdm(papers, desc="Matching papers", unit="paper"),
        start=1,
    ):
        title = str(paper["title"]).strip()
        authors = paper.get("authors", "")
        source_csv = str(paper.get("source_csv", ""))

        tqdm.write(f"[{paper_number}/{len(papers)}] {title[:100]}")

        try:
            summary, details = process_paper(
                paper_number,
                title,
                authors,
                cache,
                source_csv=source_csv,
            )
            summary["csv_row"] = paper.get("csv_row", "")
            summary_rows.append(summary)
            candidate_rows.extend(details)
            tqdm.write(
                f"  status={summary['status']} candidates={summary['api_candidates']}"
            )
        except Exception as e:
            tqdm.write(f"ERROR: {e}")
            summary_rows.append(
                {
                    "row_number": paper_number,
                    "source_csv": source_csv,
                    "csv_row": paper.get("csv_row", ""),
                    "input_title": title,
                    "input_authors": authors,
                    "api_candidates": "",
                    "strong_candidates": "",
                    "status": "ERROR",
                    "best_arxiv_id": "",
                    "best_arxiv_title": "",
                    "title_similarity": "",
                    "author_overlap": "",
                    "combined_score": "",
                    "published": "",
                    "arxiv_url": "",
                }
            )

        if paper_number % CHECKPOINT_EVERY == 0:
            save_checkpoint(summary_rows, candidate_rows)
            tqdm.write(f"Checkpoint saved after {paper_number} papers.")

    # ============================================
    # Final save
    # ============================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False
    )

    candidate_df.to_csv(
        OUTPUT_CANDIDATES_CSV,
        index=False
    )

    # ============================================
    # Print final statistics
    # ============================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MATCHING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nStatus counts:"
    )

    print(
        summary_df[
            "status"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nSummary saved to:"
    )

    print(
        OUTPUT_SUMMARY_CSV
    )

    print(
        "\nCandidate details saved to:"
    )

    print(
        OUTPUT_CANDIDATES_CSV
    )

    print(
        "\nCache saved to:"
    )

    print(
        CACHE_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()