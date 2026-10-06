"""Task 5b: accumulate ICLR papers with both rating and confidence.

Walks ReviewsICLR including reject / withdrawn / desk-rejected folders.
Does not fit OLS. Does not modify paper_review_scores.csv or Task 5 scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ICLR_ANALYSIS = PROJECT_ROOT / "ICLR_Analysis"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ICLR_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ICLR_ANALYSIS))

import extract_review_scores as ext
import task4_iclr_submission_acceptance as t4

ICLR_REVIEWS = PROJECT_ROOT / "ReviewsICLR" / "Reviews"
OLDOUTPUTS_ICLR = PROJECT_ROOT / "OldOutputs" / "ICLR"
EXISTING_SCORES = SCRIPT_DIR / "paper_review_scores.csv"
OLDARXIV_ICLR = PROJECT_ROOT / "OldArXiv" / "ICLR"
DEADLINES_CSV = PROJECT_ROOT / "ConfusionMatrix" / "conference_deadlines.csv"
TASK4_DATASET = ICLR_ANALYSIS / "task4_submission_dataset.csv"
OUTPUT_SAMPLE = SCRIPT_DIR / "task5b_iclr_sample.csv"
OUTPUT_REPORT = SCRIPT_DIR / "task5b_iclr_sample_report.txt"

SKIP_FOLDER_NAMES = {
    "submitted",
    ".ipynb_checkpoints",
}
REJECT_MARKERS = (
    "desk_rejected",
    "withdrawn_rejected",
    "withdrawn",
    "reject",
    "rejected",
)
ACCEPT_MARKERS = (
    "poster",
    "oral",
    "spotlight",
    "top_5",
    "top25",
    "accepted",
)
FIELDNAMES = [
    "Paper Title",
    "Conference",
    "Year",
    "rating",
    "confidence",
    "Group",
    "Date",
    "PreprintVisible",
    "Country",
    "Rank",
    "outcome",
    "source_folder",
    "source_file",
]


def normalize_title(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_outcome(path: Path) -> str:
    parts = {ext.normalize_name(part) for part in path.parts}
    if any(marker in parts for marker in REJECT_MARKERS):
        return "rejected"
    if any(any(marker in part for marker in ACCEPT_MARKERS) for part in parts):
        return "accepted"
    return "unknown"


def should_skip_folder(path: Path) -> bool:
    return any(ext.normalize_name(part) in SKIP_FOLDER_NAMES for part in path.parts)


def extract_scores_keep_rejects(json_path: Path) -> dict | None:
    year = ext.year_from_path(json_path)
    if year is None:
        return None
    with json_path.open(encoding="utf-8") as fp:
        data = json.load(fp)
    notes = data.get("notes", [])
    if not isinstance(notes, list):
        return None
    ratings: list[float] = []
    confidences: list[float] = []
    for note in notes:
        if not isinstance(note, dict) or not ext.is_official_review(note):
            continue
        content = note.get("content", {})
        rating = ext.extract_numeric_score(
            content.get(
                "rating",
                content.get("recommendation", content.get("overall_recommendation")),
            )
        )
        confidence = ext.extract_numeric_score(content.get("confidence"))
        if rating is not None:
            ratings.append(rating)
        if confidence is not None:
            confidences.append(confidence)
    if not ratings or not confidences:
        return None
    return {
        "Paper Title": ext.extract_title(notes, json_path.stem),
        "Conference": "ICLR",
        "Year": year,
        "rating": ext.mean_or_blank(ratings),
        "confidence": ext.mean_or_blank(confidences),
    }


def load_oldoutputs_lookup() -> dict[tuple[int, str], dict[str, str]]:
    lookup: dict[tuple[int, str], dict[str, str]] = {}
    if not OLDOUTPUTS_ICLR.exists():
        return lookup
    for csv_path in OLDOUTPUTS_ICLR.rglob("*.csv"):
        year = ext.year_from_path(csv_path)
        if year is None:
            continue
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                title = str(row.get("Title", "")).strip()
                if not title:
                    continue
                key = (year, normalize_title(title))
                lookup[key] = {
                    "Country": str(row.get("Country", "")).strip(),
                    "Rank": str(row.get("Rank_first", "")).strip(),
                    "Date": str(row.get("Date", "")).strip(),
                }
    return lookup


def load_existing_iclr_scores() -> dict[tuple[int, str], dict[str, object]]:
    out: dict[tuple[int, str], dict[str, object]] = {}
    if not EXISTING_SCORES.exists():
        return out
    df = pd.read_csv(EXISTING_SCORES)
    df = df[df["Conference"].astype(str).str.strip() == "ICLR"].copy()
    for row in df.to_dict(orient="records"):
        title = str(row.get("Paper Title", "")).strip()
        try:
            year = int(row.get("Year"))
        except (TypeError, ValueError):
            continue
        rating = pd.to_numeric(row.get("rating"), errors="coerce")
        confidence = pd.to_numeric(row.get("confidence"), errors="coerce")
        if pd.isna(rating) or pd.isna(confidence) or not title:
            continue
        out[(year, normalize_title(title))] = {
            "Paper Title": title,
            "Conference": "ICLR",
            "Year": year,
            "rating": float(rating),
            "confidence": float(confidence),
            "Group": str(row.get("Group", "")).strip() or "No Preprint",
            "Country": str(row.get("Country", "")).strip(),
            "Rank": str(row.get("Rank", "")).strip(),
            "outcome": "accepted",
            "source_folder": "paper_review_scores.csv",
            "source_file": "paper_review_scores.csv",
        }
    return out


def prefer_row(existing: dict[str, object], new: dict[str, object]) -> dict[str, object]:
    if existing.get("outcome") == "accepted" and new.get("outcome") != "accepted":
        merged = dict(existing)
    elif new.get("outcome") == "accepted" and existing.get("outcome") != "accepted":
        merged = dict(new)
    else:
        merged = dict(existing)
    for field in ("Country", "Rank", "Group"):
        if not str(merged.get(field, "")).strip() and str(new.get(field, "")).strip():
            merged[field] = new[field]
    return merged


def collect_from_reviews() -> dict[tuple[int, str], dict[str, object]]:
    records: dict[tuple[int, str], dict[str, object]] = {}
    json_paths = [
        path
        for path in sorted(ICLR_REVIEWS.rglob("*.json"))
        if not should_skip_folder(path)
    ]
    print(f"Scanning {len(json_paths)} ICLR review JSON files...")
    kept = 0
    for i, json_path in enumerate(json_paths, start=1):
        if i % 2000 == 0:
            print(f"  {i}/{len(json_paths)} scanned, {kept} with both scores")
        try:
            scores = extract_scores_keep_rejects(json_path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skipping unreadable file {json_path}: {exc}")
            continue
        if not scores:
            continue
        title = str(scores["Paper Title"]).strip()
        year = int(scores["Year"])
        key = (year, normalize_title(title))
        rel = json_path.relative_to(PROJECT_ROOT)
        row = {
            "Paper Title": title,
            "Conference": "ICLR",
            "Year": year,
            "rating": scores["rating"],
            "confidence": scores["confidence"],
            "Group": "No Preprint",
            "Country": "",
            "Rank": "",
            "outcome": classify_outcome(json_path),
            "source_folder": json_path.parent.name,
            "source_file": str(rel),
        }
        existing = records.get(key)
        records[key] = prefer_row(existing, row) if existing else row
        kept += 1
    print(f"Review JSONs with both scores (pre-dedupe hits): {kept}")
    print(f"Unique year+title from reviews: {len(records)}")
    return records


def attach_metadata(
    records: dict[tuple[int, str], dict[str, object]],
    oldoutputs: dict[tuple[int, str], dict[str, str]],
    group_lookup: dict[tuple[str, int, str], str],
) -> None:
    for (year, norm_title), row in records.items():
        meta = oldoutputs.get((year, norm_title))
        if meta:
            if not str(row.get("Country", "")).strip():
                row["Country"] = meta.get("Country", "")
            if not str(row.get("Rank", "")).strip():
                row["Rank"] = meta.get("Rank", "")
        group = group_lookup.get(("ICLR", year, ext.normalize_title(row["Paper Title"])))
        if group:
            row["Group"] = group


def _remember_date(
    lookup: dict[tuple[int, str], date],
    year: int,
    title: object,
    raw: object,
) -> None:
    parsed = t4.parse_date(raw)
    if parsed is None:
        return
    key = (year, normalize_title(title))
    current = lookup.get(key)
    if current is None or parsed < current:
        lookup[key] = parsed


def load_date_lookup() -> dict[tuple[int, str], date]:
    lookup: dict[tuple[int, str], date] = {}

    if OLDOUTPUTS_ICLR.exists():
        for csv_path in OLDOUTPUTS_ICLR.rglob("*.csv"):
            year = ext.year_from_path(csv_path)
            if year is None:
                continue
            with csv_path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    _remember_date(lookup, year, row.get("Title", ""), row.get("Date", ""))

    if OLDARXIV_ICLR.exists():
        for csv_path in OLDARXIV_ICLR.rglob("*_arxiv.csv"):
            year = ext.year_from_path(csv_path)
            if year is None:
                continue
            with csv_path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    _remember_date(
                        lookup,
                        year,
                        row.get("Title", ""),
                        row.get("Submission Date", row.get("Date", "")),
                    )

    if TASK4_DATASET.exists():
        task4 = pd.read_csv(TASK4_DATASET)
        for row in task4.to_dict(orient="records"):
            try:
                year = int(row.get("year"))
            except (TypeError, ValueError):
                continue
            _remember_date(lookup, year, row.get("title", ""), row.get("date_raw", ""))

    return lookup


def fill_preprint_visible(df: pd.DataFrame) -> pd.DataFrame:
    deadlines = t4.load_deadline_windows(DEADLINES_CSV)
    dates = load_date_lookup()
    visible: list[int] = []
    date_text: list[str] = []
    groups: list[str] = []
    title_col = "Paper Title" if "Paper Title" in df.columns else df.columns[0]
    for title, year in zip(df[title_col], df["Year"]):
        parsed = dates.get((int(year), normalize_title(title)))
        window = deadlines.get(("ICLR", int(year)))
        if parsed is None or window is None:
            visible.append(0)
            date_text.append("")
            groups.append("No Preprint")
            continue
        submission, _review = window
        is_visible = int(parsed < submission)
        visible.append(is_visible)
        date_text.append(parsed.isoformat())
        groups.append("Preprint" if is_visible else "No Preprint")
    out = df.copy()
    out["Date"] = date_text
    out["PreprintVisible"] = visible
    out["Group"] = groups
    return out


def write_report(df: pd.DataFrame) -> str:
    both = int((df["rating"].notna() & df["confidence"].notna()).sum())
    lines = [
        "Task 5b ICLR sample accumulation",
        f"Unique papers with both rating and confidence: {both}",
        f"Accepted: {int((df['outcome']=='accepted').sum())}",
        f"Rejected: {int((df['outcome']=='rejected').sum())}",
        f"Unknown outcome: {int((df['outcome']=='unknown').sum())}",
        "",
        "By year:",
    ]
    for year, grp in df.groupby("Year"):
        lines.append(
            f"  {year}: n={len(grp)} | accepted={int((grp['outcome']=='accepted').sum())} "
            f"| rejected={int((grp['outcome']=='rejected').sum())}"
        )
    lines.extend(["", "By source folder:"])
    for folder, count in df["source_folder"].value_counts().items():
        lines.append(f"  {folder}: {count}")
    if "PreprintVisible" in df.columns:
        vis = int(pd.to_numeric(df["PreprintVisible"], errors="coerce").fillna(0).sum())
        n = len(df)
        pct = 100 * vis / n if n else 0
        lines.extend(
            [
                "",
                "PreprintVisible (date strictly before ICLR submission deadline):",
                f"  1: {vis} ({pct:.1f}%)",
                f"  0: {n - vis} ({100 - pct:.1f}%)",
            ]
        )
    return "\n".join(lines)


def load_group_lookup() -> dict[tuple[str, int, str], str]:
    try:
        return ext.build_group_lookup()
    except FileNotFoundError:
        return {}


def main() -> None:
    group_lookup = load_group_lookup()
    oldoutputs = load_oldoutputs_lookup()
    records = collect_from_reviews()
    existing = load_existing_iclr_scores()
    added_from_csv = 0
    for key, row in existing.items():
        if key not in records:
            records[key] = row
            added_from_csv += 1
        else:
            records[key] = prefer_row(records[key], row)
    print(f"Added ICLR rows only in paper_review_scores.csv: {added_from_csv}")

    attach_metadata(records, oldoutputs, group_lookup)
    rows = list(records.values())
    df = pd.DataFrame(rows)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["rating"].notna() & df["confidence"].notna()].copy()
    df = fill_preprint_visible(df)
    df = df.sort_values(["Year", "outcome", "Paper Title"]).reset_index(drop=True)
    save_sample(df)


def save_sample(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_SAMPLE, index=False, encoding="utf-8-sig")
    report = write_report(df)
    _ = OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved sample: {OUTPUT_SAMPLE}")
    print(f"Saved report: {OUTPUT_REPORT}")


def fill_existing_sample() -> None:
    df = pd.read_csv(OUTPUT_SAMPLE)
    df = fill_preprint_visible(df)
    save_sample(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fill-only",
        action="store_true",
        help="Fill PreprintVisible on the existing Task 5b sample without rescanning JSONs.",
    )
    args = parser.parse_args()
    if args.fill_only:
        fill_existing_sample()
    else:
        main()
