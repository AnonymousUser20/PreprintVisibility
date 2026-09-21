"""Task 3: Metadata availability and missing-data summary by venue and year."""

# pyright: reportMissingModuleSource=false, reportAny=false

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OLD_OUTPUTS = PROJECT_ROOT / "OldOutputs"
OUTPUTS = PROJECT_ROOT / "Outputs"
REVIEWER_PLOTS = PROJECT_ROOT / "ReviewerPlots"
REVIEW_SCORES_CSV = REVIEWER_PLOTS / "paper_review_scores.csv"
NEURIPS_DB_REVIEWS = PROJECT_ROOT / "NeurIPS Dataset and Benchmark" / "Reviews"
REVIEWS_ROOTS = {
    "ICLR": PROJECT_ROOT / "ReviewsICLR" / "Reviews",
    "ICML": PROJECT_ROOT / "ReviewsICML" / "Reviews",
    "NeurIPS": PROJECT_ROOT / "ReviewsNeurIPS" / "Reviews",
}

OUTPUT_TABLE = SCRIPT_DIR / "table3_metadata_availability.csv"
OUTPUT_DATASET = SCRIPT_DIR / "task3_paper_metadata_dataset.csv"
OUTPUT_REPORT = SCRIPT_DIR / "task3_missing_data_report.txt"

# Row order matches Table 3 template.
TABLE_ROWS: list[tuple[str, int]] = [
    ("ICLR", 2023),
    ("ICLR", 2024),
    ("ICLR", 2025),
    ("ICML", 2023),
    ("ICML", 2024),
    ("ICML", 2025),
    ("NeurIPS", 2023),
    ("NeurIPS", 2024),
    ("NeurIPS", 2025),
    ("AAAI", 2023),
    ("AAAI", 2024),
    ("AAAI", 2025),
    ("ACL", 2023),
    ("ACL", 2024),
    ("ACL", 2025),
    ("EMNLP", 2023),
    ("EMNLP", 2024),
    ("EMNLP", 2025),
    ("COLING", 2024),
    ("COLING", 2025),
    ("KDD", 2023),
    ("KDD", 2024),
    ("KDD", 2025),
    ("CVPR", 2023),
    ("CVPR", 2024),
    ("CVPR", 2025),
]

TARGET_VENUES = sorted({venue for venue, _year in TABLE_ROWS})
TARGET_VENUES_UPPER = {v.upper(): v for v in TARGET_VENUES}
REJECT_COUNT_VENUES = {"ICLR", "ICML", "NeurIPS"}

NULL_LIKE = {"", "nan", "none", "null", "na", "not found", "unknown"}
SKIP_DIR_NAMES = {"institute_batches"}
SKIP_FILE_NAMES = {"rank_fill_summary.csv"}

# Track labels used for Papers = sum(tracks).
TRACK_MAIN = "Main"
TRACK_FINDINGS = "Findings"
TRACK_DEMO = "Demo"
TRACK_DB = "D&B"
TRACK_APPL = "Appl."
TRACK_INDUSTRY = "Industry"
TRACK_REJECTED = "Rejected"


def is_non_null(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in NULL_LIKE


def normalize_title(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_venue_year_from_path(path: Path) -> tuple[str | None, int | None]:
    text = str(path).replace("\\", "/")
    venue: str | None = None
    for upper, canonical in TARGET_VENUES_UPPER.items():
        if re.search(rf"(?i)(?:^|/)({re.escape(upper)})(?:/|$|\s|_)", text):
            venue = canonical
            break
        # Match folder-style and flat names like kdd2024_..., acl_2024_main_...
        if re.search(rf"(?i)(?:^|[/_\s-]){re.escape(upper)}(?=$|[/_\s-]|\d{{4}})", text):
            venue = canonical
            break
    year_match = re.search(r"(20(?:23|24|25))", text)
    year = int(year_match.group(1)) if year_match else None
    return venue, year


def classify_track(path: Path, venue: str) -> str:
    """Map source path/filename onto Table 2-style tracks."""
    text = f"{path.stem} {' '.join(path.parts)}".lower().replace("-", "_")

    if any(m in text for m in ("reject", "withdrawn", "desk_rejected", "retracted")):
        return TRACK_REJECTED
    if "dataset" in text or "benchmark" in text or "d&b" in text or "d_and_b" in text:
        return TRACK_DB
    if "finding" in text:
        return TRACK_FINDINGS
    if "demo" in text:
        return TRACK_DEMO
    if any(m in text for m in ("industry", "ind.", "_ind_", "ads_author", "applied")):
        # Table 2 maps KDD applied / ads lists onto Industry.
        return TRACK_INDUSTRY
    if "appl" in text and venue == "KDD":
        return TRACK_APPL
    return TRACK_MAIN


def extract_title(row: dict[str, str]) -> str:
    for key in ("Title", "paper_title", "Paper Title", "title"):
        text = str(row.get(key, "")).strip()
        if is_non_null(text):
            return text
    file_name = str(row.get("file_name", "")).strip()
    if is_non_null(file_name):
        return Path(file_name).stem
    return ""


def extract_affiliation(row: dict[str, str]) -> str:
    for key in (
        "Institutes",
        "Institutions",
        "Authors_Institutes",
        "affiliations",
        "Institution",
    ):
        text = str(row.get(key, "")).strip()
        if is_non_null(text):
            return text
    return ""


def extract_country(row: dict[str, str]) -> str:
    text = str(row.get("Country", "")).strip()
    return text if is_non_null(text) else ""


def extract_rank(row: dict[str, str]) -> str:
    for key in ("Rank_first", "Rank_last", "rank", "Rank"):
        text = str(row.get(key, "")).strip()
        if not is_non_null(text):
            continue
        try:
            _ = float(text)
            return text
        except ValueError:
            continue
    return ""


def extract_date(row: dict[str, str]) -> str:
    for key in ("Date", "Submission Date", "arxiv_date", "Preprint Date"):
        text = str(row.get(key, "")).strip()
        if is_non_null(text):
            return text
    return ""


def metadata_score(item: dict[str, str | int]) -> int:
    return (
        int(item["has_arxiv"])
        + int(item["has_affiliation"])
        + int(item["has_country"])
        + int(item["has_tier"])
        + int(item["has_rating"])
        + int(item["has_confidence"])
    )


def upsert_paper(
    records: dict[tuple[str, int, str], dict[str, str | int]],
    *,
    venue: str,
    year: int,
    title: str,
    track: str,
    has_arxiv: int = 0,
    has_affiliation: int = 0,
    has_country: int = 0,
    has_tier: int = 0,
    has_rating: int = 0,
    has_confidence: int = 0,
    source_file: str = "",
) -> None:
    key = (venue, year, normalize_title(title))
    entry: dict[str, str | int] = {
        "venue": venue,
        "year": year,
        "title": title,
        "track": track,
        "has_arxiv": has_arxiv,
        "has_affiliation": has_affiliation,
        "has_country": has_country,
        "has_tier": has_tier,
        "has_rating": has_rating,
        "has_confidence": has_confidence,
        "source_file": source_file,
    }
    existing = records.get(key)
    if existing is None:
        records[key] = entry
        return

    # Prefer richer metadata; keep Rejected/D&B/Findings/etc. over generic Main
    # only when the existing track is Main/empty.
    merged = dict(existing)
    for field in (
        "has_arxiv",
        "has_affiliation",
        "has_country",
        "has_tier",
        "has_rating",
        "has_confidence",
    ):
        merged[field] = max(int(existing[field]), int(entry[field]))
    if metadata_score(entry) >= metadata_score(existing):
        merged["source_file"] = entry["source_file"] or existing["source_file"]
        merged["title"] = entry["title"]

    existing_track = str(existing["track"])
    if existing_track == TRACK_MAIN and track != TRACK_MAIN:
        merged["track"] = track
    elif existing_track == TRACK_REJECTED and track != TRACK_REJECTED:
        # Accepted record wins over rejected duplicate titles.
        merged["track"] = track
    records[key] = merged


def load_csv_papers(records: dict[tuple[str, int, str], dict[str, str | int]]) -> None:
    for root in (OLD_OUTPUTS, OUTPUTS):
        if not root.exists():
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            if csv_path.name.lower() in SKIP_FILE_NAMES:
                continue
            if any(part.lower() in SKIP_DIR_NAMES for part in csv_path.parts):
                continue

            venue, year = infer_venue_year_from_path(csv_path)
            if venue is None or year is None or (venue, year) not in TABLE_ROWS:
                continue

            track = classify_track(csv_path, venue)
            # Only ICLR/ICML/NeurIPS contribute rejected papers to Table 3.
            if track == TRACK_REJECTED and venue not in REJECT_COUNT_VENUES:
                continue

            try:
                with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None:
                        continue
                    for row in reader:
                        title = extract_title(row)
                        if not title:
                            continue
                        upsert_paper(
                            records,
                            venue=venue,
                            year=year,
                            title=title,
                            track=track,
                            has_arxiv=int(bool(extract_date(row))),
                            has_affiliation=int(bool(extract_affiliation(row))),
                            has_country=int(bool(extract_country(row))),
                            has_tier=int(bool(extract_rank(row))),
                            source_file=str(csv_path.relative_to(PROJECT_ROOT)),
                        )
            except (OSError, UnicodeDecodeError, csv.Error):
                continue


def load_neurips_db_papers(
    records: dict[tuple[str, int, str], dict[str, str | int]],
) -> None:
    if not NEURIPS_DB_REVIEWS.exists():
        return
    for year in (2023, 2024, 2025):
        year_dir = NEURIPS_DB_REVIEWS / str(year)
        if not year_dir.exists():
            continue
        for json_path in year_dir.rglob("*.json"):
            title = json_path.stem
            upsert_paper(
                records,
                venue="NeurIPS",
                year=year,
                title=title,
                track=TRACK_DB,
                source_file=str(json_path.relative_to(PROJECT_ROOT)),
            )


def load_reject_json_papers(
    records: dict[tuple[str, int, str], dict[str, str | int]],
) -> None:
    """Add rejected papers from Reviews* folders when OldOutputs rejects are incomplete."""
    for venue, root in REVIEWS_ROOTS.items():
        if not root.exists():
            continue
        for year_dir in sorted(root.glob(f"{venue} 20*")):
            year_match = re.search(r"(2023|2024|2025)", year_dir.name)
            if not year_match:
                continue
            year = int(year_match.group(1))
            if (venue, year) not in TABLE_ROWS:
                continue
            for sub in year_dir.iterdir():
                if not sub.is_dir():
                    continue
                track = classify_track(sub, venue)
                if track != TRACK_REJECTED:
                    continue
                for json_path in sub.glob("*.json"):
                    title = json_path.stem
                    # Prefer title from JSON if present.
                    try:
                        payload = json.loads(json_path.read_text(encoding="utf-8"))
                        if isinstance(payload, dict):
                            for key in ("title", "Title", "paper_title"):
                                raw = str(payload.get(key, "")).strip()
                                if is_non_null(raw):
                                    title = raw
                                    break
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                        pass
                    upsert_paper(
                        records,
                        venue=venue,
                        year=year,
                        title=title,
                        track=TRACK_REJECTED,
                        source_file=str(json_path.relative_to(PROJECT_ROOT)),
                    )


def attach_review_scores(
    records: dict[tuple[str, int, str], dict[str, str | int]],
) -> None:
    """
    Match ratings/confidence and, for ICLR/ICML/NeurIPS, add missing accepted
    Main-track papers from paper_review_scores (aligns with Table 2 Main counts).
    """
    if not REVIEW_SCORES_CSV.exists():
        return

    with open(REVIEW_SCORES_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            venue = str(row.get("Conference", "")).strip()
            try:
                year = int(str(row.get("Year", "")).strip())
            except ValueError:
                continue
            title = str(row.get("Paper Title", "")).strip()
            if not title or (venue, year) not in TABLE_ROWS:
                continue

            key = (venue, year, normalize_title(title))
            entry = records.get(key)

            has_rating = 0
            has_confidence = 0
            rating_text = str(row.get("rating", "")).strip()
            conf_text = str(row.get("confidence", "")).strip()
            if is_non_null(rating_text):
                try:
                    _ = float(rating_text)
                    has_rating = 1
                except ValueError:
                    pass
            if is_non_null(conf_text):
                try:
                    _ = float(conf_text)
                    has_confidence = 1
                except ValueError:
                    pass

            if entry is None:
                # Only seed Main accepted papers for venues that use review scores
                # as the Main-track inventory (ICLR/ICML/NeurIPS).
                if venue not in REJECT_COUNT_VENUES:
                    continue
                upsert_paper(
                    records,
                    venue=venue,
                    year=year,
                    title=title,
                    track=TRACK_MAIN,
                    has_country=int(bool(extract_country(row))),
                    has_tier=int(bool(extract_rank(row))),
                    has_rating=has_rating,
                    has_confidence=has_confidence,
                    source_file=str(REVIEW_SCORES_CSV.relative_to(PROJECT_ROOT)),
                )
                continue

            entry["has_rating"] = max(int(entry["has_rating"]), has_rating)
            entry["has_confidence"] = max(int(entry["has_confidence"]), has_confidence)
            if not int(entry["has_country"]) and extract_country(row):
                entry["has_country"] = 1
            if not int(entry["has_tier"]) and extract_rank(row):
                entry["has_tier"] = 1
            # If a rejected duplicate later matched an accepted review paper, keep accepted.
            if str(entry["track"]) == TRACK_REJECTED:
                entry["track"] = TRACK_MAIN


def format_count_pct(count: int, total: int) -> str:
    if total <= 0:
        return "0 (0.0%)"
    pct = 100.0 * count / total
    return f"{count} ({pct:.1f}%)"


def build_table3(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for venue, year in TABLE_ROWS:
        subset = df[(df["venue"] == venue) & (df["year"] == year)]
        n = int(len(subset))
        track_sum = int(subset["track"].value_counts().sum()) if n else 0
        if track_sum != n:
            raise RuntimeError(
                f"Track sum mismatch for {venue} {year}: papers={n}, track_sum={track_sum}"
            )

        rows.append(
            {
                "Venue": venue,
                "Year": year,
                "Papers": n,
                "arXiv": format_count_pct(int(subset["has_arxiv"].sum()), n) if n else "0 (0.0%)",
                "Affil.": format_count_pct(int(subset["has_affiliation"].sum()), n)
                if n
                else "0 (0.0%)",
                "Country": format_count_pct(int(subset["has_country"].sum()), n)
                if n
                else "0 (0.0%)",
                "Tier": format_count_pct(int(subset["has_tier"].sum()), n) if n else "0 (0.0%)",
                "Rating": format_count_pct(int(subset["has_rating"].sum()), n) if n else "0 (0.0%)",
                "Conf.": format_count_pct(int(subset["has_confidence"].sum()), n)
                if n
                else "0 (0.0%)",
                "Track": "",
            }
        )
    return pd.DataFrame(rows)


def missing_data_report(df: pd.DataFrame, table3: pd.DataFrame) -> str:
    lines = [
        "Task 3 metadata availability report",
        "Sources: OldOutputs + Outputs + ReviewerPlots/paper_review_scores.csv",
        "         + NeurIPS Dataset and Benchmark (D&B)",
        "         + Reviews* reject folders for ICLR/ICML/NeurIPS",
        "Papers = sum of tracks (Main/Findings/Demo/D&B/Appl./Industry[/Rejected]).",
        "Track column in Table 3 left blank by design.",
        "",
        f"Total unique papers: {len(df)}",
        "",
        "Track breakdown by venue-year (should sum to Papers):",
    ]
    for venue, year in TABLE_ROWS:
        subset = df[(df["venue"] == venue) & (df["year"] == year)]
        counts = subset["track"].value_counts().to_dict()
        parts = [f"{track}={count}" for track, count in sorted(counts.items())]
        lines.append(
            f"  {venue:8} {year}: Papers={len(subset):5} | "
            + (", ".join(parts) if parts else "(none)")
        )

    lines.extend(["", "Table 3 preview:"])
    for _, row in table3.iterrows():
        preview = (
            f"  {row['Venue']:8} {row['Year']}  papers={row['Papers']:>5}  "
            + f"arXiv={row['arXiv']:>16}  Affil.={row['Affil.']:>16}  "
            + f"Country={row['Country']:>16}  Tier={row['Tier']:>16}  "
            + f"Rating={row['Rating']:>16}  Conf.={row['Conf.']:>16}"
        )
        lines.append(preview)
    return "\n".join(lines)


def main() -> None:
    records: dict[tuple[str, int, str], dict[str, str | int]] = {}
    load_csv_papers(records)
    load_neurips_db_papers(records)
    load_reject_json_papers(records)
    attach_review_scores(records)

    df = pd.DataFrame(list(records.values()))
    if df.empty:
        raise RuntimeError("No papers loaded for Task 3.")

    df = df.sort_values(["venue", "year", "title"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")

    table3 = build_table3(df)
    table3.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8")

    report = missing_data_report(df, table3)
    _ = OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved paper dataset: {OUTPUT_DATASET}")
    print(f"Saved Table 3: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
