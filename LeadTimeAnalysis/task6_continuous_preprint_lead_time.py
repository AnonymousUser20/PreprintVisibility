"""Task 6: Continuous preprint lead-time analysis by venue and year."""

# pyright: reportMissingModuleSource=false, reportAny=false

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OLD_OUTPUTS = PROJECT_ROOT / "OldOutputs"
OUTPUTS = PROJECT_ROOT / "Outputs"
DEADLINES_CSV = PROJECT_ROOT / "ConfusionMatrix" / "conference_deadlines.csv"

OUTPUT_TABLE = SCRIPT_DIR / "table6_continuous_preprint_lead_time.csv"
OUTPUT_DATASET = SCRIPT_DIR / "task6_lead_time_dataset.csv"
OUTPUT_REPORT = SCRIPT_DIR / "task6_missing_data_report.txt"

TARGET_VENUES = ("ICLR", "ICML", "NeurIPS")
TARGET_VENUES_UPPER = {v.upper(): v for v in TARGET_VENUES}
TARGET_YEARS = (2023, 2024, 2025)

NULL_LIKE = {"", "nan", "none", "null", "na", "not found"}
DATE_PATTERNS = [
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %B, %Y",
    "%d %b, %Y",
    "%B %Y",
    "%b %Y",
]

SKIP_DIR_NAMES = {"institute_batches"}
SKIP_FILE_NAMES = {"rank_fill_summary.csv"}
REJECTED_FILE_MARKERS = ("reject", "withdrawn", "desk_rejected", "retracted")


def is_rejected_source(path: Path) -> bool:
    stem = path.stem.lower()
    return any(marker in stem for marker in REJECTED_FILE_MARKERS)


def is_non_null(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in NULL_LIKE


def normalize_title(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_date(value: object) -> date | None:
    text = str(value).strip()
    if not is_non_null(text):
        return None
    text = re.sub(r"\s*\(.*?\)\s*", "", text).strip()
    for pattern in DATE_PATTERNS:
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    return None


def parse_deadline_day(value: object) -> date | None:
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", str(value))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%d.%m.%Y").date()


def load_submission_deadlines(path: Path) -> dict[tuple[str, int], date]:
    deadlines: dict[tuple[str, int], date] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            conf_raw = str(row.get("Conference", "")).strip().upper()
            conf = TARGET_VENUES_UPPER.get(conf_raw)
            if conf is None:
                continue
            try:
                year = int(str(row.get("Year", "")).strip())
            except ValueError:
                continue
            submission = parse_deadline_day(row.get("Submission Deadline", ""))
            if submission is not None:
                deadlines[(conf, year)] = submission
    return deadlines


def infer_venue_year_from_path(path: Path) -> tuple[str | None, int | None]:
    text = str(path).replace("\\", "/")
    venue: str | None = None
    for upper, canonical in TARGET_VENUES_UPPER.items():
        if re.search(rf"(?i)(?:^|/)({re.escape(upper)})(?:/|$|\s|_)", text):
            venue = canonical
            break
        if re.search(rf"(?i)\b{re.escape(upper)}\b", path.name):
            venue = canonical
            break

    year_match = re.search(r"(20(?:23|24|25))", text)
    year = int(year_match.group(1)) if year_match else None
    return venue, year


def infer_venue_from_row(row: dict[str, str], path: Path) -> str | None:
    for key in ("conference_name", "Conference", "conference"):
        raw = str(row.get(key, "")).strip().upper()
        if not raw:
            continue
        for upper, canonical in TARGET_VENUES_UPPER.items():
            if upper in raw:
                return canonical
    venue, _year = infer_venue_year_from_path(path)
    return venue


def extract_title(row: dict[str, str]) -> str:
    for key in ("Title", "paper_title", "Paper Title", "title"):
        text = str(row.get(key, "")).strip()
        if text:
            return text
    return ""


def extract_date_raw(row: dict[str, str]) -> str:
    for key in ("Date", "Submission Date", "arxiv_date", "Preprint Date"):
        text = str(row.get(key, "")).strip()
        if is_non_null(text):
            return text
    return ""


def iter_candidate_csvs() -> list[Path]:
    paths: list[Path] = []

    for root in (OLD_OUTPUTS, OUTPUTS):
        if not root.exists():
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            if csv_path.name.lower() in SKIP_FILE_NAMES:
                continue
            if any(part.lower() in SKIP_DIR_NAMES for part in csv_path.parts):
                continue
            if is_rejected_source(csv_path):
                continue
            venue, year = infer_venue_year_from_path(csv_path)
            # Keep Outputs files that may encode venue in row fields.
            if root == OLD_OUTPUTS and (venue is None or year is None):
                continue
            if root == OLD_OUTPUTS and venue not in TARGET_VENUES:
                continue
            paths.append(csv_path)

    return paths


def load_lead_time_records(
    deadlines: dict[tuple[str, int], date],
) -> pd.DataFrame:
    # Keep earliest preprint date per (venue, year, normalized title).
    best: dict[tuple[str, int, str], dict[str, object]] = {}

    for csv_path in iter_candidate_csvs():
        path_venue, path_year = infer_venue_year_from_path(csv_path)
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                for row in reader:
                    venue = infer_venue_from_row(row, csv_path) or path_venue
                    year = path_year
                    if year is None:
                        for key in ("Year", "year"):
                            raw = str(row.get(key, "")).strip()
                            if raw.isdigit():
                                year = int(raw)
                                break
                    if venue not in TARGET_VENUES or year not in TARGET_YEARS:
                        continue

                    title = extract_title(row)
                    if not title:
                        continue
                    date_raw = extract_date_raw(row)
                    preprint_date = parse_date(date_raw)
                    if preprint_date is None:
                        continue

                    deadline = deadlines.get((venue, year))
                    if deadline is None:
                        continue

                    lead_days = (deadline - preprint_date).days
                    key = (venue, year, normalize_title(title))
                    entry = {
                        "venue": venue,
                        "year": year,
                        "title": title,
                        "date_raw": date_raw,
                        "preprint_date": preprint_date.isoformat(),
                        "submission_deadline": deadline.isoformat(),
                        "lead_time_days": lead_days,
                        "source_file": str(csv_path.relative_to(PROJECT_ROOT)),
                    }
                    existing = best.get(key)
                    if existing is None:
                        best[key] = entry
                    else:
                        # Prefer earlier preprint (larger positive lead time).
                        if lead_days > int(existing["lead_time_days"]):
                            best[key] = entry
        except (OSError, UnicodeDecodeError, csv.Error):
            continue

    rows = list(best.values())
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["venue", "year", "title"]).reset_index(drop=True)


def format_iqr(q1: float, q3: float) -> str:
    return f"[{q1:.0f}, {q3:.0f}]"


def format_range(lo: float, hi: float) -> str:
    return f"[{lo:.0f}, {hi:.0f}]"


def build_table6(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for venue in TARGET_VENUES:
        for year in TARGET_YEARS:
            subset = df[(df["venue"] == venue) & (df["year"] == year)]
            if subset.empty:
                rows.append(
                    {
                        "Venue": venue,
                        "Year": year,
                        "N": 0,
                        "Median lead time": "-",
                        "IQR": "-",
                        "Range": "-",
                    }
                )
                continue

            values = subset["lead_time_days"].astype(float)
            q1 = float(values.quantile(0.25))
            median = float(values.quantile(0.50))
            q3 = float(values.quantile(0.75))
            lo = float(values.min())
            hi = float(values.max())
            rows.append(
                {
                    "Venue": venue,
                    "Year": year,
                    "N": int(len(values)),
                    "Median lead time": f"{median:.0f}",
                    "IQR": format_iqr(q1, q3),
                    "Range": format_range(lo, hi),
                }
            )
    return pd.DataFrame(rows)


def missing_data_report(df: pd.DataFrame, deadlines: dict[tuple[str, int], date]) -> str:
    lines = [
        "Task 6 missing-data / coverage report",
        "Lead time (days) = submission_deadline - preprint_date",
        "  positive = preprint before submission deadline",
        "  negative = preprint after submission deadline",
        "",
        f"Papers with computable lead time: {len(df)}",
        "",
        "Submission deadlines used:",
    ]
    for venue in TARGET_VENUES:
        for year in TARGET_YEARS:
            deadline = deadlines.get((venue, year))
            lines.append(
                f"  {venue} {year}: {deadline.isoformat() if deadline else 'MISSING'}"
            )

    lines.extend(["", "By venue-year (N with lead time):"])
    if df.empty:
        lines.append("  (no rows)")
    else:
        for (venue, year), grp in df.groupby(["venue", "year"]):
            lines.append(f"  {venue} {year}: n={len(grp)}")

    if not df.empty:
        lines.extend(
            [
                "",
                f"Lead time overall: median={df['lead_time_days'].median():.0f} | "
                f"min={df['lead_time_days'].min():.0f} | max={df['lead_time_days'].max():.0f}",
                f"Share with lead_time > 0 (before deadline): "
                f"{100 * float((df['lead_time_days'] > 0).mean()):.1f}%",
                f"Share with lead_time < 0 (after deadline): "
                f"{100 * float((df['lead_time_days'] < 0).mean()):.1f}%",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    if not DEADLINES_CSV.exists():
        raise FileNotFoundError(f"Deadlines file not found: {DEADLINES_CSV}")

    deadlines = load_submission_deadlines(DEADLINES_CSV)
    df = load_lead_time_records(deadlines)

    report = missing_data_report(df, deadlines)
    _ = OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(report)

    if not df.empty:
        df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
        print(f"\nSaved lead-time dataset: {OUTPUT_DATASET}")

    table6 = build_table6(df)
    table6.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8")
    print(f"\nSaved Table 6: {OUTPUT_TABLE}")
    print("\nTable 6: Continuous preprint lead time by venue and year")
    print(
        f"{'Venue':10} {'Year':4} {'N':>6}  {'Median':>8}  {'IQR':>14}  {'Range':>16}"
    )
    for _, row in table6.iterrows():
        print(
            f"{str(row['Venue']):10} {str(row['Year']):4} {str(row['N']):>6}  "
            f"{str(row['Median lead time']):>8}  {str(row['IQR']):>14}  {str(row['Range']):>16}"
        )


if __name__ == "__main__":
    main()
