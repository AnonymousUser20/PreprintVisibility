# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnusedCallResult=false

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OLD_OUTPUTS_DIR = PROJECT_ROOT / "OldOutputs"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
DEADLINES_CSV = PROJECT_ROOT / "ConfusionMatrix" / "conference_deadlines.csv"

OUTPUT_PNG = SCRIPT_DIR / "horizontal_glow_stacked_plot.png"
OUTPUT_PDF = SCRIPT_DIR / "horizontal_glow_stacked_plot.pdf"
SUMMARY_CSV = SCRIPT_DIR / "ops_bucket_counts.csv"

YEARS = [2023, 2024, 2025]
BUCKETS = ["CfP->Review", "<30d", "<60d", "<90d", "<180d", ">180d"]
COLORS = [
    "#E57373",  # rose red
    "#FFB74D",  # amber orange
    "#FFF176",  # soft yellow
    "#81C784",  # green
    "#9575CD",  # violet
    "#64B5F6",  # blue
]
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

# These OldOutputs files already encode presentation type in the file name/path.
OLD_PRESENTATION_SOURCES = [
    ("ICLR", 2023, "O", OLD_OUTPUTS_DIR / "ICLR" / "ICLR 2023" / "top_5.csv"),
    ("ICLR", 2023, "P", OLD_OUTPUTS_DIR / "ICLR" / "ICLR 2023" / "Poster.csv"),
    ("ICLR", 2023, "S", OLD_OUTPUTS_DIR / "ICLR" / "ICLR 2023" / "top25.csv"),
    ("ICLR", 2024, "O", OLD_OUTPUTS_DIR / "ICLR" / "ICLR 2024" / "accepted(oral).csv"),
    ("ICLR", 2024, "P", OLD_OUTPUTS_DIR / "ICLR" / "ICLR 2024" / "accepted(poster).csv"),
    (
        "ICLR",
        2024,
        "S",
        OLD_OUTPUTS_DIR / "ICLR" / "ICLR 2024" / "accepted(spotlight).csv",
    ),
    ("ICLR", 2025, "P", OLD_OUTPUTS_DIR / "ICLR" / "ICLR 2025" / "accepted(poster).csv"),
    (
        "ICLR",
        2025,
        "S",
        OLD_OUTPUTS_DIR / "ICLR" / "ICLR 2025" / "accepted(spotlight).csv",
    ),
    (
        "ICML",
        2023,
        "O/P",
        OLD_OUTPUTS_DIR / "ICML" / "ICML 2023" / "accepted (oral & poster).csv",
    ),
    ("ICML", 2024, "O", OLD_OUTPUTS_DIR / "ICML" / "ICML 2024" / "accept(oral).csv"),
    ("ICML", 2024, "P", OLD_OUTPUTS_DIR / "ICML" / "ICML 2024" / "accept(poster).csv"),
    (
        "ICML",
        2024,
        "S",
        OLD_OUTPUTS_DIR / "ICML" / "ICML 2024" / "accept(spotlight).csv",
    ),
    ("ICML", 2025, "O", OLD_OUTPUTS_DIR / "ICML" / "ICML 2025" / "accept(oral).csv"),
    ("ICML", 2025, "P", OLD_OUTPUTS_DIR / "ICML" / "ICML 2025" / "accept(poster).csv"),
    (
        "ICML",
        2025,
        "S",
        OLD_OUTPUTS_DIR / "ICML" / "ICML 2025" / "accept(spotlight).csv",
    ),
    (
        "NeurIPS",
        2023,
        "O",
        OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2023" / "accept(oral).csv",
    ),
    (
        "NeurIPS",
        2023,
        "P",
        OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2023" / "accept(poster).csv",
    ),
    (
        "NeurIPS",
        2023,
        "S",
        OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2023" / "accept(spotlight).csv",
    ),
    (
        "NeurIPS",
        2024,
        "O",
        OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2024" / "accept(oral).csv",
    ),
    (
        "NeurIPS",
        2024,
        "P",
        OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2024" / "accept(poster).csv",
    ),
    (
        "NeurIPS",
        2024,
        "S",
        OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2024" / "accept(spotlight).csv",
    ),
    ("NeurIPS", 2025, "O", OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2025" / "oral.csv"),
    ("NeurIPS", 2025, "P", OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2025" / "poster.csv"),
    (
        "NeurIPS",
        2025,
        "S",
        OLD_OUTPUTS_DIR / "NeurIPS" / "NeurIPS 2025" / "spotlight.csv",
    ),
]

# These are title-only oral lists, so they are matched against Outputs/OldOutputs records.
ORAL_TITLE_SOURCES = [
    ("EMNLP", 2023, "O", SCRIPT_DIR / "2023_emnlp_oral.csv"),
    ("AAAI", 2024, "O", SCRIPT_DIR / "AAAI2024_oral_papers.csv"),
    ("ACL", 2025, "O", SCRIPT_DIR / "oral_acl_2025.csv"),
    ("COLING", 2025, "O", SCRIPT_DIR / "oral_coling_2025.csv"),
    ("EMNLP", 2025, "O", SCRIPT_DIR / "oral_emnlp_2025.csv"),
]


@dataclass(frozen=True)
class PaperRecord:
    title: str
    conference: str
    year: int
    preprint_date: date | None
    source: Path


def is_non_null(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in NULL_LIKE


def normalize_title(value: object) -> str:
    text = str(value or "").lower().replace(".pdf", "")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_title(value: object) -> str:
    return normalize_title(value).replace(" ", "")


def parse_date(value: object) -> date | None:
    text = str(value).strip()
    if not text or text.lower() in NULL_LIKE:
        return None

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


def load_deadline_windows(deadlines_csv: Path) -> dict[tuple[str, int], tuple[date, date]]:
    windows: dict[tuple[str, int], tuple[date, date]] = {}
    with open(deadlines_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = str(row.get("Conference", "")).strip().upper()
            try:
                year = int(str(row.get("Year", "")).strip())
            except ValueError:
                continue

            submission = parse_deadline_day(row.get("Submission Deadline", ""))
            review = parse_deadline_day(row.get("Review deadline", ""))
            if submission is not None and review is not None:
                windows[(conf, year)] = (submission, review)
    return windows


def date_to_bucket(
    preprint_date: date | None,
    conference: str,
    year: int,
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> int | None:
    if preprint_date is None:
        return None

    window = deadline_windows.get((conference.upper(), year))
    if window is None:
        return None

    submission, review = window
    if submission <= preprint_date <= review:
        return 0
    if submission - timedelta(days=30) <= preprint_date < submission:
        return 1
    if submission - timedelta(days=60) <= preprint_date < submission - timedelta(days=30):
        return 2
    if submission - timedelta(days=90) <= preprint_date < submission - timedelta(days=60):
        return 3
    if submission - timedelta(days=180) <= preprint_date < submission - timedelta(days=90):
        return 4
    if preprint_date < submission - timedelta(days=180):
        return 5
    return None


def title_from_row(row: dict[str, str]) -> str:
    for column in ("Title", "Paper Title", "paper_title", "file_name"):
        value = row.get(column)
        if is_non_null(value):
            return str(value).replace(".pdf", "").strip()
    return ""


def count_oldoutput_source(
    conference: str,
    year: int,
    csv_path: Path,
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> np.ndarray:
    counts = np.zeros(len(BUCKETS), dtype=int)
    if not csv_path.exists():
        return counts

    seen_titles: set[str] = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title_key = normalize_title(title_from_row(row))
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            bucket = date_to_bucket(
                parse_date(row.get("Date", "")),
                conference,
                year,
                deadline_windows,
            )
            if bucket is not None:
                counts[bucket] += 1
    return counts


def infer_conference_year(path: Path) -> tuple[str | None, int | None]:
    text = str(path).upper()
    conference = None
    for candidate in ("ICLR", "ICML", "NEURIPS", "KDD", "ACL", "CVPR", "EMNLP", "COLING", "AAAI"):
        if candidate in text:
            conference = "NeurIPS" if candidate == "NEURIPS" else candidate
            break

    year_match = re.search(r"20\d{2}", text)
    year = int(year_match.group(0)) if year_match else None
    return conference, year


def collect_records(target_pairs: set[tuple[str, int]]) -> list[PaperRecord]:
    records_by_key: dict[tuple[str, int, str], PaperRecord] = {}
    csv_paths = sorted(OUTPUTS_DIR.glob("*.csv")) + sorted(OLD_OUTPUTS_DIR.glob("**/*.csv"))

    for csv_path in csv_paths:
        conference, year = infer_conference_year(csv_path)
        if conference is None or year is None:
            continue
        if (conference.upper(), year) not in target_pairs:
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = title_from_row(row)
                norm = normalize_title(title)
                if not norm:
                    continue

                key = (conference.upper(), year, norm)
                current = records_by_key.get(key)
                parsed_date = parse_date(row.get("Date", ""))
                if current is None or (current.preprint_date is None and parsed_date is not None):
                    records_by_key[key] = PaperRecord(
                        title=title,
                        conference=conference,
                        year=year,
                        preprint_date=parsed_date,
                        source=csv_path,
                    )

    return list(records_by_key.values())


def load_oral_titles(csv_path: Path) -> list[str]:
    titles: list[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        title_column = "Title" if "Title" in fieldnames else "Paper Title"
        for row in reader:
            title = str(row.get(title_column, "")).strip()
            if title:
                titles.append(title)
    return titles


def build_record_indexes(
    records: list[PaperRecord],
) -> tuple[
    dict[tuple[str, int, str], PaperRecord],
    dict[tuple[str, int, str], PaperRecord],
    dict[tuple[str, int], list[PaperRecord]],
]:
    by_norm: dict[tuple[str, int, str], PaperRecord] = {}
    by_compact: dict[tuple[str, int, str], PaperRecord] = {}
    by_conf_year: dict[tuple[str, int], list[PaperRecord]] = {}

    for record in records:
        conf_key = record.conference.upper()
        norm = normalize_title(record.title)
        compact = compact_title(record.title)
        by_norm[(conf_key, record.year, norm)] = record
        by_compact[(conf_key, record.year, compact)] = record
        by_conf_year.setdefault((conf_key, record.year), []).append(record)

    return by_norm, by_compact, by_conf_year


def match_oral_title(
    title: str,
    conference: str,
    year: int,
    by_norm: dict[tuple[str, int, str], PaperRecord],
    by_compact: dict[tuple[str, int, str], PaperRecord],
    by_conf_year: dict[tuple[str, int], list[PaperRecord]],
) -> PaperRecord | None:
    conf_key = conference.upper()
    norm = normalize_title(title)
    compact = compact_title(title)

    exact = by_norm.get((conf_key, year, norm)) or by_compact.get((conf_key, year, compact))
    if exact is not None:
        return exact

    norm_tokens = norm.split()
    if not norm_tokens:
        return None
    token_gate = set(norm_tokens[:3])
    best: tuple[float, PaperRecord] | None = None
    for record in by_conf_year.get((conf_key, year), []):
        candidate_norm = normalize_title(record.title)
        if not token_gate.intersection(candidate_norm.split()[:8]):
            continue
        candidate_compact = compact_title(record.title)
        if compact and candidate_compact:
            shorter, longer = sorted((compact, candidate_compact), key=len)
            if len(shorter) >= 20 and shorter in longer:
                score = len(shorter) / len(longer)
            else:
                score = SequenceMatcher(None, compact, candidate_compact).ratio()
        else:
            score = SequenceMatcher(None, norm, candidate_norm).ratio()

        if score >= 0.92 and (best is None or score > best[0]):
            best = (score, record)

    return best[1] if best is not None else None


def count_oral_title_source(
    conference: str,
    year: int,
    csv_path: Path,
    records: list[PaperRecord],
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> tuple[np.ndarray, int, set[tuple[str, int, str]]]:
    counts = np.zeros(len(BUCKETS), dtype=int)
    by_norm, by_compact, by_conf_year = build_record_indexes(records)
    titles = load_oral_titles(csv_path)
    matched_keys: set[tuple[str, int, str]] = set()

    for title in titles:
        record = match_oral_title(title, conference, year, by_norm, by_compact, by_conf_year)
        if record is None:
            continue

        record_key = (
            record.conference.upper(),
            record.year,
            normalize_title(record.title),
        )
        if record_key in matched_keys:
            continue
        matched_keys.add(record_key)

        bucket = date_to_bucket(record.preprint_date, conference, year, deadline_windows)
        if bucket is not None:
            counts[bucket] += 1

    return counts, len(titles), matched_keys


def record_key(record: PaperRecord) -> tuple[str, int, str]:
    return (record.conference.upper(), record.year, normalize_title(record.title))


def count_remaining_records_as_posters(
    conference: str,
    year: int,
    records: list[PaperRecord],
    oral_record_keys: set[tuple[str, int, str]],
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> tuple[np.ndarray, int]:
    counts = np.zeros(len(BUCKETS), dtype=int)
    remaining_records = [
        record
        for record in records
        if record.conference.upper() == conference.upper()
        and record.year == year
        and record_key(record) not in oral_record_keys
    ]

    for record in remaining_records:
        bucket = date_to_bucket(record.preprint_date, conference, year, deadline_windows)
        if bucket is not None:
            counts[bucket] += 1

    return counts, len(remaining_records)


def ordered_labels(raw_counts: dict[tuple[str, str], dict[int, np.ndarray]]) -> list[tuple[str, str]]:
    preferred = ["ICLR", "ICML", "NeurIPS", "KDD", "ACL", "EMNLP", "COLING", "AAAI"]
    suffix_order = {"O": 0, "O/P": 1, "P": 2, "S": 3}

    return sorted(
        raw_counts,
        key=lambda item: (
            preferred.index(item[0]) if item[0] in preferred else len(preferred),
            suffix_order.get(item[1], 99),
            item[1],
        ),
    )


def write_summary(
    labels: list[tuple[str, str]],
    raw_counts: dict[tuple[str, str], dict[int, np.ndarray]],
    match_summaries: list[dict[str, object]],
) -> None:
    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Year", *BUCKETS, "Total bucketed"])
        for conference, suffix in labels:
            label = f"{conference}-{suffix}"
            for year in YEARS:
                counts = raw_counts[(conference, suffix)].get(year, np.zeros(len(BUCKETS), dtype=int))
                writer.writerow([label, year, *counts.tolist(), int(counts.sum())])

        writer.writerow([])
        writer.writerow(["Matched oral title sources"])
        writer.writerow(
            ["Conference", "Year", "Source", "Titles", "Matched records", "Poster records"]
        )
        for summary in match_summaries:
            writer.writerow(
                [
                    summary["conference"],
                    summary["year"],
                    summary["source"],
                    summary["titles"],
                    summary["matched"],
                    summary["poster_records"],
                ]
            )


def draw_plot(labels: list[tuple[str, str]], raw_counts: dict[tuple[str, str], dict[int, np.ndarray]]) -> None:
    label_text = [f"{conference}-{suffix}" for conference, suffix in labels]
    row_count = len(label_text)
    figure_height = max(10, row_count * 0.65)

    fig, axes = plt.subplots(1, len(YEARS), figsize=(24, figure_height), sharex=True)
    fig.patch.set_facecolor("white")

    for idx, year in enumerate(YEARS):
        ax = axes[idx]
        values = np.vstack(
            [
                raw_counts[(conference, suffix)].get(year, np.zeros(len(BUCKETS), dtype=int))
                for conference, suffix in labels
            ]
        ).astype(float)

        row_sums = values.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized = values / row_sums

        y_positions = np.arange(row_count)
        left = np.zeros(row_count)
        for bucket_index, bucket in enumerate(BUCKETS):
            bars = ax.barh(
                y_positions,
                normalized[:, bucket_index],
                left=left,
                color=COLORS[bucket_index],
                edgecolor="white",
                linewidth=1.2,
                height=0.72,
                label=bucket,
            )
            for bar in bars:
                bar.set_alpha(0.95)
            left += normalized[:, bucket_index]

        ax.set_title(str(year), fontsize=24, weight="bold", pad=15)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(label_text, fontsize=24, fontweight="bold")
        ax.tick_params(axis="x", labelsize=24)
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=24,
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.88))
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    deadline_windows = load_deadline_windows(DEADLINES_CSV)
    target_pairs = {(conference.upper(), year) for conference, year, _suffix, _path in ORAL_TITLE_SOURCES}
    records = collect_records(target_pairs)
    raw_counts: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    match_summaries: list[dict[str, object]] = []

    for conference, year, suffix, csv_path in OLD_PRESENTATION_SOURCES:
        if not csv_path.exists():
            continue

        key = (conference, suffix)
        raw_counts.setdefault(key, {})[year] = count_oldoutput_source(
            conference,
            year,
            csv_path,
            deadline_windows,
        )

    for conference, year, suffix, csv_path in ORAL_TITLE_SOURCES:
        if not csv_path.exists():
            continue

        counts, title_count, oral_record_keys = count_oral_title_source(
            conference,
            year,
            csv_path,
            records,
            deadline_windows,
        )
        key = (conference, suffix)
        raw_counts.setdefault(key, {})[year] = counts

        poster_counts, poster_record_count = count_remaining_records_as_posters(
            conference,
            year,
            records,
            oral_record_keys,
            deadline_windows,
        )
        raw_counts.setdefault((conference, "P"), {})[year] = poster_counts

        match_summaries.append(
            {
                "conference": conference,
                "year": year,
                "source": csv_path.name,
                "titles": title_count,
                "matched": len(oral_record_keys),
                "poster_records": poster_record_count,
            }
        )

    labels = ordered_labels(raw_counts)
    write_summary(labels, raw_counts, match_summaries)
    draw_plot(labels, raw_counts)

    print(f"Saved plot: {OUTPUT_PNG}")
    print(f"Saved summary: {SUMMARY_CSV}")
    for summary in match_summaries:
        print(
            (
                f"{summary['conference']} {summary['year']} {summary['source']}: "
                f"{summary['matched']}/{summary['titles']} titles matched; "
                f"{summary['poster_records']} remaining records shown as posters"
            )
        )


if __name__ == "__main__":
    main()