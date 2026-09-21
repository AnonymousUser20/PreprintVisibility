"""Task 7: Robustness of institutional-tier analyses under alternative CSRankings thresholds."""

# pyright: reportMissingModuleSource=false, reportAny=false

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OLD_OUTPUTS = PROJECT_ROOT / "OldOutputs"
OUTPUTS = PROJECT_ROOT / "Outputs"
DEADLINES_CSV = PROJECT_ROOT / "ConfusionMatrix" / "conference_deadlines.csv"
LOOKUP_RANK_DIR = PROJECT_ROOT / "lookup_rank"

OUTPUT_TABLE = SCRIPT_DIR / "table7_institutional_tier_robustness.csv"
OUTPUT_DATASET = SCRIPT_DIR / "task7_model_dataset.csv"
OUTPUT_REPORT = SCRIPT_DIR / "task7_missing_data_report.txt"

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
REJECTED_FILE_MARKERS = ("reject", "withdrawn", "desk_rejected", "retracted")
SKIP_DIR_NAMES = {"institute_batches"}
SKIP_FILE_NAMES = {"rank_fill_summary.csv"}


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


def is_rejected_source(path: Path) -> bool:
    stem = path.stem.lower()
    return any(marker in stem for marker in REJECTED_FILE_MARKERS)


def load_submission_deadlines(path: Path) -> dict[tuple[str, int], date]:
    deadlines: dict[tuple[str, int], date] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            conf = TARGET_VENUES_UPPER.get(str(row.get("Conference", "")).strip().upper())
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


def load_max_ranks() -> dict[int, int]:
    max_ranks: dict[int, int] = {}
    for year in TARGET_YEARS:
        path = LOOKUP_RANK_DIR / f"{year}_cs_ranks.csv"
        if not path.exists():
            continue
        ranks = pd.read_csv(path)["rank"]
        max_ranks[year] = int(pd.to_numeric(ranks, errors="coerce").max())
    # Fallbacks aligned with project CSRankings lists if lookup missing.
    max_ranks.setdefault(2023, 486)
    max_ranks.setdefault(2024, 495)
    max_ranks.setdefault(2025, 497)
    return max_ranks


def bottom_threshold(year: int, k: int, max_ranks: dict[int, int]) -> int:
    return int(max_ranks[year]) - k + 1


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


def extract_rank(row: dict[str, str]) -> float | None:
    for key in ("Rank_first", "rank"):
        text = str(row.get(key, "")).strip()
        if not is_non_null(text):
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return None


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
            if venue is None or year is None:
                continue
            if venue not in TARGET_VENUES or year not in TARGET_YEARS:
                continue
            paths.append(csv_path)
    return paths


def load_papers(
    deadlines: dict[tuple[str, int], date],
) -> pd.DataFrame:
    best: dict[tuple[str, int, str], dict[str, object]] = {}

    for csv_path in iter_candidate_csvs():
        venue, year = infer_venue_year_from_path(csv_path)
        if venue is None or year is None:
            continue
        deadline = deadlines.get((venue, year))
        if deadline is None:
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
                    rank = extract_rank(row)
                    if rank is None:
                        continue

                    preprint_date = parse_date(extract_date_raw(row))
                    # Outcome requires a known preprint date relative to submission.
                    # Papers with no date are treated as not pre-submission-visible.
                    pre_submission = int(
                        preprint_date is not None and preprint_date < deadline
                    )

                    key = (venue, year, normalize_title(title))
                    entry = {
                        "venue": venue,
                        "year": year,
                        "title": title,
                        "rank": rank,
                        "preprint_date": (
                            preprint_date.isoformat() if preprint_date else ""
                        ),
                        "submission_deadline": deadline.isoformat(),
                        "pre_submission_visible": pre_submission,
                        "has_preprint_date": int(preprint_date is not None),
                        "source_file": str(csv_path.relative_to(PROJECT_ROOT)),
                    }
                    existing = best.get(key)
                    if existing is None:
                        best[key] = entry
                    else:
                        # Prefer better (lower) rank if duplicate titles appear.
                        if float(rank) < float(existing["rank"]):
                            best[key] = entry
        except (OSError, UnicodeDecodeError, csv.Error):
            continue

    df = pd.DataFrame(list(best.values()))
    if df.empty:
        return df
    return df.sort_values(["venue", "year", "title"]).reset_index(drop=True)


def format_pvalue(p: float) -> str:
    if np.isnan(p):
        return "-"
    text = f"{p:.1e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def fit_tier_logit(
    df: pd.DataFrame,
    *,
    treatment: str,
    reference: str,
) -> tuple[float, float, float, float, int]:
    model_df = df.copy()
    model_df["tier"] = pd.Categorical(
        model_df["tier"],
        categories=[reference, treatment],
    )
    model_df["venue"] = model_df["venue"].astype("category")
    model_df["year"] = model_df["year"].astype("category")

    result = smf.logit(
        "pre_submission_visible ~ C(tier, Treatment(reference='%s')) + C(venue) + C(year)"
        % reference,
        data=model_df,
    ).fit(disp=False, maxiter=200)

    param = f"C(tier, Treatment(reference='{reference}'))[T.{treatment}]"
    coef = float(result.params[param])
    se = float(result.bse[param])
    p = float(result.pvalues[param])
    or_val = float(np.exp(coef))
    ci_lo = float(np.exp(coef - 1.96 * se))
    ci_hi = float(np.exp(coef + 1.96 * se))
    return or_val, ci_lo, ci_hi, p, int(result.nobs)


def analyze_top_vs_bottom(
    df: pd.DataFrame,
    k: int,
    max_ranks: dict[int, int],
) -> tuple[float, float, float, float, int, str]:
    rows: list[pd.DataFrame] = []
    for year in TARGET_YEARS:
        bot = bottom_threshold(year, k, max_ranks)
        year_df = df[df["year"] == year].copy()
        year_df["tier"] = np.where(
            year_df["rank"] <= k,
            "top",
            np.where(year_df["rank"] >= bot, "bottom", None),
        )
        year_df = year_df[year_df["tier"].isin(["top", "bottom"])]
        rows.append(year_df)

    subset = pd.concat(rows, ignore_index=True)
    n_top = int((subset["tier"] == "top").sum())
    n_bottom = int((subset["tier"] == "bottom").sum())
    note = f"n_top={n_top}, n_bottom={n_bottom}"
    or_val, ci_lo, ci_hi, p, n = fit_tier_logit(
        subset, treatment="top", reference="bottom"
    )
    return or_val, ci_lo, ci_hi, p, n, note


def analyze_top_vs_nontop(
    df: pd.DataFrame,
    k: int,
) -> tuple[float, float, float, float, int, str]:
    subset = df.copy()
    subset["tier"] = np.where(subset["rank"] <= k, "top", "non_top")
    n_top = int((subset["tier"] == "top").sum())
    n_non = int((subset["tier"] == "non_top").sum())
    note = f"n_top={n_top}, n_non_top={n_non}"
    or_val, ci_lo, ci_hi, p, n = fit_tier_logit(
        subset, treatment="top", reference="non_top"
    )
    return or_val, ci_lo, ci_hi, p, n, note


def build_table7(df: pd.DataFrame, max_ranks: dict[int, int]) -> pd.DataFrame:
    specs = [
        ("Top-10 vs Bottom-10", "top_bottom", 10),
        ("Top-20 vs Bottom-20", "top_bottom", 20),
        ("Top-50 vs Non-top-50", "top_nontop", 50),
    ]
    rows: list[dict[str, str | int]] = []
    for label, kind, k in specs:
        if kind == "top_bottom":
            or_val, ci_lo, ci_hi, p, n, note = analyze_top_vs_bottom(df, k, max_ranks)
        else:
            or_val, ci_lo, ci_hi, p, n, note = analyze_top_vs_nontop(df, k)

        rows.append(
            {
                "Tier definition": label,
                "Outcome": "Pre-submission preprint visibility",
                "Odds ratio": f"{or_val:.2f}",
                "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                "p-value": format_pvalue(p),
                "N": n,
                "Notes": note,
            }
        )
    return pd.DataFrame(rows)


def missing_data_report(df: pd.DataFrame, max_ranks: dict[int, int]) -> str:
    lines = [
        "Task 7 missing-data / coverage report",
        "Outcome: pre_submission_visible = 1 if preprint_date < submission_deadline",
        "Sample: accepted ICLR/ICML/NeurIPS papers with Rank_first (rejects excluded)",
        "",
        f"Papers in analysis dataset: {len(df)}",
        f"Pre-submission visible: {int(df['pre_submission_visible'].sum())} "
        f"({100 * float(df['pre_submission_visible'].mean()):.1f}%)",
        f"Has preprint date: {int(df['has_preprint_date'].sum())} "
        f"({100 * float(df['has_preprint_date'].mean()):.1f}%)",
        "",
        "CSRankings max ranks used for bottom-k thresholds:",
    ]
    for year in TARGET_YEARS:
        lines.append(
            f"  {year}: max={max_ranks[year]} | "
            f"bottom-10>={bottom_threshold(year, 10, max_ranks)} | "
            f"bottom-20>={bottom_threshold(year, 20, max_ranks)}"
        )

    lines.extend(["", "By venue-year:"])
    for (venue, year), grp in df.groupby(["venue", "year"]):
        lines.append(
            f"  {venue} {year}: n={len(grp)} | "
            f"pre-sub visible={int(grp['pre_submission_visible'].sum())}"
        )
    return "\n".join(lines)


def main() -> None:
    deadlines = load_submission_deadlines(DEADLINES_CSV)
    max_ranks = load_max_ranks()
    df = load_papers(deadlines)

    report = missing_data_report(df, max_ranks)
    _ = OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(report)

    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
    print(f"\nSaved model dataset: {OUTPUT_DATASET}")

    table7 = build_table7(df, max_ranks)
    table7.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8")
    print(f"\nSaved Table 7: {OUTPUT_TABLE}")
    print("\nTable 7: Robustness of institutional-tier analyses under alternative CSRankings thresholds")
    for _, row in table7.iterrows():
        print(
            f"{str(row['Tier definition']):24}  OR={str(row['Odds ratio']):>6}  "
            f"CI={str(row['95% CI']):>16}  p={str(row['p-value']):>10}  "
            f"N={row['N']}  ({row['Notes']})"
        )


if __name__ == "__main__":
    main()
