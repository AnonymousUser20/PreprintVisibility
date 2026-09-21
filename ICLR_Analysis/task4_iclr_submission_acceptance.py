"""Task 4: ICLR submission-level logistic regression for acceptance outcome."""

# pyright: reportMissingModuleSource=false, reportAny=false

from __future__ import annotations

import csv
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast, TypedDict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ICLR_DIR = PROJECT_ROOT / "OldOutputs" / "ICLR"
DEADLINES_CSV = PROJECT_ROOT / "ConfusionMatrix" / "conference_deadlines.csv"
OUTPUT_TABLE = SCRIPT_DIR / "table4_iclr_submission_logistic.csv"
OUTPUT_US_COUNTRY = SCRIPT_DIR / "table4_us_country_contrasts.csv"
OUTPUT_CHINA_COUNTRY = SCRIPT_DIR / "table4_china_country_contrasts.csv"
OUTPUT_DATASET = SCRIPT_DIR / "task4_submission_dataset.csv"
OUTPUT_MISSING = SCRIPT_DIR / "task4_missing_data_report.txt"

COUNTRY_REFERENCE = "Canada"
US_COUNTRY = "United States"
CHINA_COUNTRY = "China"

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

ACCEPTED_MARKERS = ("poster", "oral", "spotlight", "top_5", "top25")
REJECTED_MARKERS = ("withdrawn_rejected", "reject")

BOTTOM_RANK_THRESHOLD = {2023: 465, 2024: 475, 2025: 476}

TIMING_REFERENCE = "cfp_to_review"
TIMING_CONTRAST = "within_30d"
TIMING_LABELS = {
    "no_preprint": "No preprint date",
    "cfp_to_review": "CfP to review window",
    "within_30d": "<30 days before submission",
    "within_60d": "30-60 days before submission",
    "within_90d": "60-90 days before submission",
    "within_180d": "90-180 days before submission",
    "more_than_180d": ">180 days before submission",
}


class SubmissionRecord(TypedDict):
    title: str
    year: int
    accepted: int
    source_file: str
    preprint_visible: int
    preprint_timing: str
    institution_tier: str
    country: str
    rank_first: str
    date_raw: str


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
        for row in csv.DictReader(f):
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


def infer_year_from_path(path: Path) -> int | None:
    match = re.search(r"20(23|24|25)", str(path))
    return int(match.group(0)) if match else None


def classify_outcome(filename: str) -> int | None:
    name = filename.lower()
    if any(marker in name for marker in ACCEPTED_MARKERS):
        return 1
    if any(marker in name for marker in REJECTED_MARKERS):
        return 0
    return None


def primary_country(raw: object) -> str:
    text = str(raw).strip()
    if not is_non_null(text):
        return "Unknown"
    first = text.split(";")[0].strip()
    return first if first else "Unknown"


def institution_tier(rank_raw: object, year: int) -> str:
    text = str(rank_raw).strip()
    if not is_non_null(text):
        return "not_listed"
    try:
        rank = int(float(text))
    except ValueError:
        return "not_listed"
    if rank <= 20:
        return "top"
    if rank >= BOTTOM_RANK_THRESHOLD.get(year, 9999):
        return "bottom"
    return "not_listed"


def preprint_timing_bucket(
    preprint_date: date | None,
    year: int,
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> str:
    if preprint_date is None:
        return "no_preprint"

    window = deadline_windows.get(("ICLR", year))
    if window is None:
        return "no_preprint"

    submission, review = window
    if submission <= preprint_date <= review:
        return "cfp_to_review"
    if submission - timedelta(days=30) <= preprint_date < submission:
        return "within_30d"
    if submission - timedelta(days=60) <= preprint_date < submission - timedelta(days=30):
        return "within_60d"
    if submission - timedelta(days=90) <= preprint_date < submission - timedelta(days=60):
        return "within_90d"
    if submission - timedelta(days=180) <= preprint_date < submission - timedelta(days=90):
        return "within_180d"
    if preprint_date < submission - timedelta(days=180):
        return "more_than_180d"
    return "no_preprint"


def preprint_visible_flag(
    preprint_date: date | None,
    year: int,
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> int:
    """1 if preprint is on/after (CfP - 30d); 0 if earlier than that or missing.

    CfP is the ICLR submission deadline in conference_deadlines.csv.
    """
    if preprint_date is None:
        return 0
    window = deadline_windows.get(("ICLR", year))
    if window is None:
        return 0
    cfp, _review = window
    if preprint_date < cfp - timedelta(days=30):
        return 0
    return 1


def load_submissions(
    iclr_dir: Path,
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> pd.DataFrame:
    records: dict[tuple[int, str], SubmissionRecord] = {}

    for csv_path in sorted(iclr_dir.glob("**/*.csv")):
        outcome = classify_outcome(csv_path.stem)
        year = infer_year_from_path(csv_path)
        if outcome is None or year is None:
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                title = str(row.get("Title", "")).strip()
                if not title:
                    continue
                key = (year, normalize_title(title))
                preprint_date = parse_date(row.get("Date", ""))
                entry: SubmissionRecord = {
                    "title": title,
                    "year": year,
                    "accepted": outcome,
                    "source_file": csv_path.name,
                    "preprint_visible": preprint_visible_flag(
                        preprint_date, year, deadline_windows
                    ),
                    "preprint_timing": preprint_timing_bucket(
                        preprint_date, year, deadline_windows
                    ),
                    "institution_tier": institution_tier(row.get("Rank_first", ""), year),
                    "country": primary_country(row.get("Country", "")),
                    "rank_first": str(row.get("Rank_first", "")).strip(),
                    "date_raw": str(row.get("Date", "")).strip(),
                }

                existing = records.get(key)
                if existing is None:
                    records[key] = entry
                elif outcome == 1 and existing["accepted"] == 0:
                    records[key] = entry
                elif outcome == existing["accepted"] == 1:
                    if _metadata_score(entry) > _metadata_score(existing):
                        records[key] = entry

    rows: list[SubmissionRecord] = list(records.values())
    df = pd.DataFrame(rows)
    df = df.sort_values(["year", "accepted", "title"]).reset_index(drop=True)
    return df


def _metadata_score(item: SubmissionRecord) -> int:
    score = 0
    if item["preprint_visible"]:
        score += 2
    if item["country"] != "Unknown":
        score += 1
    if item["institution_tier"] != "not_listed":
        score += 1
    return score


def collapse_countries(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    counts = df["country"].value_counts()
    keep: set[str] = {str(country) for country in counts.head(top_n).index}
    keep.discard("Unknown")
    keep.add("Unknown")
    out = df.copy()
    out["country_group"] = out["country"].where(out["country"].isin(list(keep)), other="Other")
    return out


def missing_data_report(df: pd.DataFrame) -> str:
    lines = [
        "Task 4 missing-data report (ICLR submissions)",
        f"Total unique submissions: {len(df)}",
        f"Accepted: {int(df['accepted'].sum())} | Rejected: {int((1 - df['accepted']).sum())}",
        "",
        "By year:",
    ]
    for year, grp in df.groupby("year"):
        accepted_count = int(grp["accepted"].sum())
        rejected_count = int((1 - grp["accepted"]).sum())
        lines.append(
            f"  {year}: n={len(grp)} | accepted={accepted_count} | rejected={rejected_count}"
        )

    preprint_visible_count = int(df["preprint_visible"].sum())
    preprint_visible_pct = 100 * float(df["preprint_visible"].mean())
    no_preprint_count = int((df["preprint_timing"] == "no_preprint").sum())
    no_preprint_pct = 100 * float((df["preprint_timing"] == "no_preprint").mean())
    not_listed_count = int((df["institution_tier"] == "not_listed").sum())
    not_listed_pct = 100 * float((df["institution_tier"] == "not_listed").mean())
    unknown_country_count = int((df["country"] == "Unknown").sum())
    unknown_country_pct = 100 * float((df["country"] == "Unknown").mean())

    lines.extend(
        [
            "",
            "Predictor completeness:",
            f"  preprint_visible=1: {preprint_visible_count} ({preprint_visible_pct:.1f}%)",
            f"  preprint_timing=no_preprint: {no_preprint_count} ({no_preprint_pct:.1f}%)",
            f"  institution_tier=not_listed: {not_listed_count} ({not_listed_pct:.1f}%)",
            f"  country=Unknown: {unknown_country_count} ({unknown_country_pct:.1f}%)",
            "",
            "Preprint timing distribution:",
        ]
    )
    for label, count in df["preprint_timing"].value_counts().items():
        lines.append(f"  {label}: {count}")

    lines.extend(["", "Institution tier distribution:"])
    for label, count in df["institution_tier"].value_counts().items():
        lines.append(f"  {label}: {count}")

    return "\n".join(lines)


def fit_logit(df: pd.DataFrame) -> sm.BinaryResultsWrapper:
    model_df = df.copy()
    model_df["year"] = model_df["year"].astype("category")
    model_df["institution_tier"] = pd.Categorical(
        model_df["institution_tier"],
        categories=["not_listed", "top", "bottom"],
    )
    model_df["preprint_timing"] = model_df["preprint_timing"].astype("category")

    formula = (
        "accepted ~ preprint_visible + "
        f"C(preprint_timing, Treatment(reference='{TIMING_REFERENCE}')) + "
        "C(institution_tier) + C(country_group) + C(year)"
    )
    return smf.logit(formula, data=model_df).fit(disp=False, maxiter=200)


def or_ci_p(result: sm.BinaryResultsWrapper, param: str) -> tuple[float, float, float, float]:
    if param not in result.params.index:
        return float("nan"), float("nan"), float("nan"), float("nan")
    coef = float(result.params[param])
    se = float(result.bse[param])
    p = float(result.pvalues[param])
    or_val = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))
    return or_val, ci_low, ci_high, p


def country_group_param(level: str) -> str:
    return f"C(country_group)[T.{level}]"


def wald_contrast_or_ci_p(
    result: sm.BinaryResultsWrapper,
    contrast: np.ndarray,
) -> tuple[float, float, float, float]:
    effect = float(contrast @ result.params.to_numpy())
    wald = result.wald_test(contrast, scalar=True)
    p = float(wald.pvalue)
    class _HasCovParams(Protocol):
        def cov_params(self) -> object: ...

    cov_params = cast(_HasCovParams, cast(object, result)).cov_params()
    cov = np.asarray(cov_params)  # cov_params() often returns a DataFrame
    se = float(np.sqrt(contrast @ cov @ contrast.T))
    or_val = float(np.exp(effect))
    ci_low = float(np.exp(effect - 1.96 * se))
    ci_high = float(np.exp(effect + 1.96 * se))
    return or_val, ci_low, ci_high, p


def focal_vs_country_contrast(
    result: sm.BinaryResultsWrapper,
    focal: str,
    other: str,
) -> tuple[float, float, float, float]:
    contrast = np.zeros(len(result.params))
    if focal != COUNTRY_REFERENCE:
        focal_param = country_group_param(focal)
        if focal_param not in result.params.index:
            return float("nan"), float("nan"), float("nan"), float("nan")
        contrast[list(result.params.index).index(focal_param)] = 1.0
    if other != COUNTRY_REFERENCE:
        other_param = country_group_param(other)
        if other_param not in result.params.index:
            return float("nan"), float("nan"), float("nan"), float("nan")
        contrast[list(result.params.index).index(other_param)] = -1.0
    return wald_contrast_or_ci_p(result, contrast)


def other_country_examples(df: pd.DataFrame, top_n: int = 8) -> str:
    mask = df["country_group"] == "Other"
    if not mask.any():
        return ""
    counts = df.loc[mask, "country"].value_counts()
    parts = [f"{country} ({count})" for country, count in counts.head(top_n).items()]
    remaining = int(counts.shape[0] - min(top_n, counts.shape[0]))
    suffix = f"; +{remaining} more countries" if remaining > 0 else ""
    return ", ".join(parts) + suffix


def build_country_contrast_table(
    result: sm.BinaryResultsWrapper,
    df: pd.DataFrame,
    focal_country: str,
) -> pd.DataFrame:
    groups = sorted(df["country_group"].unique())
    other_examples = other_country_examples(df)
    rows: list[dict[str, str | int]] = []

    for group in groups:
        if group == focal_country:
            continue
        n_group = int((df["country_group"] == group).sum())
        or_val, ci_lo, ci_hi, p = focal_vs_country_contrast(result, focal_country, group)
        direction = "higher" if or_val >= 1 else "lower"
        pct = abs(or_val - 1.0) * 100
        notes = ""
        if group == "Other" and other_examples:
            notes = f"Includes: {other_examples}"
        rows.append(
            {
                "Comparison": f"{focal_country} vs {group}",
                "Reference group": group,
                "N (reference group)": n_group,
                "Odds ratio": f"{or_val:.2f}" if not np.isnan(or_val) else "—",
                "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(or_val) else "—",
                "p-value": format_pvalue(p),
                "Interpretation": (
                    f"{focal_country} submissions have {pct:.0f}% {direction} odds of acceptance than {group} submissions (OR={or_val:.2f}), adjusted for preprint, institution tier, and year."
                    if not np.isnan(or_val)
                    else f"Could not estimate {focal_country} vs {group}."
                ),
                "Notes": notes,
            }
        )

    return pd.DataFrame(rows)


def save_and_print_country_contrasts(
    result: sm.BinaryResultsWrapper,
    df: pd.DataFrame,
    focal_country: str,
    output_path: Path,
) -> None:
    table = build_country_contrast_table(result, df, focal_country)
    table.to_csv(output_path, index=False, encoding="utf-8")
    focal_n = int((df["country_group"] == focal_country).sum())
    print(f"\nSaved {focal_country} country contrasts: {output_path}")
    print(
        f"\n{focal_country} vs other country groups (model reference: {COUNTRY_REFERENCE}; {focal_country} n={focal_n})"
    )
    for _, row in table.iterrows():
        comparison = str(row["Comparison"])
        odds_ratio = str(row["Odds ratio"])
        ci = str(row["95% CI"])
        p_value = str(row["p-value"])
        n_ref = str(row["N (reference group)"])
        print(f"{comparison:32} OR={odds_ratio:>6}  95% CI={ci:>16}  p={p_value}  n={n_ref}")
        notes = str(row["Notes"])
        if notes:
            print(f"  -> {notes}")


def wald_omnibus_p(result: sm.BinaryResultsWrapper, param_prefix: str) -> float:
    names = [name for name in result.params.index if name.startswith(param_prefix)]
    if not names:
        return float("nan")
    r_matrix = np.zeros((len(names), len(result.params)))
    for i, name in enumerate(names):
        r_matrix[i, list(result.params.index).index(name)] = 1.0
    wald = result.wald_test(r_matrix, scalar=True)
    return float(wald.pvalue)


def format_pvalue(p: float) -> str:
    if np.isnan(p):
        return "—"
    text = f"{p:.1e}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def pct_change(or_val: float) -> str:
    return f"{abs(or_val - 1.0) * 100:.0f}%"


def simple_interpretation(
    predictor: str,
    or_val: float,
    *,
    contrast: str | None = None,
    omnibus: bool = False,
) -> str:
    if np.isnan(or_val):
        return f"Could not estimate a clear pattern for {predictor.lower()}."

    if predictor == "Preprint visible":
        return (
            f"Papers whose preprint appears on or after 30 days before the CfP "
            f"(submission deadline) had about {or_val:.1f}x the odds of acceptance "
            "compared with papers with no preprint or an earlier preprint, after "
            "adjusting for timing, institution tier, country, and year."
        )

    if predictor == "Preprint timing":
        direction = "lower" if or_val < 1 else "higher"
        pct = abs(or_val - 1.0) * 100
        ref_label = TIMING_LABELS.get(TIMING_REFERENCE, TIMING_REFERENCE)
        return (
            f"Compared with the reference bucket ({ref_label}), posting within 30 days "
            f"of submission is associated with {pct:.0f}% {direction} odds of acceptance "
            f"(OR={or_val:.2f}), after adjusting for preprint visibility and other covariates."
        )

    if predictor == "Institution tier":
        return (
            f"Submissions whose first author comes from a top-20 CS-ranked institute "
            f"had about {pct_change(or_val)} higher odds of acceptance than submissions "
            "from unlisted or mid-ranked institutes."
        )

    if predictor == "Country":
        label = contrast or "the reference country group"
        direction = "higher" if or_val >= 1 else "lower"
        return (
            "Author country is linked to acceptance differences in the model. "
            f"Overall, country groups differ significantly; for example, {label} shows "
            f"{direction} odds than the reference group (OR={or_val:.2f})."
        )

    if predictor == "Year":
        return (
            f"Later ICLR years show much lower acceptance odds than 2023 "
            f"(2025 OR={or_val:.2f}, about {100 - or_val * 100:.0f}% lower odds), "
            "reflecting stronger competition or changing acceptance rates over time."
        )

    if omnibus:
        return f"{predictor} is significantly associated with acceptance in the adjusted model."
    return (
        f"{predictor} is associated with {pct_change(or_val)} "
        f"{'higher' if or_val >= 1 else 'lower'} odds of acceptance."
    )


def timing_contrast_param(result: sm.BinaryResultsWrapper) -> str:
    marker = f"[T.{TIMING_CONTRAST}]"
    matches = [name for name in result.params.index if "preprint_timing" in name and marker in name]
    return matches[0] if matches else ""


def timing_omnibus_prefix(result: sm.BinaryResultsWrapper) -> str:
    matches = [
        name
        for name in result.params.index
        if name.startswith("C(preprint_timing") and "[T." in name
    ]
    if not matches:
        return "C(preprint_timing)"
    return matches[0].split("[T.", 1)[0]


def build_table4(result: sm.BinaryResultsWrapper) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    or_val, ci_lo, ci_hi, p = or_ci_p(result, "preprint_visible")
    rows.append(
        {
            "Predictor": "Preprint visible",
            "Odds ratio": f"{or_val:.2f}" if not np.isnan(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(or_val) else "—",
            "p-value": format_pvalue(p),
            "Interpretation": simple_interpretation("Preprint visible", or_val),
        }
    )

    timing_param = timing_contrast_param(result)
    or_val, ci_lo, ci_hi, _p_level = or_ci_p(result, timing_param)
    p_omni = wald_omnibus_p(result, timing_omnibus_prefix(result))
    rows.append(
        {
            "Predictor": "Preprint timing",
            "Odds ratio": f"{or_val:.2f}" if not np.isnan(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(or_val) else "—",
            "p-value": format_pvalue(p_omni),
            "Interpretation": simple_interpretation("Preprint timing", or_val),
        }
    )

    tier_param = "C(institution_tier)[T.top]"
    or_val, ci_lo, ci_hi, p = or_ci_p(result, tier_param)
    p_omni = wald_omnibus_p(result, "C(institution_tier)")
    rows.append(
        {
            "Predictor": "Institution tier",
            "Odds ratio": f"{or_val:.2f}" if not np.isnan(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(or_val) else "—",
            "p-value": format_pvalue(p_omni),
            "Interpretation": simple_interpretation("Institution tier", or_val),
        }
    )

    country_param = "C(country_group)[T.United States]"
    if country_param not in result.params.index:
        country_candidates = [
            n for n in result.params.index if n.startswith("C(country_group)[T.")
        ]
        country_param = country_candidates[0] if country_candidates else ""
    or_val, ci_lo, ci_hi, p = or_ci_p(result, country_param)
    p_omni = wald_omnibus_p(result, "C(country_group)")
    country_label = country_param.replace("C(country_group)[T.", "").rstrip("]") if country_param else "reference"
    rows.append(
        {
            "Predictor": "Country",
            "Odds ratio": f"{or_val:.2f}" if not np.isnan(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(or_val) else "—",
            "p-value": format_pvalue(p_omni),
            "Interpretation": simple_interpretation(
                "Country", or_val, contrast=country_label
            ),
        }
    )

    year_param = "C(year)[T.2025]"
    or_val, ci_lo, ci_hi, p = or_ci_p(result, year_param)
    p_omni = wald_omnibus_p(result, "C(year)")
    rows.append(
        {
            "Predictor": "Year",
            "Odds ratio": f"{or_val:.2f}" if not np.isnan(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(or_val) else "—",
            "p-value": format_pvalue(p_omni),
            "Interpretation": simple_interpretation("Year", or_val),
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    if not ICLR_DIR.exists():
        raise FileNotFoundError(f"ICLR data folder not found: {ICLR_DIR}")

    deadline_windows = load_deadline_windows(DEADLINES_CSV)
    df = load_submissions(ICLR_DIR, deadline_windows)
    df = collapse_countries(df, top_n=8)

    missing_text = missing_data_report(df)
    _ = OUTPUT_MISSING.write_text(missing_text, encoding="utf-8")
    print(missing_text)
    print()

    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
    print(f"Saved submission dataset: {OUTPUT_DATASET}")

    result = fit_logit(df)
    print(result.summary())

    table4 = build_table4(result)
    table4.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8")
    print(f"\nSaved Table 4: {OUTPUT_TABLE}")
    print("\nTable 4: ICLR submission-level logistic regression for acceptance outcome")
    for _, row in table4.iterrows():
        predictor = str(row["Predictor"])
        odds_ratio = str(row["Odds ratio"])
        ci = str(row["95% CI"])
        p_value = str(row["p-value"])
        interpretation = str(row["Interpretation"])
        print(f"{predictor:20} OR={odds_ratio:>6}  95% CI={ci:>16}  p={p_value}")
        print(f"  -> {interpretation}")

    save_and_print_country_contrasts(result, df, US_COUNTRY, OUTPUT_US_COUNTRY)
    save_and_print_country_contrasts(result, df, CHINA_COUNTRY, OUTPUT_CHINA_COUNTRY)


if __name__ == "__main__":
    main()
