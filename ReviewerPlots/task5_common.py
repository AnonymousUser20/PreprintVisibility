"""Shared helpers for Task 5 OLS models (rating / confidence)."""

# pyright: reportMissingModuleSource=false, reportAny=false

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "paper_review_scores.csv"
OUTPUT_TABLE5 = SCRIPT_DIR / "table5_reviewer_rating_confidence.csv"
OUTPUT_RATING_ROWS = SCRIPT_DIR / "table5_rating_model_rows.csv"
OUTPUT_CONFIDENCE_ROWS = SCRIPT_DIR / "table5_confidence_model_rows.csv"
OUTPUT_DATASET = SCRIPT_DIR / "task5_model_dataset.csv"
OUTPUT_MISSING = SCRIPT_DIR / "task5_missing_data_report.txt"

NULL_LIKE = {"", "nan", "none", "null", "na", "not found"}
BOTTOM_RANK_THRESHOLD = {2023: 465, 2024: 475, 2025: 476}
COUNTRY_REFERENCE = "Canada"
COUNTRY_EXAMPLE = "United States"
TIER_REFERENCE = "not_listed"
TIER_EXAMPLE = "top"
VENUE_REFERENCE = "ICLR"
YEAR_REFERENCE = 2023


def is_non_null(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in NULL_LIKE


def primary_country(raw: object) -> str:
    text = str(raw).strip()
    if not is_non_null(text):
        return "Unknown"
    first = text.split(";")[0].strip()
    return first if first else "Unknown"


def institution_tier(rank_raw: object, year: object) -> str:
    text = str(rank_raw).strip()
    if not is_non_null(text):
        return "not_listed"
    try:
        rank = int(float(text))
        year_int = int(float(str(year)))
    except (TypeError, ValueError):
        return "not_listed"
    if rank <= 20:
        return "top"
    if rank >= BOTTOM_RANK_THRESHOLD.get(year_int, 9999):
        return "bottom"
    return "not_listed"


def collapse_countries(series: pd.Series, top_n: int = 8) -> pd.Series:
    counts = series.value_counts()
    keep: set[str] = {str(country) for country in counts.head(top_n).index}
    keep.discard("Unknown")
    keep.add("Unknown")
    return series.where(series.isin(list(keep)), other="Other")


def standardize_within_venue_year(df: pd.DataFrame, column: str) -> pd.Series:
    def _z(group: pd.Series) -> pd.Series:
        values = pd.to_numeric(group, errors="coerce")
        std = float(values.std(ddof=0))
        if std == 0.0 or np.isnan(std):
            return pd.Series(np.nan, index=group.index)
        return (values - float(values.mean())) / std

    return df.groupby(["venue", "year"], group_keys=False)[column].transform(_z)


def load_and_prepare(csv_path: Path = INPUT_CSV) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    required = {
        "Conference",
        "Year",
        "rating",
        "confidence",
        "Group",
        "Country",
        "Rank",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    df = pd.DataFrame(
        {
            "title": raw["Paper Title"].astype(str),
            "venue": raw["Conference"].astype(str).str.strip(),
            "year": pd.to_numeric(raw["Year"], errors="coerce").astype("Int64"),
            "rating": pd.to_numeric(raw["rating"], errors="coerce"),
            "confidence": pd.to_numeric(raw["confidence"], errors="coerce"),
            "group": raw["Group"].replace({"Non-Preprint": "No Preprint"}),
            "country": raw["Country"].map(primary_country),
            "rank": pd.to_numeric(raw["Rank"], errors="coerce"),
        }
    )
    df = df[df["venue"].isin(["ICLR", "ICML", "NeurIPS"])].copy()
    df = df[df["year"].isin([2023, 2024, 2025])].copy()
    df["preprint_visible"] = (df["group"] == "Preprint").astype(int)
    df["institution_tier"] = [
        institution_tier(rank, year) for rank, year in zip(df["rank"], df["year"])
    ]
    df["country_group"] = collapse_countries(df["country"], top_n=8)
    df["rating_z"] = standardize_within_venue_year(df, "rating")
    df["confidence_z"] = standardize_within_venue_year(df, "confidence")
    return df.reset_index(drop=True)


def missing_data_report(df: pd.DataFrame) -> str:
    lines = [
        "Task 5 missing-data report (reviewer scores)",
        f"Total papers: {len(df)}",
        "",
        "By venue:",
    ]
    for venue, grp in df.groupby("venue"):
        conf_n = int(grp["confidence"].notna().sum())
        lines.append(
            f"  {venue}: n={len(grp)} | rating non-null={int(grp['rating'].notna().sum())} "
            f"| confidence non-null={conf_n}"
        )

    lines.extend(
        [
            "",
            "Predictor completeness:",
            f"  preprint_visible=1: {int(df['preprint_visible'].sum())} "
            f"({100 * float(df['preprint_visible'].mean()):.1f}%)",
            f"  institution_tier=not_listed: {int((df['institution_tier'] == 'not_listed').sum())} "
            f"({100 * float((df['institution_tier'] == 'not_listed').mean()):.1f}%)",
            f"  country=Unknown: {int((df['country'] == 'Unknown').sum())} "
            f"({100 * float((df['country'] == 'Unknown').mean()):.1f}%)",
            "",
            "Institution tier distribution:",
        ]
    )
    for label, count in df["institution_tier"].value_counts().items():
        lines.append(f"  {label}: {count}")

    lines.extend(["", "Country group distribution:"])
    for label, count in df["country_group"].value_counts().items():
        lines.append(f"  {label}: {count}")

    return "\n".join(lines)


def fit_ols(df: pd.DataFrame, outcome: str) -> sm.RegressionResultsWrapper:
    model_df = df.dropna(subset=[outcome]).copy()
    model_df["year"] = model_df["year"].astype(int).astype("category")

    # Drop venues with no outcome variation (e.g. ICML has no confidence scores).
    venue_counts = model_df.groupby("venue", observed=False)[outcome].count()
    usable_venues = [str(v) for v, n in venue_counts.items() if int(n) > 0]
    preferred = [v for v in ["ICLR", "ICML", "NeurIPS"] if v in usable_venues]
    model_df = model_df[model_df["venue"].isin(preferred)].copy()
    model_df["venue"] = pd.Categorical(model_df["venue"], categories=preferred)

    model_df["institution_tier"] = pd.Categorical(
        model_df["institution_tier"],
        categories=["not_listed", "top", "bottom"],
    )
    model_df["country_group"] = model_df["country_group"].astype("category")

    formula = (
        f"{outcome} ~ preprint_visible + "
        "C(institution_tier) + C(country_group) + C(venue) + C(year)"
    )
    return smf.ols(formula, data=model_df).fit()


def format_pvalue(p: float) -> str:
    if np.isnan(p):
        return "—"
    text = f"{p:.1e}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def coef_ci_p(
    result: sm.RegressionResultsWrapper, param: str
) -> tuple[float, float, float, float]:
    if param not in result.params.index:
        return float("nan"), float("nan"), float("nan"), float("nan")
    coef = float(result.params[param])
    se = float(result.bse[param])
    p = float(result.pvalues[param])
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    return coef, ci_low, ci_high, p


def wald_omnibus_p(result: sm.RegressionResultsWrapper, param_prefix: str) -> float:
    names = [name for name in result.params.index if name.startswith(param_prefix)]
    if not names:
        return float("nan")
    r_matrix = np.zeros((len(names), len(result.params)))
    for i, name in enumerate(names):
        r_matrix[i, list(result.params.index).index(name)] = 1.0
    wald = result.wald_test(r_matrix, scalar=True)
    return float(wald.pvalue)


def build_outcome_rows(
    result: sm.RegressionResultsWrapper,
    outcome_label: str,
    n: int,
) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []

    est, lo, hi, p = coef_ci_p(result, "preprint_visible")
    rows.append(
        {
            "Outcome": outcome_label,
            "Predictor": "Preprint visible",
            "Estimate": f"{est:.3f}" if not np.isnan(est) else "—",
            "95% CI": f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(est) else "—",
            "p-value": format_pvalue(p),
            "N": n,
        }
    )

    est, lo, hi, _p_level = coef_ci_p(result, f"C(institution_tier)[T.{TIER_EXAMPLE}]")
    p_omni = wald_omnibus_p(result, "C(institution_tier)")
    rows.append(
        {
            "Outcome": outcome_label,
            "Predictor": "Institution tier",
            "Estimate": f"{est:.3f}" if not np.isnan(est) else "—",
            "95% CI": f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(est) else "—",
            "p-value": format_pvalue(p_omni),
            "N": n,
        }
    )

    country_param = f"C(country_group)[T.{COUNTRY_EXAMPLE}]"
    if country_param not in result.params.index:
        candidates = [
            name for name in result.params.index if name.startswith("C(country_group)[T.")
        ]
        country_param = candidates[0] if candidates else ""
    est, lo, hi, _p_level = coef_ci_p(result, country_param)
    p_omni = wald_omnibus_p(result, "C(country_group)")
    rows.append(
        {
            "Outcome": outcome_label,
            "Predictor": "Country",
            "Estimate": f"{est:.3f}" if not np.isnan(est) else "—",
            "95% CI": f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(est) else "—",
            "p-value": format_pvalue(p_omni),
            "N": n,
        }
    )

    return pd.DataFrame(rows)


def merge_table5(
    rating_rows: pd.DataFrame | None = None,
    confidence_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if rating_rows is None and OUTPUT_RATING_ROWS.exists():
        rating_rows = pd.read_csv(OUTPUT_RATING_ROWS, dtype={"p-value": str})
    if confidence_rows is None and OUTPUT_CONFIDENCE_ROWS.exists():
        confidence_rows = pd.read_csv(OUTPUT_CONFIDENCE_ROWS, dtype={"p-value": str})
    if rating_rows is not None and not rating_rows.empty:
        frames.append(rating_rows)
    if confidence_rows is not None and not confidence_rows.empty:
        frames.append(confidence_rows)
    if not frames:
        return pd.DataFrame(
            columns=["Outcome", "Predictor", "Estimate", "95% CI", "p-value", "N"]
        )
    return pd.concat(frames, ignore_index=True)


def print_table_rows(table: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    for _, row in table.iterrows():
        print(
            f"{str(row['Outcome']):12} | {str(row['Predictor']):18} | "
            f"est={str(row['Estimate']):>8} | CI={str(row['95% CI']):>18} | "
            f"p={str(row['p-value']):>10} | N={row['N']}"
        )
