"""Task 5b: ICLR-only OLS for standardized rating and confidence.

Uses task5b_iclr_sample.csv. Does not overwrite original Task 5 tables.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import task5_common as t5

SAMPLE_CSV = SCRIPT_DIR / "task5b_iclr_sample.csv"
OLDOUTPUTS_ICLR = PROJECT_ROOT / "OldOutputs" / "ICLR"
OUTPUT_TABLE = SCRIPT_DIR / "table5b_iclr_rating_confidence.csv"
OUTPUT_DATASET = SCRIPT_DIR / "task5b_iclr_model_dataset.csv"


def normalize_title(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_rank_lookup() -> dict[tuple[int, str], tuple[str, str]]:
    lookup: dict[tuple[int, str], tuple[str, str]] = {}
    if not OLDOUTPUTS_ICLR.exists():
        return lookup
    for csv_path in OLDOUTPUTS_ICLR.rglob("*.csv"):
        match = re.search(r"20(23|24|25)", str(csv_path))
        if not match:
            continue
        year = int(match.group(0))
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                title = str(row.get("Title", "")).strip()
                if not title:
                    continue
                lookup[(year, normalize_title(title))] = (
                    str(row.get("Rank_first", "")).strip(),
                    str(row.get("Rank_last", "")).strip(),
                )
    return lookup


def choose_rank(existing: object, first: str, last: str) -> object:
    if t5.is_non_null(existing):
        return existing
    if t5.is_non_null(first):
        return first
    if t5.is_non_null(last):
        return last
    return existing


def prepare_task5b() -> pd.DataFrame:
    raw = pd.read_csv(SAMPLE_CSV)
    ranks = load_rank_lookup()
    chosen_ranks: list[object] = []
    for title, year, rank in zip(raw["Paper Title"], raw["Year"], raw["Rank"]):
        first, last = ranks.get((int(year), normalize_title(title)), ("", ""))
        chosen_ranks.append(choose_rank(rank, first, last))

    df = pd.DataFrame(
        {
            "title": raw["Paper Title"].astype(str),
            "venue": "ICLR",
            "year": pd.to_numeric(raw["Year"], errors="coerce").astype("Int64"),
            "rating": pd.to_numeric(raw["rating"], errors="coerce"),
            "confidence": pd.to_numeric(raw["confidence"], errors="coerce"),
            "preprint_visible": pd.to_numeric(raw["PreprintVisible"], errors="coerce")
            .fillna(0)
            .astype(int),
            "country": raw["Country"].map(t5.primary_country),
            "rank": chosen_ranks,
            "outcome": raw["outcome"].astype(str),
        }
    )
    df = df[df["rating"].notna() & df["confidence"].notna()].copy()
    df["institution_tier"] = [
        t5.institution_tier(rank, year) for rank, year in zip(df["rank"], df["year"])
    ]
    df["country_group"] = t5.collapse_countries(df["country"], top_n=8)
    df["rating_z"] = t5.standardize_within_venue_year(df, "rating")
    df["confidence_z"] = t5.standardize_within_venue_year(df, "confidence")
    return df.reset_index(drop=True)


def fit_ols_iclr(df: pd.DataFrame, outcome: str) -> RegressionResultsWrapper:
    model_df = df.dropna(subset=[outcome]).copy()
    model_df["year"] = model_df["year"].astype(int).astype("category")
    model_df["institution_tier"] = pd.Categorical(
        model_df["institution_tier"],
        categories=["not_listed", "top", "bottom"],
    )
    model_df["country_group"] = model_df["country_group"].astype("category")
    formula = (
        f"{outcome} ~ preprint_visible + "
        "C(institution_tier) + "
        f"C(country_group, Treatment(reference='{t5.COUNTRY_REFERENCE}')) + "
        "C(year)"
    )
    return smf.ols(formula, data=model_df).fit(cov_type="HC3")


def wald_contrast(
    result: RegressionResultsWrapper,
    contrast: np.ndarray,
) -> tuple[float, float, float, float]:
    effect = float(contrast @ result.params.to_numpy())
    wald = result.wald_test(contrast, scalar=True)
    p = float(wald.pvalue)
    cov = np.asarray(result.cov_params())
    se = float(np.sqrt(contrast @ cov @ contrast.T))
    return effect, effect - 1.96 * se, effect + 1.96 * se, p


def top_vs_bottom(result: RegressionResultsWrapper) -> tuple[float, float, float, float]:
    names = list(result.params.index)
    contrast = np.zeros(len(names))
    top = "C(institution_tier)[T.top]"
    bottom = "C(institution_tier)[T.bottom]"
    if top not in names or bottom not in names:
        return float("nan"), float("nan"), float("nan"), float("nan")
    contrast[names.index(top)] = 1.0
    contrast[names.index(bottom)] = -1.0
    return wald_contrast(result, contrast)


def country_param(level: str) -> str:
    return (
        f"C(country_group, Treatment(reference='{t5.COUNTRY_REFERENCE}'))[T.{level}]"
    )


def prefix_for(result: RegressionResultsWrapper, token: str) -> str:
    matches = [name for name in result.params.index if token in name and "[T." in name]
    if not matches:
        return token
    return matches[0].split("[T.", 1)[0]


def interpret(
    outcome: str,
    predictor: str,
    est: float,
    test: str,
) -> str:
    outcome_l = outcome.lower()
    if test == "joint Wald test":
        if predictor.startswith("Institution"):
            target = "institution-tier groups"
        elif predictor.startswith("Country"):
            target = "country groups"
        else:
            target = "year groups"
        return (
            f"Joint Wald test that all {target} have the same mean {outcome_l}, "
            "after adjusting for the other covariates."
        )
    if np.isnan(est):
        return f"Could not estimate a clear pattern for {predictor.lower()}."
    direction = "higher" if est >= 0 else "lower"
    mag = abs(est)
    if predictor == "Preprint visibility":
        return (
            f"Papers with a preprint dated before the submission deadline have "
            f"{mag:.3f} SD {direction} mean {outcome_l} than papers with no such "
            "preprint, after adjusting for institution tier, country, and year."
        )
    if predictor.startswith("Institution tier"):
        return (
            f"Top-20 first-author institutes have {mag:.3f} SD {direction} mean "
            f"{outcome_l} than bottom-ranked institutes (top vs bottom), after "
            "adjusting for preprint visibility, country, and year."
        )
    if predictor == "China vs. US":
        return (
            f"China-affiliated papers have {mag:.3f} SD {direction} mean "
            f"{outcome_l} than US-affiliated papers, after adjusting for "
            "preprint visibility, institution tier, and year."
        )
    if predictor == "2025 vs 2023":
        return (
            f"ICLR 2025 papers have {mag:.3f} SD {direction} mean {outcome_l} "
            "than ICLR 2023 papers, after adjusting for preprint visibility, "
            "institution tier, and country."
        )
    return (
        f"{predictor} is associated with {mag:.3f} SD {direction} mean {outcome_l}."
    )


def row(
    outcome: str,
    predictor: str,
    est: float,
    lo: float,
    hi: float,
    p: float,
    test: str,
) -> dict[str, str]:
    has_est = test != "joint Wald test" and not np.isnan(est)
    return {
        "Outcome": outcome,
        "Predictor": predictor,
        "Estimate": f"{est:.3f}" if has_est else "—",
        "95% CI": f"[{lo:.3f}, {hi:.3f}]" if has_est else "—",
        "p-value": t5.format_pvalue(p),
        "Test": test,
        "Interpretation": interpret(outcome, predictor, est, test),
    }


def build_rows(
    result: RegressionResultsWrapper,
    outcome: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    est, lo, hi, p = t5.coef_ci_p(result, "preprint_visible")
    rows.append(row(outcome, "Preprint visibility", est, lo, hi, p, "individual co-efficient"))

    est, lo, hi, p = top_vs_bottom(result)
    rows.append(row(outcome, "Institution tier (top vs bottom)", est, lo, hi, p, "individual co-efficient"))
    p_omni = t5.wald_omnibus_p(result, "C(institution_tier)")
    rows.append(row(outcome, "Institution tier, overall", float("nan"), float("nan"), float("nan"), p_omni, "joint Wald test"))

    est, lo, hi, p = t5.coef_ci_p(result, country_param("China"))
    rows.append(row(outcome, "China vs. US", est, lo, hi, p, "individual co-efficient"))
    p_omni = t5.wald_omnibus_p(result, prefix_for(result, "country_group"))
    rows.append(row(outcome, "Country, overall", float("nan"), float("nan"), float("nan"), p_omni, "joint Wald test"))

    est, lo, hi, p = t5.coef_ci_p(result, "C(year)[T.2025]")
    rows.append(row(outcome, "2025 vs 2023", est, lo, hi, p, "individual co-efficient"))
    p_omni = t5.wald_omnibus_p(result, "C(year)")
    rows.append(row(outcome, "Year, overall", float("nan"), float("nan"), float("nan"), p_omni, "joint Wald test"))
    return rows


def main() -> None:
    df = prepare_task5b()
    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
    print(f"Task 5b model dataset n={len(df)}")
    print(df["institution_tier"].value_counts().to_string())
    print(df["country_group"].value_counts().to_string())
    print(f"PreprintVisible=1: {int(df['preprint_visible'].sum())}")

    rating = fit_ols_iclr(df, "rating_z")
    conf = fit_ols_iclr(df, "confidence_z")
    print(rating.summary())
    print(conf.summary())

    table = pd.DataFrame(
        build_rows(rating, "Reviewer Rating") + build_rows(conf, "Confidence")
    )
    table.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8")
    print(f"\nSaved {OUTPUT_TABLE}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
