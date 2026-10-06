"""Task 4 Model B: preprint-timing logistic on all 14,883 ICLR submissions.

Does not modify task4_iclr_submission_acceptance.py or its Model A outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.base.wrapper import ResultsWrapper as BinaryResultsWrapper

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import task4_iclr_submission_acceptance as t4

OUTPUT_TABLE = SCRIPT_DIR / "table4_model_b_timing.csv"
OUTPUT_LEVELS = SCRIPT_DIR / "table4_model_b_timing_levels.csv"
OUTPUT_US_COUNTRY = SCRIPT_DIR / "table4_model_b_us_country_contrasts.csv"
OUTPUT_CHINA_COUNTRY = SCRIPT_DIR / "table4_model_b_china_country_contrasts.csv"
OUTPUT_DATASET = SCRIPT_DIR / "task4_model_b_dataset.csv"
OUTPUT_REPORT = SCRIPT_DIR / "task4_model_b_report.txt"

TIMING_REFERENCE = "cfp_to_review"
TIMING_ORDER = [
    "cfp_to_review",
    "more_than_180d",
    "within_180d",
    "within_90d",
    "within_60d",
    "within_30d",
    "no_preprint",
]
TIMING_LABELS = {
    "no_preprint": "No usable date or posted after review",
    "cfp_to_review": "CfP to review window",
    "within_30d": "<30 days before submission",
    "within_60d": "30-60 days before submission",
    "within_90d": "60-90 days before submission",
    "within_180d": "90-180 days before submission",
    "more_than_180d": ">180 days before submission",
}


def prepare_model_b_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["preprint_timing"] = out["preprint_timing"].astype(str).str.strip()
    return out


def fit_model_b(df: pd.DataFrame) -> BinaryResultsWrapper:
    model_df = df.copy()
    model_df["year"] = model_df["year"].astype("category")
    model_df["institution_tier"] = pd.Categorical(
        model_df["institution_tier"],
        categories=["not_listed", "top", "bottom"],
    )
    model_df["preprint_timing"] = pd.Categorical(
        model_df["preprint_timing"],
        categories=TIMING_ORDER,
    )
    formula = (
        f"accepted ~ C(preprint_timing, Treatment(reference='{TIMING_REFERENCE}')) + "
        "C(institution_tier) + C(country_group) + C(year)"
    )
    return smf.logit(formula, data=model_df).fit(disp=False, maxiter=200)


def timing_param(level: str) -> str:
    return f"C(preprint_timing, Treatment(reference='{TIMING_REFERENCE}'))[T.{level}]"


def timing_omnibus_prefix(result: BinaryResultsWrapper) -> str:
    matches = [
        name
        for name in result.params.index
        if name.startswith("C(preprint_timing") and "[T." in name
    ]
    if not matches:
        return "C(preprint_timing)"
    return matches[0].split("[T.", 1)[0]


def build_timing_levels_table(
    result: BinaryResultsWrapper,
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    ref_n = int((df["preprint_timing"] == TIMING_REFERENCE).sum())
    rows.append(
        {
            "Timing category": TIMING_LABELS[TIMING_REFERENCE],
            "Code": TIMING_REFERENCE,
            "N": ref_n,
            "Odds ratio": "1.00 (reference)",
            "95% CI": "—",
            "p-value": "—",
        }
    )
    for level in TIMING_ORDER:
        if level == TIMING_REFERENCE:
            continue
        or_val, ci_lo, ci_hi, p = t4.or_ci_p(result, timing_param(level))
        rows.append(
            {
                "Timing category": TIMING_LABELS[level],
                "Code": level,
                "N": int((df["preprint_timing"] == level).sum()),
                "Odds ratio": f"{or_val:.2f}" if not pd.isna(or_val) else "—",
                "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not pd.isna(or_val) else "—",
                "p-value": t4.format_pvalue(p),
            }
        )
    return pd.DataFrame(rows)


def build_summary_table(
    result: BinaryResultsWrapper,
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    or_val, ci_lo, ci_hi, p = t4.or_ci_p(result, timing_param("within_30d"))
    rows.append(
        {
            "Predictor": "within_30d vs CfP-review",
            "Odds ratio": f"{or_val:.2f}" if not pd.isna(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not pd.isna(or_val) else "—",
            "p-value": t4.format_pvalue(p),
            "Test": "pairwise coefficient",
            "Interpretation": (
                f"Posting within 30 days of submission had about {or_val:.2f}x the "
                "odds of acceptance of posting in the CfP-to-review window, "
                "after adjusting for institution tier, country, and year."
            ),
        }
    )

    p_omni = t4.wald_omnibus_p(result, timing_omnibus_prefix(result))
    rows.append(
        {
            "Predictor": "Preprint timing, overall",
            "Odds ratio": "—",
            "95% CI": "—",
            "p-value": t4.format_pvalue(p_omni),
            "Test": "joint Wald test",
            "Interpretation": (
                "Joint Wald test that all preprint-timing coefficients are jointly "
                "zero, after adjusting for institution tier, country, and year."
            ),
        }
    )

    or_val, ci_lo, ci_hi, p = t4.tier_top_vs_bottom_contrast(result)
    rows.append(
        {
            "Predictor": "Institution tier",
            "Odds ratio": f"{or_val:.2f}" if not pd.isna(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not pd.isna(or_val) else "—",
            "p-value": t4.format_pvalue(p),
            "Test": "pairwise coefficient",
            "Interpretation": (
                f"Top-20 first-author institutes had about {or_val:.1f}x the odds of "
                "acceptance of bottom-ranked institutes (top vs bottom)."
            ),
        }
    )

    or_val, ci_lo, ci_hi, p = t4.focal_vs_country_contrast(
        result, t4.CHINA_COUNTRY, t4.US_COUNTRY
    )
    rows.append(
        {
            "Predictor": "China vs. US",
            "Odds ratio": f"{or_val:.2f}" if not pd.isna(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not pd.isna(or_val) else "—",
            "p-value": t4.format_pvalue(p),
            "Test": "pairwise coefficient",
            "Interpretation": (
                f"China-affiliated submissions had about {or_val:.2f}x the odds of "
                "acceptance of US-affiliated submissions (pairwise coefficient, "
                "China vs US)."
            ),
        }
    )

    p_omni = t4.wald_omnibus_p(result, "C(country_group)")
    rows.append(
        {
            "Predictor": "Country, overall",
            "Odds ratio": "—",
            "95% CI": "—",
            "p-value": t4.format_pvalue(p_omni),
            "Test": "joint Wald test",
            "Interpretation": (
                "Joint Wald test that all country-group coefficients are jointly "
                "zero, after adjusting for preprint timing, institution tier, and year."
            ),
        }
    )

    or_val, ci_lo, ci_hi, p = t4.or_ci_p(result, "C(year)[T.2025]")
    rows.append(
        {
            "Predictor": "2025 vs 2023",
            "Odds ratio": f"{or_val:.2f}" if not pd.isna(or_val) else "—",
            "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not pd.isna(or_val) else "—",
            "p-value": t4.format_pvalue(p),
            "Test": "pairwise coefficient",
            "Interpretation": (
                f"ICLR 2025 submissions had about {or_val:.2f}x the odds of "
                "acceptance of ICLR 2023 submissions (2025 vs 2023)."
            ),
        }
    )

    p_omni = t4.wald_omnibus_p(result, "C(year)")
    rows.append(
        {
            "Predictor": "Year, overall",
            "Odds ratio": "—",
            "95% CI": "—",
            "p-value": t4.format_pvalue(p_omni),
            "Test": "joint Wald test",
            "Interpretation": (
                "Joint Wald test that all year coefficients are jointly zero, "
                "after adjusting for preprint timing, institution tier, and country."
            ),
        }
    )
    return pd.DataFrame(rows)


def timing_report(df: pd.DataFrame) -> str:
    lines = [
        "Task 4 Model B report (ICLR submissions)",
        f"Total unique submissions: {len(df)}",
        f"Accepted: {int(df['accepted'].sum())} | Rejected: {int((1 - df['accepted']).sum())}",
        "",
        "Timing encoding: original buckets kept, including cfp_to_review.",
        "no_preprint = missing date or posted after the review deadline.",
        f"Reference category: {TIMING_REFERENCE} ({TIMING_LABELS[TIMING_REFERENCE]}).",
        "",
        "Preprint timing distribution:",
    ]
    counts = df["preprint_timing"].value_counts()
    for code in TIMING_ORDER:
        lines.append(f"  {code}: {int(counts.get(code, 0))}")
    return "\n".join(lines)


def main() -> None:
    if t4.OUTPUT_DATASET.exists():
        df = pd.read_csv(t4.OUTPUT_DATASET)
    else:
        deadline_windows = t4.load_deadline_windows(t4.DEADLINES_CSV)
        df = t4.collapse_countries(t4.load_submissions(t4.ICLR_DIR, deadline_windows))

    df = prepare_model_b_frame(df)
    report = timing_report(df)
    _ = OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(report)
    print()

    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
    print(f"Saved Model B dataset: {OUTPUT_DATASET}")

    result = fit_model_b(df)
    print(result.summary())

    summary = build_summary_table(result, df)
    summary.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8")
    print(f"\nSaved Model B summary: {OUTPUT_TABLE}")
    for _, row in summary.iterrows():
        print(
            f"{row['Predictor']:28} OR={row['Odds ratio']:>6}  "
            f"95% CI={row['95% CI']:>16}  p={row['p-value']}  {row['Test']}"
        )

    levels = build_timing_levels_table(result, df)
    levels.to_csv(OUTPUT_LEVELS, index=False, encoding="utf-8")
    print(f"\nSaved Model B timing levels: {OUTPUT_LEVELS}")
    print(levels.to_string(index=False))

    t4.save_and_print_country_contrasts(result, df, t4.US_COUNTRY, OUTPUT_US_COUNTRY)
    t4.save_and_print_country_contrasts(result, df, t4.CHINA_COUNTRY, OUTPUT_CHINA_COUNTRY)


if __name__ == "__main__":
    main()
