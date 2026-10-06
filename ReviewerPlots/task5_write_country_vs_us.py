"""Write US-reference country contrast CSVs for Task 5 rating and confidence OLS."""

from __future__ import annotations

from task5_common import (
    COUNTRY_VS_US_DIR,
    OUTPUT_CONFIDENCE_COUNTRY_VS_US,
    OUTPUT_RATING_COUNTRY_VS_US,
    build_country_contrasts_vs_us,
    fit_ols,
    load_and_prepare,
    write_per_country_vs_us_csvs,
)


def main() -> None:
    df = load_and_prepare()

    rating_df = df.dropna(subset=["rating_z"]).copy()
    rating_result = fit_ols(rating_df, "rating_z")
    rating_n = int(rating_result.nobs)
    rating_table = build_country_contrasts_vs_us(
        rating_result, rating_df, "Rating", rating_n
    )
    rating_table.to_csv(OUTPUT_RATING_COUNTRY_VS_US, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_RATING_COUNTRY_VS_US}")

    conf_df = df.dropna(subset=["confidence_z"]).copy()
    conf_result = fit_ols(conf_df, "confidence_z")
    conf_n = int(conf_result.nobs)
    conf_table = build_country_contrasts_vs_us(
        conf_result, conf_df, "Confidence", conf_n
    )
    conf_table.to_csv(OUTPUT_CONFIDENCE_COUNTRY_VS_US, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_CONFIDENCE_COUNTRY_VS_US}")

    written = write_per_country_vs_us_csvs(rating_table, conf_table)
    print(f"Saved {len(written)} per-country CSVs in {COUNTRY_VS_US_DIR}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
