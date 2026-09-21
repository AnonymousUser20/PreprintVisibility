"""Task 5 Model 1: OLS for venue-year standardized reviewer rating."""

# pyright: reportMissingModuleSource=false, reportAny=false

from __future__ import annotations

from task5_common import (
    OUTPUT_DATASET,
    OUTPUT_MISSING,
    OUTPUT_RATING_ROWS,
    OUTPUT_TABLE5,
    build_outcome_rows,
    fit_ols,
    load_and_prepare,
    merge_table5,
    missing_data_report,
    print_table_rows,
)


def main() -> None:
    df = load_and_prepare()
    missing_text = missing_data_report(df)
    _ = OUTPUT_MISSING.write_text(missing_text, encoding="utf-8")
    print(missing_text)

    df.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
    print(f"\nSaved model dataset: {OUTPUT_DATASET}")

    model_df = df.dropna(subset=["rating_z"]).copy()
    result = fit_ols(model_df, "rating_z")
    print(result.summary())

    n = int(result.nobs)
    rows = build_outcome_rows(result, "Rating", n)
    rows.to_csv(OUTPUT_RATING_ROWS, index=False, encoding="utf-8")
    print(f"\nSaved rating model rows: {OUTPUT_RATING_ROWS}")
    print_table_rows(rows, "Rating model coefficients (Table 5 rows)")

    table5 = merge_table5(rating_rows=rows)
    table5.to_csv(OUTPUT_TABLE5, index=False, encoding="utf-8")
    print(f"\nUpdated Table 5: {OUTPUT_TABLE5}")
    print_table_rows(table5, "Table 5 (current)")


if __name__ == "__main__":
    main()
