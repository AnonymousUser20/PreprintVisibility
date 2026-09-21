from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable


def to_int(value: str) -> int:
    return int(float(str(value).strip() or 0))


def main() -> None:
    cmh_dir = Path(__file__).resolve().parent
    input_csv = cmh_dir / "pre-print_or_not_30.csv"
    output_csv = cmh_dir / "top_bottom_30.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    tables: list[np.ndarray] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            before_30_usa = to_int(row.get("preprint before 30_USA", "0"))
            within_30_usa = to_int(row.get("30 days before submission_USA", "0"))
            before_30_china = to_int(row.get("preprint before 30_CHINA", "0"))
            within_30_china = to_int(row.get("30 days before submission_CHINA", "0"))

            table = np.array(
                [
                    [before_30_usa, within_30_usa],
                    [before_30_china, within_30_china],
                ],
                dtype=float,
            )

            # Skip invalid strata where one row has no observations.
            if np.any(table.sum(axis=1) == 0):
                continue
            tables.append(table)

    if not tables:
        raise ValueError("No valid strata found to run CMH test.")

    st = StratifiedTable(tables)
    ci_low, ci_high = st.oddsratio_pooled_confint()

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "script_filename",
                "common_odds_ratio_mh",
                "ci_95_low",
                "ci_95_high",
                "cmh_p_value",
                "num_strata",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "script_filename": Path(__file__).name,
                "common_odds_ratio_mh": st.oddsratio_pooled,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "cmh_p_value": st.test_null_odds().pvalue,
                "num_strata": len(tables),
            }
        )

    print(f"Saved CMH summary: {output_csv}")


if __name__ == "__main__":
    main()
