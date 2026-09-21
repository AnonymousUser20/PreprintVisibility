from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "Outputs"
CSV_GLOB = "acl_2023*_preprint.csv"

# CMH rows: compare these two countries (must match values in Country column)
ROW_COUNTRY_TOP = "United States"
ROW_COUNTRY_BOT = "China"

NULL_DATE = {"", "nan", "none", "null", "na"}


def date_has_value(raw: str) -> bool:
    s = str(raw).strip()
    return bool(s) and s.lower() not in NULL_DATE


def load_country_date_counts(csv_path: Path) -> tuple[int, int, int, int]:
    """Returns P_usa, N_usa, P_china, N_china (author-row counts)."""
    p_usa = n_usa = p_china = n_china = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        country_key = next(
            (k for k in (reader.fieldnames or []) if k.strip().lower() == "country"),
            None,
        )
        date_key = next(
            (k for k in (reader.fieldnames or []) if k.strip().lower() == "date"),
            None,
        )
        if country_key is None or date_key is None:
            raise ValueError(f"Missing Country or Date column in {csv_path}")

        for row in reader:
            country = str(row.get(country_key, "")).strip()
            has_date = date_has_value(row.get(date_key, ""))

            if country == ROW_COUNTRY_TOP:
                if has_date:
                    p_usa += 1
                else:
                    n_usa += 1
            elif country == ROW_COUNTRY_BOT:
                if has_date:
                    p_china += 1
                else:
                    n_china += 1

    return p_usa, n_usa, p_china, n_china


def main() -> None:
    csv_paths = sorted(OUTPUTS_DIR.glob(CSV_GLOB))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSVs found in {OUTPUTS_DIR} matching pattern: {CSV_GLOB}"
        )

    tables: list[np.ndarray] = []
    for csv_path in csv_paths:
        p_usa, n_usa, p_china, n_china = load_country_date_counts(csv_path)
        table = np.array(
            [
                [p_usa, n_usa],
                [p_china, n_china],
            ],
            dtype=float,
        )

        print(f"\nSource: {csv_path}")
        print(
            f"{ROW_COUNTRY_TOP}: P_country (Date present)={p_usa}, "
            f"N_country (Date null/empty)={n_usa}"
        )
        print(
            f"{ROW_COUNTRY_BOT}: P_country (Date present)={p_china}, "
            f"N_country (Date null/empty)={n_china}"
        )
        print("2x2 table [rows: country, cols: Date present | Date null]:")
        print(table)

        if np.any(table.sum(axis=1) == 0):
            print("Skipping this file: one country has zero rows.")
            continue

        tables.append(table)

    if not tables:
        raise ValueError("No valid ACL 2023 tables available for CMH test.")

    st = StratifiedTable(tables)
    print(f"\nUsing {len(tables)} strata (ACL 2023 CSV files).")
    print("\nCommon odds ratio (MH):", st.oddsratio_pooled)
    print("95% CI:", st.oddsratio_pooled_confint())
    print("CMH p-value:", st.test_null_odds().pvalue)


if __name__ == "__main__":
    main()
