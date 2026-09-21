"""Apply date / country / rank pipelines to specific ICLR reject OldOutputs CSVs only."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))

from update_oldoutputs_date_from_oldarxiv import build_title_date_map, normalize_title
from update_oldoutputs_country_from_institutes import (
    affiliations_from_authors_institutes,
    build_batch_lookup,
    build_country_aliases,
    build_institutions_lookup,
    country_from_lookup_fallback,
    find_institute_column,
    normalize_text,
    split_institutes,
)
from update_oldoutputs_ranks_from_lookup import (
    first_last_from_authors_institutes,
    infer_year_from_path,
    load_rank_lookup,
    match_rank,
    parse_institutes,
    find_institute_column as find_institute_column_rank,
)

TARGET_CSVS = [
    PROJECT_ROOT / "OldOutputs" / "ICLR" / "ICLR 2023" / "Withdrawn_Rejected.csv",
    PROJECT_ROOT / "OldOutputs" / "ICLR" / "ICLR 2024" / "reject.csv",
    PROJECT_ROOT / "OldOutputs" / "ICLR" / "ICLR 2025" / "reject.csv",
]


def apply_dates(csv_path: Path, title_to_date: dict[str, str]) -> tuple[int, int]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0

    updated = 0
    for row in rows:
        title = normalize_title(row.get("Title", ""))
        mapped_date = title_to_date.get(title, "")
        # Treat "Not Found" arxiv dates as missing.
        if mapped_date.strip().lower() in {"", "not found", "nan", "none", "null", "na"}:
            mapped_date = ""
        if str(row.get("Date", "")) != mapped_date:
            row["Date"] = mapped_date
            updated += 1
        else:
            row["Date"] = mapped_date

    out_fields = list(rows[0].keys())
    if "Date" not in out_fields:
        out_fields.append("Date")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), updated


def apply_country(
    csv_path: Path,
    batch_lookup: dict[str, str],
    institution_lookup: dict,
    country_by_alpha2_region: dict,
    country_by_alpha2: dict,
) -> tuple[int, int]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0

    fieldnames = list(rows[0].keys())
    institute_col = find_institute_column(fieldnames)
    unmapped = 0

    for row in rows:
        if institute_col is None:
            row["Country"] = ""
            unmapped += 1
            continue

        raw_institutes = row.get(institute_col, "")
        if institute_col.strip().lower() == "authors_institutes":
            institutes = affiliations_from_authors_institutes(raw_institutes)
        else:
            # Prefer list-aware parse used by ranks; fall back to country splitter.
            institutes = parse_institutes(raw_institutes)
            if not institutes:
                institutes = split_institutes(raw_institutes)

        mapped_countries: list[str] = []
        for inst in institutes:
            norm_inst = normalize_text(inst)
            country = batch_lookup.get(norm_inst, "")
            if not country:
                country = (
                    country_from_lookup_fallback(
                        affiliation=inst,
                        institution_lookup=institution_lookup,
                        country_by_alpha2_region=country_by_alpha2_region,
                        country_by_alpha2=country_by_alpha2,
                    )
                    or ""
                )
            if country:
                mapped_countries.append(country)

        unique_countries = list(dict.fromkeys(c for c in mapped_countries if c))
        row["Country"] = "; ".join(unique_countries)
        if not row["Country"].strip():
            unmapped += 1

    out_fields = list(rows[0].keys())
    if "Country" not in out_fields:
        out_fields.append("Country")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), unmapped


def apply_ranks(csv_path: Path, rank_lookup_by_year: dict[int, dict[str, str]]) -> tuple[int, int]:
    year = infer_year_from_path(csv_path)
    if year not in rank_lookup_by_year:
        raise ValueError(f"Cannot infer year for {csv_path}")
    rank_lookup = rank_lookup_by_year[year]

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0

    fieldnames = list(rows[0].keys())
    institute_col = find_institute_column_rank(fieldnames)
    both_filled = 0

    for row in rows:
        raw_institutes = row.get(institute_col, "") if institute_col else ""
        if institute_col and institute_col.strip().lower() == "authors_institutes":
            first_inst, last_inst = first_last_from_authors_institutes(raw_institutes)
        else:
            institutes = parse_institutes(raw_institutes) if institute_col else []
            first_inst = institutes[0] if institutes else ""
            last_inst = institutes[-1] if institutes else ""

        rank_first = match_rank(first_inst, rank_lookup) if first_inst else ""
        rank_last = match_rank(last_inst, rank_lookup) if last_inst else ""
        row["Rank_first"] = rank_first
        row["Rank_last"] = rank_last
        if rank_first and rank_last:
            both_filled += 1

    out_fields = list(rows[0].keys())
    if "Rank_first" not in out_fields:
        out_fields.append("Rank_first")
    if "Rank_last" not in out_fields:
        out_fields.append("Rank_last")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), both_filled


def missing_summary(csv_path: Path) -> None:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        print(f"  {csv_path.name}: empty")
        return

    def is_missing(val: object) -> bool:
        return str(val or "").strip().lower() in {"", "nan", "none", "null", "na", "not found"}

    cols = ["Date", "Country", "Rank_first", "Rank_last", "Institutes"]
    print(f"  {csv_path.name} | rows={n} | cols={list(rows[0].keys())}")
    for col in cols:
        if col not in rows[0]:
            print(f"    {col}: column absent")
            continue
        miss = sum(1 for r in rows if is_missing(r.get(col, "")))
        print(f"    {col}: missing={miss} ({100.0 * miss / n:.1f}%)")


def main() -> None:
    for path in TARGET_CSVS:
        if not path.exists():
            raise FileNotFoundError(path)

    print("=== Step 1: Date from OldArXiv ===")
    title_to_date, conflicts = build_title_date_map(PROJECT_ROOT / "OldArXiv")
    print(f"Titles mapped from OldArXiv: {len(title_to_date)} | conflicts={conflicts}")
    for path in TARGET_CSVS:
        rows, updated = apply_dates(path, title_to_date)
        print(f"  {path.name}: rows={rows} | Date updates={updated}")

    print("\n=== Step 2: Country from institutes ===")
    countries_csv = PROJECT_ROOT / "lookup" / "countries.csv"
    institutions_csv = PROJECT_ROOT / "lookup" / "institutions.csv"
    institute_batches_dir = PROJECT_ROOT / "OldOutputs" / "institute_batches"
    country_alias_to_name, country_by_alpha2_region, country_by_alpha2 = build_country_aliases(
        countries_csv
    )
    batch_lookup = build_batch_lookup(institute_batches_dir, country_alias_to_name)
    institution_lookup = build_institutions_lookup(institutions_csv)
    for path in TARGET_CSVS:
        rows, unmapped = apply_country(
            path,
            batch_lookup,
            institution_lookup,
            country_by_alpha2_region,
            country_by_alpha2,
        )
        print(f"  {path.name}: rows={rows} | Country unmapped={unmapped}")

    print("\n=== Step 3: Ranks from lookup_rank ===")
    lookup_rank_dir = PROJECT_ROOT / "lookup_rank"
    rank_lookup_by_year = {
        2023: load_rank_lookup(lookup_rank_dir / "2023_cs_ranks.csv"),
        2024: load_rank_lookup(lookup_rank_dir / "2024_cs_ranks.csv"),
        2025: load_rank_lookup(lookup_rank_dir / "2025_cs_ranks.csv"),
    }
    for path in TARGET_CSVS:
        rows, both = apply_ranks(path, rank_lookup_by_year)
        print(f"  {path.name}: rows={rows} | both ranks filled={both}")

    print("\n=== Missing-data summary ===")
    for path in TARGET_CSVS:
        missing_summary(path)


if __name__ == "__main__":
    main()
