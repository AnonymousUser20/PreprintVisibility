from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


REGION_TO_COUNTRY_REGION = {
    "africa": {"Africa"},
    "asia": {"Asia"},
    "europe": {"Europe"},
    "northamerica": {"Americas"},
    "southamerica": {"Americas"},
    "latinamerica": {"Americas"},
    "americas": {"Americas"},
    "australasia": {"Oceania"},
    "oceania": {"Oceania"},
}


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    replacements = {
        "univ.": "university",
        "univ ": "university ",
        "inst.": "institute",
        "dept.": "department",
        "dep.": "department",
        "&": " and ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_affiliations(raw_value: str) -> list[str]:
    value = str(raw_value).strip()
    if not value or value.lower() == "nan":
        return []

    # The field is typically a serialized Python list like "['University X']".
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    # Fallback for malformed list strings.
    value = value.strip("[]")
    parts = [part.strip().strip("'\"") for part in value.split(",")]
    return [part for part in parts if part]


def build_country_lookup(
    countries_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    lookup: dict[tuple[str, str], str] = {}
    by_code: dict[str, str] = {}
    for row in countries_rows:
        alpha2 = str(row.get("alpha_2", "")).strip().lower()
        region = str(row.get("region", "")).strip()
        country_name = str(row.get("name", "")).strip()
        if alpha2 and region and country_name and alpha2 != "nan":
            lookup[(alpha2, region)] = country_name
        if alpha2 and country_name and alpha2 != "nan" and alpha2 not in by_code:
            by_code[alpha2] = country_name
    return lookup, by_code


def build_institution_lookup(institutions_rows: list[dict[str, str]]) -> dict[str, list[tuple[str, str]]]:
    lookup: dict[str, list[tuple[str, str]]] = {}
    for row in institutions_rows:
        institution = str(row.get("institution", "")).strip()
        region = str(row.get("region", "")).strip().lower()
        country_abbrv = str(row.get("countryabbrv", "")).strip().lower()
        if not institution or institution == "nan" or not country_abbrv or country_abbrv == "nan":
            continue
        key = normalize_text(institution)
        lookup.setdefault(key, []).append((country_abbrv, region))
    return lookup


def best_institution_matches(affiliation: str, institution_lookup: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    normalized_aff = normalize_text(affiliation)
    if not normalized_aff:
        return []

    # 1) Exact normalized match.
    if normalized_aff in institution_lookup:
        return institution_lookup[normalized_aff]

    # 2) Partial containment match (helps with dept names around institution names).
    matched: list[tuple[str, str]] = []
    for inst_name, values in institution_lookup.items():
        if inst_name in normalized_aff or normalized_aff in inst_name:
            matched.extend(values)
    return matched


def country_from_matches(
    inst_matches: list[tuple[str, str]],
    country_lookup: dict[tuple[str, str], str],
    country_by_code: dict[str, str],
) -> str | None:
    # First pass: strict match using countryabbrv + mapped region.
    for country_abbrv, inst_region in inst_matches:
        valid_country_regions = REGION_TO_COUNTRY_REGION.get(inst_region, set())
        for country_region in valid_country_regions:
            country = country_lookup.get((country_abbrv, country_region))
            if country:
                return country

    # Second pass: fallback by countryabbrv only when region mismatches.
    for country_abbrv, _ in inst_matches:
        country = country_by_code.get(country_abbrv)
        if country:
            return country

    return None


def map_affiliation_to_country(
    raw_affiliations: str,
    institution_lookup: dict[str, list[tuple[str, str]]],
    country_lookup: dict[tuple[str, str], str],
    country_by_code: dict[str, str],
) -> str:
    affiliations = parse_affiliations(raw_affiliations)
    mapped_countries: list[str] = []

    for affiliation in affiliations:
        inst_matches = best_institution_matches(affiliation, institution_lookup)
        country = country_from_matches(inst_matches, country_lookup, country_by_code)
        if country:
            mapped_countries.append(country)

    # Keep unique values in insertion order.
    unique_countries = list(dict.fromkeys(mapped_countries))
    return "; ".join(unique_countries)


def run_mapping(
    author_csv: Path,
    institutions_csv: Path,
    countries_csv: Path,
    output_csv: Path | None,
) -> None:
    with open(institutions_csv, "r", encoding="utf-8-sig", newline="") as f:
        institutions_rows = list(csv.DictReader(f))
    with open(countries_csv, "r", encoding="utf-8-sig", newline="") as f:
        countries_rows = list(csv.DictReader(f))
    with open(author_csv, "r", encoding="utf-8-sig", newline="") as f:
        author_rows = list(csv.DictReader(f))

    institution_lookup = build_institution_lookup(institutions_rows)
    country_lookup, country_by_code = build_country_lookup(countries_rows)

    for row in author_rows:
        row["Country"] = map_affiliation_to_country(
            row.get("affiliations", ""),
            institution_lookup,
            country_lookup,
            country_by_code,
        )

    out_path = output_csv if output_csv else author_csv
    fieldnames = list(author_rows[0].keys()) if author_rows else ["Country"]
    if "Country" not in fieldnames:
        fieldnames.append("Country")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(author_rows)

    total = len(author_rows)
    mapped = sum(1 for row in author_rows if str(row.get("Country", "")).strip())
    unmapped = total - mapped
    print(f"Saved: {out_path}")
    print(f"Rows: {total}, mapped: {mapped}, unmapped: {unmapped}")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Map affiliations to countries using institution and country reference tables."
    )
    parser.add_argument(
        "--author-csv",
        type=Path,
        default=base_dir / "emnlp_2024_main_author_output.csv",
        help="Path to EMNLP main author output CSV.",
    )
    parser.add_argument(
        "--institutions-csv",
        type=Path,
        default=base_dir / "institutions.csv",
        help="Path to institutions CSV.",
    )
    parser.add_argument(
        "--countries-csv",
        type=Path,
        default=base_dir / "countries.csv",
        help="Path to countries CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, updates the author CSV in place.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mapping(
        author_csv=args.author_csv,
        institutions_csv=args.institutions_csv,
        countries_csv=args.countries_csv,
        output_csv=args.output_csv,
    )
