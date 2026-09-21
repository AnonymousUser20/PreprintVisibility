from __future__ import annotations

import csv
import re
from pathlib import Path

NULL_LIKE = {"", "nan", "none", "null", "na"}
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
INSTITUTE_COLUMN_CANDIDATES = {
    "institute",
    "institutes",
    "institution",
    "institutions",
    "insitiute",
    "authors_institutes",
}


def affiliations_from_authors_institutes(raw: object) -> list[str]:
    text = str(raw).strip()
    if not text or text.lower() in NULL_LIKE:
        return []

    institutes: list[str] = []
    for block in re.findall(r"\(([^)]+)\)", text):
        for part in block.split(";"):
            part = part.strip()
            if part and part.lower() not in NULL_LIKE:
                institutes.append(part)
    return institutes


def normalize_text(text: object) -> str:
    value = str(text).strip().lower()
    replacements = {
        "univ.": "university",
        "univ ": "university ",
        "inst.": "institute",
        "dept.": "department",
        "dep.": "department",
        "&": " and ",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_institutes(raw: object) -> list[str]:
    text = str(raw).strip()
    if not text or text.lower() in NULL_LIKE:
        return []

    parts = [p.strip() for p in text.split(";")]
    if len(parts) == 1:
        # fallback when a row uses comma delimiter only
        parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p and p.lower() not in NULL_LIKE]


def build_country_aliases(countries_csv: Path) -> tuple[dict[str, str], dict[tuple[str, str], str], dict[str, str]]:
    alias_to_country: dict[str, str] = {}
    by_alpha2_region: dict[tuple[str, str], str] = {}
    by_alpha2: dict[str, str] = {}

    with open(countries_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = str(row.get("name", "")).strip()
            alpha2 = str(row.get("alpha_2", "")).strip().lower()
            alpha3 = str(row.get("alpha_3", "")).strip().lower()
            region = str(row.get("region", "")).strip()
            if not name:
                continue

            alias_to_country[normalize_text(name)] = name
            if alpha2:
                alias_to_country[normalize_text(alpha2)] = name
                by_alpha2[alpha2] = name
            if alpha3:
                alias_to_country[normalize_text(alpha3)] = name
            if alpha2 and region:
                by_alpha2_region[(alpha2, region)] = name

    # Common aliases seen in institute_batches.
    alias_to_country.setdefault(normalize_text("USA"), "United States")
    alias_to_country.setdefault(normalize_text("UK"), "United Kingdom")
    return alias_to_country, by_alpha2_region, by_alpha2


def build_batch_lookup(
    institute_batches_dir: Path,
    country_alias_to_name: dict[str, str],
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for batch_csv in sorted(institute_batches_dir.glob("institute_batch_*.csv")):
        with open(batch_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name", "")).strip()
                country_raw = str(row.get("country", "")).strip()
                if not name:
                    continue
                canonical_country = country_alias_to_name.get(
                    normalize_text(country_raw),
                    country_raw,
                )
                lookup[normalize_text(name)] = canonical_country
    return lookup


def build_institutions_lookup(institutions_csv: Path) -> dict[str, list[tuple[str, str]]]:
    lookup: dict[str, list[tuple[str, str]]] = {}
    with open(institutions_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            institution = str(row.get("institution", "")).strip()
            region = str(row.get("region", "")).strip().lower()
            country_abbrv = str(row.get("countryabbrv", "")).strip().lower()
            if not institution or not country_abbrv:
                continue
            key = normalize_text(institution)
            lookup.setdefault(key, []).append((country_abbrv, region))
    return lookup


def best_institution_matches(affiliation: str, institution_lookup: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    normalized_aff = normalize_text(affiliation)
    if not normalized_aff:
        return []
    if normalized_aff in institution_lookup:
        return institution_lookup[normalized_aff]

    matched: list[tuple[str, str]] = []
    for inst_name, values in institution_lookup.items():
        if inst_name in normalized_aff or normalized_aff in inst_name:
            matched.extend(values)
    return matched


def country_from_lookup_fallback(
    affiliation: str,
    institution_lookup: dict[str, list[tuple[str, str]]],
    country_by_alpha2_region: dict[tuple[str, str], str],
    country_by_alpha2: dict[str, str],
) -> str | None:
    matches = best_institution_matches(affiliation, institution_lookup)
    for country_abbrv, inst_region in matches:
        valid_regions = REGION_TO_COUNTRY_REGION.get(inst_region, set())
        for country_region in valid_regions:
            country = country_by_alpha2_region.get((country_abbrv, country_region))
            if country:
                return country
    for country_abbrv, _ in matches:
        country = country_by_alpha2.get(country_abbrv)
        if country:
            return country
    return None


def find_institute_column(fieldnames: list[str]) -> str | None:
    for key in fieldnames:
        if str(key).strip().lower() in INSTITUTE_COLUMN_CANDIDATES:
            return key
    return None


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    oldoutputs_dir = project_root / "OldOutputs"
    institute_batches_dir = oldoutputs_dir / "institute_batches"
    lookup_dir = project_root / "lookup"

    countries_csv = lookup_dir / "countries.csv"
    institutions_csv = lookup_dir / "institutions.csv"

    country_alias_to_name, country_by_alpha2_region, country_by_alpha2 = build_country_aliases(
        countries_csv
    )
    batch_lookup = build_batch_lookup(institute_batches_dir, country_alias_to_name)
    institution_lookup = build_institutions_lookup(institutions_csv)

    files_updated = 0
    total_rows = 0
    total_unmapped = 0
    per_file_unmapped: list[tuple[str, int, int]] = []

    for csv_path in sorted(oldoutputs_dir.glob("**/*.csv")):
        if "institute_batches" in {part.lower() for part in csv_path.parts}:
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        fieldnames = list(rows[0].keys())
        institute_col = find_institute_column(fieldnames)
        if institute_col is None:
            # ensure Country column exists even if there is no institute field.
            for row in rows:
                row["Country"] = ""
            out_fields = list(rows[0].keys())
            if "Country" not in out_fields:
                out_fields.append("Country")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=out_fields)
                writer.writeheader()
                writer.writerows(rows)
            files_updated += 1
            total_rows += len(rows)
            total_unmapped += len(rows)
            per_file_unmapped.append((str(csv_path), len(rows), len(rows)))
            continue

        unmapped_in_file = 0
        for row in rows:
            raw_institutes = row.get(institute_col, "")
            if institute_col.strip().lower() == "authors_institutes":
                institutes = affiliations_from_authors_institutes(raw_institutes)
            else:
                institutes = split_institutes(raw_institutes)
            mapped_countries: list[str] = []

            for inst in institutes:
                norm_inst = normalize_text(inst)
                country = batch_lookup.get(norm_inst, "")
                if not country:
                    country = country_from_lookup_fallback(
                        affiliation=inst,
                        institution_lookup=institution_lookup,
                        country_by_alpha2_region=country_by_alpha2_region,
                        country_by_alpha2=country_by_alpha2,
                    ) or ""
                if country:
                    mapped_countries.append(country)

            unique_countries = list(dict.fromkeys(c for c in mapped_countries if c))
            row["Country"] = "; ".join(unique_countries)
            if not row["Country"].strip():
                unmapped_in_file += 1

        out_fields = list(rows[0].keys())
        if "Country" not in out_fields:
            out_fields.append("Country")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(rows)

        files_updated += 1
        total_rows += len(rows)
        total_unmapped += unmapped_in_file
        per_file_unmapped.append((str(csv_path), len(rows), unmapped_in_file))

    print(f"Files updated: {files_updated}")
    print(f"Total rows processed: {total_rows}")
    print(f"Rows not mapped to any country: {total_unmapped}")
    print("\nPer-file unmapped counts:")
    for file_path, rows_count, unmapped_count in per_file_unmapped:
        print(f"{file_path} | rows={rows_count} | unmapped={unmapped_count}")


if __name__ == "__main__":
    main()
