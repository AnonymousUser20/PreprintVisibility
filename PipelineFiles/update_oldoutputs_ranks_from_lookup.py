from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

NULL_LIKE = {"", "nan", "none", "null", "na", "not found"}
INSTITUTE_COLUMN_CANDIDATES = {
    "institute",
    "institutes",
    "institution",
    "institutions",
    "insitiute",
    "intitute",
    "authors_institutes",
}


def first_last_from_authors_institutes(raw_value: object) -> tuple[str, str]:
    text = str(raw_value).strip()
    if not text or text.lower() in NULL_LIKE:
        return "", ""

    blocks = [block.strip() for block in re.findall(r"\(([^)]+)\)", text) if block.strip()]
    if not blocks:
        return "", ""

    first_parts = [part.strip() for part in blocks[0].split(";") if part.strip()]
    last_parts = [part.strip() for part in blocks[-1].split(";") if part.strip()]
    return (first_parts[0] if first_parts else "", last_parts[-1] if last_parts else "")


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


def parse_institutes(raw_value: object) -> list[str]:
    value = str(raw_value).strip()
    if not value or value.lower() in NULL_LIKE:
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip() and str(item).strip().lower() not in NULL_LIKE
            ]
    except (ValueError, SyntaxError):
        pass

    parts = [p.strip() for p in value.split(";")]
    if len(parts) == 1:
        parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p and p.lower() not in NULL_LIKE]


def load_rank_lookup(lookup_csv: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with open(lookup_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            aff = str(row.get("affiliations", "")).strip()
            rank = str(row.get("rank", "")).strip()
            if not aff or not rank:
                continue
            lookup[normalize_text(aff)] = rank
    return lookup


def match_rank(inst: str, rank_lookup: dict[str, str]) -> str:
    norm = normalize_text(inst)
    if not norm:
        return ""

    exact = rank_lookup.get(norm)
    if exact:
        return exact

    # Fallback containment match for variants with dept/domain suffixes.
    for key, rank in rank_lookup.items():
        if key in norm or norm in key:
            return rank
    return ""


def infer_year_from_path(csv_path: Path) -> int | None:
    text = str(csv_path)
    match = re.search(r"20(23|24|25)", text)
    if not match:
        return None
    return int(match.group(0))


def find_institute_column(fieldnames: list[str]) -> str | None:
    for key in fieldnames:
        if str(key).strip().lower() in INSTITUTE_COLUMN_CANDIDATES:
            return key
    return None


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    oldoutputs_dir = project_root / "OldOutputs"
    lookup_rank_dir = project_root / "lookup_rank"

    rank_lookup_by_year = {
        2023: load_rank_lookup(lookup_rank_dir / "2023_cs_ranks.csv"),
        2024: load_rank_lookup(lookup_rank_dir / "2024_cs_ranks.csv"),
        2025: load_rank_lookup(lookup_rank_dir / "2025_cs_ranks.csv"),
    }

    files_updated = 0
    total_rows = 0
    both_non_empty_total = 0

    for csv_path in sorted(oldoutputs_dir.glob("**/*.csv")):
        if "institute_batches" in {part.lower() for part in csv_path.parts}:
            continue

        year = infer_year_from_path(csv_path)
        if year not in rank_lookup_by_year:
            continue
        rank_lookup = rank_lookup_by_year[year]

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        fieldnames = list(rows[0].keys())
        institute_col = find_institute_column(fieldnames)

        both_non_empty_file = 0
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
                both_non_empty_file += 1

        out_fields = list(rows[0].keys())
        if "Rank_first" not in out_fields:
            out_fields.append("Rank_first")
        if "Rank_last" not in out_fields:
            out_fields.append("Rank_last")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(rows)

        files_updated += 1
        total_rows += len(rows)
        both_non_empty_total += both_non_empty_file
        print(f"{csv_path} | rows={len(rows)} | both_rank_filled={both_non_empty_file}")

    print("\nSummary")
    print(f"Files updated: {files_updated}")
    print(f"Total rows processed: {total_rows}")
    print(f"Rows with both Rank_first and Rank_last non-empty: {both_non_empty_total}")


if __name__ == "__main__":
    main()
