from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


def parse_affiliations(raw_value: str) -> list[str]:
    value = str(raw_value).strip()
    if not value or value.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    value = value.strip("[]")
    parts = [part.strip().strip("'\"") for part in value.split(",")]
    return [part for part in parts if part]


def normalize_text(text: str) -> str:
    cleaned = str(text).strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_year_from_filename(file_path: Path) -> str:
    matches = re.findall(r"(20\d{2})", file_path.stem)
    if not matches:
        raise ValueError(f"Could not find year in input filename: {file_path.name}")
    return matches[0]


def load_rank_lookup(rank_csv: Path) -> dict[str, str]:
    if not rank_csv.exists():
        raise FileNotFoundError(f"Rank lookup CSV not found: {rank_csv}")

    lookup: dict[str, str] = {}
    with open(rank_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            affiliation = str(row.get("affiliations", "")).strip()
            rank = str(row.get("rank", "")).strip()
            if affiliation and rank:
                lookup[normalize_text(affiliation)] = rank
    return lookup


def rank_for_row(affiliations_raw: str, rank_lookup: dict[str, str]) -> str:
    affiliations = parse_affiliations(affiliations_raw)
    matched_ranks: list[tuple[int, str]] = []

    for aff in affiliations:
        normalized_aff = normalize_text(aff)
        if not normalized_aff:
            continue

        # First pass: exact normalized match
        rank = rank_lookup.get(normalized_aff)
        if rank:
            try:
                matched_ranks.append((int(rank), rank))
            except ValueError:
                pass
            continue

        # Second pass: containment matching for minor naming variations
        for ranked_aff, ranked_value in rank_lookup.items():
            if ranked_aff in normalized_aff or normalized_aff in ranked_aff:
                try:
                    matched_ranks.append((int(ranked_value), ranked_value))
                except ValueError:
                    continue

    if not matched_ranks:
        return ""

    # Use the best (lowest) rank when multiple affiliations match.
    matched_ranks.sort(key=lambda x: x[0])
    return matched_ranks[0][1]


def process_file(input_csv: Path, lookup_rank_dir: Path) -> Path:
    year = extract_year_from_filename(input_csv)
    rank_csv = lookup_rank_dir / f"{year}_cs_ranks.csv"
    rank_lookup = load_rank_lookup(rank_csv)

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Input CSV has no rows: {input_csv}")

    for row in rows:
        row["rank"] = rank_for_row(row.get("affiliations", ""), rank_lookup)

    fieldnames = list(rows[0].keys())
    if "rank" not in fieldnames:
        fieldnames.append("rank")

    output_csv = input_csv.with_name(f"{input_csv.stem}_ranked.csv")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mapped = sum(1 for row in rows if str(row.get("rank", "")).strip())
    print(f"Processed: {input_csv.name} (year={year})")
    print(f"Lookup used: {rank_csv.name}")
    print(f"Mapped ranks: {mapped}/{len(rows)}")
    print(f"Saved: {output_csv}")
    return output_csv


def parse_args() -> argparse.Namespace:
    pipeline_dir = Path(__file__).resolve().parent
    project_root = pipeline_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Map ranks for *_country_mapped.csv files using year-specific lookup_rank CSVs."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=project_root,
        help="Base directory containing *_country_mapped.csv files.",
    )
    parser.add_argument(
        "--lookup-rank-dir",
        type=Path,
        default=project_root / "lookup_rank",
        help="Directory containing {year}_cs_ranks.csv files.",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit input *_country_mapped.csv files. If omitted, auto-discovers in base-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    lookup_rank_dir = args.lookup_rank_dir

    if args.inputs:
        input_files = args.inputs
    else:
        input_files = sorted(base_dir.glob("*_country_mapped.csv"))

    if not input_files:
        raise FileNotFoundError(f"No *_country_mapped.csv files found in: {base_dir}")

    for input_csv in input_files:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        process_file(input_csv=input_csv, lookup_rank_dir=lookup_rank_dir)

    print("Rank mapping pipeline complete.")


if __name__ == "__main__":
    main()
