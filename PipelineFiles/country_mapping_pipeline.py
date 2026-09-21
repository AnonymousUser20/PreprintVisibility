from __future__ import annotations

import argparse
import ast
import csv
import shutil
import subprocess
import sys
from collections import OrderedDict
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


def run_python_script(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    subprocess.run(cmd, check=True)


def sanitize_input_csv(input_csv: Path, sanitized_dir: Path) -> Path:
    """
    Remove NUL bytes from input CSV to avoid csv reader failures.
    Returns path to sanitized CSV. If no NUL found, returns original input path.
    """
    raw = input_csv.read_bytes()
    if b"\x00" not in raw:
        return input_csv

    sanitized_dir.mkdir(parents=True, exist_ok=True)
    sanitized_path = sanitized_dir / input_csv.name
    cleaned = raw.replace(b"\x00", b"")
    sanitized_path.write_bytes(cleaned)
    print(f"Sanitized NUL bytes in input: {input_csv} -> {sanitized_path}")
    return sanitized_path


def export_unmapped_affiliations(input_csv: Path, output_csv: Path) -> None:
    counter: dict[str, int] = {}

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("Country", "")).strip():
                continue
            affiliations = parse_affiliations(row.get("affiliations", ""))
            for affiliation in affiliations:
                counter[affiliation] = counter.get(affiliation, 0) + 1

    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0].lower()))
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["affiliation", "frequency"])
        for affiliation, frequency in sorted_items:
            writer.writerow([affiliation, frequency])

    print(f"Exported unmapped affiliations: {output_csv}")
    print(f"Distinct unmapped affiliations: {len(counter)}")


def load_affiliation_country_map(unmapped_csv: Path) -> dict[str, str]:
    aff_to_country: dict[str, str] = {}
    with open(unmapped_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            affiliation = str(row.get("affiliation", "")).strip()
            country = str(row.get("Country", "")).strip()
            if affiliation and country:
                aff_to_country[affiliation] = country
    return aff_to_country


def merge_back_to_mapped(mapped_csv: Path, unmapped_csv: Path) -> None:
    aff_to_country = load_affiliation_country_map(unmapped_csv)
    if not aff_to_country:
        print("No countries found in unmapped file to merge.")
        return

    with open(mapped_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Mapped CSV is empty.")

    updated_rows = 0
    for row in rows:
        existing_country = str(row.get("Country", "")).strip()
        if existing_country:
            continue

        affiliations = parse_affiliations(row.get("affiliations", ""))
        inferred: list[str] = []
        for affiliation in affiliations:
            country = aff_to_country.get(affiliation, "")
            if country:
                inferred.append(country)

        unique_inferred = list(OrderedDict.fromkeys(inferred))
        if unique_inferred:
            row["Country"] = "; ".join(unique_inferred)
            updated_rows += 1

    fieldnames = list(rows[0].keys())
    if "Country" not in fieldnames:
        fieldnames.append("Country")

    with open(mapped_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    mapped = sum(1 for row in rows if str(row.get("Country", "")).strip())
    unmapped = total - mapped
    print(f"Merged back into mapped CSV: {mapped_csv}")
    print(f"Rows updated from OpenRouter unmapped map: {updated_rows}")
    print(f"Final mapped rows: {mapped}, final unmapped rows: {unmapped}")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    parser = argparse.ArgumentParser(description="Run full country mapping pipeline.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "InputFiles2",
        help="Directory containing input author CSV files.",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Optional CSV filename to resume from (inclusive), e.g. acl_2024_main_author_output.csv",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Explicit input author CSV paths (absolute or relative). "
            "If set, only these files are processed; --input-dir and --start-from are ignored."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    lookup_dir = project_root / "lookup"
    countries_csv = lookup_dir / "countries.csv"
    institutions_csv = lookup_dir / "institutions.csv"
    intermediate_dir = project_root / "Intermediate_files"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    if not countries_csv.exists():
        raise FileNotFoundError(f"countries.csv not found: {countries_csv}")
    if not institutions_csv.exists():
        raise FileNotFoundError(f"institutions.csv not found: {institutions_csv}")

    if args.inputs:
        input_files = [p.resolve() for p in args.inputs]
        for p in input_files:
            if not p.exists():
                raise FileNotFoundError(f"Input CSV not found: {p}")
            if p.suffix.lower() != ".csv":
                raise ValueError(f"Input path must be a .csv file: {p}")
    else:
        input_dir = args.input_dir
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        input_files = sorted(input_dir.glob("*.csv"))
        if not input_files:
            raise FileNotFoundError(f"No CSV files found in input directory: {input_dir}")

        if args.start_from:
            start_name = args.start_from.strip()
            matched_index = next(
                (idx for idx, path in enumerate(input_files) if path.name == start_name),
                None,
            )
            if matched_index is None:
                raise FileNotFoundError(
                    f"--start-from file not found in input directory: {start_name}"
                )
            input_files = input_files[matched_index:]

    sanitized_dir = intermediate_dir / "sanitized_inputs"

    for input_csv in input_files:
        print(f"\nProcessing input file: {input_csv}")
        input_name = input_csv.stem
        mapped_csv = intermediate_dir / f"{input_name}_country_mapped.csv"
        unmapped_csv = intermediate_dir / f"{input_name}_unmapped.csv"
        author_input_csv = sanitize_input_csv(input_csv=input_csv, sanitized_dir=sanitized_dir)

        print("Step 1: Running country_mapping.py...")
        run_python_script(
            script_path=base_dir / "country_mapping.py",
            args=[
                "--author-csv",
                str(author_input_csv),
                "--institutions-csv",
                str(institutions_csv),
                "--countries-csv",
                str(countries_csv),
                "--output-csv",
                str(mapped_csv),
            ],
        )

        print("Step 2: Exporting unmapped affiliations...")
        export_unmapped_affiliations(input_csv=mapped_csv, output_csv=unmapped_csv)

        print("Step 3: Running openrouter_country_fallback.py on unmapped CSV (in-place)...")
        run_python_script(
            script_path=base_dir / "openrouter_country_fallback.py",
            args=["--input-csv", str(unmapped_csv), "--output-csv", str(unmapped_csv)],
        )

        print("Step 4: Merging unmapped countries back into mapped CSV...")
        merge_back_to_mapped(mapped_csv=mapped_csv, unmapped_csv=unmapped_csv)

        final_copy_csv = project_root / f"{input_name}_country_mapped.csv"
        shutil.copy2(mapped_csv, final_copy_csv)
        print(f"Final copy created in base directory: {final_copy_csv}")

        print("Pipeline complete for this input.")
        print(f"Mapped CSV: {mapped_csv}")
        print(f"Unmapped CSV (with OpenRouter Country): {unmapped_csv}")
        print(f"Final mapped CSV copy: {final_copy_csv}")


if __name__ == "__main__":
    main()
