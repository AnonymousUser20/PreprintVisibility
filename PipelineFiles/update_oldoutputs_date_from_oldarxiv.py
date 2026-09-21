from __future__ import annotations

import csv
from pathlib import Path


def normalize_title(title: object) -> str:
    return str(title).strip()


def build_title_date_map(oldarxiv_dir: Path) -> tuple[dict[str, str], int]:
    mapping: dict[str, str] = {}
    conflicts = 0

    for csv_path in sorted(oldarxiv_dir.glob("**/*.csv")):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            if "Title" not in reader.fieldnames or "Submission Date" not in reader.fieldnames:
                continue

            for row in reader:
                title = normalize_title(row.get("Title", ""))
                date_value = str(row.get("Submission Date", "")).strip()
                if not title:
                    continue

                existing = mapping.get(title)
                if existing is None:
                    mapping[title] = date_value
                elif existing != date_value and date_value:
                    # Keep first seen value but count mismatches.
                    conflicts += 1

    return mapping, conflicts


def update_oldoutputs(oldoutputs_dir: Path, title_to_date: dict[str, str]) -> tuple[int, int, int]:
    files_updated = 0
    rows_updated = 0
    files_skipped_no_title = 0

    for csv_path in sorted(oldoutputs_dir.glob("**/*.csv")):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
            fieldnames = rows[0].keys() if rows else []

        if not rows:
            continue
        if "Title" not in fieldnames:
            files_skipped_no_title += 1
            continue

        updated_this_file = 0
        for row in rows:
            title = normalize_title(row.get("Title", ""))
            mapped_date = title_to_date.get(title, "")
            if str(row.get("Date", "")) != mapped_date:
                row["Date"] = mapped_date
                updated_this_file += 1

        out_fieldnames = list(rows[0].keys())
        if "Date" not in out_fieldnames:
            out_fieldnames.append("Date")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        files_updated += 1
        rows_updated += updated_this_file

    return files_updated, rows_updated, files_skipped_no_title


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    oldarxiv_dir = project_root / "OldArXiv"
    oldoutputs_dir = project_root / "OldOutputs"

    if not oldarxiv_dir.exists():
        raise FileNotFoundError(f"OldArXiv folder not found: {oldarxiv_dir}")
    if not oldoutputs_dir.exists():
        raise FileNotFoundError(f"OldOutputs folder not found: {oldoutputs_dir}")

    title_to_date, conflicts = build_title_date_map(oldarxiv_dir)
    files_updated, rows_updated, files_skipped_no_title = update_oldoutputs(
        oldoutputs_dir, title_to_date
    )

    print(f"Titles mapped from OldArXiv: {len(title_to_date)}")
    print(f"Conflicting Title->Submission Date mappings: {conflicts}")
    print(f"OldOutputs CSV files updated (with Title column): {files_updated}")
    print(f"Total row Date updates applied: {rows_updated}")
    print(f"OldOutputs CSV files skipped (no Title column): {files_skipped_no_title}")


if __name__ == "__main__":
    main()
