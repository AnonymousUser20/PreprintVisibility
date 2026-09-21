import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_CSV = SCRIPT_DIR / "paper_review_scores.csv"
OLD_OUTPUTS_ROOT = PROJECT_ROOT / "OldOutputs"

TITLE_COLUMN = "Paper Title"
COUNTRY_COLUMN = "Country"
RANK_COLUMN = "Rank"


def normalize_title(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def year_from_path(path: Path) -> int | None:
    match = re.search(r"\b(2023|2024|2025)\b", str(path))
    return int(match.group(1)) if match else None


def conference_from_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "iclr" in parts:
        return "ICLR"
    if "icml" in parts:
        return "ICML"
    if "neurips" in parts:
        return "NeurIPS"
    return None


def should_use_old_output(path: Path, target_pairs: set[tuple[str, int]]) -> bool:
    conference = conference_from_path(path)
    year = year_from_path(path)
    if conference is None or year is None:
        return False

    if (conference, year) not in target_pairs:
        return False

    if conference == "ICML" and year != 2025:
        return False

    return conference in {"ICLR", "ICML", "NeurIPS"}


def choose_rank(row: dict[str, str]) -> str:
    rank_first = row.get("Rank_first", "").strip()
    rank_last = row.get("Rank_last", "").strip()
    return rank_first or rank_last


def load_score_rows() -> tuple[list[dict[str, str]], list[str]]:
    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for column in (COUNTRY_COLUMN, RANK_COLUMN):
        if column not in fieldnames:
            fieldnames.append(column)

    return rows, fieldnames


def target_conference_year_pairs(rows: list[dict[str, str]]) -> set[tuple[str, int]]:
    pairs = set()
    for row in rows:
        conference = row.get("Conference", "").strip()
        year_text = row.get("Year", "").strip()
        if not conference or not year_text:
            continue
        try:
            year = int(year_text)
        except ValueError:
            continue
        pairs.add((conference, year))
    return pairs


def load_old_output_lookup(
    target_pairs: set[tuple[str, int]],
) -> dict[tuple[str, int, str], tuple[str, str]]:
    lookup = {}

    for csv_path in sorted(OLD_OUTPUTS_ROOT.rglob("*.csv")):
        if not should_use_old_output(csv_path, target_pairs):
            continue

        conference = conference_from_path(csv_path)
        year = year_from_path(csv_path)
        if conference is None or year is None:
            continue

        with csv_path.open(newline="", encoding="utf-8-sig") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                title = row.get("Title", "").strip()
                if not title:
                    continue

                key = (conference, year, normalize_title(title))
                country = row.get("Country", "").strip()
                rank = choose_rank(row)

                # Keep the first non-empty match for duplicate titles.
                existing = lookup.get(key)
                if existing is None or (not existing[0] and country) or (not existing[1] and rank):
                    lookup[key] = (country, rank)

    return lookup


def enrich_rows(
    rows: list[dict[str, str]],
    old_output_lookup: dict[tuple[str, int, str], tuple[str, str]],
) -> int:
    matched = 0

    for row in rows:
        try:
            year = int(row.get("Year", "").strip())
        except ValueError:
            row[COUNTRY_COLUMN] = ""
            row[RANK_COLUMN] = ""
            continue

        key = (
            row.get("Conference", "").strip(),
            year,
            normalize_title(row.get(TITLE_COLUMN, "")),
        )
        country, rank = old_output_lookup.get(key, ("", ""))
        if country or rank:
            matched += 1

        row[COUNTRY_COLUMN] = country
        row[RANK_COLUMN] = rank

    return matched


def write_score_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with INPUT_CSV.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, fieldnames = load_score_rows()
    target_pairs = target_conference_year_pairs(rows)
    old_output_lookup = load_old_output_lookup(target_pairs)
    matched = enrich_rows(rows, old_output_lookup)
    write_score_rows(rows, fieldnames)

    print(f"Loaded {len(old_output_lookup)} title lookup entries from OldOutputs.")
    print(f"Updated {len(rows)} rows in {INPUT_CSV}.")
    print(f"Rows with Country or Rank populated: {matched}.")


if __name__ == "__main__":
    main()
