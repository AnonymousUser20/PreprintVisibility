import ast
import csv
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
INPUT_CSV = BASE / "unmapped_affiliations_frequencies_with_country.csv"
OUTPUT_CSV = BASE / "unmapped_affiliations_frequencies.csv"


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


def main() -> None:
    counter: Counter[str] = Counter()

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("Country", "")).strip():
                continue
            for affiliation in parse_affiliations(row.get("affiliations", "")):
                counter[affiliation] += 1

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["affiliations", "frequency"])
        for affiliation, frequency in sorted(counter.items(), key=lambda x: (-x[1], x[0].lower())):
            writer.writerow([affiliation, frequency])

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Distinct unmapped affiliations: {len(counter)}")
    print(f"Total unmapped affiliation occurrences: {sum(counter.values())}")


if __name__ == "__main__":
    main()
