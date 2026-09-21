import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEADLINES_PATH = PROJECT_ROOT / "ConfusionMatrix" / "conference_deadlines.csv"
OLD_ARXIV_ROOTS = {
    "ICLR": PROJECT_ROOT / "OldArXiv" / "ICLR",
    "ICML": PROJECT_ROOT / "OldArXiv" / "ICML",
    "NeurIPS": PROJECT_ROOT / "OldArXiv" / "NeurIPS",
}
REVIEW_ROOTS = {
    "ICLR": PROJECT_ROOT / "ReviewsICLR" / "Reviews",
    "ICML": PROJECT_ROOT / "ReviewsICML" / "Reviews",
    "NeurIPS": PROJECT_ROOT / "ReviewsNeurIPS" / "Reviews",
}
EXCLUDED_FOLDER_NAMES = {
    "withdrawn",
    "withdrawn_rejected",
    "rejected",
    "reject",
    "desk_rejected",
    "dataset_and_benchmark",
    "dataset_and_benchmarks",
    "datasets_and_benchmark",
    "datasets_and_benchmarks",
    "retracted",
    "retracted_acceptance",
    "submitted",
}
FIELDNAMES = ["Paper Title", "Conference", "Year", "rating", "confidence", "Group"]


def normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value


def normalize_title(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def parse_date(value: str) -> datetime | None:
    value = str(value).strip()
    if not value or value.lower() == "not found":
        return None

    value = re.sub(r"\s*\(.*?\)\s*", "", value).strip()
    for date_format in ("%d.%m.%Y", "%d %B, %Y", "%B %Y"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    return None


def load_submission_deadlines() -> dict[tuple[str, int], datetime]:
    deadlines = {}
    with DEADLINES_PATH.open(newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            conference = row.get("Conference", "").strip()
            if conference not in OLD_ARXIV_ROOTS:
                continue

            year = int(row["Year"])
            deadline = parse_date(row["Submission Deadline"])
            if deadline:
                deadlines[(conference, year)] = deadline
    return deadlines


def load_arxiv_dates() -> dict[tuple[str, int, str], datetime]:
    arxiv_dates = {}
    for conference, root in OLD_ARXIV_ROOTS.items():
        if not root.is_dir():
            continue

        for csv_path in sorted(root.rglob("*_arxiv.csv")):
            year = year_from_path(csv_path)
            if year is None:
                continue

            with csv_path.open(newline="", encoding="utf-8-sig") as fp:
                for row in csv.DictReader(fp):
                    title = row.get("Title", "").strip()
                    arxiv_date = parse_date(row.get("Submission Date", ""))
                    if not title or not arxiv_date:
                        continue

                    key = (conference, year, normalize_title(title))
                    current = arxiv_dates.get(key)
                    if current is None or arxiv_date < current:
                        arxiv_dates[key] = arxiv_date
    return arxiv_dates


def build_group_lookup() -> dict[tuple[str, int, str], str]:
    deadlines = load_submission_deadlines()
    arxiv_dates = load_arxiv_dates()
    groups = {}

    for key, arxiv_date in arxiv_dates.items():
        conference, year, title = key
        deadline = deadlines.get((conference, year))
        if not deadline:
            continue

        group = "Preprint" if arxiv_date < deadline else "No Preprint"
        groups[(conference, year, title)] = group

    return groups


def should_skip_path(path: Path) -> bool:
    normalized_parts = {normalize_name(part) for part in path.parts}
    return any(part in EXCLUDED_FOLDER_NAMES for part in normalized_parts)


def year_from_path(path: Path) -> int | None:
    match = re.search(r"\b(2023|2024|2025)\b", str(path))
    return int(match.group(1)) if match else None


def unwrap_openreview_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def extract_numeric_score(value) -> float | None:
    value = unwrap_openreview_value(value)
    if value is None:
        return None

    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def invitation_text(note: dict) -> str:
    invitations = note.get("invitations") or []
    invitation = note.get("invitation")
    return " ".join(str(item) for item in [*invitations, invitation or ""])


def extract_title(notes: list[dict], fallback: str) -> str:
    candidates = []
    for note in notes:
        content = note.get("content", {})
        title = unwrap_openreview_value(content.get("title"))
        if title:
            title = re.sub(r"\s+", " ", str(title)).strip()
            text = invitation_text(note)
            is_discussion_note = any(
                token in text
                for token in ("Official_Comment", "Official_Review", "Meta_Review", "Decision")
            )
            has_venue = "venue" in content or "venueid" in content
            is_submission = "Submission" in text or "Blind_Submission" in text
            candidates.append((title, has_venue, is_submission, is_discussion_note))

    for title, has_venue, _, is_discussion_note in candidates:
        if has_venue and not is_discussion_note:
            return title

    for title, _, is_submission, is_discussion_note in candidates:
        if is_submission and not is_discussion_note:
            return title

    for title, _, _, is_discussion_note in candidates:
        if not is_discussion_note:
            return title

    return fallback


def is_official_review(note: dict) -> bool:
    content = note.get("content", {})
    has_score = any(
        field in content
        for field in ("rating", "recommendation", "overall_recommendation")
    )
    has_confidence = "confidence" in content
    if not (has_score or has_confidence):
        return False

    if "Official_Review" in invitation_text(note):
        return True

    # Older OpenReview dumps in this project do not always include invitation metadata.
    return has_score and has_confidence


def mean_or_blank(values: list[float]) -> float | str:
    if not values:
        return ""
    return round(sum(values) / len(values), 6)


def extract_scores(json_path: Path, conference: str) -> dict | None:
    year = year_from_path(json_path)
    if year is None or should_skip_path(json_path):
        return None

    with json_path.open(encoding="utf-8") as fp:
        data = json.load(fp)

    notes = data.get("notes", [])
    if not isinstance(notes, list):
        return None

    ratings = []
    confidences = []
    for note in notes:
        if not isinstance(note, dict) or not is_official_review(note):
            continue

        content = note.get("content", {})
        rating = extract_numeric_score(
            content.get(
                "rating",
                content.get("recommendation", content.get("overall_recommendation")),
            )
        )
        confidence = extract_numeric_score(content.get("confidence"))
        if rating is not None:
            ratings.append(rating)
        if confidence is not None:
            confidences.append(confidence)

    if not ratings and not confidences:
        return None

    return {
        "Paper Title": extract_title(notes, json_path.stem),
        "Conference": conference,
        "Year": year,
        "rating": mean_or_blank(ratings),
        "confidence": mean_or_blank(confidences),
    }


def collect_rows() -> list[dict]:
    rows = []
    seen = set()
    group_lookup = build_group_lookup()

    for conference, review_root in REVIEW_ROOTS.items():
        if not review_root.is_dir():
            print(f"Skipping missing review folder: {review_root}")
            continue

        for json_path in sorted(review_root.rglob("*.json")):
            try:
                row = extract_scores(json_path, conference)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Skipping unreadable file {json_path}: {exc}")
                continue

            if not row:
                continue

            dedupe_key = (
                row["Conference"],
                row["Year"],
                normalize_name(row["Paper Title"]),
            )
            if dedupe_key in seen:
                continue

            row["Group"] = group_lookup.get(
                (
                    row["Conference"],
                    row["Year"],
                    normalize_title(row["Paper Title"]),
                ),
                "No Preprint",
            )

            seen.add(dedupe_key)
            rows.append(row)

    rows.sort(key=lambda item: (item["Conference"], item["Year"], item["Paper Title"].lower()))
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract average review rating and confidence by paper."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper_review_scores.csv",
        help="CSV output path. Defaults to paper_review_scores.csv in the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows()
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
