from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SERPER_URL = "https://google.serper.dev/search"
NULL_LIKE = {"", "nan", "none", "null", "na"}
MIN_TITLE_MATCH_RATIO = 0.60
# Serper sometimes puts author-year snippets in `date` instead of a publication date.
AUTHOR_YEAR_SNIPPET = re.compile(r"^by\s+.+\s·\s*\d{4}$", re.IGNORECASE)


def normalize_title_from_filename(file_name: str) -> str:
    cleaned = str(file_name).strip()
    if not cleaned:
        return ""
    return re.sub(r"\.pdf$", "", cleaned, flags=re.IGNORECASE).strip()


def normalize_for_similarity(text: str) -> str:
    cleaned = str(text).strip().lower()
    cleaned = re.sub(r"[^a-z0-9\\s]", " ", cleaned)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    return cleaned


def title_similarity(a: str, b: str) -> float:
    a_norm = normalize_for_similarity(a)
    b_norm = normalize_for_similarity(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def is_usable_date(value: str) -> bool:
    date_value = str(value).strip()
    if date_value.lower() in NULL_LIKE:
        return False
    if AUTHOR_YEAR_SNIPPET.match(date_value):
        return False
    return True


def fetch_arxiv_date_from_title(title: str, api_key: str, timeout_seconds: int = 30) -> str:
    query = normalize_title_from_filename(title)
    if not query:
        return ""

    payload = json.dumps({"q": f"{query} site:arxiv.org"})
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            SERPER_URL, headers=headers, data=payload, timeout=timeout_seconds
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return ""

    for item in data.get("organic", []):
        link = str(item.get("link", "")).lower()
        result_title = str(item.get("title", "")).strip()
        similarity = title_similarity(query, result_title)
        if "arxiv.org" not in link or similarity < MIN_TITLE_MATCH_RATIO:
            continue
        date_value = str(item.get("date", "")).strip()
        if is_usable_date(date_value):
            return date_value
    return ""


def write_rows(output_csv: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    if "Date" not in fieldnames:
        fieldnames.append("Date")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_input_csv(input_csv: Path, api_key: str, delay_seconds: float) -> Path:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Input CSV has no rows: {input_csv}")
    if "file_name" not in rows[0]:
        raise ValueError(f"'file_name' column not found in: {input_csv}")

    title_cache: dict[str, str] = {}
    searched = 0
    for row in tqdm(rows, desc=f"Searching arXiv date: {input_csv.name}", unit="row"):
        title = normalize_title_from_filename(row.get("file_name", ""))
        if not title:
            row["Date"] = ""
            continue

        existing_date = str(row.get("Date", "")).strip()
        if title in title_cache:
            row["Date"] = title_cache[title]
            continue
        if is_usable_date(existing_date):
            title_cache[title] = existing_date
            continue

        date_value = fetch_arxiv_date_from_title(title=title, api_key=api_key)
        title_cache[title] = date_value
        row["Date"] = date_value or ""
        searched += 1
        if date_value:
            message = f"Scraped date: {date_value} | title: {title}"
            try:
                print(message)
            except UnicodeEncodeError:
                print(message.encode("ascii", "replace").decode("ascii"))

        if delay_seconds > 0:
            time.sleep(delay_seconds)
        if searched % 25 == 0:
            write_rows(input_csv, rows)

    write_rows(input_csv, rows)

    found = sum(1 for row in rows if str(row.get("Date", "")).strip())
    print(f"Processed: {input_csv.name}")
    print(f"Unique titles searched: {len(title_cache)}")
    print(f"Dates found: {found}/{len(rows)}")
    print(f"Saved: {input_csv}")
    return input_csv


def parse_args() -> argparse.Namespace:
    pipeline_dir = Path(__file__).resolve().parent
    project_root = pipeline_dir.parent
    parser = argparse.ArgumentParser(
        description="Map arXiv dates for *mapped_ranked.csv files using Serper results."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=project_root,
        help="Base directory containing *mapped_ranked.csv files.",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit *mapped_ranked.csv files. If omitted, auto-discovers in base-dir.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("SERPER_API_KEY", ""),
        help="Serper API key. Defaults to SERPER_API_KEY from .env or the environment.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.2,
        help="Delay between unique API requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not str(args.api_key).strip():
        raise ValueError(
            "Missing Serper API key. Put SERPER_API_KEY in the project .env, "
            "set the environment variable, or pass --api-key."
        )

    if args.inputs:
        input_files = args.inputs
    else:
        input_files = sorted(args.base_dir.glob("*mapped_ranked.csv"))

    if not input_files:
        raise FileNotFoundError(f"No *mapped_ranked.csv files found in: {args.base_dir}")

    for input_csv in input_files:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        process_input_csv(
            input_csv=input_csv,
            api_key=args.api_key,
            delay_seconds=args.delay_seconds,
        )

    print("arXiv mapping pipeline complete.")


if __name__ == "__main__":
    main()