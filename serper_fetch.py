"""Fetch remaining COLING 2025 main-track papers from the ACL Anthology.

Scrapes only papers not already in the existing COLING 2025 main CSVs.
Uses SERPER_API_KEY from the project .env (never a hardcoded key) to look up
arXiv dates. Writes a separate Outputs CSV using the project naming convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

EVENTS_URL = "https://aclanthology.org/events/coling-2025/"
MAIN_VOLUME_URL = "https://aclanthology.org/volumes/2025.coling-main/"
MAIN_VOLUME_XML = "https://aclanthology.org/volumes/2025.coling-main.xml"
SERPER_URL = "https://google.serper.dev/search"
CONFERENCE_NAME = "Proceedings of the 31st International Conference on Computational Linguistics"

EXISTING_MAIN_CSVS = [
    PROJECT_ROOT / "InputFiles" / "coling_2025_main_author_output.csv",
    PROJECT_ROOT
    / "Outputs"
    / "coling_2025_main_author_output_country_mapped_ranked_preprint.csv",
]
OUTPUT_CSV = (
    PROJECT_ROOT
    / "Outputs"
    / "coling_2025_main_remaining_author_output_country_mapped_ranked_preprint.csv"
)

NULL_LIKE = {"", "nan", "none", "null", "na"}
MIN_TITLE_MATCH_RATIO = 0.60
FRONT_MATTER_PREFIX = "proceedings of the 31st international conference"
NS = {"m": "http://www.loc.gov/mods/v3"}
REQUEST_HEADERS = {
    "User-Agent": "COLING2025RemainingScraper/1.0 (academic research)"
}


def normalize_title(text: object) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def title_similarity(a: str, b: str) -> float:
    a_norm = normalize_title(a)
    b_norm = normalize_title(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def clean_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:180]


def author_full_name(name_el: ET.Element) -> str:
    given = [
        str(part.text or "").strip()
        for part in name_el.findall("m:namePart[@type='given']", NS)
        if part.text
    ]
    family = [
        str(part.text or "").strip()
        for part in name_el.findall("m:namePart[@type='family']", NS)
        if part.text
    ]
    return " ".join([*given, *family]).strip()


def is_front_matter(mods_el: ET.Element) -> bool:
    roles = {
        str(role.text or "").strip().lower()
        for role in mods_el.findall(".//m:roleTerm", NS)
    }
    return "author" not in roles


def parse_main_volume_papers(xml_text: str) -> list[dict[str, object]]:
    root = ET.fromstring(xml_text)
    papers: list[dict[str, object]] = []
    for mods in root.findall("m:mods", NS):
        if is_front_matter(mods):
            continue
        title_el = mods.find("m:titleInfo/m:title", NS)
        title = str(title_el.text or "").strip() if title_el is not None else ""
        if not title or normalize_title(title).startswith(FRONT_MATTER_PREFIX):
            continue
        authors = []
        for name_el in mods.findall("m:name", NS):
            name_roles = {
                str(role.text or "").strip().lower()
                for role in name_el.findall(".//m:roleTerm", NS)
            }
            if "author" not in name_roles:
                continue
            full = author_full_name(name_el)
            if full:
                authors.append(full)
        url_el = mods.find("m:location/m:url", NS)
        anthology_url = str(url_el.text or "").strip() if url_el is not None else ""
        papers.append(
            {
                "title": title,
                "authors": authors,
                "anthology_url": anthology_url,
            }
        )
    return papers


def fetch_main_volume_xml() -> str:
    # Events page is the user-facing entry; the main-track volume XML is the
    # structured source for the 2025.coling-main papers listed there.
    response = requests.get(MAIN_VOLUME_XML, headers=REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    return response.text


def is_already_scraped(title: str, scraped_norms: set[str]) -> bool:
    norm = normalize_title(title)
    if not norm:
        return False
    if norm in scraped_norms:
        return True
    # Existing PDFs were often truncated to ~100 filename characters.
    for scraped in scraped_norms:
        if len(scraped) < 30:
            continue
        if norm.startswith(scraped) or scraped.startswith(norm):
            return True
    return False


def load_already_scraped_titles() -> set[str]:
    scraped: set[str] = set()
    for csv_path in EXISTING_MAIN_CSVS:
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                file_name = str(row.get("file_name", "")).strip()
                paper_title = str(row.get("paper_title", "")).strip()
                for raw in (file_name, paper_title):
                    norm = normalize_title(raw)
                    if norm:
                        scraped.add(norm)
    return scraped


def authorship_role(index: int, n_authors: int) -> str:
    if n_authors == 1:
        return "FIRST_AUTHOR / LAST_AUTHOR"
    if index == 0:
        return "FIRST_AUTHOR"
    if index == n_authors - 1:
        return "LAST_AUTHOR"
    return "CO_AUTHOR"


def fetch_arxiv_date_from_title(title: str, api_key: str, timeout_seconds: int = 30) -> str:
    query = str(title).strip()
    if not query:
        return ""
    payload = json.dumps({"q": query})
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
    except Exception as exc:
        tqdm.write(f"Serper error for {title[:80]!r}: {exc}")
        return ""

    for item in data.get("organic", []):
        link = str(item.get("link", ""))
        result_title = str(item.get("title", "")).strip()
        if "arxiv" in link.lower() and title_similarity(query, result_title) >= MIN_TITLE_MATCH_RATIO:
            date_value = str(item.get("date", "")).strip()
            if date_value.lower() in NULL_LIKE:
                return ""
            return date_value
    return ""


def resolve_api_key(cli_key: str) -> str:
    if cli_key.strip():
        return cli_key.strip()
    return os.getenv("SERPER_API_KEY", "").strip()


def write_rows(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "author",
        "affiliations",
        "authorship_role",
        "file_name",
        "conference_name",
        "paper_title",
        "Country",
        "rank",
        "Date",
        "anthology_url",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape remaining COLING 2025 main-track papers from the ACL Anthology "
            f"({EVENTS_URL}) and look up arXiv dates with Serper."
        )
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Serper API key. Defaults to SERPER_API_KEY in .env. Do not hardcode keys.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.2,
        help="Delay between Serper requests.",
    )
    parser.add_argument(
        "--skip-serper",
        action="store_true",
        help="Write remaining papers without calling Serper (Date left blank).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Events page: {EVENTS_URL}")
    print(f"Main-track volume: {MAIN_VOLUME_URL}")

    xml_text = fetch_main_volume_xml()
    all_papers = parse_main_volume_papers(xml_text)
    already = load_already_scraped_titles()
    remaining = [
        paper
        for paper in all_papers
        if not is_already_scraped(str(paper["title"]), already)
    ]

    print(f"Anthology main-track papers: {len(all_papers)}")
    print(f"Already scraped: {len(all_papers) - len(remaining)}")
    print(f"Remaining to fetch: {len(remaining)}")

    api_key = ""
    if not args.skip_serper:
        api_key = resolve_api_key(args.api_key)
        if not api_key:
            raise ValueError(
                "Missing Serper API key. Put SERPER_API_KEY in the project .env "
                "or pass --api-key. The hardcoded key in older versions of this "
                "script is not used."
            )

    rows: list[dict[str, str]] = []
    for paper in tqdm(remaining, desc="Remaining COLING 2025 main", unit="paper"):
        title = str(paper["title"])
        authors = list(paper["authors"]) if isinstance(paper["authors"], list) else []
        date_value = ""
        if api_key:
            date_value = fetch_arxiv_date_from_title(title, api_key)
            time.sleep(args.delay_seconds)

        n_authors = len(authors) or 1
        if not authors:
            authors = [""]
        file_name = clean_filename(title) + ".pdf"
        for index, author in enumerate(authors):
            rows.append(
                {
                    "author": str(author),
                    "affiliations": "",
                    "authorship_role": authorship_role(index, n_authors),
                    "file_name": file_name,
                    "conference_name": CONFERENCE_NAME,
                    "paper_title": title,
                    "Country": "",
                    "rank": "",
                    "Date": date_value,
                    "anthology_url": str(paper.get("anthology_url", "")),
                }
            )

    write_rows(rows, args.output_csv)
    n_papers = len(remaining)
    n_with_date = len({row["file_name"] for row in rows if row["Date"]})
    print(f"Saved: {args.output_csv}")
    print(f"Author rows: {len(rows)}")
    print(f"Papers: {n_papers}")
    print(f"Papers with arXiv date: {n_with_date}")


if __name__ == "__main__":
    main()
