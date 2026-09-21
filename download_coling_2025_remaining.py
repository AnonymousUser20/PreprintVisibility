"""Download remaining COLING 2025 main-track PDFs from the ACL Anthology."""

from __future__ import annotations

import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = (
    PROJECT_ROOT
    / "Outputs"
    / "coling_2025_main_remaining_author_output_country_mapped_ranked_preprint.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "COLING 2025 Remaining"
FAILED_LOG = OUTPUT_DIR / "_failed_downloads.csv"
MAX_WORKERS = 6
TIMEOUT_SECONDS = 90
RETRIES = 4
REQUEST_HEADERS = {
    "User-Agent": "COLING2025RemainingScraper/1.0 (academic research)",
    "Accept": "application/pdf,*/*",
}


def anthology_pdf_url(page_url: str) -> str:
    url = str(page_url).strip().rstrip("/")
    if not url:
        return ""
    if url.lower().endswith(".pdf"):
        return url
    return url + ".pdf"


def safe_pdf_name(name: str, out_dir: Path) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', "", str(name).strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    max_len = max(80, 240 - len(str(out_dir)))
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 4].rstrip() + ".pdf"
    return cleaned


def load_papers() -> list[tuple[str, str]]:
    papers: dict[str, str] = {}
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = str(row.get("anthology_url", "")).strip()
            file_name = str(row.get("file_name", "")).strip()
            if url and url not in papers:
                papers[url] = file_name
    return list(papers.items())


def download_one(page_url: str, dest: Path) -> tuple[str, str]:
    pdf_url = anthology_pdf_url(page_url)
    if not pdf_url:
        return page_url, "missing anthology url"
    if dest.exists() and dest.stat().st_size > 1000:
        return page_url, "ok"
    last_error = "unknown error"
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(pdf_url, headers=REQUEST_HEADERS, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            content_type = str(response.headers.get("Content-Type", "")).lower()
            body = response.content
            if "pdf" not in content_type and not body.startswith(b"%PDF"):
                last_error = f"not a pdf ({content_type[:80]})"
            elif len(body) < 1000:
                last_error = f"too small ({len(body)} bytes)"
            else:
                dest.write_bytes(body)
                return page_url, "ok"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(min(2 ** attempt, 8))
    return page_url, last_error


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers = load_papers()
    jobs = [
        (page_url, OUTPUT_DIR / safe_pdf_name(file_name, OUTPUT_DIR))
        for page_url, file_name in papers
    ]
    already = sum(1 for _, dest in jobs if dest.exists() and dest.stat().st_size > 1000)
    print(f"Papers: {len(jobs)}")
    print(f"Already downloaded: {already}")
    print(f"Folder: {OUTPUT_DIR}")

    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_one, page_url, dest): (page_url, dest)
            for page_url, dest in jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDFs", unit="pdf"):
            page_url, status = future.result()
            if status != "ok":
                failed.append((page_url, status))

    if failed:
        with open(FAILED_LOG, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["anthology_url", "error"])
            writer.writerows(failed)
    elif FAILED_LOG.exists():
        FAILED_LOG.unlink()

    downloaded = sum(1 for _, dest in jobs if dest.exists() and dest.stat().st_size > 1000)
    print(f"Downloaded: {downloaded}/{len(jobs)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failure log: {FAILED_LOG}")


if __name__ == "__main__":
    main()
