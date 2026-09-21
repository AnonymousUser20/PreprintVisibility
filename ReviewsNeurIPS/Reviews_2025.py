import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

BASE_DIR = Path(__file__).resolve().parent
INPUT_ROOT = BASE_DIR / "NeurIPS 2025"
OUTPUT_ROOT = BASE_DIR / "Reviews" / "NeurIPS 2025"

OPENREVIEW_API = (
    "https://api2.openreview.net/notes"
    "?count=true"
    "&details=writable%2Csignatures%2Cinvitation%2Cpresentation%2Ctags"
    "&domain={domain}"
    "&forum={forum_id}"
    "&limit=1000"
    "&trash=true"
)

REQUEST_DELAY_SECONDS = 1.0
RATE_LIMIT_WAIT_SECONDS = 10.0
DEFAULT_DOMAIN = "NeurIPS.cc/2025/Conference"


def _title_for_filename(note: dict, fallback: str) -> str:
    raw = note.get("content", {}).get("title", {}).get("value", "")
    title = str(raw).strip()
    if not title:
        return fallback
    title = re.sub(r"\s+", " ", title)
    for char in r'\/:*?"<>|':
        title = title.replace(char, "_")
    title = title.strip(" .")
    if len(title) > 180:
        title = title[:180].rstrip(" .")
    return title or fallback


def _domain_from_note(note: dict) -> str:
    domain = str(note.get("domain", "")).strip()
    if domain:
        return domain
    venueid = note.get("content", {}).get("venueid", {}).get("value", "")
    if venueid:
        return str(venueid).strip()
    return DEFAULT_DOMAIN


def _unique_output_path(out_dir: Path, stem: str, forum_id: str) -> Path:
    path = out_dir / f"{stem}.json"
    if not path.exists():
        return path
    alt = out_dir / f"{stem}__{forum_id}.json"
    if not alt.exists():
        return alt
    n = 2
    while True:
        candidate = out_dir / f"{stem}__{forum_id}_{n}.json"
        if not candidate.exists():
            return candidate
        n += 1


def _fetch_reviews(forum_id: str, domain: str) -> dict:
    domain_encoded = quote(domain, safe="")
    url = OPENREVIEW_API.format(domain=domain_encoded, forum_id=forum_id)
    response = requests.get(url, timeout=60)

    if response.status_code == 429:
        print("Rate limit hit. Waiting...")
        time.sleep(RATE_LIMIT_WAIT_SECONDS)
        response = requests.get(url, timeout=60)

    response.raise_for_status()
    return response.json()


def _page_sort_key(path: Path) -> tuple:
    match = re.search(r"page(\d+)", path.stem, re.IGNORECASE)
    return (int(match.group(1)) if match else 0, path.name)


def collect_neurips_2025_reviews() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input directory not found: {INPUT_ROOT}")

    category_dirs = sorted(
        d for d in INPUT_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not category_dirs:
        raise FileNotFoundError(f"No category folders found under: {INPUT_ROOT}")

    total_saved = 0
    total_skipped = 0

    for category_dir in category_dirs:
        out_dir = OUTPUT_ROOT / category_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        page_files = sorted(category_dir.glob("*.json"), key=_page_sort_key)
        if not page_files:
            print(f"No JSON files in {category_dir.name}, skipping.")
            continue

        print(f"\n{category_dir.name}: {len(page_files)} page file(s)")

        for page_path in page_files:
            with open(page_path, encoding="utf-8") as fp:
                content = json.load(fp)

            for note in content.get("notes", []):
                forum_id = note.get("id") or note.get("forum")
                if not forum_id:
                    print(f"  Skipping note without id in {page_path.name}")
                    continue

                stem = _title_for_filename(note, forum_id)
                out_path = _unique_output_path(out_dir, stem, forum_id)

                if out_path.exists():
                    total_skipped += 1
                    continue

                domain = _domain_from_note(note)
                print(f"  Fetching: {stem}")
                data = _fetch_reviews(forum_id=forum_id, domain=domain)

                with open(out_path, "w", encoding="utf-8") as fp:
                    json.dump(data, fp, ensure_ascii=False)

                total_saved += 1
                time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. Saved {total_saved} review file(s), skipped {total_skipped} existing.")


def main() -> None:
    collect_neurips_2025_reviews()


if __name__ == "__main__":
    main()
