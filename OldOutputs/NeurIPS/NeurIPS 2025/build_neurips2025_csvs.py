from __future__ import annotations

import csv
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

OPENREVIEW_PROFILE_API = "https://api.openreview.net/profiles"
TRACKS = ("oral", "poster", "reject", "spotlight")


def pick_institution_names(profile_content: dict) -> list[str]:
    names: list[str] = []

    history = profile_content.get("history", [])
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            institution = item.get("institution", {})
            if isinstance(institution, dict):
                name = str(institution.get("name", "")).strip()
                if name:
                    names.append(name)

    relations = profile_content.get("relations", [])
    if isinstance(relations, list):
        for item in relations:
            if not isinstance(item, dict):
                continue
            relation = item.get("relation", {})
            if isinstance(relation, dict):
                name = str(relation.get("name", "")).strip()
                if name:
                    names.append(name)

    deduped: list[str] = []
    seen = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def fetch_institutes_for_authorid(authorid: str) -> str:
    authorid = str(authorid).strip()
    if not authorid:
        return ""

    try:
        response = requests.get(
            OPENREVIEW_PROFILE_API, params={"id": authorid}, timeout=20
        )
        response.raise_for_status()
        profiles = response.json().get("profiles", [])
    except Exception:
        return ""

    if not profiles:
        return ""

    content = profiles[0].get("content", {})
    if not isinstance(content, dict):
        return ""

    institutes = pick_institution_names(content)
    return "; ".join(institutes)


def parse_notes_from_json(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes = data.get("notes", [])
    return notes if isinstance(notes, list) else []


def build_track_csv(track_dir: Path, author_cache: dict[str, str]) -> Path:
    rows: list[dict[str, str]] = []
    json_files = sorted(track_dir.glob("*.json"))
    print(f"\nTrack: {track_dir.name} | JSON files: {len(json_files)}")

    all_authorids: set[str] = set()

    for json_file in tqdm(
        json_files,
        desc=f"{track_dir.name}: scan JSON",
        unit="file",
    ):
        notes = parse_notes_from_json(json_file)
        for note in notes:
            if not isinstance(note, dict):
                continue
            content = note.get("content", {})
            if not isinstance(content, dict):
                continue

            title = str(content.get("title", {}).get("value", "")).strip()
            authors = content.get("authors", {}).get("value", [])
            authorids = content.get("authorids", {}).get("value", [])

            if not isinstance(authors, list):
                authors = []
            if not isinstance(authorids, list):
                authorids = []

            author_names = [str(a).strip() for a in authors if str(a).strip()]
            author_ids = [str(aid).strip() for aid in authorids if str(aid).strip()]
            all_authorids.update(author_ids)

    missing_authorids = [aid for aid in all_authorids if aid not in author_cache]
    print(
        f"Track: {track_dir.name} | unique authorids: {len(all_authorids)} | "
        f"to fetch: {len(missing_authorids)}"
    )

    if missing_authorids:
        with ThreadPoolExecutor(max_workers=24) as executor:
            future_by_authorid = {
                executor.submit(fetch_institutes_for_authorid, aid): aid
                for aid in missing_authorids
            }
            for future in tqdm(
                as_completed(future_by_authorid),
                total=len(future_by_authorid),
                desc=f"{track_dir.name}: fetch OpenReview profiles",
                unit="author",
            ):
                aid = future_by_authorid[future]
                try:
                    author_cache[aid] = future.result()
                except Exception:
                    author_cache[aid] = ""

    for json_file in tqdm(
        json_files,
        desc=f"{track_dir.name}: build rows",
        unit="file",
    ):
        notes = parse_notes_from_json(json_file)
        for note in notes:
            if not isinstance(note, dict):
                continue
            content = note.get("content", {})
            if not isinstance(content, dict):
                continue

            title = str(content.get("title", {}).get("value", "")).strip()
            authors = content.get("authors", {}).get("value", [])
            authorids = content.get("authorids", {}).get("value", [])

            if not isinstance(authors, list):
                authors = []
            if not isinstance(authorids, list):
                authorids = []

            author_names = [str(a).strip() for a in authors if str(a).strip()]
            author_ids = [str(aid).strip() for aid in authorids if str(aid).strip()]
            institutes_per_author: list[str] = []
            for aid in author_ids:
                institute_text = author_cache.get(aid, "")
                if institute_text:
                    institutes_per_author.append(institute_text)

            rows.append(
                {
                    "Title": title,
                    "Authors": "; ".join(author_names),
                    "Institutes": " | ".join(institutes_per_author),
                }
            )

    output_csv = track_dir / f"{track_dir.name}.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Title", "Authors", "Institutes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_csv} | rows: {len(rows)}")
    return output_csv


def main() -> None:
    root = Path(__file__).resolve().parent
    author_cache: dict[str, str] = {}

    for track in TRACKS:
        track_dir = root / track
        if not track_dir.exists() or not track_dir.is_dir():
            print(f"Skipping missing track folder: {track_dir}")
            continue
        build_track_csv(track_dir=track_dir, author_cache=author_cache)

    print(f"\nDone. Cached OpenReview authorids: {len(author_cache)}")


if __name__ == "__main__":
    main()
