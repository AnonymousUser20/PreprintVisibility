"""Fill NeurIPS 2025 OldOutputs institutes from OpenReview JSON, then map country/rank."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEURIPS_2025_DIR = PROJECT_ROOT / "OldOutputs" / "NeurIPS" / "NeurIPS 2025"
JSON_ROOT = PROJECT_ROOT / "ReviewsNeurIPS" / "NeurIPS 2025"
CACHE_PATH = NEURIPS_2025_DIR / "author_institute_cache.json"
TRACKS = ("oral", "poster", "spotlight")
CSV_BY_TRACK = {
    "oral": NEURIPS_2025_DIR / "oral.csv",
    "poster": NEURIPS_2025_DIR / "poster.csv",
    "spotlight": NEURIPS_2025_DIR / "spotlight.csv",
}


def load_build_module():
    build_path = NEURIPS_2025_DIR / "build_neurips2025_csvs.py"
    spec = importlib.util.spec_from_file_location("build_neurips2025_csvs", build_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {build_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_title(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def load_author_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    return {}


def save_author_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as fp:
        json.dump(cache, fp, ensure_ascii=False, indent=2)


def collect_submission_notes(json_root: Path) -> list[dict]:
    notes: list[dict] = []
    for track in TRACKS:
        track_dir = json_root / track
        if not track_dir.is_dir():
            continue
        for json_path in sorted(track_dir.glob("*.json")):
            with json_path.open(encoding="utf-8") as fp:
                payload = json.load(fp)
            for note in payload.get("notes", []):
                if not isinstance(note, dict):
                    continue
                content = note.get("content", {})
                if not isinstance(content, dict):
                    continue
                title = str(content.get("title", {}).get("value", "")).strip()
                authorids = content.get("authorids", {}).get("value", [])
                if not title or not isinstance(authorids, list):
                    continue
                notes.append(
                    {
                        "title": title,
                        "authorids": [str(aid).strip() for aid in authorids if str(aid).strip()],
                    }
                )
    return notes


def fetch_missing_authorids(build_mod, author_cache: dict[str, str], authorids: set[str]) -> None:
    missing = sorted(aid for aid in authorids if aid not in author_cache)
    if not missing:
        return

    print(f"Fetching {len(missing)} author profiles from OpenReview...")
    with ThreadPoolExecutor(max_workers=24) as executor:
        future_by_authorid = {
            executor.submit(build_mod.fetch_institutes_for_authorid, aid): aid
            for aid in missing
        }
        for future in tqdm(
            as_completed(future_by_authorid),
            total=len(future_by_authorid),
            desc="OpenReview profiles",
            unit="author",
        ):
            aid = future_by_authorid[future]
            try:
                author_cache[aid] = future.result()
            except Exception:
                author_cache[aid] = ""

    save_author_cache(author_cache)


def institutes_for_authorids(authorids: list[str], author_cache: dict[str, str]) -> str:
    per_author: list[str] = []
    for aid in authorids:
        institute_text = author_cache.get(aid, "").strip()
        per_author.append(institute_text if institute_text else "Not Found")
    return str(per_author)


def build_title_to_institutes(build_mod) -> dict[str, str]:
    notes = collect_submission_notes(JSON_ROOT)
    author_cache = load_author_cache()
    all_authorids = {aid for note in notes for aid in note["authorids"]}
    print(f"Submission notes: {len(notes)} | unique authorids: {len(all_authorids)}")
    fetch_missing_authorids(build_mod, author_cache, all_authorids)

    title_to_institutes: dict[str, str] = {}
    for note in notes:
        key = normalize_title(note["title"])
        institutes = institutes_for_authorids(note["authorids"], author_cache)
        title_to_institutes[key] = institutes
    return title_to_institutes


def update_csv_institutes(csv_path: Path, title_to_institutes: dict[str, str]) -> tuple[int, int]:
    with csv_path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "Institutes" not in fieldnames:
        fieldnames.append("Institutes")

    matched = 0
    institutes_filled = 0
    for row in rows:
        key = normalize_title(row.get("Title", ""))
        institutes = title_to_institutes.get(key, "")
        if institutes:
            matched += 1
            row["Institutes"] = institutes
            if institutes not in ("[]", ""):
                institutes_filled += 1

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return matched, institutes_filled


def map_country_and_rank(csv_path: Path) -> dict[str, int]:
    sys.path.insert(0, str(PROJECT_ROOT / "PipelineFiles"))
    from update_oldoutputs_country_from_institutes import (
        build_batch_lookup,
        build_country_aliases,
        build_institutions_lookup,
        country_from_lookup_fallback,
        find_institute_column,
        normalize_text,
        split_institutes,
    )
    from update_oldoutputs_ranks_from_lookup import (
        load_rank_lookup,
        match_rank,
        parse_institutes,
    )

    lookup_dir = PROJECT_ROOT / "lookup"
    country_alias_to_name, country_by_alpha2_region, country_by_alpha2 = build_country_aliases(
        lookup_dir / "countries.csv"
    )
    batch_lookup = build_batch_lookup(
        PROJECT_ROOT / "OldOutputs" / "institute_batches", country_alias_to_name
    )
    institution_lookup = build_institutions_lookup(lookup_dir / "institutions.csv")
    rank_lookup = load_rank_lookup(PROJECT_ROOT / "lookup_rank" / "2025_cs_ranks.csv")

    with csv_path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for column in ("Country", "Rank_first", "Rank_last"):
        if column not in fieldnames:
            fieldnames.append(column)

    institute_col = find_institute_column(fieldnames)
    country_filled = rank_first_filled = rank_last_filled = both_rank_filled = 0

    for row in rows:
        institutes = parse_institutes(row.get(institute_col, "")) if institute_col else []
        countries: list[str] = []
        for inst in institutes:
            country = batch_lookup.get(normalize_text(inst), "")
            if not country:
                country = country_from_lookup_fallback(
                    affiliation=inst,
                    institution_lookup=institution_lookup,
                    country_by_alpha2_region=country_by_alpha2_region,
                    country_by_alpha2=country_by_alpha2,
                ) or ""
            if country:
                countries.append(country)
        row["Country"] = "; ".join(dict.fromkeys(countries))
        if row["Country"].strip():
            country_filled += 1

        first_inst = institutes[0] if institutes else ""
        last_inst = institutes[-1] if institutes else ""
        row["Rank_first"] = match_rank(first_inst, rank_lookup) if first_inst else ""
        row["Rank_last"] = match_rank(last_inst, rank_lookup) if last_inst else ""
        if row["Rank_first"].strip():
            rank_first_filled += 1
        if row["Rank_last"].strip():
            rank_last_filled += 1
        if row["Rank_first"].strip() and row["Rank_last"].strip():
            both_rank_filled += 1

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "rows": len(rows),
        "country_filled": country_filled,
        "rank_first_filled": rank_first_filled,
        "rank_last_filled": rank_last_filled,
        "both_rank_filled": both_rank_filled,
    }


def main() -> None:
    if not JSON_ROOT.is_dir():
        raise FileNotFoundError(f"NeurIPS 2025 JSON root not found: {JSON_ROOT}")

    build_mod = load_build_module()
    title_to_institutes = build_title_to_institutes(build_mod)
    print(f"Titles with institutes from OpenReview JSON: {len(title_to_institutes)}")

    for track, csv_path in CSV_BY_TRACK.items():
        if not csv_path.exists():
            print(f"Skipping missing CSV: {csv_path}")
            continue
        matched, institutes_filled = update_csv_institutes(csv_path, title_to_institutes)
        stats = map_country_and_rank(csv_path)
        print(
            f"\n{track} ({csv_path.name})"
            f"\n  title matches: {matched}/{stats['rows']}"
            f"\n  institutes filled: {institutes_filled}/{stats['rows']}"
            f"\n  country filled: {stats['country_filled']}/{stats['rows']}"
            f"\n  rank_first filled: {stats['rank_first_filled']}/{stats['rows']}"
            f"\n  rank_last filled: {stats['rank_last_filled']}/{stats['rows']}"
            f"\n  both ranks filled: {stats['both_rank_filled']}/{stats['rows']}"
        )


if __name__ == "__main__":
    main()
