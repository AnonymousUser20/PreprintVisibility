"""Count first/last author affiliations that fail rank matching due to naming variants."""

from __future__ import annotations

import ast
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rank_mapping_pipeline import parse_affiliations, rank_for_row  # noqa: E402
from update_oldoutputs_ranks_from_lookup import (  # noqa: E402
    first_last_from_authors_institutes,
    find_institute_column,
    infer_year_from_path,
    load_rank_lookup,
    match_rank,
    normalize_text,
    parse_institutes,
)

NULL_LIKE = {"", "nan", "none", "null", "na", "not found"}
SKIP_DIR_NAMES = {"institute_batches"}
SKIP_FILE_NAMES = {"rank_fill_summary.csv"}

STOP_TOKENS = {
    "a",
    "an",
    "and",
    "at",
    "de",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "ac",
    "ai",
    "cs",
    "lab",
    "labs",
    "department",
    "school",
    "faculty",
    "center",
    "centre",
    "division",
    "group",
    "research",
    "science",
    "engineering",
    "technology",
    "institute",
    "university",
    "college",
}


@dataclass(frozen=True)
class AffiliationHit:
    source: str
    file: str
    role: str
    raw: str
    year: int | None
    matched_rank: str
    recoverable_rank: str
    recoverable_key: str
    reason: str


def is_null(value: object) -> bool:
    text = str(value).strip()
    return not text or text.lower() in NULL_LIKE


def strip_parenthetical(raw: str) -> str:
    return re.sub(r"\([^)]*\)", " ", raw).strip()


def institute_candidates(raw: str) -> list[str]:
    """Generate cleaned variants of an affiliation string for recovery checks."""
    text = strip_parenthetical(raw).strip()
    if not text:
        return []

    candidates = [text]
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if parts:
            candidates.append(parts[-1])
        if len(parts) >= 2:
            candidates.append(parts[-2])
            candidates.append(", ".join(parts[-2:]))
    return list(dict.fromkeys(c for c in candidates if c))


def token_set(text: str) -> set[str]:
    tokens = set(normalize_text(text).split())
    return {t for t in tokens if t and t not in STOP_TOKENS and len(t) > 1}


def find_token_recoverable_match(
    raw: str, rank_lookup: dict[str, str]
) -> tuple[str, str, str]:
    """Return (rank, lookup_key, reason) if a naming-variant match is likely."""
    for candidate in institute_candidates(raw):
        rank = match_rank(candidate, rank_lookup)
        if rank:
            if candidate != raw:
                return rank, normalize_text(candidate), "cleaned_string"
            return rank, normalize_text(candidate), "direct"

    aff_tokens = token_set(raw)
    if not aff_tokens:
        return "", "", ""

    best: tuple[int, str, str] | None = None
    for key, rank in rank_lookup.items():
        key_tokens = token_set(key)
        if not key_tokens:
            continue
        overlap = aff_tokens & key_tokens
        if not overlap:
            continue
        if key_tokens <= aff_tokens:
            score = len(key_tokens) * 10 + len(overlap)
            if best is None or score > best[0]:
                best = (score, rank, key)

    if best is None:
        return "", "", ""
    return best[1], best[2], "token_subset"


def classify_affiliation(raw: str, rank_lookup: dict[str, str]) -> tuple[str, str, str, str]:
    matched = match_rank(raw, rank_lookup)
    if matched:
        return matched, "", "", "matched"

    recoverable_rank, recoverable_key, reason = find_token_recoverable_match(
        raw, rank_lookup
    )
    if recoverable_rank:
        return "", recoverable_rank, recoverable_key, reason
    return "", "", "", "unmatched"


def infer_year_from_outputs_path(path: Path) -> int | None:
    match = re.search(r"(20(23|24|25))", path.stem)
    return int(match.group(1)) if match else None


def process_oldoutputs(
    oldoutputs_dir: Path, rank_lookup_by_year: dict[int, dict[str, str]]
) -> list[AffiliationHit]:
    hits: list[AffiliationHit] = []

    for csv_path in sorted(oldoutputs_dir.glob("**/*.csv")):
        if csv_path.name in SKIP_FILE_NAMES:
            continue
        if any(part.lower() in SKIP_DIR_NAMES for part in csv_path.parts):
            continue

        year = infer_year_from_path(csv_path)
        if year not in rank_lookup_by_year:
            continue
        rank_lookup = rank_lookup_by_year[year]

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        institute_col = find_institute_column(list(rows[0].keys()))
        if not institute_col:
            continue

        for row in rows:
            raw_institutes = row.get(institute_col, "")
            if institute_col.strip().lower() == "authors_institutes":
                first_inst, last_inst = first_last_from_authors_institutes(raw_institutes)
            else:
                institutes = parse_institutes(raw_institutes)
                first_inst = institutes[0] if institutes else ""
                last_inst = institutes[-1] if institutes else ""

            for role, inst in (("first", first_inst), ("last", last_inst)):
                if is_null(inst):
                    continue
                matched, rec_rank, rec_key, reason = classify_affiliation(
                    inst, rank_lookup
                )
                if reason == "matched":
                    continue
                hits.append(
                    AffiliationHit(
                        source="OldOutputs",
                        file=str(csv_path.relative_to(oldoutputs_dir.parent)),
                        role=role,
                        raw=inst,
                        year=year,
                        matched_rank=matched,
                        recoverable_rank=rec_rank,
                        recoverable_key=rec_key,
                        reason=reason,
                    )
                )
    return hits


def process_outputs(
    outputs_dir: Path, rank_lookup_by_year: dict[int, dict[str, str]]
) -> list[AffiliationHit]:
    hits: list[AffiliationHit] = []

    for csv_path in sorted(outputs_dir.glob("**/*.csv")):
        year = infer_year_from_outputs_path(csv_path)
        if year not in rank_lookup_by_year:
            continue
        rank_lookup = rank_lookup_by_year[year]

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        papers: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
        for row in rows:
            role = str(row.get("authorship_role", "")).strip().upper()
            if role not in {"FIRST_AUTHOR", "LAST_AUTHOR"}:
                continue
            file_name = str(row.get("file_name", "")).strip()
            paper_title = str(row.get("paper_title", "")).strip()
            key = (file_name, paper_title)
            papers[key][role] = parse_affiliations(row.get("affiliations", ""))

        for (file_name, paper_title), roles in papers.items():
            for role_label, role_key in (
                ("first", "FIRST_AUTHOR"),
                ("last", "LAST_AUTHOR"),
            ):
                affiliations = roles.get(role_key, [])
                if not affiliations:
                    continue
                raw_joined = "; ".join(affiliations)
                current_rank = rank_for_row(str(affiliations), rank_lookup)
                if current_rank:
                    continue

                best_rec_rank = ""
                best_rec_key = ""
                best_reason = "unmatched"
                for aff in affiliations:
                    if is_null(aff):
                        continue
                    _, rec_rank, rec_key, reason = classify_affiliation(
                        aff, rank_lookup
                    )
                    if rec_rank and (
                        not best_rec_rank
                        or int(rec_rank) < int(best_rec_rank)
                    ):
                        best_rec_rank = rec_rank
                        best_rec_key = rec_key
                        best_reason = reason

                if best_rec_rank:
                    hits.append(
                        AffiliationHit(
                            source="Outputs",
                            file=str(csv_path.relative_to(outputs_dir.parent)),
                            role=role_label,
                            raw=raw_joined,
                            year=year,
                            matched_rank="",
                            recoverable_rank=best_rec_rank,
                            recoverable_key=best_rec_key,
                            reason=best_reason,
                        )
                    )
                else:
                    hits.append(
                        AffiliationHit(
                            source="Outputs",
                            file=str(csv_path.relative_to(outputs_dir.parent)),
                            role=role_label,
                            raw=raw_joined,
                            year=year,
                            matched_rank="",
                            recoverable_rank="",
                            recoverable_key="",
                            reason="unmatched",
                        )
                    )
    return hits


def count_total_affiliations(
    oldoutputs_dir: Path, outputs_dir: Path, rank_lookup_by_year: dict[int, dict[str, str]]
) -> tuple[int, int, int, int]:
    old_total = 0
    out_total = 0
    old_matched = 0
    out_matched = 0

    for csv_path in sorted(oldoutputs_dir.glob("**/*.csv")):
        if csv_path.name in SKIP_FILE_NAMES:
            continue
        if any(part.lower() in SKIP_DIR_NAMES for part in csv_path.parts):
            continue
        year = infer_year_from_path(csv_path)
        if year not in rank_lookup_by_year:
            continue
        rank_lookup = rank_lookup_by_year[year]
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        institute_col = find_institute_column(list(rows[0].keys())) if rows else None
        if not institute_col:
            continue
        for row in rows:
            raw_institutes = row.get(institute_col, "")
            if institute_col.strip().lower() == "authors_institutes":
                first_inst, last_inst = first_last_from_authors_institutes(raw_institutes)
            else:
                institutes = parse_institutes(raw_institutes)
                first_inst = institutes[0] if institutes else ""
                last_inst = institutes[-1] if institutes else ""
            for inst in (first_inst, last_inst):
                if is_null(inst):
                    continue
                old_total += 1
                if match_rank(inst, rank_lookup):
                    old_matched += 1

    for csv_path in sorted(outputs_dir.glob("**/*.csv")):
        year = infer_year_from_outputs_path(csv_path)
        if year not in rank_lookup_by_year:
            continue
        rank_lookup = rank_lookup_by_year[year]
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        papers: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
        for row in rows:
            role = str(row.get("authorship_role", "")).strip().upper()
            if role not in {"FIRST_AUTHOR", "LAST_AUTHOR"}:
                continue
            key = (
                str(row.get("file_name", "")).strip(),
                str(row.get("paper_title", "")).strip(),
            )
            papers[key][role] = parse_affiliations(row.get("affiliations", ""))
        for roles in papers.values():
            for role_key in ("FIRST_AUTHOR", "LAST_AUTHOR"):
                affiliations = roles.get(role_key, [])
                if not affiliations:
                    continue
                out_total += 1
                if rank_for_row(str(affiliations), rank_lookup):
                    out_matched += 1

    return old_total, old_matched, out_total, out_matched


def main() -> None:
    project_root = SCRIPT_DIR.parent
    lookup_rank_dir = project_root / "lookup_rank"
    oldoutputs_dir = project_root / "OldOutputs"
    outputs_dir = project_root / "Outputs"

    rank_lookup_by_year = {
        2023: load_rank_lookup(lookup_rank_dir / "2023_cs_ranks.csv"),
        2024: load_rank_lookup(lookup_rank_dir / "2024_cs_ranks.csv"),
        2025: load_rank_lookup(lookup_rank_dir / "2025_cs_ranks.csv"),
    }

    old_total, old_matched, out_total, out_matched = count_total_affiliations(
        oldoutputs_dir, outputs_dir, rank_lookup_by_year
    )

    old_hits = process_oldoutputs(oldoutputs_dir, rank_lookup_by_year)
    out_hits = process_outputs(outputs_dir, rank_lookup_by_year)

    def summarize(hits: list[AffiliationHit], source: str, total: int, matched: int) -> None:
        recoverable = [h for h in hits if h.recoverable_rank]
        unmatched = [h for h in hits if not h.recoverable_rank]
        print(f"\n=== {source} ===")
        print(f"First/last author affiliations (non-empty): {total}")
        print(f"Currently matched: {matched} ({100*matched/total:.1f}%)")
        print(f"Currently blank: {total - matched} ({100*(total-matched)/total:.1f}%)")
        print(
            f"Blank but recoverable (naming variant): {len(recoverable)} "
            f"({100*len(recoverable)/total:.1f}% of all; "
            f"{100*len(recoverable)/(total-matched) if total-matched else 0:.1f}% of blank)"
        )
        print(f"Truly unlisted / no recovery: {len(unmatched)}")

        reason_counts = Counter(h.reason for h in recoverable)
        print("Recovery reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")

        iit_like = [
            h
            for h in recoverable
            if "indian institute of technology" in h.raw.lower()
            or re.search(r"\biit\b", h.raw, re.I)
        ]
        print(f"IIT-related recoverable cases: {len(iit_like)}")

        top_patterns = Counter(
            (normalize_text(h.raw)[:80], h.recoverable_key, h.recoverable_rank)
            for h in recoverable
        )
        print("Top recoverable patterns (affiliation -> lookup key -> rank):")
        for (raw, key, rank), count in top_patterns.most_common(15):
            print(f"  [{count}x] {raw!r} -> {key!r} ({rank})")

    summarize(old_hits, "OldOutputs", old_total, old_matched)
    summarize(out_hits, "Outputs", out_total, out_matched)

    combined_recoverable = [h for h in old_hits + out_hits if h.recoverable_rank]
    combined_unmatched = [h for h in old_hits + out_hits if not h.recoverable_rank]
    print("\n=== Combined ===")
    print(
        f"Total first/last affiliations: {old_total + out_total:,}"
    )
    print(
        f"Naming-variant recoverable (like expanded IIT names): {len(combined_recoverable):,}"
    )
    print(f"Remaining unmatched: {len(combined_unmatched):,}")

    out_csv = SCRIPT_DIR / "unmatched_affiliation_analysis.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "file",
                "role",
                "year",
                "raw",
                "recoverable_rank",
                "recoverable_key",
                "reason",
            ],
        )
        writer.writeheader()
        for hit in sorted(old_hits + out_hits, key=lambda h: (h.source, h.file, h.role)):
            writer.writerow(
                {
                    "source": hit.source,
                    "file": hit.file,
                    "role": hit.role,
                    "year": hit.year,
                    "raw": hit.raw,
                    "recoverable_rank": hit.recoverable_rank,
                    "recoverable_key": hit.recoverable_key,
                    "reason": hit.reason,
                }
            )
    print(f"\nDetailed rows saved to: {out_csv}")


if __name__ == "__main__":
    main()
