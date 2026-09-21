"""Table 2 audit: affiliation, country, CSRankings, and institutional-tier mapping."""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from update_oldoutputs_ranks_from_lookup import (  # noqa: E402
    find_institute_column,
    first_last_from_authors_institutes,
    infer_year_from_path,
    load_rank_lookup,
    match_rank,
    normalize_text,
    parse_institutes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLD_OUTPUTS = PROJECT_ROOT / "OldOutputs"
OUTPUTS = PROJECT_ROOT / "Outputs"
LOOKUP_DIR = PROJECT_ROOT / "lookup_rank"
PAPERS_PER_CSV = 20
SKIP_DIR_NAMES = {"institute_batches"}
SKIP_FILE_NAMES = {"rank_fill_summary.csv"}
NULL_LIKE = {"", "nan", "none", "null", "na", "not found"}
BOTTOM_RANK_THRESHOLD = {2023: 465, 2024: 475, 2025: 476}

INDUSTRY_ORGS = {
    "google",
    "google research",
    "google deepmind",
    "deepmind",
    "alphabet",
    "meta",
    "meta ai",
    "facebook",
    "facebook ai",
    "fair",
    "amazon",
    "amazon com",
    "aws",
    "aws ai",
    "aws ai labs",
    "microsoft",
    "microsoft research",
    "apple",
    "ibm",
    "ibm research",
    "nvidia",
    "openai",
    "anthropic",
    "bytedance",
    "tencent",
    "alibaba",
    "huawei",
    "samsung",
    "adobe",
    "intel",
    "qualcomm",
    "salesforce",
    "uber",
    "twitter",
    "x corp",
    "cisco",
    "oracle",
    "baidu",
    "yahoo",
    "netflix",
    "snap",
    "linkedin",
    "tiktok",
}

COUNTRY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bunited states\b|\busa\b|\bu\.s\.a\b|\bu\.s\.\b", re.I), "United States"),
    (re.compile(r"\bunited kingdom\b|\bengland\b|\bscotland\b|\bu\.k\.\b|\buk\b", re.I), "United Kingdom"),
    (re.compile(r"\bchina\b|\bprc\b", re.I), "China"),
    (re.compile(r"\bindia\b", re.I), "India"),
    (re.compile(r"\bcanada\b", re.I), "Canada"),
    (re.compile(r"\bgermany\b|\bdeutschland\b", re.I), "Germany"),
    (re.compile(r"\bfrance\b", re.I), "France"),
    (re.compile(r"\bjapan\b", re.I), "Japan"),
    (re.compile(r"\bsingapore\b", re.I), "Singapore"),
    (re.compile(r"\bswitzerland\b", re.I), "Switzerland"),
    (re.compile(r"\baustralia\b", re.I), "Australia"),
    (re.compile(r"\bsouth korea\b|\bkorea\b", re.I), "South Korea"),
    (re.compile(r"\bisrael\b", re.I), "Israel"),
    (re.compile(r"\bnetherlands\b", re.I), "Netherlands"),
    (re.compile(r"\bhong kong\b", re.I), "Hong Kong"),
    (re.compile(r"\btaiwan\b", re.I), "Taiwan"),
    (re.compile(r"\bitaly\b", re.I), "Italy"),
    (re.compile(r"\bspain\b", re.I), "Spain"),
    (re.compile(r"\bbrazil\b", re.I), "Brazil"),
    (re.compile(r"\brussia\b", re.I), "Russia"),
    (re.compile(r"\bsweeden\b|\bsweden\b", re.I), "Sweden"),
    (re.compile(r"\bfinland\b", re.I), "Finland"),
    (re.compile(r"\bnorway\b", re.I), "Norway"),
    (re.compile(r"\bdenmark\b", re.I), "Denmark"),
    (re.compile(r"\baustria\b", re.I), "Austria"),
    (re.compile(r"\bbelgium\b", re.I), "Belgium"),
    (re.compile(r"\bireland\b", re.I), "Ireland"),
    (re.compile(r"\bpoland\b", re.I), "Poland"),
    (re.compile(r"\bportugal\b", re.I), "Portugal"),
    (re.compile(r"\bmexico\b", re.I), "Mexico"),
    (re.compile(r"\buae\b|\bunited arab emirates\b", re.I), "United Arab Emirates"),
    (re.compile(r"\bqatar\b", re.I), "Qatar"),
    (re.compile(r"\bsaudi arabia\b", re.I), "Saudi Arabia"),
    (re.compile(r"\bnew zealand\b", re.I), "New Zealand"),
    (re.compile(r"\bpakistan\b", re.I), "Pakistan"),
    (re.compile(r"\bbangladesh\b", re.I), "Bangladesh"),
    (re.compile(r"\bvietnam\b", re.I), "Vietnam"),
    (re.compile(r"\bthailand\b", re.I), "Thailand"),
    (re.compile(r"\bmalaysia\b", re.I), "Malaysia"),
    (re.compile(r"\bindonesia\b", re.I), "Indonesia"),
    (re.compile(r"\bgreece\b", re.I), "Greece"),
    (re.compile(r"\bturkey\b|\btürkiye\b", re.I), "Turkey"),
    (re.compile(r"\bczech\b", re.I), "Czech Republic"),
    (re.compile(r"\bhungary\b", re.I), "Hungary"),
    (re.compile(r"\bromania\b", re.I), "Romania"),
    (re.compile(r"\bsouth africa\b", re.I), "South Africa"),
    (re.compile(r"\bargentina\b", re.I), "Argentina"),
    (re.compile(r"\bchile\b", re.I), "Chile"),
    (re.compile(r"\bcolombia\b", re.I), "Colombia"),
    (re.compile(r"\begypt\b", re.I), "Egypt"),
    (re.compile(r"\biran\b", re.I), "Iran"),
]

IIT_RE = re.compile(r"indian\s+institute\s+of\s+technology|\biit\b", re.I)
ILLINOIS_IIT_RE = re.compile(r"illinois\s+institute\s+of\s+technology", re.I)
IIT_CAMPUS_RE = re.compile(
    r"\b(delhi|bombay|mumbai|madras|chennai|kanpur|kharagpur|roorkee|"
    r"guwahati|hyderabad|indore|jodhpur|patna|ropar|tirupati|gandhinagar|"
    r"bhu|varanasi|dhanbad|jammu|mandi|bhubaneswar|palakkad|goa|dharwad|"
    r"bhilai|jammu)\b",
    re.I,
)


def is_indian_iit(raw: str) -> bool:
    text = strip_domain(raw)
    if ILLINOIS_IIT_RE.search(text):
        return False
    return bool(IIT_RE.search(text))


def is_null(value: object) -> bool:
    text = str(value).strip()
    return not text or text.lower() in NULL_LIKE


def strip_domain(raw: str) -> str:
    return re.sub(r"\([^)]*\)", " ", raw).strip()


def affiliation_has_country(raw: str) -> str:
    text = strip_domain(raw)
    for pattern, country in COUNTRY_PATTERNS:
        if pattern.search(text):
            return country
    return ""


def is_industry_org(raw: str) -> bool:
    norm = normalize_text(strip_domain(raw))
    if not norm:
        return False
    if norm in INDUSTRY_ORGS:
        return True
    tokens = set(norm.split())
    for org in INDUSTRY_ORGS:
        org_tokens = set(org.split())
        if org_tokens <= tokens and len(org_tokens) >= 1:
            if org in {"google", "meta", "amazon", "microsoft", "apple", "ibm", "nvidia"}:
                if org in tokens:
                    return True
            elif org in norm:
                return True
    return False


def is_multi_location(raw: str) -> bool:
    if is_industry_org(raw):
        return True
    norm = normalize_text(strip_domain(raw))
    generic = {
        "university of california",
        "iit",
        "indian institute of technology",
        "national university",
        "state university",
        "institute of technology",
    }
    if norm in generic:
        return True
    if norm in {"iit", "indian institute of technology"}:
        return True
    if is_indian_iit(raw) and not IIT_CAMPUS_RE.search(strip_domain(raw)):
        return True
    return False


def unique_enough(raw: str) -> bool:
    if affiliation_has_country(raw):
        return True
    if is_multi_location(raw):
        return False
    norm = normalize_text(strip_domain(raw))
    if len(norm.split()) >= 2:
        return True
    return False


def classify_affiliation(raw: str) -> str:
    if is_null(raw):
        return "incorrect"
    if not affiliation_has_country(raw) and is_multi_location(raw):
        return "uncertain"
    if unique_enough(raw):
        return "correct"
    if not affiliation_has_country(raw) and is_industry_org(raw):
        return "uncertain"
    return "correct"


def primary_country(raw: object) -> str:
    text = str(raw).strip()
    if is_null(text) or text.lower() == "unknown":
        return ""
    return text.split(";")[0].strip()


def classify_country(affiliation: str, mapped_country: str) -> str:
    if is_null(affiliation):
        return "incorrect"
    if is_industry_org(affiliation) and not affiliation_has_country(affiliation):
        return "uncertain"
    if not affiliation_has_country(affiliation) and is_multi_location(affiliation):
        return "uncertain"

    mentioned = affiliation_has_country(affiliation)
    mapped = primary_country(mapped_country)

    if mentioned:
        if not mapped:
            return "incorrect"
        if mentioned.lower() in mapped.lower() or mapped.lower() in mentioned.lower():
            return "correct"
        aliases = {
            "united states": {"usa", "us", "america", "united states of america"},
            "united kingdom": {"uk", "britain", "england", "scotland", "wales"},
            "south korea": {"korea", "republic of korea"},
        }
        for canonical, alts in aliases.items():
            if mentioned.lower() == canonical and mapped.lower() in alts | {canonical}:
                return "correct"
        return "incorrect"

    if mapped:
        return "correct"
    return "incorrect"


def iit_alias_match(raw: str, lookup: dict[str, str]) -> str:
    s = re.sub(r"\([^)]*\)", " ", raw)
    s = re.sub(r",?\s*india\s*$", "", s, flags=re.I).strip(" ,")
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for part in reversed(parts):
            if IIT_RE.search(part):
                s = part
                break
        else:
            s = parts[-1]
    norm = IIT_RE.sub("iit", normalize_text(s))
    norm = re.sub(r"\bmumbai\b", "bombay", norm)
    norm = re.sub(r"\bchennai\b", "madras", norm)
    return match_rank(s, lookup) or lookup.get(norm, "") or match_rank(norm, lookup)


def classify_csrankings(raw: str, assigned_rank: str, lookup: dict[str, str]) -> str:
    if is_null(raw) or not is_indian_iit(raw):
        return ""
    if not IIT_CAMPUS_RE.search(raw):
        return "uncertain"
    current = assigned_rank if not is_null(assigned_rank) else match_rank(raw, lookup)
    recoverable = iit_alias_match(raw, lookup)
    if current:
        if recoverable and current != recoverable:
            return "incorrect"
        return "correct"
    if recoverable:
        return "incorrect"
    return "uncertain"


def institution_tier(rank_raw: object, year: int) -> str:
    text = str(rank_raw).strip()
    if is_null(text):
        return "not_listed"
    try:
        rank = int(float(text))
    except ValueError:
        return "not_listed"
    if rank <= 20:
        return "top"
    if rank >= BOTTOM_RANK_THRESHOLD.get(year, 9999):
        return "bottom"
    return "not_listed"


def classify_tier(cs_status: str, assigned_rank: str, recoverable_rank: str, year: int) -> str:
    if cs_status == "uncertain":
        return "uncertain"
    if cs_status == "incorrect":
        return "incorrect"
    assigned_tier = institution_tier(assigned_rank, year)
    true_tier = institution_tier(recoverable_rank or assigned_rank, year)
    if assigned_tier == true_tier:
        return "correct"
    return "incorrect"


def parse_author_list(raw: object) -> list[str]:
    if is_null(raw):
        return []
    text = str(raw).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [p.strip() for p in re.split(r"[;,]", text) if p.strip()]


def discover_csvs() -> list[Path]:
    files: list[Path] = []
    for root in (OLD_OUTPUTS, OUTPUTS):
        for path in sorted(root.glob("**/*.csv")):
            if path.name in SKIP_FILE_NAMES:
                continue
            if any(part.lower() in SKIP_DIR_NAMES for part in path.parts):
                continue
            files.append(path)
    return files


def infer_year(path: Path) -> int | None:
    year = infer_year_from_path(path)
    if year:
        return year
    match = re.search(r"(2023|2024|2025)", path.stem)
    return int(match.group(1)) if match else None


def load_oldoutputs_papers(path: Path, lookup: dict[str, str], year: int) -> list[dict[str, object]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    col = find_institute_column(list(rows[0].keys()))
    papers: list[dict[str, object]] = []
    for row in rows[:PAPERS_PER_CSV]:
        raw = row.get(col, "") if col else ""
        if col and col.strip().lower() == "authors_institutes":
            first_inst, last_inst = first_last_from_authors_institutes(raw)
        else:
            institutes = parse_institutes(raw)
            first_inst = institutes[0] if institutes else ""
            last_inst = institutes[-1] if institutes else ""
        mapped_countries = [c.strip() for c in str(row.get("Country", "")).split(";") if c.strip()]
        papers.append(
            {
                "source": str(path.relative_to(PROJECT_ROOT)),
                "year": year,
                "first_aff": first_inst,
                "last_aff": last_inst,
                "first_country": mapped_countries[0] if mapped_countries else "",
                "last_country": mapped_countries[-1] if mapped_countries else "",
                "rank_first": str(row.get("Rank_first", "")).strip(),
                "rank_last": str(row.get("Rank_last", "")).strip(),
            }
        )
    return papers


def load_outputs_papers(path: Path, lookup: dict[str, str], year: int) -> list[dict[str, object]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        title = str(row.get("paper_title", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        if not title or title.lower() in NULL_LIKE:
            title = file_name
        if not title:
            continue
        key = (file_name, title)
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        role = str(row.get("authorship_role", "")).strip().upper()
        affs = parse_institutes(row.get("affiliations", ""))
        grouped[key][role] = {
            "aff": affs[0] if affs else "",
            "country": str(row.get("Country", "")).strip(),
            "rank": str(row.get("rank", "")).strip(),
        }

    papers: list[dict[str, object]] = []
    for key in order[:PAPERS_PER_CSV]:
        roles = grouped[key]
        first = roles.get("FIRST_AUTHOR", {})
        last = roles.get("LAST_AUTHOR", {})
        papers.append(
            {
                "source": str(path.relative_to(PROJECT_ROOT)),
                "year": year,
                "first_aff": first.get("aff", ""),
                "last_aff": last.get("aff", ""),
                "first_country": first.get("country", ""),
                "last_country": last.get("country", ""),
                "rank_first": first.get("rank", ""),
                "rank_last": last.get("rank", ""),
            }
        )
    return papers


def tally(labels: list[str]) -> dict[str, int | float | str]:
    counts = Counter(labels)
    sample = len(labels)
    correct = counts.get("correct", 0)
    incorrect = counts.get("incorrect", 0)
    uncertain = counts.get("uncertain", 0)
    accuracy = (correct / sample) if sample else 0.0
    return {
        "sample_size": sample,
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "accuracy": accuracy,
    }


def main() -> None:
    lookups = {
        2023: load_rank_lookup(LOOKUP_DIR / "2023_cs_ranks.csv"),
        2024: load_rank_lookup(LOOKUP_DIR / "2024_cs_ranks.csv"),
        2025: load_rank_lookup(LOOKUP_DIR / "2025_cs_ranks.csv"),
    }
    csvs = discover_csvs()
    papers: list[dict[str, object]] = []
    for path in csvs:
        year = infer_year(path)
        if year not in lookups:
            continue
        lookup = lookups[year]
        try:
            path.relative_to(OUTPUTS)
            loaded = load_outputs_papers(path, lookup, year)
        except ValueError:
            loaded = load_oldoutputs_papers(path, lookup, year)
        papers.extend(loaded)

    first_labels: list[str] = []
    last_labels: list[str] = []
    country_labels: list[str] = []
    cs_labels: list[str] = []
    tier_labels: list[str] = []
    iit_examples: list[dict[str, object]] = []

    for paper in papers:
        year = int(paper["year"])
        lookup = lookups[year]
        first_aff = str(paper["first_aff"])
        last_aff = str(paper["last_aff"])

        first_labels.append(classify_affiliation(first_aff))
        last_labels.append(classify_affiliation(last_aff))

        for aff, mapped in (
            (first_aff, str(paper["first_country"])),
            (last_aff, str(paper["last_country"])),
        ):
            if is_null(aff):
                continue
            country_labels.append(classify_country(aff, mapped))

        for role, aff, rank in (
            ("first", first_aff, str(paper["rank_first"])),
            ("last", last_aff, str(paper["rank_last"])),
        ):
            if is_null(aff) or not is_indian_iit(aff):
                continue
            cs_status = classify_csrankings(aff, rank, lookup)
            if not cs_status:
                continue
            recoverable = iit_alias_match(aff, lookup)
            assigned = rank if not is_null(rank) else match_rank(aff, lookup)
            cs_labels.append(cs_status)
            tier_labels.append(classify_tier(cs_status, assigned, recoverable, year))
            iit_examples.append(
                {
                    "source": paper["source"],
                    "role": role,
                    "affiliation": aff,
                    "assigned_rank": assigned,
                    "recoverable_rank": recoverable,
                    "cs_status": cs_status,
                    "assigned_tier": institution_tier(assigned, year),
                    "true_tier": institution_tier(recoverable or assigned, year),
                }
            )

    table = {
        "First-author affiliation": tally(first_labels),
        "Last-author affiliation": tally(last_labels),
        "Country mapping": tally(country_labels),
        "CSRankings match": tally(cs_labels),
        "Institutional tier": tally(tier_labels),
    }

    out_json = PROJECT_ROOT / "MetadataAvailability" / "table2_mapping_audit.json"
    out_csv = PROJECT_ROOT / "MetadataAvailability" / "table2_mapping_audit.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_csvs": len(csvs),
        "n_papers": len(papers),
        "papers_per_csv": PAPERS_PER_CSV,
        "table": table,
        "iit_examples": iit_examples[:40],
        "iit_status_counts": dict(Counter(x["cs_status"] for x in iit_examples)),
        "notes": {
            "country": (
                "Industry orgs (Google, Meta, Amazon, Microsoft, etc.) without a "
                "country in the affiliation string are marked uncertain; they are "
                "excluded from institution_tier regressions because they are not "
                "listed in CSRankings."
            ),
            "csrankings": (
                "Sample is IIT / Indian Institute of Technology first- or last-author "
                "affiliations in the top-20-per-CSV audit set. Incorrect = expanded "
                "name that should match a CSRankings campus but did not. Uncertain = "
                "no campus specified."
            ),
            "tier": (
                "CSRankings errors carry forward: unmatched IIT campuses become "
                "not_listed instead of their year-specific AI/DS rank band. Rankings "
                "are year-specific (2023/2024/2025) from the CSRankings UI, Artificial "
                "Intelligence / Data Science domain."
            ),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Field audited", "Sample size", "Correct", "Incorrect", "Uncertain", "Accuracy"])
        for field, stats in table.items():
            writer.writerow(
                [
                    field,
                    stats["sample_size"],
                    stats["correct"],
                    stats["incorrect"],
                    stats["uncertain"],
                    f"{100 * float(stats['accuracy']):.1f}%",
                ]
            )

    print(f"Papers audited: {len(papers)} from {len(csvs)} CSVs (top {PAPERS_PER_CSV} each)")
    print()
    print(f"{'Field audited':<28} {'N':>6} {'Correct':>8} {'Incorrect':>10} {'Uncertain':>10} {'Accuracy':>10}")
    for field, stats in table.items():
        print(
            f"{field:<28} {stats['sample_size']:>6} {stats['correct']:>8} "
            f"{stats['incorrect']:>10} {stats['uncertain']:>10} "
            f"{100 * float(stats['accuracy']):>9.1f}%"
        )
    print()
    print("IIT cases:", len(iit_examples), dict(Counter(x["cs_status"] for x in iit_examples)))
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
