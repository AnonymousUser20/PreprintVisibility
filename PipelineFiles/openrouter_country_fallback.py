from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-5.5"

SYSTEM_PROMPT = (
    "You are a data-normalization assistant for affiliation->country mapping. "
    "Always return strict JSON only."
)

USER_PROMPT_TEMPLATE = """You must map a single affiliation string to a country using these rules:
1) If a country name is explicitly mentioned in the affiliation, return that country.
2) If no country is mentioned and affiliation is a specific institute/university/lab/company with a clear primary location for this context, 
infer and return country.
3) If no country is mentioned and affiliation is a global organization with many offices (ambiguous location), return empty string.

Output JSON exactly in this schema:
{{"country": "<country or empty>", "confidence": <0-1 float>, "reason": "<short reason>"}}

Affiliation: "{affiliation}"
"""


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


def call_openrouter(
    api_key: str,
    affiliation: str,
    timeout_seconds: int = 60,
    max_retries: int = 4,
) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(affiliation=affiliation)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    backoff = 1.5
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError):
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("OpenRouter call failed unexpectedly.")


def normalize_country(candidate: str) -> str:
    country = str(candidate).strip()
    if not country or country.lower() in {"none", "null", "n/a", "unknown"}:
        return ""
    return country


def fallback_country_for_row(
    api_key: str,
    affiliations_raw: str,
    cache: dict[str, str],
    sleep_seconds: float,
) -> str:
    affiliations = parse_affiliations(affiliations_raw)
    if not affiliations:
        return ""

    countries: list[str] = []
    for affiliation in affiliations:
        if affiliation in cache:
            inferred = cache[affiliation]
        else:
            result = call_openrouter(api_key=api_key, affiliation=affiliation)
            inferred = normalize_country(result.get("country", ""))
            cache[affiliation] = inferred
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if inferred:
            countries.append(inferred)

    unique = list(OrderedDict.fromkeys(countries))
    return "; ".join(unique)


def fallback_country_for_affiliation(
    api_key: str,
    affiliation: str,
    cache: dict[str, str],
    sleep_seconds: float,
) -> str:
    cleaned = str(affiliation).strip()
    if not cleaned:
        return ""

    if cleaned in cache:
        return cache[cleaned]

    result = call_openrouter(api_key=api_key, affiliation=cleaned)
    inferred = normalize_country(result.get("country", ""))
    cache[cleaned] = inferred

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return inferred


def run_fallback(
    input_csv: Path,
    output_csv: Path,
    api_key: str,
    sleep_seconds: float,
) -> None:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Input CSV has no rows.")

    cache: dict[str, str] = {}
    filled = 0
    skipped_already_mapped = 0

    for row in tqdm(rows, desc="Mapping countries in main author CSV", unit="row"):
        existing_country = str(row.get("Country", "")).strip()
        if existing_country:
            skipped_already_mapped += 1
            continue

        inferred_country = fallback_country_for_row(
            api_key=api_key,
            affiliations_raw=row.get("affiliations", ""),
            cache=cache,
            sleep_seconds=sleep_seconds,
        )
        if inferred_country:
            row["Country"] = inferred_country
            filled += 1

    fieldnames = list(rows[0].keys())
    if "Country" not in fieldnames:
        fieldnames.append("Country")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    mapped = sum(1 for row in rows if str(row.get("Country", "")).strip())
    unmapped = total - mapped

    print(f"Saved: {output_csv}")
    print(f"Total rows: {total}")
    print(f"Already mapped before fallback: {skipped_already_mapped}")
    print(f"Filled by OpenRouter fallback: {filled}")
    print(f"Mapped total after fallback: {mapped}")
    print(f"Unmapped total after fallback: {unmapped}")
    print(f"Unique affiliations queried: {len(cache)}")


def run_affiliation_frequency_fallback(
    input_csv: Path,
    output_csv: Path,
    api_key: str,
    sleep_seconds: float,
) -> None:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Input affiliation frequency CSV has no rows.")
    if "affiliation" not in rows[0]:
        raise ValueError("Expected 'affiliation' column in input CSV.")

    cache: dict[str, str] = {}
    filled = 0

    for row in tqdm(rows, desc="Mapping countries in affiliation frequency CSV", unit="affiliation"):
        affiliation = row.get("affiliation", "")
        inferred_country = fallback_country_for_affiliation(
            api_key=api_key,
            affiliation=affiliation,
            cache=cache,
            sleep_seconds=sleep_seconds,
        )
        row["Country"] = inferred_country
        if inferred_country:
            filled += 1

    fieldnames = list(rows[0].keys())
    if "Country" not in fieldnames:
        fieldnames.append("Country")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    unmapped = total - filled
    print(f"Saved: {output_csv}")
    print(f"Distinct affiliations processed: {total}")
    print(f"Country populated: {filled}")
    print(f"Still unmapped: {unmapped}")
    print(f"Unique affiliations queried: {len(cache)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fallback country mapping using OpenRouter model openai/gpt-5.5."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input CSV path. Defaults to unmapped_affiliations_frequencies.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to unmapped_affiliations_frequencies_with_country.csv.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY from .env or the environment.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Optional text file containing OpenRouter API key (first line).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between uncached API calls to reduce rate-limit risk.",
    )
    return parser.parse_args()


def resolve_api_key(cli_key: str, api_key_file: Path | None) -> str:
    if cli_key.strip():
        return cli_key.strip()

    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    if api_key_file and api_key_file.exists():
        with open(api_key_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line:
            return first_line

    return ""


if __name__ == "__main__":
    args = parse_args()
    resolved_key = resolve_api_key(args.api_key, args.api_key_file)
    if not resolved_key:
        raise ValueError(
            "Missing OpenRouter API key. Put OPENROUTER_API_KEY in the project .env, "
            "set the environment variable, or pass --api-key / --api-key-file."
        )
    base_dir = Path(__file__).resolve().parent
    input_csv = args.input_csv or (base_dir / "unmapped_affiliations_frequencies.csv")
    output_csv = args.output_csv or (base_dir / "unmapped_affiliations_frequencies_with_country.csv")
    run_affiliation_frequency_fallback(
        input_csv=input_csv,
        output_csv=output_csv,
        api_key=resolved_key,
        sleep_seconds=args.sleep_seconds,
    )
