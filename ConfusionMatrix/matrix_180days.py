from __future__ import annotations

import ast
import csv
import re
from datetime import datetime, date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

YEARS = (2023, 2024, 2025)
BOTTOM_RANK_THRESHOLD_BY_YEAR = {
    2023: 465, #20
    2024: 475,#20
    2025: 476, #20
}
NULL_LIKE = {"", "nan", "none", "null", "na"}
CONFERENCES = ("ICML", "ICLR", "NEURIPS", "KDD", "ACL", "CVPR", "EMNLP", "COLING", "AAAI")
DATE_PATTERNS = [
    "%Y-%m-%d",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %B, %Y",
    "%d %b, %Y",
    "%B %Y",
    "%b %Y",
]

# Matrix rows/cols bucket order:
# 0 -> top rank (<=10)
# 1 -> bottom rank (>= year threshold)
# 2 -> empty rank
BUCKET_LABELS = ["Top 20 \nRanked Institutes", "Bottom 20 \nRanked Institutes", "Not Listed \non CSRanking"]

# Bottom-right cell [2, 2]: count only papers where FIRST/LAST authors have at least one
# academic-institution affiliation (exclude companies / typical corporate orgs).
_ACADEMIC_HINT = re.compile(
    r"\b(university|college|polytechnic|faculty of|school of)\b",
    re.IGNORECASE,
)
_INSTITUTE_HINT = re.compile(
    r"\b(institute of|institute for|academy of sciences|academy of)\b",
    re.IGNORECASE,
)
_CORP_HINT = re.compile(
    r"\b("
    r"inc\.?|corp\.?|corporation|company|co\.|llc|ltd\.?|plc|gmbh|"
    r"google|alphabet|meta|facebook|microsoft|amazon|apple|nvidia|intel|oracle|"
    r"salesforce|adobe|ibm\b|openai|anthropic|deepmind|baidu|bytedance|alibaba|"
    r"tencent|huawei|netease|youdao|samsung|sony|ebay|yahoo|linkedin|twitter|"
    r"uber|lyft|airbnb|spotify|netflix|snapchat|tiktok"
    r")\b",
    re.IGNORECASE,
)


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


def load_institution_names(lookup_institutions_csv: Path) -> list[str]:
    names: list[str] = []
    with open(lookup_institutions_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = str(row.get("institution", "")).strip()
            if inst:
                names.append(inst.lower())
    return names


def affiliation_is_institution(
    affiliation: str, institution_names_lower: list[str]
) -> bool:
    text = str(affiliation).strip()
    if not text:
        return False

    lower = text.lower()
    if _CORP_HINT.search(lower):
        return False

    for inst in institution_names_lower:
        if inst and (inst in lower or lower in inst):
            return True

    if _ACADEMIC_HINT.search(lower) or _INSTITUTE_HINT.search(lower):
        return True

    return False


def paper_roles_have_institution_affiliation(
    first_affiliations: list[str],
    last_affiliations: list[str],
    institution_names_lower: list[str],
) -> bool:
    first_ok = any(
        affiliation_is_institution(a, institution_names_lower) for a in first_affiliations
    )
    last_ok = any(
        affiliation_is_institution(a, institution_names_lower) for a in last_affiliations
    )
    return first_ok and last_ok


def is_non_null(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in NULL_LIKE


def parse_date(value: object) -> date | None:
    text = str(value).strip()
    if not text or text.lower() in NULL_LIKE:
        return None

    for pattern in DATE_PATTERNS:
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.date()
        except ValueError:
            continue
    return None


def parse_deadline_day(value: str) -> date | None:
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%d.%m.%Y").date()


def load_deadline_windows(deadlines_csv: Path) -> dict[tuple[str, int], tuple[date, date]]:
    windows: dict[tuple[str, int], tuple[date, date]] = {}
    with open(deadlines_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = str(row.get("Conference", "")).strip().upper()
            year_text = str(row.get("Year", "")).strip()
            if not conf or not year_text:
                continue
            try:
                year = int(year_text)
            except ValueError:
                continue
            sub = parse_deadline_day(row.get("Submission Deadline", ""))
            rev = parse_deadline_day(row.get("Review deadline", ""))
            if sub is None or rev is None:
                continue
            windows[(conf, year)] = (sub, rev)
    return windows


def infer_conference_from_path(file_path: Path) -> str | None:
    text = str(file_path).upper()
    for conf in CONFERENCES:
        if conf in text:
            return conf
    return None


def is_in_180_to_90_day_window(
    file_path: Path,
    year: int,
    raw_date: object,
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> bool:
    conf = infer_conference_from_path(file_path)
    if conf is None:
        return False
    window = deadline_windows.get((conf, year))
    if window is None:
        return False

    parsed = parse_date(raw_date)
    if parsed is None:
        return False

    submission, _review = window
    start = submission - timedelta(days=180)
    end = submission - timedelta(days=90)
    return start <= parsed <= end


def rank_to_bucket(rank_raw: object, year: int) -> int | None:
    rank_text = str(rank_raw).strip()
    if not is_non_null(rank_text):
        return 2

    try:
        rank_value = int(float(rank_text))
    except ValueError:
        return None

    if rank_value <= 20:
        return 0
    if rank_value >= BOTTOM_RANK_THRESHOLD_BY_YEAR[year]:
        return 1
    return None


def choose_bucket(buckets: set[int]) -> int | None:
    # Priority: top -> bottom -> empty
    for bucket in (0, 1, 2):
        if bucket in buckets:
            return bucket
    return None


def build_year_matrix_from_outputs(
    outputs_dir: Path,
    year: int,
    institution_names_lower: list[str],
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    year_token = str(year)

    csv_files = sorted(
        path
        for path in outputs_dir.glob("*_preprint.csv")
        if year_token in path.name and path.name.endswith("_preprint.csv")
    )

    for csv_file in csv_files:
        papers: dict[str, dict[str, object]] = {}

        with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                paper_id = str(row.get("file_name", "")).strip()
                if not paper_id:
                    continue

                state = papers.setdefault(
                    paper_id,
                    {
                        "date_non_null": False,
                        "first_buckets": set(),
                        "last_buckets": set(),
                        "first_affiliations": [],
                        "last_affiliations": [],
                    },
                )

                if is_in_180_to_90_day_window(
                    file_path=csv_file,
                    year=year,
                    raw_date=row.get("Date", ""),
                    deadline_windows=deadline_windows,
                ):
                    state["date_non_null"] = True

                role = str(row.get("authorship_role", "")).strip().upper()
                bucket = rank_to_bucket(row.get("rank", ""), year)
                if bucket is None:
                    continue

                affs = parse_affiliations(row.get("affiliations", ""))
                if role == "FIRST_AUTHOR":
                    state["first_buckets"].add(bucket)
                    state["first_affiliations"].extend(affs)
                elif role == "LAST_AUTHOR":
                    state["last_buckets"].add(bucket)
                    state["last_affiliations"].extend(affs)

        for state in papers.values():
            if not state["date_non_null"]:
                continue

            first_bucket = choose_bucket(state["first_buckets"])
            last_bucket = choose_bucket(state["last_buckets"])
            if first_bucket is None or last_bucket is None:
                continue

            if first_bucket == 2 and last_bucket == 2:
                if not paper_roles_have_institution_affiliation(
                    state["first_affiliations"],
                    state["last_affiliations"],
                    institution_names_lower,
                ):
                    continue

            matrix[first_bucket, last_bucket] += 1

    return matrix


def build_year_matrix_from_oldoutputs(
    oldoutputs_dir: Path,
    year: int,
    institution_names_lower: list[str],
    deadline_windows: dict[tuple[str, int], tuple[date, date]],
) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    year_token = str(year)

    csv_files = sorted(
        path for path in oldoutputs_dir.glob("**/*.csv") if year_token in str(path)
    )

    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        fieldnames = set(rows[0].keys())
        # OldOutputs expected schema.
        if not {"Title", "Date", "Rank_first", "Rank_last"}.issubset(fieldnames):
            continue

        papers: dict[str, dict[str, object]] = {}
        for row in rows:
            paper_id = str(row.get("Title", "")).strip()
            if not paper_id:
                continue

            state = papers.setdefault(
                paper_id,
                {
                    "date_non_null": False,
                    "first_buckets": set(),
                    "last_buckets": set(),
                    "first_affiliations": [],
                    "last_affiliations": [],
                },
            )

            if is_in_180_to_90_day_window(
                file_path=csv_file,
                year=year,
                raw_date=row.get("Date", ""),
                deadline_windows=deadline_windows,
            ):
                state["date_non_null"] = True

            first_bucket = rank_to_bucket(row.get("Rank_first", ""), year)
            last_bucket = rank_to_bucket(row.get("Rank_last", ""), year)
            if first_bucket is not None:
                state["first_buckets"].add(first_bucket)
            if last_bucket is not None:
                state["last_buckets"].add(last_bucket)

            institutes = parse_affiliations(row.get("Institutes", ""))
            if institutes:
                state["first_affiliations"].append(institutes[0])
                state["last_affiliations"].append(institutes[-1])

        for state in papers.values():
            if not state["date_non_null"]:
                continue

            first_bucket = choose_bucket(state["first_buckets"])
            last_bucket = choose_bucket(state["last_buckets"])
            if first_bucket is None or last_bucket is None:
                continue

            if first_bucket == 2 and last_bucket == 2:
                if not paper_roles_have_institution_affiliation(
                    state["first_affiliations"],
                    state["last_affiliations"],
                    institution_names_lower,
                ):
                    continue

            matrix[first_bucket, last_bucket] += 1

    return matrix


def draw_heatmap_on_axis(
    ax: plt.Axes, matrix: np.ndarray, year: int, show_xticks: bool
) -> None:
    image = ax.imshow(matrix, cmap="YlOrRd")

    if show_xticks:
        ax.set_xticks(range(3), BUCKET_LABELS, rotation=90)
    else:
        ax.set_xticks([])
    ax.tick_params(axis="x", labelsize=22)
    ax.set_yticks([])
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.set_xlabel("Last Author", fontsize=22, fontweight="bold")
    ax.set_ylabel("")

    for row_idx in range(3):
        for col_idx in range(3):
            cell_value = matrix[row_idx, col_idx]
            red, green, blue, _ = image.cmap(image.norm(cell_value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(
                col_idx,
                row_idx,
                str(cell_value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=28,
            )

    return image


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root / "Outputs"
    oldoutputs_dir = project_root / "OldOutputs"
    institutions_csv = project_root / "lookup" / "institutions.csv"
    deadlines_csv = Path(__file__).resolve().parent / "conference_deadlines.csv"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    if not oldoutputs_dir.exists():
        raise FileNotFoundError(f"OldOutputs directory not found: {oldoutputs_dir}")
    if not institutions_csv.exists():
        raise FileNotFoundError(f"institutions.csv not found: {institutions_csv}")
    if not deadlines_csv.exists():
        raise FileNotFoundError(f"conference_deadlines.csv not found: {deadlines_csv}")

    institution_names_lower = load_institution_names(institutions_csv)
    deadline_windows = load_deadline_windows(deadlines_csv)

    matrices_by_year: dict[int, np.ndarray] = {}
    for year in YEARS:
        matrix_outputs = build_year_matrix_from_outputs(
            outputs_dir=outputs_dir,
            year=year,
            institution_names_lower=institution_names_lower,
            deadline_windows=deadline_windows,
        )
        matrix_oldoutputs = build_year_matrix_from_oldoutputs(
            oldoutputs_dir=oldoutputs_dir,
            year=year,
            institution_names_lower=institution_names_lower,
            deadline_windows=deadline_windows,
        )
        matrix = matrix_outputs + matrix_oldoutputs
        matrices_by_year[year] = matrix
        print(f"\n{year} matrix (rows=FIRST_AUTHOR, cols=LAST_AUTHOR; Top/Bottom/Empty):")
        print(matrix)

    # Compile all year plots vertically in one figure.
    fig, axes = plt.subplots(nrows=len(YEARS), ncols=1, figsize=(8, 18))
    if len(YEARS) == 1:
        axes = [axes]

    for idx, (ax, year) in enumerate(zip(axes, YEARS)):
        show_xticks = idx == len(YEARS) - 1
        image = draw_heatmap_on_axis(
            ax,
            matrices_by_year[year],
            year,
            show_xticks=show_xticks,
        )
        fig.colorbar(image, ax=ax, label="Unique papers")

    fig.suptitle("<180 days", fontsize=30, fontweight="bold")
    output_path = Path(__file__).resolve().parent / "180_days.png"
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
