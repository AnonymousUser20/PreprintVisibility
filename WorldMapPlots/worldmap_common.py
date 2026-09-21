from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import plotly.express as px
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

NULL_LIKE = {"", "nan", "none", "null", "na", "n/a", "not found"}
CONFERENCES = (
    "ACL",
    "EMNLP",
    "COLING",
    "ICLR",
    "ICML",
    "NEURIPS",
    "KDD",
    "AAAI",
    "CVPR",
)

COUNTRY_ALIASES = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "uae": "United Arab Emirates",
    "viet nam": "Vietnam",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "russia": "Russian Federation",
}

TITLE_FONT_SIZE = 36
BASE_FONT_SIZE = 28
COLORBAR_TITLE_SIZE = 28
COLORBAR_TICK_SIZE = 28
SHARED_COLORBAR_LABEL_SIZE = 36


@dataclass(frozen=True)
class DateWindow:
    submission: date
    review: date


def is_non_null(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text not in NULL_LIKE


def parse_day(raw: object) -> date | None:
    text = str(raw or "").strip()
    if not text or text.lower() in NULL_LIKE:
        return None
    # Extract first date-like token even when extra text exists.
    match = re.search(r"(\d{4}[/-]\d{2}[/-]\d{2}|\d{2}[./-]\d{2}[./-]\d{4})", text)
    if match:
        text = match.group(1)
    else:
        text = text.split(";")[0].strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %B, %Y",
        "%d %b, %Y",
        "%B %Y",
        "%b %Y",
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def load_deadline_windows(deadlines_csv: Path) -> dict[tuple[str, int], DateWindow]:
    windows: dict[tuple[str, int], DateWindow] = {}
    with deadlines_csv.open("r", encoding="utf-8-sig", newline="") as f:
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
            submission = parse_day(row.get("Submission Deadline", ""))
            review = parse_day(row.get("Review deadline", ""))
            if submission is None or review is None:
                continue
            windows[(conf, year)] = DateWindow(submission=submission, review=review)
    return windows


def infer_conference_from_path(path: Path) -> str | None:
    up = str(path).upper()
    for conf in CONFERENCES:
        if conf in up:
            return conf
    return None


def infer_year_from_text(text: str) -> int | None:
    match = re.search(r"(20\d{2})", text)
    if not match:
        return None
    return int(match.group(1))


def resolve_outputs_conf_year(
    csv_file: Path, windows: dict[tuple[str, int], DateWindow]
) -> tuple[str, int] | None:
    """
    Resolve conference/year from Outputs filename (all tracks), then validate
    against available (conference, year) keys in conference_deadlines.csv.
    """
    name_up = csv_file.name.upper()
    candidates: list[tuple[str, int]] = []
    for conf, year in windows.keys():
        if conf in name_up and str(year) in name_up:
            candidates.append((conf, year))
    if not candidates:
        return None
    # Most specific/latest year if multiple matches.
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def resolve_oldoutputs_conf_year(
    csv_file: Path, windows: dict[tuple[str, int], DateWindow]
) -> tuple[str, int] | None:
    """
    Resolve conference/year from full OldOutputs path (all categories), then
    validate against available (conference, year) keys.
    """
    path_up = str(csv_file).upper()
    candidates: list[tuple[str, int]] = []
    for conf, year in windows.keys():
        if conf in path_up and str(year) in path_up:
            candidates.append((conf, year))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def parse_countries(raw: object) -> list[str]:
    text = str(raw or "").strip()
    if not text or text.lower() in NULL_LIKE:
        return []
    parts = re.split(r"\s*[;|,/]\s*", text)
    countries: list[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        normalized = COUNTRY_ALIASES.get(token.lower(), token)
        countries.append(normalized)
    return countries


def choose_country(first: str | None, last: str | None, all_countries: Iterable[str]) -> str | None:
    if first and last and first == last:
        return first
    if first and last and first != last:
        counts = Counter([c for c in all_countries if c])
        if counts:
            return counts.most_common(1)[0][0]
        return first
    if first:
        return first
    counts = Counter([c for c in all_countries if c])
    if counts:
        return counts.most_common(1)[0][0]
    return last


def in_window(raw_day: object, window: DateWindow, mode: str) -> bool:
    day = parse_day(raw_day)
    if day is None:
        return False
    sub = window.submission
    rev = window.review
    if mode == "between":
        return sub <= day <= rev
    if mode == "30":
        return sub - timedelta(days=30) <= day <= sub
    if mode == "60":
        return sub - timedelta(days=60) <= day <= sub - timedelta(days=30)
    if mode == "90":
        return sub - timedelta(days=90) <= day <= sub - timedelta(days=60)
    if mode == "180":
        return sub - timedelta(days=180) <= day <= sub - timedelta(days=90)
    if mode == "gt180":
        return day < sub - timedelta(days=180)
    raise ValueError(f"Unsupported mode: {mode}")


def counts_from_outputs(outputs_dir: Path, windows: dict[tuple[str, int], DateWindow], mode: str) -> Counter:
    result: Counter = Counter()
    for csv_file in sorted(outputs_dir.glob("*_preprint.csv")):
        resolved = resolve_outputs_conf_year(csv_file, windows)
        if resolved is None:
            continue
        conf, year = resolved
        window = windows[(conf, year)]

        papers: dict[str, dict[str, object]] = {}
        with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                paper_id = str(row.get("file_name", "")).strip()
                if not paper_id:
                    continue
                state = papers.setdefault(
                    paper_id,
                    {"eligible": False, "first": None, "last": None, "all": []},
                )
                if in_window(row.get("Date", ""), window, mode):
                    state["eligible"] = True
                countries = parse_countries(row.get("Country", ""))
                state["all"].extend(countries)
                role = str(row.get("authorship_role", "")).strip().upper()
                if countries and role == "FIRST_AUTHOR" and state["first"] is None:
                    state["first"] = countries[0]
                if countries and role == "LAST_AUTHOR" and state["last"] is None:
                    state["last"] = countries[0]

        for state in papers.values():
            if not state["eligible"]:
                continue
            chosen = choose_country(state["first"], state["last"], state["all"])
            if chosen:
                result[chosen] += 1
    return result


def counts_from_oldoutputs(oldoutputs_dir: Path, windows: dict[tuple[str, int], DateWindow], mode: str) -> Counter:
    result: Counter = Counter()
    for csv_file in sorted(oldoutputs_dir.glob("**/*.csv")):
        if "institute_batches" in str(csv_file).lower():
            continue
        resolved = resolve_oldoutputs_conf_year(csv_file, windows)
        if resolved is None:
            continue
        conf, year = resolved
        window = windows[(conf, year)]

        try:
            with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        fields = set(rows[0].keys())
        if not {"Title", "Date", "Country"}.issubset(fields):
            continue

        papers: dict[str, dict[str, object]] = {}
        for row in rows:
            paper_id = str(row.get("Title", "")).strip()
            if not paper_id:
                continue
            state = papers.setdefault(
                paper_id,
                {"eligible": False, "first": None, "last": None, "all": []},
            )
            if in_window(row.get("Date", ""), window, mode):
                state["eligible"] = True
            countries = parse_countries(row.get("Country", ""))
            state["all"].extend(countries)
            if countries:
                if state["first"] is None:
                    state["first"] = countries[0]
                state["last"] = countries[-1]

        for state in papers.values():
            if not state["eligible"]:
                continue
            chosen = choose_country(state["first"], state["last"], state["all"])
            if chosen:
                result[chosen] += 1
    return result


def build_country_counter(project_root: Path, mode: str) -> Counter:
    if not (project_root / "ConfusionMatrix").exists() and (
        project_root.parent / "ConfusionMatrix"
    ).exists():
        project_root = project_root.parent

    deadlines_csv = project_root / "ConfusionMatrix" / "conference_deadlines.csv"
    outputs_dir = project_root / "Outputs"
    oldoutputs_dir = project_root / "OldOutputs"
    windows = load_deadline_windows(deadlines_csv)
    total = Counter()
    total.update(counts_from_outputs(outputs_dir, windows, mode))
    total.update(counts_from_oldoutputs(oldoutputs_dir, windows, mode))
    return total


def plot_world(
    counter: Counter,
    title: str,
    output_png: Path,
    zmin: int | None = None,
    zmax: int | None = None,
) -> None:
    countries = sorted(counter.keys())
    values = [counter[c] for c in countries]
    range_color = None
    if zmin is not None and zmax is not None:
        range_color = (zmin, zmax)
    fig = px.choropleth(
        locations=countries,
        locationmode="country names",
        color=values,
        color_continuous_scale="plasma",
        range_color=range_color,
        title=f"<b>{title}</b>",
    )
    fig.update_traces(showscale=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        title_x=0.5,
        font=dict(size=BASE_FONT_SIZE),
        title_font=dict(size=TITLE_FONT_SIZE),
        coloraxis_showscale=False,
    )
    fig.write_image(str(output_png), width=1600, height=900, scale=2)
    print(f"Saved world heatmap: {output_png}")


def stitch_worldmaps(
    project_root: Path,
    zmin: int,
    zmax: int,
    color_scale: str = "plasma",
) -> Path:
    files = [
        "Between CfP-Review.png",
        "30 days before CfP.png",
        "60 days before CfP.png",
        "90 days before CfP.png",
        "180 days before CfP.png",
        "greater than 180 days before CfP.png",
    ]
    images = [Image.open(project_root / file_name).convert("RGB") for file_name in files]
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    normalized = [
        img.resize((cell_w, cell_h), Image.Resampling.LANCZOS) if img.size != (cell_w, cell_h) else img
        for img in images
    ]
    main_canvas = Image.new("RGB", (cell_w * 2, cell_h * 3), "white")
    for idx, img in enumerate(normalized):
        row = idx // 2
        col = idx % 2
        main_canvas.paste(img, (col * cell_w, row * cell_h))

    bar_margin = 30
    bar_width = 60
    final_width = main_canvas.width + (bar_margin * 3) + bar_width + 120
    final_canvas = Image.new("RGB", (final_width, main_canvas.height), "white")
    final_canvas.paste(main_canvas, (0, 0))

    draw = ImageDraw.Draw(final_canvas)
    try:
        label_font = ImageFont.truetype("arialbd.ttf", SHARED_COLORBAR_LABEL_SIZE)
    except OSError:
        try:
            label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", SHARED_COLORBAR_LABEL_SIZE)
        except OSError:
            label_font = ImageFont.load_default()
    cmap = cm.get_cmap(color_scale)
    bar_left = main_canvas.width + bar_margin
    bar_top = int(main_canvas.height * 0.08)
    bar_bottom = int(main_canvas.height * 0.92)
    bar_height = max(1, bar_bottom - bar_top)

    for y in range(bar_height):
        ratio = 1.0 - (y / max(1, bar_height - 1))
        r, g, b, _ = cmap(ratio)
        color = (int(r * 255), int(g * 255), int(b * 255))
        draw.line(
            [(bar_left, bar_top + y), (bar_left + bar_width, bar_top + y)],
            fill=color,
            width=1,
        )

    draw.rectangle(
        [bar_left, bar_top, bar_left + bar_width, bar_bottom],
        outline="black",
        width=2,
    )
    label_x = bar_left + bar_width + 15
    tick_values = list(range(int(zmin), int(zmax) + 1, 500))
    if not tick_values:
        tick_values = [int(zmin), int(zmax)]
    if tick_values[0] != int(zmin):
        tick_values.insert(0, int(zmin))
    if tick_values[-1] != int(zmax):
        tick_values.append(int(zmax))
    tick_values = sorted(set(tick_values))

    value_span = max(1, int(zmax) - int(zmin))
    for tick in tick_values:
        ratio = (tick - int(zmin)) / value_span
        y = bar_bottom - int(ratio * bar_height)
        draw.line(
            [(bar_left + bar_width + 2, y), (bar_left + bar_width + 12, y)],
            fill="black",
            width=2,
        )
        draw.text((label_x + 12, y - 18), str(tick), fill="black", font=label_font)

    draw.text((bar_left - 2, bar_top - 48), "Papers", fill="black", font=label_font)

    output = project_root / "worldmaps_6panel.png"
    final_canvas.save(output, quality=95)
    print(f"Saved stitched panel: {output}")
    return output


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    project_root = output_dir.parent
    specs = [
        ("between", "(a)", "Between CfP-Review.png"),
        ("30", "(b)", "30 days before CfP.png"),
        ("60", "(c)", "60 days before CfP.png"),
        ("90", "(d)", "90 days before CfP.png"),
        ("180", "(e)", "180 days before CfP.png"),
        ("gt180", "(f)", "greater than 180 days before CfP.png"),
    ]
    counters = {mode: build_country_counter(project_root, mode) for mode, _, _ in specs}
    global_max = max([max(counter.values()) for counter in counters.values() if counter] or [1])
    for mode, title, png_name in specs:
        plot_world(counters[mode], title, output_dir / png_name, zmin=0, zmax=global_max)
    stitch_worldmaps(output_dir, zmin=0, zmax=global_max, color_scale="plasma")


if __name__ == "__main__":
    main()
