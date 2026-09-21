# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "paper_review_score_lists.csv"
OUTPUT_PNG = PROJECT_ROOT / "reviewer_rating_confidence_list_violins.png"

CONFERENCES = ["ICLR", "ICML", "NeurIPS"]
GROUPS = ["Preprint", "No Preprint"]
GROUP_COLORS = {
    "Preprint": "#dbeafe",  # pastel blue
    "No Preprint": "#fde7d3",  # pastel orange
}


def parse_score_list(value) -> list[float]:
    if pd.isna(value):
        return []

    try:
        scores = json.loads(str(value))
    except json.JSONDecodeError:
        return []

    if not isinstance(scores, list):
        return []

    parsed_scores = []
    for score in scores:
        try:
            parsed_scores.append(float(score))
        except (TypeError, ValueError):
            continue
    return parsed_scores


def load_review_score_lists(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"Paper Title", "Conference", "Year", "rating", "confidence", "Group"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Group"] = df["Group"].replace({"Non-Preprint": "No Preprint"})
    return df


def aggregate_scores(df: pd.DataFrame, metric: str) -> dict[tuple[str, int, str], list[float]]:
    aggregated = {}

    for row in df.to_dict("records"):
        if pd.isna(row["Year"]):
            continue

        key = (row["Conference"], int(row["Year"]), row["Group"])
        aggregated.setdefault(key, []).extend(parse_score_list(row[metric]))

    return aggregated


def add_violinplot(
    ax,
    aggregated_scores: dict[tuple[str, int, str], list[float]],
    conference: str,
    years: list[int],
) -> None:
    data = []
    positions = []
    colors = []

    pos = 1.0
    for year in years:
        for offset, group in enumerate(GROUPS):
            values = aggregated_scores.get((conference, year, group), [])

            if not values:
                continue

            data.append(values)
            positions.append(pos + (offset * 0.35))
            colors.append(GROUP_COLORS[group])
        pos += 1.2

    if data:
        violinplot = ax.violinplot(
            data,
            positions=positions,
            widths=0.28,
            showmeans=False,
            showmedians=True,
            showextrema=True,
            bw_method=0.8,   # smooth KDE
        )

        for body, color in zip(violinplot["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("gray")
            body.set_alpha(0.9)

        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in violinplot:
                violinplot[key].set_color("gray")
                violinplot[key].set_linewidth(1.1)

    ax.set_xticks([1.175 + (i * 1.2) for i in range(len(years))])
    ax.set_xticklabels(years)
    ax.grid(alpha=0.18)


def style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def main() -> None:
    paper_df = load_review_score_lists(INPUT_CSV)
    rating_scores = aggregate_scores(paper_df, "rating")
    confidence_scores = aggregate_scores(paper_df, "confidence")
    years = sorted(int(year) for year in paper_df["Year"].dropna().unique())

    fig, axes = plt.subplots(len(CONFERENCES), 2, figsize=(16, 15))
    fig.patch.set_facecolor("white")

    for i, conference in enumerate(CONFERENCES):
        rating_ax = axes[i, 0]
        add_violinplot(rating_ax, rating_scores, conference, years)
        style_axis(rating_ax)
        rating_ax.set_title(f"{conference} - Ratings", fontsize=18, weight="bold")
        rating_ax.set_ylabel("Reviewer Rating")

        confidence_ax = axes[i, 1]
        if conference == "ICML":
            confidence_ax.remove()
            continue

        add_violinplot(confidence_ax, confidence_scores, conference, years)
        style_axis(confidence_ax)
        confidence_ax.set_title(
            f"{conference} - Confidence", fontsize=18, weight="bold"
        )
        confidence_ax.set_ylabel("Reviewer Confidence")

    legend_elements = [
        Patch(facecolor=GROUP_COLORS[group], edgecolor="gray", label=group)
        for group in GROUPS
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        prop={"size": 18, "weight": "bold"},
        frameon=False,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
