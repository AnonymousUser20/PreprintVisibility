# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "paper_review_scores.csv"
OUTPUT_PNG = PROJECT_ROOT / "reviewer_rating_confidence_violins_usa_rank20.png"

CONFERENCES = ["ICLR", "ICML", "NeurIPS"]
GROUPS = ["Preprint", "No Preprint"]
GROUP_COLORS = {
    "Preprint": "#dbeafe",  # pastel blue
    "No Preprint": "#fde7d3",  # pastel orange
}


def load_review_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {
        "Conference",
        "Year",
        "rating",
        "confidence",
        "Group",
        "Country",
        "Rank",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Group"] = df["Group"].replace({"Non-Preprint": "No Preprint"})

    country_text = df["Country"].fillna("")
    usa_mask = country_text.str.contains(
        r"(?:^|;\s*)(?:USA|United States)(?:\s*;|$)",
        case=False,
        regex=True,
    )
    return df[usa_mask & (df["Rank"] <= 20)].copy()


def add_boxplot(ax, conf_df: pd.DataFrame, years: list[int], value_column: str) -> None:
    data = []
    positions = []
    colors = []

    pos = 1.0
    for year in years:
        for offset, group in enumerate(GROUPS):
            values = conf_df[
                (conf_df["Year"] == year) & (conf_df["Group"] == group)
            ][value_column].dropna()

            if values.empty:
                continue

            data.append(values)
            positions.append(pos + (offset * 0.35))
            colors.append(GROUP_COLORS[group])
        pos += 1.2

    if data:
        boxplot = ax.boxplot(
            data,
            positions=positions,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
        )

        for box, color in zip(boxplot["boxes"], colors):
            box.set_facecolor(color)
            box.set_edgecolor("gray")
            box.set_alpha(0.9)

        for key in ("medians", "whiskers", "caps"):
            for artist in boxplot[key]:
                artist.set_color("gray")
                artist.set_linewidth(1.1)

    ax.set_xticks([1.175 + (i * 1.2) for i in range(len(years))])
    ax.set_xticklabels(years)
    ax.grid(alpha=0.18)


def main() -> None:
    df = load_review_scores(INPUT_CSV)
    years = sorted(int(year) for year in df["Year"].dropna().unique())

    fig, axes = plt.subplots(len(CONFERENCES), 2, figsize=(16, 15))
    fig.patch.set_facecolor("white")

    for i, conference in enumerate(CONFERENCES):
        conf_df = df[df["Conference"] == conference]

        rating_ax = axes[i, 0]
        add_boxplot(rating_ax, conf_df, years, "rating")
        rating_ax.tick_params(axis="both", labelsize=18)
        for label in rating_ax.get_xticklabels() + rating_ax.get_yticklabels():
            label.set_fontweight("bold")
        rating_ax.set_title(f"{conference} - Ratings", fontsize=18, weight="bold")
        rating_ax.set_ylabel("Reviewer Rating")

        confidence_ax = axes[i, 1]
        if conference == "ICML":
            confidence_ax.remove()
            continue

        add_boxplot(confidence_ax, conf_df, years, "confidence")
        confidence_ax.tick_params(axis="both", labelsize=18)
        for label in confidence_ax.get_xticklabels() + confidence_ax.get_yticklabels():
            label.set_fontweight("bold")
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

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
