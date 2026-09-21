# pyright: reportMissingModuleSource=false, reportArgumentType=false, reportAssignmentType=false

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OLD_OUTPUTS_ROOT = PROJECT_ROOT / "OldOutputs"
OUTPUTS_ROOT = PROJECT_ROOT / "Outputs"
DEADLINES_CSV = PROJECT_ROOT / "ConfusionMatrix" / "conference_deadlines.csv"
OUTPUT_PNG = SCRIPT_DIR / "rank_overall_no_preprint_ratio.png"

YEARS = [2023, 2024, 2025]
PREFERRED_CONFERENCE_ORDER = ["AAAI", "CVPR", "ICLR", "ICML", "KDD", "NeurIPS"]
BOTTOM_RANK_THRESHOLD = {
    2023: 466,
    2024: 475,
    2025: 477,
}
RANK_GROUPS = ["Top 20 \nRanked Institutes", "Bottom 20 \nRanked Institutes"]
CONFERENCES_WITHOUT_YTICKS = {"CVPR", "ICLR", "KDD", "NeurIPS", "EMNLP", "COLING"}


def parse_date(value: str):
    value = str(value).strip()
    if not value or value.lower() == "not found" or value.lower() == "nan":
        return pd.NaT

    value = value.split(";", 1)[0]
    value = value.split("(", 1)[0].strip()
    for date_format in ("%d.%m.%Y", "%d %B, %Y", "%B %Y"):
        parsed = pd.to_datetime(value, format=date_format, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    return pd.to_datetime(value, errors="coerce")


def load_review_deadlines() -> dict[tuple[str, int], pd.Timestamp]:
    deadlines = {}
    with DEADLINES_CSV.open(newline="", encoding="utf-8-sig") as fp:
        deadline_df = pd.read_csv(fp)

    for _, row in deadline_df.iterrows():
        conference = str(row["Conference"]).strip()
        year = int(row["Year"])
        review_deadline = parse_date(str(row["Review deadline"]))
        if not pd.isna(review_deadline):
            deadlines[(conference, year)] = review_deadline
    return deadlines


def load_deadline_conference_order() -> list[str]:
    with DEADLINES_CSV.open(newline="", encoding="utf-8-sig") as fp:
        deadline_df = pd.read_csv(fp)

    conferences = []
    seen = set()
    for conference in deadline_df["Conference"]:
        conference = str(conference).strip()
        if conference and conference not in seen:
            seen.add(conference)
            conferences.append(conference)
    return conferences


def conference_from_path(path: Path, conference_order: list[str]) -> str | None:
    path_text = str(path).lower()
    for conference in conference_order:
        conference_lower = conference.lower()
        if any(part.lower() == conference_lower for part in path.parts):
            return conference
        if conference_lower in path.stem.lower().replace("-", "_").split("_"):
            return conference
        if conference_lower in path_text:
            return conference
    return None


def year_from_path(path: Path) -> int | None:
    match = pd.Series([str(path)]).str.extract(r"(2023|2024|2025)")[0].iloc[0]
    if pd.isna(match):
        return None
    return int(match)


def rank_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rank_first_raw = df["Rank_first"] if "Rank_first" in df.columns else pd.Series(np.nan, index=df.index)
    rank_last_raw = df["Rank_last"] if "Rank_last" in df.columns else pd.Series(np.nan, index=df.index)
    rank_first = pd.Series(pd.to_numeric(rank_first_raw, errors="coerce"), index=df.index)
    rank_last = pd.Series(pd.to_numeric(rank_last_raw, errors="coerce"), index=df.index)
    return rank_first, rank_last


def accepted_paper_files(conference_order: list[str]) -> list[Path]:
    skipped_names = {"rank_fill_summary.csv"}
    files = []
    for root in (OLD_OUTPUTS_ROOT, OUTPUTS_ROOT):
        if not root.exists():
            continue
        for csv_path in root.rglob("*.csv"):
            if csv_path.name in skipped_names:
                continue
            if "institute_batches" in {part.lower() for part in csv_path.parts}:
                continue
            if conference_from_path(csv_path, conference_order) is None or year_from_path(csv_path) is None:
                continue
            files.append(csv_path)
    return sorted(files)


def paper_keys(df: pd.DataFrame) -> pd.Series | None:
    """Paper identifier for author-level Outputs rows (title, or file_name if title missing)."""
    keys: pd.Series | None = None
    for column in ("Title", "paper_title", "Paper Title"):
        if column not in df.columns:
            continue
        candidate = df[column].astype(str).str.strip()
        candidate = candidate.mask(candidate.str.lower().isin({"", "nan", "none"}))
        if keys is None:
            keys = candidate
        else:
            keys = keys.fillna(candidate)

    if keys is not None and "file_name" in df.columns:
        file_keys = df["file_name"].astype(str).str.strip()
        keys = keys.fillna(file_keys)

    if keys is None and "file_name" in df.columns:
        keys = df["file_name"].astype(str).str.strip()

    if keys is None:
        return None
    return keys.mask(keys.str.lower().isin({"", "nan", "none"}))


def output_rank_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ranks = pd.Series(pd.to_numeric(df["rank"], errors="coerce"), index=df.index)
    roles = df["authorship_role"].astype(str).str.upper()
    rank_first = pd.Series(
        ranks.where(roles.str.contains("FIRST_AUTHOR", na=False)), index=df.index
    )
    rank_last = pd.Series(
        ranks.where(roles.str.contains("LAST_AUTHOR", na=False)), index=df.index
    )
    return rank_first, rank_last


def collapse_author_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = paper_keys(df)
    if keys is None or not {"Date", "rank", "authorship_role"}.issubset(df.columns):
        return pd.DataFrame(columns=["Paper Date", "Rank_first", "Rank_last"])

    rank_first, rank_last = output_rank_columns(df)
    working = pd.DataFrame(
        {
            "Paper Key": keys,
            "Paper Date": df["Date"].map(parse_date),
            "Rank_first": rank_first,
            "Rank_last": rank_last,
        }
    )
    working = working[working["Paper Key"].notna()]

    def first_valid(series: pd.Series):
        valid = series.dropna()
        return valid.iloc[0] if not valid.empty else np.nan

    collapsed = (
        working.groupby("Paper Key", as_index=False)
        .agg(
            {
                "Paper Date": first_valid,
                "Rank_first": first_valid,
                "Rank_last": first_valid,
            }
        )
        .drop(columns=["Paper Key"])
    )
    return pd.DataFrame(collapsed)


def load_paper_rows(csv_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if OUTPUTS_ROOT in csv_path.parents:
        return collapse_author_rows(df)

    if "Date" not in df.columns:
        return pd.DataFrame(columns=["Paper Date", "Rank_first", "Rank_last"])

    rank_first, rank_last = rank_columns(df)
    return pd.DataFrame(
        {
            "Paper Date": df["Date"].map(parse_date),
            "Rank_first": rank_first,
            "Rank_last": rank_last,
        }
    )


def load_scores() -> pd.DataFrame:
    deadlines = load_review_deadlines()
    conference_order = load_deadline_conference_order()
    frames = []

    for csv_path in accepted_paper_files(conference_order):
        conference = conference_from_path(csv_path, conference_order)
        year = year_from_path(csv_path)
        if conference is None or year is None:
            continue
        if (conference, year) not in deadlines:
            continue

        df = pd.read_csv(csv_path)
        paper_rows = load_paper_rows(csv_path, df)
        if paper_rows.empty:
            continue

        review_deadline = deadlines[(conference, year)]
        paper_rows["Conference"] = conference
        paper_rows["Year"] = year
        paper_rows["Review deadline"] = review_deadline
        frames.append(
            paper_rows[
                [
                    "Conference",
                    "Year",
                    "Paper Date",
                    "Review deadline",
                    "Rank_first",
                    "Rank_last",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "Conference",
                "Year",
                "Paper Date",
                "Review deadline",
                "Rank_first",
                "Rank_last",
            ]
        )

    scores = pd.concat(frames, ignore_index=True)
    scores["Has Preprint Before Review Deadline"] = (
        scores["Paper Date"].notna()
        & (scores["Paper Date"] < scores["Review deadline"])
    )
    scores["No Preprint Before Review Deadline"] = ~scores[
        "Has Preprint Before Review Deadline"
    ]
    return scores


def no_preprint_ratio_in_rank_group(
    conference_year_rows: pd.DataFrame,
    rank_group_mask: pd.Series,
) -> float:
    if conference_year_rows.empty:
        return np.nan

    numerator = conference_year_rows.loc[
        rank_group_mask, "No Preprint Before Review Deadline"
    ].sum()
    return float(numerator / len(conference_year_rows))


def build_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Year"].isin(YEARS)].copy()
    df["Bottom Rank Threshold"] = df["Year"].map(BOTTOM_RANK_THRESHOLD)
    df["Top Rank Group"] = (df["Rank_first"] <= 20) | (df["Rank_last"] <= 20)
    df["Bottom Rank Group"] = (
        (df["Rank_first"] >= df["Bottom Rank Threshold"])
        | (df["Rank_last"] >= df["Bottom Rank Threshold"])
    )

    available_conferences = set(df["Conference"].dropna().unique())
    deadline_order = load_deadline_conference_order()
    ordered = PREFERRED_CONFERENCE_ORDER + [
        conference for conference in deadline_order if conference not in PREFERRED_CONFERENCE_ORDER
    ]
    conferences = [
        conference for conference in ordered if conference in available_conferences
    ]
    heatmap_df = pd.DataFrame(index=RANK_GROUPS)

    for conference in conferences:
        for year in YEARS:
            column = f"{conference}-{year}"
            subset = df[(df["Conference"] == conference) & (df["Year"] == year)]
            heatmap_df[column] = [
                no_preprint_ratio_in_rank_group(subset, subset["Top Rank Group"]),
                no_preprint_ratio_in_rank_group(subset, subset["Bottom Rank Group"]),
            ]

        heatmap_df[f"{conference}_space"] = np.nan

    return heatmap_df.iloc[:, :-1]


def ordered_conferences(df: pd.DataFrame) -> list[str]:
    available_conferences = set(df["Conference"].dropna().unique())
    deadline_order = load_deadline_conference_order()
    ordered = PREFERRED_CONFERENCE_ORDER + [
        conference for conference in deadline_order if conference not in PREFERRED_CONFERENCE_ORDER
    ]
    return [
        conference for conference in ordered if conference in available_conferences
    ]


def build_conference_heatmap_data(df: pd.DataFrame, conference: str) -> pd.DataFrame:
    df = df[df["Year"].isin(YEARS)].copy()
    df["Bottom Rank Threshold"] = df["Year"].map(BOTTOM_RANK_THRESHOLD)
    df["Top Rank Group"] = (df["Rank_first"] <= 20) | (df["Rank_last"] <= 20)
    df["Bottom Rank Group"] = (
        (df["Rank_first"] >= df["Bottom Rank Threshold"])
        | (df["Rank_last"] >= df["Bottom Rank Threshold"])
    )

    heatmap_df = pd.DataFrame(index=RANK_GROUPS)
    for year in YEARS:
        subset = df[(df["Conference"] == conference) & (df["Year"] == year)]
        heatmap_df[str(year)] = [
            no_preprint_ratio_in_rank_group(subset, subset["Top Rank Group"]),
            no_preprint_ratio_in_rank_group(subset, subset["Bottom Rank Group"]),
        ]
    return heatmap_df


def display_labels(columns: pd.Index) -> list[str]:
    return ["" if "space" in column else str(column) for column in columns]


def scientific_label(value: float) -> str:
    if pd.isna(value):
        return ""
    if value == 0:
        return "0 *\n10^0"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f} *\n10^{exponent}"


def scientific_annotations(heatmap_df: pd.DataFrame) -> pd.DataFrame:
    return heatmap_df.map(scientific_label)


def main() -> None:
    df = load_scores()
    conferences = ordered_conferences(df)
    conferences_per_row = 3
    rows = int(np.ceil(len(conferences) / conferences_per_row))

    fig, axes = plt.subplots(
        rows,
        conferences_per_row,
        figsize=(conferences_per_row * 5, rows * 3),
        squeeze=False,
    )
    cbar_ax = fig.add_axes((0.92, 0.18, 0.015, 0.64))

    for index, conference in enumerate(conferences):
        row = index // conferences_per_row
        col = index % conferences_per_row
        ax = axes[row][col]
        heatmap_df = build_conference_heatmap_data(df, conference)
        show_cbar = index == 0

        sns.heatmap(
            heatmap_df,
            ax=ax,
            annot=scientific_annotations(heatmap_df),
            fmt="",
            cmap="crest",
            linewidths=1,
            linecolor="white",
            vmin=0,
            vmax=1,
            cbar=show_cbar,
            cbar_ax=cbar_ax if show_cbar else None,
            cbar_kws={
                "label": "",
                "shrink": 0.8,
            },
            annot_kws={
                "fontsize": 20,
                "weight": "bold",
            },
            mask=heatmap_df.isna(),
        )
        ax.set_title(conference, fontsize=24, weight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=0,
            fontsize=24,
            fontweight="bold",
        )
        if conference in CONFERENCES_WITHOUT_YTICKS:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0,
                fontsize=24,
                fontweight="bold",
            )
        ax.set_facecolor("#fafafa")

    for index in range(len(conferences), rows * conferences_per_row):
        row = index // conferences_per_row
        col = index % conferences_per_row
        axes[row][col].axis("off")

    '''fig.suptitle(
        "No-Preprint Accepted Papers in Rank Groups",
        fontsize=18,
        weight="bold",
        y=0.98,
    )'''
    fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.08, wspace=0.35, hspace=0.5)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()