from pathlib import Path

from worldmap_common import build_country_counter, plot_world


def main() -> None:
    project_root = Path(__file__).resolve().parent
    counter = build_country_counter(project_root, mode="gt180")
    output_png = project_root / "greater than 180 days before CfP.png"
    plot_world(counter, "(f)", output_png)


if __name__ == "__main__":
    main()
