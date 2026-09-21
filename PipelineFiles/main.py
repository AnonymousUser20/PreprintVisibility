from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(
    step_no: int,
    script_name: str,
    pipeline_dir: Path,
    extra_args: list[str] | None = None,
) -> None:
    script_path = pipeline_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\nStep {step_no}: Running {script_name} ...")
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)
    print(f"Step {step_no} complete: {script_name}")


def ensure_preprint_csvs_from_ranked(project_root: Path) -> None:
    """Experiment.py copies *_preprint.csv; bridge from *mapped_ranked.csv naming."""
    for ranked in sorted(project_root.glob("*mapped_ranked.csv")):
        if ranked.name.endswith("_preprint.csv"):
            continue
        dest = ranked.with_name(f"{ranked.stem}_preprint{ranked.suffix}")
        shutil.copy2(ranked, dest)
        print(f"Prepared *_preprint.csv for Experiment step: {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project pipeline steps.")
    parser.add_argument(
        "--skip-step-1",
        action="store_true",
        help="Skip Step 1 (country_mapping_pipeline.py).",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional explicit input author CSV paths for Step 1 only "
            "(forwarded to country_mapping_pipeline.py --inputs)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_dir = Path(__file__).resolve().parent
    project_root = pipeline_dir.parent

    if args.skip_step_1:
        print("\nSkipping Step 1: country_mapping_pipeline.py")
    else:
        step1_extra: list[str] | None = None
        if args.inputs:
            step1_extra = ["--inputs"] + [str(p.resolve()) for p in args.inputs]
        run_step(1, "country_mapping_pipeline.py", pipeline_dir, step1_extra)
    run_step(2, "rank_mapping_pipeline.py", pipeline_dir)
    run_step(3, "arXiv_mapping.py", pipeline_dir)
    ensure_preprint_csvs_from_ranked(project_root)
    run_step(4, "Experiment.py", pipeline_dir)

    print("\nAll pipeline steps completed successfully.")


if __name__ == "__main__":
    main()
