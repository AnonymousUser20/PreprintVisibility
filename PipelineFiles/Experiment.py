from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_arxiv_mapping(pipeline_dir: Path, delay_seconds: float | None) -> None:
    script = pipeline_dir / "arXiv_mapping.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    cmd = [sys.executable, str(script)]
    if delay_seconds is not None:
        cmd.extend(["--delay-seconds", str(delay_seconds)])
    print("Running arXiv_mapping.py on all *_ranked.csv in project root...")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy *_preprint.csv from project root into Outputs/. "
            "Use --refresh-arxiv to fill Date via arXiv first."
        )
    )
    parser.add_argument(
        "--refresh-arxiv",
        action="store_true",
        help=(
            "Run arXiv_mapping.py before copying so *_preprint.csv have Date populated "
            "(slow: one API lookup per unique title)."
        ),
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=None,
        help="Only used with --refresh-arxiv: delay between unique arXiv requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_dir = Path(__file__).resolve().parent
    project_root = pipeline_dir.parent
    outputs_dir = project_root / "Outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_arxiv:
        run_arxiv_mapping(pipeline_dir, args.delay_seconds)

    preprint_files = sorted(project_root.glob("*_preprint.csv"))
    if not preprint_files:
        print(f"No *_preprint.csv files found in: {project_root}")
        return

    copied = 0
    for src in preprint_files:
        dst = outputs_dir / src.name
        shutil.copy2(src, dst)
        copied += 1
        print(f"Copied: {src} -> {dst}")

    print(f"Done. Copied {copied} file(s) to: {outputs_dir}")


if __name__ == "__main__":
    main()
