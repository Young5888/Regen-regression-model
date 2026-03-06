"""
Run full pipeline: 1. Download → 2. Data exploration → 3. Preprocessing → 4. Regression.
Outputs final results to results/run_YYYY-MM-DD_HH-MM-SS_results.txt

Checkpointing: download scripts skip if data is up to date (see raw_data/.checkpoints/).
Use --skip-download to skip downloads when you already have raw_data.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_SCRIPTS = [
    PROJECT_ROOT / "1_scripts" / "download_studies.py",
    PROJECT_ROOT / "1_scripts" / "download_sponsors.py",
    PROJECT_ROOT / "1_scripts" / "download_browse_conditions.py",
    PROJECT_ROOT / "1_scripts" / "download_interventions.py",
    PROJECT_ROOT / "1_scripts" / "download_eligibilities.py",
    PROJECT_ROOT / "1_scripts" / "download_calculated_values.py",
    PROJECT_ROOT / "1_scripts" / "download_facilities.py",
    PROJECT_ROOT / "1_scripts" / "download_countries.py",
    PROJECT_ROOT / "1_scripts" / "download_designs.py",
    PROJECT_ROOT / "1_scripts" / "download_design_groups.py",
    PROJECT_ROOT / "1_scripts" / "download_design_outcomes.py",
    PROJECT_ROOT / "1_scripts" / "download_browse_interventions.py",
]


def run_script(script_path: Path, step_name: str, quiet: bool = False) -> bool:
    """Run a Python script, return True if successful."""
    print(f"\n{step_name}")
    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        **kwargs,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download steps (use when raw_data already exists)",
    )
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = RESULTS_DIR / f"run_{run_timestamp}_results.txt"

    if args.skip_download:
        print("Skipping download steps (--skip-download)")

    # Step 1: Download (quiet — checkpointing handles skips)
    if not args.skip_download:
        print("\n1. Download")
        for script_path in DOWNLOAD_SCRIPTS:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                print("ERROR: 1. Download failed")
                sys.exit(1)

    # Steps 2–4 (quiet for 2 and 3; regression prints metrics)
    steps = [
        (PROJECT_ROOT / "2_data_exploration" / "run_all.py", "2. Data exploration", True),
        (PROJECT_ROOT / "3_preprocessing" / "preprocess.py", "3. Preprocessing", True),
        (PROJECT_ROOT / "4_regression" / "train_regression.py", "4. Regression", False),
    ]

    for script_path, step_name, quiet in steps:
        if not run_script(script_path, step_name, quiet=quiet):
            print(f"ERROR: {step_name} failed")
            sys.exit(1)

    # Collect final results (metrics only)
    regression_report = RESULTS_DIR / "regression_report.txt"
    if regression_report.exists():
        results_path.write_text(regression_report.read_text())
    print(f"\nFinal results saved to: {results_path}")


if __name__ == "__main__":
    main()
