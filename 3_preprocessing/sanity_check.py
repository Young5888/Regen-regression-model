"""
Sanity check on clean_data: duration_days min, max, median, mean, quantiles.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DATA = PROJECT_ROOT / "clean_data"
OUTPUT_DIR = CLEAN_DATA


def main() -> None:
    studies = pd.read_csv(CLEAN_DATA / "studies.csv", low_memory=False)
    duration = pd.to_numeric(studies["duration_days"], errors="coerce").dropna()

    stats = {
        "count": len(duration),
        "min": duration.min(),
        "max": duration.max(),
        "mean": round(duration.mean(), 1),
        "median": round(duration.median(), 1),
        "std": round(duration.std(), 1),
        "q1 (25%)": duration.quantile(0.25),
        "q2 (50%)": duration.quantile(0.50),
        "q3 (75%)": duration.quantile(0.75),
        "q5%": duration.quantile(0.05),
        "q95%": duration.quantile(0.95),
    }

    lines = []
    lines.append("=" * 50)
    lines.append("DURATION SANITY CHECK (primary_completion_date - start_date)")
    lines.append("=" * 50)
    lines.append(f"Rows with valid duration: {stats['count']:,}")
    lines.append("")
    lines.append("Summary statistics (days):")
    lines.append(f"  min:    {stats['min']:,.0f}")
    lines.append(f"  max:    {stats['max']:,.0f}")
    lines.append(f"  mean:   {stats['mean']:,.1f}")
    lines.append(f"  median: {stats['median']:,.1f}")
    lines.append(f"  std:    {stats['std']:,.1f}")
    lines.append("")
    lines.append("Quantiles:")
    lines.append(f"  5%:  {stats['q5%']:,.0f} days")
    lines.append(f"  25%: {stats['q1 (25%)']:,.0f} days")
    lines.append(f"  50%: {stats['q2 (50%)']:,.0f} days")
    lines.append(f"  75%: {stats['q3 (75%)']:,.0f} days")
    lines.append(f"  95%: {stats['q95%']:,.0f} days")
    lines.append("=" * 50)

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "duration_sanity_check.txt").write_text(report)
    print(f"\nSaved to {OUTPUT_DIR / 'duration_sanity_check.txt'}")


if __name__ == "__main__":
    main()
