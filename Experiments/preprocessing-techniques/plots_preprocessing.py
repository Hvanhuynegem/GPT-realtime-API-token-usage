# plots_preprocessing.py
# Input: PreprocessingTimes_YYYYMMDD_HHMMSS.csv
# Produces:
# - boxplot_execution_time.png (box-and-whisker per technique)
# - barchart_avg_time.png (avg per technique)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs/preprocessing-techniques")
PATTERN = "PreprocessingTimes_*.csv"

# Find latest CSV
csv_files = sorted(
    OUT_DIR.glob(PATTERN),
    key=lambda p: p.stat().st_mtime
)

if not csv_files:
    raise FileNotFoundError(f"No files matching {PATTERN} in {OUT_DIR}")

CSV_PATH = csv_files[-1]
print(f"Using latest file: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Box-and-whisker: execution time per image
techniques = sorted(df["technique"].unique().tolist())
data = [df[df["technique"] == t]["time_ms"].values for t in techniques]

plt.figure()
plt.boxplot(data, labels=techniques, showfliers=True)
plt.xlabel("preprocessing technique")
plt.ylabel("time in ms")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
out1 = Path(CSV_PATH).with_name("boxplot_execution_time.png")
plt.savefig(out1, dpi=200)
plt.close()

# Bar chart: average time per technique
avg = df.groupby("technique")["time_ms"].mean().sort_values()
plt.figure()
plt.bar(avg.index.tolist(), avg.values)
plt.xlabel("preprocessing technique")
plt.ylabel("average time in ms")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
out2 = Path(CSV_PATH).with_name("barchart_avg_time.png")
plt.savefig(out2, dpi=200)
plt.close()

print("Wrote:", out1)
print("Wrote:", out2)


# --- NEW: size ratio scatter plot ---
sizes_csv = sorted(OUT_DIR.glob("PreprocessingSizes_*.csv"), key=lambda p: p.stat().st_mtime)
if not sizes_csv:
    print("No PreprocessingSizes_*.csv found. Skipping size scatter plot.")
else:
    SIZES_PATH = sizes_csv[-1]
    print(f"Using latest size file: {SIZES_PATH}")
    ds = pd.read_csv(SIZES_PATH)

    plt.figure()
    for tech in sorted(ds["technique"].unique()):
        sub = ds[ds["technique"] == tech]
        # plt.scatter(sub["original_bytes"], sub["compression_ratio"], label=tech, s=10)
        plt.scatter(sub["processed_bytes"], sub["compression_ratio"], label=tech, s=10)

    plt.xlabel("processed size (bytes)")
    plt.ylabel("compression ratio (processed/original)")
    plt.legend()
    plt.tight_layout()

    out3 = Path(SIZES_PATH).with_name("scatter_size_ratio.png")
    plt.savefig(out3, dpi=200)
    plt.close()
    print("Wrote:", out3)