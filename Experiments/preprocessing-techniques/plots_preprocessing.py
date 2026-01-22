# plots_preprocessing.py
# Input:
# - PreprocessingTimes_YYYYMMDD_HHMMSS.csv
# - PreprocessingSizes_YYYYMMDD_HHMMSS.csv
# Produces:
# - boxplot_execution_time.png
# - barchart_avg_time.png
# - scatter_size_ratio.png
# - barchart_avg_processed_bytes.png

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

# Box-and-whisker: execution time per image (excluding YOLOv12)
BOXPLOT_EXCLUDE = "YoloV12SalientRoi+GlobalThumb"
df_box = df[df["technique"] != BOXPLOT_EXCLUDE]

techniques = sorted(df_box["technique"].unique().tolist())
data = [df_box[df_box["technique"] == t]["time_ms"].values for t in techniques]

plt.figure()
plt.boxplot(data, labels=techniques, showfliers=True)
plt.xlabel("Preprocessing technique")
plt.ylabel("Time in ms")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

out1 = Path(CSV_PATH).with_name("boxplot_execution_time.png")
plt.savefig(out1, dpi=200)
plt.close()

# Bar chart: average time per technique
avg = df.groupby("technique")["time_ms"].mean().sort_values()
plt.figure()
plt.bar(avg.index.tolist(), avg.values)
plt.xlabel("Preprocessing technique")
plt.ylabel("Average time in ms")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
out2 = Path(CSV_PATH).with_name("barchart_avg_time.png")
plt.savefig(out2, dpi=200)
plt.close()

print("Wrote:", out1)
print("Wrote:", out2)

# --- Size plots ---
sizes_csv = sorted(OUT_DIR.glob("PreprocessingSizes_*.csv"), key=lambda p: p.stat().st_mtime)
if not sizes_csv:
    print("No PreprocessingSizes_*.csv found. Skipping size plots.")
else:
    SIZES_PATH = sizes_csv[-1]
    print(f"Using latest size file: {SIZES_PATH}")
    ds = pd.read_csv(SIZES_PATH)

    # Normalize technique name for filtering
    ds["technique_lower"] = ds["technique"].astype(str).str.lower()

    # 1) Scatter: compression ratio vs original bytes, excluding BMP
    ds_scatter = ds[~ds["technique_lower"].str.contains("bmp", na=False)].copy()

    plt.figure()
    for tech in sorted(ds_scatter["technique"].unique()):
        sub = ds_scatter[ds_scatter["technique"] == tech]
        plt.scatter(sub["original_binary_bytes"], sub["compression_ratio"], label=tech, s=10)

    plt.xlabel("Original size (bytes)")
    plt.ylabel("Compression factor (processed/original)")
    plt.legend()
    plt.tight_layout()

    out3 = Path(SIZES_PATH).with_name("scatter_size_ratio.png")
    plt.savefig(out3, dpi=200)
    plt.close()
    print("Wrote:", out3)

    # 2) Bar chart: average processed bytes per technique (includes BMP by default)
    
    # Exclude BMP from bar chart
    ds_bar = ds[~ds["technique_lower"].str.contains("bmp", na=False)]
    avg_processed = (
        ds_bar.groupby("technique")["processed_binary_bytes"]
        .mean()
        .sort_values()
    )


    plt.figure()
    plt.bar(avg_processed.index.tolist(), avg_processed.values)
    plt.xlabel("Preprocessing technique")
    plt.ylabel("Average processed bytes")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out4 = Path(SIZES_PATH).with_name("barchart_avg_processed_bytes.png")
    plt.savefig(out4, dpi=200)
    plt.close()
    print("Wrote:", out4)
    
    # Bar chart: standard deviation of time per technique
    std = df.groupby("technique")["time_ms"].std().sort_values()

    plt.figure()
    plt.bar(std.index.tolist(), std.values)
    plt.xlabel("Preprocessing technique")
    plt.ylabel("Standard deviation of time in ms")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_std = Path(CSV_PATH).with_name("barchart_std_time.png")
    plt.savefig(out_std, dpi=200)
    plt.close()

    print("Wrote:", out_std)

    # Bar chart: coefficient of variation (std / mean) per technique
    stats = (
        df.groupby("technique")["time_ms"]
        .agg(["mean", "std"])
    )

    stats["cv"] = stats["std"] / stats["mean"]
    stats = stats.sort_values("cv")

    plt.figure()
    plt.bar(stats.index.tolist(), stats["cv"].values)
    plt.xlabel("Preprocessing technique")
    plt.ylabel("Coefficient of variation (std / mean)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_cv = Path(CSV_PATH).with_name("barchart_cv_time.png")
    plt.savefig(out_cv, dpi=200)
    plt.close()

    print("Wrote:", out_cv)

