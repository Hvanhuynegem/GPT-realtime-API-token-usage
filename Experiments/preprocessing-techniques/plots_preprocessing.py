# # plots_preprocessing.py
# # Input:
# # - PreprocessingTimes_YYYYMMDD_HHMMSS.csv
# # - PreprocessingSizes_YYYYMMDD_HHMMSS.csv
# # Produces:
# # - boxplot_execution_time.png
# # - barchart_avg_time.png
# # - scatter_size_ratio.png
# # - barchart_avg_processed_bytes.png

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# OUT_DIR = Path("outputs/preprocessing-techniques")
# PATTERN = "PreprocessingTimes_*.csv"

# # Find latest CSV
# csv_files = sorted(
#     OUT_DIR.glob(PATTERN),
#     key=lambda p: p.stat().st_mtime
# )

# if not csv_files:
#     raise FileNotFoundError(f"No files matching {PATTERN} in {OUT_DIR}")

# CSV_PATH = csv_files[-1]
# print(f"Using latest file: {CSV_PATH}")

# df = pd.read_csv(CSV_PATH)

# # Box-and-whisker: execution time per image (excluding YOLOv12)
# BOXPLOT_EXCLUDE = "YoloV12SalientRoi+GlobalThumb"
# df_box = df[df["technique"] != BOXPLOT_EXCLUDE]

# techniques = sorted(df_box["technique"].unique().tolist())
# data = [df_box[df_box["technique"] == t]["time_ms"].values for t in techniques]

# plt.figure()
# plt.boxplot(data, labels=techniques, showfliers=True)
# plt.xlabel("Preprocessing technique")
# plt.ylabel("Time in ms")
# plt.xticks(rotation=30, ha="right")
# plt.tight_layout()

# out1 = Path(CSV_PATH).with_name("boxplot_execution_time.png")
# plt.savefig(out1, dpi=200)
# plt.close()

# # Bar chart: average time per technique
# avg = df.groupby("technique")["time_ms"].mean().sort_values()
# plt.figure()
# plt.bar(avg.index.tolist(), avg.values)
# plt.xlabel("Preprocessing technique")
# plt.ylabel("Average time in ms")
# plt.xticks(rotation=30, ha="right")
# plt.tight_layout()
# out2 = Path(CSV_PATH).with_name("barchart_avg_time.png")
# plt.savefig(out2, dpi=200)
# plt.close()

# print("Wrote:", out1)
# print("Wrote:", out2)

# # --- Size plots ---
# sizes_csv = sorted(OUT_DIR.glob("PreprocessingSizes_*.csv"), key=lambda p: p.stat().st_mtime)
# if not sizes_csv:
#     print("No PreprocessingSizes_*.csv found. Skipping size plots.")
# else:
#     SIZES_PATH = sizes_csv[-1]
#     print(f"Using latest size file: {SIZES_PATH}")
#     ds = pd.read_csv(SIZES_PATH)

#     # Normalize technique name for filtering
#     ds["technique_lower"] = ds["technique"].astype(str).str.lower()

#     # 1) Scatter: compression ratio vs original bytes, excluding BMP
#     ds_scatter = ds[~ds["technique_lower"].str.contains("bmp", na=False)].copy()

#     plt.figure()
#     for tech in sorted(ds_scatter["technique"].unique()):
#         sub = ds_scatter[ds_scatter["technique"] == tech]
#         plt.scatter(sub["original_binary_bytes"], sub["compression_ratio"], label=tech, s=10)

#     plt.xlabel("Original size (bytes)")
#     plt.ylabel("Compression factor (processed/original)")
#     plt.legend()
#     plt.tight_layout()

#     out3 = Path(SIZES_PATH).with_name("scatter_size_ratio.png")
#     plt.savefig(out3, dpi=200)
#     plt.close()
#     print("Wrote:", out3)

#     # 2) Bar chart: average processed bytes per technique (includes BMP by default)
    
#     # Exclude BMP from bar chart
#     ds_bar = ds[~ds["technique_lower"].str.contains("bmp", na=False)]
#     avg_processed = (
#         ds_bar.groupby("technique")["processed_binary_bytes"]
#         .mean()
#         .sort_values()
#     )


#     plt.figure()
#     plt.bar(avg_processed.index.tolist(), avg_processed.values)
#     plt.xlabel("Preprocessing technique")
#     plt.ylabel("Average processed bytes")
#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()

#     out4 = Path(SIZES_PATH).with_name("barchart_avg_processed_bytes.png")
#     plt.savefig(out4, dpi=200)
#     plt.close()
#     print("Wrote:", out4)
    
#     # Bar chart: standard deviation of time per technique
#     std = df.groupby("technique")["time_ms"].std().sort_values()

#     plt.figure()
#     plt.bar(std.index.tolist(), std.values)
#     plt.xlabel("Preprocessing technique")
#     plt.ylabel("Standard deviation of time in ms")
#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()

#     out_std = Path(CSV_PATH).with_name("barchart_std_time.png")
#     plt.savefig(out_std, dpi=200)
#     plt.close()

#     print("Wrote:", out_std)

#     # Bar chart: coefficient of variation (std / mean) per technique
#     stats = (
#         df.groupby("technique")["time_ms"]
#         .agg(["mean", "std"])
#     )

#     stats["cv"] = stats["std"] / stats["mean"]
#     stats = stats.sort_values("cv")

#     plt.figure()
#     plt.bar(stats.index.tolist(), stats["cv"].values)
#     plt.xlabel("Preprocessing technique")
#     plt.ylabel("Coefficient of variation (std / mean)")
#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()

#     out_cv = Path(CSV_PATH).with_name("barchart_cv_time.png")
#     plt.savefig(out_cv, dpi=200)
#     plt.close()

#     print("Wrote:", out_cv)


# plots_preprocessing.py
# Input (latest in outputs/preprocessing-techniques):
# - PreprocessingTimes_YYYYMMDD_HHMMSS.csv
# - PreprocessingSizes_YYYYMMDD_HHMMSS.csv
# - PreprocessingRois_YYYYMMDD_HHMMSS.csv   (NEW, produced by your C# update)
#
# Produces (in same folder as latest CSVs):
# - boxplot_execution_time.png
# - barchart_avg_time.png
# - scatter_size_ratio.png
# - barchart_avg_processed_bytes.png
# - barchart_std_time.png
# - barchart_cv_time.png
# - overlap_scatter_gaze_vs_saliency.png        (NEW)
# - overlap_boxplot_gaze_saliency.png           (NEW)
# - overlap_corr_heatmap_gaze_saliency.png      (NEW)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs/preprocessing-techniques")

# -----------------------------
# Helpers: overlap computations
# -----------------------------

def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b

def rect_iou(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    """IoU of two axis-aligned rectangles."""
    if any(v is None for v in [ax, ay, aw, ah, bx, by, bw, bh]):
        return float("nan")

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0.0, aw) * max(0.0, ah)
    b_area = max(0.0, bw) * max(0.0, bh)
    union = a_area + b_area - inter
    return _safe_div(inter, union)

def rect_overlap_fraction_of_a(ax, ay, aw, ah, bx, by, bw, bh) -> float:
    """Intersection area divided by area(A). Asymmetric overlap."""
    if any(v is None for v in [ax, ay, aw, ah, bx, by, bw, bh]):
        return float("nan")

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0.0, aw) * max(0.0, ah)
    return _safe_div(inter, a_area)

def pick_single_roi(df_tech: pd.DataFrame) -> pd.DataFrame:
    """
    If a technique produces multiple ROIs for the same image,
    select the largest ROI (by roi_box_w * roi_box_h) per image_id.
    """
    df = df_tech.copy()
    # If box dims missing, fallback to roi_pixels_w/h (still useful for size)
    df["box_w_eff"] = df["roi_box_w"].where(df["roi_box_w"].notna(), df["roi_pixels_w"])
    df["box_h_eff"] = df["roi_box_h"].where(df["roi_box_h"].notna(), df["roi_pixels_h"])
    df["area_eff"] = df["box_w_eff"].fillna(0) * df["box_h_eff"].fillna(0)

    df = df.sort_values(["image_id", "area_eff"], ascending=[True, False])
    return df.groupby("image_id", as_index=False).head(1)

def require_bbox_cols(df: pd.DataFrame, label: str) -> None:
    need = ["roi_x", "roi_y", "roi_box_w", "roi_box_h"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} ROI CSV is missing bbox columns {missing}. "
            "Your C# ROI CSV must include roi_x, roi_y, roi_box_w, roi_box_h. "
            "If your ROI objects do not expose these, add them to the ROI type and export them."
        )

# -----------------------------
# Timing plots (existing)
# -----------------------------

PATTERN = "PreprocessingTimes_*.csv"
csv_files = sorted(OUT_DIR.glob(PATTERN), key=lambda p: p.stat().st_mtime)

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

# -----------------------------
# Size plots (existing)
# -----------------------------

sizes_csv = sorted(OUT_DIR.glob("PreprocessingSizes_*.csv"), key=lambda p: p.stat().st_mtime)
if not sizes_csv:
    print("No PreprocessingSizes_*.csv found. Skipping size plots.")
else:
    SIZES_PATH = sizes_csv[-1]
    print(f"Using latest size file: {SIZES_PATH}")
    ds = pd.read_csv(SIZES_PATH)

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

    # 2) Bar chart: average processed bytes per technique (exclude BMP)
    ds_bar = ds[~ds["technique_lower"].str.contains("bmp", na=False)]
    avg_processed = ds_bar.groupby("technique")["processed_binary_bytes"].mean().sort_values()

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

        # Bar chart: coefficient of variation for processed size (std / mean) per technique
    size_stats = (
        ds_bar.groupby("technique")["processed_binary_bytes"]
        .agg(["mean", "std"])
    )

    # Avoid division by zero
    size_stats["cv"] = size_stats["std"] / size_stats["mean"].replace(0, pd.NA)
    size_stats = size_stats.dropna(subset=["cv"]).sort_values("cv")

    plt.figure()
    plt.bar(size_stats.index.tolist(), size_stats["cv"].values)
    plt.xlabel("Preprocessing technique")
    plt.ylabel("Processed size CV (std / mean)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_cv_size = Path(SIZES_PATH).with_name("barchart_cv_processed_bytes.png")
    plt.savefig(out_cv_size, dpi=200)
    plt.close()
    print("Wrote:", out_cv_size)


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
    stats = df.groupby("technique")["time_ms"].agg(["mean", "std"])
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

# -----------------------------
# NEW: ROI overlap analysis
# -----------------------------

roi_csv = sorted(OUT_DIR.glob("PreprocessingRois_*.csv"), key=lambda p: p.stat().st_mtime)
if not roi_csv:
    print("No PreprocessingRois_*.csv found. Skipping ROI overlap plots.")
else:
    ROI_PATH = roi_csv[-1]
    print(f"Using latest ROI file: {ROI_PATH}")
    dr = pd.read_csv(ROI_PATH)

    # Techniques to compare
    GAZE_TECH = "GazeRoi+GlobalThumb"
    SAL_TECH = "SalientRoi+GlobalThumb"

    gaze = dr[dr["technique"] == GAZE_TECH].copy()
    sal = dr[dr["technique"] == SAL_TECH].copy()

    if gaze.empty or sal.empty:
        print(f"Missing ROI rows for gaze or saliency. gaze={len(gaze)} sal={len(sal)}. Skipping overlap plots.")
    else:
        # Need bbox to compute overlap meaningfully
        require_bbox_cols(gaze, "Gaze")
        require_bbox_cols(sal, "Saliency")

        # Ensure numeric
        for col in ["roi_x", "roi_y", "roi_box_w", "roi_box_h"]:
            gaze[col] = pd.to_numeric(gaze[col], errors="coerce")
            sal[col] = pd.to_numeric(sal[col], errors="coerce")

        # If multiple ROIs per image: pick a single ROI (largest) for each technique
        gaze1 = pick_single_roi(gaze)
        sal1 = pick_single_roi(sal)

        # Inner join by image_id
        m = gaze1.merge(
            sal1,
            on="image_id",
            suffixes=("_gaze", "_sal"),
            how="inner"
        )

        if m.empty:
            print("No overlapping image_id rows between gaze and saliency. Skipping overlap plots.")
        else:
            # Compute:
            # - IoU (symmetric)
            # - overlap fraction gaze->sal (intersection / gaze area)
            # - overlap fraction sal->gaze (intersection / sal area)
            m["iou"] = m.apply(
                lambda r: rect_iou(
                    r["roi_x_gaze"], r["roi_y_gaze"], r["roi_box_w_gaze"], r["roi_box_h_gaze"],
                    r["roi_x_sal"],  r["roi_y_sal"],  r["roi_box_w_sal"],  r["roi_box_h_sal"]
                ),
                axis=1
            )

            m["gaze_in_sal"] = m.apply(
                lambda r: rect_overlap_fraction_of_a(
                    r["roi_x_gaze"], r["roi_y_gaze"], r["roi_box_w_gaze"], r["roi_box_h_gaze"],
                    r["roi_x_sal"],  r["roi_y_sal"],  r["roi_box_w_sal"],  r["roi_box_h_sal"]
                ),
                axis=1
            )

            m["sal_in_gaze"] = m.apply(
                lambda r: rect_overlap_fraction_of_a(
                    r["roi_x_sal"],  r["roi_y_sal"],  r["roi_box_w_sal"],  r["roi_box_h_sal"],
                    r["roi_x_gaze"], r["roi_y_gaze"], r["roi_box_w_gaze"], r["roi_box_h_gaze"]
                ),
                axis=1
            )

            # Clean invalid
            m = m.dropna(subset=["iou", "gaze_in_sal", "sal_in_gaze"]).copy()

            # Correlations (Pearson) between the two asymmetric overlaps
            corr = m[["gaze_in_sal", "sal_in_gaze", "iou"]].corr(method="pearson")

            # 1) Scatter plot: gaze_in_sal vs sal_in_gaze
            plt.figure()
            plt.scatter(m["gaze_in_sal"], m["sal_in_gaze"], s=12)
            plt.xlabel("Overlap fraction: (Gaze ROI ∩ Saliency ROI) / Gaze ROI")
            plt.ylabel("Overlap fraction: (Gaze ROI ∩ Saliency ROI) / Saliency ROI")
            plt.tight_layout()

            out_overlap_scatter = Path(ROI_PATH).with_name("overlap_scatter_gaze_vs_saliency.png")
            plt.savefig(out_overlap_scatter, dpi=250)
            plt.close()
            print("Wrote:", out_overlap_scatter)

            # 2) Boxplot: distributions of overlaps + IoU
            plt.figure()
            plt.boxplot(
                [m["gaze_in_sal"].values, m["sal_in_gaze"].values, m["iou"].values],
                labels=["Gaze ROI in Saliency ROI", "Saliency ROI in Gaze ROI", "Intersection over Union"],
                showfliers=True
            )
            plt.ylabel("Overlap (0..1)")
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()

            out_overlap_box = Path(ROI_PATH).with_name("overlap_boxplot_gaze_saliency.png")
            plt.savefig(out_overlap_box, dpi=250)
            plt.close()
            print("Wrote:", out_overlap_box)

            # 3) Correlation heatmap (no seaborn)
            plt.figure()
            mat = corr.values
            plt.imshow(mat, aspect="auto")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=20, ha="right")
            plt.yticks(range(len(corr.index)), corr.index)

            # annotate cells
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")

            plt.tight_layout()
            out_corr = Path(ROI_PATH).with_name("overlap_corr_heatmap_gaze_saliency.png")
            plt.savefig(out_corr, dpi=250)
            plt.close()
            print("Wrote:", out_corr)

            # Optional: print quick stats for your log
            print()
            print(f"Overlap stats (n={len(m)} matched images):")
            print(f"- mean Intersection over Union:     {m['iou'].mean():.3f}")
            print(f"- mean gaze_in_sal:                 {m['gaze_in_sal'].mean():.3f}")
            print(f"- mean sal_in_gaze:                 {m['sal_in_gaze'].mean():.3f}")
            print("Pearson correlation matrix:")
            print(corr.round(3))


# -----------------------------
# NEW: YOLOv12 ROI count plot
# -----------------------------

roi_csv = sorted(OUT_DIR.glob("PreprocessingRois_*.csv"), key=lambda p: p.stat().st_mtime)
if not roi_csv:
    print("No PreprocessingRois_*.csv found. Skipping YOLO ROI count plot.")
else:
    ROI_PATH = roi_csv[-1]
    dr = pd.read_csv(ROI_PATH)

    YOLO_TECH = "YoloV12SalientRoi+GlobalThumb"
    yolo = dr[dr["technique"] == YOLO_TECH].copy()

    if yolo.empty:
        print("No YOLOv12 ROI rows found. Skipping YOLO ROI count plot.")
    else:
        # Count ROIs per image
        roi_counts = (
            yolo.groupby("image_id")
            .size()
            .values
        )

        plt.figure()
        plt.boxplot(roi_counts, showfliers=True)
        plt.ylabel("Number of ROIs per image")
        plt.xticks([1], ["YOLOv12"])
        plt.tight_layout()

        out_yolo = Path(ROI_PATH).with_name("boxplot_yolov12_roi_count.png")
        plt.savefig(out_yolo, dpi=250)
        plt.close()

        print("Wrote:", out_yolo)
        print(f"YOLOv12 ROI count stats: "
              f"min={roi_counts.min()}, "
              f"median={int(pd.Series(roi_counts).median())}, "
              f"mean={roi_counts.mean():.2f}, "
              f"max={roi_counts.max()}")

