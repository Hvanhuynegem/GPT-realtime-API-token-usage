import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Models to process (latest run per model)
TARGET_MODELS = [
    "gpt-realtime-mini",
    "gpt-realtime",
    "gpt-5.1-2025-11-13",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
]


def _pick_payload_bytes_column(df: pd.DataFrame) -> str | None:
    candidates = [
        # Most realistic network payload first
        "data_url_bytes_utf8_total",      # actual string sent
        "processed_binary_bytes",         # compressed binary
        "original_binary_bytes",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

import re

def _coco_image_id_to_sample_id(image_id: str) -> str:
    """
    Converts: COCO_val2014_000000000338.jpg -> 338002
    If the pattern does not match, returns the original string (so you can see failures).
    """
    m = re.search(r"COCO_(?:val|train)2014_(\d+)\.jpg$", str(image_id))
    if not m:
        return str(image_id)

    coco_num = int(m.group(1))  # 000000000338 -> 338
    return f"{coco_num}002"


def load_latest_preprocessing_sizes(preproc_root: str) -> pd.DataFrame:
    root = Path(preproc_root)

    files = sorted(root.glob("PreprocessingSizes_*.csv"))
    if not files:
        raise RuntimeError(f"No PreprocessingSizes_*.csv found in {root}")

    latest = files[-1]
    print(f"Using preprocessing sizes file: {latest.name}")

    df = pd.read_csv(latest)

    df["image_id"] = df["image_id"].astype(str)
    df["technique"] = df["technique"].astype(str)

    # Create sample_id to match metrics.csv
    df["sample_id"] = df["image_id"].map(_coco_image_id_to_sample_id)

    # Debug: show how many conversions worked
    bad = df["sample_id"].eq(df["image_id"]).mean() * 100
    if bad > 0:
        print(f"Warning: {bad:.1f}% of image_id values did not match expected COCO pattern.")

    df["sample_id"] = df["sample_id"].astype(str)

    return df



def load_all_metrics(log_root: str = "logs") -> pd.DataFrame:
    """
    Walk through the logs folder and collect all metrics.csv files.
    For each experiment, also read config.json and attach metadata columns.
    """
    log_root = Path(log_root)
    all_rows = []

    for exp_dir in log_root.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics_path = exp_dir / "metrics.csv"
        config_path = exp_dir / "config.json"

        if not metrics_path.exists() or not config_path.exists():
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        df = pd.read_csv(metrics_path)

        if "preprocessor" not in df.columns:
            df["preprocessor"] = "Unknown"

        df["experiment_id"] = cfg.get("experiment_id", exp_dir.name)
        df["model"] = cfg.get("model", "unknown")
        df["dataset_name"] = cfg.get("dataset_name", "unknown")
        df["timestamp"] = cfg.get("timestamp")

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"No metrics.csv files found in {log_root.resolve()}")

    df_all = pd.concat(all_rows, ignore_index=True)

    if "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")

    return df_all


def select_latest_run_for_model(df: pd.DataFrame, target_model: str) -> pd.DataFrame:
    """
    Filter to a single model and return only the latest experiment_id
    based on the timestamp column.
    """
    df_model = df[df["model"] == target_model].copy()
    if df_model.empty:
        raise RuntimeError(f"No runs found for model {target_model!r}")

    if "timestamp" not in df_model.columns or df_model["timestamp"].isna().all():
        latest_exp_id = df_model["experiment_id"].iloc[-1]
    else:
        exp_times = df_model.groupby("experiment_id")["timestamp"].max()
        latest_exp_id = exp_times.idxmax()

    df_latest = df_model[df_model["experiment_id"] == latest_exp_id].copy()
    print(f"Using latest run for model {target_model!r}: experiment_id = {latest_exp_id}")
    return df_latest


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def _compute_sample_info(df: pd.DataFrame, group_by: str) -> str:
    """
    Returns:
    'Samples per technique: X'
    where X is the average number of samples per preprocessing technique.
    """
    counts = df[group_by].astype(str).value_counts(dropna=False)

    if counts.empty:
        return "Samples per technique: 0"

    avg_samples = int(round(counts.mean()))
    return f"Samples per technique: {avg_samples}"


def _plot_payload_vs_latency(
    df: pd.DataFrame,
    group_by: str,
    save_dir: Path | None,
    title_prefix: str,
) -> None:
    """
    Scatter plot: payload bytes vs end_to_end_ms, with a simple linear trend line.
    """
    if "end_to_end_ms" not in df.columns:
        return

    bytes_col = _pick_payload_bytes_column(df)
    if bytes_col is None:
        print("Skipping payload-vs-latency plot: no payload-bytes column found.")
        return

    d = df[[bytes_col, "end_to_end_ms", group_by]].copy()
    d[bytes_col] = pd.to_numeric(d[bytes_col], errors="coerce")
    d["end_to_end_ms"] = pd.to_numeric(d["end_to_end_ms"], errors="coerce")
    d = d.dropna(subset=[bytes_col, "end_to_end_ms"])

    if d.empty:
        print("Skipping payload-vs-latency plot: no valid numeric rows.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter by group (label only once per group)
    for label, sub in d.groupby(group_by, dropna=False):
        ax.scatter(sub[bytes_col], sub["end_to_end_ms"], s=10, alpha=0.6, label=str(label))

    # Linear trend line on all points (for a quick relationship view)
    x = d[bytes_col].to_numpy()
    y = d["end_to_end_ms"].to_numpy()
    if len(x) >= 2 and np.nanstd(x) > 0:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        ax.plot(x_line, y_line, linewidth=2)

    ax.set_title(f"Network payload size vs observed end-to-end latency")
    ax.set_xlabel(f"{bytes_col} (bytes)")
    ax.set_ylabel("end_to_end_ms (ms)")

    # Avoid huge legends if you have many preprocessors; still useful if you have a few.
    if d[group_by].nunique() <= 12:
        ax.legend(fontsize=8, loc="best")

    sample_info = _compute_sample_info(df, group_by)
    fig.suptitle(f"{title_prefix}Payload vs latency\n{sample_info}".strip(), fontsize=11)

    plt.tight_layout()

    if save_dir:
        fig.savefig(save_dir / "payload_vs_end_to_end_ms.png", dpi=200)

    plt.close(fig)


def plot_boxplots_separate_images(
    df: pd.DataFrame,
    group_by: str,
    save_dir: Path | None = None,
    title_prefix: str = "",
) -> None:
    """
    Create boxplots and save each metric as its own image:
    - latency: one image per latency metric
    - tokens:  one image per token metric
    - cost:    one image

    Also adds:
    - bar chart of average end_to_end_ms per preprocessing technique
    - scatter: network payload size vs end_to_end_ms
    """
    latency_cols = [
        "end_to_end_ms",
        "send_to_response_created_ms",
        "response_created_to_first_token_ms",
        "first_token_to_done_ms",
    ]
    token_cols = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "text_input_tokens",
        "image_input_tokens",
    ]
    cost_col = "total_cost_usd"

    latency_cols = [c for c in latency_cols if c in df.columns]
    token_cols = [c for c in token_cols if c in df.columns]
    has_cost = cost_col in df.columns

    if group_by not in df.columns:
        raise RuntimeError(
            f"group_by column {group_by!r} not in dataframe columns: {list(df.columns)}"
        )

    df = df.copy()
    df[group_by] = df[group_by].astype(str)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # -----------------
    # Latency: one fig per metric
    # -----------------
    for col in latency_cols:
        fig, ax = plt.subplots(figsize=(7, 5))
        df.boxplot(column=col, by=group_by, ax=ax)
        ax.set_title(col)
        ax.set_xlabel(group_by)
        ax.set_ylabel("milliseconds")
        ax.tick_params(axis="x", rotation=60)

        sample_info = _compute_sample_info(df, group_by)
        fig.suptitle(f"{title_prefix}{col} boxplot\n{sample_info}".strip(), fontsize=11)

        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / f"latency_boxplot_{_safe_filename(col)}.png", dpi=200)

        plt.close(fig)

    # -----------------
    # Tokens: one fig per metric
    # -----------------
    for col in token_cols:
        fig, ax = plt.subplots(figsize=(7, 5))
        df.boxplot(column=col, by=group_by, ax=ax)
        ax.set_title(col)
        ax.set_xlabel(group_by)
        ax.set_ylabel("tokens")
        ax.tick_params(axis="x", rotation=60)

        sample_info = _compute_sample_info(df, group_by)
        fig.suptitle(f"{title_prefix}{col} boxplot\n{sample_info}".strip(), fontsize=11)

        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / f"tokens_boxplot_{_safe_filename(col)}.png", dpi=200)

        plt.close(fig)

    # -----------------
    # Cost: one fig
    # -----------------
    if has_cost:
        fig, ax = plt.subplots(figsize=(7, 5))
        df.boxplot(column=cost_col, by=group_by, ax=ax)
        ax.set_title("Total cost per call")
        ax.set_xlabel(group_by)
        ax.set_ylabel("USD")
        ax.tick_params(axis="x", rotation=60)

        sample_info = _compute_sample_info(df, group_by)
        fig.suptitle(f"{title_prefix}Cost boxplot\n{sample_info}".strip(), fontsize=11)

        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / "cost_boxplot.png", dpi=200)

        plt.close(fig)

    # -----------------
    # Extra plot: average end_to_end per preprocessing technique
    # -----------------
    if "end_to_end_ms" in df.columns:
        means = (
            df.groupby(group_by, dropna=False)["end_to_end_ms"]
            .mean()
            .sort_values(ascending=True)
        )

        print("Average end_to_end_ms per preprocessing technique:")
        for technique, mean_ms in means.items():
            print(f"{technique} - {mean_ms:.2f} ms")

        fig, ax = plt.subplots(figsize=(9, 5))
        means.plot(kind="bar", ax=ax)
        ax.set_title("Average end_to_end_ms per preprocessing technique")
        ax.set_xlabel(group_by)
        ax.set_ylabel("milliseconds")
        ax.tick_params(axis="x", rotation=60)

        sample_info = _compute_sample_info(df, group_by)
        fig.suptitle(f"{title_prefix}Average end_to_end_ms\n{sample_info}".strip(), fontsize=11)

        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / "avg_end_to_end_ms_by_preprocessor.png", dpi=200)

        plt.close(fig)

    # -----------------
    # NEW: payload size vs latency scatter
    # -----------------
    _plot_payload_vs_latency(df, group_by=group_by, save_dir=save_dir, title_prefix=title_prefix)


def plot_preprocessors_for_latest_run_per_model(logs_root: str = "logs") -> None:
    """
    For each model in TARGET_MODELS:
      - pick the latest experiment
      - plot metrics grouped by preprocessor
      - save under logs/<experiment_id>/plots_by_preprocessor/
    """
    df_all = load_all_metrics(logs_root)

    for model in TARGET_MODELS:
        df_latest = select_latest_run_for_model(df_all, model)

        # Load preprocessing sizes
        sizes_df = load_latest_preprocessing_sizes(
            "outputs/preprocessing-techniques/sample-visualization_20260210_135855"
        )

        # Align naming (your metrics uses "preprocessor")
        sizes_df = sizes_df.rename(columns={"technique": "preprocessor"})

        # Merge on sample_id + preprocessor
        df_latest["sample_id"] = df_latest["sample_id"].astype(str)

        df_latest = df_latest.merge(
            sizes_df[
                [
                    "sample_id",
                    "preprocessor",
                    "processed_binary_bytes",
                    "data_url_bytes_utf8_total",
                ]
            ],
            on=["sample_id", "preprocessor"],
            how="left",
        )


        exp_id = df_latest["experiment_id"].iloc[0]
        base_dir = Path(logs_root) / exp_id
        out_dir = base_dir / "plots_by_preprocessor"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_boxplots_separate_images(
            df_latest,
            group_by="preprocessor",
            save_dir=out_dir,
            title_prefix=f"{model} | ",
        )


def plot_models_by_preprocessor_across_latest_runs(logs_root: str = "logs") -> None:
    """
    Builds a combined dataframe containing ONLY the latest run of each model,
    then produces plots where the grouping label is "preprocessor | model".
    """
    df_all = load_all_metrics(logs_root)

    latest_dfs = []
    for model in TARGET_MODELS:
        try:
            latest_dfs.append(select_latest_run_for_model(df_all, model))
        except RuntimeError as e:
            print(f"Skipping {model!r}: {e}")

    if not latest_dfs:
        raise RuntimeError("No latest runs found for any target models")

    df_latest_all = pd.concat(latest_dfs, ignore_index=True)

    df_latest_all["preprocessor_model"] = (
        df_latest_all["preprocessor"].astype(str) + " | " + df_latest_all["model"].astype(str)
    )

    out_dir = Path(logs_root) / "_combined_latest" / "plots_by_preprocessor_and_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_boxplots_separate_images(
        df_latest_all,
        group_by="preprocessor_model",
        save_dir=out_dir,
        title_prefix="Combined latest runs | ",
    )


def main():
    plot_preprocessors_for_latest_run_per_model("logs")
    # plot_models_by_preprocessor_across_latest_runs("logs")


if __name__ == "__main__":
    main()
