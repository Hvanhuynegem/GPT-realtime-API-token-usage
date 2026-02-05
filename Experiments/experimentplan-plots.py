import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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

        # Load config metadata
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Load metrics
        df = pd.read_csv(metrics_path)

        # Ensure new-format column exists (old logs will not have it)
        if "preprocessor" not in df.columns:
            df["preprocessor"] = "Unknown"

        # Attach metadata columns for later grouping
        df["experiment_id"] = cfg.get("experiment_id", exp_dir.name)
        df["model"] = cfg.get("model", "unknown")
        df["dataset_name"] = cfg.get("dataset_name", "unknown")
        df["timestamp"] = cfg.get("timestamp")

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"No metrics.csv files found in {log_root.resolve()}")

    df_all = pd.concat(all_rows, ignore_index=True)

    # Parse timestamp if present
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


def plot_boxplots(
    df: pd.DataFrame,
    group_by: str,
    save_dir: Path | None = None,
    title_prefix: str = "",
) -> None:
    """
    Create box and whisker plots for:
    - latency
    - token usage
    - cost

    Grouped by the given column (e.g. "preprocessor").
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

    # Make grouping column exist
    if group_by not in df.columns:
        raise RuntimeError(f"group_by column {group_by!r} not in dataframe columns: {list(df.columns)}")

    # Helpful: stable order of categories
    df[group_by] = df[group_by].astype(str)

    # Latency
    if latency_cols:
        fig, axes = plt.subplots(1, len(latency_cols), figsize=(4 * len(latency_cols), 5), squeeze=False)
        axes = axes[0]

        for ax, col in zip(axes, latency_cols):
            df.boxplot(column=col, by=group_by, ax=ax)
            ax.set_title(col)
            ax.set_xlabel(group_by)
            ax.set_ylabel("milliseconds")
            ax.tick_params(axis="x", rotation=60)

        fig.suptitle(f"{title_prefix}Latency boxplots".strip())
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / "latency_boxplots.png", dpi=200)

        plt.close(fig)

    # Tokens
    if token_cols:
        fig, axes = plt.subplots(1, len(token_cols), figsize=(4 * len(token_cols), 5), squeeze=False)
        axes = axes[0]

        for ax, col in zip(axes, token_cols):
            df.boxplot(column=col, by=group_by, ax=ax)
            ax.set_title(col)
            ax.set_xlabel(group_by)
            ax.set_ylabel("tokens")
            ax.tick_params(axis="x", rotation=60)

        fig.suptitle(f"{title_prefix}Token usage boxplots".strip())
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / "tokens_boxplots.png", dpi=200)

        plt.close(fig)

    # Cost
    if has_cost:
        fig, ax = plt.subplots(figsize=(7, 5))
        df.boxplot(column=cost_col, by=group_by, ax=ax)
        ax.set_title("Total cost per call")
        ax.set_xlabel(group_by)
        ax.set_ylabel("USD")
        ax.tick_params(axis="x", rotation=60)

        fig.suptitle(f"{title_prefix}Cost boxplot".strip())
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / "cost_boxplot.png", dpi=200)

        plt.close(fig)


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

        exp_id = df_latest["experiment_id"].iloc[0]
        base_dir = Path(logs_root) / exp_id
        out_dir = base_dir / "plots_by_preprocessor"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Group by preprocessing technique
        plot_boxplots(
            df_latest,
            group_by="preprocessor",
            save_dir=out_dir,
            title_prefix=f"{model} | ",
        )


# -----------------------------
# Optional: compare MODELS per PREPROCESSOR across all latest runs
# -----------------------------
def plot_models_by_preprocessor_across_latest_runs(logs_root: str = "logs") -> None:
    """
    Builds a combined dataframe containing ONLY the latest run of each model,
    then produces plots where the grouping label is "preprocessor | model".
    This lets you compare models within each preprocessor (in one figure).
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

    # Composite group label: preprocessor + model
    df_latest_all["preprocessor_model"] = (
        df_latest_all["preprocessor"].astype(str) + " | " + df_latest_all["model"].astype(str)
    )

    out_dir = Path(logs_root) / "_combined_latest" / "plots_by_preprocessor_and_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_boxplots(
        df_latest_all,
        group_by="preprocessor_model",
        save_dir=out_dir,
        title_prefix="Combined latest runs | ",
    )


def main():
    plot_preprocessors_for_latest_run_per_model("logs")

    # Uncomment if you also want the combined comparison plots
    # plot_models_by_preprocessor_across_latest_runs("logs")


if __name__ == "__main__":
    main()
