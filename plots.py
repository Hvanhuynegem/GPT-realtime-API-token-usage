import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# "Define" the model you want to plot
TARGET_MODEL = ["gpt-realtime-mini", "gpt-realtime","gpt-5.1-2025-11-13", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"]



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
    # Filter by model
    df_model = df[df["model"] == target_model].copy()
    if df_model.empty:
        raise RuntimeError(f"No runs found for model {target_model!r}")

    if "timestamp" not in df_model.columns or df_model["timestamp"].isna().all():
        # If no usable timestamps, just take the most recent experiment_id by name
        latest_exp_id = df_model["experiment_id"].iloc[-1]
    else:
        # For each experiment, take its max timestamp, then choose the latest one
        exp_times = df_model.groupby("experiment_id")["timestamp"].max()
        latest_exp_id = exp_times.idxmax()

    df_latest = df_model[df_model["experiment_id"] == latest_exp_id].copy()
    print(f"Using latest run for model {target_model!r}: experiment_id = {latest_exp_id}")
    return df_latest


def plot_boxplots(df: pd.DataFrame, group_by: str = "model", save_dir: Path | None = None) -> None:
    """
    Create box and whisker plots for
    - latency
    - token usage
    - cost

    Grouped by the given column (for example model or experiment_id).
    """
    # You can adjust which columns you want to show
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

    # Filter to only columns that actually exist
    latency_cols = [c for c in latency_cols if c in df.columns]
    token_cols = [c for c in token_cols if c in df.columns]
    has_cost = cost_col in df.columns

    # Latency boxplots
    if latency_cols:
        fig, axes = plt.subplots(1, len(latency_cols), figsize=(4 * len(latency_cols), 5), squeeze=False)
        axes = axes[0]

        for ax, col in zip(axes, latency_cols):
            df.boxplot(column=col, by=group_by, ax=ax)
            ax.set_title(col)
            ax.set_xlabel(group_by)
            ax.set_ylabel("milliseconds")

        fig.suptitle("Latency boxplots")
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / "latency_boxplots.png", dpi=200)

        plt.close(fig)


    # Token usage boxplots
    if token_cols:
        fig, axes = plt.subplots(1, len(token_cols), figsize=(4 * len(token_cols), 5), squeeze=False)
        axes = axes[0]

        for ax, col in zip(axes, token_cols):
            df.boxplot(column=col, by=group_by, ax=ax)
            ax.set_title(col)
            ax.set_xlabel(group_by)
            ax.set_ylabel("tokens")

        fig.suptitle("Token usage boxplots")
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / "tokens_boxplots.png", dpi=200)

        plt.close(fig)


    # Cost boxplot
    if has_cost:
        fig, ax = plt.subplots(figsize=(6, 5))
        df.boxplot(column=cost_col, by=group_by, ax=ax)
        ax.set_title("Total cost per call")
        ax.set_xlabel(group_by)
        ax.set_ylabel("USD")
        fig.suptitle("Cost boxplot")
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / "cost_boxplot.png", dpi=200)

        plt.close(fig)



def main():
    df_all = load_all_metrics("logs")
    # Run for each target model
    for model in TARGET_MODEL:
        # Keep only the latest run for the selected model
        df_latest = select_latest_run_for_model(df_all, model)

        # Determine folder of the latest experiment
        exp_id = df_latest["experiment_id"].iloc[0]
        output_dir = Path("logs") / exp_id
        output_dir.mkdir(exist_ok=True)

        # Produce and save the plots
        plot_boxplots(df_latest, group_by="experiment_id", save_dir=output_dir)


if __name__ == "__main__":
    main()
