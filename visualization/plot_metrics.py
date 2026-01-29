import os
import re
import math
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _nice_step(raw_step: float) -> float:
    """Round raw_step to a 'nice' step in {1,2,5}*10^k."""
    if raw_step <= 0 or not math.isfinite(raw_step):
        return 1.0
    exp = math.floor(math.log10(raw_step))
    base = 10.0 ** exp
    mant = raw_step / base
    if mant <= 1:
        nice = 1.0
    elif mant <= 2:
        nice = 2.0
    elif mant <= 5:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * base
    return step if step > 0 else 1.0


def _set_adaptive_xticks(ax: plt.Axes, x_min: float, x_max: float) -> Tuple[float, float]:
    rng = float(x_max - x_min) if math.isfinite(x_max - x_min) else 0.0
    if rng <= 0:
        major_step = 1.0
        minor_step = 0.2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
        return major_step, minor_step

    target_major_ticks = 7
    raw_major = rng / target_major_ticks
    major_step = _nice_step(raw_major)
    minor_step = major_step / 5.0

    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_step))

    approx_major_ticks = rng / major_step
    if approx_major_ticks >= 9:
        ax.tick_params(axis="x", which="major", labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        ax.tick_params(axis="x", which="major", labelrotation=0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")

    return major_step, minor_step


def plot_metrics(
    df: pd.DataFrame,
    eta: Optional[float] = None,
    beta: Optional[float] = None,
    batch_size: Optional[int] = None,
    other_params: Optional[dict] = None,
) -> plt.Figure:
    """
    Plot batch_sharpness, lambda_max, stability bounds, and full_loss.
    """
    other_params = other_params or {}
    fig, ax = plt.subplots(figsize=(13, 7.5))
    y_candidates = []

    # Stability bounds
    if eta is not None and beta is not None:
        upper = 2 * (1 + beta) / eta
        lower = 2 * (1 - beta) / eta
        ax.axhline(y=upper, color="black", linestyle="--", linewidth=2.5,
                   label=r"$2(1+\beta)/\eta$")
        ax.axhline(y=lower, color="black", linestyle=":", linewidth=2.5,
                   label=r"$2(1-\beta)/\eta$")
        y_candidates.extend([upper, lower])

    # Batch sharpness
    batch_sharp = df[["step", "batch_sharpness"]].dropna()
    if not batch_sharp.empty:
        ax.plot(batch_sharp["step"], batch_sharp["batch_sharpness"],
                label="Batch Sharpness", color="#2CA02C",
                linewidth=2.8)
        y_candidates.append(batch_sharp["batch_sharpness"].max())

    # Lambda max
    lmax = df[["step", "lambda_max"]].dropna()
    if not lmax.empty:
        ax.plot(lmax["step"], lmax["lambda_max"],
                label=r"$\lambda_{\max}$", color="#1F77B4",
                linewidth=2.8)
        y_candidates.append(lmax["lambda_max"].max())

    ax.set_xlabel("Training Step", fontsize=18)
    ax.set_ylabel("Sharpness", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)

    if y_candidates:
        ax.set_ylim(0, 1.15 * max(y_candidates))

    # Loss (secondary axis)
    ax_loss = ax.twinx()
    loss = df[["step", "full_loss"]].dropna()
    if not loss.empty:
        ax_loss.plot(loss["step"], loss["full_loss"],
                     color="gray", alpha=0.18, linewidth=2.0,
                     label="Full Batch Loss")
        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("Loss (log scale)", fontsize=18)

    # Adaptive x ticks
    if "step" in df.columns and df["step"].notna().any():
        _set_adaptive_xticks(ax, float(df["step"].min()), float(df["step"].max()))

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_loss.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc="upper left", fontsize=14)

    plt.tight_layout()
    return fig


_B_REGEX = re.compile(r"(?:^|[^0-9])b(?:atch)?[_-]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)

def infer_batch_size_from_name(filename: str) -> Optional[int]:
    m = _B_REGEX.search(filename)
    return int(m.group(1)) if m else None


def load_results_txt(path: Path) -> pd.DataFrame:
    """Load results.txt format: comment header lines starting with #, then CSV data."""
    # Standard column names matching the comment header format
    columns = [
        "epoch", "step", "batch_loss", "full_loss", "lambda_max",
        "step_sharpness", "batch_sharpness", "gni", "accuracy"
    ]
    df = pd.read_csv(path, comment="#", header=None, names=columns, skipinitialspace=True)
    return df


def render_folder(
    input_dir: str,
    output_dir: str,
    *,
    eta: Optional[float] = None,
    beta: Optional[float] = None,
    other_params: Optional[Dict] = None,
    batch_size_override: Optional[int] = None,
    glob_pattern: str = "**/results.txt",
    dpi: int = 200,
) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_paths = sorted(in_path.glob(glob_pattern))
    if not data_paths:
        raise FileNotFoundError(f"No files matching '{glob_pattern}' found in {in_path.resolve()}")

    other_params = other_params or {}

    for data_path in data_paths:
        # Load based on file extension
        if data_path.suffix == ".txt":
            df = load_results_txt(data_path)
        else:
            df = pd.read_csv(data_path)

        if "step" in df.columns:
            df = df.sort_values("step")

        # Infer batch size from parent directory name (run folder)
        bs = batch_size_override or infer_batch_size_from_name(data_path.parent.name)

        fig = plot_metrics(df, eta=eta, beta=beta, batch_size=bs,
                           other_params=other_params)

        # Use parent directory name for output filename since all files are results.txt
        run_name = data_path.parent.name
        out_file = out_path / f"{run_name}_plot.png"
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(data_paths)} plots to {out_path.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training metrics from CSV files")
    parser.add_argument("input_dir", help="Directory containing run folders with results.txt files")
    parser.add_argument("output_dir", help="Directory to save plots")
    parser.add_argument("--eta", type=float, default=None, help="Learning rate (optional, for stability bounds)")
    parser.add_argument("--beta", type=float, default=None, help="Momentum (optional, for stability bounds)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--glob", default="**/results.txt", help="Glob pattern for data files (default: **/results.txt)")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for output images")

    args = parser.parse_args()

    render_folder(
        args.input_dir,
        args.output_dir,
        eta=args.eta,
        beta=args.beta,
        batch_size_override=args.batch_size,
        glob_pattern=args.glob,
        dpi=args.dpi,
    )

