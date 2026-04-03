"""
Plot cross-validation training/validation loss curves for a paper.

Reads TensorBoard event files from fold_0/ ... fold_N/ inside a crossval save_dir,
then plots mean ± std across folds with a shaded confidence band.

Usage:
    python plot_crossval_loss.py --save_dir final_training/window/config1 --out loss_plot.pdf

Requirements:
    pip install tensorboard scipy matplotlib
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── TensorBoard reader ────────────────────────────────────────────────────────

def read_tb_scalar(event_file_dir, tag):
    """
    Read all (step, value) pairs for a given tag from a TensorBoard event directory.
    Checks both 'scalars' and 'tensors' (the latter is used by logger.logkv / log_loss).
    Returns a sorted (N,) array of steps and values, or (None, None) if tag not found.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(event_file_dir, size_guidance={"scalars": 0, "tensors": 0})
    ea.Reload()

    # Try scalars first, then tensors
    if tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events], dtype=np.float32)
        values = np.array([e.value for e in events], dtype=np.float32)
    elif tag in ea.Tags().get("tensors", []):
        events = ea.Tensors(tag)
        steps  = np.array([e.step for e in events], dtype=np.float32)
        # tensor_proto for a scalar: value lives in float_val[0] (DT_FLOAT)
        # or double_val[0] (DT_DOUBLE) — check both
        def _proto_to_scalar(proto):
            if proto.float_val:
                return proto.float_val[0]
            if proto.double_val:
                return proto.double_val[0]
            # fallback: raw bytes as float32
            return np.frombuffer(proto.tensor_content, dtype=np.float32)[0]
        values = np.array([_proto_to_scalar(e.tensor_proto) for e in events], dtype=np.float32)
    else:
        return None, None

    order = np.argsort(steps)
    return steps[order], values[order]


def interpolate_to_grid(steps, values, grid):
    """Linearly interpolate values onto a common step grid."""
    return np.interp(grid, steps, values)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_crossval_loss(
    save_dir,
    num_folds=5,
    train_tag="train_loss",
    val_tag="val_loss",
    smooth_window=1,
    n_grid=500,
    out_path="crossval_loss.pdf",
    dpi=300,
    separate=False,
    log_scale=False,
    ymin=None,
    ymax=None,
):
    """
    Args:
        save_dir     : base crossval directory containing fold_0/, fold_1/, ...
        num_folds    : number of folds
        train_tag    : TensorBoard scalar tag for training loss
        val_tag      : TensorBoard scalar tag for validation loss
        smooth_window: uniform smoothing kernel size (1 = no smoothing)
        n_grid       : number of points on the interpolated step grid
        out_path     : output file path (.pdf or .png)
    """
    train_curves, val_curves = [], []
    all_steps = []

    for fold_idx in range(num_folds):
        fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
        if not os.path.isdir(fold_dir):
            print(f"  [warn] fold_{fold_idx} not found, skipping.")
            continue

        # TensorBoard logs are written to a 'tb/' subdirectory
        tb_dir = os.path.join(fold_dir, "tb") if os.path.isdir(os.path.join(fold_dir, "tb")) else fold_dir
        steps_t, vals_t = read_tb_scalar(tb_dir, train_tag)
        steps_v, vals_v = read_tb_scalar(tb_dir, val_tag)

        if steps_t is None:
            print(f"  [warn] tag '{train_tag}' not found in fold_{fold_idx}.")
            continue

        all_steps.append(steps_t[-1])
        train_curves.append((steps_t, vals_t))
        if steps_v is not None:
            val_curves.append((steps_v, vals_v))

    if not train_curves:
        raise RuntimeError("No training curves found. Check save_dir and tag names.")

    # Common step grid from 0 to max shared step
    max_step = min(all_steps)
    grid = np.linspace(0, max_step, n_grid)

    def aggregate(curves):
        interped = np.stack([interpolate_to_grid(s, v, grid) for s, v in curves])
        if smooth_window > 1:
            interped = uniform_filter1d(interped, size=smooth_window, axis=1)
        return interped.mean(0), interped.std(0)

    train_mean, train_std = aggregate(train_curves)
    has_val = len(val_curves) > 0
    if has_val:
        val_mean, val_std = aggregate(val_curves)

    # ── Figure ────────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,   # embeds fonts in PDF (required by most venues)
        "ps.fonttype":  42,
    })

    config_name = os.path.basename(save_dir)

    def _apply_axis(ax):
        if log_scale:
            ax.set_yscale("log")
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

    def _save_single(mean, std, label, color, title, out):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(grid, mean, color=color, linewidth=1.5, label=f"{label} (mean)")
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.20, label="±1 std")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        _apply_axis(ax)
        ax.legend(framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.close(fig)

    base, ext = os.path.splitext(out_path)

    if separate:
        _save_single(train_mean, train_std, "Train loss", "#2166ac",
                     f"Training loss — {config_name}", f"{base}_train{ext}")
        if has_val:
            _save_single(val_mean, val_std, "Val loss", "#d6604d",
                         f"Validation loss — {config_name}", f"{base}_val{ext}")
    else:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(grid, train_mean, color="#2166ac", linewidth=1.5, label="Train loss (mean)")
        ax.fill_between(grid, train_mean - train_std, train_mean + train_std,
                        color="#2166ac", alpha=0.20, label="Train ±1 std")
        if has_val:
            ax.plot(grid, val_mean, color="#d6604d", linewidth=1.5, label="Val loss (mean)")
            ax.fill_between(grid, val_mean - val_std, val_mean + val_std,
                            color="#d6604d", alpha=0.20, label="Val ±1 std")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_title(f"5-fold cross-validation loss — {config_name}")
        _apply_axis(ax)
        ax.legend(framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",     required=True,             help="Crossval base dir (contains fold_0/ ...)")
    parser.add_argument("--num_folds",    default=5,   type=int)
    parser.add_argument("--train_tag",    default="train_loss",      help="TensorBoard tag for train loss")
    parser.add_argument("--val_tag",      default="val_loss",        help="TensorBoard tag for val loss")
    parser.add_argument("--smooth",       default=1,   type=int,     help="Smoothing window (1=none, try 20-50)")
    parser.add_argument("--n_grid",       default=500, type=int,     help="Interpolation grid points")
    parser.add_argument("--out",          default="crossval_loss.pdf")
    parser.add_argument("--dpi",          default=300, type=int)
    parser.add_argument("--separate",     action="store_true",      help="Save train and val as separate files")
    parser.add_argument("--log",          action="store_true",      help="Use log scale on y-axis")
    parser.add_argument("--ymin",         default=None, type=float, help="Y-axis lower limit (e.g. 0.01)")
    parser.add_argument("--ymax",         default=None, type=float, help="Y-axis upper limit (e.g. 1.0)")
    args = parser.parse_args()

    plot_crossval_loss(
        save_dir     = args.save_dir,
        num_folds    = args.num_folds,
        train_tag    = args.train_tag,
        val_tag      = args.val_tag,
        smooth_window= args.smooth,
        n_grid       = args.n_grid,
        out_path     = args.out,
        dpi          = args.dpi,
        separate     = args.separate,
        log_scale    = args.log,
        ymin         = args.ymin,
        ymax         = args.ymax,
    )
