"""
Plot cross-validation training/validation loss curves for a paper.

Single model:
    python plot_crossval_loss.py --save_dirs path/to/config --out loss.pdf

Compare up to 3 models:
    python plot_crossval_loss.py \
        --save_dirs path/config1 path/config2 path/config3 \
        --labels "Model A" "Model B" "Model C" \
        --out comparison.pdf

Requirements:
    pip install tensorboard scipy matplotlib
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Colorblind-friendly palette: blue, orange, green
_COLORS = ["#2166ac", "#d6604d", "#4dac26"]

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

    if tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events], dtype=np.float32)
        values = np.array([e.value for e in events], dtype=np.float32)
    elif tag in ea.Tags().get("tensors", []):
        events = ea.Tensors(tag)
        steps  = np.array([e.step for e in events], dtype=np.float32)
        def _proto_to_scalar(proto):
            if proto.float_val:
                return proto.float_val[0]
            if proto.double_val:
                return proto.double_val[0]
            return np.frombuffer(proto.tensor_content, dtype=np.float32)[0]
        values = np.array([_proto_to_scalar(e.tensor_proto) for e in events], dtype=np.float32)
    else:
        return None, None

    order = np.argsort(steps)
    return steps[order], values[order]


# ── Curve loading ─────────────────────────────────────────────────────────────

def load_model_curves(save_dir, num_folds, train_tag, val_tag, smooth_window, n_grid, max_epochs):
    """
    Load and aggregate fold curves for one model directory.
    Returns (grid, train_mean, train_std, val_mean, val_std).
    val_mean/val_std are None if val_tag was not found.
    """
    train_curves, val_curves, all_steps = [], [], []

    for fold_idx in range(num_folds):
        fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
        if not os.path.isdir(fold_dir):
            print(f"  [warn] {save_dir}/fold_{fold_idx} not found, skipping.")
            continue

        tb_dir = os.path.join(fold_dir, "tb") if os.path.isdir(os.path.join(fold_dir, "tb")) else fold_dir
        steps_t, vals_t = read_tb_scalar(tb_dir, train_tag)
        steps_v, vals_v = read_tb_scalar(tb_dir, val_tag)
        print(vals_v)

        if steps_t is None:
            print(f"  [warn] tag '{train_tag}' not found in {save_dir}/fold_{fold_idx}.")
            continue

        all_steps.append(steps_t[-1])
        train_curves.append((steps_t, vals_t))
        if steps_v is not None:
            val_curves.append((steps_v, vals_v))

    if not train_curves:
        raise RuntimeError(f"No training curves found in {save_dir}. Check tag names.")

    max_step = min(all_steps)
    grid = np.linspace(0, max_epochs, n_grid)

    def aggregate(curves):
        interped = np.stack([
            np.interp(grid, s * (max_epochs / max_step), v)
            for s, v in curves
        ])
        if smooth_window > 1:
            interped = uniform_filter1d(interped, size=smooth_window, axis=1)
        return interped.mean(0), interped.std(0)

    train_mean, train_std = aggregate(train_curves)
    val_mean, val_std = aggregate(val_curves) if val_curves else (None, None)

    return grid, train_mean, train_std, val_mean, val_std


# ── Shared axis style ─────────────────────────────────────────────────────────

def _style_ax(ax, log_scale, ymin, ymax):
    if log_scale:
        ax.set_yscale("log")
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)


def _apply_rcparams():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })


# ── Single-model plot ─────────────────────────────────────────────────────────

def plot_single(label, grid, train_mean, train_std, val_mean, val_std,
                std_scale, log_scale, ymin, ymax, separate, out_path, dpi):
    base, ext = os.path.splitext(out_path)
    color_t, color_v = _COLORS[0], _COLORS[1]

    def _one(mean, std, curve_label, color, title, path):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(grid, mean, color=color, linewidth=1.5, label=f"{curve_label} (mean)")
        ax.fill_between(grid, mean - std_scale * std, mean + std_scale * std,
                        color=color, alpha=0.20, label=f"±{std_scale} std")
        ax.set_title(title)
        _style_ax(ax, log_scale, ymin, ymax)
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    if separate:
        _one(train_mean, train_std, "Train loss", color_t,
             f"Training loss — {label}", f"{base}_train{ext}")
        if val_mean is not None:
            _one(val_mean, val_std, "Val loss", color_v,
                 f"Validation loss — {label}", f"{base}_val{ext}")
    else:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(grid, train_mean, color=color_t, linewidth=1.5, label="Train loss (mean)")
        ax.fill_between(grid, train_mean - std_scale * train_std, train_mean + std_scale * train_std,
                        color=color_t, alpha=0.20, label=f"Train ±{std_scale} std")
        if val_mean is not None:
            ax.plot(grid, val_mean, color=color_v, linewidth=1.5, label="Val loss (mean)")
            ax.fill_between(grid, val_mean - std_scale * val_std, val_mean + std_scale * val_std,
                            color=color_v, alpha=0.20, label=f"Val ±{std_scale} std")
        ax.set_title(f"5-fold cross-validation loss — {label}")
        _style_ax(ax, log_scale, ymin, ymax)
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out_path}")
        plt.close(fig)


# ── Multi-model comparison plot ───────────────────────────────────────────────

def plot_comparison(model_data, std_scale, log_scale, ymin, ymax, separate, out_path, dpi):
    """
    model_data: list of (label, grid, train_mean, train_std, val_mean, val_std)
    solid line = train loss, dashed line = val loss, one color per model.
    With --separate: saves _train.pdf and _val.pdf each containing all models.
    """
    base, ext = os.path.splitext(out_path)

    def _make_ax(title):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.set_title(title)
        _style_ax(ax, log_scale, ymin, ymax)
        return fig, ax

    def _add_model(ax, grid, mean, std, color, label, linestyle):
        ax.plot(grid, mean, color=color, linewidth=1.5, linestyle=linestyle, label=label)
        ax.fill_between(grid, mean - std_scale * std, mean + std_scale * std,
                        color=color, alpha=0.15)

    def _finish(fig, ax, path):
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    if separate:
        fig_t, ax_t = _make_ax("Training loss comparison")
        fig_v, ax_v = _make_ax("Validation loss comparison")
        has_val = False
        #labelmap={"config1": "Window 30", "config4": "Full Sequence", "config6": "Window 60"}
        labelmap={"config13": "Soft-DTW", "config6": "Full Sequence", "config6": "L_simple+Velocity"}
        for i, (label, grid, train_mean, train_std, val_mean, val_std) in enumerate(model_data):
            color = _COLORS[i % len(_COLORS)]
            print(f"Plotting model: {label}")
            _add_model(ax_t, grid, train_mean, train_std, color, labelmap[label], "-")
            if val_mean is not None:
                _add_model(ax_v, grid, val_mean, val_std, color, labelmap[label], "-")
                has_val = True
        _finish(fig_t, ax_t, f"{base}_train{ext}")
        if has_val:
            _finish(fig_v, ax_v, f"{base}_val{ext}")
    else:
        fig, ax = _make_ax("Cross-validation loss comparison")
        for i, (label, grid, train_mean, train_std, val_mean, val_std) in enumerate(model_data):
            color = _COLORS[i % len(_COLORS)]
            _add_model(ax, grid, train_mean, train_std, color, f"{label} train", "-")
            if val_mean is not None:
                _add_model(ax, grid, val_mean, val_std, color, f"{label} val", "--")
        _finish(fig, ax, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dirs",  required=True, nargs="+",
                        help="One crossval dir (single plot) or 2–3 dirs (comparison)")
    parser.add_argument("--labels",     nargs="+", default=None,
                        help="Legend labels for each model (default: folder name)")
    parser.add_argument("--num_folds",  default=5,    type=int)
    parser.add_argument("--train_tag",  default="loss",      help="TensorBoard tag for train loss")
    parser.add_argument("--val_tag",    default="val_dtw_loss",  help="TensorBoard tag for val loss")
    parser.add_argument("--smooth",     default=1,    type=int,   help="Smoothing window (1=none, try 20-50)")
    parser.add_argument("--n_grid",     default=500,  type=int)
    parser.add_argument("--out",        default="crossval_loss.png")
    parser.add_argument("--dpi",        default=300,  type=int)
    parser.add_argument("--separate",   action="store_true", help="Save train and val as separate files")
    parser.add_argument("--log",        action="store_true", help="Log scale on y-axis")
    parser.add_argument("--ymin",       default=None, type=float)
    parser.add_argument("--ymax",       default=None, type=float)
    parser.add_argument("--std_scale",  default=0.5,  type=float, help="Std band multiplier (default 0.7)")
    parser.add_argument("--max_epochs", default=1500, type=int,   help="Value at the last x-axis tick")
    args = parser.parse_args()

    if len(args.save_dirs) > 3:
        parser.error("At most 3 models can be compared.")

    _apply_rcparams()

    labels = args.labels or [os.path.basename(d.rstrip("/\\")) for d in args.save_dirs]
    if len(labels) != len(args.save_dirs):
        parser.error("--labels count must match --save_dirs count.")

    model_data = []
    for save_dir, label in zip(args.save_dirs, labels):
        print(f"\nLoading: {save_dir}  ({label})")
        grid, train_mean, train_std, val_mean, val_std = load_model_curves(
            save_dir    = save_dir,
            num_folds   = args.num_folds,
            train_tag   = args.train_tag,
            val_tag     = args.val_tag,
            smooth_window = args.smooth,
            n_grid      = args.n_grid,
            max_epochs  = args.max_epochs,
        )
        model_data.append((label, grid, train_mean, train_std, val_mean, val_std))

    shared = dict(std_scale=args.std_scale, log_scale=args.log,
                  ymin=args.ymin, ymax=args.ymax, separate=args.separate,
                  out_path=args.out, dpi=args.dpi)

    if len(model_data) == 1:
        label, grid, train_mean, train_std, val_mean, val_std = model_data[0]
        plot_single(label, grid, train_mean, train_std, val_mean, val_std, **shared)
    else:
        plot_comparison(model_data, **shared)
