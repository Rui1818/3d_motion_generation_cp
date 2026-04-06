"""
Summarize cross-validation results produced by run_crossval_eval.sh.

For each model directory, loads crossval_metrics_{checkpoint}.npy and
fid_result_{checkpoint}.npy, aggregates mean ± std across all samples,
and prints two DataFrames: one for motion metrics, one for FID.

CSVs are also written next to this script.
"""

import os
import json
import numpy as np
import pandas as pd

# ── Model directories (from run_crossval_eval.sh) ────────────────────────────

MODEL_DIRS = [
    "my_training/transformer/config1",
    "my_training/transformer/config2",
    "my_training/transformer/config3",
    "my_training/transformer/config4",
    "my_training/transformer/config5",
    "my_training/transformer/config6",
    "my_training/transformer/config7",
    "my_training/transformer/config8",
    "my_training/transformer/config9",
    "my_training/transformer/config10",
    "my_training/transformer/config11",
]

SKIP_KEYS = {"sample_id", "action"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_fold0_args(model_dir: str) -> dict:
    path = os.path.join(model_dir, "fold_0", "args.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _load_npy(path: str):
    """Load a .npy file; return None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return data.item() if data.ndim == 0 else list(data)


def detect_checkpoint_suffixes(model_dir: str) -> list[str]:
    """
    Return the list of suffixes for which metric files exist in model_dir.
    Prefers _latest/_best (from --checkpoint both); falls back to '' (single run).
    """
    suffixes = []
    for s in ("_latest", "_best"):
        if os.path.exists(os.path.join(model_dir, f"crossval_metrics{s}.npy")):
            suffixes.append(s)
    if not suffixes and os.path.exists(os.path.join(model_dir, "crossval_metrics.npy")):
        suffixes.append("")
    return suffixes


def aggregate(samples: list[dict]) -> dict:
    """Compute mean and std for every numeric metric key."""
    if not samples:
        return {}
    keys = [k for k in samples[0] if k not in SKIP_KEYS]
    out = {}
    for key in keys:
        vals = [
            float(s[key]) for s in samples
            if key in s and isinstance(s[key], (int, float, np.floating))
        ]
        if vals:
            out[f"{key}_mean"] = np.mean(vals)
            out[f"{key}_std"]  = np.std(vals)
    return out


def checkpoint_label(suffix: str) -> str:
    return suffix.lstrip("_") if suffix else "single"


# ── Build DataFrames ──────────────────────────────────────────────────────────

def build_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_metrics = []
    rows_fid     = []

    for model_dir in MODEL_DIRS:
        if not os.path.isdir(model_dir):
            print(f"  SKIP (not found): {model_dir}")
            continue

        args       = load_fold0_args(model_dir)
        model_name = os.path.basename(model_dir)
        info = {
            "model":        model_name,
            "keypointtype": args.get("keypointtype", "?"),
            "window":       args.get("input_motion_length", "?"),
            "use_dct":      args.get("use_dct", "?"),
        }

        suffixes = detect_checkpoint_suffixes(model_dir)
        if not suffixes:
            print(f"  SKIP (no metric files): {model_dir}")
            continue

        for suffix in suffixes:
            ckpt = checkpoint_label(suffix)
            row_base = {**info, "checkpoint": ckpt}

            # ── Per-window metrics ────────────────────────────────────────────
            samples = _load_npy(os.path.join(model_dir, f"crossval_metrics{suffix}.npy"))
            if samples is not None:
                agg = aggregate(samples)
                rows_metrics.append({**row_base, "metric_type": "per_window", **agg})

            # ── Full-concat metrics (sliding-window models only) ──────────────
            concat_samples = _load_npy(os.path.join(model_dir, f"crossval_metrics_concat{suffix}.npy"))
            if concat_samples is not None:
                agg_concat = aggregate(concat_samples)
                rows_metrics.append({**row_base, "metric_type": "full_concat", **agg_concat})

            # ── FID ───────────────────────────────────────────────────────────
            fid_data = _load_npy(os.path.join(model_dir, f"fid_result{suffix}.npy"))
            if fid_data is not None:
                rows_fid.append({
                    **row_base,
                    "FID":          fid_data.get("FID_global", float("nan")),
                    "real_windows": fid_data.get("real_windows"),
                    "gen_windows":  fid_data.get("gen_windows"),
                })

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_fid)


# ── Display ───────────────────────────────────────────────────────────────────

ID_COLS = ["model", "checkpoint", "metric_type", "keypointtype", "window", "use_dct"]


def print_metrics_df(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no metric files found)")
        return

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    std_cols  = [c for c in df.columns if c.endswith("_std")]
    id_present = [c for c in ID_COLS if c in df.columns]

    print("\n  -- Means --")
    print(df[id_present + mean_cols].to_string(index=False))

    print("\n  -- Standard Deviations --")
    print(df[id_present + std_cols].to_string(index=False))


def print_fid_df(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (no FID files found — was --autoencoder_path provided?)")
        return

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    id_present = [c for c in ["model", "checkpoint", "keypointtype", "window", "use_dct"]
                  if c in df.columns]
    print(df[id_present + ["FID", "real_windows", "gen_windows"]].to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Scanning model directories...")
    df_metrics, df_fid = build_dataframes()

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION METRICS SUMMARY")
    print("=" * 80)
    print_metrics_df(df_metrics)

    print("\n" + "=" * 80)
    print("FID SCORES")
    print("=" * 80)
    print_fid_df(df_fid)

    # Save to CSV
    if not df_metrics.empty:
        df_metrics.to_csv("crossval_summary_metrics.csv", index=False)
        print("\nMetrics CSV → crossval_summary_metrics.csv")
    if not df_fid.empty:
        df_fid.to_csv("crossval_summary_fid.csv", index=False)
        print("FID CSV     → crossval_summary_fid.csv")


if __name__ == "__main__":
    main()
