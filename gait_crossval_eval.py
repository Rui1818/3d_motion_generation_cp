"""
Cross-validation evaluation script.
For each fold in a crossval save_dir, loads the trained model, runs generation on
the held-out val subjects, and computes the same metrics as gait_generate.py.
"""
import os
import re
import json
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm

from gait_generate import (
    load_diffusion_model,
    sample,
    calculate_metrics,
    transform_motion_back,
)
from data_loaders.dataloader3d import TestDataset, load_data, get_dataloader
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


NUM_FOLDS = 5


def find_latest_checkpoint(fold_dir):
    """Return the path to the checkpoint with the highest step number in fold_dir."""
    ckpts = [
        f for f in os.listdir(fold_dir)
        if re.match(r"model\d+\.pt$", f)
    ]
    if not ckpts:
        raise FileNotFoundError(f"No model checkpoints found in {fold_dir}")
    # Sort by the numeric part
    ckpts.sort(key=lambda f: int(re.search(r"\d+", f).group()))
    return os.path.join(fold_dir, ckpts[-1])


def eval_fold(fold_dir, dataset_path):
    """Evaluate one fold: load model, run generation on val subjects, return metrics list."""
    args_path = os.path.join(fold_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {fold_dir}")

    with open(args_path) as f:
        saved = json.load(f)

    val_subjects = saved["val_subjects"]
    print(f"  Val subjects: {val_subjects}")

    # Build a minimal args namespace from the saved config
    args = argparse.Namespace(**saved)
    args.model_path = find_latest_checkpoint(fold_dir)
    print(f"  Using checkpoint: {args.model_path}")

    # Load DCT stats if needed
    dct_stats = None
    if args.use_dct:
        dct_stats_path = os.path.join(fold_dir, "dct_stats.pt")
        if os.path.exists(dct_stats_path):
            dct_stats = torch.load(dct_stats_path, map_location="cpu")
            print("  Loaded DCT stats.")
        else:
            print("  WARNING: use_dct=True but dct_stats.pt not found.")

    # Load model
    model, diffusion = load_diffusion_model(args)

    # Load val data as TestDataset (same format as gait_generate.py)
    motion_clean, motion_w_o, betas = load_data(
        dataset_path,
        split="test",
        keypointtype=args.keypointtype,
        subjects=val_subjects,
    )
    val_dataset = TestDataset(
        "gait",
        motion_clean,
        motion_w_o,
        betas=betas,
        input_motion_length=args.input_motion_length,
    )
    dataloader = get_dataloader(val_dataset, "test", batch_size=1, num_workers=1)
    print(f"  Val samples: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_sliding_window = args.input_motion_length < 240

    fold_metrics = []
    for i, batch in enumerate(tqdm(dataloader, desc="  Evaluating")):
        reference, condition, batch_betas = batch
        condition = condition.to(device)

        if use_sliding_window:
            _, generated_motion = sample(
                model, diffusion, condition, args,
                use_sliding_window=True, dct_stats=dct_stats
            )
        else:
            generated_motion = sample(
                model, diffusion, condition, args,
                use_sliding_window=False, dct_stats=dct_stats
            )

        generated_np = generated_motion.squeeze(0).cpu().numpy()
        reference_np = reference.squeeze(0).cpu().numpy()

        res = calculate_metrics(reference_np, generated_np, args.keypointtype, i)

        if args.keypointtype == "6d":
            gen_back, ref_back = transform_motion_back(args, batch_betas, generated_np.copy(), reference_np.copy())
            mpjpe_6d, pampjpe_6d, jitter = calculate_metrics(ref_back, gen_back, "6d_transformed", i)
            res["MPJPE_6d"] = mpjpe_6d
            res["PAMPJPE_6d"] = pampjpe_6d
            res["jitter"] = jitter

        fold_metrics.append(res)

    return fold_metrics


def print_summary(all_metrics, keypointtype):
    """Print mean ± std for each metric across all samples from all folds."""
    if not all_metrics:
        print("No metrics to summarize.")
        return

    keys = [k for k in all_metrics[0].keys() if k != "sample_id"]
    print("\n=== Cross-Validation Metrics Summary ===")
    print(f"Total samples: {len(all_metrics)}")
    for key in keys:
        vals = [m[key] for m in all_metrics if key in m]
        print(f"  {key:30s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
              f"[min={np.min(vals):.4f}, max={np.max(vals):.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, type=str,
                        help="Base crossval save directory (contains fold_0/, fold_1/, ...)")
    parser.add_argument("--dataset_path", required=True, type=str,
                        help="Path to final_dataset (same as used for training)")
    parser.add_argument("--num_folds", default=NUM_FOLDS, type=int)
    parser.add_argument("--seed", default=10, type=int)
    args_main = parser.parse_args()

    random.seed(args_main.seed)
    np.random.seed(args_main.seed)
    torch.manual_seed(args_main.seed)

    all_metrics = []

    for fold_idx in range(args_main.num_folds):
        fold_dir = os.path.join(args_main.save_dir, f"fold_{fold_idx}")
        if not os.path.exists(fold_dir):
            print(f"Fold {fold_idx}: directory not found, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args_main.num_folds}  ({fold_dir})")
        print(f"{'='*60}")

        try:
            fold_metrics = eval_fold(fold_dir, args_main.dataset_path)
        except Exception as e:
            print(f"  ERROR evaluating fold {fold_idx}: {e}")
            continue

        # Save per-fold metrics
        metrics_path = os.path.join(fold_dir, "eval_metrics.npy")
        np.save(metrics_path, fold_metrics)
        print(f"  Saved fold metrics → {metrics_path}")

        # Per-fold summary
        keys = [k for k in fold_metrics[0].keys() if k != "sample_id"]
        for key in keys:
            vals = [m[key] for m in fold_metrics if key in m]
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        all_metrics.extend(fold_metrics)

    # Detect keypointtype from the first fold's args.json
    first_fold_args_path = os.path.join(args_main.save_dir, "fold_0", "args.json")
    keypointtype = "openpose"
    if os.path.exists(first_fold_args_path):
        with open(first_fold_args_path) as f:
            keypointtype = json.load(f).get("keypointtype", "openpose")

    print_summary(all_metrics, keypointtype)

    # Save all metrics
    all_metrics_path = os.path.join(args_main.save_dir, "crossval_metrics.npy")
    np.save(all_metrics_path, all_metrics)
    print(f"\nAll metrics saved → {all_metrics_path}")


if __name__ == "__main__":
    main()
