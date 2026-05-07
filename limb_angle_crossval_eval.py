"""
Cross-validation evaluation of DTW-aligned limb angle errors.
Mirrors gait_crossval_eval.py for motion generation, then computes
per-limb angle MAE (hip, knee, ankle — left and right) using DTW alignment.
"""
import os
import re
import json
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm

from gait_generate import load_diffusion_model, sample, transform_motion_back
from gait_crossval_eval import (
    find_latest_checkpoint,
    find_best_checkpoint,
    _sliding_window_start_indices,
)
from data_loaders.dataloader3d import TestDataset, load_data, get_dataloader, sample_matching_startframe
from limb_angles import dtw_angle_error, dtw_angle_correlation

NUM_FOLDS = 5


def _to_joints3d(motion_np: np.ndarray, keypointtype: str) -> np.ndarray:
    """
    Convert a flat motion array to (frames, joints, 3).
      openpose      : (frames, 69)  -> (frames, 23, 3)
      6d_transformed: (frames, 22, 3) already correct
    """
    if keypointtype == "openpose":
        return motion_np.reshape(-1, 23, 3)
    elif keypointtype in ("6d", "6d_transformed"):
        if motion_np.ndim == 2:
            return motion_np.reshape(-1, 22, 3)
        return motion_np  # already (frames, 22, 3)
    else:
        raise ValueError(f"Unsupported keypointtype for limb angles: {keypointtype}")


def eval_fold_limb_angles(fold_dir, dataset_path, checkpoint_type="latest"):
    """
    Load model for one fold, generate motions on val subjects, and compute
    DTW-aligned limb angle errors.

    Returns:
        list of per-sample dicts with keys:
            sample_id, action, left_hip, right_hip, left_knee, right_knee,
            left_ankle, right_ankle
    """
    args_path = os.path.join(fold_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {fold_dir}")

    with open(args_path) as f:
        saved = json.load(f)

    val_subjects = saved["val_subjects"]
    print(f"  Val subjects: {val_subjects}")

    # backwards compat: sparse_dim was renamed to cond_dim
    if "cond_dim" not in saved and "sparse_dim" in saved:
        saved["cond_dim"] = saved["sparse_dim"]
    # backwards compat: lambda_transl added later with default 1.0
    saved.setdefault("lambda_transl", 1.0)
    saved.setdefault("lambda_rot", 1.0)

    args = argparse.Namespace(**saved)
    if checkpoint_type == "best":
        args.model_path = find_best_checkpoint(fold_dir)
    else:
        args.model_path = find_latest_checkpoint(fold_dir)
    print(f"  Using checkpoint ({checkpoint_type}): {args.model_path}")

    dct_stats = None
    if args.use_dct:
        dct_stats_path = os.path.join(fold_dir, "dct_stats.pt")
        if os.path.exists(dct_stats_path):
            dct_stats = torch.load(dct_stats_path, map_location="cpu")

    model, diffusion = load_diffusion_model(args)

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
    ws = args.input_motion_length

    match_dict = None
    if use_sliding_window:
        match_dict_path = (
            "prepare_data/match_dict_window30_final.npy" if ws == 30
            else "prepare_data/match_dict_window60_final.npy"
        )
        match_dict = np.load(match_dict_path, allow_pickle=True).item()

    fold_results = []

    for i, batch in enumerate(tqdm(dataloader, desc="  Evaluating")):
        reference, condition, batch_betas, action_label, match_key = batch
        action_label = action_label[0]
        match_key = match_key[0]
        condition = condition.to(device)

        if use_sliding_window:
            generated_motion_windows, generated_motion = sample(
                model, diffusion, condition, args,
                use_sliding_window=True, dct_stats=dct_stats
            )

            gen_windows_np   = generated_motion_windows.squeeze(0).cpu().numpy()
            reference_tensor = reference.squeeze(0)
            total_frames     = condition.shape[1]
            start_indices    = _sliding_window_start_indices(total_frames, ws)
            n_windows        = len(start_indices)

            window_errors = []
            for k in range(n_windows):
                gen_win = gen_windows_np[k * ws : (k + 1) * ws]
                ref_win = sample_matching_startframe(
                    reference_tensor, match_dict, match_key, start_indices[k], ws
                ).numpy()

                # Transform to 3D joint positions when needed
                if args.keypointtype == "6d":
                    win_betas = batch_betas[:, :ws, :] if batch_betas.dim() == 3 else batch_betas
                    gen_win_3d, ref_win_3d = transform_motion_back(
                        args, win_betas, gen_win.copy(), ref_win.copy()
                    )
                else:
                    gen_win_3d = gen_win
                    ref_win_3d = ref_win

                gen_joints = _to_joints3d(gen_win_3d, args.keypointtype)
                ref_joints = _to_joints3d(ref_win_3d, args.keypointtype)
                window_errors.append(dtw_angle_error(ref_joints, gen_joints))
                window_errors[-1].update({
                    f"{k}_r": v
                    for k, v in dtw_angle_correlation(ref_joints, gen_joints).items()
                })

            # Average per-window errors into one sample-level result
            res = {"sample_id": i, "action": action_label}
            for limb in window_errors[0]:
                res[limb] = float(np.mean([e[limb] for e in window_errors]))

        else:
            generated_motion = sample(
                model, diffusion, condition, args,
                use_sliding_window=False, dct_stats=dct_stats
            )
            generated_np = generated_motion.squeeze(0).cpu().numpy()
            reference_np = reference.squeeze(0).cpu().numpy()

            if args.keypointtype == "6d":
                generated_np, reference_np = transform_motion_back(
                    args, batch_betas, generated_np.copy(), reference_np.copy()
                )

            gen_joints = _to_joints3d(generated_np, args.keypointtype)
            ref_joints = _to_joints3d(reference_np, args.keypointtype)

            errors = dtw_angle_error(ref_joints, gen_joints)
            corrs = {f"{k}_r": v for k, v in dtw_angle_correlation(ref_joints, gen_joints).items()}
            res = {"sample_id": i, "action": action_label, **errors, **corrs}

        fold_results.append(res)

    return fold_results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results):
    """Print mean ± std per limb angle, total and broken down by action."""
    if not all_results:
        print("No results to summarize.")
        return

    skip = {"sample_id", "action"}
    all_keys  = [k for k in all_results[0] if k not in skip]
    mae_keys  = [k for k in all_keys if not k.endswith("_r")]
    corr_keys = [k for k in all_keys if k.endswith("_r")]

    print("\n=== Limb Angle MAE (DTW-aligned, degrees) ===")
    print(f"Total samples: {len(all_results)}")
    for key in mae_keys:
        vals = [r[key] for r in all_results if key in r]
        print(f"  {key:<15}: {np.mean(vals):.2f} ± {np.std(vals):.2f}  "
              f"[min={np.min(vals):.2f}, max={np.max(vals):.2f}]")

    if corr_keys:
        print("\n=== Limb Angle Pearson r (DTW-aligned) ===")
        for key in corr_keys:
            vals = [r[key] for r in all_results if key in r]
            print(f"  {key:<18}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  "
                  f"[min={np.min(vals):.3f}, max={np.max(vals):.3f}]")

    action_labels = sorted(set(r["action"] for r in all_results if "action" in r))
    if action_labels:
        print("\n--- Per-action breakdown ---")
        for action in action_labels:
            subset = [r for r in all_results if r.get("action") == action]
            print(f"\n  [Action: {action}]  n={len(subset)}")
            for key in mae_keys:
                vals = [r[key] for r in subset if key in r]
                if vals:
                    r_vals = [r[key + "_r"] for r in subset if key + "_r" in r]
                    r_str = f"  r={np.mean(r_vals):.3f}" if r_vals else ""
                    print(f"    {key:<15}: {np.mean(vals):.2f} ± {np.std(vals):.2f}{r_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, type=str,
                        help="Base crossval save directory (contains fold_0/, fold_1/, ...)")
    parser.add_argument("--dataset_path", required=True, type=str,
                        help="Path to final_dataset")
    parser.add_argument("--num_folds", default=NUM_FOLDS, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--checkpoint", default="best", choices=["latest", "best", "both"],
                        help="Which checkpoint to evaluate.")
    args_main = parser.parse_args()

    random.seed(args_main.seed)
    np.random.seed(args_main.seed)
    torch.manual_seed(args_main.seed)

    checkpoint_types = (
        ["latest", "best"] if args_main.checkpoint == "both" else [args_main.checkpoint]
    )

    results = {ct: [] for ct in checkpoint_types}

    for fold_idx in range(args_main.num_folds):
        fold_dir = os.path.join(args_main.save_dir, f"fold_{fold_idx}")
        if not os.path.exists(fold_dir):
            print(f"Fold {fold_idx}: directory not found, skipping.")
            continue

        for ct in checkpoint_types:
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx + 1}/{args_main.num_folds}  [{ct}]  ({fold_dir})")
            print(f"{'='*60}")

            try:
                fold_results = eval_fold_limb_angles(fold_dir, args_main.dataset_path, checkpoint_type=ct)
            except Exception as e:
                print(f"  ERROR evaluating fold {fold_idx} [{ct}]: {e}")
                continue

            suffix = f"_{ct}" if args_main.checkpoint == "both" else ""
            metrics_path = os.path.join(fold_dir, f"limb_angle_metrics{suffix}.npy")
            np.save(metrics_path, fold_results)
            print(f"  Saved fold limb angle metrics → {metrics_path}")

            skip = {"sample_id", "action"}
            limbs = [k for k in fold_results[0] if k not in skip]
            for limb in limbs:
                vals = [r[limb] for r in fold_results]
                print(f"    {limb:<15}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

            results[ct].extend(fold_results)

    for ct in checkpoint_types:
        suffix = f"_{ct}" if args_main.checkpoint == "both" else ""

        print(f"\n{'#'*60}")
        print(f"CHECKPOINT: {ct.upper()}")
        print(f"{'#'*60}")
        print_summary(results[ct])

        all_path = os.path.join(args_main.save_dir, f"crossval_limb_angle_metrics{suffix}.npy")
        np.save(all_path, results[ct])
        print(f"\nAll limb angle metrics saved → {all_path}")


if __name__ == "__main__":
    main()
