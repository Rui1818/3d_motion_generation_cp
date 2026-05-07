"""
Baseline limb angle evaluation: DTW-aligned MAE between condition (c2, without
orthosis) and reference (c1, with orthosis) loaded directly from the dataset.
No model is involved — this measures the raw difference between conditions.
Only supports keypointtype='openpose'.
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from data_loaders.dataloader3d import load_data
from limb_angles import dtw_angle_error, dtw_angle_correlation


def _to_joints3d(motion: torch.Tensor) -> np.ndarray:
    """(frames, 69) -> (frames, 23, 3)"""
    return motion.cpu().numpy().reshape(-1, 23, 3)


def eval_limb_angles(dataset_path, subjects=None):
    """
    Iterate over all (c1, c2) pairs and compute per-pair DTW limb angle error.

    Returns:
        list of dicts with keys: sample_id, action, left_hip, right_hip,
        left_knee, right_knee, left_ankle, right_ankle
    """
    motion_clean, motion_w_o, _ = load_data(
        dataset_path,
        split="test",
        keypointtype="openpose",
        subjects=subjects,
    )

    results = []
    sample_id = 0

    for action_key in tqdm(sorted(motion_clean.keys()), desc="Actions"):
        c1_takes = motion_clean[action_key]
        c2_takes = motion_w_o[action_key]
        action = action_key.rsplit("_", 1)[-1]

        for i, c1 in enumerate(c1_takes):
            for j, c2 in enumerate(c2_takes):
                ref_joints = _to_joints3d(c1)
                cond_joints = _to_joints3d(c2)

                errors = dtw_angle_error(ref_joints, cond_joints)
                corrs = dtw_angle_correlation(ref_joints, cond_joints)
                corrs_renamed = {f"{k}_r": v for k, v in corrs.items()}
                results.append({"sample_id": sample_id, "action": action, **errors, **corrs_renamed})
                sample_id += 1

    return results


def print_summary(all_results):
    if not all_results:
        print("No results.")
        return

    skip = {"sample_id", "action"}
    all_keys = [k for k in all_results[0] if k not in skip]
    mae_keys  = [k for k in all_keys if not k.endswith("_r")]
    corr_keys = [k for k in all_keys if k.endswith("_r")]

    print("\n=== Baseline Limb Angle MAE (c1 vs c2, DTW-aligned, degrees) ===")
    print(f"Total samples: {len(all_results)}")
    for key in mae_keys:
        vals = [r[key] for r in all_results if key in r]
        print(f"  {key:<15}: {np.mean(vals):.2f} ± {np.std(vals):.2f}  "
              f"[min={np.min(vals):.2f}, max={np.max(vals):.2f}]")

    print("\n=== Baseline Limb Angle Pearson r (c1 vs c2, DTW-aligned) ===")
    for key in corr_keys:
        vals = [r[key] for r in all_results if key in r]
        print(f"  {key:<18}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  "
              f"[min={np.min(vals):.3f}, max={np.max(vals):.3f}]")

    action_labels = sorted(set(r["action"] for r in all_results))
    if action_labels:
        print("\n--- Per-action breakdown ---")
        for action in action_labels:
            subset = [r for r in all_results if r["action"] == action]
            print(f"\n  [Action: {action}]  n={len(subset)}")
            for key in mae_keys:
                vals = [r[key] for r in subset if key in r]
                if vals:
                    print(f"    {key:<15}: MAE {np.mean(vals):.2f} ± {np.std(vals):.2f}"
                          f"  r={np.mean([r[key+'_r'] for r in subset if key+'_r' in r]):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to final_dataset directory")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Optional list of subjects to evaluate (e.g. gait_011 gait_052)")
    parser.add_argument("--save_path", default=None,
                        help="Optional .npy path to save results")
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = eval_limb_angles(args.dataset_path, subjects=args.subjects)
    print_summary(results)

    if args.save_path:
        np.save(args.save_path, results)
        print(f"\nResults saved → {args.save_path}")


if __name__ == "__main__":
    main()
