"""
Generate motion for a single subject from a trained model fold.

Usage:
    python subject_generate.py \
        --subject gait_011 \
        --model_dir results/dctmlp_60/config_dctmlp10 \
        --dataset_path final_dataset \
        [--checkpoint best|latest] \
        [--output_dir path/to/output]

Outputs (in output_dir):
    reference_motion_{i}.npy          - reference (clean) motion
    generated_motion_{i}.npy          - per-window concatenated output (raw windows)
    generated_motion_concat_{i}.npy   - blended/stitched output (sliding window)
"""

import os

import re
import json
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm

from data_loaders.dataloader3d import TestDataset, load_data, get_dataloader
from gait_generate import (
    sample,
    transform_motion_back,
    load_diffusion_model,
)


def find_latest_checkpoint(fold_dir):
    ckpts = [f for f in os.listdir(fold_dir) if re.match(r"model\d+\.pt$", f)]
    if not ckpts:
        raise FileNotFoundError(f"No model checkpoints found in {fold_dir}")
    ckpts.sort(key=lambda f: int(re.search(r"\d+", f).group()))
    return os.path.join(fold_dir, ckpts[-1])


def find_best_checkpoint(fold_dir):
    path = os.path.join(fold_dir, "best_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"best_model.pt not found in {fold_dir}")
    return path


def load_args_from_fold(fold_dir, checkpoint_type="best"):
    args_path = os.path.join(fold_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {fold_dir}")

    with open(args_path) as f:
        saved = json.load(f)

    args = argparse.Namespace(**saved)

    if checkpoint_type == "best":
        args.model_path = find_best_checkpoint(fold_dir)
    else:
        args.model_path = find_latest_checkpoint(fold_dir)

    # Strip "diffusion_" prefix that some saved configs include
    if hasattr(args, "arch") and args.arch.startswith("diffusion_"):
        args.arch = args.arch[len("diffusion_"):]

    return args


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate motion for a single subject using a trained fold model."
    )
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        help="Subject folder name in the dataset (e.g. gait_011).",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Path to the model fold folder (must contain args.json and a checkpoint).",
    )
    parser.add_argument(
        "--dataset_path",
        default="final_dataset",
        type=str,
        help="Path to the dataset root folder.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        choices=["best", "latest"],
        help="Which checkpoint to load: 'best' (best_model.pt) or 'latest' (highest step).",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Output directory. Defaults to <model_dir>/<subject>/.",
    )
    parser.add_argument(
        "--seed",
        default=10,
        type=int,
        help="Random seed for reproducibility.",
    )
    cli = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = False
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model args from fold ────────────────────────────────────────────
    args = load_args_from_fold(cli.model_dir, checkpoint_type=cli.checkpoint)
    print(f"Checkpoint : {args.model_path}")
    print(f"Keypoint type: {args.keypointtype}")
    print(f"Window size  : {args.input_motion_length}")

    model, diffusion = load_diffusion_model(args)

    # ── DCT stats ────────────────────────────────────────────────────────────
    dct_stats = None
    if getattr(args, "use_dct", False):
        dct_stats_path = os.path.join(cli.model_dir, "dct_stats.pt")
        if os.path.exists(dct_stats_path):
            dct_stats = torch.load(dct_stats_path, map_location="cpu")
            print("Loaded DCT normalization stats.")
        else:
            print("WARNING: use_dct=True but no dct_stats.pt found — running without DCT normalization.")

    # ── Load subject data ────────────────────────────────────────────────────
    if not os.path.isdir(cli.dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {cli.dataset_path}")
    subject_dir = os.path.join(cli.dataset_path, cli.subject)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(
            f"Subject folder not found: {subject_dir}\n"
            f"Available subjects: {sorted(os.listdir(cli.dataset_path))}"
        )

    print(f"Loading data for subject: {cli.subject}")
    motion_clean, motion_w_o, betas = load_data(
        cli.dataset_path,
        split="test",
        keypointtype=args.keypointtype,
        subjects=[cli.subject],
    )
    dataset = TestDataset(
        "gait",
        motion_clean,
        motion_w_o,
        betas=betas,
        input_motion_length=args.input_motion_length,
    )
    dataloader = get_dataloader(dataset, "test", batch_size=1, num_workers=1)
    print(f"Samples for subject {cli.subject}: {len(dataset)}")

    # ── Output directory ─────────────────────────────────────────────────────
    output_dir = cli.output_dir if cli.output_dir else os.path.join(cli.model_dir, cli.subject)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    use_sliding_window = args.input_motion_length < 240
    print(f"Sliding window: {use_sliding_window}")

    # ── Generation loop ──────────────────────────────────────────────────────
    for i, batch in enumerate(tqdm(dataloader, desc="Generating")):
        reference, condition, batch_betas, action_label, match_key = batch
        condition = condition.to(device)

        ref_path        = os.path.join(output_dir, f"reference_motion_{i}.npy")
        gen_path        = os.path.join(output_dir, f"generated_motion_{i}.npy")
        gen_concat_path = os.path.join(output_dir, f"generated_motion_concat_{i}.npy")

        if use_sliding_window:
            generated_windows, generated_concat = sample(
                model, diffusion, condition, args,
                use_sliding_window=True, dct_stats=dct_stats,
            )

            generated_windows_np = generated_windows.squeeze(0).cpu().numpy()
            generated_concat_np  = generated_concat.squeeze(0).cpu().numpy()
            reference_np         = reference.squeeze(0).cpu().numpy()

            generated_windows_np, reference_np = transform_motion_back(
                args, batch_betas, generated_windows_np, reference_np
            )
            generated_concat_np, _ = transform_motion_back(
                args, batch_betas, generated_concat_np, None
            )

            np.save(gen_concat_path, generated_concat_np)

        else:
            generated_motion = sample(
                model, diffusion, condition, args,
                use_sliding_window=False, dct_stats=dct_stats,
            )

            generated_windows_np = generated_motion.squeeze(0).cpu().numpy()
            reference_np         = reference.squeeze(0).cpu().numpy()

            generated_windows_np, reference_np = transform_motion_back(
                args, batch_betas, generated_windows_np, reference_np
            )

        np.save(ref_path, reference_np)
        np.save(gen_path, generated_windows_np)

    print(f"\nGeneration complete. {len(dataset)} sample(s) saved to: {output_dir}")


if __name__ == "__main__":
    main()
