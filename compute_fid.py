"""
Compute FID between motion (with orthosis) and motion_w_o (without orthosis)
using a pre-trained MotionAutoencoder for feature extraction.

Usage:
    python compute_fid.py \
        --dataset_path /path/to/final_dataset \
        --autoencoder_path /path/to/best_autoencoder.pt \
        --keypointtype 6d \
        --input_motion_length 196
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.dataloader3d import MotionDataset, load_data
from gait_crossval_eval import load_encoder, compute_fid


@torch.no_grad()
def collect_features(dataloader, encoder, device):
    """
    Iterate through a MotionDataset dataloader and encode each windowed
    motion / motion_w_o pair into latent features.

    Returns:
        real_feats : (N, latent_dim) numpy array  — motion (with orthosis)
        gen_feats  : (N, latent_dim) numpy array  — motion_w_o (without orthosis)
    """
    real_feats = []
    gen_feats = []

    for _, motion, motion_w_o in tqdm(dataloader, desc="Extracting features"):
        motion = motion.to(device)        # (B, T, D)
        motion_w_o = motion_w_o.to(device)  # (B, T, D)

        real_feats.append(encoder(motion).cpu().numpy())
        gen_feats.append(encoder(motion_w_o).cpu().numpy())

    return np.concatenate(real_feats, axis=0), np.concatenate(gen_feats, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID between motion (with orthosis) and motion_w_o (without orthosis)."
    )
    parser.add_argument("--dataset_path", required=True, type=str,
                        help="Path to final_dataset root directory.")
    parser.add_argument("--autoencoder_path", required=True, type=str,
                        help="Path to best_autoencoder.pt checkpoint.")
    parser.add_argument("--keypointtype", default="6d", choices=["6d", "openpose", "smpl"],
                        help="Keypoint representation type (must match autoencoder training).")
    parser.add_argument("--input_motion_length", default=196, type=int,
                        help="Window size in frames.")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Whitelist of subject folder names to include (default: all).")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Which data split to load.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_repeats", default=10, type=int,
                        help="How many random windows to sample per sequence pair. "
                             "Higher = more stable FID estimate. Uses train_dataset_repeat_times.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load encoder ──────────────────────────────────────────────────────────
    encoder, latent_dim = load_encoder(args.autoencoder_path, device)
    print(f"Encoder ready  (latent_dim={latent_dim})")

    # ── Load motion data ──────────────────────────────────────────────────────
    print(f"\nLoading data ({args.split}) from: {args.dataset_path}")
    if args.split == "test":
        motion_clean, motion_w_o, _ = load_data(
            args.dataset_path,
            split="test",
            keypointtype=args.keypointtype,
            subjects=args.subjects,
        )
    else:
        motion_clean, motion_w_o = load_data(
            args.dataset_path,
            split="train",
            keypointtype=args.keypointtype,
            subjects=args.subjects,
        )

    dataset = MotionDataset(
        dataset="gait",
        motion_clean=motion_clean,
        motion_without_orth=motion_w_o,
        input_motion_length=args.input_motion_length,
        no_normalization=True,
        train_dataset_repeat_times=args.num_repeats,
    )
    print(f"Dataset pairs : {len(dataset) // args.num_repeats}  x{args.num_repeats} repeats = {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ── Extract features ──────────────────────────────────────────────────────
    real_feats, gen_feats = collect_features(dataloader, encoder, device)
    print(f"\nReal (motion)      windows: {len(real_feats)}")
    print(f"Gen  (motion_w_o)  windows: {len(gen_feats)}")

    # ── Compute FID ───────────────────────────────────────────────────────────
    fid = compute_fid(real_feats, gen_feats)
    print(f"\nFID (motion vs motion_w_o): {fid:.4f}")


if __name__ == "__main__":
    main()
