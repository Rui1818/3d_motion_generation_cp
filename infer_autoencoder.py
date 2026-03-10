"""
Inference script for the trained MotionAutoencoder.

Given a trained checkpoint and an input motion sequence, encodes it to a
latent vector and decodes it back to a reconstructed sequence.

Supports two input formats:
  - .npy / .npz  : raw numpy arrays  (shape: frames × D)
  - .npz (6d)    : SMPL-X parameter file processed via smplx_to_6d()

Usage examples:

  # 6D SMPL-X param file:
  python infer_autoencoder.py \
      --checkpoint checkpoints/autoencoder/best_autoencoder.pt \
      --input     data/subject01/smplx-params_cut.npz \
      --keypointtype 6d \
      --output    output/recon.npy

  # Pre-extracted .npy array (shape: T×D):
  python infer_autoencoder.py \
      --checkpoint checkpoints/autoencoder/best_autoencoder.pt \
      --input     data/motion.npy \
      --keypointtype openpose \
      --output    output/recon.npy

Outputs
-------
  <output>.npy          Reconstructed sequence, shape (T, D), original scale
  <output>_latent.npy   Latent vector, shape (latent_dim,)
"""

import argparse
import os

import numpy as np
import torch

from data_loaders.dataloader3d import normalize_motion
from model.motion_autoencoder import MotionAutoencoder
from utils.transformation_sixd import smplx_to_6d


KEYPOINTTYPE_TO_DIM = {
    "6d": 135,
    "openpose": 69,
    "smpl": 66,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MotionAutoencoder inference")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to best_autoencoder.pt checkpoint")
    parser.add_argument("--input", required=True, type=str,
                        help="Input motion file (.npy array or .npz SMPL-X params)")
    parser.add_argument("--keypointtype", default="6d",
                        choices=["6d", "openpose", "smpl"], type=str)
    parser.add_argument("--output", default=None, type=str,
                        help="Output path prefix (default: <input>_recon)")
    parser.add_argument("--window", default=None, type=int,
                        help="If set, process a single window of this length "
                             "starting at --start_frame (default: full sequence)")
    parser.add_argument("--start_frame", default=0, type=int,
                        help="Start frame for windowed inference")
    parser.add_argument("--device", default="cuda", type=str,
                        help="'cuda', 'cuda:0', 'cpu', etc.")
    return parser.parse_args()


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_input(path: str, keypointtype: str) -> torch.Tensor:
    """Load a motion file and return a float32 tensor of shape (T, D)."""
    if keypointtype == "6d":
        res = smplx_to_6d(path)          # handles axis-angle → 6D conversion
        motion_6d = torch.tensor(res["motion_6d"], dtype=torch.float32)
        transl     = torch.tensor(res["transl"],    dtype=torch.float32)
        motion = torch.cat([transl, motion_6d], dim=1)   # (T, 135)
    else:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # pick the first array in the file
            key = list(arr.keys())[0]
            arr = arr[key]
        motion = torch.tensor(arr, dtype=torch.float32)  # (T, D)

    expected_dim = KEYPOINTTYPE_TO_DIM[keypointtype]
    if motion.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} feature dims for '{keypointtype}', "
            f"got {motion.shape[1]}"
        )
    return motion


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args  = ckpt["args"]

    model = MotionAutoencoder(
        input_dim  = KEYPOINTTYPE_TO_DIM[args["keypointtype"]],
        latent_dim = args["latent_dim"],
        hidden_dim = args["hidden_dim"],
        dropout    = args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = ckpt.get("norm_mean", None)
    std  = ckpt.get("norm_std",  None)
    if mean is not None:
        mean = mean.to(device)
        std  = std.to(device)

    return model, mean, std


# ── Core inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct(
    model: MotionAutoencoder,
    motion: torch.Tensor,       # (T, D)  in original scale
    mean: torch.Tensor | None,
    std:  torch.Tensor | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    recon  : np.ndarray  (T, D)  in original (un-normalised) scale
    latent : np.ndarray  (latent_dim,)
    """
    motion = motion.to(device)          # (T, D)
    x = motion.unsqueeze(0)             # (1, T, D)

    if mean is not None:
        x = (x - mean) / std

    recon_norm, z = model(x)            # (1, T, D),  (1, latent_dim)

    if mean is not None:
        recon = recon_norm * std + mean
    else:
        recon = recon_norm

    return recon.squeeze(0).cpu().numpy(), z.squeeze(0).cpu().numpy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, mean, std = load_checkpoint(args.checkpoint, device)
    if mean is None:
        print("  Warning: checkpoint has no normalisation stats — running without normalisation.")
    else:
        print(f"  Normalisation stats loaded  (mean range [{mean.min():.3f}, {mean.max():.3f}])")

    # Load input
    print(f"Loading input: {args.input}")
    motion = load_input(args.input, args.keypointtype)  # (T, D)
    print(f"  Raw sequence shape: {tuple(motion.shape)}")

    # Root-centre (same as training)
    motion = normalize_motion(motion.clone())
    print(f"  After normalize_motion: {tuple(motion.shape)}")

    # Optionally slice a window
    if args.window is not None:
        end = args.start_frame + args.window
        if end > motion.shape[0]:
            # pad with last frame
            pad = motion[-1:].repeat(end - motion.shape[0], 1)
            motion = torch.cat([motion[args.start_frame:], pad], dim=0)
        else:
            motion = motion[args.start_frame:end]
        print(f"  Window [{args.start_frame}:{args.start_frame + args.window}] → shape: {tuple(motion.shape)}")

    # Reconstruct
    recon, latent = reconstruct(model, motion, mean, std, device)
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Latent shape:         {latent.shape}")

    l1_err = np.abs(recon - motion.numpy()).mean()
    print(f"  Mean L1 reconstruction error: {l1_err:.5f}")

    # Save outputs
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + "_recon"

    recon_path  = args.output + ".npy"
    latent_path = args.output + "_latent.npy"
    os.makedirs(os.path.dirname(os.path.abspath(recon_path)), exist_ok=True)
    input_path = args.output + "_input.npy"
    np.save(input_path,  motion.numpy())
    np.save(recon_path,  recon)
    np.save(latent_path, latent)
    print(f"Saved input          → {input_path}")
    print(f"Saved reconstruction → {recon_path}")
    print(f"Saved latent         → {latent_path}")


if __name__ == "__main__":
    main()
