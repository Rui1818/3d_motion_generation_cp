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
from scipy.linalg import sqrtm

from gait_generate import (
    load_diffusion_model,
    sample,
    calculate_metrics,
    transform_motion_back,
)
from data_loaders.dataloader3d import TestDataset, load_data, get_dataloader
from model.motion_autoencoder import MotionAutoencoder
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

# Must match train_autoencoder.py
_TRANSL_SCALE = 3.0
_KEYPOINTTYPE_TO_DIM = {"6d": 135, "openpose": 69, "smpl": 66}


NUM_FOLDS = 5

# ── FID utilities ─────────────────────────────────────────────────────────────

def load_encoder(path, device):
    """
    Load MotionEncoder from a full autoencoder checkpoint saved by train_autoencoder.py.
    Returns (encoder, latent_dim).
    """
    ckpt = torch.load(path, map_location="cpu")
    saved_args = ckpt["args"]
    input_dim = _KEYPOINTTYPE_TO_DIM[saved_args["keypointtype"]]
    model = MotionAutoencoder(
        input_dim=input_dim,
        latent_dim=saved_args["latent_dim"],
        hidden_dim=saved_args.get("hidden_dim", 256),
        dropout=saved_args.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"Loaded encoder: {path}  (epoch {ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss', float('nan')):.4f})")
    return model.encoder, saved_args["latent_dim"]


def _preprocess_window(window):
    """
    Apply the same normalisation used during autoencoder training to a single
    (window_len, D) numpy window.  Modifies a copy; does not mutate the input.
    """
    w = window.copy()
    D = w.shape[1]
    if D == 135:  # 6D representation
        w[:, :3] -= w[0, :3]           # subtract first-frame root translation
        w[:, :3] /= _TRANSL_SCALE      # scale to match rotation range
    elif D == 69:  # OpenPose 23 joints
        joints = w.reshape(-1, 23, 3)
        root = (joints[0, 7, :] + joints[0, 10, :]) / 2
        joints -= root
        w = joints.reshape(-1, 69)
    return w


@torch.no_grad()
def extract_features(motion_list, encoder, window_len, device, batch_size=64):
    """
    Encode a list of (T, D) numpy arrays into latent feature vectors.

    Each sequence is sliced into non-overlapping windows of `window_len` frames;
    any remainder shorter than a full window is dropped.

    Returns an (N_windows, latent_dim) numpy array.
    """
    windows = []
    for motion in motion_list:
        if isinstance(motion, torch.Tensor):
            motion = motion.numpy()
        T = motion.shape[0]
        for start in range(0, T - window_len + 1, window_len):
            w = _preprocess_window(motion[start : start + window_len])
            windows.append(w)

    if not windows:
        return np.zeros((0, 1), dtype=np.float32)

    all_feats = []
    for i in range(0, len(windows), batch_size):
        batch = torch.tensor(
            np.stack(windows[i : i + batch_size]), dtype=torch.float32
        ).to(device)
        all_feats.append(encoder(batch).cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def compute_fid(real_feats, gen_feats):
    """
    Fréchet Inception Distance between two (N, D) feature arrays.
    A small epsilon is added to the covariance diagonals for numerical stability.
    """
    n_real, n_gen = len(real_feats), len(gen_feats)
    latent_dim = real_feats.shape[1]
    if n_real < latent_dim * 2 or n_gen < latent_dim * 2:
        print(
            f"  WARNING: FID may be unreliable — "
            f"real windows={n_real}, gen windows={n_gen}, latent_dim={latent_dim}. "
            f"Recommended minimum: {latent_dim * 2} windows per split."
        )

    eps = 1e-6
    mu_r, mu_g = real_feats.mean(axis=0), gen_feats.mean(axis=0)
    sigma_r = np.cov(real_feats, rowvar=False) + np.eye(latent_dim) * eps
    sigma_g = np.cov(gen_feats, rowvar=False) + np.eye(latent_dim) * eps

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        if np.abs(covmean.imag).max() > 1e-3:
            print(
                f"  WARNING: FID covmean has large imaginary component "
                f"({np.abs(covmean.imag).max():.4f}). Result may be inaccurate."
            )
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))


# ── Fold evaluation ───────────────────────────────────────────────────────────

def _sliding_window_start_indices(total_frames, window_size):
    """
    Replicates the start-index logic from gait_generate.sample() so that
    per-window reference slices align with the generated windows.
    """
    if window_size == 60:
        step = 40
    elif window_size == 30:
        step = 20
    else:
        raise ValueError(f"Unsupported sliding window size: {window_size}")

    indices = list(range(0, total_frames - window_size + 1, step))
    if not indices:
        indices = [0]
    elif indices[-1] + window_size < total_frames:
        indices.append(total_frames - window_size)
    return indices


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


def eval_fold(fold_dir, dataset_path, encoder=None):
    """
    Evaluate one fold: load model, run generation on val subjects, return metrics.

    Returns:
        fold_metrics  : list of per-sample metric dicts
        real_raw      : list of (T, D) numpy arrays — reference sequences
        gen_raw       : list of (T, D) numpy arrays — generated sequences (pre-SMPL)
        window_len    : input_motion_length used by this fold's model
    """
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
    real_raw = []   # full reference sequences; extract_features will window them for FID
    gen_raw = []    # individual generated windows (window_size, D) for FID

    ws = args.input_motion_length

    for i, batch in enumerate(tqdm(dataloader, desc="  Evaluating")):
        reference, condition, batch_betas = batch
        condition = condition.to(device)

        if use_sliding_window:
            generated_motion_windows, generated_motion = sample(
                model, diffusion, condition, args,
                use_sliding_window=True, dct_stats=dct_stats
            )

            gen_windows_np = generated_motion_windows.squeeze(0).cpu().numpy()  # (n_windows*ws, D)
            reference_np   = reference.squeeze(0).cpu().numpy()                 # (T, D)
            total_frames   = condition.shape[1]
            start_indices  = _sliding_window_start_indices(total_frames, ws)
            n_windows      = len(start_indices)

            # ── Per-window metrics ────────────────────────────────────────────
            window_metrics = []
            for k in range(n_windows):
                gen_win = gen_windows_np[k * ws : (k + 1) * ws]   # (ws, D)

                # Slice the corresponding reference window; pad if at sequence end
                ref_start = start_indices[k]
                ref_end   = ref_start + ws
                if ref_end > reference_np.shape[0]:
                    ref_win  = reference_np[ref_start:]
                    pad      = np.tile(ref_win[-1:], (ref_end - reference_np.shape[0], 1))
                    ref_win  = np.concatenate([ref_win, pad], axis=0)
                else:
                    ref_win = reference_np[ref_start:ref_end]

                win_res = calculate_metrics(ref_win, gen_win, args.keypointtype, k)

                if args.keypointtype == "6d":
                    # betas are per-subject (constant), slice to window length
                    win_betas = batch_betas[:, :ws, :] if batch_betas.dim() == 3 else batch_betas
                    gen_win_back, ref_win_back = transform_motion_back(
                        args, win_betas, gen_win.copy(), ref_win.copy()
                    )
                    mpjpe_6d, pampjpe_6d, jitter = calculate_metrics(
                        ref_win_back, gen_win_back, "6d_transformed", k
                    )
                    win_res["MPJPE_6d"]   = mpjpe_6d
                    win_res["PAMPJPE_6d"] = pampjpe_6d
                    win_res["jitter"]     = jitter

                window_metrics.append(win_res)

                # Each generated window is one FID sample
                gen_raw.append(gen_win)

            # Average per-window metrics into a single sample-level result
            res = {"sample_id": i}
            for key in [key for key in window_metrics[0].keys() if key != "sample_id"]:
                vals = [m[key] for m in window_metrics if key in m]
                res[key] = float(np.mean(vals))

            # Full reference sequence for FID (extract_features will window it)
            real_raw.append(reference_np)

        else:
            generated_motion = sample(
                model, diffusion, condition, args,
                use_sliding_window=False, dct_stats=dct_stats
            )

            generated_np = generated_motion.squeeze(0).cpu().numpy()
            reference_np = reference.squeeze(0).cpu().numpy()

            res = calculate_metrics(reference_np, generated_np, args.keypointtype, i)

            if args.keypointtype == "6d":
                gen_back, ref_back = transform_motion_back(
                    args, batch_betas, generated_np.copy(), reference_np.copy()
                )
                mpjpe_6d, pampjpe_6d, jitter = calculate_metrics(
                    ref_back, gen_back, "6d_transformed", i
                )
                res["MPJPE_6d"]   = mpjpe_6d
                res["PAMPJPE_6d"] = pampjpe_6d
                res["jitter"]     = jitter

            # Single generated window and full reference for FID
            gen_raw.append(generated_np)
            real_raw.append(reference_np)

        fold_metrics.append(res)

    return fold_metrics, real_raw, gen_raw, ws


# ── Summary ───────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, type=str,
                        help="Base crossval save directory (contains fold_0/, fold_1/, ...)")
    parser.add_argument("--dataset_path", required=True, type=str,
                        help="Path to final_dataset (same as used for training)")
    parser.add_argument("--num_folds", default=NUM_FOLDS, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--autoencoder_path", default=None, type=str,
                        help="Path to best_autoencoder.pt for FID computation (optional).")
    args_main = parser.parse_args()

    random.seed(args_main.seed)
    np.random.seed(args_main.seed)
    torch.manual_seed(args_main.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder once, shared across all folds
    encoder = None
    if args_main.autoencoder_path:
        encoder, _ = load_encoder(args_main.autoencoder_path, device)

    all_metrics = []
    all_real_raw = []
    all_gen_raw = []
    window_len = None  # filled from first successfully evaluated fold

    for fold_idx in range(args_main.num_folds):
        fold_dir = os.path.join(args_main.save_dir, f"fold_{fold_idx}")
        if not os.path.exists(fold_dir):
            print(f"Fold {fold_idx}: directory not found, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args_main.num_folds}  ({fold_dir})")
        print(f"{'='*60}")

        try:
            fold_metrics, real_raw, gen_raw, fold_window_len = eval_fold(
                fold_dir, args_main.dataset_path, encoder=encoder
            )
        except Exception as e:
            print(f"  ERROR evaluating fold {fold_idx}: {e}")
            continue

        if window_len is None:
            window_len = fold_window_len

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
        all_real_raw.extend(real_raw)
        all_gen_raw.extend(gen_raw)

    # Detect keypointtype from the first fold's args.json
    first_fold_args_path = os.path.join(args_main.save_dir, "fold_0", "args.json")
    keypointtype = "openpose"
    if os.path.exists(first_fold_args_path):
        with open(first_fold_args_path) as f:
            keypointtype = json.load(f).get("keypointtype", "openpose")

    print_summary(all_metrics, keypointtype)

    # ── Global FID ────────────────────────────────────────────────────────────
    if encoder is not None and all_real_raw and window_len is not None:
        print(f"\nExtracting FID features (window_len={window_len})...")
        real_feats = extract_features(all_real_raw, encoder, window_len, device)
        gen_feats  = extract_features(all_gen_raw,  encoder, window_len, device)
        print(f"  Real windows: {len(real_feats)}  |  Gen windows: {len(gen_feats)}")

        fid = compute_fid(real_feats, gen_feats)
        print(f"\n  FID (global, all folds): {fid:.4f}")

        # Append FID to the saved all-metrics file so it is recorded
        fid_record = {"FID_global": fid, "real_windows": len(real_feats), "gen_windows": len(gen_feats)}
        fid_path = os.path.join(args_main.save_dir, "fid_result.npy")
        np.save(fid_path, fid_record)
        print(f"  FID result saved → {fid_path}")

    # Save all per-sample metrics
    all_metrics_path = os.path.join(args_main.save_dir, "crossval_metrics.npy")
    np.save(all_metrics_path, all_metrics)
    print(f"\nAll metrics saved → {all_metrics_path}")


if __name__ == "__main__":
    main()
