"""
Training script for the MotionAutoencoder used for FID feature extraction.

Loss:
    L = L1(recon, motion)   applied separately for c1 and c2 with individual weight updates

c1 and c2 are treated as fully independent motion samples. The dataloader returns
pairs but the pairing is ignored — each sequence gets its own forward pass and
gradient update, equivalent to training on a flat dataset of all sequences.

Usage:
    python train_autoencoder.py \
        --dataset_path ./dataset/mydataset \
        --keypointtype 6d \
        --save_dir ./checkpoints/autoencoder \
        --input_motion_length 60 \
        --latent_dim 256 \
        --epochs 200 \
        --lr 1e-3
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loaders.dataloader3d import MotionDataset, get_dataloader, load_data
from model.motion_autoencoder import MotionAutoencoder


# ── Input dimension lookup ────────────────────────────────────────────────────
KEYPOINTTYPE_TO_DIM = {
    "6d": 135,        # 3 transl + 132 rotation (6d)
    "openpose": 69,   # 23 joints * 3
    "smpl": 66,       # 22 joints * 3
}

# Scale factor applied to translation dims (0-2) for 6d to match rotation range [-1, 1].
TRANSL_SCALE = 3.0


def scale_transl(motion):
    """Scale translation dims in-place on a (B, T, 135) tensor."""
    motion = motion.clone()
    motion[:, :, :3] = motion[:, :, :3] / TRANSL_SCALE
    return motion


def parse_args():
    parser = argparse.ArgumentParser(description="Train MotionAutoencoder for FID")

    # data
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument(
        "--keypointtype",
        default="6d",
        choices=["6d", "openpose", "smpl"],
        type=str,
    )
    parser.add_argument(
        "--input_motion_length",
        default=60,
        type=int,
        help="Window size in frames. Use 60 for augmented windowed training.",
    )
    parser.add_argument(
        "--val_dataset_path",
        default=None,
        type=str,
        help="Optional separate validation data path. Defaults to dataset_path.",
    )

    # model
    parser.add_argument("--latent_dim", default=128, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    # training
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--repeat_times",
        default=10,
        type=int,
        help="How many times to repeat the dataset per epoch (random windows differ each time).",
    )
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=0, type=int)

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for _, motion_c1, motion_c2 in loader:
        motion_c1 = motion_c1.to(device)
        motion_c2 = motion_c2.to(device)

        for motion in (motion_c1, motion_c2):
            if motion.shape[-1] == 135:
                motion = scale_transl(motion)
            recon, _ = model(motion)
            loss = F.l1_loss(recon, motion)
            total_loss += loss.item() * motion.shape[0]
            n += motion.shape[0]

    return total_loss / n


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = KEYPOINTTYPE_TO_DIM[args.keypointtype]
    print(f"Motion feature dimension: {input_dim}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading training data...")
    motion_clean, motion_w_o = load_data(
        args.dataset_path, "train", keypointtype=args.keypointtype
    )
    train_dataset = MotionDataset(
        dataset="gait",
        motion_clean=motion_clean,
        motion_without_orth=motion_w_o,
        input_motion_length=args.input_motion_length,
        train_dataset_repeat_times=args.repeat_times,
        mode="train",
    )
    print(f"Train dataset size: {len(train_dataset)}")
    train_loader = get_dataloader(
        train_dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )

    val_path = args.val_dataset_path or args.dataset_path.replace("mydataset", "test_dataset")
    print("Loading validation data...")
    val_clean, val_w_o = load_data(val_path, "train", keypointtype=args.keypointtype)
    val_dataset = MotionDataset(
        dataset="gait",
        motion_clean=val_clean,
        motion_without_orth=val_w_o,
        input_motion_length=args.input_motion_length,
        train_dataset_repeat_times=1,
        mode="test",
    )
    print(f"Val dataset size: {len(val_dataset)}")
    val_loader = get_dataloader(
        val_dataset, "test", batch_size=args.batch_size, num_workers=0
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MotionAutoencoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MotionAutoencoder: {n_params:.2f}M parameters")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Data scale diagnostics ────────────────────────────────────────────────
    first_batch = next(iter(train_loader))
    _, sample_c1, _ = first_batch
    if args.keypointtype == "6d":
        print(f"[Diagnostics] Translation (dims 0-2)  min={sample_c1[:, :, :3].min():.4f}  max={sample_c1[:, :, :3].max():.4f}  std={sample_c1[:, :, :3].std():.4f}")
        print(f"[Diagnostics] Rotation    (dims 3-134) min={sample_c1[:, :, 3:].min():.4f}  max={sample_c1[:, :, 3:].max():.4f}  std={sample_c1[:, :, 3:].std():.4f}")
    else:
        print(f"[Diagnostics] Motion min={sample_c1.min():.4f}  max={sample_c1.max():.4f}  std={sample_c1.std():.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"))

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        n = 0
        n_updates = 0

        for _, motion_c1, motion_c2 in train_loader:
            motion_c1 = motion_c1.to(device)
            motion_c2 = motion_c2.to(device)

            for motion in (motion_c1, motion_c2):
                if motion.shape[-1] == 135:
                    motion = scale_transl(motion)
                recon, _ = model(motion)
                loss = F.l1_loss(recon, motion)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * motion.shape[0]
                total_grad_norm += grad_norm.item()
                n += motion.shape[0]
                n_updates += 1

        scheduler.step()

        avg_train = total_loss / n
        avg_grad_norm = total_grad_norm / n_updates
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={avg_train:.4f} | val loss={val_loss:.4f} | "
            f"grad norm={avg_grad_norm:.4f}"
        )
        writer.add_scalar("Loss/train", avg_train, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("GradNorm/train", avg_grad_norm, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, "best_autoencoder.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            # Also save the encoder alone for convenient FID feature extraction
            encoder_path = os.path.join(args.save_dir, "best_encoder.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "val_loss": val_loss,
                    "input_dim": input_dim,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "args": vars(args),
                },
                encoder_path,
            )
            print(f"  → New best val loss {val_loss:.4f}, checkpoint saved.")

    writer.close()

    # Save final model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        },
        os.path.join(args.save_dir, "final_autoencoder.pt"),
    )
    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
