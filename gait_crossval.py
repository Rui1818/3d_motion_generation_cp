import os
import copy
import json
import random

import numpy as np
import torch

from gait_train import train_diffusion_model
from data_loaders.dataloader3d import get_dataloader, load_data, MotionDataset, compute_dct_stats
from utils.parser_util import train_args


NUM_FOLDS = 5


def run_fold(args, train_subjects, val_subjects, fold_save_dir):
    """Run training for a single fold with subject-level train/val split."""
    args = copy.copy(args)
    args.save_dir = fold_save_dir

    os.makedirs(fold_save_dir, exist_ok=True)

    # Save args + fold info
    fold_info = vars(args).copy()
    fold_info["train_subjects"] = train_subjects
    fold_info["val_subjects"] = val_subjects
    with open(os.path.join(fold_save_dir, "args.json"), "w") as fw:
        json.dump(fold_info, fw, indent=4, sort_keys=True)

    # Load training data (filtered by train_subjects)
    print("Loading train data...")
    motion_clean, motion_w_o = load_data(
        args.dataset_path,
        "train",
        keypointtype=args.keypointtype,
        subjects=train_subjects,
    )

    dataset = MotionDataset(
        args.dataset,
        motion_clean,
        motion_w_o,
        input_motion_length=args.input_motion_length,
        use_dct=args.use_dct,
    )
    print("train dataset size:", len(dataset))

    dct_mean, dct_std = None, None
    if args.use_dct:
        print("Computing DCT normalization statistics from train fold...")
        dct_mean, dct_std = compute_dct_stats(dataset)
        print(f"DCT stats computed: mean range [{dct_mean.min():.3f}, {dct_mean.max():.3f}], "
              f"std range [{dct_std.min():.3f}, {dct_std.max():.3f}]")
        torch.save({"dct_mean": dct_mean, "dct_std": dct_std},
                   os.path.join(fold_save_dir, "dct_stats.pt"))
        dataset.dct_mean = dct_mean
        dataset.dct_std = dct_std

    dataloader = get_dataloader(
        dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Load val data (same dataset_path, filtered by val_subjects)
    print("Loading val data...")
    val_motion_clean, val_motion_w_o = load_data(
        args.dataset_path,
        "train",
        keypointtype=args.keypointtype,
        subjects=val_subjects,
    )

    val_dataset = MotionDataset(
        args.dataset,
        val_motion_clean,
        val_motion_w_o,
        input_motion_length=args.input_motion_length,
        mode="test",
        use_dct=args.use_dct,
        dct_mean=dct_mean,
        dct_std=dct_std,
    )
    print("val dataset size:", len(val_dataset))

    val_dataloader = get_dataloader(
        val_dataset, "test", batch_size=args.batch_size, num_workers=1
    )

    dct_stats = {"dct_mean": dct_mean, "dct_std": dct_std} if dct_mean is not None else None
    train_diffusion_model(args, dataloader, val_dataloader, dct_stats=dct_stats)


def main():
    args = train_args()

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")

    # Discover all subject folders in the dataset path
    all_subjects = sorted([
        d for d in os.listdir(args.dataset_path)
        if os.path.isdir(os.path.join(args.dataset_path, d))
    ])
    print(f"Found {len(all_subjects)} subjects in '{args.dataset_path}':")
    for s in all_subjects:
        print(f"  {s}")

    # Split subjects into NUM_FOLDS folds as evenly as possible
    folds = [list(a) for a in np.array_split(all_subjects, NUM_FOLDS)]
    print(f"\n{NUM_FOLDS}-fold subject split:")
    for i, fold in enumerate(folds):
        print(f"  Fold {i}: val={fold}")

    base_save_dir = args.save_dir

    for fold_idx, val_subjects in enumerate(folds):
        train_subjects = [s for s in all_subjects if s not in val_subjects]
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_idx}")

        if os.path.exists(fold_save_dir) and not args.overwrite:
            print(f"\nSkipping fold {fold_idx}: '{fold_save_dir}' already exists.")
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{NUM_FOLDS}")
        print(f"  Train ({len(train_subjects)}): {train_subjects}")
        print(f"  Val   ({len(val_subjects)}):   {val_subjects}")
        print(f"{'='*60}")

        run_fold(args, train_subjects, val_subjects, fold_save_dir)

    print("\nCross-validation complete.")


if __name__ == "__main__":
    main()
