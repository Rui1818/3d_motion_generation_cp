# 3D Motion Generation for Children with Walking Disabilities

Master's Thesis project

This project trains a diffusion model to generate corrected 3D gait motions (with orthosis) conditioned on the pathological gait of a patient. The model is evaluated on a dataset of 15 subjects using 5-fold subject-level cross-validation.

The diffusion model code is adapted from [AGRoL](https://github.com/facebookresearch/AGRoL) (Du et al., CVPR 2023).

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate gait3d
```

---

## Project Structure

```
Thesis_project/
├── gait_train.py              # Single-run training
├── gait_crossval.py           # 5-fold cross-validation training
├── gait_crossval_eval.py      # Cross-validation evaluation
├── gait_generate.py           # Motion generation + metric computation
├── train_autoencoder.py       # Train autoencoder for FID computation
├── compute_fid.py             # Standalone FID computation
├── plot_crossval_loss.py      # Plot training/validation loss curves
├── summarize_crossval.py      # Summarize cross-val metrics across folds
├── run_gait_training_final.sh # Main experiment runner (calls gait_crossval.py)
├── run_crossval_eval.sh       # Evaluation runner (calls gait_crossval_eval.py)
├── data_loaders/              # Dataset and dataloader
├── diffusion/                 # Diffusion process (DDPM, Gaussian diffusion)
├── model/                     # DiffMLP and DiffTransformer architectures
├── runner/                    # Training loop
├── utils/                     # Metrics, transforms, rotation utils, DCT
├── final_dataset/             # Training data (subject folders)
├── test_dataset/              # Held-out test data
└── results/                   # Generated motions and metrics
```

---

## Dataset

The dataset contains 3D gait recordings from 15 subjects. Each subject folder holds paired motion files:
- **Pathological gait** — input condition to the model
- **Corrected gait (with orthosis)** — generation target

Two motion representations are supported:
- `openpose` — 23 joints × 3 coordinates (69-dim)
- `6d` — 6D rotation representation + translation (135-dim)

The dataset should be placed at `final_dataset/`, with one sub-folder per subject:

```
final_dataset/
├── subject_01/
├── subject_02/
└── ...
```

---

## Training

### Cross-validation (recommended)

Runs 5-fold subject-level cross-validation using `gait_crossval.py`. Use the provided shell script to reproduce experiments:

```bash
bash run_gait_training_final.sh
```

Or run a single configuration manually:

```bash
python gait_crossval.py \
    --save_dir results/my_experiment \
    --dataset_path final_dataset \
    --dataset gait \
    --arch diffusion_DiffMLP \
    --keypointtype openpose \
    --input_motion_length 30 \
    --motion_nfeat 69 \
    --sparse_dim 69 \
    --latent_dim 512 \
    --layers 8 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --num_steps 70000 \
    --lr_anneal_steps 30000 \
    --batch_size 8 \
    --loss_func mse \
    --overwrite
```

Key arguments:

| Argument | Description |
|---|---|
| `--keypointtype` | `openpose` (69-dim) or `6d` (135-dim) |
| `--input_motion_length` | Window size in frames (30, 60, or 240) |
| `--arch` | `diffusion_DiffMLP` or `diffusion_DiffTransformer` |
| `--loss_func` | `mse` or `softdtw` |
| `--use_dct` | Apply DCT in frequency domain before diffusion |
| `--lambda_rot_vel` | Weight for rotational velocity auxiliary loss |
| `--lambda_transl_vel` | Weight for translational velocity auxiliary loss |
| `--cond_mask_prob` | Condition dropout probability (classifier-free guidance) |

### Single-run training

```bash
python gait_train.py \
    --save_dir results/single_run \
    --dataset_path final_dataset \
    --dataset gait \
    --keypointtype openpose \
    --input_motion_length 30 \
    --motion_nfeat 69 \
    --sparse_dim 69 \
    --arch diffusion_DiffMLP \
    --latent_dim 512 \
    --layers 8 \
    --lr 2e-4 \
    --num_steps 70000 \
    --batch_size 8 \
    --overwrite
```

---

## Evaluation

### Cross-validation evaluation

Runs generation on each fold's held-out subjects and aggregates metrics (MPJPE, PAMPJPE, DTW, jitter, FID):

```bash
bash run_crossval_eval.sh
```

Or run directly:

```bash
python gait_crossval_eval.py \
    --save_dir results/my_experiment \
    --dataset_path final_dataset \
    --num_folds 5 \
    --autoencoder_path checkpoints/best_autoencoder.pt
```

### Single-run generation

```bash
python gait_generate.py \
    <path/to/model.pt> \
    --output_dir results/generated
```

Metrics computed per sample: MPJPE, PAMPJPE, DTW (geodesic and trajectory), jitter.

### Summarize results

```bash
python summarize_crossval.py
```

### Plot training curves

```bash
python plot_crossval_loss.py
```

---

## FID Computation

FID requires a trained motion autoencoder as the feature extractor.

**1. Train the autoencoder:**

```bash
python train_autoencoder.py \
    --dataset_path final_dataset \
    --keypointtype openpose \
    --save_dir checkpoints/autoencoder
```

**2. Compute FID** (standalone, or done automatically in `gait_crossval_eval.py`):

```bash
python compute_fid.py \
    --autoencoder_path checkpoints/autoencoder/best_autoencoder.pt \
    --real_dir final_dataset \
    --generated_dir results/generated
```

---

## Visualization

Three interactive viewers are provided (require [aitviewer](https://github.com/eth-siplab/AitViewer)):

```bash
# View generated skeleton motions
python generationviewer.py

# View skeleton from .npy file
python skeletonviewer.py

# View SMPL-X body model
python smplviewer.py
```

---

## Credits

The diffusion model architecture and training loop are adapted from:

> **Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model**
> Y. Du, R. Kips, A. Pumarola, S. Starke, A. Thabet, A. Sanakoyeu — CVPR 2023
> [[Paper]](https://arxiv.org/abs/2304.08577) [[Code]](https://github.com/facebookresearch/AGRoL)
