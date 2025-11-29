# Parser / CLI Arguments (brief)

A compact export of all argparse options defined in utils/parser_util.py. Each line shows the flag, type & default, and a one-line comment.

## Base
- `--cuda` (bool, default=True) — use CUDA device; otherwise CPU.
- `--device` (int, default=0) — GPU device id.
- `--seed` (int, default=10) — random seed.
- `--batch_size` (int, default=64) — training / eval batch size.
- `--timestep_respacing` (str, default="") — ddim timestep respacing string.

## Diffusion
- `--noise_schedule` (str, default="cosine") — noise schedule type ("linear" or "cosine").
- `--diffusion_steps` (int, default=1000) — number of diffusion timesteps (T).
- `--sigma_small` (bool, default=True) — use smaller sigma values.

## Model
- `--arch` (str, default="DiffMLP") — model architecture name.
- `--motion_nfeat` (int, default=132) — per-frame motion feature dimension.
- `--sparse_dim` (int, default=54) — sparse conditioning feature dimension.
- `--layers` (int, default=8) — number of model layers.
- `--latent_dim` (int, default=512) — model hidden / embedding width.
- `--cond_mask_prob` (float, default=0.0) — probability to mask condition (classifier-free guidance).
- `--input_motion_length` (int, default=196) — max frames / input sequence length.
- `--no_normalization` (flag) — disable data normalization.

## Dataset / Data options
- `--dataset` (str, default=None, choices=["amass"]) — dataset name.
- `--dataset_path` (str, default="./dataset/AMASS/") — dataset root path.
- `--keypointtype` (str, default="openpose", choices=["6d", "smpl", "openpose"]) — type of keypoints to use.

## Training
- `--save_dir` (str, required) — directory to save checkpoints & outputs.
- `--overwrite` (flag) — allow reusing existing save_dir.
- `--train_platform_type` (str, default="NoPlatform") — logging platform choice.
- `--lr` (float, default=2e-4) — learning rate.
- `--weight_decay` (float, default=0.0) — optimizer weight decay.
- `--lr_anneal_steps` (int, default=0) — LR anneal steps.
- `--lambda_rot_vel` (float, default=0.0) — weight for rotation velocity loss.
- `--lambda_transl_vel` (float, default=0.0) — weight for translation velocity loss.
- `--train_dataset_repeat_times` (int, default=1000) — repeat factor for training dataset.
- `--eval_during_training` (flag) — run evaluation during training.
- `--log_interval` (int, default=100) — log every N steps.
- `--save_interval` (int, default=5000) — save/checkpoint every N steps.
- `--num_steps` (int, default=6000000) — total training steps.
- `--resume_checkpoint` (str, default="") — path to checkpoint to resume from.
- `--load_optimizer` (flag) — also load optimizer state when resuming.
- `--num_workers` (int, default=8) — dataloader worker count.

## Sampling
- `--overlapping_test` (flag) — enable overlapping test mode.
- `--num_per_batch` (int, default=256) — per-split batch size for non-overlapping test.
- `--sld_wind_size` (int, default=70) — sliding window size.
- `--vis` (flag) — visualize outputs.
- `--fix_noise` (flag) — fix initial noise for reproducible sampling.
- `--fps` (int, default=30) — output frames per second.
- `--model_path` (str, required for sampling) — path to model####.pt for sampling.
- `--output_dir` (str, default="") — results output directory.
- `--support_dir` (str) — path to SMPLH / DMPL support dirs.

## Evaluation
- `--model_path` (str, required for eval) — path to model####.pt for evaluation.

## Notes
- `parse_and_load_from_model(parser)` will load saved model args from `args.json` next to the checkpoint and overwrite the arguments in the `dataset`, `model`, and `diffusion` groups (except it preserves `--dataset` if the user specified it).
- Use `train_args()`, `sample_args()`, or `evaluation_parser()` to get the correct set of user-specified flags for each mode.
