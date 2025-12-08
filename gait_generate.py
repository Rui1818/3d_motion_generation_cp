import os
import random
import numpy as np
import torch
from tqdm import tqdm

from utils.parser_util import sample_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.dataloader3d import TestDataset, load_data, MotionDataset, get_dataloader
from utils.transformation_sixd import sixd_to_smplx
from scipy.ndimage import gaussian_filter1d
from utils.metrics import calculate_motion_dtw

def remove_padding_3d_numpy(sequence):
    # sequence shape: (frames, x, y)

    diffs = sequence[1:] - sequence[:-1]

    non_zero_diff = np.any(diffs != 0, axis=(1, 2))
    change_indices = np.where(non_zero_diff)[0]
    
    if len(change_indices) == 0:
        return sequence[:1]
    
    last_real_index = change_indices[-1] + 2
    
    return sequence[:last_real_index]

def load_diffusion_model(args):
    """
    Loads the diffusion model and its configuration from a checkpoint.
    """
    print("Creating model and diffusion...")
    # The model architecture is stored with a prefix e.g., "diffusion_DiffMLP"
    # We need to remove the prefix to get the actual architecture name.
    if args.arch.startswith("diffusion_"):
        args.arch = args.arch[len("diffusion_"):]
    
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model.to("cuda:0")
    model.eval()  # Set model to evaluation mode
    return model, diffusion

def prepare_conditional_motion(file_path, input_motion_length, keypointtype):
    """
    Loads and preprocesses the conditional motion data from a given file path.
    """
    clean, cond, betas = load_data(file_path, split="test", keypointtype=keypointtype)
    dataset = TestDataset(
        "gait",
        clean,
        cond,
        betas=betas,
        input_motion_length=input_motion_length,
    )

    #return motion_w_o.unsqueeze(0)  # Add batch dimension
    return get_dataloader(dataset, "test", batch_size=1, num_workers=1)

def sample(model, diffusion, cond_motion, args):
    """
    Generates motion samples using the diffusion model conditioned on the provided motion.
    """
    batch_size = cond_motion.shape[0]
    output_shape = (batch_size, args.input_motion_length, args.motion_nfeat)

    sample_fn = diffusion.p_sample_loop

    generated_motion = sample_fn(
        model,
        output_shape,
        sparse=cond_motion,  # Pass the conditional motion here
        clip_denoised=False,
        model_kwargs=None,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    return generated_motion

def main():
    """
    Main function to run the motion generation process.
    """
    # Parse arguments, loading model configuration from the saved args.json
    args = sample_args()

    # Set random seeds for reproducibility
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained diffusion model
    model, diffusion = load_diffusion_model(args)

    # --- Prepare a single data sample for inference ---
    # This part is a simplified version of your dataloader to get one sample.
    # You can modify this to select a specific file.
    print("Loading a sample from the dataset...")
    
    


    #TODO change sample dataset
    file_path = "test_dataset"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Conditional motion file not found: {file_path}")

    # The model expects a batch, so add a batch dimension
    dataloader = prepare_conditional_motion(file_path, args.input_motion_length, args.keypointtype)
    #cond_motion = prepare_conditional_motion(file_path, args.input_motion_length).to(device)
    #cond_motion_turn = prepare_conditional_motion(file_path_turn, args.input_motion_length).to(device)
    print("Generating motion...")
        # Create an output directory if it doesn't exist
    if not args.output_dir:
        name, _ = os.path.splitext(os.path.basename(args.model_path))
        args.output_dir="results/"
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize list to store DTW metrics
    dtw_metrics = []

    for i, batch in enumerate(tqdm(dataloader)):
        #6d case| keypoint case
        #referenceshape: (1, frames, 22, 3)| (1,frames,69)
        #condition shape: (1, frames, 135)| (1,frames,69)
        #betas shape: (1,frames, 10/11)|_
        reference, condition, betas = batch
        condition=condition.to(device)
        print(reference.shape, condition.shape)  # shapes of the batch
        print(f"Condition motion shape: {condition.shape}")

        # --- Run the generation ---
        generated_motion = sample(model, diffusion, condition, args)
        generated_motion_np = generated_motion.squeeze(0).cpu().numpy()
        reference_np=reference.squeeze(0).cpu().numpy()
        if args.keypointtype=='6d':
            assert generated_motion_np.shape[1]==135
            betas_np=betas.squeeze(0).cpu().numpy()
            generated_motion_np = gaussian_filter1d(generated_motion_np, sigma=1, axis=0)
            generated_motion_np=sixd_to_smplx({'motion_6d': generated_motion_np[:,3:], 'transl': generated_motion_np[:,:3], 'betas': betas_np})
        elif args.keypointtype=='openpose':
            assert generated_motion_np.shape[1]==69
            reference_np=reference_np.reshape(-1,23,3)
            generated_motion_np=generated_motion_np.reshape(-1,23,3)
        ref_path=os.path.join(args.output_dir, "reference_motion_"+str(i)+".npy")
        gen_path=os.path.join(args.output_dir, "generated_motion_"+str(i)+".npy")
        np.save(ref_path, reference_np)
        np.save(gen_path, generated_motion_np)

        # Calculate DTW between reference and generated motion
        try:
            #remove padding
            #generated_motion_np=remove_padding_3d_numpy(generated_motion_np)
            dtw_distance, _ = calculate_motion_dtw(reference_np, generated_motion_np)
            dtw_metrics.append({
                'sample_id': i,
                'dtw_distance': dtw_distance,
                'reference_frames': reference_np.shape[0],
                'generated_frames': generated_motion_np.shape[0]
            })
            print(f"Sample {i} - DTW Distance: {dtw_distance:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate DTW for sample {i}: {e}")

    print("Motion generation complete.")

    # Log summary statistics
    if dtw_metrics:
        dtw_distances = [m['dtw_distance'] for m in dtw_metrics]
        print("\n=== DTW Metrics Summary ===")
        print(f"Total samples: {len(dtw_metrics)}")
        print(f"Mean DTW Distance: {np.mean(dtw_distances):.4f}")
        print(f"Std DTW Distance: {np.std(dtw_distances):.4f}")
        print(f"Min DTW Distance: {np.min(dtw_distances):.4f}")
        print(f"Max DTW Distance: {np.max(dtw_distances):.4f}")

        # Save metrics to file
        metrics_path = os.path.join(args.output_dir, "dtw_metrics.npy")
        np.save(metrics_path, dtw_metrics)
        print(f"\nDTW metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
