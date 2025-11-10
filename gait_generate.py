import os
import random
import numpy as np
import torch
from tqdm import tqdm

from utils.parser_util import sample_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.dataloader3d import load_data, MotionDataset, drop_duplicate_frames, subtract_root

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

def prepare_conditional_motion(file_path, input_motion_length):
    """
    Loads and preprocesses the conditional motion data from a given file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Conditional motion file not found: {file_path}")

    motion_w_o_np = drop_duplicate_frames(np.load(file_path))
    motion_w_o_np = motion_w_o_np[..., :3]
    motion_w_o_np = subtract_root(motion_w_o_np)
    motion_w_o_np = motion_w_o_np.reshape(-1, 72)
    motion_w_o = torch.tensor(motion_w_o_np, dtype=torch.float32)

    # Crop or pad the conditional motion to the model's input length
    seqlen_wo = motion_w_o.shape[0]
    if seqlen_wo <= input_motion_length:
        if seqlen_wo > 0:
            frames_to_add = input_motion_length - seqlen_wo
            last_frame_wo = motion_w_o[-1:]
            padding_wo = last_frame_wo.repeat(frames_to_add, 1)
            motion_w_o = torch.cat([motion_w_o, padding_wo], dim=0)
    else:
        # If longer, just take the beginning
        motion_w_o = motion_w_o[:input_motion_length]

    return motion_w_o.unsqueeze(0)  # Add batch dimension

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
    
    


    
    file_path = os.path.join("observations", "753", "vitpose_c2_a1", "vitpose", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
    file_path_turn = os.path.join("observations", "753", "vitpose_c2_a3", "vitpose", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Conditional motion file not found: {file_path}")

    # The model expects a batch, so add a batch dimension
    cond_motion = prepare_conditional_motion(file_path, args.input_motion_length).to(device)
    cond_motion_turn = prepare_conditional_motion(file_path_turn, args.input_motion_length).to(device)

    print(f"Condition motion shape: {cond_motion.shape}")

    # --- Run the generation ---
    print("Generating motion...")

    generated_motion = sample(model, diffusion, cond_motion, args)
    generated_motion_turn = sample(model, diffusion, cond_motion_turn, args)

    print("Motion generation complete.")

    # --- Save the output ---
    # The output is a tensor on the GPU, move it to CPU and convert to numpy
    generated_motion_np = generated_motion.squeeze(0).cpu().numpy()
    generated_motion_turn_np = generated_motion_turn.squeeze(0).cpu().numpy()

    # Create an output directory if it doesn't exist
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), "generated_motions")
    
    os.makedirs(args.output_dir, exist_ok=True)

    output_filename = f"generated_motion_c1_a1.npy"
    output_path = os.path.join(args.output_dir, output_filename)
    output_filename_turn = f"generated_motion_c1_a3.npy"
    output_path_turn = os.path.join(args.output_dir, output_filename_turn)

    np.save(output_path, generated_motion_np)
    np.save(output_path_turn, generated_motion_turn_np)
    print(f"Generated motion saved to: {output_path}")
    print(f"Shape of saved motion: {generated_motion_np.shape}")
    print(f"Generated motion saved to: {output_path_turn}")
    print(f"Shape of saved motion_turn: {generated_motion_turn_np.shape}")

if __name__ == "__main__":
    main()
