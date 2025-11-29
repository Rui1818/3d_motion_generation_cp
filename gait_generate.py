import os
import random
import numpy as np
import torch
from tqdm import tqdm

from utils.parser_util import sample_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.dataloader3d import load_data, MotionDataset, drop_duplicate_frames, subtract_root, get_dataloader

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
    clean, cond = load_data(file_path, keypointtype)
    dataset = MotionDataset(
        "gait",
        clean,
        cond,
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
    file_path = ""
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
        args.output_dir = os.path.join(os.path.dirname(args.model_path), "generated_motions"+name)
    for i, batch in enumerate(tqdm(dataloader)):
        reference, condition = batch
        condition=condition.to(device)
        print(reference.shape, condition.shape)  # shapes of the batch
        print(f"Condition motion shape: {condition.shape}")

        # --- Run the generation ---
        generated_motion = sample(model, diffusion, condition, args)
        generated_motion_np = generated_motion.squeeze(0).cpu().numpy()
        reference_np=reference.squeeze(0).cpu().numpy()
        ref_path=os.path.join(args.outputdir, "reference_motion_"+i+".npy")
        gen_path=os.path.join(args.outputdir, "generated_motion_"+i+".npy")
        np.save(ref_path, reference_np)
        np.save(gen_path, generated_motion_np)

    print("Motion generation complete.")

if __name__ == "__main__":
    main()
