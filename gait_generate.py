import os
import random
import fastdtw
import numpy as np
import torch
from tqdm import tqdm

from utils.parser_util import sample_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.dataloader3d import TestDataset, load_data, MotionDataset, get_dataloader
from utils.transformation_sixd import sixd_to_smplx, smplx_to_6d
from scipy.ndimage import gaussian_filter1d
from utils.metrics import calculate_motion_dtw, pose_distance_metric

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

def change_motion_position(motion, offset=None):
    #motion: tensor (1, frames, dim)
    #only batch size 1 supported
    if motion.shape[1]==22:
        root=motion[0,0,:]
        motion=motion - root
        return motion

    motion=motion.squeeze(0)  # (frames, dim)
    
    offset_val = None
    if offset is not None:
        offset_val = offset.reshape(-1, motion.shape[1])[0]

    if motion.shape[1]==69:
        # 1. Calculate the root (average of joint 7 and joint 10) 
        # Joint 7 is indices 21:24, Joint 10 is indices 30:33
        root=(motion[0, 21:24] + motion[0, 30:33]) / 2
        root = root if offset is None else root-((offset_val[21:24] + offset_val[30:33]) / 2)
        # 2. Subtract the root from the motion data
        # We subtract root[0] from all X's, root[1] from all Y's, and root[2] from all Z's
        motion[:, 0::3] -= root[0]  # All X coordinates
        motion[:, 1::3] -= root[1]  # All Y coordinates
        motion[:, 2::3] -= root[2]  # All Z coordinates
    elif motion.shape[1]==135:
        root = motion[0, :3] if offset is None else motion[0, :3]-offset_val[:3]
        motion[:, :3] = motion[:, :3] - root
    else:
        raise ValueError("Unknown motion dimension for normalization.")
    motion=motion.unsqueeze(0)
    return motion

def linear_blend_motion(motion1, motion2):
    # motion1: (B, L, D) - fading out
    # motion2: (B, L, D) - fading in
    
    length = motion1.shape[1]
    device = motion1.device
    
    alpha = torch.linspace(0, 1, length, device=device).view(1, -1, 1)
    blended = (1 - alpha) * motion1 + alpha * motion2
    return blended


def sample(model, diffusion, cond_motion, args, use_sliding_window=False, sliding_window_step=20):
    """
    Generates motion samples using the diffusion model conditioned on the provided motion.

    Args:
        model: The diffusion model
        diffusion: Diffusion process
        cond_motion: Conditional motion input (batch_size, total_frames, features)
        args: Arguments containing model configuration
        use_sliding_window: If True, generates motion using sliding windows
        sliding_window_step: Number of frames to slide the window after each generation

    Returns:
        Generated motion tensor
    """
    batch_size = cond_motion.shape[0]
    total_frames = cond_motion.shape[1]
    window_size = args.input_motion_length
    

    if not use_sliding_window:
        # Original behavior: generate entire motion at once
        output_shape = (batch_size, args.input_motion_length, args.motion_nfeat)
        sample_fn = diffusion.p_sample_loop

        generated_motion = sample_fn(
            model,
            output_shape,
            sparse=cond_motion,
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

    # Sliding window generation
    sample_fn = diffusion.p_sample_loop

    # Initialize output tensor to store all generated frames
    all_generated_frames = []
    generated_frames_concat=[]

    # Calculate start indices for windows
    start_indices = list(range(0, total_frames - window_size + 1, sliding_window_step))
    if not start_indices:
        start_indices = [0]
    elif start_indices[-1] + window_size < total_frames:
        start_indices.append(total_frames - window_size)
    
    last_concat_end_idx = 0

    for window_idx, start_idx in enumerate(start_indices):
        end_idx = min(start_idx + window_size, total_frames)

        # Extract window from conditional motion
        cond_window = cond_motion[:, start_idx:end_idx, :].clone()
        #root normalization for inference
        cond_window = change_motion_position(cond_window)

        # Pad if necessary (last window might be shorter)
        if cond_window.shape[1] < window_size:
            padding_length = window_size - cond_window.shape[1]
            last_frame = cond_window[:, -1:, :]
            padding = last_frame.repeat(1, padding_length, 1)
            cond_window = torch.cat([cond_window, padding], dim=1)

        # Generate motion for this window
        output_shape = (batch_size, window_size, args.motion_nfeat)
        generated_window = sample_fn(
            model,
            output_shape,
            sparse=cond_window,
            clip_denoised=False,
            model_kwargs=None,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # For the first window, take all frames
        if window_idx == 0:
            all_generated_frames.append(generated_window)
            generated_frames_concat.append(generated_window)
            last_concat_end_idx = window_size
        # For subsequent windows, only take the new frames (after the overlap)
        else:
            #TODO apply smoothing between windows
            # Take only the frames that extend beyond previous windows
            frames_to_take = start_idx + window_size - last_concat_end_idx

            frame_offset = generated_frames_concat[-1][:, -1:, :]
            overlaplen=window_size - frames_to_take  
            # append full window
            all_generated_frames.append(generated_window)

            # overlap motion blending
            previously_generated_overlap = generated_frames_concat[-1][:, -overlaplen:, :].clone()
            generated_window_overlap = generated_window[:, :overlaplen, :].clone()
            generated_window_blended = linear_blend_motion(previously_generated_overlap, generated_window_overlap)
            generated_frames_concat[-1][:, -overlaplen:, :] = generated_window_blended
            generated_window_slice = generated_window[:, -frames_to_take:, :].clone()
            generated_window_concat=change_motion_position(generated_window_slice, offset=frame_offset)
            generated_frames_concat.append(generated_window_concat)
            
            last_concat_end_idx += frames_to_take

    # Concatenate all generated frames
    generated_motion = torch.cat(all_generated_frames, dim=1)

    generated_motion_concat = torch.cat(generated_frames_concat, dim=1)
    """
    # Trim to match the original total_frames if needed
    if generated_motion.shape[1] > total_frames:
        generated_motion = generated_motion[:, :total_frames, :]"""

    return generated_motion, generated_motion_concat

def transform_motion_back(args, betas, generated_motion_np, reference_np):
    if args.keypointtype=='6d':
        assert generated_motion_np.shape[1]==135
        betas_np=betas.squeeze(0).cpu().numpy()
        #generated_motion_np = gaussian_filter1d(generated_motion_np, sigma=1, axis=0)
        generated_motion_np=sixd_to_smplx({'motion_6d': generated_motion_np[:,3:], 'transl': generated_motion_np[:,:3], 'betas': betas_np})
        if reference_np is not None:
            reference_np=sixd_to_smplx({'motion_6d': reference_np[:,3:], 'transl': reference_np[:,:3], 'betas': betas_np})
    elif args.keypointtype=='openpose':
        assert generated_motion_np.shape[1]==69
        generated_motion_np=generated_motion_np.reshape(-1,23,3)
        if reference_np is not None:
            reference_np=reference_np.reshape(-1,23,3)
    return generated_motion_np, reference_np

def calculate_metrics(reference_np, generated_motion_np, keypointtype, index):
    if keypointtype=='openpose':
        assert generated_motion_np.shape[1]==69
        generated_motion_np=change_motion_position(generated_motion_np)
        reference_np=change_motion_position(reference_np)
        dtw_distance, path = fastdtw.fastdtw(reference_np, generated_motion_np)
        dtw_distance=dtw_distance/len(path)  # normalize by length of path
        res={'sample_id': index, 'dtw_distance': dtw_distance, 'reference_frames': reference_np.shape[0], 'generated_frames': generated_motion_np.shape[0]}
        return res
    elif keypointtype=='6d':
        assert generated_motion_np.shape[1]==135
        ref_sixd=reference_np[:,3:]
        gen_sixd=generated_motion_np[:,3:]
        dtw_distance, path = fastdtw.fastdtw(reference_np[:, :3], generated_motion_np[:, :3]) # translation part
        dtw_distance=dtw_distance/len(path)  # normalize by length of path
        dtw_distance_geodesic, path = fastdtw.fastdtw(ref_sixd, gen_sixd, dist=pose_distance_metric)
        dtw_distance_geodesic=dtw_distance_geodesic/len(path)  # normalize by length of path
        res={'sample_id': index, 'dtw_distance_geodesic':dtw_distance_geodesic, 'dtw_distance_transl': dtw_distance, 'reference_frames': reference_np.shape[0], 'generated_frames': generated_motion_np.shape[0]}
        return res
    elif keypointtype=='6d_transformed':
        assert generated_motion_np.shape[1]==22
        generated_motion_np=change_motion_position(generated_motion_np)
        reference_np=change_motion_position(reference_np)
        gen=generated_motion_np.reshape(-1,66)
        ref=reference_np.reshape(-1,66)
        dtw_distance, path = fastdtw.fastdtw(ref, gen)
        dtw_distance=dtw_distance/len(path)  # normalize by length of path
        return dtw_distance
    else:
        raise ValueError("Unknown keypoint type for DTW calculation.")

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
    use_sliding_window = False if args.input_motion_length >= 240 else True
    print(f"Using sliding window: {use_sliding_window}")


    #TODO change sample dataset
    file_path = "test_dataset"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Conditional motion file not found: {file_path}")

    # The model expects a batch, so add a batch dimension
    dataloader = prepare_conditional_motion(file_path, args.input_motion_length, args.keypointtype)
    #cond_motion = prepare_conditional_motion(file_path, args.input_motion_length).to(device)
    #cond_motion_turn = prepare_conditional_motion(file_path_turn, args.input_motion_length).to(device)
    print("Generating motion...")
    print(args.keypointtype)
        # Create an output directory if it doesn't exist
    if not args.output_dir:
        name, _ = os.path.splitext(os.path.basename(args.model_path))
        args.output_dir="results/"
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize list to store DTW metrics
    dtw_metrics = []
    #dtw_geodesic_metrics = []

    for i, batch in enumerate(tqdm(dataloader)):
        #6d case| keypoint case
        #referenceshape: (1, frames, 22, 3)| (1,frames,69)
        #condition shape: (1, frames, 135)| (1,frames,69)
        #betas shape: (1,frames, 10/11)|_
        reference, condition, betas = batch
        condition=condition.to(device)
        #print(reference.shape, condition.shape)  # shapes of the batch
        #print(f"Condition motion shape: {condition.shape}")
        ref_path=os.path.join(args.output_dir, "reference_motion_"+str(i)+".npy")
        gen_path=os.path.join(args.output_dir, "generated_motion_"+str(i)+".npy")
        res={}
        # --- Run the generation ---
        if use_sliding_window:
            generated_motion, generated_motion_concat = sample(model, diffusion, condition, args, use_sliding_window=use_sliding_window, sliding_window_step=10)
            generated_motion_np = generated_motion.squeeze(0).cpu().numpy()
            generated_motion_concat_np = generated_motion_concat.squeeze(0).cpu().numpy()
            reference_np=reference.squeeze(0).cpu().numpy()
            res=calculate_metrics(reference_np, generated_motion_concat_np, args.keypointtype, i)
            generated_motion_np, reference_np = transform_motion_back(args, betas, generated_motion_np, reference_np)
            generated_motion_concat_np, _ = transform_motion_back(args, betas, generated_motion_concat_np, None)

            gen_concat_path=os.path.join(args.output_dir, "generated_motion_concat_"+str(i)+".npy")
            np.save(gen_concat_path, generated_motion_concat_np)
        else:
            generated_motion = sample(model, diffusion, condition, args, use_sliding_window=use_sliding_window, sliding_window_step=10)

            generated_motion_np = generated_motion.squeeze(0).cpu().numpy()
            reference_np=reference.squeeze(0).cpu().numpy()
            res=calculate_metrics(reference_np, generated_motion_np, args.keypointtype, i)
            generated_motion_np, reference_np = transform_motion_back(args, betas, generated_motion_np, reference_np)
        
        if args.keypointtype=='6d':
            #calculate dtw only on for 6d after transformation
            res['dtw_distance']=calculate_metrics(reference_np, generated_motion_np, '6d_transformed', i) # normalize to be comparable with openpose
        dtw_metrics.append(res)

        np.save(ref_path, reference_np)
        np.save(gen_path, generated_motion_np)

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
