import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from data_loaders.dataloader3d import load_data, MotionDataset
import torch
C.smplx_models = "smpl_models/"


def rotate_keypoints_y_180(keypoints_path: str):
    """
    Loads a 3D keypoints file, rotates the keypoints 180 degrees around the Y-axis,
    and saves the modified keypoints back to the original file.

    Args:
        keypoints_path (str): The full path to the .npy file containing the keypoints.
                             Expected shape: (frames, 135, 5), where the last 3 values
                             of the final dimension represent (x, y, z) coordinates.
    """
    if not os.path.exists(keypoints_path):
        print(f"Error: Keypoints file not found at '{keypoints_path}'")
        return

    print(f"Loading keypoints from: {keypoints_path}")
    base_path, ext = os.path.splitext(keypoints_path)
    

    keypoints = np.load(keypoints_path)
    # Define the 180-degree rotation matrix around the Y-axis
    rotation_matrix_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # Extract the XYZ coordinates (first 3 values of the last dimension)
    xyz_coords = keypoints[..., :3]
    
    # Apply the rotation to the XYZ coordinates
    rotated_xyz = xyz_coords @ rotation_matrix_y_180
    
    # Create the output array with the same shape as input
    rotated_keypoints = keypoints.copy()
    rotated_keypoints[..., :3] = rotated_xyz

    
    np.save(base_path, rotated_keypoints)
    print(f" -> Modified and saved to: {base_path}")

#cut frames
def cut_frames(gait,take, start, end):
    root=gait
    keypointspart="split_subjects/0/keypoints_3d/smpl-keypoints-3d.npy"
    #keypointspart="split_subjects/0/fit-smplx/new-smpl-keypoints.npy"
    newkeypoints_name="smpl-keypoints-3d_cut.npy"
    keypoints_path = os.path.join(root, take, keypointspart)
    keypoints = np.load(keypoints_path)
    cut_keypoints = keypoints[start:end]
    new_keypoints_path = os.path.join(root, take, "split_subjects/0/keypoints_3d", newkeypoints_name)
    np.save(new_keypoints_path, cut_keypoints)
    print(f"Cut keypoints saved to: {new_keypoints_path} with frames from {start} to {end}")


def cut_smplx_frames(root, take, start, end):
    """
    Loads an SMPL-X parameters file (npz/npy), cuts the data to the specified 
    frame range, and saves it to a new file.

    Args:
        root (str): The root directory (e.g., 'gait').
        take (str): The specific take or sequence folder.
        start (int): Start frame index.
        end (int): End frame index.
    """
    
    # --- Configuration of Paths ---
    # Adjust these filenames to match your specific folder structure
    relative_folder = "split_subjects/0/fit-smplx"
    input_filename = "smplx-params.npz" # Or whatever your source file is named
    output_filename = "smplx-params_cut.npz"

    input_path = os.path.join(root, take, relative_folder, input_filename)
    output_path = os.path.join(root, take, relative_folder, output_filename)

    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    print(f"Loading from: {input_path}")
    
    # 1. Load the data
    data = np.load(input_path, allow_pickle=True)
    
    # Convert NpzFile or 0-d array to a standard dictionary
    if isinstance(data, np.lib.npyio.NpzFile):
        params = dict(data)
    elif data.ndim == 0 and data.dtype == 'O':
        params = data.item()
    else:
        params = dict(data)

    # 2. Determine the original number of frames
    total_frames = params['transl'].shape[0]

    # 3. Slice the data
    cut_params = {}
    
    for key, value in params.items():
        # Only slice arrays that match the total frame count (dynamic data)
        cut_params[key] = value[start:end]
            
    # 4. Save the result
    # If the input was .npz, we save as .npz (compressed)
    if input_filename.endswith('.npz'):
        np.savez(output_path, **cut_params)
    else:
        # If it was .npy, we save as a pickled dictionary
        np.save(output_path, cut_params)

    print(f" -> Cut saved to: {output_path}")

def create_smpl_keypoints(path):
    #function to create smpl keypoints from smplx parameters
    data = np.load(path)
    
    # smplx parameters
    body_pose = data['body_pose']           # (419, 63)
    print(body_pose.shape)
    global_orient = data['global_orient']   # (419, 3)
    betas = data['betas']                   # (419, 11)
    transl = data['transl']                 # (419, 3)
    left_hand_pose = data['left_hand_pose'] # (419, 12) - PCA components
    right_hand_pose = data['right_hand_pose'] # (419, 12) - PCA components
    
    # Create SMPL-X layer with PCA hand pose support
    if betas.shape[1]==11:
        smpl_layer = SMPLLayer(
            model_type="smplx", 
            gender="neutral", 
            device=C.device,
            age="kid",
            kid_template_path="", #INSERT KID TEMPLATE PATH
            use_pca=True, 
            num_pca_comps=12,  
            flat_hand_mean=False
        )
        smpl_layer.num_betas += 1
    else:
        smpl_layer = SMPLLayer(
            model_type="smplx", 
            gender="neutral", 
            device=C.device,
            use_pca=True, 
            num_pca_comps=12,  
            flat_hand_mean=False
        )

    num_frames = body_pose.shape[0]
    all_joints = []
    
    # Convert numpy arrays to torch tensors
    body_pose_torch = torch.from_numpy(body_pose).float().to(C.device)
    global_orient_torch = torch.from_numpy(global_orient).float().to(C.device)
    betas_torch = torch.from_numpy(betas).float().to(C.device)
    transl_torch = torch.from_numpy(transl).float().to(C.device)
    left_hand_pose_torch = torch.from_numpy(left_hand_pose).float().to(C.device)
    right_hand_pose_torch = torch.from_numpy(right_hand_pose).float().to(C.device)

    for i in range(num_frames):
        output = smpl_layer(
            poses_body=body_pose_torch[i:i+1],
            poses_root=global_orient_torch[i:i+1],
            betas=betas_torch[i:i+1],
            trans=transl_torch[i:i+1],
            poses_left_hand=left_hand_pose_torch[i:i+1],
            poses_right_hand=right_hand_pose_torch[i:i+1]
        )
        # Unpack the output tuple - joints are the second element
        _, joints = output
        joints_np = joints.cpu().numpy()  # Shape: (1, num_joints, 3)
        all_joints.append(joints_np[0])
    
    all_joints = np.array(all_joints)  # Shape: (num_frames, num_joints, 3)
    all_joints=all_joints[:,:22,:]  #only smpl body joints
    return all_joints

import numpy as np

def get_transformation_matrix(v):
    """
    Creates a matrix that rotates v to the X-axis, mirrors Z, 
    and rotates back.
    """
    # 1. Define input and target
    v = np.array(v, dtype=float)
    target = np.array([1, 0, 0], dtype=float)
    
    # Normalize input
    norm_v = np.linalg.norm(v)
    if norm_v == 0: return np.eye(3)
    a = v / norm_v
    b = target

    # 2. Compute Rotation (Rodrigues' Formula)
    # Axis of rotation (k)
    k = np.cross(a, b)
    s = np.linalg.norm(k) # sin of angle
    c = np.dot(a, b)      # cos of angle

    if s == 0:
        # Vector is already on x-axis (parallel)
        if c > 0: return np.diag([1, 1, -1]) # Just mirror
        # If anti-parallel (-x), we usually just flip indices, 
        # but technically needs 180 rotation. Simplified here:
        return np.diag([1, 1, -1]) 

    # Skew-symmetric matrix K
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    # Rotation Matrix (aligns v -> x)
    # Note: We want R that takes v to x. 
    # The formula R = I + K + ... rotates a to b.
    R = np.eye(3) + K + (K @ K) * ((1 - c) / (s**2))

    # 3. Mirror Matrix (Mirror across XY plane means Z -> -Z)
    M = np.diag([1, 1, -1])

    # 4. Combine: Un-rotate * Mirror * Rotate
    # Note: If R takes v->x, then R.T takes x->v.
    # Order depends on if we view R as transforming the basis or the vector.
    # Standard: T_final = R.T @ M @ R
    return R.T @ M @ R
def mirror_keypoints(path, offset):
    data=np.load(path)
    root=data[0,2,:]+data[0, 5, :]/2
    dest=data[offset,2,:]+data[offset, 5, :]/2
    vec=dest - root
    vec=vec[:3]
    transform_matrix = get_transformation_matrix(vec)
    transformed_data = data[..., :3] @ transform_matrix.T
    data[..., :3] = transformed_data
    np.save(path, data)
    return

def mirror_smplx_gait(npz_path, gait_direction, output_path=None):
    """
    Mirrors SMPL-X gait motion based on a specified gait direction.
    
    Args:
        npz_path (str): Path to the input .npz file.
        gait_direction (list or np.array): A 2D or 3D vector [x, y, z] indicating 
                                           the forward direction of the gait. 
                                           (e.g., [1, 0, 1] for diagonal).
        output_path (str, optional): Path to save the result. If None, returns data dict.

    Returns:
        dict: The mirrored SMPL-X data dictionary.
    """
    
    # 1. Load Data
    data = np.load(npz_path)
    # Convert to mutable dict
    new_data = {k: v.copy() for k, v in data.items()}
    
    # Extract main parameters
    # shape: (N, 3)
    global_orient = data['global_orient'] 
    transl = data['transl']
    
    # SMPL-X Body Pose is usually (N, 63) -> 21 joints * 3
    body_pose = data['body_pose'].reshape(data['body_pose'].shape[0], -1, 3)
    
    N = global_orient.shape[0]

    # ---------------------------------------------------------
    # 2. Compute Alignment Rotation
    # ---------------------------------------------------------
    # We define the mirror plane relative to the gait direction.
    # We rotate the world so gait_direction aligns with Z (Forward).
    
    gait_vec = np.array(gait_direction, dtype=np.float32)
    gait_vec[1] = 0  # Project to ground plane (ignore vertical component)
    gait_vec = gait_vec / np.linalg.norm(gait_vec)
    
    canonical_forward = np.array([0, 0, 1], dtype=np.float32)
    
    # Calculate rotation from gait_dir to canonical_forward (Z)
    # We use cross product to find axis and dot product for angle
    rotation_axis = np.cross(gait_vec, canonical_forward)
    
    # Handle case where gait is already aligned or opposite
    norm_axis = np.linalg.norm(rotation_axis)
    if norm_axis < 1e-6:
        # Vectors are parallel
        if np.dot(gait_vec, canonical_forward) > 0:
            align_rot = R.from_matrix(np.eye(3)) # No rotation needed
        else:
            align_rot = R.from_euler('y', 180, degrees=True) # 180 turn
    else:
        # Calculate angle
        angle = np.arctan2(norm_axis, np.dot(gait_vec, canonical_forward))
        align_rot = R.from_rotvec(rotation_axis / norm_axis * angle)

    # Convert rotations to Scipy objects for batch processing
    r_root = R.from_rotvec(global_orient)
    
    # ---------------------------------------------------------
    # 3. Align to Canonical Space (Gait -> Z+)
    # ---------------------------------------------------------
    # Rotate root orientation and translation to align with Z-axis
    # For translation: simple vector rotation
    transl_aligned = align_rot.apply(transl)
    
    # For root orientation: R_aligned = R_align * R_original
    # Note: We apply align_rot to the global root orientation
    r_root_aligned = align_rot * r_root

    # ---------------------------------------------------------
    # 4. Perform Mirroring (Reflection across YZ plane)
    # ---------------------------------------------------------
    
    # A. Mirror Translation: Flip X coordinate
    transl_mirrored = transl_aligned.copy()
    transl_mirrored[:, 0] *= -1
    
    # B. Mirror Root Orientation
    # To mirror a rotation matrix across X-axis:
    # 1. Convert to axis-angle
    # 2. Flip signs of Y and Z components: [rx, ry, rz] -> [rx, -ry, -rz]
    # (This is the standard rule when reflecting across the sagittal plane)
    
    root_vecs = r_root_aligned.as_rotvec()
    root_vecs[:, 1] *= -1
    root_vecs[:, 2] *= -1
    r_root_mirrored = R.from_rotvec(root_vecs)

    # C. Mirror Body Poses (Swap L/R and Flip Signs)
    # Define SMPL-X Body Joint Mapping (21 joints)
    # Indices based on standard SMPL topology excluding root (which is 0)
    # 0: Pelvis (Center)
    # 1: L_Hip <-> 2: R_Hip
    # 3: Spine1 (Center)
    # 4: L_Knee <-> 5: R_Knee
    # 6: Spine2 (Center)
    # 7: L_Ankle <-> 8: R_Ankle
    # 9: Spine3 (Center)
    # 10: L_Foot <-> 11: R_Foot
    # 12: Neck (Center)
    # 13: L_Collar <-> 14: R_Collar
    # 15: Head (Center)
    # 16: L_Shoulder <-> 17: R_Shoulder
    # 18: L_Elbow <-> 19: R_Elbow
    # 20: L_Wrist <-> 21: R_Wrist
    
    # Map for the 21 joints in body_pose
    smplx_perm = np.arange(21)
    pairs = [
        (1, 2), (4, 5), (7, 8), (10, 11), 
        (13, 14), (16, 17), (18, 19), (20, 21)
    ]
    # Adjust for 0-indexing in body_pose array (which starts at joint 1 usually? 
    # No, usually body_pose is joints 1-21. So index 0 is joint 1.
    # Let's double check standard SMPL-X body_pose structure. 
    # Usually body_pose is 21 joints (excluding pelvis/root).
    # If so: 
    # 0,1 (Hips), 2 (Spine), 3,4 (Knees)...
    
    # Re-mapped indices assuming body_pose starts at Joint 1 (Pelvis is global_orient)
    # 0: L_Hip, 1: R_Hip
    # 2: Spine1
    # 3: L_Knee, 4: R_Knee
    # 5: Spine2
    # 6: L_Ankle, 7: R_Ankle
    # 8: Spine3
    # 9: L_Foot, 10: R_Foot
    # 11: Neck
    # 12: L_Collar, 13: R_Collar
    # 14: Head
    # 15: L_Shoulder, 16: R_Shoulder
    # 17: L_Elbow, 18: R_Elbow
    # 19: L_Wrist, 20: R_Wrist
    
    perm_idxs = np.arange(21)
    swap_pairs = [
        (0, 1), (3, 4), (6, 7), (9, 10), 
        (12, 13), (15, 16), (17, 18), (19, 20)
    ]
    
    for (l, r) in swap_pairs:
        perm_idxs[l] = r
        perm_idxs[r] = l
        
    # Apply swapping
    body_pose_mirrored = body_pose[:, perm_idxs, :]
    
    # Apply Axis-Angle Flip [x, -y, -z] to body pose
    body_pose_mirrored[:, :, 1] *= -1
    body_pose_mirrored[:, :, 2] *= -1
    
    # D. Mirror Hands (Swap Left parameter block with Right)
    if 'left_hand_pose' in data and 'right_hand_pose' in data:
        l_hand = data['left_hand_pose'].copy()
        r_hand = data['right_hand_pose'].copy()
        
        # Apply axis flips
        # Reshape to (N, 15, 3) for easier axis manipulation if needed, 
        # or just flat array manipulation
        l_hand = l_hand.reshape(N, -1, 3)
        r_hand = r_hand.reshape(N, -1, 3)
        
        l_hand[:, :, 1] *= -1
        l_hand[:, :, 2] *= -1
        r_hand[:, :, 1] *= -1
        r_hand[:, :, 2] *= -1
        
        new_data['left_hand_pose'] = r_hand.reshape(N, -1)
        new_data['right_hand_pose'] = l_hand.reshape(N, -1)

    # ---------------------------------------------------------
    # 5. De-Align (Rotate back to Global)
    # ---------------------------------------------------------
    # Apply inverse of the alignment rotation
    inv_align = align_rot.inv()
    
    # Transform translation back
    final_transl = inv_align.apply(transl_mirrored)
    
    # Transform root orientation back
    final_root_rot = inv_align * r_root_mirrored
    final_global_orient = final_root_rot.as_rotvec()

    # ---------------------------------------------------------
    # 6. Update and Save
    # ---------------------------------------------------------
    new_data['global_orient'] = final_global_orient
    new_data['transl'] = final_transl
    new_data['body_pose'] = body_pose_mirrored.reshape(N, -1)
    
    # Handle Jaw/Eye/Expression if needed (Optional but recommended)
    # Jaw is central (usually), just flip axis
    if 'jaw_pose' in new_data:
        jp = new_data['jaw_pose'].reshape(N, -1, 3)
        jp[:, :, 1] *= -1
        jp[:, :, 2] *= -1
        new_data['jaw_pose'] = jp.reshape(N, -1)
        
    if output_path:
        np.savez(output_path, **new_data)
        print(f"Mirrored gait saved to {output_path}")
    # Example Usage:
    # If the person walks diagonally at 45 degrees (x=1, z=1)
    # mirror_smplx_gait('my_motion.npz', gait_direction=[1, 0, 1], output_path='my_motion_mirrored.npz')
        
    return new_data

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_smplx_params_y_180(smplx_path: str):
    """
    Loads an SMPL-X parameters file, rotates the global orientation and translation
    180 degrees around the Y-axis, and saves the result back to the file.

    Args:
        smplx_path (str): Path to the .npz or .npy file containing the dictionary 
                          of SMPL-X parameters.
    """
    if not os.path.exists(smplx_path):
        print(f"Error: File not found at '{smplx_path}'")
        return

    print(f"Loading SMPL-X params from: {smplx_path}")
    
    # Load the data
    # allow_pickle is needed if the .npy contains a dictionary object
    data = np.load(smplx_path, allow_pickle=True)
    
    # Convert to a mutable dictionary based on file type
    if isinstance(data, np.lib.npyio.NpzFile):
        # If it's a .npz (zipped), convert to dict
        params = dict(data)
    elif data.ndim == 0 and data.dtype == 'O':
        # If it's a 0-d array wrapping a dict (common in some datasets)
        params = data.item()
    else:
        # If it's already a dict or structured array
        params = dict(data)

    # ---------------------------------------------------------
    # 1. Rotate Translation (transl)
    # ---------------------------------------------------------
    if 'transl' in params:
        # Transformation: (x, y, z) -> (-x, y, -z)
        # We multiply x and z by -1
        transl = params['transl'].copy()
        transl[:, 0] *= -1  # Invert X
        transl[:, 2] *= -1  # Invert Z
        params['transl'] = transl

    # ---------------------------------------------------------
    # 2. Rotate Global Orientation (global_orient)
    # ---------------------------------------------------------
    if 'global_orient' in params:
        global_orient = params['global_orient'] # Shape (N, 3) - Axis Angle
        
        # Convert current axis-angle to Rotation object
        rot_original = R.from_rotvec(global_orient)
        
        # Create a 180-degree rotation around Y-axis
        rot_180_y = R.from_euler('y', 180, degrees=True)
        
        # Apply the rotation: New = Rot180 * Old
        # (We left-multiply to apply the rotation to the global frame)
        rot_new = rot_180_y * rot_original
        
        # Convert back to axis-angle and ensure correct datatype
        params['global_orient'] = rot_new.as_rotvec().astype(global_orient.dtype)

    # ---------------------------------------------------------
    # Save Output
    # ---------------------------------------------------------
    base_path, ext = os.path.splitext(smplx_path)
    
    if ext == '.npz':
        np.savez(smplx_path, **params)
    else:
        np.save(smplx_path, params)
        
    print(f" -> Modified and saved to: {smplx_path}")

def find_best_matching_window(seq1, seq2, start_frame1, window_size):
    """
    Finds the starting frame in seq2 that minimizes the MSE with the window 
    defined by start_frame1 and window_size in seq1.

    Args:
        seq1 (np.ndarray): The first sequence (reference). Condition
        seq2 (np.ndarray): The second sequence (search). Ground truth
        start_frame1 (int): The starting frame index in seq1.
        window_size (int): The size of the window to compare.

    Returns:
        tuple: (best_start_frame2, min_mse)
            best_start_frame2 (int): The starting index in seq2 that gives the lowest MSE.
            min_mse (float): The calculated Mean Squared Error.
    """
    # Extract the target window from seq1
    target_window = seq1[start_frame1 : start_frame1 + window_size]
    
    # Check if the window is valid within seq1
    if target_window.shape[0] != window_size:
        print("seq1 shape:", seq1.shape)
        print("seq2 shape:", seq2.shape)
        raise ValueError(f"Window size {window_size} exceeds bounds of seq1 starting at {start_frame1}")

    best_mse = 5000000.0
    best_start_frame2 = -1
    
    # Number of valid starting positions in seq2
    num_valid_starts = seq2.shape[0] - window_size + 1
    
    if num_valid_starts <= 0:
        return 0, best_mse

    # Iterate through all possible windows in seq2
    for i in range(num_valid_starts):
        candidate_window = seq2[i : i + window_size]
        
        # Calculate Mean Squared Error
        mse=torch.mean((target_window - candidate_window) ** 2) 
        
        if mse < best_mse:
            best_mse = mse
            best_start_frame2 = i
            
    return best_start_frame2, best_mse


def window_matching(root, WINDOW_SIZE):
    a,b = load_data(root, split='train', keypointtype='6d')
    dataset = MotionDataset(
        "gait",
        a,
        b,
        input_motion_length=WINDOW_SIZE,
    )
    data_pairs = dataset.data_pairs
    
    match_dict={}
    for action, idx_clean, idx_no_orth in data_pairs:
        orth_motion=dataset.motion_clean[action][idx_clean][:,3:]
        no_orth_motion=dataset.motion_without_orth[action][idx_no_orth][:,3:]
        orth_seq_len = orth_motion.shape[0]
        no_orth_seq_len = no_orth_motion.shape[0]
        #c2 match with c1
        key=action+str(idx_no_orth)+str(idx_clean)
        print("Processing:",key)
        list_best_frames=[]
        #case when orth_seq_len > WINDOW_SIZE and no_orth_seq_len > WINDOW_SIZE
        if no_orth_seq_len > WINDOW_SIZE:
            for start_frame in range(0, no_orth_seq_len-WINDOW_SIZE+1):
                best_start, min_mse = find_best_matching_window(
                    no_orth_motion,
                    orth_motion,
                    start_frame,
                    WINDOW_SIZE
                )
                #print(f"Action: {action}, Start Frame Orth: {start_frame}, Best Start Frame No Orth: {best_start}, Min MSE: {min_mse}")
                list_best_frames.append((start_frame, best_start))
        else:
            best_start, min_mse = find_best_matching_window(
                no_orth_motion,
                orth_motion,
                0,
                no_orth_seq_len
            )
            list_best_frames.append((0, best_start))
        match_dict[key]=list_best_frames
    return match_dict

if __name__ == "__main__":
    #IMPORTANT: take out the takes mentioned in the preprocessing word document before running this file (data_preprocessing.docx on the last page)


    #parameters
    WINDOW_SIZE=[30,60]
    root="data/dataset"

    ### FRAME TRIMMING ###
    gaitfile=np.load("prepare_data/gaitlist.npy", allow_pickle=True).item()
    for gaitname in gaitfile:
        gaitdict=gaitfile[gaitname]
        print("cutting:",gaitname)
        gait=root+'/'+gaitname
        for take in os.listdir(gait):
            cur_take=take.split('_')[1:]
            key='_'.join(cur_take)
            key=key.replace('Take','t')
            print(key)
            if key in gaitdict:
                if gaitdict[key] is not None:
                    cut_smplx_frames(gait,take, gaitdict[key][0], gaitdict[key][1])
                    cut_frames(gait,take, gaitdict[key][0], gaitdict[key][1])

    ### MIRRORING and ROTATION ###

    # openpose keypoint rotation paths, rotate around the y-axis by 180 degrees
    pathlist=[
        "gait_753/20250617_c1_a2_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_753/20250617_c1_a4_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_753/20250617_c1_a5_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_753/20250617_c2_a2_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_753/20250617_c2_a3_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_753/20250617_c2_a3_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_753/20250617_c2_a3_Take3/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_753/20250617_c2_a4_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_809/20250819_c1_a2_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_809/20250819_c1_a2_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_809/20250819_c2_a2_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
        "gait_809/20250819_c2_a4_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy"
    ]

    for path in pathlist:
        path=root+'/'+path
        rotate_keypoints_y_180(path)

    #smpl-keypoints rotation
    pathlist=[
        "gait_753/20250617_c1_a2_Take1/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_753/20250617_c1_a4_Take2/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_753/20250617_c1_a5_Take1/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_753/20250617_c2_a2_Take1/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_753/20250617_c2_a3_Take1/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_753/20250617_c2_a3_Take2/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_753/20250617_c2_a3_Take3/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_753/20250617_c2_a4_Take2/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_809/20250819_c1_a2_Take1/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_809/20250819_c1_a2_Take2/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_809/20250819_c2_a2_Take2/split_subjects/0/fit-smplx/smplx-params_cut.npz",
        "gait_809/20250819_c2_a4_Take2/split_subjects/0/fit-smplx/smplx-params_cut.npz"
    ]

    for path in pathlist:
        path=root+'/'+path
        rotate_smplx_params_y_180(path)

    mirrorpathlist=[
        ("gait_753/20250617_c2_a3_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy", 75),
        ("gait_753/20250617_c2_a3_Take3/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy", 20),
        ("gait_766/20251001_c2_a3_Take3/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy", 44)
    ]

    for path, offset in mirrorpathlist:
        path=root+'/'+path
        mirror_keypoints(path, offset)

    mirrorpathlist=[
        ("gait_753/20250617_c2_a3_Take2/split_subjects/0/fit-smplx/smplx-params_cut.npz", 75),
        ("gait_753/20250617_c2_a3_Take3/split_subjects/0/fit-smplx/smplx-params_cut.npz", 20),
        ("gait_766/20251001_c2_a3_Take3/split_subjects/0/fit-smplx/smplx-params_cut.npz", 44)
    ]

    for path, offset in mirrorpathlist:
        path=root+'/'+path
        data=np.load(path)
        data=data['transl']
        print(data.shape)
        rootd=data[offset,:]+data[0, :]   
        mirror_smplx_gait(path, rootd, output_path=path)


    ### MATCHING for the context windows ###
    for w_size in WINDOW_SIZE:
        match_dict=window_matching(root, w_size)
        np.save(f"match_dict_{w_size}_final.npy", match_dict)
        

 
