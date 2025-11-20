import os
import numpy as np
import torch
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from utils import utils_transform


def smplx_to_6d(input_path, output_path=None):
    """
    Convert SMPL-X poses from axis-angle to 6D representation and save with translation and betas.
    returns a motions dictionary containing 6D poses, translations, and betas
    """
    data = np.load(input_path)
    # 1. Extract global orientation and body pose (first 21 joints for SMPL compatibility)
    global_orient = torch.from_numpy(data['global_orient']).float()  # (frames, 3)
    body_pose = torch.from_numpy(data['body_pose']).float()      # (frames, 63)
    transl = torch.from_numpy(data['transl']).float()            # (frames, 3)
    betas = torch.from_numpy(data['betas']).float() 
                 # (frames, 11)
    # discard 0 frames
    is_nonzero_row = (transl != 0).any(axis=1)
    ind = np.where(is_nonzero_row)[0]
    global_orient = global_orient[ind]
    body_pose = body_pose[ind]
    transl = transl[ind]
    betas = betas[ind]

    #root normalization
    transl = transl - transl[0:1, :]

    # 2. Concatenate to get full SMPL body pose (1 root + 21 joints = 22 joints)
    # Shape: (frames, 66)
    smpl_poses_aa = torch.cat([global_orient, body_pose], dim=1)

    # Reshape to (frames, 22, 3) to process each joint's rotation
    smpl_poses_aa_reshaped = smpl_poses_aa.reshape(smpl_poses_aa.shape[0], 22, 3)

    # 3. Convert from axis-angle to 6D representation
    smpl_poses_6d = utils_transform.aa2sixd(smpl_poses_aa_reshaped.reshape(-1, 3)).reshape(smpl_poses_aa.shape[0], 22, 6)

    # Reshape back to a flat vector per frame: (frames, 22 * 6) -> (frames, 132)
    smpl_poses_6d_flat = smpl_poses_6d.reshape(smpl_poses_6d.shape[0], -1)

    """
    # 4. Save the result
    output_filename = "motion_6d_with_transl.npz"
    output_path = os.path.join(output_path, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, motion_6d=smpl_poses_6d_flat.numpy(), transl=transl.numpy(), betas=betas.numpy())
    print(f"Successfully converted poses to 6D and saved with translation.")
    print(f"Original shape (axis-angle): {smpl_poses_aa.shape}")
    print(f"Output shape (6D): {smpl_poses_6d_flat.shape}")
    print(f"Saved result to: {output_path}")
    """
    motion={"motion_6d": smpl_poses_6d_flat.numpy(), "transl": transl.numpy(), "betas": betas.numpy()}
    return motion

def sixd_to_smplx(input_path, output_path, smplmodel_path, kid_template_path=None):
    #TODO: how to handle betas? Load 
    C.smplx_models = "smpl_models/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load the 6D rotation and translation data from the .npz file
    data = np.load(input_path)
    motion_6d_flat = torch.from_numpy(data['motion_6d']).float().to(device)
    transl = torch.from_numpy(data['transl']).float().to(device)
    num_frames = motion_6d_flat.shape[0]
    betas = torch.from_numpy(data['betas']).float().to(device) if 'betas' in data else torch.zeros(num_frames, 11, device=device)

    # 2. Convert 6D rotations back to axis-angle
    # Reshape from (frames, 132) to (frames * 22, 6) for batch conversion
    motion_aa = utils_transform.sixd2aa(motion_6d_flat.reshape(-1, 6))
    # Reshape back to (frames, 66)
    motion_aa_flat = motion_aa.reshape(num_frames, -1)

    # 3. Prepare parameters for the body model
    body_params = {
        'root_orient': motion_aa_flat[:, :3],    # First 3 values are the root rotation
        'pose_body': motion_aa_flat[:, 3:66],  # The rest are the body joint rotations
        'trans': transl, # Use the loaded translation
        'betas': betas
    }
    # 4. Instantiate body model and perform forward kinematics
    print("Performing forward kinematics to calculate joint positions...")
    if kid_template_path is None:
        print("Using generic SMPL-X model for forward kinematics.")
        smpl_layer = SMPLLayer(
            model_type="smplx", 
            gender="neutral", 
            device=C.device,
            num_betas=11,
            use_pca=True, 
            num_pca_comps=12,  
            flat_hand_mean=False
        )
    else:
        print("Using kid template for SMPL-X model forward kinematics.")
        smpl_layer = SMPLLayer(
            model_type="smplx", 
            gender="neutral", 
            device=C.device,
            age="kid",
            kid_template_path=r"C:\Users\Rui\Vorlesungskript\Master\Thesis\test\smpl_models\smplx\smplx_kid_template.npy",
            use_pca=True, 
            num_pca_comps=12,  
            flat_hand_mean=False
        )
        smpl_layer.num_betas += 1
    num_frames = motion_aa_flat.shape[0]
    all_joints = []
    
    print(f"Extracting joints for {num_frames} frames...")
    
    # Convert numpy arrays to torch tensors
    body_pose_torch = motion_aa_flat[:, 3:66].float().to(C.device)
    global_orient_torch = motion_aa_flat[:, :3].float().to(C.device)
    betas_torch = betas.float().to(C.device)
    transl_torch = transl.float().to(C.device)

    for i in range(num_frames):
        # Forward pass through SMPL layer to get joints
        # The layer returns a tuple: (vertices, joints)
        output = smpl_layer(
            poses_body=body_pose_torch[i:i+1],
            poses_root=global_orient_torch[i:i+1],
            betas=betas_torch[i:i+1],
            trans=transl_torch[i:i+1]
        )
        # Unpack the output tuple - joints are the second element
        vertices, joints = output
        joints_np = joints.cpu().numpy()  # Shape: (1, num_joints, 3)
        all_joints.append(joints_np[0])
    
    all_joints = np.array(all_joints)  # Shape: (num_frames, num_joints, 3)
    all_joints=all_joints[:,:22,:]
    # Extract the 3D joint positions for the 22 SMPL joints
    keypoints_3d = all_joints

    # 5. Save the reconstructed keypoints
    output_path = os.path.join(output_path, "reconstructed_keypoints_3d_ref.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, keypoints_3d)

    print(f"\nSuccessfully reconstructed 3D keypoints.")
    print(f"Output shape: {keypoints_3d.shape}")
    print(f"Saved result to: {output_path}")