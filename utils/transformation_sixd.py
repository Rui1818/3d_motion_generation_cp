import argparse
import os
import numpy as np
import torch
import utils.utils_transform as utils_transform
from utils.body_model import BodyModel as BM



class BodyModel(torch.nn.Module):
    """
    Wrapper for the human_body_prior BodyModel to make it a simple Module.
    """
    def __init__(self, support_dir, device, kid_template_path=None):
        super().__init__()
        # Use a generic male model, as is common in this project
        bm_fname = os.path.join(support_dir, "smplx/SMPLX_MALE.npz")
        if not os.path.exists(bm_fname):
            raise FileNotFoundError(f"SMPL-X model not found at: {bm_fname}")

        num_betas = 11
        self.v_template = None
        if kid_template_path:
            self.v_template = torch.from_numpy(np.load(kid_template_path)).float().to(device)
            num_betas = 10 # Kid template uses 10 betas

        body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
        ).to(device)
        self.body_model = body_model.eval()

    def forward(self, body_params):
        with torch.no_grad():
            # Pass the v_template during the forward call if it exists
            if self.v_template is not None:
                body_params['v_template'] = self.v_template
            return self.body_model(**body_params)

def smplx_to_6d(input_path, output_path):
    data = np.load(input_path)
    # 1. Extract global orientation and body pose (first 21 joints for SMPL compatibility)
    global_orient = torch.from_numpy(data['global_orient']).float()  # (frames, 3)
    body_pose = torch.from_numpy(data['body_pose']).float()      # (frames, 63)
    transl = torch.from_numpy(data['transl']).float()            # (frames, 3)
    betas = torch.from_numpy(data['betas']).float()              # (frames, 11)

    # 2. Concatenate to get full SMPL body pose (1 root + 21 joints = 22 joints)
    # Shape: (frames, 66)
    smpl_poses_aa = torch.cat([global_orient, body_pose], dim=1)

    # Reshape to (frames, 22, 3) to process each joint's rotation
    smpl_poses_aa_reshaped = smpl_poses_aa.reshape(smpl_poses_aa.shape[0], 22, 3)

    # 3. Convert from axis-angle to 6D representation
    smpl_poses_6d = utils_transform.aa2sixd(smpl_poses_aa_reshaped.reshape(-1, 3)).reshape(smpl_poses_aa.shape[0], 22, 6)

    # Reshape back to a flat vector per frame: (frames, 22 * 6) -> (frames, 132)
    smpl_poses_6d_flat = smpl_poses_6d.reshape(smpl_poses_6d.shape[0], -1)

    # 4. Save the result
    output_filename = "motion_6d_with_transl.npz"
    output_path = os.path.join(output_path, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, motion_6d=smpl_poses_6d_flat.numpy(), transl=transl.numpy(), betas=betas.numpy())
    print(f"Successfully converted poses to 6D and saved with translation.")
    print(f"Original shape (axis-angle): {smpl_poses_aa.shape}")
    print(f"Output shape (6D): {smpl_poses_6d_flat.shape}")
    print(f"Saved result to: {output_path}")

def sixd_to_smplx(input_path, output_path, smplmodel_path, kid_template_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load the 6D rotation and translation data from the .npz file
    data = np.load(input_path)
    motion_6d_flat = torch.from_numpy(data['motion_6d']).float().to(device)
    transl = torch.from_numpy(data['transl']).float().to(device)
    betas = torch.from_numpy(data['betas']).float().to(device)
    num_frames = motion_6d_flat.shape[0]

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
        'betas': betas[:, :10] if kid_template_path else betas
    }
    # 4. Instantiate body model and perform forward kinematics
    print("Performing forward kinematics to calculate joint positions...")
    body_model = BodyModel(smplmodel_path, device=device, kid_template_path=kid_template_path)
    body_pose = body_model(body_params)

    # Extract the 3D joint positions for the 22 SMPL joints
    keypoints_3d = body_pose.Jtr[:, :22, :].cpu().numpy()

    # 5. Save the reconstructed keypoints
    output_path = os.path.join(output_path, "reconstructed_keypoints_3d.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, keypoints_3d)

    print(f"\nSuccessfully reconstructed 3D keypoints.")
    print(f"Output shape: {keypoints_3d.shape}")
    print(f"Saved result to: {output_path}")