import os
import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.point_clouds import PointClouds
from plyfile import PlyData
from aitviewer.renderables.skeletons import Skeletons
import argparse



class BODY25Skeletons(Skeletons):
    SKELETON = np.asarray([
        (-1, 0), (0, 8), (9, 10), (10, 11), (8, 9), (8, 12),
        (12, 13), (13, 14), (0, 2), (2, 3), (3, 4), (2, 17),
        (0, 5), (5, 6), (6, 7), (5, 18), (0, 15),
        (0, 16), (15, 17), (16, 18), (14, 19), (19, 20), (14, 21),
        (11, 22), (22, 23), (11, 24)
    ])

    def __init__(self, joints, **kwargs):
        kwargs['color'] = (0.5, 0.0, 0.0, 1.0)
        super().__init__(joints, __class__.SKELETON, **kwargs)

def subtract_root(data):
    #only after frames have been cut
    root = (data[0,8,:]+data[0, 9, :])/2
    return data - root

def drop_duplicate_frames(data):
    first_row = data[:, 0:1, :]  # Shape: (frames, 1, 5)
    all_rows_same = np.all(data == first_row, axis=(1,2))

    mask = ~all_rows_same
    return data[mask]

def add_keypoints(path, viewer, thisname, color=(1.0, 0.0, 0.0, 1)):
    # Load keypoints
    keypoints = np.load(path)

    if keypoints.shape[-1] > 5:
        keypoints = keypoints.reshape(-1, 24, 3)
    elif keypoints.shape[-1] == 5:
        keypoints=drop_duplicate_frames(keypoints)
        keypoints = keypoints[..., :3]
        keypoints=subtract_root(keypoints)

    

    keypoints_pc = PointClouds(
        keypoints,
        name=thisname,
        point_size=3.0,
        color=color
    )
    if keypoints.shape[1]==24:
        keypoints=np.insert(keypoints, 1,0, axis=1)

    skeleton=BODY25Skeletons(keypoints, name=thisname)
    viewer.scene.add(skeleton)
    #viewer.scene.add(keypoints_pc)
    return


def visualize_gait(keypoints_path, reference_path=None, smplseq_path=None):
    v = Viewer()
    add_keypoints(keypoints_path, v, "my Keypoints")

    if reference_path is not None:
        add_keypoints(reference_path, v, "reference Keypoints", color=(0.0, 1.0, 0.0, 1))

    if smplseq_path is not None:
        data = np.load(smplseq_path)
    
        # smplx parameters
        body_pose = data['body_pose']           # (419, 63)
        #print(body_pose.shape)
        global_orient = data['global_orient']   # (419, 3)
        betas = data['betas']                   # (419, 11)
        transl = data['transl']                 # (419, 3)
        left_hand_pose = data['left_hand_pose'] # (419, 12) - PCA components
        right_hand_pose = data['right_hand_pose'] # (419, 12) - PCA components
        
        print(f"Loaded sequence with {body_pose.shape[0]} frames")
        
        # Create SMPL-X layer with PCA hand pose support
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
        
        # Create SMPL-X sequence
        # When use_pca=True, the layer expects 12-dimensional hand poses
        smpl_sequence = SMPLSequence(
            poses_body=body_pose,
            poses_root=global_orient,
            betas=betas,
            trans=transl,
            poses_left_hand=left_hand_pose,
            poses_right_hand=right_hand_pose,
            smpl_layer=smpl_layer,
            name="SMPL-X Sequence",
        )
        v.scene.add(smpl_sequence)

    v.run()
    

if __name__ == "__main__":
    # Set the path to your SMPL models
    C.smplx_models = "smpl_models/"
    """
    parser = argparse.ArgumentParser(description="My command-line program")

    # Add arguments
    parser.add_argument("--keypointspath", type=str, help="Path to the keypoints file")
    parser.add_argument("--smplseqpath", type=str, default=None, help="Path to the SMPL sequence file (optional)")
    """
    take="_c1_a1"
    root="overfit_training_sample"
    keypoints_path = "generated_motion_c1_a1.npy"
    keypoints_path = os.path.join(root, keypoints_path)
    reference_path = os.path.join("observations", "753", "vitpose"+take, "vitpose", "keypoints_3d", "smpl-keypoints-3d_cut.npy")


    visualize_gait(keypoints_path, reference_path)