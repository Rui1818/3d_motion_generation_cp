import os
import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.point_clouds import PointClouds
from plyfile import PlyData
from aitviewer.renderables.skeletons import Skeletons

import torch



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
    data=np.delete((data - root), (1,8), axis=1)
    return data

def drop_duplicate_frames(data):
    first_row = data[:, 0:1, :]  # Shape: (frames, 1, 5)
    all_rows_same = np.all(data == first_row, axis=(1,2))

    mask = ~all_rows_same
    return data[mask]

def add_keypoints(path, viewer, thisname, color=(1.0, 0.0, 0.0, 1)):
    # Load keypoints
    keypoints = np.load(path)

    if keypoints.shape[-1] > 5:
        keypoints = keypoints.reshape(-1, 23, 3)
    elif keypoints.shape[-1] == 5:
        #keypoints=drop_duplicate_frames(keypoints)
        keypoints = keypoints[..., :3]
        keypoints=subtract_root(keypoints)

    

    keypoints_pc = PointClouds(
        keypoints,
        name=thisname,
        point_size=3.0,
        color=color
    )
    #if keypoints.shape[1]==24:
    #    keypoints=np.insert(keypoints, 1,0, axis=1)

    #skeleton=BODY25Skeletons(keypoints, name=thisname, color=color)
    #viewer.scene.add(skeleton)
    viewer.scene.add(keypoints_pc)
    return


def visualize_gait(keypoints_path, reference_path=None, condition_path=None, smplseq_path=None, smplseq_reference_path=None):
    v = Viewer()
    add_keypoints(keypoints_path, v, "my Keypoints")
    # Load floor point cloud from PLY file
    ply_data = PlyData.read("floor_c1_a3.ply")
    vertices = ply_data['vertex']
    floor_points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Create point cloud (add a frame dimension if needed)
    floor_pc = PointClouds(floor_points[np.newaxis, :, :], name="MyFloor", point_size=2.0)
    v.scene.add(floor_pc)

    if reference_path is not None:
        add_keypoints(reference_path, v, "reference Keypoints", color=(0.0, 0.0, 1.0, 1))

    if condition_path is not None:
        add_keypoints(condition_path, v, "condition Keypoints", color=(0.0, 1.0, 0.0, 1))

    if smplseq_path is not None:
        load_smpl_sequence(smplseq_path, v, name="c1 seq")
    if smplseq_reference_path is not None:
        load_smpl_sequence(smplseq_reference_path, v, name="c2 seq")

    v.run()

def load_smpl_sequence(smplseq_path, v, name="SMPL-X Sequence"):
    data = np.load(smplseq_path)
    
    # smplx parameters
    body_pose = data['body_pose']           # (419, 63)
    #print(body_pose.shape)
    global_orient = data['global_orient']   # (419, 3)
    betas = data['betas']                   # (419, 11)
    transl = data['transl']                 # (419, 3)
    left_hand_pose = data['left_hand_pose'] # (419, 12) - PCA components
    right_hand_pose = data['right_hand_pose'] # (419, 12) - PCA components
        
    #print(f"Loaded sequence with {body_pose.shape[0]} frames")
        
        # Create SMPL-X layer with PCA hand pose support
    if betas.shape[1]==11:
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
    else:
        smpl_layer = SMPLLayer(
                model_type="smplx", 
                gender="neutral", 
                device=C.device,
                use_pca=True, 
                num_pca_comps=12,  
                flat_hand_mean=False
        )
        
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
            name=name,
        )
    v.scene.add(smpl_sequence)
    return

def visualize_smpl_keypoints(smplkeypoints_path):
    data = np.load(smplkeypoints_path)
    
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
    print("num_betas:", smpl_layer.num_betas)
    smpl_layer.num_betas += 1

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

    num_frames = body_pose.shape[0]
    all_joints = []
    
    print(f"Extracting joints for {num_frames} frames...")
    
    # Convert numpy arrays to torch tensors
    body_pose_torch = torch.from_numpy(body_pose).float().to(C.device)
    global_orient_torch = torch.from_numpy(global_orient).float().to(C.device)
    betas_torch = torch.from_numpy(betas).float().to(C.device)
    transl_torch = torch.from_numpy(transl).float().to(C.device)
    left_hand_pose_torch = torch.from_numpy(left_hand_pose).float().to(C.device)
    right_hand_pose_torch = torch.from_numpy(right_hand_pose).float().to(C.device)

    for i in range(num_frames):
        # Forward pass through SMPL layer to get joints
        # The layer returns a tuple: (vertices, joints)
        output = smpl_layer(
            poses_body=body_pose_torch[i:i+1],
            poses_root=global_orient_torch[i:i+1],
            betas=betas_torch[i:i+1],
            trans=transl_torch[i:i+1],
            poses_left_hand=left_hand_pose_torch[i:i+1],
            poses_right_hand=right_hand_pose_torch[i:i+1]
        )
        # Unpack the output tuple - joints are the second element
        vertices, joints = output
        joints_np = joints.cpu().numpy()  # Shape: (1, num_joints, 3)
        all_joints.append(joints_np[0])
    
    all_joints = np.array(all_joints)  # Shape: (num_frames, num_joints, 3)
    right=all_joints[:,43,:]
    left=all_joints[:,29,:]
    all_joints=all_joints[:,:22,:]
    all_joints=np.concatenate((all_joints,right[:,np.newaxis,:]),axis=1)
    all_joints=np.concatenate((all_joints,left[:,np.newaxis,:]),axis=1)
    
    #indx 43
    print(f"Extracted joints shape: {all_joints.shape}")
    
    # Create point cloud from SMPL joints
    smpl_joints_pc = PointClouds(
        all_joints,
        name="SMPL Joints",
        point_size=10.0,
        color=(0.0, 0.0, 0.0, 1.0)  # Black color
    )
    v=Viewer()
    v.scene.add(smpl_sequence)
    v.scene.add(smpl_joints_pc)
    v.run()

def visualiza_gait_batch(root):
    v=Viewer()
    """
    ply_data = PlyData.read("floor_c1_a3.ply")
    vertices = ply_data['vertex']
    floor_points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Create point cloud (add a frame dimension if needed)
    floor_pc = PointClouds(floor_points[np.newaxis, :, :], name="MyFloor", point_size=2.0)
    v.scene.add(floor_pc)"""
    for take in os.listdir(root):
        cond=take.split("_")
        
        if cond[2]!="a3":
            continue
        if cond[1]=="c1":
            c2 = cond[0]+'_c2_'+"_".join(cond[2:])
            keypointspart="split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy"
            #keypointspart="split_subjects/0/fit-smplx/smpl-keypoints-3d_cut.npy"
            smplseqpart="split_subjects/0/fit-smplx/smplx-params_cut.npz"
            print(take)
            print(c2)
            keypoints_path = os.path.join(root, take, keypointspart)
            keypoints_path2 = os.path.join(root, c2, keypointspart)
            smplseq_path= os.path.join(root, take, smplseqpart)
            smplseq_reference_path= os.path.join(root, c2, smplseqpart)
            add_keypoints(keypoints_path, v, take)
            add_keypoints(keypoints_path2, v, c2, color=(0.0, 0.0, 1.0, 1))
            #load_smpl_sequence(smplseq_path, v, name=take)
            #load_smpl_sequence(smplseq_reference_path, v, name=c2)
    
    v.run()
    return

if __name__ == "__main__":
    # Set the path to your SMPL models
    C.smplx_models = "smpl_models/"

    root="mydataset"
    #root="test_dataset"
    smplpart="split_subjects/0/fit-smplx/smplx-params.npz"
    keypointspart="split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy"
    fin_take="_a4_Take2"
    take="gait_753/20250617_c1"+fin_take
    ref_take="gait_753/20250617_c2"+fin_take
    keypoints_path = os.path.join(root, take, keypointspart)
    keypoints_path2 = os.path.join(root, ref_take, keypointspart)
    condition_path = os.path.join(root, "gait_753/20250617_c2_a4_Take2", "split_subjects/0/fit-smplx/smpl-keypoints-3d_cut.npy")
    smplseq= os.path.join(root, take, smplpart)
    smplseq2= os.path.join(root, ref_take, smplpart)
    #visualize_gait(keypoints_path, reference_path=keypoints_path2, condition_path=condition_path, smplseq_path=None, smplseq_reference_path=None)
    #visualize_smpl_keypoints(smplseq)
    visualiza_gait_batch(root+"/gait_766")
    #visualize_gait('mydataset/gait_682/20250919_c1_a3_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy', 'mydataset/gait_682/20250919_c2_a3_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy')

