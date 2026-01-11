import os
import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.point_clouds import PointClouds
from plyfile import PlyData
from aitviewer.renderables.skeletons import Skeletons



class BODY25Skeletons(Skeletons):
    SKELETON = np.asarray([
        (-1, 0), (0, 1), (1, 8),(1,2), (1,5), (9, 10), (10, 11), (8, 9), (8, 12),
        (12, 13), (13, 14), (2, 3), (3, 4),
        (5, 6), (6, 7), (0, 15),
        (0, 16), (15, 17), (16, 18), (14, 19), (19, 20), (14, 21),
        (11, 22), (22, 23), (11, 24)
    ])

    def __init__(self, joints, **kwargs):
        kwargs.setdefault('color', (0.5, 0.0, 0.0, 1.0))
        super().__init__(joints, __class__.SKELETON, **kwargs)

class SMPLSkeletons(Skeletons):
    """
    Skeleton definition for the first 22 SMPL joints (Pelvis to Wrists).
    Format: (Parent, Child)
    """
    SKELETON = np.asarray([
        (-1, 0),  # Pelvis (Root)
        (0, 1),   # Pelvis -> L_Hip
        (0, 2),   # Pelvis -> R_Hip
        (0, 3),   # Pelvis -> Spine1 (Waist)
        (1, 4),   # L_Hip -> L_Knee
        (2, 5),   # R_Hip -> R_Knee
        (3, 6),   # Spine1 -> Spine2 (Back)
        (4, 7),   # L_Knee -> L_Ankle
        (5, 8),   # R_Knee -> R_Ankle
        (6, 9),   # Spine2 -> Spine3 (Chest)
        (7, 10),  # L_Ankle -> L_Foot
        (8, 11),  # R_Ankle -> R_Foot
        (9, 12),  # Spine3 -> Neck
        (9, 13),  # Spine3 -> L_Collar
        (9, 14),  # Spine3 -> R_Collar
        (12, 15), # Neck -> Head
        (13, 16), # L_Collar -> L_Shoulder
        (14, 17), # R_Collar -> R_Shoulder
        (16, 18), # L_Shoulder -> L_Elbow
        (17, 19), # R_Shoulder -> R_Elbow
        (18, 20), # L_Elbow -> L_Wrist
        (19, 21)  # R_Elbow -> R_Wrist
    ])

    def __init__(self, joints, **kwargs):
        kwargs.setdefault('color', (0.5, 0.0, 0.0, 1.0))
        super().__init__(joints, __class__.SKELETON, **kwargs)

def add_keypoints(data, viewer, thisname, color=(1.0, 0.0, 0.0, 1)):
    # Load keypoints
    keypoints = data
    keypoints=keypoints[..., :3]

    keypoints_pc = PointClouds(
        keypoints,
        name=thisname,
        point_size=3.0,
        color=color
    )

    if keypoints.shape[1]==25:
        skeleton=BODY25Skeletons(keypoints, name=thisname, color=color)
        viewer.scene.add(skeleton)
    elif keypoints.shape[1]==22:
        skeleton=SMPLSkeletons(keypoints, name=thisname, color=color)
        viewer.scene.add(skeleton)
    elif keypoints.shape[1]==23:
        keypoints=repair_data(keypoints)
        skeleton=BODY25Skeletons(keypoints, name=thisname, color=color)
        viewer.scene.add(skeleton)
    else:
        viewer.scene.add(keypoints_pc)
    return

def subtract_root(data):
    #only after frames have been cut
    if data.shape[1]==25:
        root = (data[0,8,:]+data[0, 9, :])/2
        data=np.delete((data - root), (1,8), axis=1)
    elif data.shape[1]==22:
        root = data[0,0,:]
        data=data - root
    return data

def repair_data(data):
    if(data.shape[1]==23):
        midhip = (data[:,7,:]+data[:, 10, :])/2
        neck= (data[:,1,:]+data[:, 4, :])/2
        data=np.insert(data, 7, midhip, axis=1)
        data=np.insert(data, 1, neck, axis=1)
    elif(data.shape[1]==25): 
        midhip = (data[:,8,:]+data[:, 9, :])/2
        neck= (data[:,2,:]+data[:, 5, :])/2
        data[:,8,:]=midhip
        data[:,1,:]=neck
    return data

    

if __name__ == "__main__":
    # Set the path to your SMPL models
    conditionpaths=["test_dataset/gait_700/20250820_c2_a3_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
                    "test_dataset/gait_700/20250820_c2_a5_Take3/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
                    "test_dataset/gait_753/20250617_c2_a1_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
                    "test_dataset/gait_753/20250617_c2_a3_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
                    "test_dataset/gait_809/20250819_c2_a2_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
                    "test_dataset/gait_809/20250819_c2_a2_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
                    "test_dataset/gait_809/20250819_c2_a4_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
                    "test_dataset/gait_809/20250819_c2_a4_Take2/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",
                    "test_dataset/gait_809/20250819_c2_a5_Take1/split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy",]
    modellist=["model000080002", 
               "model000070213", 
               "model000059813"
               ]
    C.smplx_models = "smpl_models/"
    # Set the playback speed to 30 frames per second.
    C.playback_fps = 30
    
    configlist=[
        "configsoftdtw",

    ]
    root="results/window60_concat"
    v=Viewer()
    reference=None
    i=5
    #conf="config16"
    for config in os.listdir(root):
        """
        if config not in configlist:
            continue"""
        config_path=os.path.join(root, config)
        for model in os.listdir(config_path):
            #if model in modellist:
            model_path=os.path.join(config_path, model)
            if reference is None:
                reference_path=os.path.join(model_path, "reference_motion_"+str(i)+".npy")
                reference=np.load(reference_path)
                reference=repair_data(reference)
                reference=subtract_root(reference)
                #print(reference.shape)
                condition=np.load(conditionpaths[i])
                condition=subtract_root(condition)
                add_keypoints(condition, v, "Condition Motion", color=(0.0, 1.0, 0.0, 1))
                add_keypoints(reference, v, "Reference Motion", color=(0.0, 0.0, 1.0, 1))
            condition_path=os.path.join(model_path, "generated_motion_"+str(i)+".npy")
            gen=np.load(condition_path)
            #gen=subtract_root(gen)
            # Create a viewer instance
            add_keypoints(gen, v, "Generated Motion_"+config+model[-5:], color=(0.5, 0.0, 0.0, 1))
            
    v.run()

    #visualize_gait(keypoints_path, reference_path, condition_path)