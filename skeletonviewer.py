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
        (-1, 0), (0, 1), (1, 8),(1,2), (1,5), (9, 10), (10, 11), (8, 9), (8, 12),
        (12, 13), (13, 14), (2, 3), (3, 4),
        (5, 6), (6, 7), (0, 15),
        (0, 16), (15, 17), (16, 18), (14, 19), (19, 20), (14, 21),
        (11, 22), (22, 23), (11, 24)
    ])

    def __init__(self, joints, **kwargs):
        kwargs['color'] = (0.5, 0.0, 0.0, 1.0)
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

    #skeleton=BODY25Skeletons(keypoints, name=thisname, color=color)
    #viewer.scene.add(skeleton)
    viewer.scene.add(keypoints_pc)
    return

def add_skeleton(data, viewer, thisname, color=(1.0, 0.0, 0.0, 1)): 
    # Load keypoints
    keypoints = data
    keypoints=keypoints[..., :3]

    skeleton=BODY25Skeletons(keypoints, name=thisname, color=color)
    viewer.scene.add(skeleton)
    return

def repair_data(data):
    midhip = (data[:,8,:]+data[:, 9, :])/2
    neck= (data[:,2,:]+data[:, 5, :])/2
    data[:,8,:]=midhip
    data[:,1,:]=neck
    return data
    

if __name__ == "__main__":
    # Set the path to your SMPL models
    C.smplx_models = "smpl_models/"
    keypoints=np.load("mydataset/gait_052/20250912_c1_a1_Take1\\split_subjects\\0\\fit-smplx\\new-smpl-keypoints_cut.npy")
    keypoints=np.load("mydataset/gait_052/20250912_c1_a1_Take1\\split_subjects\\0\\keypoints_3d\\smpl-keypoints-3d_cut.npy")
    #keypoints=repair_data(keypoints)
    
    v=Viewer()
    for i in range(keypoints.shape[1]):
        add_keypoints(keypoints[:,i:i+1,:], v, str(i))
    #add_skeleton(keypoints,v, "skeleton", color=(0.0, 1.0, 0.0, 1))
    v.run()
    #visualize_gait(keypoints_path, reference_path, condition_path)