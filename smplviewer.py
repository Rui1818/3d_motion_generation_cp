import glob
import os

import numpy as np
from PIL import Image
from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.point_clouds import PointClouds
from plyfile import PlyData
from aitviewer.renderables.skeletons import Skeletons

import torch



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
    print(f"Loaded keypoints from {path} with shape {keypoints.shape}")

    if keypoints.shape[-1] > 5:
        keypoints = keypoints.reshape(-1, 23, 3)
    elif keypoints.shape[-1] == 5:
        #keypoints=drop_duplicate_frames(keypoints)
        keypoints = keypoints[..., :3]
        #keypoints=subtract_root(keypoints)

    

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

def visualize_gait_batch(root):
    v=Viewer()
    
    ply_data = PlyData.read("floor_c1_a3.ply")
    vertices = ply_data['vertex']
    floor_points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Create point cloud (add a frame dimension if needed)
    floor_pc = PointClouds(floor_points[np.newaxis, :, :], name="MyFloor", point_size=2.0)
    v.scene.add(floor_pc)
    for take in os.listdir(root):
        cond=take.split("_")
        
        if cond[2]!="a5":
            continue
        if cond[1]=="c1":
            c2 = cond[0]+'_c2_'+"_".join(cond[2:])
            keypointspart="split_subjects/0/keypoints_3d/smpl-keypoints-3d_cut.npy"
            #keypointspart="split_subjects/0/fit-smplx/smpl-keypoints-3d_cut.npy"
            smplseqpart="split_subjects/0/fit-smplx/smplx-params_cut.npz"
            #print(take)
            #print(c2)
            keypoints_path = os.path.join(root, take, keypointspart)
            keypoints_path2 = os.path.join(root, c2, keypointspart)
            smplseq_path= os.path.join(root, take, smplseqpart)
            smplseq_reference_path= os.path.join(root, c2, smplseqpart)
            add_keypoints(keypoints_path, v, take)
            add_keypoints(keypoints_path2, v, c2, color=(0.0, 0.0, 1.0, 1))
            load_smpl_sequence(smplseq_path, v, name=take)
            load_smpl_sequence(smplseq_reference_path, v, name=c2)
    
    v.run()
    return

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

def subtract_root(data):
    #only after frames have been cut
    if data.shape[1]==25:
        root = (data[0,8,:]+data[0, 9, :])/2
        data=np.delete((data - root), (1,8), axis=1)
    elif data.shape[1]==22:
        root = data[0,0,:]
        data=data - root
    return data

def add_keypoints_result(data, viewer, thisname, color=(1.0, 0.0, 0.0, 1)):
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
        keypoints=subtract_root(keypoints)
        keypoints=repair_data(keypoints)
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


_DEFAULT_COLORS = [
    (0.8, 0.1, 0.1, 1.0),  # red
    (0.1, 0.1, 0.8, 1.0),  # blue
    (0.1, 0.7, 0.1, 1.0),  # green
    (0.8, 0.5, 0.0, 1.0),  # orange
    (0.6, 0.0, 0.8, 1.0),  # purple
    (0.0, 0.7, 0.7, 1.0),  # cyan
]


def capture_motion_frames(
    output_dir,
    smplseq_path=None,
    keypoints_path=None,
    frame_indices=None,
    n_frames=None,
    size=(1920, 1080),
    prefix="frame",
    transparent=False,
    colors=None,
    viewer=None,
):
    """
    Render individual motion frames to PNG images using the headless renderer.

    :param output_dir: Directory where the PNG images are saved.
    :param smplseq_path: Path (or list of paths) to .npz files with SMPL-X parameters (optional).
    :param keypoints_path: Path (or list of paths) to .npy keypoints files (optional).
      Pass a list to render multiple skeletons in the same image.
    :param frame_indices: List of specific frame indices to capture. If None, uses n_frames
      evenly spaced frames (or all frames if n_frames is also None).
    :param n_frames: Number of evenly spaced frames to capture. Ignored when frame_indices is given.
    :param size: Render resolution as (width, height).
    :param prefix: Filename prefix for saved images.
    :param transparent: Render with transparent background (required for overlay_motion_frames).
    :param colors: List of (R, G, B, A) colors, one per skeleton. Cycles through _DEFAULT_COLORS
      when None.
    """
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    os.makedirs(output_dir, exist_ok=True)

    # Normalise single paths → lists so the rest of the code is uniform
    if smplseq_path is None:
        smplseq_paths = []
    elif isinstance(smplseq_path, (str, os.PathLike)):
        smplseq_paths = [smplseq_path]
    else:
        smplseq_paths = list(smplseq_path)

    if keypoints_path is None:
        keypoints_paths = []
    elif isinstance(keypoints_path, (str, os.PathLike)):
        keypoints_paths = [keypoints_path]
    else:
        keypoints_paths = list(keypoints_path)

    total = len(smplseq_paths) + len(keypoints_paths)
    if colors is None:
        colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(total)]

    if viewer is None:
        v = HeadlessRenderer(size=size)
        C.smplx_models = "smpl_models/"
        v._default_nodes = list(v.scene.nodes)
    else:
        v = viewer
        default_nodes = getattr(v, '_default_nodes', [])
        for node in list(v.scene.nodes):
            if node not in default_nodes:
                v.scene.remove(node)

    for i, path in enumerate(smplseq_paths):
        load_smpl_sequence(path, v, name=f"SMPL_{i}")

    for i, path in enumerate(keypoints_paths):
        color = colors[len(smplseq_paths) + i]
        add_keypoints_result(np.load(path), v, f"Keypoints_{i}", color=color)

    v._init_scene()
    v.scene.camera.load_cam()
    total_frames = v.scene.n_frames

    if frame_indices is None:
        if n_frames is None:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(round(i)) for i in np.linspace(0, total_frames - 1, n_frames)]

    print(f"Capturing {len(frame_indices)} frames to '{output_dir}' ...")
    for idx in frame_indices:
        v.scene.current_frame_id = idx
        out_path = os.path.join(output_dir, f"{prefix}_{idx:04d}.png")
        v.export_frame(out_path, transparent_background=transparent)
        print(f"  Saved {out_path}")
    print("Done.")

def overlay_motion_frames(
    frame_dir,
    output_path,
    prefix="frame",
    bg_color=(255, 255, 255),
    colormap=None,
    alpha=0.7,
):
    """
    Overlay all captured frames onto a single image to show the full motion trail.

    Frames must have been captured with transparent=True so the background is
    transparent and only the skeleton/mesh is visible.

    :param frame_dir: Directory containing the transparent PNG frames.
    :param output_path: Output file path (e.g. "motion_overlay.png").
    :param prefix: Filename prefix used during capture.
    :param bg_color: Background color as (R, G, B).
    :param colormap: Optional matplotlib colormap name (e.g. "cool", "plasma", "viridis")
      to tint frames from start (cold) to end (warm), making temporal order visible.
      Requires matplotlib. Set to None to use original colors.
    :param alpha: Opacity of each frame layer (0.0 fully transparent, 1.0 opaque).
      Lower values let multiple overlapping poses show through each other.
    """
    paths = sorted(glob.glob(os.path.join(frame_dir, f"{prefix}_*.png")))
    if not paths:
        raise FileNotFoundError(f"No frames matching '{prefix}_*.png' found in '{frame_dir}'")

    base = Image.open(paths[0]).convert("RGBA")
    canvas = Image.new("RGBA", base.size, (*bg_color, 255))

    if colormap is not None:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)

    n = len(paths)
    frames = [np.array(Image.open(p).convert("RGB"), dtype=np.float32) for p in paths]

    # Estimate background as the median across all frames.
    # Pixels occupied by the (moving) skeleton will vary; static background pixels stay constant.
    bg = np.median(np.stack(frames, axis=0), axis=0)  # H x W x 3

    # Start canvas from the clean background
    canvas = bg.copy()

    for i, frame in enumerate(frames):
        # Find foreground pixels: those that differ from the background
        diff = np.max(np.abs(frame - bg), axis=-1)  # H x W
        fg_mask = diff > 10  # threshold in pixel intensity (0-255)

        if not fg_mask.any():
            continue

        fg = frame.copy()
        if colormap is not None:
            r, g, b, _ = cmap(i / max(n - 1, 1))
            tint = np.array([r * 255, g * 255, b * 255], dtype=np.float32)
            fg[..., :3] = fg[..., :3] * 0.5 + tint * 0.5

        # Alpha-blend foreground pixels into the canvas
        canvas[fg_mask] = canvas[fg_mask] * (1 - alpha) + fg[fg_mask] * alpha

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB").save(output_path)
    print(f"Saved overlay image ({n} frames) → {output_path}")

def visualize_result(folder_path, condition_path=None):
    v = Viewer()
    npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
    pathcondition=condition_path
    cond=np.load(pathcondition)
    cond=repair_data(cond)
    cond=subtract_root(cond)
    for i, path in enumerate(npy_files):
        name = os.path.splitext(os.path.basename(path))[0]
        color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        if "generated_motion_concat_" in name or "reference_motion_" in name:
            add_keypoints_result(np.load(path), v, name, color=color)
    add_keypoints_result(cond, v, 'condition', color=color)
    v.run()



if __name__ == "__main__":
    # Set the path to your SMPL models
    C.smplx_models = "smpl_models/"
    C.playback_fps = 30

    root="final_mydataset"
    #visualize_gait(keypoints_path, reference_path=keypoints_path2, condition_path=condition_path, smplseq_path=None, smplseq_reference_path=None)
    #visualize_smpl_keypoints(smplseq)
    #visualize_gait_batch(root+"/gait_114")
    #visualize_gait('test.npy')
    condition=os.path.join("final_dataset/gait_983/20251222_c2_a5_Take1/split_subjects\\0\\keypoints_3d\\smpl-keypoints-3d_cut.npy")
    visualize_result("test/weightmlp_rot", condition_path=condition)
    

    """
    (0.8, 0.1, 0.1, 1.0),  # red, gen
    (0.1, 0.1, 0.8, 1.0),  # blue, cond
    (0.1, 0.7, 0.1, 1.0),  # green, ref
    model="window_rot2"
    pathlista3=[
        (['test/' + model + '/generated_motion_concat_12.npy'], (0.8, 0.1, 0.1, 1.0), 'motion_turn_key'),
        (['test/' + model + '/reference_motion_11.npy'], (0.1, 0.7, 0.1, 1.0), 'motion_turn_key_ref'),
        (["final_dataset/gait_983/20251222_c2_a3_Take1/split_subjects\\0\\keypoints_3d\\smpl-keypoints-3d_cut.npy"], (0.1, 0.1, 0.8, 1.0), 'motion_turn_key_cond'),
    ]
    pathlista4=[
        (['test/' + model + '/generated_motion_concat_15.npy'], (0.8, 0.1, 0.1, 1.0), 'motion_cross_key'),
        (['test/' + model + '/reference_motion_14.npy'], (0.1, 0.7, 0.1, 1.0), 'motion_cross_key_ref'),
        (["final_dataset/gait_983/20251222_c2_a4_Take1/split_subjects\\0\\keypoints_3d\\smpl-keypoints-3d_cut.npy"], (0.1, 0.1, 0.8, 1.0), 'motion_cross_key_cond'),
    ]
    pathlista5=[
        (['test/' + model + '/generated_motion_concat_17.npy'], (0.8, 0.1, 0.1, 1.0), 'motion_pick_key'),
        (['test/' + model + '/reference_motion_17.npy'], (0.1, 0.7, 0.1, 1.0), 'motion_pick_key_ref'),
        (["final_dataset/gait_983/20251222_c2_a5_Take1/split_subjects\\0\\keypoints_3d\\smpl-keypoints-3d_cut.npy"], (0.1, 0.1, 0.8, 1.0), 'motion_pick_key_cond'),
    ]
    """
    """
    rotlist=["test/transformer_rot2/generated_motion_concat_5.npy", 
             "test/transformer_rot2/generated_motion_concat_11.npy",
             "test/transformer_rot2/generated_motion_concat_13.npy",
             "test/transformer_rot2/generated_motion_concat_18.npy",]
    
    headless_v = HeadlessRenderer(size=(1920, 1080))
    headless_v._default_nodes = list(headless_v.scene.nodes)
    for path in rotlist:
        capture_motion_frames(
            output_dir="result_motion",
            smplseq_path=None,
            keypoints_path=[path],
            n_frames=10,
            size=(1920, 1080),
            prefix="frame",
            colors=[(0.8, 0.1, 0.1, 1.0)],
            viewer=headless_v,
        )

        overlay_motion_frames(
            frame_dir="result_motion",
            output_path=os.path.basename(path).replace(".npy", "_overlay.png"),
            colormap="cool",
            alpha=0.6,
        )
    C.smplx_models = "smpl_models/"
    pathlist=[pathlista3, pathlista4, pathlista5]
    for pathgroup in pathlist:
      for path, color, name in pathgroup:
        capture_motion_frames(
            output_dir="result_motion",
            smplseq_path=None,
            keypoints_path=path,
            n_frames=10,
            size=(1920, 1080),
            prefix="frame",
            colors=[color],
            viewer=headless_v,
        )

        overlay_motion_frames(
            frame_dir="result_motion",
            output_path=name+".png",
            colormap="cool",
            alpha=0.6,
        )
    """

