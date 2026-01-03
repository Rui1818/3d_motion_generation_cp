import os

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from utils.transformation_sixd import smplx_to_6d


def drop_duplicate_frames(data):
    """
    Remove frames that are the same for all coordinates.
    """
    first_row = data[:, 0:1, :]  # Shape: (frames, 1, 5)
    all_rows_same = np.all(data == first_row, axis=(1,2))

    mask = ~all_rows_same
    return data[mask]

def subtract_root(data, keypointtype):
    #only after frames have been cut
    #also deletes 0 row
    if keypointtype=="openpose":
        root = (data[0,8,:]+data[0, 9, :])/2
        data=np.delete((data - root), (1,8), axis=1)
        assert data.shape[1]==23
    elif keypointtype=="smpl":
        root = data[0,0,:]
        data= data - root
        assert data.shape[1]==22
    else:
        raise ValueError(f"Unknown keypoint type: {keypointtype}")

    return data

def load_pure_keypoints(keypointspath, motionlist, keypointtype):
    keypoints = np.load(keypointspath)  # shape (frames, 25, 5) or (frames, 22, 3)
    keypoints = drop_duplicate_frames(keypoints)
    #reshape to (frame, num_joints*3)
    keypoints=keypoints[...,:3]
    keypoints = subtract_root(keypoints, keypointtype)
    if keypointtype=="openpose":
        keypoints = keypoints.reshape(-1, 23 * 3)
    elif keypointtype=="smpl":
        keypoints = keypoints.reshape(-1, 22 * 3)
    tensor = torch.tensor(keypoints, dtype=torch.float32)
    motionlist.append(tensor)
    return motionlist

def load_6drotations(motion_6dpath, motionlist):
    res = smplx_to_6d(motion_6dpath)
    motion_6d = res["motion_6d"]  # shape (frames, 132)
    transl = res["transl"]       # shape (frames, 3)
    betas = res["betas"]          # shape (frames, 11)
    motion_6d = torch.tensor(motion_6d, dtype=torch.float32)
    transl = torch.tensor(transl, dtype=torch.float32)
    result=torch.cat((transl, motion_6d), dim=1)  #shape (frames, 135)
    motionlist.append(result)
    return motionlist, betas

def normalize_motion(motion):
    #motion: tensor (frames, dim)
    if motion.shape[1]==69:
        motion=motion.reshape(-1,23,3)
        root = (motion[0,7,:]+motion[0, 10, :])/2
        motion = motion - root
        motion=motion.reshape(-1,69)
    elif motion.shape[1]==135:
        root = motion[0, :3]
        motion[:, :3] = motion[:, :3] - root
    return motion

class MotionDataset(Dataset):
    def __init__(
        self,
        dataset,
        motion_clean,
        motion_without_orth,
        mean=0,
        std=1,
        input_motion_length=196,
        train_dataset_repeat_times=1,
        no_normalization=False,
    ):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length

        # motion_clean and motion_without_orth are now dictionaries
        # with keys as action identifiers (e.g., 's01_a1')
        # and values as lists of tensors (different takes)
        self.motion_clean = motion_clean
        self.motion_without_orth = motion_without_orth

        # Create a list of action keys for indexing
        self.action_keys = list(motion_clean.keys())

        # Calculate base dataset length
        self.base_length = sum(len(takes) for takes in motion_clean.values())

    def __len__(self):
        return self.base_length * self.train_dataset_repeat_times

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx):
        # Map idx to an action
        idx = idx % self.base_length

        cumulative = 0
        selected_action = None
        for action_key in self.action_keys:
            num_c1_takes = len(self.motion_clean[action_key])
            if idx < cumulative + num_c1_takes:
                selected_action = action_key
                break
            cumulative += num_c1_takes

        # Randomly sample one take from c1 and one from c2 for this action
        c1_takes = self.motion_clean[selected_action]
        c2_takes = self.motion_without_orth[selected_action]

        motion = c1_takes[torch.randint(0, len(c1_takes), (1,)).item()]
        motion_w_o = c2_takes[torch.randint(0, len(c2_takes), (1,)).item()]

        seqlen = motion.shape[0]
        seqlen_wo = motion_w_o.shape[0]

        
        if seqlen <= self.input_motion_length:
            if seqlen > 0:
                frames_to_add = self.input_motion_length - seqlen
                last_frame = motion[-1:]
                padding = last_frame.repeat(frames_to_add, 1)
                motion = torch.cat([motion, padding], dim=0)
        else:
            sampleidx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]
            motion = motion[sampleidx:sampleidx+self.input_motion_length]
            motion=normalize_motion(motion)

        if seqlen_wo <= self.input_motion_length:
            if seqlen_wo > 0:
                frames_to_add = self.input_motion_length - seqlen_wo
                last_frame_wo = motion_w_o[-1:]
                padding_wo = last_frame_wo.repeat(frames_to_add, 1)
                motion_w_o = torch.cat([motion_w_o, padding_wo], dim=0)
        else:
            sampleidx_wo = torch.randint(0, int(seqlen_wo - self.input_motion_length), (1,))[0]
            motion_w_o = motion_w_o[sampleidx_wo:sampleidx_wo+self.input_motion_length]
            motion_w_o=normalize_motion(motion_w_o)

        """
        # Normalization
        if not self.no_normalization:
            motion = (motion - self.mean) / (self.std + 1e-8)
        """
        seqlen = seqlen if seqlen < self.input_motion_length else self.input_motion_length

        return seqlen, motion.float(), motion_w_o.float()
    
class TestDataset(Dataset):
    def __init__(
        self,
        dataset,
        motion_clean,
        motion_without_orth,
        betas=None,
        mean=0,
        std=1,
        input_motion_length=196,
        train_dataset_repeat_times=1,
        no_normalization=False,
    ):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.motion_clean = motion_clean
        self.motion_without_orth = motion_without_orth
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.betas = betas

        self.motion_clean = motion_clean
        self.motion_without_orth = motion_without_orth

        self.input_motion_length = input_motion_length

    def __len__(self):
        return len(self.motion_clean) * self.train_dataset_repeat_times

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx):
        motion = self.motion_clean[idx % len(self.motion_clean)]
        motion_w_o = self.motion_without_orth[idx % len(self.motion_clean)]
        beta=self.betas[idx % len(self.motion_clean)] if self.betas is not None else torch.tensor(0.0)
        seqlen_wo = motion_w_o.shape[0]

        if seqlen_wo <= self.input_motion_length:
            if seqlen_wo > 0:
                frames_to_add = self.input_motion_length - seqlen_wo
                last_frame_wo = motion_w_o[-1:]
                padding_wo = last_frame_wo.repeat(frames_to_add, 1)
                motion_w_o = torch.cat([motion_w_o, padding_wo], dim=0)
        else:
            # Truncate to input_motion_length if sequence is too long
            motion_w_o = motion_w_o[:self.input_motion_length]
            motion_w_o = normalize_motion(motion_w_o)

        """
        # Normalization
        if not self.no_normalization:
            motion = (motion - self.mean) / (self.std + 1e-8)
        """

        return motion.float(), motion_w_o.float(), beta

    


def load_data(motion_path, split, keypointtype="openpose",**kwargs):
    """
    Load SMPL keypoint .npy files from a folder into dictionaries organized by action.

    Args:
        dataset_path (str): Path to the folder containing .npy files.
        split (str): 'train' or 'test'
        keypointtype (str): Type of keypoints to load
        **kwargs: Optional keyword arguments for future extensions.

    Returns:
        For train split:
            motion_clean (dict): {action_key: [list of c1 tensors]}
            motion_w_o (dict): {action_key: [list of c2 tensors]}
        For test split:
            motion_clean (list): List of tensors
            motion_w_o (list): List of tensors
            betas (list or None): List of betas
    """
    if split == "train":
        motion_clean = {}
        motion_w_o = {}

        for patient in sorted(os.listdir(motion_path)):
            patient_path = os.path.join(motion_path, patient)
            for file in sorted(os.listdir(patient_path)):
                take = file.split('_')
                if take[1] != 'c1':
                    continue

                action = take[2]
                action_key = f"{patient}_{action}"

                if action_key not in motion_clean:
                    motion_clean[action_key] = []
                    motion_w_o[action_key] = []

                # Construct file paths
                def get_file_path(category, keypoint_type):
                    filename = take[0] + f'_{category}_' + "_".join(take[2:])
                    if keypoint_type == "6d":
                        return os.path.join(patient_path, filename, "split_subjects", "0", "fit-smplx", "smplx-params_cut.npz")
                    elif keypoint_type == "openpose":
                        return os.path.join(patient_path, filename, "split_subjects", "0", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                    elif keypoint_type == "smpl":
                        return os.path.join(patient_path, filename, "split_subjects", "0", "fit-smplx", "smpl-keypoints-3d_cut.npy")
                    else:
                        raise ValueError(f"Unknown keypoint type: {keypoint_type}")

                # Load c1 (with orthosis)
                c1_path = get_file_path('c1', keypointtype)
                temp_list = []
                if keypointtype == "6d":
                    temp_list, _ = load_6drotations(c1_path, temp_list)
                else:
                    temp_list = load_pure_keypoints(c1_path, temp_list, keypointtype)
                motion_clean[action_key].append(temp_list[0])

                # Load c2 (without orthosis)
                c2_path = get_file_path('c2', keypointtype)
                temp_list = []
                if keypointtype == "6d":
                    temp_list, _ = load_6drotations(c2_path, temp_list)
                else:
                    temp_list = load_pure_keypoints(c2_path, temp_list, keypointtype)
                motion_w_o[action_key].append(temp_list[0])

        if not motion_clean:
            raise FileNotFoundError(f"No files found in directory: {motion_path}")
        return motion_clean, motion_w_o
    elif split == "test":
        motion_clean = []
        motion_w_o = []
        betas = [] if keypointtype == "6d" else None

        for patient in sorted(os.listdir(motion_path)):
            patient_path = os.path.join(motion_path, patient)
            for file in sorted(os.listdir(patient_path)):
                take = file.split('_')
                if take[1] != 'c1':
                    continue

                no_orth_path = take[0] + '_c2_' + "_".join(take[2:])

                if keypointtype == "6d":
                    c1_path = os.path.join(patient_path, file, "split_subjects", "0", "fit-smplx", "smpl-keypoints-3d_cut.npy")
                    c2_path = os.path.join(patient_path, no_orth_path, "split_subjects", "0", "fit-smplx", "smplx-params_cut.npz")
                    motion_clean.append(torch.tensor(np.load(c1_path), dtype=torch.float32))
                    motion_w_o, beta = load_6drotations(c2_path, motion_w_o)
                    betas.append(beta)
                elif keypointtype == "openpose":
                    c1_path = os.path.join(patient_path, file, "split_subjects", "0", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                    c2_path = os.path.join(patient_path, no_orth_path, "split_subjects", "0", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                    motion_clean = load_pure_keypoints(c1_path, motion_clean, keypointtype)
                    motion_w_o = load_pure_keypoints(c2_path, motion_w_o, keypointtype)
                elif keypointtype == "smpl":
                    raise NotImplementedError("SMPL keypoints not implemented for test split.")
                else:
                    raise ValueError(f"Unknown keypoint type: {keypointtype}")

        return motion_clean, motion_w_o, betas



def get_dataloader(
    dataset,
    split,
    batch_size,
    num_workers=8,
):
    shuffle = split == "train"
    drop_last = split == "train"
    num_workers = num_workers if split == "train" else 1

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader