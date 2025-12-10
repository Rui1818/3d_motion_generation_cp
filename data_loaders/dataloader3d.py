import os

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
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
        self.motion_clean = motion_clean
        self.motion_without_orth = motion_without_orth
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization

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
        seqlen = motion.shape[0]
        seqlen_wo = motion_w_o.shape[0]
        if seqlen <= self.input_motion_length:
            if seqlen > 0:
                frames_to_add = self.input_motion_length - seqlen
                last_frame = motion[-1:]
                padding = last_frame.repeat(frames_to_add, 1)
                motion = torch.cat([motion, padding], dim=0)
        else:
            motion = motion[0:self.input_motion_length]

        if seqlen_wo <= self.input_motion_length:
            if seqlen_wo > 0:
                frames_to_add = self.input_motion_length - seqlen_wo
                last_frame_wo = motion_w_o[-1:]
                padding_wo = last_frame_wo.repeat(frames_to_add, 1)
                motion_w_o = torch.cat([motion_w_o, padding_wo], dim=0)
        else:
            motion_w_o = motion_w_o[0:self.input_motion_length]

        """
        # Normalization
        if not self.no_normalization:
            motion = (motion - self.mean) / (self.std + 1e-8)
        """

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
            motion_w_o = motion_w_o[0:self.input_motion_length]

        """
        # Normalization
        if not self.no_normalization:
            motion = (motion - self.mean) / (self.std + 1e-8)
        """

        return motion.float(), motion_w_o.float(), beta

    


def load_data(motion_path, split, keypointtype="openpose",**kwargs):
    """
    Load SMPL keypoint .npy files from a folder into a list of PyTorch tensors.

    Args:
        dataset_path (str): Path to the folder containing .npy files.
        **kwargs: Optional keyword arguments for future extensions.

    Returns: 
        list[torch.Tensor]: A list of tensors, one per file, each of shape (frames, 72).
    """
    if split == "train":
        motion_clean =[]
        motion_w_o=[]
        for patient in sorted(os.listdir(motion_path)):
            for file in sorted(os.listdir(os.path.join(motion_path, patient))):
                take=file.split('_')
                if take[1]=='c1':
                    #motion with orthosis
                    if keypointtype=="6d":
                        file_path = os.path.join(motion_path, patient, file, "split_subjects", "0", "fit-smplx", "smplx-params_cut.npz")
                        motion_clean,_=load_6drotations(file_path, motion_clean)
                        no_orth_path = take[0]+'_c2_'+"_".join(take[2:])
                        file_path_wo = os.path.join(motion_path, patient, no_orth_path, "split_subjects", "0", "fit-smplx", "smplx-params_cut.npz")
                        motion_w_o,_=load_6drotations(file_path_wo, motion_w_o)
                    elif keypointtype=="openpose" or keypointtype=="smpl":
                        if keypointtype=="openpose":
                            file_path = os.path.join(motion_path, patient, file, "split_subjects", "0", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                            no_orth_path = take[0]+'_c2_'+"_".join(take[2:])
                            file_path_wo = os.path.join(motion_path, patient, no_orth_path, "split_subjects", "0", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                        elif keypointtype=="smpl":
                            file_path = os.path.join(motion_path, patient, file, "split_subjects", "0", "fit-smplx", "smpl-keypoints-3d_cut.npy")
                            no_orth_path = take[0]+'_c2_'+"_".join(take[2:])
                            file_path_wo = os.path.join(motion_path, patient, no_orth_path, "split_subjects", "0", "fit-smplx", "smpl-keypoints-3d_cut.npy")
                        motion_clean=load_pure_keypoints(file_path, motion_clean, keypointtype)
                        motion_w_o=load_pure_keypoints(file_path_wo, motion_w_o, keypointtype)
                    else:
                        raise ValueError(f"Unknown keypoint type: {keypointtype}")
        if not motion_clean:
            raise FileNotFoundError(f"No files found in directory: {motion_path}")
        return motion_clean, motion_w_o
    elif split == "test":
        motion_clean =[]
        motion_w_o=[]
        betas=[] if keypointtype=="6d" else None
        for patient in sorted(os.listdir(motion_path)):
            for file in sorted(os.listdir(os.path.join(motion_path, patient))):
                take=file.split('_')
                if take[1]=='c1':
                    #motion with orthosis
                    if keypointtype=="6d":
                        file_path = os.path.join(motion_path, patient, file, "split_subjects", "0", "fit-smplx", "smpl-keypoints-3d_cut.npy")
                        motion_clean.append(torch.tensor(np.load(file_path), dtype=torch.float32))
                        no_orth_path = take[0]+'_c2_'+"_".join(take[2:])
                        file_path_wo = os.path.join(motion_path, patient, no_orth_path, "split_subjects", "0", "fit-smplx", "smplx-params_cut.npz")
                        motion_w_o,beta=load_6drotations(file_path_wo, motion_w_o)
                        betas.append(beta)
                    elif keypointtype=="openpose" or keypointtype=="smpl":
                        if keypointtype=="openpose":
                            file_path = os.path.join(motion_path, patient, file, "split_subjects", "0", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                            no_orth_path = take[0]+'_c2_'+"_".join(take[2:])
                            file_path_wo = os.path.join(motion_path, patient, no_orth_path, "split_subjects", "0", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                        elif keypointtype=="smpl":
                            raise NotImplementedError("SMPL keypoints not implemented for test split.")
                        motion_clean=load_pure_keypoints(file_path, motion_clean, keypointtype)
                        motion_w_o=load_pure_keypoints(file_path_wo, motion_w_o, keypointtype)
                    else:
                        raise ValueError(f"Unknown keypoint type: {keypointtype}")
        return motion_clean, motion_w_o, betas



def get_dataloader(
    dataset,
    split,
    batch_size,
    num_workers=8,
):

    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader