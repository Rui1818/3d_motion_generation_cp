# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs

import numpy as np
import torch

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def geodesic_distance_matrix(R1, R2):
    """
    Computes Geodesic Distance between two Rotation Matrices.
    d = arccos( (Tr(R1 * R2^T) - 1) / 2 )
    
    Input: R1, R2 with shape (..., 3, 3)
    Output: distances with shape (...)
    """
    # Compute R1 * R2^T
    # swapaxes to transpose the last two dimensions
    R_diff = np.matmul(R1, np.swapaxes(R2, -2, -1))
    
    # Trace is the sum of diagonal elements
    trace = np.trace(R_diff, axis1=-2, axis2=-1)
    
    # Clamp values to [-1, 1] to avoid NaN in arccos due to float errors
    trace = np.clip((trace - 1) / 2.0, -1.0, 1.0)
    
    # Return angle in radians
    return np.arccos(trace)

def bgs_numpy(d6s):
    a1 = d6s[..., :3]
    a2 = d6s[..., 3:]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)

def pose_distance_metric(frame_A, frame_B):
    """
    Custom distance function for DTW.
    Input: Two flattened frames of 6D data (Shape: [Num_Joints * 6])
    Output: Scalar distance (Average angular error in radians)
    """
    # 1. Reshape back to (Num_Joints, 6)
    # Assuming standard SMPL 24 joints, change '24' to your joint count
    n_joints = 22
    
    fA = frame_A.reshape(n_joints, 6)
    fB = frame_B.reshape(n_joints, 6)
    
    # 2. Convert to Rotation Matrices (Num_Joints, 3, 3)
    rmAT = bgs_numpy(fA)
    rmBT = bgs_numpy(fB)
    
    # 3. Calculate Geodesic Distance for each joint
    # Returns shape (Num_Joints,)
    joint_errors = geodesic_distance_matrix(rmAT, rmBT)
    
    # 4. Return average error across all joints for this frame
    return np.mean(joint_errors)

def calculate_motion_dtw(motion1, motion2, distance_metric=euclidean):
    """
    Calculates DTW distance between two human motion sequences.
    
    Args:
        motion1 (np.ndarray): Shape (frames_A, joints, 3)
        motion2 (np.ndarray): Shape (frames_B, joints, 3)
        
    Returns:
        float: The DTW distance.
        list: The warping path (list of tuples aligning frame i to frame j).
    """
    
    # 1. Input Validation
    if motion1.shape[1:] != motion2.shape[1:]:
        raise ValueError("Both motions must have the same number of joints and dimensions.")
    
    seq_a = motion1.reshape(motion1.shape[0], -1)
    seq_b = motion2.reshape(motion2.shape[0], -1)
    distance, path = fastdtw(seq_a, seq_b, dist=distance_metric)
    
    return distance, path


def calculate_jitter(motion, fps=30):
    """
    Calculates the jitter (jerk) of a motion sequence.

    Args:
        motion (torch.Tensor or np.ndarray): Shape (frames, joints, 3)
        fps (float): Frames per second. Default is 30.

    Returns:
        float: The average jitter across all joints and frames.
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    # Ensure motion is float
    motion = motion.float()

    # Calculate third derivative (jerk) using finite differences
    # p(t+3) - 3p(t+2) + 3p(t+1) - p(t)
    jerk = (
        motion[3:]
        - 3 * motion[2:-1]
        + 3 * motion[1:-2]
        - motion[:-3]
    )

    # Scale by FPS cubed
    jerk = jerk * (fps**3)

    # Calculate magnitude of jerk vectors (L2 norm along the coordinate dimension)
    # Shape: (frames-3, joints)
    jerk_norm = torch.norm(jerk, dim=2)

    # Return mean jitter across all valid frames and joints
    return jerk_norm.mean().item()


def pred_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pred_jitter = (
        (
            (
                predicted_position[3:]
                - 3 * predicted_position[2:-1]
                + 3 * predicted_position[1:-2]
                - predicted_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return pred_jitter


def gt_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    gt_jitter = (
        (
            (
                gt_position[3:]
                - 3 * gt_position[2:-1]
                + 3 * gt_position[1:-2]
                - gt_position[:-3]
            )
            * (fps**3)
        )
        .norm(dim=2)
        .mean()
    )
    return gt_jitter


def mpjre(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    diff = gt_angle - predicted_angle
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = torch.mean(torch.absolute(diff))
    return rot_error


def mpjpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))
    )
    return pos_error


def handpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error_hands = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [20, 21]
        ]
    )
    return pos_error_hands


def upperpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    upper_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., upper_index
        ]
    )
    return upper_body_error


def lowerpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    lower_body_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., lower_index
        ]
    )
    return lower_body_error


def rootpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    pos_error_root = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_position - predicted_position), axis=-1))[
            ..., [0]
        ]
    )
    return pos_error_root


def mpjve(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
):
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (
        predicted_position[1:, ...] - predicted_position[:-1, ...]
    ) * fps
    vel_error = torch.mean(
        torch.sqrt(torch.sum(torch.square(gt_velocity - predicted_velocity), axis=-1))
    )
    return vel_error


metric_funcs_dict = {
    "mpjre": mpjre,
    "mpjpe": mpjpe,
    "mpjve": mpjve,
    "handpe": handpe,
    "upperpe": upperpe,
    "lowerpe": lowerpe,
    "rootpe": rootpe,
    "pred_jitter": pred_jitter,
    "gt_jitter": gt_jitter,
}


def get_metric_function(metric):
    return metric_funcs_dict[metric]
