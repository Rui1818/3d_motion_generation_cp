# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs

import numpy as np
import torch

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def calculate_motion_dtw(motion1, motion2):
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
    distance, path = fastdtw(seq_a, seq_b, dist=euclidean)
    
    return distance, path


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
