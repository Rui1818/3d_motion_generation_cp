import numpy as np
from tslearn.metrics import dtw_path_from_metric


def calculate_lower_body_angles(skeletonmotion: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calculate hip, knee, and ankle angles from a skeleton sequence.

    Args:
        skeletonmotion: array of shape (frames, joints, 3)

    Returns:
        dict mapping angle name -> array of shape (frames,) in degrees
    """
    # Each entry: (joint_before_vertex, vertex, joint_after_vertex)
    # Replace the placeholder integers with your actual joint indices.
    joints=skeletonmotion.shape[1]

    if joints == 23:
    #openpose
        limbs = {
            "left_hip":   (4,10,11),   # e.g. (LShoulder, LHip, LKnee)
            "right_hip":  (1,7,8),   # e.g. (RShoulder, RHip, RKnee)
            "left_knee":  (10,11,12),   # e.g. (LHip, LKnee, LAnkle)
            "right_knee": (7,8,9),   # e.g. (RHip, RKnee, RAnkle)
            "left_ankle": (11,12,17),   # e.g. (LKnee, LAnkle, LBigToe)
            "right_ankle":(8,9,20),   # e.g. (RKnee, RAnkle, RBigToe)
        }
    elif joints == 22:
    #smpl rotation
        limbs = {
            "left_hip":   (16, 1, 4),   # LShoulder, LHip, LKnee
            "right_hip":  (17, 2, 5),   # RShoulder, RHip, RKnee
            "left_knee":  (1, 4, 7),  # LHip, LKnee, LAnkle
            "right_knee": (2, 5, 8), # RHip, RKnee, RAnkle
            "left_ankle": (4, 7, 10), # LKnee, LAnkle, LBigToe
            "right_ankle":(5, 8, 11) # RKnee, RAnkle, RBigToe
        }
    else:
        raise ValueError(f"Unsupported number of joints: {joints}")

    angles = {}
    for name, (j1, vertex, j3) in limbs.items():
        p1 = skeletonmotion[:, j1]      # (frames, 3)
        p2 = skeletonmotion[:, vertex]  # (frames, 3)  — the joint being measured
        p3 = skeletonmotion[:, j3]      # (frames, 3)

        v1 = p1 - p2
        v2 = p3 - p2

        dot      = np.sum(v1 * v2, axis=1)
        norm_v1  = np.linalg.norm(v1, axis=1)
        norm_v2  = np.linalg.norm(v2, axis=1)

        cos_a = dot / (norm_v1 * norm_v2 + 1e-8)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles[name] = np.degrees(np.arccos(cos_a))   # 180° = straight, 0° = fully bent

    return angles


def test_from_file(file_path: str) -> dict[str, np.ndarray]:
    """
    Load a skeleton motion from a .npy file and return its lower-body angles.

    The file should contain an array of shape (frames, joints, 3).
    If the array has more than 3 dimensions (e.g. a 4th confidence channel),
    only the first 3 values along the last axis are used.

    Args:
        file_path: path to the .npy file

    Returns:
        dict mapping angle name -> array of shape (frames,) in degrees
    """
    motion = np.load(file_path)
    print(f"Loaded '{file_path}'  shape: {motion.shape}  dtype: {motion.dtype}")

    if motion.ndim != 3:
        raise ValueError(f"Expected 3-D array (frames, joints, 3), got shape {motion.shape}")

    motion = motion[..., :3]  # drop any extra channels (e.g. confidence)

    angles = calculate_lower_body_angles(motion)

    print(f"\n{'Joint':<15} {'min':>8} {'mean':>8} {'max':>8}  (degrees)")
    print("-" * 45)
    for name, vals in angles.items():
        print(f"{name:<15} {vals.min():>8.1f} {vals.mean():>8.1f} {vals.max():>8.1f}")

    return angles


def _angle_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error between two scalar angle values (used as DTW local metric)."""
    return float(np.abs(a - b).item())


def dtw_angle_error(
    motion_ref: np.ndarray,
    motion_gen: np.ndarray,
) -> dict[str, float]:
    """
    Compute DTW-aligned mean absolute angle error per limb between two motion sequences.

    Both sequences are aligned independently per limb using DTW, matching the approach
    in gait_generate.py (tslearn dtw_path_from_metric, path-length-normalised).

    Args:
        motion_ref: shape (frames_ref, joints, 3)
        motion_gen: shape (frames_gen, joints, 3)

    Returns:
        dict mapping limb name -> mean absolute angle error in degrees (DTW-aligned)
    """
    angles_ref = calculate_lower_body_angles(motion_ref[..., :3])
    angles_gen = calculate_lower_body_angles(motion_gen[..., :3])

    errors = {}
    for name in angles_ref:
        ref_seq = angles_ref[name].reshape(-1, 1)  # (frames_ref, 1)
        gen_seq = angles_gen[name].reshape(-1, 1)  # (frames_gen, 1)

        path, x = dtw_path_from_metric(ref_seq, gen_seq, metric=_angle_mae)
        errors[name] = x / len(path)

    return errors


if __name__ == "__main__":
    motion1 = "test/transformer_key/generated_motion_concat_0.npy"
    motion2 = "test/transformer_key/reference_motion_0.npy"
    errors= dtw_angle_error(np.load(motion1), np.load(motion2))
    print("\nDTW-aligned mean absolute angle errors (degrees):")
    for limb, error in errors.items():
        print(f"{limb:<15} {error:.2f}")