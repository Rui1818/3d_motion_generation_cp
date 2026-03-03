import numpy as np
from utils.metrics import _procrustes_align


def rotation_matrix_z(angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [ np.cos(a), -np.sin(a), 0],
        [ np.sin(a),  np.cos(a), 0],
        [         0,          0, 1],
    ])


# Fixed body pose used across tests: 5 joints, clearly asymmetric
A = np.array([
    [ 0.0,  0.0,  0.0],  # root
    [ 1.0,  0.0,  0.0],  # right
    [-1.0,  0.0,  0.0],  # left
    [ 0.5,  1.0,  0.3],  # upper-right
    [-0.3,  1.5, -0.2],  # upper-left
])


def test_identity():
    """Aligning A to itself should return A unchanged."""
    result = _procrustes_align(A, A.copy())
    assert np.allclose(result, A, atol=1e-10), f"Identity failed:\n{result}\n!=\n{A}"
    print("PASS test_identity")


def test_translation_only():
    """B = A + t  →  aligned B should recover A."""
    t = np.array([5.0, -3.0, 2.0])
    B = A + t
    result = _procrustes_align(A, B)
    assert np.allclose(result, A, atol=1e-10), f"Translation failed:\n{result}\n!=\n{A}"
    print("PASS test_translation_only")


def test_scale_only():
    """B = s * A  →  aligned B should recover A (centred at A's mean)."""
    s = 3.7
    # Shift A away from origin so mu_A != 0
    A_shifted = A + np.array([2.0, 1.0, -1.0])
    B = s * A_shifted
    result = _procrustes_align(A_shifted, B)
    assert np.allclose(result, A_shifted, atol=1e-10), f"Scale failed:\n{result}\n!=\n{A_shifted}"
    print("PASS test_scale_only")


def test_rotation_only():
    """B = A @ R  →  aligned B should recover A."""
    R = rotation_matrix_z(47)           # arbitrary non-trivial angle
    B = A @ R.T                         # rotate each joint row-vector
    result = _procrustes_align(A, B)
    assert np.allclose(result, A, atol=1e-8), f"Rotation failed:\n{result}\n!=\n{A}"
    print("PASS test_rotation_only")


def test_combined():
    """B = s * (A @ R) + t  →  aligned B should recover A."""
    R = rotation_matrix_z(123)
    s = 2.5
    t = np.array([-1.0, 4.0, 0.5])
    B = s * (A @ R.T) + t
    result = _procrustes_align(A, B)
    assert np.allclose(result, A, atol=1e-8), f"Combined failed:\n{result}\n!=\n{A}"
    print("PASS test_combined")


def test_result_is_proper_rotation():
    """The implicit rotation inside _procrustes_align should have det = +1 (no reflection)."""
    R = rotation_matrix_z(200)
    B = A @ R.T
    # Recover R numerically: since both A and B are centered at origin already,
    # the aligned B = A means B_aligned = A, i.e., B0_unit @ R_internal = A0_unit
    mu_A = A.mean(0);  A0 = (A - mu_A);  A0 /= np.sqrt((A0**2).sum())
    mu_B = B.mean(0);  B0 = (B - mu_B);  B0 /= np.sqrt((B0**2).sum())
    U, _, Vt = np.linalg.svd(B0.T @ A0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    # whichever formula is used in the implementation, check det == +1
    R_internal = U @ D @ Vt
    assert np.isclose(np.linalg.det(R_internal), 1.0, atol=1e-10), \
        f"Rotation det = {np.linalg.det(R_internal):.6f}, expected +1"
    print("PASS test_result_is_proper_rotation")


if __name__ == "__main__":
    test_identity()
    test_translation_only()
    test_scale_only()
    test_rotation_only()
    test_combined()
    test_result_is_proper_rotation()
    print("\nAll tests passed.")
