"""
Orthonormal DCT-II and IDCT-II via matrix multiplication (GPU-compatible).

Applies the transform along the temporal axis (dim=1) of tensors with shape (B, T, D).
Uses orthonormal normalization so that the transform preserves L2 norms,
which is important for diffusion models (Gaussian noise properties are preserved).
"""

import torch
import math

# Cache for DCT matrices to avoid recomputation
_dct_matrix_cache = {}


def _get_dct_matrix(N, dtype, device):
    """Get or create the orthonormal DCT-II matrix of size (N, N)."""
    key = (N, dtype, device)
    if key not in _dct_matrix_cache:
        n = torch.arange(N, dtype=dtype, device=device)
        k = torch.arange(N, dtype=dtype, device=device)
        # DCT-II basis: C[k, n] = cos(pi * k * (2n + 1) / (2N))
        C = torch.cos(math.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * N))
        # Orthonormal normalization
        C[0, :] *= 1.0 / math.sqrt(N)
        C[1:, :] *= math.sqrt(2.0 / N)
        _dct_matrix_cache[key] = C
    return _dct_matrix_cache[key]


def dct(x):
    """
    Orthonormal DCT-II along dim=1 (temporal axis).

    Input:  (B, T, D)
    Output: (B, T, D) — DCT coefficients
    """
    N = x.shape[1]
    C = _get_dct_matrix(N, x.dtype, x.device)
    # Y[b, k, d] = sum_n C[k, n] * x[b, n, d]
    return torch.einsum("kn,bnd->bkd", C, x)


def idct(X):
    """
    Orthonormal IDCT-II (inverse DCT-II) along dim=1 (temporal axis).

    Input:  (B, T, D) — DCT coefficients
    Output: (B, T, D) — reconstructed temporal signal
    """
    N = X.shape[1]
    C = _get_dct_matrix(N, X.dtype, X.device)
    # Since C is orthogonal, C^{-1} = C^T
    # x[b, n, d] = sum_k C^T[n, k] * X[b, k, d] = sum_k C[k, n] * X[b, k, d]
    return torch.einsum("kn,bkd->bnd", C, X)
