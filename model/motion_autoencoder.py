import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    """Residual block with a dilated depthwise conv + pointwise conv."""

    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=pad
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.conv(x)))
        x = self.drop(x)
        x = self.bn2(self.pointwise(x))
        return self.act(x + residual)


class MotionEncoder(nn.Module):
    """
    Encodes a motion sequence (B, T, D) into a fixed-size latent vector (B, latent_dim).

    Architecture:
        stem   : D → 256 channels
        res1   : local temporal patterns        (dilation=1)
        down1  : stride-2 downsample, T → T/2
        res2   : mid-range patterns             (dilation=2)
        down2  : stride-2 downsample, T/2 → T/4
        res3   : gait-cycle scale patterns      (dilation=2)
        GAP    : global average pooling → (B, 512)
        fc     : (B, 512) → (B, latent_dim)
    """

    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res1 = ResBlock1d(hidden_dim, kernel_size=5, dilation=1, dropout=dropout)
        self.down1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res2 = ResBlock1d(hidden_dim, kernel_size=5, dilation=2, dropout=dropout)
        self.down2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
        )
        self.res3 = ResBlock1d(hidden_dim * 2, kernel_size=5, dilation=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x.permute(0, 2, 1)          # (B, D, T)
        x = self.stem(x)                 # (B, 256, T)
        x = self.res1(x)
        x = self.down1(x)                # (B, 256, T/2)
        x = self.res2(x)
        x = self.down2(x)                # (B, 512, T/4)
        x = self.res3(x)
        x = self.pool(x).squeeze(-1)     # (B, 512)
        return self.fc(x)                # (B, latent_dim)


class MotionDecoder(nn.Module):
    """
    Decodes a latent vector (B, latent_dim) back to a motion sequence (B, T, D).

    Upsampling path: base_t=16 → T//4 → T//2 → T  (3 bilinear interpolations)
    This is intentionally simple since we only need the decoder as a training signal.
    """

    def __init__(
        self,
        output_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        base_t: int = 16,
    ):
        super().__init__()
        self.base_t = base_t
        self.fc = nn.Linear(latent_dim, hidden_dim * 2 * base_t)

        self.res1 = ResBlock1d(hidden_dim * 2, kernel_size=5, dilation=2, dropout=dropout)
        self.up1 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res2 = ResBlock1d(hidden_dim, kernel_size=5, dilation=2, dropout=dropout)
        self.up2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res3 = ResBlock1d(hidden_dim, kernel_size=5, dilation=1, dropout=dropout)
        self.out_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=5, padding=2)

    def forward(self, z: torch.Tensor, T: int) -> torch.Tensor:
        # z: (B, latent_dim)
        B = z.shape[0]
        hidden_dim2 = self.up1[0].in_channels          # hidden_dim * 2
        x = self.fc(z).view(B, hidden_dim2, self.base_t)

        x = self.res1(x)
        x = F.interpolate(x, size=max(T // 4, 1), mode="linear", align_corners=False)
        x = self.up1(x)                                # (B, 256, T/4)

        x = self.res2(x)
        x = F.interpolate(x, size=max(T // 2, 1), mode="linear", align_corners=False)
        x = self.up2(x)                                # (B, 256, T/2)

        x = self.res3(x)
        x = F.interpolate(x, size=T, mode="linear", align_corners=False)
        x = self.out_conv(x)                           # (B, D, T)
        return x.permute(0, 2, 1)                      # (B, T, D)


class MotionAutoencoder(nn.Module):
    """
    Full autoencoder for motion sequences.

    Input : (B, T, D)
    Latent: (B, latent_dim)  ← use encoder.forward() to extract FID features
    Output: (B, T, D)        ← reconstruction for training signal
    """

    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = MotionEncoder(input_dim, latent_dim, hidden_dim, dropout)
        self.decoder = MotionDecoder(input_dim, latent_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor):
        T = x.shape[1]
        z = self.encoder(x)
        recon = self.decoder(z, T)
        return recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent features for FID computation."""
        return self.encoder(x)
