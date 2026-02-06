# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn


###############################
############ Layers ###########
###############################


class MLPblock(nn.Module):
    def __init__(self, dim, seq0, seq1, first=False, w_embed=True):
        super().__init__()

        self.w_embed = w_embed
        self.fc0 = nn.Conv1d(seq0, seq1, 1)

        if self.w_embed:
            if first:
                self.conct = nn.Linear(dim * 2, dim)
            else:
                self.conct = nn.Identity()
            self.emb_fc = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, inputs):

        if self.w_embed:
            x = inputs[0]
            embed = inputs[1]
            x = self.conct(x) + self.emb_fc(self.act(embed))
        else:
            x = inputs

        x_ = self.norm0(x)
        x_ = self.fc0(x_)
        x_ = self.act(x_)
        x = x + x_

        x_ = self.norm1(x)
        x_ = self.fc1(x_)
        x_ = self.act(x_)

        x = x + x_

        if self.w_embed:
            return x, embed
        else:
            return x


class BaseMLP(nn.Module):
    def __init__(self, dim, seq, num_layers, w_embed=True):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                MLPblock(dim, seq, seq, first=i == 0 and w_embed, w_embed=w_embed)
            )

        self.mlps = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlps(x)
        return x


###############################
########### Networks ##########
###############################


class DiffMLP(nn.Module):
    def __init__(self, latent_dim=512, seq=98, num_layers=12):
        super(DiffMLP, self).__init__()

        self.motion_mlp = BaseMLP(dim=latent_dim, seq=seq, num_layers=num_layers)

    def forward(self, motion_input, embed):

        motion_feats = self.motion_mlp([motion_input, embed])[0]

        return motion_feats


class DiffTransformer(nn.Module):
    """Transformer-based backbone for diffusion models (based on MDM architecture)."""

    def __init__(
        self,
        latent_dim=512,
        seq=98,
        num_layers=8,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        activation="gelu",
    ):
        super(DiffTransformer, self).__init__()

        self.latent_dim = latent_dim
        self.seq = seq
        self.num_layers = num_layers

        # Input projection: 2*latent_dim -> latent_dim (like MLPblock's first conct)
        # This handles the concatenated input from MetaModel (sparse_emb + x)
        self.input_proj = nn.Linear(latent_dim * 2, latent_dim)

        # Positional encoding
        self.sequence_pos_encoder = PositionalEncoding(
            latent_dim, dropout, max_len=max(5000, seq + 1)
        )

        # Timestep embedding MLP (converts sinusoidal to learned)
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, motion_input, embed):
        """
        motion_input: [batch_size, seq, latent_dim * 2] - concatenated sparse + input
        embed: [batch_size, 1, latent_dim] - timestep embedding from sinusoidal encoding
        """
        # Project concatenated input to latent_dim
        x = self.input_proj(motion_input)  # [bs, seq, latent_dim]

        # Reshape embed from [bs, 1, d] to [1, bs, d] for transformer
        embed = embed.permute(1, 0, 2)  # [1, bs, latent_dim]

        # Process timestep embedding through MLP
        time_emb = self.time_embed(embed)  # [1, bs, latent_dim]

        # Reshape motion for transformer: [seq, batch_size, latent_dim]
        x = x.permute(1, 0, 2)

        # Prepend timestep token to sequence: [seq+1, bs, latent_dim]
        xseq = torch.cat((time_emb, x), dim=0)

        # Add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # Pass through transformer encoder
        output = self.transformer_encoder(xseq)

        # Remove timestep token: [seq, bs, latent_dim]
        output = output[1:]

        # Reshape back: [batch_size, seq, latent_dim]
        output = output.permute(1, 0, 2)

        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (based on MDM)."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class PureMLP(nn.Module):
    def __init__(
        self, latent_dim=512, seq=98, num_layers=12, input_dim=54, output_dim=132
    ):
        super(PureMLP, self).__init__()

        self.input_fc = nn.Linear(input_dim, latent_dim)
        self.motion_mlp = BaseMLP(
            dim=latent_dim, seq=seq, num_layers=num_layers, w_embed=False
        )
        self.output_fc = nn.Linear(latent_dim, output_dim)

    def forward(self, motion_input):

        motion_feats = self.input_fc(motion_input)
        motion_feats = self.motion_mlp(motion_feats)
        motion_feats = self.output_fc(motion_feats)

        return motion_feats
