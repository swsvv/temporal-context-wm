import math

import torch
from torch import nn


class InitGoalProjector(nn.Module):
    def __init__(self, input_dim: int, repr_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = repr_dim

        self.init_goal_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(repr_dim),
        )

    def forward(self, z_0: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Helper to encode init(s_0)/goal(s_g) latent state"""
        context_input = torch.cat([z_0, z_g], dim=-1)

        # [B, D]
        context_enc = self.init_goal_projector(context_input)

        return context_enc


class StateProjector(nn.Module):
    # This is for env that using MeNet (e.g., Diverse PointMaze)
    def __init__(self, in_channels=18, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.reshape(T * B, C, H, W)
        out = self.net(x_flat)
        return out.reshape(T, B, -1)


class ContextTransformer(nn.Module):
    def __init__(
        self,
        # repr_dim=512,
        embed_dim=256,
        nhead=4,
        num_layers=2,
        condition_on_initial_goal=False,
        context_window_k=3,
        mask_random=True,
        mask_ratio=0.2,
    ) -> None:
        super().__init__()

        self.condition_on_initial_goal = condition_on_initial_goal

        self.mask_state = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_enc = PositionalEncoding(embed_dim, max_len=100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(embed_dim, embed_dim)

        self.mask_random = mask_random
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if self.condition_on_initial_goal:
            center_idx = (seq_len - 2) // 2
        else:
            center_idx = seq_len // 2

        x[:, center_idx, :] = self.mask_state.squeeze(0)

        if self.mask_random is True:
            probs = torch.rand(batch_size, seq_len, device=x.device)
            mask_bool = probs < self.mask_ratio

            if self.condition_on_initial_goal:
                mask_bool[:, -2:] = False

            mask_bool = mask_bool.unsqueeze(-1)
            x = torch.where(mask_bool, self.mask_state, x)

        x = self.pos_enc(x)

        out = self.transformer(x)

        center_out = out[:, center_idx, :]
        context_enc = self.output_head(center_out)

        return context_enc


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        # Add position encoding up to the current sequence length
        return x + self.pe[: x.size(1), :].unsqueeze(0)
