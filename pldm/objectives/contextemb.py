from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from pldm.configs import ConfigBase
from pldm.models.context_projectors import ContextTransformer, StateProjector
from pldm.models.jepa import ForwardResult


class ContextEMBLossInfo(NamedTuple):
    total_loss: torch.Tensor
    context_loss: torch.Tensor
    context_t_loss: torch.Tensor

    source_norm: torch.Tensor
    target_norm: torch.Tensor

    loss_name: str = "contextemb"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_context_loss": self.context_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_context_t_loss": self.context_t_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_source_norm": self.source_norm.item(),
            f"{self.name_prefix}/{self.loss_name}_target_norm": self.target_norm.item(),
        }


@dataclass
class ContextEMBObjectiveConfig(ConfigBase):
    repr_loss: str = "mse"

    use_context_objective: str = "context_transformer"

    condition_on_initial_goal: bool = False

    # Weight for the context loss
    context_objective_coeff: float = 1.0

    # Number of surround states to use (e.g., 3 -> total # of states to use are 6)
    context_window_k: int = 3

    context_t_loss_coeff: float = 0.1

    embed_dim: int = 512


class ContextEMBObjective(torch.nn.Module):
    def __init__(
        self, config: ContextEMBObjectiveConfig, repr_dim: int, name_prefix: str = ""
    ):
        super().__init__()
        self.repr_loss = config.repr_loss
        self.name_prefix = name_prefix
        self.use_context_objective = config.use_context_objective
        self.condition_on_initial_goal = config.condition_on_initial_goal
        self.context_objective_coeff = config.context_objective_coeff
        self.context_window_k = config.context_window_k
        self.context_t_loss_coeff = config.context_t_loss_coeff
        self.repr_dim = repr_dim
        self.embed_dim = config.embed_dim

        if isinstance(self.repr_dim, tuple):
            self.state_proj = StateProjector(
                in_channels=self.repr_dim[0], embed_dim=self.embed_dim
            ).cuda()
        else:
            self.state_proj = nn.Linear(self.repr_dim, self.embed_dim).cuda()

        if self.use_context_objective == "context_transformer":
            self.context_transformer = ContextTransformer(
                embed_dim=self.embed_dim,
                nhead=4,
                num_layers=2,
                condition_on_initial_goal=self.condition_on_initial_goal,
            ).cuda()
        else:
            raise ValueError(f"Unknown context projector: {self.use_context_objective}")

        if self.context_objective_coeff == 0.0:
            raise NotImplementedError("ContextEMBObjective is not used!")

    def __call__(
        self, _batch, result: List[ForwardResult]
    ) -> Optional[ContextEMBLossInfo]:
        # This objective only works on level1 of HJEPA
        l1_result = result[0]

        # MeNet: [T, B, 18, 26, 26]
        raw_encodings = l1_result.backbone_output.encodings

        # Project to 1D Latent Space: [T, B, embed_dim]
        projected_encodings = self.state_proj(raw_encodings)

        if projected_encodings.shape[0] <= self.context_window_k * 2:
            raise ValueError(
                "ContextEMBObjective received sequence too short to process!"
            )

        state_encs = projected_encodings

        # Making input dimension suitable for objective's input dimension
        # state_encs: [T - k * 2, B, k * 2, D] == [Num_Windows, B, Window_Size, D]
        expanded_state_encs = self._get_context_by_sliding_transformer(state_encs)

        num_windows, batch_size, window_len, dim = expanded_state_encs.shape

        # This treats every time-step's window as an independent sequence.
        # shape: [Num_Windows * B, Window_Len, D]
        expanded_state_encs = expanded_state_encs.reshape(-1, window_len, dim)

        if self.condition_on_initial_goal:
            # init_goal_enc: [B, 2D]
            init_goal_enc = torch.cat(
                [
                    projected_encodings[0],  # Init state
                    projected_encodings[-1],  # Goal state
                ],
                dim=-1,
            )

            init_goal_enc = init_goal_enc.unsqueeze(0).expand(num_windows, -1, -1)
            init_goal_enc = init_goal_enc.reshape(-1, 2, dim)

            # expanded_state_encs: [Super Batch, (win_len + 2), D]
            # win_len includes center step
            expanded_state_encs = torch.cat(
                [expanded_state_encs, init_goal_enc],
                dim=1,
            )

        # sources: [Num_Windows * B, Window_Len, D]
        sources = self.context_transformer(expanded_state_encs)

        # Reshape it back to [Num_Windows, B, D]
        sources = sources.reshape(num_windows, batch_size, -1)

        # targets: [T, B, D]
        # targets = raw_encodings.detach()
        targets = projected_encodings.detach()

        # targets: [T - k * 2, B, D]
        targets = targets[self.context_window_k : len(targets) - self.context_window_k]

        context_loss, context_t_loss = self._context_loss(sources, targets)
        total_loss = self.context_objective_coeff * (context_loss + context_t_loss)

        source_norm = sources.norm(dim=-1).mean()
        target_norm = targets.norm(dim=-1).mean()

        return ContextEMBLossInfo(
            total_loss=total_loss,
            context_loss=context_loss,
            context_t_loss=context_t_loss,
            name_prefix=self.name_prefix,
            source_norm=source_norm,
            target_norm=target_norm,
        )

    def _get_context_by_sliding_transformer(
        self, state_encs: torch.Tensor
    ) -> torch.Tensor:
        """
        state_encs: [T, B, D]
        Returns: [Num_Windows, B, Window_Size, D]
        """
        window_size = 2 * self.context_window_k + 1

        # 1. Permute to [B, D, T] because unfold operates on the last dim usually,
        # or we can unfold dim 0 directly. Let's unfold dim 0 (Time).

        # Unfold creates a sliding window view.
        # Input: [T, B, D] -> Unfold dim 0
        # Output: [Num_Windows, B, D, Window_Size]
        windows = state_encs.unfold(dimension=0, size=window_size, step=1)

        # 2. We need to reshape
        # Current: [Num_Windows, B, D, Window_Size]
        # Reshape: [Num_Windows, B, Window_Size, D]
        windows = windows.permute(0, 1, 3, 2).contiguous()

        return windows

    def _context_loss(
        self, source_contexts: torch.Tensor, target_contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sources_norm = F.normalize(source_contexts, dim=-1)
        targets_norm = F.normalize(target_contexts, dim=-1)

        if self.repr_loss == "mse":
            context_loss = F.mse_loss(sources_norm, targets_norm)
        elif self.repr_loss == "l1":
            context_loss = F.l1_loss(sources_norm, targets_norm)
        elif self.repr_loss == "cosine":
            context_loss = 2 - 2 * (sources_norm * targets_norm).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss function: {self.repr_loss}")

        diff_sq = (sources_norm[1:] - sources_norm[:-1]).pow(2).sum(dim=-1)
        loss_temporal = F.relu(diff_sq - 0.1).mean()

        context_t_loss = self.context_t_loss_coeff * loss_temporal

        return context_loss, context_t_loss
