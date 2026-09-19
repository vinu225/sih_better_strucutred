import torch
import torch.nn as nn
from backend.config import VISION_FEATURE_DIM, PROJECTOR_HIDDEN_DIM


class VisionProjector(nn.Module):
    """
    Trainable multimodal projection network mapping 640-dim multispectral
    satellite embeddings into the hidden embedding space of the language model.
    """
    def __init__(
        self,
        input_dim: int = VISION_FEATURE_DIM,
        output_dim: int = 896,
        hidden_dim: int = PROJECTOR_HIDDEN_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Visual features of shape (B, input_dim) or (B, N, input_dim)
        Returns:
            Projected embeddings of shape (B, output_dim) or (B, N, output_dim)
        """
        return self.projector(x)
