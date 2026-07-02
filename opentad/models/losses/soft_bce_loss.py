import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class SoftBCELoss(nn.Module):
    """Binary cross-entropy with logits for soft targets in [0, 1]."""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.as_tensor(self.pos_weight, device=inputs.device, dtype=inputs.dtype)
        loss = F.binary_cross_entropy_with_logits(inputs.float(), targets.float(), reduction="none", pos_weight=pos_weight)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return loss

    def __repr__(self):
        return f"{self.__class__.__name__}(pos_weight={self.pos_weight})"
