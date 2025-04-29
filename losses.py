import torch
import torchvision

class FocalLossBCE(torch.nn.Module):
    """
    Combines BCEWithLogitsLoss and sigmoid_focal_loss.
    Useful for situations where standard BCE might struggle with class imbalance,
    allowing adjustment between the two loss types.
    """
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 0.6,
            focal_weight: float = 1.4,
    ):
        super().__init__()
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha should be between 0 and 1, got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma should be non-negative, got {gamma}")
        if bce_weight < 0 or focal_weight < 0:
            raise ValueError("Loss weights cannot be negative")
        if bce_weight == 0 and focal_weight == 0:
            raise ValueError("At least one of bce_weight or focal_weight must be positive")
            
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Use pos_weight if provided? Currently not parameterized.
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        bce_loss = 0.0
        focal_loss = 0.0
        
        # Ensure targets are float for BCE and Focal loss
        targets = targets.float()
        
        if self.bce_weight > 0:
             bce_loss = self.bce(logits, targets)
             
        if self.focal_weight > 0:
            # torchvision's focal loss expects targets in the same format as input logits
            focal_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
                inputs=logits,
                targets=targets,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction,
            )
            
        # Combine the losses
        total_loss = self.bce_weight * bce_loss + self.focal_weight * focal_loss
        return total_loss