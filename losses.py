import torch
import torchvision

class FocalLossBCE(torch.nn.Module):
    """
    Combines BCEWithLogitsLoss and sigmoid_focal_loss.
    Useful for situations where standard BCE might struggle with class imbalance,
    allowing adjustment between the two loss types.
    Accepts configuration via a config object.
    """
    def __init__(self, config, reduction: str = "mean"):
        super().__init__()

        self.alpha = config.focal_loss_alpha
        self.gamma = config.focal_loss_gamma
        self.bce_weight = config.focal_loss_bce_weight
        self.focal_weight = max(0.0, 2.0 - self.bce_weight)
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits, targets):
        bce_loss = 0.0
        focal_loss = 0.0
        
        targets = targets.float()
        
        if self.bce_weight > 0:
             bce_loss = self.bce(logits, targets)
             
        if self.focal_weight > 0:
            focal_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
                inputs=logits,
                targets=targets,
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction,
            )
            
        total_loss = self.bce_weight * bce_loss + self.focal_weight * focal_loss
        return total_loss