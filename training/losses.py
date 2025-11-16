"""
Loss functions for chest X-ray classification
Includes: BCE, Focal Loss, Asymmetric Loss (ASL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss to address class imbalance and hard sample mining

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for positive samples
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) [B, C]
            targets: Ground truth labels [B, C]

        Returns:
            loss: Computed loss
        """
        # Apply sigmoid to convert logits to probabilities
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Focal Loss modification
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) to handle class imbalance and hard negative mining

    Reference: Ridnik et al. "Asymmetric Loss for Multi-Label Classification" (ICCV 2021)
    Used in Phase 2 training as per the paper
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Clip the predictions to prevent extreme values
            eps: Small epsilon to prevent log(0)
            reduction: 'mean', 'sum', or 'none'
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """
        Args:
            x: Model predictions (logits) [B, C]
            y: Ground truth labels [B, C]

        Returns:
            loss: Computed loss
        """
        # Convert to probabilities
        x_sigmoid = torch.sigmoid(x)

        # Clip predictions to prevent extreme values
        xs_min = x_sigmoid.clamp(min=self.eps)
        xs_max = x_sigmoid.clamp(max=1 - self.eps)

        # Asymmetric term for positive and negative samples
        loss_pos = -y * torch.log(xs_min) * torch.pow(1 - xs_min, self.gamma_pos)
        loss_neg = -(1 - y) * torch.log(1 - xs_max) * torch.pow(xs_max, self.gamma_neg)

        loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


def get_loss_function(loss_name='bce', **kwargs):
    """
    Get loss function by name

    Args:
        loss_name: 'bce', 'focal', or 'asymmetric'
        **kwargs: Additional arguments for the loss function

    Returns:
        loss_fn: Loss function
    """
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()

    elif loss_name == 'focal':
        alpha = kwargs.get('alpha', 1)
        gamma = kwargs.get('gamma', 2)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name == 'asymmetric':
        gamma_neg = kwargs.get('gamma_neg', 4)
        gamma_pos = kwargs.get('gamma_pos', 1)
        clip = kwargs.get('clip', 0.05)
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    else:
        raise ValueError(f"Unknown loss: {loss_name}. Supported: bce, focal, asymmetric")


if __name__ == '__main__':
    # Test loss functions
    print("Testing loss functions...")

    batch_size, num_classes = 4, 14
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()

    print(f"\nLogits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    # BCE
    bce_loss = get_loss_function('bce')
    print(f"\nBCE Loss: {bce_loss(logits, labels).item():.4f}")

    # Focal Loss
    focal_loss = get_loss_function('focal', alpha=1, gamma=2)
    print(f"Focal Loss: {focal_loss(logits, labels).item():.4f}")

    # Asymmetric Loss
    asl_loss = get_loss_function('asymmetric', gamma_neg=4, gamma_pos=1)
    print(f"Asymmetric Loss: {asl_loss(logits, labels).item():.4f}")
