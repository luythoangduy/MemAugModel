"""
FastAI Learner creation following mem_model.ipynb structure
"""

from fastai.vision.all import *
import torch.nn as nn
import torch
from copy import deepcopy


# Focal Loss Implementation (from notebook)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss to address class imbalance and hard sample mining

        Args:
            alpha (float): Weighting factor for positive samples
            gamma (float): Focusing parameter
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss

        Args:
            inputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss
        """
        # Apply sigmoid to convert logits to probabilities
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Focal Loss modification
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# Asymmetric Loss Implementation (from notebook)
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        """
        Asymmetric Loss to handle class imbalance and hard negative mining

        Args:
            gamma_neg (float): Focusing parameter for negative samples
            gamma_pos (float): Focusing parameter for positive samples
            clip (float): Clip the predictions to prevent extreme values
            eps (float): Small epsilon to prevent log(0)
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """
        Compute asymmetric loss

        Args:
            x (torch.Tensor): Model predictions (logits)
            y (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss
        """
        # Convert to probabilities
        x_sigmoid = torch.sigmoid(x)

        # Clip predictions to prevent extreme values
        xs_min = x_sigmoid.clamp(min=self.eps)
        xs_max = x_sigmoid.clamp(max=1-self.eps)

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


# Custom Loss Wrapper for FastAI
class ChestXrayLoss(Module):
    def __init__(self, loss_type, **kwargs):
        super().__init__()
        if loss_type == 'focal':
            self.loss = FocalLoss(
                alpha=kwargs.get('focal_alpha', 1),
                gamma=kwargs.get('focal_gamma', 2)
            )
        elif loss_type == 'asymmetric':
            self.loss = AsymmetricLoss(
                gamma_neg=kwargs.get('asymmetric_gamma_neg', 4),
                gamma_pos=kwargs.get('asymmetric_gamma_pos', 1)
            )
        elif loss_type == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, preds, targets):
        return self.loss(preds, targets)


# Custom Callback for Momentum Update (placeholder from notebook)
class MomentumUpdateCallback(Callback):
    def __init__(self, warmup_epochs):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def after_batch(self):
        if hasattr(self.learn.model, 'momentum_final_block'):
            # Apply warm-up during the first few epochs
            is_warmup = self.learn.epoch < self.warmup_epochs
            self.learn.model.momentum_final_block.update(
                self.learn.model.final_block, warmup=is_warmup
            )


# Model Wrapper for FastAI
class ModelWrapper(nn.Module):
    """Wrapper to ensure model works with FastAI"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def create_fastai_learner(
    dls,                            # DataLoaders object
    model=None,                     # Pre-instantiated model
    num_classes=14,                 # Number of output classes
    lr=1e-4,                        # Learning rate
    momentum=0.9,                   # Momentum for memory bank
    dropout_rate=0.3,               # Dropout rate for classifier
    mixup=False,                    # Whether to use mixup augmentation
    wd=1e-2,                        # Weight decay
    cbs=None,                       # Additional callbacks
    warmup_epochs=0,                # Number of warm-up epochs
    loss_type='bce',                # Loss type: 'bce', 'focal', or 'asymmetric'
    focal_alpha=1,                  # Focal loss alpha parameter
    focal_gamma=2,                  # Focal loss gamma parameter
    asymmetric_gamma_neg=4,         # Asymmetric loss gamma for negative samples
    asymmetric_gamma_pos=1,         # Asymmetric loss gamma for positive samples
    update_strategy='rarity',       # Memory bank update strategy
    bank_size=512,                  # Memory bank size
    model_name='efficientnet_b0',   # Model architecture
    use_fp16=True                   # Use mixed precision training
):
    """
    Create a FastAI Learner for ChestXrayModel
    Following the exact structure from mem_model.ipynb

    Args:
        dls: FastAI DataLoaders
        model: Pre-instantiated model (if None, will create one)
        num_classes: Number of disease classes
        lr: Learning rate
        momentum: Memory bank momentum
        dropout_rate: Dropout rate
        mixup: Use mixup augmentation
        wd: Weight decay
        cbs: Additional callbacks
        warmup_epochs: Warmup epochs for momentum
        loss_type: Loss function type
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        asymmetric_gamma_neg: ASL gamma for negatives
        asymmetric_gamma_pos: ASL gamma for positives
        update_strategy: Memory bank strategy
        bank_size: Memory bank size
        model_name: Backbone architecture
        use_fp16: Use mixed precision

    Returns:
        learn: FastAI Learner
    """
    # Import model here to avoid circular import
    from models.model import ChestXrayModel

    # Create model if not provided
    if model is None:
        model = ChestXrayModel(
            num_classes=num_classes,
            model_name=model_name,
            dropout_rate=dropout_rate,
            bank_size=bank_size,
            update_strategy=update_strategy,
            rarity_threshold=0.2,
            diversity_weight=0.5,
            memory_momentum=momentum
        )
        print(f"\nCreated model: {model_name}")
        print(f"  Update strategy: {update_strategy}")
        print(f"  Bank size: {bank_size}")
        print(f"  Memory momentum: {momentum}")

    # Prepare default callbacks
    default_cbs = [
        MomentumUpdateCallback(warmup_epochs),  # Custom callback with warm-up
        SaveModelCallback(monitor='valid_loss'),  # Save best model
        EarlyStoppingCallback(monitor='valid_loss', patience=3)  # Early stopping
    ]

    # Add user-specified callbacks
    if cbs is not None:
        if isinstance(cbs, list):
            default_cbs.extend(cbs)
        else:
            default_cbs.append(cbs)

    # Create the learner with custom model and loss
    learn = Learner(
        dls,
        model,
        loss_func=ChestXrayLoss(
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            asymmetric_gamma_neg=asymmetric_gamma_neg,
            asymmetric_gamma_pos=asymmetric_gamma_pos
        ),
        metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()],  # Multi-label metrics
        wd=wd,
        cbs=default_cbs
    )

    # Wrap the model for FastAI compatibility
    learn.model = ModelWrapper(learn.model)

    # Enable mixed precision training if requested
    if use_fp16:
        learn.to_fp16()

    # Add mixup if requested (suitable for multi-label with BCE)
    if mixup:
        learn.add_cb(MixUp())

    # Add progress bar callback
    learn.add_cb(ProgressCallback())

    # Add CSV logger to track metrics
    learn.add_cb(CSVLogger())

    print(f"\nLearner created successfully!")
    print(f"  Loss: {loss_type}")
    print(f"  Weight decay: {wd}")
    print(f"  Mixed precision: {use_fp16}")
    print(f"  Metrics: accuracy_multi, F1ScoreMulti, RocAucMulti")

    return learn


if __name__ == '__main__':
    from data.fastai_data import prepare_chestxray14_dataframe, create_dataloaders

    # Test learner creation
    print("Testing learner creation...")

    data_dir = '/kaggle/input/data'
    train_val_df, disease_labels = prepare_chestxray14_dataframe(data_dir, seed=85, filter_normal=True)
    dls = create_dataloaders(train_val_df, disease_labels, batch_size=64)

    # Create learner
    learn = create_fastai_learner(
        dls,
        num_classes=14,
        loss_type='bce',
        update_strategy='rarity',
        model_name='efficientnet_b0'
    )

    print("\nLearner created successfully!")
    print(f"Model: {learn.model}")
