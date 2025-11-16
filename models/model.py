"""
Main Model Architecture for Memory-Augmented Chest X-Ray Classification
Based on the paper: "Mitigating Class Imbalance in Chest X-Ray Classification with Memory-Augmented Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_backbone
from .memory_bank import MemoryBank


class ChestXrayModel(nn.Module):
    """
    Memory-Augmented Chest X-ray Classification Model

    Architecture (Fig. 1 in paper):
    1. Backbone (ResNet50/DenseNet121/EfficientNetV2) extracts features
    2. Global Average Pooling to get compact representation (Eq. 2)
    3. Memory Bank stores and retrieves pathological features
    4. Feature enhancement via memory retrieval (Eq. 6)
    5. Classification head for multi-label prediction
    """

    def __init__(self,
                 num_classes=14,
                 backbone='efficientnet_v2_s',
                 pretrained=True,
                 dropout_rate=0.3,
                 use_memory=True,
                 bank_size=512,
                 update_strategy='rarity',
                 rarity_threshold=0.2,
                 diversity_weight=0.5,
                 memory_momentum=0.9,
                 top_k=3,
                 normalize_retrieved=True):
        """
        Args:
            num_classes: Number of disease classes
            backbone: Backbone architecture name
            pretrained: Use pretrained weights
            dropout_rate: Dropout rate in classifier
            use_memory: Whether to use memory bank
            bank_size: Size of memory bank
            update_strategy: Memory update strategy
            rarity_threshold: Threshold for rarity-based filtering
            diversity_weight: Weight for diversity in hybrid strategy
            memory_momentum: Momentum for running statistics
            top_k: Number of memories to retrieve
            normalize_retrieved: Normalize retrieved features
        """
        super(ChestXrayModel, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_memory = use_memory
        self.top_k = top_k

        # Backbone and Final Block (Eq. 1)
        self.backbone, self.final_block, self.feature_dim = get_backbone(
            backbone, pretrained
        )

        # Memory Bank
        if self.use_memory:
            self.memory_bank = MemoryBank(
                feature_dim=self.feature_dim,
                bank_size=bank_size,
                update_strategy=update_strategy,
                rarity_threshold=rarity_threshold,
                diversity_weight=diversity_weight,
                momentum=memory_momentum,
                normalize_retrieved=normalize_retrieved
            )
        else:
            self.memory_bank = None

        # Classifier (Section III.D in paper)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            logits: Class predictions [B, num_classes]
        """
        # Extract features (Eq. 1)
        backbone_features = self.backbone(x)
        main_features = self.final_block(backbone_features)

        # Global Average Pooling (Eq. 2)
        # z = (1/H'W') * sum(k(h,w))
        pooled_features = F.adaptive_avg_pool2d(main_features, (1, 1)).flatten(1)

        # Memory augmentation
        if self.use_memory and self.memory_bank is not None:
            # Update memory bank during training (Eq. 4)
            if self.training:
                self.memory_bank.update(pooled_features.detach())

            # Retrieve from memory bank (Eq. 5-6)
            # z' = z + sum(w_j * m_j)
            memory_features = self.memory_bank.retrieve(pooled_features, k=self.top_k)
            enhanced_features = pooled_features + memory_features
        else:
            enhanced_features = pooled_features

        # Classification
        logits = self.classifier(enhanced_features)

        return logits

    def get_features(self, x, return_memory=False):
        """
        Extract features (useful for visualization)

        Args:
            x: Input images [B, 3, H, W]
            return_memory: Also return memory-enhanced features

        Returns:
            features: Pooled features [B, D]
            enhanced_features: Memory-enhanced features [B, D] (if return_memory=True)
        """
        backbone_features = self.backbone(x)
        main_features = self.final_block(backbone_features)
        pooled_features = F.adaptive_avg_pool2d(main_features, (1, 1)).flatten(1)

        if return_memory and self.use_memory and self.memory_bank is not None:
            memory_features = self.memory_bank.retrieve(pooled_features, k=self.top_k)
            enhanced_features = pooled_features + memory_features
            return pooled_features, enhanced_features

        return pooled_features

    def get_memory_stats(self):
        """Get memory bank statistics"""
        if self.memory_bank is not None:
            return self.memory_bank.get_memory_stats()
        return None

    def freeze_backbone(self):
        """Freeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.final_block.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.final_block.parameters():
            param.requires_grad = True


def create_model(config):
    """
    Create model from configuration

    Args:
        config: Configuration dict with 'backbone' and 'memory' keys

    Returns:
        model: ChestXrayModel instance
    """
    backbone_cfg = config['backbone']
    memory_cfg = config['memory']

    model = ChestXrayModel(
        num_classes=config.get('num_classes', 14),
        backbone=backbone_cfg['name'],
        pretrained=backbone_cfg.get('pretrained', True),
        dropout_rate=config.get('dropout_rate', 0.3),
        use_memory=memory_cfg.get('use_memory', True),
        bank_size=memory_cfg.get('bank_size', 512),
        update_strategy=memory_cfg.get('update_strategy', 'rarity'),
        rarity_threshold=memory_cfg.get('rarity_threshold', 0.2),
        diversity_weight=memory_cfg.get('diversity_weight', 0.5),
        memory_momentum=memory_cfg.get('memory_momentum', 0.9),
        top_k=memory_cfg.get('top_k', 3),
        normalize_retrieved=memory_cfg.get('normalize_retrieved', True)
    )

    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...")

    # Test with memory
    model_with_memory = ChestXrayModel(
        num_classes=14,
        backbone='efficientnet_v2_s',
        use_memory=True,
        update_strategy='rarity'
    )

    # Test without memory
    model_without_memory = ChestXrayModel(
        num_classes=14,
        backbone='efficientnet_v2_s',
        use_memory=False
    )

    x = torch.randn(4, 3, 224, 224)

    print("\n=== Model with Memory ===")
    model_with_memory.train()
    out = model_with_memory(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Memory stats: {model_with_memory.get_memory_stats()}")

    print("\n=== Model without Memory ===")
    model_without_memory.train()
    out = model_without_memory(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Memory stats: {model_without_memory.get_memory_stats()}")

    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameters (with memory): {count_parameters(model_with_memory):,}")
    print(f"Parameters (without memory): {count_parameters(model_without_memory):,}")
