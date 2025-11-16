"""
Backbone models for chest X-ray classification
Supports: ResNet50, DenseNet121, EfficientNetV2-S/M/L
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(backbone_name='efficientnet_v2_s', pretrained=True):
    """
    Get backbone model and feature dimension

    Args:
        backbone_name: Name of the backbone architecture
        pretrained: Whether to use pretrained weights

    Returns:
        backbone: Feature extraction layers (excluding final classification layer)
        final_block: Final feature processing block
        feature_dim: Dimension of extracted features
    """

    if backbone_name == 'resnet50':
        base_model = models.resnet50(pretrained=pretrained)

        # Split into backbone and final block
        backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3
        )
        final_block = base_model.layer4
        feature_dim = 2048

    elif backbone_name == 'densenet121':
        base_model = models.densenet121(pretrained=pretrained)

        # Split DenseNet features
        features = list(base_model.features.children())
        backbone = nn.Sequential(*features[:-1])
        final_block = nn.Sequential(features[-1])
        feature_dim = 1024

    elif backbone_name == 'efficientnet_v2_s':
        base_model = models.efficientnet_v2_s(pretrained=pretrained)

        features = list(base_model.features)
        backbone = nn.Sequential(*features[:-1])
        final_block = nn.Sequential(features[-1])
        feature_dim = 1280

    elif backbone_name == 'efficientnet_v2_m':
        base_model = models.efficientnet_v2_m(pretrained=pretrained)

        features = list(base_model.features)
        backbone = nn.Sequential(*features[:-1])
        final_block = nn.Sequential(features[-1])
        feature_dim = 1280

    elif backbone_name == 'efficientnet_v2_l':
        base_model = models.efficientnet_v2_l(pretrained=pretrained)

        features = list(base_model.features)
        backbone = nn.Sequential(*features[:-1])
        final_block = nn.Sequential(features[-1])
        feature_dim = 1280

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                        f"Supported: resnet50, densenet121, efficientnet_v2_s/m/l")

    return backbone, final_block, feature_dim


class BackboneWrapper(nn.Module):
    """
    Wrapper for backbone models to extract features
    """
    def __init__(self, backbone_name='efficientnet_v2_s', pretrained=True):
        super(BackboneWrapper, self).__init__()

        self.backbone_name = backbone_name
        self.backbone, self.final_block, self.feature_dim = get_backbone(
            backbone_name, pretrained
        )

    def forward(self, x):
        """
        Extract features from input images

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: Extracted feature maps [B, C, H', W']
        """
        # Extract features through backbone
        backbone_features = self.backbone(x)

        # Process through final block
        features = self.final_block(backbone_features)

        return features

    def get_feature_dim(self):
        """Return the feature dimension"""
        return self.feature_dim


if __name__ == '__main__':
    # Test different backbones
    print("Testing backbone models...")

    backbones = ['resnet50', 'densenet121', 'efficientnet_v2_s',
                 'efficientnet_v2_m', 'efficientnet_v2_l']

    x = torch.randn(2, 3, 224, 224)

    for backbone_name in backbones:
        print(f"\n{backbone_name}:")
        model = BackboneWrapper(backbone_name, pretrained=False)
        features = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature dim: {model.get_feature_dim()}")
