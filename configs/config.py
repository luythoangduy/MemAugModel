"""
Configuration file for Memory-Augmented Chest X-Ray Classification
Based on the paper: "Mitigating Class Imbalance in Chest X-Ray Classification with Memory-Augmented Models"
"""

# Random seed for reproducibility
SEED = 85

# Dataset paths
DATA_DIR = '/kaggle/input/data'
TRAIN_VAL_LIST = 'train_val_list.txt'
TEST_LIST = 'test_list.txt'
DATA_ENTRY = 'Data_Entry_2017.csv'

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE_PHASE1 = 64  # Phase 1: abnormal images only
BATCH_SIZE_PHASE2 = 128  # Phase 2: full dataset

# Training settings
EPOCHS_FREEZE = 3
EPOCHS_TOTAL = 20
WEIGHT_DECAY = 1e-2
DROPOUT_RATE = 0.3

# Disease labels
DISEASE_LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
NUM_CLASSES = len(DISEASE_LABELS)

# Data split ratios
TRAIN_VAL_RATIO = 0.8  # 80% train, 20% val from train_val set
VAL_PCT = 0.125  # 12.5% validation from training set (1 - 0.875 = 0.125)

# ============== Model Backbone Configurations ==============
BACKBONE_CONFIGS = {
    'resnet50': {
        'name': 'resnet50',
        'pretrained': True,
        'feature_dim': 2048,
        'description': 'ResNet-50 backbone'
    },
    'densenet121': {
        'name': 'densenet121',
        'pretrained': True,
        'feature_dim': 1024,
        'description': 'DenseNet-121 backbone'
    },
    'efficientnet_v2_s': {
        'name': 'efficientnet_v2_s',
        'pretrained': True,
        'feature_dim': 1280,
        'description': 'EfficientNetV2-S backbone (default)'
    },
    'efficientnet_v2_m': {
        'name': 'efficientnet_v2_m',
        'pretrained': True,
        'feature_dim': 1280,
        'description': 'EfficientNetV2-M backbone'
    },
    'efficientnet_v2_l': {
        'name': 'efficientnet_v2_l',
        'pretrained': True,
        'feature_dim': 1280,
        'description': 'EfficientNetV2-L backbone'
    }
}

# Default backbone
DEFAULT_BACKBONE = 'efficientnet_v2_s'

# ============== Memory Bank Configurations ==============
MEMORY_CONFIGS = {
    # Memory enabled/disabled
    'use_memory': True,  # Set to False to disable memory bank

    # Memory bank size
    'bank_size': 512,

    # Update strategies
    'update_strategy': 'rarity',  # Options: 'rarity', 'statistical', 'entropy', 'diversity', 'hybrid', 'fifo', 'reservoir'

    # Rarity threshold for filtering features
    'rarity_threshold': 0.2,  # Features with rarity score < threshold are stored

    # Diversity weight (for hybrid strategy)
    'diversity_weight': 0.5,  # Balance between rarity and diversity

    # Momentum for running statistics
    'memory_momentum': 0.9,

    # Retrieval settings
    'top_k': 3,  # Number of similar features to retrieve
    'normalize_retrieved': True,  # Whether to normalize retrieved features

    # Self-matching threshold
    'self_match_threshold': 0.9999  # Avoid retrieving identical features
}

# Preset memory configurations
MEMORY_PRESETS = {
    'no_memory': {
        'use_memory': False,
        'bank_size': 0,
        'update_strategy': None,
        'description': 'Baseline without memory augmentation'
    },
    'rarity_based': {
        'use_memory': True,
        'bank_size': 512,
        'update_strategy': 'rarity',
        'rarity_threshold': 0.2,
        'top_k': 3,
        'normalize_retrieved': True,
        'description': 'Rarity-based memory update (default from paper)'
    },
    'statistical': {
        'use_memory': True,
        'bank_size': 512,
        'update_strategy': 'statistical',
        'rarity_threshold': 0.2,
        'top_k': 3,
        'normalize_retrieved': True,
        'memory_momentum': 0.9,
        'description': 'Statistical outlier detection with running mean'
    },
    'diversity_based': {
        'use_memory': True,
        'bank_size': 512,
        'update_strategy': 'diversity',
        'rarity_threshold': 0.2,
        'top_k': 3,
        'normalize_retrieved': True,
        'description': 'Diversity-based feature selection'
    },
    'hybrid': {
        'use_memory': True,
        'bank_size': 512,
        'update_strategy': 'hybrid',
        'rarity_threshold': 0.2,
        'diversity_weight': 0.5,
        'top_k': 3,
        'normalize_retrieved': True,
        'description': 'Hybrid rarity + diversity strategy'
    },
    'fifo': {
        'use_memory': True,
        'bank_size': 512,
        'update_strategy': 'fifo',
        'top_k': 3,
        'normalize_retrieved': True,
        'description': 'Simple FIFO memory queue'
    }
}

# ============== Loss Function Configurations ==============
LOSS_CONFIGS = {
    'bce': {
        'name': 'bce',
        'description': 'Binary Cross Entropy Loss'
    },
    'focal': {
        'name': 'focal',
        'alpha': 1,
        'gamma': 2,
        'description': 'Focal Loss for class imbalance'
    },
    'asymmetric': {
        'name': 'asymmetric',
        'gamma_neg': 4,
        'gamma_pos': 1,
        'clip': 0.05,
        'description': 'Asymmetric Loss (ASL)'
    }
}

# Training phase configurations
PHASE1_CONFIG = {
    'loss': 'bce',
    'batch_size': 64,
    'epochs': 3,
    'data_filter': 'abnormal_only',  # Only abnormal images
    'description': 'Phase 1: Train on abnormal images with BCE loss'
}

PHASE2_CONFIG = {
    'loss': 'asymmetric',
    'batch_size': 128,
    'epochs': 5,
    'data_filter': 'all',  # All images including normal
    'lr_range': (2e-5, 8e-5),
    'description': 'Phase 2: Fine-tune on full dataset with ASL'
}

# ============== Experiment Configurations ==============
# Easy configuration for running different experiments
EXPERIMENT_CONFIGS = {
    'baseline_resnet50': {
        'backbone': 'resnet50',
        'memory_preset': 'no_memory',
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'Baseline ResNet-50 without memory'
    },
    'baseline_densenet121': {
        'backbone': 'densenet121',
        'memory_preset': 'no_memory',
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'Baseline DenseNet-121 without memory'
    },
    'baseline_effv2s': {
        'backbone': 'efficientnet_v2_s',
        'memory_preset': 'no_memory',
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'Baseline EfficientNetV2-S without memory'
    },
    'memory_rarity_effv2s': {
        'backbone': 'efficientnet_v2_s',
        'memory_preset': 'rarity_based',
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'EfficientNetV2-S with rarity-based memory (paper default)'
    },
    'memory_hybrid_effv2s': {
        'backbone': 'efficientnet_v2_s',
        'memory_preset': 'hybrid',
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'EfficientNetV2-S with hybrid memory strategy'
    },
    'memory_rarity_resnet50': {
        'backbone': 'resnet50',
        'memory_preset': 'rarity_based',
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'ResNet-50 with rarity-based memory'
    },
    'memory_rarity_densenet121': {
        'backbone': 'densenet121',
        'memory_preset': 'rarity_based',
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'DenseNet-121 with rarity-based memory'
    },
    'memory_normalized_retrieve': {
        'backbone': 'efficientnet_v2_s',
        'memory_preset': 'rarity_based',
        'memory_override': {'normalize_retrieved': True},
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'With normalized retrieval'
    },
    'memory_unnormalized_retrieve': {
        'backbone': 'efficientnet_v2_s',
        'memory_preset': 'rarity_based',
        'memory_override': {'normalize_retrieved': False},
        'loss_phase1': 'bce',
        'loss_phase2': 'asymmetric',
        'description': 'Without normalized retrieval'
    }
}

# ============== Utility Functions ==============
def get_backbone_config(backbone_name=None):
    """Get backbone configuration"""
    if backbone_name is None:
        backbone_name = DEFAULT_BACKBONE
    return BACKBONE_CONFIGS.get(backbone_name, BACKBONE_CONFIGS[DEFAULT_BACKBONE])

def get_memory_config(preset_name='rarity_based', **overrides):
    """Get memory configuration with optional overrides"""
    config = MEMORY_PRESETS.get(preset_name, MEMORY_PRESETS['rarity_based']).copy()
    config.update(overrides)
    return config

def get_experiment_config(experiment_name):
    """Get complete experiment configuration"""
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")

    exp_config = EXPERIMENT_CONFIGS[experiment_name]

    # Build complete configuration
    backbone_cfg = get_backbone_config(exp_config['backbone'])
    memory_cfg = get_memory_config(
        exp_config['memory_preset'],
        **exp_config.get('memory_override', {})
    )

    return {
        'backbone': backbone_cfg,
        'memory': memory_cfg,
        'loss_phase1': exp_config['loss_phase1'],
        'loss_phase2': exp_config['loss_phase2'],
        'description': exp_config['description']
    }

def print_config(config_dict, indent=0):
    """Pretty print configuration"""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
