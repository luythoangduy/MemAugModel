"""
Configuration package for Memory-Augmented Chest X-Ray Classification
Now using YAML configuration files!
"""

from .yaml_loader import (
    get_backbone_config,
    get_memory_config,
    get_experiment_config,
    SEED,
    NUM_CLASSES,
    IMAGE_SIZE,
    DISEASE_LABELS,
    BATCH_SIZE_PHASE1,
    BATCH_SIZE_PHASE2,
    _loader as config_loader
)

__all__ = [
    # Main functions
    'get_backbone_config',
    'get_memory_config',
    'get_experiment_config',

    # Base configs
    'SEED',
    'NUM_CLASSES',
    'DISEASE_LABELS',
    'IMAGE_SIZE',
    'BATCH_SIZE_PHASE1',
    'BATCH_SIZE_PHASE2',

    # Config loader (for advanced usage)
    'config_loader',
]
