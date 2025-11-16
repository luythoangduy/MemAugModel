"""
Configuration package for Memory-Augmented Chest X-Ray Classification

Main training now uses YAML configs (configs/experiments/*.yaml).
This module provides backward compatibility with the legacy Python config.
"""

from .config import (
    get_backbone_config,
    get_memory_config,
    get_experiment_config,
    SEED,
    NUM_CLASSES,
    IMAGE_SIZE,
    DISEASE_LABELS,
    BATCH_SIZE_PHASE1,
    BATCH_SIZE_PHASE2
)

__all__ = [
    # Main functions
    'get_backbone_config',
    'get_memory_config',
    'get_experiment_config',

    # Constants
    'SEED',
    'NUM_CLASSES',
    'DISEASE_LABELS',
    'IMAGE_SIZE',
    'BATCH_SIZE_PHASE1',
    'BATCH_SIZE_PHASE2',
]
