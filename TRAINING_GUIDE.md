# Training Guide

This guide explains how to train models using the YAML configuration system.

## Quick Start

```bash
# Train with default experiment (rarity-based memory)
python train.py --config configs/experiments/memory_rarity_effv2s.yaml

# Train baseline without memory
python train.py --config configs/experiments/baseline_effv2s.yaml

# Train with different backbone
python train.py --config configs/experiments/memory_rarity_resnet50.yaml
```

## Training Workflow

The training script implements the two-phase approach from the paper:

### Phase 1: Learn from Abnormal Images
- **Data**: Only abnormal images (filter_normal=true)
- **Batch size**: 64
- **Loss**: Binary Cross-Entropy (BCE)
- **Training**:
  - Freeze backbone for 3 epochs
  - Fine-tune for 20 epochs total
  - Use LR finder to determine optimal learning rate
- **Memory momentum**: 0.9 (normal momentum)

### Phase 2: Fine-tune with All Images
- **Data**: All images including normal (filter_normal=false)
- **Batch size**: 128
- **Loss**: BCE or Asymmetric Loss (ASL)
- **Training**:
  - Load weights from Phase 1
  - Unfreeze all layers
  - fit_one_cycle for 5 epochs with lr_range=(2e-5, 8e-5)
- **Memory momentum**: 0.9999 ⚠️ **CRITICAL!**

> **Important**: Phase 2 uses very high momentum (0.9999) for the memory bank. This is critical for maintaining stable memory representations when normal images are introduced.

## Available Experiments

### 1. Baseline Models (No Memory)
```bash
python train.py --config configs/experiments/baseline_effv2s.yaml
python train.py --config configs/experiments/baseline_resnet50.yaml
```

### 2. Memory-Augmented Models

#### Different Backbones
```bash
# EfficientNetV2-S (default, balanced speed/accuracy)
python train.py --config configs/experiments/memory_rarity_effv2s.yaml

# EfficientNetV2-M (larger, better accuracy)
python train.py --config configs/experiments/memory_rarity_effv2m.yaml

# ResNet50 (classic architecture)
python train.py --config configs/experiments/memory_rarity_resnet50.yaml

# DenseNet121 (dense connections)
python train.py --config configs/experiments/memory_rarity_densenet121.yaml
```

#### Different Memory Strategies
```bash
# Rarity-based (default from paper)
python train.py --config configs/experiments/memory_rarity_effv2s.yaml

# Diversity-based
python train.py --config configs/experiments/memory_diversity_effv2s.yaml

# Hybrid (rarity + diversity)
python train.py --config configs/experiments/memory_hybrid_effv2s.yaml

# Statistical-based
python train.py --config configs/experiments/memory_statistical_effv2s.yaml

# FIFO (baseline)
python train.py --config configs/experiments/memory_fifo_effv2s.yaml
```

### 3. Ablation Studies

#### Retrieval Settings
```bash
# Without normalization
python train.py --config configs/experiments/ablation_no_normalize.yaml

# Different top-k values
python train.py --config configs/experiments/ablation_topk1.yaml   # k=1
python train.py --config configs/experiments/ablation_topk5.yaml   # k=5
```

#### Memory Bank Size
```bash
# Smaller bank (256)
python train.py --config configs/experiments/ablation_bank256.yaml

# Larger bank (1024)
python train.py --config configs/experiments/ablation_bank1024.yaml
```

## Customizing Experiments

### Create New Config File

1. Copy an existing config:
```bash
cp configs/experiments/memory_rarity_effv2s.yaml configs/experiments/my_experiment.yaml
```

2. Edit the config file:
```yaml
experiment_name: my_experiment
description: "My custom experiment"

model:
  backbone: efficientnet_v2_s  # or resnet50, densenet121, etc.

memory:
  use_memory: true
  update_strategy: rarity  # or diversity, hybrid, statistical, fifo
  bank_size: 512  # 256, 512, 1024
  top_k: 3  # 1, 3, 5
  normalize_retrieved: 'both'  # or false
```

3. Run training:
```bash
python train.py --config configs/experiments/my_experiment.yaml
```

### Key Parameters to Experiment With

| Parameter | Options | Notes |
|-----------|---------|-------|
| `model.backbone` | `efficientnet_v2_s`, `efficientnet_v2_m`, `resnet50`, `densenet121` | Larger = better accuracy but slower |
| `memory.update_strategy` | `rarity`, `diversity`, `hybrid`, `statistical`, `fifo` | Rarity is default from paper |
| `memory.bank_size` | 256, 512, 1024 | Larger = more memory coverage |
| `memory.top_k` | 1, 3, 5 | Number of retrieved features |
| `memory.normalize_retrieved` | true, false | Normalization helps gradient flow |
| `phase1.batch_size` | 32, 64 | Adjust based on GPU memory |
| `phase2.batch_size` | 64, 128 | Adjust based on GPU memory |
| `phase2.memory_momentum` | 0.9999 | **Keep at 0.9999!** |

## Output Files

After training, you'll get:

1. **Phase 1 model**: `{save_name}_phase1.pth` (in FastAI models directory)
2. **Phase 2 model**: `{save_name}_phase2.pth` (final model)
3. **Predictions**: `{experiment_name}_predictions.pt` (validation set predictions)

## Evaluation Metrics

The script automatically evaluates on the validation set and reports:
- **Mean ROC-AUC**: Average across all disease classes
- **Per-class ROC-AUC**: Individual scores for each of 14 diseases

```
ROC AUC Labels:
  Atelectasis        : 0.8234
  Cardiomegaly       : 0.8956
  ...

Mean AUC Score: 0.8345
```

## Tips for Good Results

1. **Always use Phase 2 momentum = 0.9999**: This is critical!
2. **Use LR finder in Phase 1**: Set `phase1.use_lr_finder: true`
3. **Monitor validation loss**: Training includes early stopping
4. **Start with default config**: Use `memory_rarity_effv2s.yaml` as baseline
5. **Adjust batch sizes for your GPU**:
   - 8GB VRAM: EfficientNetV2-S with bs=32/64
   - 16GB VRAM: EfficientNetV2-S with bs=64/128
   - 24GB VRAM: EfficientNetV2-M with bs=32/64

## Troubleshooting

### Out of Memory Error
Reduce batch sizes in config:
```yaml
phase1:
  batch_size: 32  # reduce from 64
phase2:
  batch_size: 64  # reduce from 128
```

### Data Directory Not Found
Update data path in config:
```yaml
data:
  data_dir: /path/to/your/data  # update this
```

### Poor Performance
Check:
1. Phase 2 momentum is 0.9999
2. Using pretrained weights (`model.pretrained: true`)
3. Sufficient training epochs (20 for Phase 1, 5 for Phase 2)
4. Data augmentation is enabled (built into FastAI DataBlock)

## Legacy Training Script

The old training script (`train_fastai.py`) still works with the modular config system:

```bash
python train_fastai.py --experiment memory_rarity_effv2s --data_dir /path/to/data
```

However, the new YAML-based approach (`train.py`) is recommended for better flexibility.

## Next Steps

1. Train baseline: `python train.py --config configs/experiments/baseline_effv2s.yaml`
2. Train with memory: `python train.py --config configs/experiments/memory_rarity_effv2s.yaml`
3. Compare results
4. Try different backbones or memory strategies
5. Run ablation studies to understand contribution of each component
