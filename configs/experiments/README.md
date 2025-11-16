# Experiment Configuration Files

This directory contains complete YAML configuration files for different experiments.

## Usage

Run training with any experiment config:

```bash
python train.py --config configs/experiments/memory_rarity_effv2s.yaml
```

## Available Experiments

### Baseline (No Memory)
- **`baseline_effv2s.yaml`**: EfficientNetV2-S without memory augmentation

### Memory-Augmented Models

#### Different Backbones
- **`memory_rarity_effv2s.yaml`**: EfficientNetV2-S with rarity memory (default)
- **`memory_rarity_effv2m.yaml`**: EfficientNetV2-M with rarity memory (larger model)
- **`memory_rarity_resnet50.yaml`**: ResNet50 with rarity memory
- **`memory_rarity_densenet121.yaml`**: DenseNet121 with rarity memory

#### Different Memory Strategies
- **`memory_rarity_effv2s.yaml`**: Rarity-based strategy (default)
- **`memory_diversity_effv2s.yaml`**: Diversity-based strategy
- **`memory_hybrid_effv2s.yaml`**: Hybrid strategy (rarity + diversity)

### Ablation Studies

#### Retrieval Settings
- **`ablation_no_normalize.yaml`**: Without normalization of retrieved features
- **`ablation_topk1.yaml`**: Retrieve only 1 memory feature (vs default 3)
- **`ablation_topk5.yaml`**: Retrieve 5 memory features (vs default 3)

#### Memory Bank Size
- **`ablation_bank256.yaml`**: Smaller bank (256 slots vs default 512)
- **`ablation_bank1024.yaml`**: Larger bank (1024 slots vs default 512)

## Config File Structure

Each config file contains:

```yaml
experiment_name: memory_rarity_effv2s
description: "Brief description"

# Model configuration
model:
  num_classes: 14
  backbone: efficientnet_v2_s
  pretrained: true
  dropout_rate: 0.3

# Memory bank configuration
memory:
  use_memory: true
  bank_size: 512
  update_strategy: rarity
  top_k: 3
  normalize_retrieved: true

# Data configuration
data:
  data_dir: /kaggle/input/data
  image_size: [224, 224]
  seed: 85
  valid_pct: 0.125

# Phase 1: Abnormal images only
phase1:
  filter_normal: true
  batch_size: 64
  loss: bce
  freeze_epochs: 3
  total_epochs: 20
  memory_momentum: 0.9

# Phase 2: All images
phase2:
  filter_normal: false
  batch_size: 128
  loss: bce
  epochs: 5
  lr_min: 0.00002
  lr_max: 0.00008
  memory_momentum: 0.9999  # Higher momentum!
  load_from_phase1: true

# Evaluation
evaluation:
  batch_size: 128
  save_predictions: true
```

## Key Parameters

### Memory Strategy
- **`rarity`**: Select rare/hard examples based on prediction confidence
- **`diversity`**: Select diverse examples to cover feature space
- **`hybrid`**: Combine rarity and diversity (use `diversity_weight`)
- **`statistical`**: Statistical-based selection
- **`entropy`**: Entropy-based selection
- **`fifo`**: First-in-first-out (baseline)
- **`reservoir`**: Reservoir sampling (baseline)

### Memory Bank Size
- **256**: Smaller bank, faster updates, less coverage
- **512**: Default, good balance
- **1024**: Larger bank, more coverage, slower updates

### Top-k Retrieval
- **1**: Only most similar feature
- **3**: Default, 3 most similar features
- **5**: More context from memory

### Normalize Retrieved
- **true**: Normalize retrieved features (default, better gradient flow)
- **false**: No normalization (ablation study)

### Phase 2 Memory Momentum
- **0.9999**: Very high (default), slow memory updates in Phase 2
- This is CRITICAL for good performance with normal images!

## Creating New Experiments

Copy an existing config and modify the parameters:

```bash
cp configs/experiments/memory_rarity_effv2s.yaml configs/experiments/my_experiment.yaml
# Edit my_experiment.yaml
python train.py --config configs/experiments/my_experiment.yaml
```

## Important Notes

1. **Data Directory**: Update `data.data_dir` to your local path
2. **Batch Size**: Adjust based on your GPU memory:
   - EfficientNetV2-S: bs=64/128 (default)
   - EfficientNetV2-M: bs=32/64 (larger model)
   - ResNet50/DenseNet121: bs=64/128
3. **Phase 2 Momentum**: Always use 0.9999 for Phase 2 (matches notebook)
4. **Save Names**: Models are saved with names from `phase1.save_name` and `phase2.save_name`
