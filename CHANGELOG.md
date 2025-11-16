# Changelog

## YAML Configuration System (Latest)

### New Features

#### 1. Complete YAML Config System
- **New training script**: `train.py` now accepts `--config` argument
- **Self-contained configs**: Each experiment has a complete YAML file
- **No hardcoded settings**: All parameters externalized to YAML

Usage:
```bash
python train.py --config configs/experiments/memory_rarity_effv2s.yaml
```

#### 2. 15 Experiment Configurations Created

**Baselines (2):**
- `baseline_effv2s.yaml`: EfficientNetV2-S without memory
- `baseline_resnet50.yaml`: ResNet50 without memory

**Memory-Augmented - Different Backbones (4):**
- `memory_rarity_effv2s.yaml`: EfficientNetV2-S (default)
- `memory_rarity_effv2m.yaml`: EfficientNetV2-M (larger)
- `memory_rarity_resnet50.yaml`: ResNet50
- `memory_rarity_densenet121.yaml`: DenseNet121

**Memory-Augmented - Different Strategies (4):**
- `memory_diversity_effv2s.yaml`: Diversity-based
- `memory_hybrid_effv2s.yaml`: Hybrid (rarity + diversity)
- `memory_statistical_effv2s.yaml`: Statistical-based
- `memory_fifo_effv2s.yaml`: FIFO baseline

**Ablation Studies (5):**
- `ablation_no_normalize.yaml`: No retrieval normalization
- `ablation_topk1.yaml`: k=1 retrieval
- `ablation_topk5.yaml`: k=5 retrieval
- `ablation_bank256.yaml`: Smaller bank (256)
- `ablation_bank1024.yaml`: Larger bank (1024)

#### 3. Complete Documentation
- **TRAINING_GUIDE.md**: Comprehensive training guide
  - Quick start examples
  - All 15 experiments documented
  - Customization instructions
  - Troubleshooting tips

- **configs/experiments/README.md**: Experiment configs documentation
  - File structure explanation
  - Parameter descriptions
  - How to create new configs

- **README.md**: Updated main README
  - New YAML workflow prominently featured
  - All experiments listed
  - Legacy workflow still documented

### Implementation Details

#### Config File Structure
Each YAML config contains:
```yaml
experiment_name: ...
description: ...

model:
  num_classes: 14
  backbone: ...
  pretrained: true
  dropout_rate: 0.3

memory:
  use_memory: true/false
  bank_size: ...
  update_strategy: ...
  top_k: ...
  normalize_retrieved: ...

data:
  data_dir: ...
  image_size: [224, 224]
  seed: 85
  valid_pct: 0.125

phase1:
  filter_normal: true
  batch_size: ...
  loss: bce
  freeze_epochs: 3
  total_epochs: 20
  memory_momentum: 0.9
  save_name: ...

phase2:
  filter_normal: false
  batch_size: ...
  loss: bce/asymmetric
  epochs: 5
  lr_min: 0.00002
  lr_max: 0.00008
  memory_momentum: 0.9999  # CRITICAL!
  load_from_phase1: true
  save_name: ...

evaluation:
  batch_size: 128
  save_predictions: true
```

#### Training Script Updates
- `train.py` completely rewritten:
  - Accepts `--config` argument (required)
  - Loads YAML config with `yaml.safe_load()`
  - Extracts all settings from config
  - Implements two-phase training exactly as in notebook
  - Saves predictions with experiment name

### Key Features Preserved

✅ **Two-Phase Training**:
- Phase 1: Abnormal images only, bs=64, BCE, freeze→fine_tune
- Phase 2: All images, bs=128, lr_range=(2e-5, 8e-5), **momentum=0.9999**

✅ **FastAI Integration**:
- DataBlock for data loading
- Learner with callbacks
- fine_tune for Phase 1
- fit_one_cycle for Phase 2
- Mixed precision (fp16)

✅ **Memory Augmentation**:
- Multiple strategies (rarity, diversity, hybrid, etc.)
- Configurable bank size, top-k, normalization
- Different momentum for Phase 1 (0.9) and Phase 2 (0.9999)

✅ **Evaluation**:
- ROC-AUC per class and mean
- Automatic prediction saving
- Validation set evaluation

### Backward Compatibility

✅ **Old config system still works**:
- `configs/config.py` unchanged
- `train_fastai.py` unchanged
- Can still use `--experiment` flag

✅ **All existing code functional**:
- Models unchanged
- Data loading unchanged
- Memory bank unchanged
- Losses unchanged

### Migration Path

**From old to new:**
```bash
# Old way
python train_fastai.py --experiment memory_rarity_effv2s --data_dir /path/to/data

# New way (recommended)
python train.py --config configs/experiments/memory_rarity_effv2s.yaml
```

### Benefits of New System

1. **Complete control**: Every parameter in one place
2. **Reproducible**: Config file is self-documenting
3. **Flexible**: Easy to create new experiments
4. **Shareable**: Send config file = send exact experiment setup
5. **Version controlled**: Track config changes in git
6. **No code changes**: Modify config, not code

### Files Modified
- ✅ `train.py`: Complete rewrite for YAML support
- ✅ `README.md`: Updated usage section
- ✅ Created: `TRAINING_GUIDE.md`
- ✅ Created: `configs/experiments/README.md`
- ✅ Created: 15 experiment YAML files

### Files Unchanged (Backward Compatible)
- ✅ `train_fastai.py`: Legacy training script
- ✅ `configs/config.py`: Python config system
- ✅ `configs/__init__.py`: Config API
- ✅ All model, data, training code

## Previous Versions

### Modular Config System (Python)
- Created modular Python configs
- Separate files for backbones, memory, training
- `configs/backbones.py`, `configs/memory.py`, etc.
- **Replaced by YAML system**

### Monolithic Config (Initial)
- Single `configs/config.py` with all experiments
- Used dictionary-based configuration
- Still present for backward compatibility

## Migration Notes

### To use new YAML system:
1. Choose an experiment from `configs/experiments/`
2. Update `data.data_dir` to your local path
3. Run: `python train.py --config configs/experiments/[name].yaml`

### To create custom experiment:
1. Copy existing config: `cp configs/experiments/memory_rarity_effv2s.yaml my_exp.yaml`
2. Edit parameters in `my_exp.yaml`
3. Run: `python train.py --config my_exp.yaml`

### To use old system:
```bash
python train_fastai.py --experiment memory_rarity_effv2s --data_dir /path/to/data
```
