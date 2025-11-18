# 1-Phase No-Validation Training Configs

## Overview

These configs train models on the **full training set** without validation split, using a warmup + main training schedule.

## Training Schedule

```
Warmup Phase: 5 epochs
  - Freeze backbone
  - Low learning rate (1e-5)
  - Allow classifier head to adapt

Main Training: 10 epochs
  - Unfreeze all layers
  - Normal learning rate (1e-4)
  - Full model training

Total: 15 epochs
```

## Key Differences from Standard Configs

| Aspect | Standard | No-Val |
|--------|----------|--------|
| **Validation Split** | 10% of train | 0% (use all data) |
| **Training Data** | 90% of train | 100% of train |
| **Early Stopping** | Yes (patience=5) | No |
| **Save Best** | Yes (on val_loss) | No (save final) |
| **Schedule** | fine_tune (freeze + unfreeze) | warmup + main |

## Available Configs

### 1. Memory-Augmented (Rarity)
**File**: `memory_rarity_effv2s_1phase_noval.yaml`

- Model: EfficientNetV2-S
- Memory: Rarity-based (512 slots)
- Batch size: 128

### 2. Baseline (No Memory)
**File**: `baseline_effv2s_1phase_noval.yaml`

- Model: EfficientNetV2-S
- Memory: None
- Batch size: 64
- Use as comparison baseline

### 3. Advanced Memory (Loss-Aware Hard Mining)
**File**: `advanced_memory_hard_1phase_noval.yaml`

- Model: EfficientNetV2-S
- Memory: Loss-aware hard sample mining
- Adaptive k retrieval (2-8)
- Batch size: 128

## Usage

```bash
# Memory-augmented with rarity
python train.py --config configs/experiments/memory_rarity_effv2s_1phase_noval.yaml

# Baseline without memory
python train.py --config configs/experiments/baseline_effv2s_1phase_noval.yaml

# Advanced loss-aware memory
python train.py --config configs/experiments/advanced_memory_hard_1phase_noval.yaml
```

## Evaluation

Since there's no validation set, evaluate on test set only:

```bash
python train.py \
  --config configs/experiments/memory_rarity_effv2s_1phase_noval.yaml \
  --evaluate \
  --model_path models/rarity_1phase_noval
```

## When to Use No-Val Training

**Use when**:
- ✅ Dataset is small → maximize training data
- ✅ You have separate test set for evaluation
- ✅ Training for fixed epochs (not early stopping)
- ✅ You know good hyperparameters already

**Don't use when**:
- ❌ Need to tune hyperparameters
- ❌ Risk of overfitting
- ❌ No separate test set

## Expected Results

Compared to standard 2-phase with validation:

| Config | Training Data | Epochs | Expected AUC |
|--------|--------------|--------|--------------|
| 2-phase + val | 90% train | 20+5 | 0.78-0.80 |
| 1-phase no-val (baseline) | 100% train | 15 | 0.79-0.81 |
| 1-phase no-val (memory) | 100% train | 15 | 0.80-0.82 |
| 1-phase no-val (advanced) | 100% train | 15 | 0.81-0.83 |

**+10% training data** may give **~1-2%** AUC improvement.

## Technical Details

### Warmup Schedule Implementation

```python
# Warmup: Freeze backbone
learn.freeze()
learn.fit_one_cycle(warmup_epochs=5, lr_max=1e-5)

# Main: Unfreeze all
learn.unfreeze()
learn.fit_one_cycle(total_epochs=10, lr_max=1e-4)
```

### No Validation DataLoader

```python
# valid_pct=0.0 → All data goes to training
dls = create_dataloaders(
    train_df,
    batch_size=128,
    valid_pct=0.0  # No validation split
)
```

### No Early Stopping

```yaml
phase1:
  early_stopping_patience: null  # Disabled
  save_best: false              # Save final, not best
```

## Validation

Run test script to verify configs:

```bash
python test_1phase_noval_config.py
```

Should output:
```
[SUCCESS] ALL CONFIGS VALIDATED
```
