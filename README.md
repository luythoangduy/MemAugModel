# Memory-Augmented Chest X-Ray Classification

Implementation of **"Mitigating Class Imbalance in Chest X-Ray Classification with Memory-Augmented Models"**

A memory-augmented neural network for multi-label chest X-ray classification on the ChestX-ray14 dataset, using FastAI and two-phase training strategy.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default config (EfficientNetV2-S + Rarity Memory)
python train.py --config configs/experiments/memory_rarity_effv2s.yaml

# Update data_dir in the YAML file first:
# data:
#   data_dir: /path/to/your/chestxray14/data
```

---

## 📁 Project Structure

```
memo_aug/
├── configs/experiments/         # 15 pre-configured YAML experiments
│   ├── baseline_*.yaml          # Baselines (no memory) - 2 configs
│   ├── memory_*.yaml            # Memory-augmented - 8 configs
│   ├── ablation_*.yaml          # Ablation studies - 5 configs
│   └── README.md                # Config documentation
│
├── models/                      # Model implementations
│   ├── backbone.py              # ResNet50, DenseNet121, EfficientNetV2-S/M/L
│   ├── memory_bank.py           # Memory bank with 7 update strategies
│   └── model.py                 # Main ChestXrayModel
│
├── data/                        # Data loading
│   └── fastai_data.py           # FastAI DataBlock for ChestX-ray14
│
├── training/                    # Training utilities
│   ├── fastai_learner.py        # FastAI Learner with memory support
│   └── losses.py                # BCE, Focal, Asymmetric losses
│
├── train.py                     # Main training script
├── evaluate_fastai.py           # Evaluation script
├── TRAINING_GUIDE.md            # Detailed training guide
└── README.md                    # This file
```

---

## ✨ Key Features

### 🧠 Memory-Augmented Architecture
- **Memory Bank**: Stores rare/hard pathological features during training
- **Feature Retrieval**: Top-k similarity-based retrieval with weighted aggregation
- **7 Update Strategies**: Rarity, Diversity, Hybrid, Statistical, Entropy, FIFO, Reservoir

### 🏗️ Model Backbones
- **ResNet50** (2048-dim features)
- **DenseNet121** (1024-dim features)
- **EfficientNetV2-S** (1280-dim, default)
- **EfficientNetV2-M** (1280-dim, larger)

### 📊 Two-Phase Training
**Phase 1**: Abnormal images only
- Batch size: 64, BCE loss
- Freeze backbone → fine-tune 20 epochs
- Memory momentum: 0.9

**Phase 2**: All images (including normal)
- Batch size: 128, BCE/ASL loss
- fit_one_cycle 5 epochs, LR: 2e-5 to 8e-5
- Memory momentum: **0.9999** (critical!)

### 🔧 Configurable Parameters
- **Memory size**: 256, 512, 1024 slots
- **Top-k retrieval**: 1, 3, 5 features
- **Normalization**: Normalize/unnormalize retrieved features
- **Loss functions**: BCE, Focal Loss, Asymmetric Loss

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch + torchvision
- FastAI
- scikit-learn, pandas, numpy
- PyYAML

---

## 🎯 Usage

### Training

All experiments use a single command with YAML configs:

```bash
python train.py --config configs/experiments/<experiment>.yaml
```

**Examples:**

```bash
# Default: EfficientNetV2-S + Rarity Memory
python train.py --config configs/experiments/memory_rarity_effv2s.yaml

# Baseline: No memory augmentation
python train.py --config configs/experiments/baseline_effv2s.yaml

# Different backbones
python train.py --config configs/experiments/memory_rarity_resnet50.yaml
python train.py --config configs/experiments/memory_rarity_densenet121.yaml

# Different memory strategies
python train.py --config configs/experiments/memory_diversity_effv2s.yaml
python train.py --config configs/experiments/memory_hybrid_effv2s.yaml

# Ablation studies
python train.py --config configs/experiments/ablation_topk1.yaml
python train.py --config configs/experiments/ablation_bank256.yaml
```

⚠️ **Before training**: Update `data.data_dir` in the YAML file to your ChestX-ray14 data path.

### 📋 Available Experiments (15 configs)

| Category | Config File | Description |
|----------|-------------|-------------|
| **Baselines** | | |
| | `baseline_effv2s.yaml` | EfficientNetV2-S without memory |
| | `baseline_resnet50.yaml` | ResNet50 without memory |
| **Different Backbones** | | |
| | `memory_rarity_effv2s.yaml` | EfficientNetV2-S + rarity (default) ⭐ |
| | `memory_rarity_effv2m.yaml` | EfficientNetV2-M + rarity (larger) |
| | `memory_rarity_resnet50.yaml` | ResNet50 + rarity |
| | `memory_rarity_densenet121.yaml` | DenseNet121 + rarity |
| **Different Strategies** | | |
| | `memory_diversity_effv2s.yaml` | Diversity-based selection |
| | `memory_hybrid_effv2s.yaml` | Hybrid (rarity + diversity) |
| | `memory_statistical_effv2s.yaml` | Statistical-based selection |
| | `memory_fifo_effv2s.yaml` | FIFO baseline |
| **Ablation Studies** | | |
| | `ablation_no_normalize.yaml` | No retrieval normalization |
| | `ablation_topk1.yaml` | k=1 (vs default k=3) |
| | `ablation_topk5.yaml` | k=5 (vs default k=3) |
| | `ablation_bank256.yaml` | Bank size 256 (vs 512) |
| | `ablation_bank1024.yaml` | Bank size 1024 (vs 512) |

📖 **Detailed guide**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### Evaluation

```bash
python evaluate_fastai.py --model_path ./models/phase2_model.pth --data_dir /path/to/data
```

**Output:**
- Per-class ROC-AUC scores (14 disease categories)
- Mean ROC-AUC score
- Saved predictions (optional)

---

## 🔬 How It Works

### Memory-Augmented Architecture

The model enhances standard CNNs with a learnable memory bank:

```
Input Image → Backbone CNN → Features (z)
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
            Memory Retrieval          Classification
         (Top-k similar features)      Head (14 classes)
                    ↓
          Augmented Features (z')
                    ↓
            Final Predictions
```

**Key mechanisms:**
1. **Update**: Store rare/hard features in memory bank
   - `rarity_score = |‖z‖ - μ| / μ` (Eq. 3-4 in paper)
2. **Retrieval**: Find top-k similar features
   - `similarity = (z · m) / (‖z‖ ‖m‖)` (Eq. 5 in paper)
3. **Aggregation**: Weighted sum of retrieved features
   - `z' = z + Σ(w_j * m_j)` (Eq. 6 in paper)

### Training Strategy

**Two-Phase Approach** (Section IV.E in paper):

| Phase | Data | Batch Size | Loss | Epochs | Memory Momentum | Method |
|-------|------|------------|------|--------|-----------------|--------|
| 1 | Abnormal only | 64 | BCE | 3+20 | 0.9 | `fine_tune` (freeze→unfreeze) |
| 2 | All images | 128 | BCE/ASL | 5 | **0.9999** | `fit_one_cycle` (2e-5→8e-5) |

⚠️ **Phase 2 momentum = 0.9999 is critical** for stable memory with normal images!

---

## 🎨 Customization

### Create Your Own Experiment

1. **Copy existing config:**
   ```bash
   cp configs/experiments/memory_rarity_effv2s.yaml configs/experiments/my_experiment.yaml
   ```

2. **Edit parameters:**
   ```yaml
   experiment_name: my_experiment

   model:
     backbone: resnet50  # or densenet121, efficientnet_v2_m

   memory:
     update_strategy: hybrid  # or diversity, statistical, fifo
     bank_size: 1024  # 256, 512, 1024
     top_k: 5  # 1, 3, 5
     normalize_retrieved: false  # true or false

   phase1:
     batch_size: 32  # adjust for GPU memory

   phase2:
     batch_size: 64  # adjust for GPU memory
   ```

3. **Run training:**
   ```bash
   python train.py --config configs/experiments/my_experiment.yaml
   ```

### Programmatic Usage

```python
from models.model import ChestXrayModel

# Memory-augmented model
model = ChestXrayModel(
    num_classes=14,
    backbone='efficientnet_v2_s',
    use_memory=True,
    update_strategy='hybrid',
    bank_size=512,
    top_k=3
)

# Baseline (no memory)
baseline = ChestXrayModel(
    num_classes=14,
    backbone='resnet50',
    use_memory=False
)
```

---

## 📊 Results

### Best Configuration: EfficientNetV2-S + Rarity Memory

**Performance:**
- 🏆 Mean AUC: **0.8633**
- 📈 Outperforms SynthEnsemble (0.8543) by **+0.009**
- ✅ Wins in **12 out of 14** disease categories

**Top-5 Disease Categories:**
| Disease | AUC |
|---------|-----|
| Hernia | 0.9530 |
| Emphysema | 0.9375 |
| Cardiomegaly | 0.9272 |
| Pneumothorax | 0.9179 |
| Edema | 0.9153 |

### Computational Efficiency

Memory operations add **minimal overhead** (Table II in paper):
- Training: +0.0032 GFLOPs (**0.04% increase**)
- Inference: +0.0033 GFLOPs (**0.11% increase**)

### Memory Bank Mechanisms

**Update** (Eq. 3-4):
```
rarity_score = |‖z‖ - μ| / μ
where μ = (1/n) * Σ‖z_i‖
```

**Retrieval** (Eq. 5-6):
```
similarity(z, m_j) = (z · m_j) / (‖z‖ ‖m_j‖)
z' = z + Σ(w_j * m_j)  # w_j = softmax(similarities)
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{hoang2024mitigating,
  title={Mitigating Class Imbalance in Chest X-Ray Classification with Memory-Augmented Models},
  author={Hoang, Khuong Duy and Nguyen, Huu Duy and Huynh, Cong Viet Ngu},
  booktitle={ISCIT Conference},
  year={2024}
}
```

---

## 📄 License

This project is for **research purposes only**.

---

## 🙏 Acknowledgments

- ChestX-ray14 dataset: [Wang et al., 2017]
- FastAI framework: [Howard & Gugger, 2020]
- Implementation based on the paper methods

---

**For detailed documentation, see:**
- 📖 [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete training guide
- 📋 [configs/experiments/README.md](configs/experiments/README.md) - Config documentation
- 📝 [CHANGELOG.md](CHANGELOG.md) - Version history
