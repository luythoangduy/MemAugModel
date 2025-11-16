"""
Main training script with YAML config support

Usage:
    python train.py --config configs/experiments/memory_rarity_effv2s.yaml
"""

import argparse
import random
import numpy as np
import torch
import os
import yaml

from fastai.vision.all import *
from data.fastai_data import prepare_chestxray14_dataframe, create_dataloaders
from training.fastai_learner import create_fastai_learner


def seed_everything(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Memory-Augmented Chest X-Ray Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment YAML config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract configurations
    exp_name = config['experiment_name']
    model_cfg = config['model']
    memory_cfg = config['memory']
    data_cfg = config['data']
    phase1_cfg = config['phase1']
    phase2_cfg = config['phase2']
    eval_cfg = config['evaluation']

    # Set random seed
    seed_everything(data_cfg['seed'])

    # Print experiment info
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"{'='*60}\n")

    print("Configuration:")
    print(f"  Backbone: {model_cfg['backbone']}")
    print(f"  Memory: {memory_cfg['use_memory']}")
    if memory_cfg['use_memory']:
        print(f"    - Strategy: {memory_cfg['update_strategy']}")
        print(f"    - Bank size: {memory_cfg['bank_size']}")
        print(f"    - Top-k: {memory_cfg['top_k']}")
        print(f"    - Normalize retrieved: {memory_cfg.get('normalize_retrieved', True)}")
    print(f"  Loss Phase 1: {phase1_cfg['loss']}")
    print(f"  Loss Phase 2: {phase2_cfg['loss']}")
    print()

    # ========== PHASE 1: Abnormal images only ==========
    print("\n" + "="*60)
    print("PHASE 1: Training on abnormal images")
    print("="*60)

    # Prepare data for Phase 1
    train_val_df_phase1, disease_labels = prepare_chestxray14_dataframe(
        data_cfg['data_dir'],
        seed=data_cfg['seed'],
        filter_normal=phase1_cfg['filter_normal']
    )

    dls_phase1 = create_dataloaders(
        train_val_df_phase1,
        disease_labels,
        batch_size=phase1_cfg['batch_size'],
        valid_pct=data_cfg.get('valid_pct', 0.125),
        seed=data_cfg['seed']
    )

    # Create callbacks for Phase 1
    cbs_phase1 = [
        SaveModelCallback(
            monitor='valid_loss',
            min_delta=phase1_cfg.get('min_delta', 0.0001),
            with_opt=True
        ),
        EarlyStoppingCallback(
            monitor='valid_loss',
            min_delta=phase1_cfg.get('min_delta', 0.001),
            patience=phase1_cfg.get('early_stopping_patience', 5)
        ),
        ShowGraphCallback()
    ]

    # Create learner for Phase 1
    learn = create_fastai_learner(
        dls_phase1,
        num_classes=model_cfg['num_classes'],
        cbs=cbs_phase1,
        loss_type=phase1_cfg['loss'],
        update_strategy=memory_cfg.get('update_strategy', 'rarity') if memory_cfg['use_memory'] else 'rarity',
        bank_size=memory_cfg.get('bank_size', 512) if memory_cfg['use_memory'] else 0,
        top_k=memory_cfg.get('top_k', 3),
        normalize_retrieved=memory_cfg.get('normalize_retrieved', True),
        rarity_threshold=memory_cfg.get('rarity_threshold', 0.2),
        diversity_weight=memory_cfg.get('diversity_weight', 0.5),
        momentum=phase1_cfg.get('memory_momentum', 0.9),
        model_name=model_cfg['backbone'],
        use_fp16=True
    )

    # LR Finder
    if phase1_cfg.get('use_lr_finder', False):
        print("\nRunning LR Finder...")
        lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
        print(f'Suggested learning rate (minimum): {lrs.minimum}')
        print(f'Suggested learning rate (valley): {lrs.valley}')
        base_lr = lrs.valley
    else:
        base_lr = phase1_cfg.get('lr', 1e-4)

    # Training Phase 1: fine_tune
    print(f"\nStarting Phase 1 training...")
    print(f"  Freeze epochs: {phase1_cfg['freeze_epochs']}")
    print(f"  Total epochs: {phase1_cfg['total_epochs']}")
    print(f"  Base LR: {base_lr}")
    print(f"  Memory momentum: {phase1_cfg.get('memory_momentum', 0.9)}")

    learn.fine_tune(
        freeze_epochs=phase1_cfg['freeze_epochs'],
        epochs=phase1_cfg['total_epochs'],
        base_lr=base_lr
    )

    # Save Phase 1 model
    phase1_save_name = phase1_cfg.get('save_name', 'phase1_model')
    learn.save(phase1_save_name)
    print(f"\nPhase 1 model saved as: {phase1_save_name}")

    # ========== PHASE 2: All images ==========
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning on all images (including normal)")
    print("="*60)

    # Prepare data for Phase 2
    train_val_df_phase2, disease_labels = prepare_chestxray14_dataframe(
        data_cfg['data_dir'],
        seed=data_cfg['seed'],
        filter_normal=phase2_cfg['filter_normal']
    )

    dls_phase2 = create_dataloaders(
        train_val_df_phase2,
        disease_labels,
        batch_size=phase2_cfg['batch_size'],
        valid_pct=data_cfg.get('valid_pct', 0.125),
        seed=data_cfg['seed']
    )

    # Create callbacks for Phase 2
    cbs_phase2 = [
        SaveModelCallback(
            monitor='valid_loss',
            min_delta=phase2_cfg.get('min_delta', 0.0001),
            with_opt=True
        ),
        EarlyStoppingCallback(
            monitor='valid_loss',
            min_delta=phase2_cfg.get('min_delta', 0.001),
            patience=phase2_cfg.get('early_stopping_patience', 5)
        ),
        ShowGraphCallback()
    ]

    # Create new learner for Phase 2
    learn_phase2 = create_fastai_learner(
        dls_phase2,
        num_classes=model_cfg['num_classes'],
        cbs=cbs_phase2,
        loss_type=phase2_cfg['loss'],
        update_strategy=memory_cfg.get('update_strategy', 'rarity') if memory_cfg['use_memory'] else 'rarity',
        bank_size=memory_cfg.get('bank_size', 512) if memory_cfg['use_memory'] else 0,
        top_k=memory_cfg.get('top_k', 3),
        normalize_retrieved=memory_cfg.get('normalize_retrieved', True),
        rarity_threshold=memory_cfg.get('rarity_threshold', 0.2),
        diversity_weight=memory_cfg.get('diversity_weight', 0.5),
        momentum=phase2_cfg.get('memory_momentum', 0.9999),  # Higher momentum for Phase 2!
        model_name=model_cfg['backbone'],
        use_fp16=True
    )

    # Load Phase 1 weights if requested
    if phase2_cfg.get('load_from_phase1', True):
        checkpoint_name = phase2_cfg.get('phase1_checkpoint', phase1_save_name)
        learn_phase2.load(checkpoint_name)
        print(f"\nLoaded Phase 1 model: {checkpoint_name}")

    # Unfreeze and train with lower LR
    learn_phase2.unfreeze()

    lr_min = phase2_cfg.get('lr_min', 2e-5)
    lr_max = phase2_cfg.get('lr_max', 8e-5)
    epochs = phase2_cfg.get('epochs', 5)

    print("\nStarting Phase 2 training...")
    print(f"  Epochs: {epochs}")
    print(f"  LR range: {lr_min} to {lr_max}")
    print(f"  Memory momentum: {phase2_cfg.get('memory_momentum', 0.9999)}")

    learn_phase2.fit_one_cycle(epochs, slice(lr_min, lr_max))

    # Save Phase 2 model
    phase2_save_name = phase2_cfg.get('save_name', 'phase2_model')
    learn_phase2.save(phase2_save_name)
    print(f"\nPhase 2 model saved as: {phase2_save_name}")

    # ========== EVALUATION ==========
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    from sklearn.metrics import roc_auc_score

    def get_roc_auc(learner, disease_labels):
        """Evaluate model on validation set"""
        learner.model.eval()
        preds, y_test = learner.get_preds(ds_idx=1)  # ds_idx=1 for validation set

        roc_auc = roc_auc_score(y_test, preds)

        scores = []
        for i in range(model_cfg['num_classes']):
            try:
                label_roc_auc_score = roc_auc_score(y_test[:, i], preds[:, i])
                scores.append(label_roc_auc_score)
            except:
                scores.append(0.0)

        print('\nROC AUC Labels:')
        for label, score in zip(disease_labels, scores):
            print(f'  {label:20s}: {score:.4f}')

        print(f'\nMean AUC Score: {roc_auc:.4f}')

        return {
            'roc_auc': roc_auc,
            'per_class_auc': scores,
            'preds': preds,
            'y_test': y_test
        }

    # Evaluate final model
    results = get_roc_auc(learn_phase2, disease_labels)

    # Save predictions if requested
    if eval_cfg.get('save_predictions', True):
        pred_save_path = f'{exp_name}_predictions.pt'
        torch.save(results['preds'], pred_save_path)
        print(f"\nPredictions saved to: {pred_save_path}")

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
