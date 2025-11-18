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
from data.fastai_data import prepare_chestxray14_dataframe, create_dataloaders, get_validation_split
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
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate mode: load model and run evaluation only')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint for evaluation (default: uses phase2_model from config)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract configurations
    exp_name = config['experiment_name']
    model_cfg = config['model']
    memory_cfg = config['memory']
    data_cfg = config['data']
    phase1_cfg = config['phase1']
    phase2_cfg = config.get('phase2', None)  # Optional for 1-phase configs
    eval_cfg = config['evaluation']

    # Determine if this is 1-phase or 2-phase training
    is_single_phase = phase2_cfg is None

    # Set random seed
    seed_everything(data_cfg['seed'])

    # Print experiment info
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"{'='*60}\n")

    print("Configuration:")
    print(f"  Training mode: {'Single-phase' if is_single_phase else 'Two-phase'}")
    print(f"  Backbone: {model_cfg['backbone']}")
    print(f"  Memory: {memory_cfg['use_memory']}")
    if memory_cfg['use_memory']:
        print(f"    - Strategy: {memory_cfg['update_strategy']}")
        print(f"    - Bank size: {memory_cfg['bank_size']}")
        print(f"    - Top-k: {memory_cfg['top_k']}")
        print(f"    - Normalize retrieved: {memory_cfg.get('normalize_retrieved', True)}")
    print(f"  Loss Phase 1: {phase1_cfg['loss']}")
    if not is_single_phase:
        print(f"  Loss Phase 2: {phase2_cfg['loss']}")
    print()

    # ========== EVALUATION MODE ==========
    if args.evaluate:
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)

        # Determine model path
        if args.model_path:
            model_path = args.model_path
        elif is_single_phase:
            model_path = phase1_cfg.get('save_name', 'phase1_model')
        else:
            model_path = phase2_cfg.get('save_name', 'phase2_model')
        print(f"Model checkpoint: {model_path}")

        # Load test data
        print("\nLoading test data...")
        _, disease_labels, test_df = prepare_chestxray14_dataframe(
            data_cfg['data_dir'],
            seed=data_cfg['seed'],
            filter_normal=False  # Use all test images
        )

        # IMPORTANT: Reset index to ensure consecutive indices 0, 1, 2, ...
        test_df = test_df.reset_index(drop=True)
        print(f"Test set: {len(test_df)} images")

        # Create test dataloader using fastai components
        from fastai.vision.all import DataBlock, ImageBlock, MultiCategoryBlock, Resize, Normalize, imagenet_stats

        item_transforms = [Resize((224, 224))]
        batch_transforms = [Normalize.from_stats(*imagenet_stats)]

        def get_x(row): return row['Paths']
        def get_y(row): return row[disease_labels].tolist()

        # Custom splitter: minimal train set (1 sample), rest in validation
        def test_splitter(items):
            # Return (train_indices, val_indices)
            # Put first sample in train (dummy), rest in validation for testing
            # FastAI requires at least 1 sample in train set
            return ([0], list(range(len(items))))

        dblock = DataBlock(
            blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=disease_labels)),
            splitter=test_splitter,  # All data in validation split
            get_x=get_x,
            get_y=get_y,
            item_tfms=item_transforms,
            batch_tfms=batch_transforms
        )

        dls_test = dblock.dataloaders(test_df, bs=eval_cfg.get('batch_size', 64))

        print(f"\nTest dataloader created:")
        print(f"  Test samples: {len(dls_test.valid_ds)}")
        print(f"  Test batches: {len(dls_test.valid)}")
        print(f"  Batch size: {eval_cfg.get('batch_size', 64)}")

        # Create learner
        learn = create_fastai_learner(
            dls_test,
            num_classes=model_cfg['num_classes'],
            cbs=[],
            loss_type=phase2_cfg['loss'],
            update_strategy=memory_cfg.get('update_strategy', 'rarity') if memory_cfg['use_memory'] else 'rarity',
            bank_size=memory_cfg.get('bank_size', 512) if memory_cfg['use_memory'] else 0,
            top_k=memory_cfg.get('top_k', 3),
            normalize_retrieved=memory_cfg.get('normalize_retrieved', True),
            rarity_threshold=memory_cfg.get('rarity_threshold', 0.2),
            diversity_weight=memory_cfg.get('diversity_weight', 0.5),
            momentum=phase2_cfg.get('memory_momentum', 0.9999),
            model_name=model_cfg['backbone'],
            use_fp16=False
        )

        # Load checkpoint
        print(f"\nLoading model from: {model_path}")
        learn.load(model_path)
        print("Model loaded successfully!")

        # Remove ProgressCallback to avoid display issues in evaluation mode
        try:
            learn.remove_cbs([ProgressCallback, CSVLogger])
        except:
            # In case callbacks are not present
            pass

        # Run evaluation
        print("\n" + "="*60)
        print("RUNNING EVALUATION ON TEST SET")
        print("="*60)

        from sklearn.metrics import roc_auc_score

        learn.model.eval()

        # Get predictions without progress bar
        print("Running inference on test set...")
        preds, y_test = learn.get_preds(ds_idx=1)  # Get predictions on validation set (which is our test set)
        print(f"Predictions shape: {preds.shape}")

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, preds)

        scores = []
        for i in range(model_cfg['num_classes']):
            try:
                label_roc_auc_score = roc_auc_score(y_test[:, i], preds[:, i])
                scores.append(label_roc_auc_score)
            except:
                scores.append(0.0)

        print('\nROC AUC per disease:')
        for label, score in zip(disease_labels, scores):
            print(f'  {label:20s}: {score:.4f}')

        print(f'\n{"="*60}')
        print(f'Mean AUC Score: {roc_auc:.4f}')
        print(f'{"="*60}')

        # Save predictions if requested
        if eval_cfg.get('save_predictions', True):
            pred_save_path = f'{exp_name}_test_predictions.pt'
            torch.save({
                'predictions': preds,
                'targets': y_test,
                'disease_labels': disease_labels,
                'per_class_auc': scores,
                'mean_auc': roc_auc
            }, pred_save_path)
            print(f"\nResults saved to: {pred_save_path}")

        print("\n" + "="*60)
        print("Evaluation completed!")
        print("="*60)

        return  # Exit after evaluation

    # ========== PREPARE CONSISTENT VALIDATION SPLIT ==========
    print("\n" + "="*60)
    print("PREPARING CONSISTENT VALIDATION SPLIT")
    print("="*60)

    # Load official TRAIN set (from train_val_list.txt, no filter)
    # This will be split into 90% train + 10% validation
    train_df_all, disease_labels, _ = prepare_chestxray14_dataframe(
        data_cfg['data_dir'],
        seed=data_cfg['seed'],
        filter_normal=False  # Load all train images (no filter) to get consistent split
    )

    # Create validation split: 10% of train images for validation
    # This ensures Phase 1 and Phase 2 use the SAME validation images
    val_image_indices = get_validation_split(
        train_df_all,
        valid_pct=data_cfg.get('valid_pct', 0.1),
        seed=data_cfg['seed']
    )
    print(f"  → Train will be split into 90% train + 10% validation")

    # ========== PHASE 1: Abnormal images only ==========
    print("\n" + "="*60)
    print("PHASE 1: Training on abnormal images")
    print("="*60)

    # Prepare data for Phase 1 (filtered)
    train_val_df_phase1, disease_labels, _ = prepare_chestxray14_dataframe(
        data_cfg['data_dir'],
        seed=data_cfg['seed'],
        filter_normal=phase1_cfg['filter_normal']
    )

    # Create DataLoaders with consistent validation split
    dls_phase1 = create_dataloaders(
        train_val_df_phase1,
        disease_labels,
        batch_size=phase1_cfg['batch_size'],
        valid_pct=data_cfg.get('valid_pct', 0.1),
        seed=data_cfg['seed'],
        val_image_indices=val_image_indices  # Use consistent validation
    )

    # Create callbacks for Phase 1 (only if validation exists)
    cbs_phase1 = []

    # Check if we have validation set
    has_validation = data_cfg.get('valid_pct', 0.1) > 0

    if has_validation:
        # Add callbacks only if validation exists
        if phase1_cfg.get('save_best', True):
            cbs_phase1.append(
                SaveModelCallback(
                    monitor='valid_loss',
                    min_delta=phase1_cfg.get('min_delta', 0.0001),
                    with_opt=True
                )
            )

        if phase1_cfg.get('early_stopping_patience') is not None:
            cbs_phase1.append(
                EarlyStoppingCallback(
                    monitor='valid_loss',
                    min_delta=phase1_cfg.get('min_delta_early_stop', 0.001),
                    patience=phase1_cfg['early_stopping_patience']
                )
            )

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
        use_fp16=False
    )

    # LR Finder
    if phase1_cfg.get('use_lr_finder', False):
        print("\nRunning LR Finder...")
        # Remove ProgressCallback to avoid display issues when running as script
        try:
            learn.remove_cbs(ProgressCallback)
        except:
            pass
        lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
        print(f'Suggested learning rate (minimum): {lrs.minimum}')
        print(f'Suggested learning rate (valley): {lrs.valley}')
        base_lr = lrs.valley
    else:
        base_lr = phase1_cfg.get('lr', 1e-4)

    # Training Phase 1
    print(f"\nStarting Phase 1 training...")

    # Check if using warmup schedule
    warmup_epochs = phase1_cfg.get('warmup_epochs', None)

    if warmup_epochs is not None and warmup_epochs > 0:
        # NEW: Warmup schedule (freeze backbone, lower LR)
        warmup_lr = phase1_cfg.get('warmup_lr', base_lr / 10)
        total_epochs = phase1_cfg['total_epochs']

        print(f"  Warmup epochs: {warmup_epochs} (LR={warmup_lr})")
        print(f"  Main epochs: {total_epochs} (LR={base_lr})")
        print(f"  Memory momentum: {phase1_cfg.get('memory_momentum', 0.9)}")

        # Remove ProgressCallback to avoid display issues
        try:
            learn.remove_cbs(ProgressCallback)
        except:
            pass

        # Warmup phase: Freeze backbone
        print(f"\n[Warmup] Training with frozen backbone for {warmup_epochs} epochs...")
        learn.freeze()
        with learn.no_logging():
            learn.fit_one_cycle(warmup_epochs, lr_max=warmup_lr)

        # Main training: Unfreeze and train
        print(f"\n[Main] Unfreezing and training for {total_epochs} epochs...")
        learn.unfreeze()
        with learn.no_logging():
            learn.fit_one_cycle(total_epochs, lr_max=base_lr)

    else:
        # ORIGINAL: fine_tune schedule
        freeze_epochs = phase1_cfg.get('freeze_epochs', 3)
        total_epochs = phase1_cfg['total_epochs']

        print(f"  Freeze epochs: {freeze_epochs}")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Base LR: {base_lr}")
        print(f"  Memory momentum: {phase1_cfg.get('memory_momentum', 0.9)}")

        with learn.no_logging():
            learn.fine_tune(
                freeze_epochs=freeze_epochs,
                epochs=total_epochs,
                base_lr=base_lr
            )
    # Save Phase 1 model
    phase1_save_name = phase1_cfg.get('save_name', 'phase1_model')

    # Only save final model if save_best is False
    # If save_best=True, SaveModelCallback already saved the best model
    if not phase1_cfg.get('save_best', True):
        learn.save(phase1_save_name)
        print(f"\nPhase 1 model saved as: {phase1_save_name}")
    else:
        print(f"\nPhase 1 best model already saved by SaveModelCallback as: {phase1_save_name}")

    # ========== PHASE 2: All images (only for 2-phase training) ==========
    if not is_single_phase:
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning on all images (including normal)")
        print("="*60)

        # IMPORTANT: Re-seed to match notebook behavior
        # Notebook resets seed before Phase 2, which ensures:
        # - Deterministic data augmentation
        # - Reproducible dropout patterns
        # - Consistent batch shuffling
        seed_everything(data_cfg['seed'])
        print(f"Re-seeded with seed={data_cfg['seed']} for Phase 2")

        # Prepare data for Phase 2 (use full train set from earlier)
        # No need to reload - we already have train_df_all
        train_val_df_phase2 = train_df_all

        # Create DataLoaders with same consistent validation split
        dls_phase2 = create_dataloaders(
            train_val_df_phase2,
            disease_labels,
            batch_size=phase2_cfg['batch_size'],
            valid_pct=data_cfg.get('valid_pct', 0.1),
            seed=data_cfg['seed'],
            val_image_indices=val_image_indices  # Use same validation as Phase 1
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
            #ShowGraphCallback()
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
            use_fp16=False
        )
        try:
                learn_phase2.remove_cbs(ProgressCallback)
        except:
            pass
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
        with learn_phase2.no_logging():
            learn_phase2.fit_one_cycle(epochs, slice(lr_min, lr_max))

        # Save Phase 2 model
        phase2_save_name = phase2_cfg.get('save_name', 'phase2_model')

        # Only save final model if save_best is False
        # If save_best=True, SaveModelCallback already saved the best model
        if not phase2_cfg.get('save_best', True):
            learn_phase2.save(phase2_save_name)
            print(f"\nPhase 2 model saved as: {phase2_save_name}")
        else:
            print(f"\nPhase 2 best model already saved by SaveModelCallback as: {phase2_save_name}")

    # ========== EVALUATION ==========
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    from sklearn.metrics import roc_auc_score

    def get_roc_auc(learner, disease_labels):
        """Evaluate model on validation set"""
        learner.model.eval()
        preds, y_test = learner.get_preds(ds_idx=0)  # ds_idx=0 for test set

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

    # Evaluate final model (use phase2 if available, else phase1)
    # Load test data for evaluation
    test_df_eval = prepare_chestxray14_dataframe(
        data_cfg['data_dir'],
        seed=data_cfg['seed'],
        filter_normal=False  # Use all test images for evaluation
    )[2]  # Get test_df only

    dls_test = create_dataloaders(
        test_df_eval,
        disease_labels,
        batch_size=eval_cfg.get('batch_size', 64),
        valid_pct=0.0,
        seed=data_cfg['seed']
    )

    # Use the final trained model (phase2 if 2-phase, else phase1)
    if is_single_phase:
        final_learner = learn
    else:
        final_learner = learn_phase2

    final_learner.dls = dls_test
    results = get_roc_auc(final_learner, disease_labels)

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
