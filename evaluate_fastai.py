"""
FastAI Evaluation script following mem_model.ipynb structure

Usage:
    python evaluate_fastai.py --model_path models/fastai_memory_model_phase2 --data_dir /path/to/data
"""

import argparse
import random
import numpy as np
import torch
import os

from fastai.vision.all import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from configs.config import SEED, NUM_CLASSES, DISEASE_LABELS
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


def get_roc_auc_simple(learner):
    """
    Simple AUC evaluation (from notebook cell 40)

    Args:
        learner: FastAI Learner

    Returns:
        results: Dictionary with ROC AUC scores
    """
    learner.model.eval()
    preds, y_test = learner.get_preds(ds_idx=1)  # Validation set
    roc_auc = roc_auc_score(y_test, preds)

    scores = []
    for i in range(0, 14):
        try:
            label_roc_auc_score = roc_auc_score(y_test[:, i], preds[:, i])
            scores.append(label_roc_auc_score)
        except:
            scores.append(0.0)

    print('\nROC AUC Labels:')
    print('-' * 60)
    for label, score in zip(DISEASE_LABELS, scores):
        print(f'  {label:20s}: {score:.4f}')
    print('-' * 60)
    print(f'  {"Mean AUC":20s}: {roc_auc:.4f}')
    print('-' * 60)

    return {
        'roc_auc': roc_auc,
        'per_class_auc': scores,
        'preds': preds,
        'y_test': y_test
    }


def get_roc_auc_detailed(learner, threshold=0.5):
    """
    Detailed evaluation with multiple metrics (from notebook cell 42)

    Args:
        learner: FastAI Learner
        threshold: Classification threshold

    Returns:
        results: Dictionary with comprehensive metrics
    """
    learner.model.eval()
    preds, y_test = learner.get_preds(ds_idx=0)

    # Calculate ROC AUC (threshold-independent)
    roc_auc = roc_auc_score(y_test, preds)

    # Apply threshold to get binary predictions
    binary_preds = (preds > threshold).float()

    # Calculate other metrics with the chosen threshold
    precision = precision_score(
        y_test.cpu().numpy(),
        binary_preds.cpu().numpy(),
        average='macro',
        zero_division=0
    )
    recall = recall_score(
        y_test.cpu().numpy(),
        binary_preds.cpu().numpy(),
        average='macro',
        zero_division=0
    )
    f1 = f1_score(
        y_test.cpu().numpy(),
        binary_preds.cpu().numpy(),
        average='macro',
        zero_division=0
    )

    # Per-class metrics
    scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i in range(0, 14):
        # ROC AUC per class (threshold-independent)
        try:
            label_roc_auc_score = roc_auc_score(y_test[:, i], preds[:, i])
        except:
            label_roc_auc_score = 0.0
        scores.append(label_roc_auc_score)

        # Precision, recall, F1 with custom threshold
        class_precision = precision_score(
            y_test[:, i].cpu().numpy(),
            binary_preds[:, i].cpu().numpy(),
            zero_division=0
        )
        class_recall = recall_score(
            y_test[:, i].cpu().numpy(),
            binary_preds[:, i].cpu().numpy(),
            zero_division=0
        )
        class_f1 = f1_score(
            y_test[:, i].cpu().numpy(),
            binary_preds[:, i].cpu().numpy(),
            zero_division=0
        )

        precision_scores.append(class_precision)
        recall_scores.append(class_recall)
        f1_scores.append(class_f1)

    # Print results
    print(f'\nUsing threshold: {threshold}')
    print('\nPer-class Performance:')
    print('-' * 80)
    print(f'{"Disease":20s} {"AUC":>8s} {"Precision":>10s} {"Recall":>10s} {"F1":>8s}')
    print('-' * 80)
    for i, label in enumerate(DISEASE_LABELS):
        print(f'{label:20s} {scores[i]:8.4f} {precision_scores[i]:10.4f} '
              f'{recall_scores[i]:10.4f} {f1_scores[i]:8.4f}')
    print('-' * 80)

    print(f'\nOverall Metrics:')
    print(f'  ROC AUC:   {roc_auc:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall:    {recall:.4f}')
    print(f'  F1 Score:  {f1:.4f}')

    return {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_auc': scores,
        'per_class_precision': precision_scores,
        'per_class_recall': recall_scores,
        'per_class_f1': f1_scores,
        'preds': preds,
        'y_test': y_test,
        'binary_preds': binary_preds
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate FastAI Chest X-Ray Model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model (without .pth extension)')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/data',
                       help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--experiment', type=str, default='memory_rarity_effv2s',
                       help='Experiment name from config.py')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed metrics with precision, recall, F1')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold for detailed metrics')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')

    args = parser.parse_args()

    # Set random seed
    seed_everything(SEED)

    print(f"\n{'='*60}")
    print(f"Evaluating model: {args.model_path}")
    print(f"{'='*60}\n")

    # Get experiment configuration
    from configs.config import get_experiment_config
    exp_config = get_experiment_config(args.experiment)

    # Prepare data
    print("Preparing data...")
    train_val_df, disease_labels, test_df = prepare_chestxray14_dataframe(
        args.data_dir,
        seed=SEED,
        filter_normal=False  # Include all images
    )

    dls = create_dataloaders(
        test_df,
        disease_labels,
        batch_size=args.batch_size,
        valid_pct=0.0,
        seed=SEED
    )

    # Create learner
    print("\nCreating learner...")
    learn = create_fastai_learner(
        dls,
        num_classes=NUM_CLASSES,
        loss_type='bce',
        update_strategy=exp_config['memory']['update_strategy'] if exp_config['memory']['use_memory'] else 'rarity',
        bank_size=exp_config['memory']['bank_size'] if exp_config['memory']['use_memory'] else 0,
        model_name=exp_config['backbone']['name'],
        use_fp16=True
    )

    # Load model
    print(f"Loading model from: {args.model_path}")
    learn.load(args.model_path)
    print("Model loaded successfully!")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    if args.detailed:
        results = get_roc_auc_detailed(learn, threshold=args.threshold)
    else:
        results = get_roc_auc_simple(learn)

    # Save predictions if requested
    if args.save_predictions:
        pred_file = f'{args.model_path}_predictions.pt'
        torch.save(results['preds'], pred_file)
        print(f"\nPredictions saved to: {pred_file}")

    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == '__main__':
    main()
