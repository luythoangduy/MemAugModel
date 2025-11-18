"""
Loss-Aware Memory Bank - Detailed Implementation with Extensive Comments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossAwareMemoryBank(nn.Module):
    """
    Memory Bank that prioritizes samples based on learning difficulty

    Key Innovation: Instead of storing based on feature statistics (norm, rarity),
    we store based on TASK DIFFICULTY (loss, uncertainty)

    Intuition:
    - Hard samples (high loss) → Model is struggling → Store to learn more
    - Easy samples (low loss) → Model already knows → Less useful
    """

    def __init__(self,
                 feature_dim,
                 bank_size=512,
                 update_mode='hard',  # 'hard', 'easy', 'curriculum'
                 loss_weight=0.6,     # Weight for loss signal
                 uncertainty_weight=0.4):  # Weight for uncertainty signal
        super().__init__()

        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.update_mode = update_mode
        self.loss_weight = loss_weight
        self.uncertainty_weight = uncertainty_weight

        # Memory storage: [bank_size, feature_dim]
        # Each row is a stored feature vector
        self.register_buffer('memory', torch.zeros(bank_size, feature_dim))

        # Current insertion index (for initial filling)
        self.register_buffer('index', torch.tensor(0))

        # Importance score for each memory slot
        # Used by priority queue to decide which slots to replace
        self.register_buffer('importance_scores', torch.zeros(bank_size))

        # Training statistics
        self.register_buffer('num_updates', torch.tensor(0))

        # For curriculum learning: track training progress
        self.register_buffer('curriculum_progress', torch.tensor(0.0))

    def compute_sample_difficulty(self, features, predictions, targets):
        """
        Compute difficulty score for each sample in the batch

        Args:
            features: [B, D] feature vectors
            predictions: [B, num_classes] model logits (before sigmoid)
            targets: [B, num_classes] ground truth multi-hot labels

        Returns:
            difficulty: [B] scalar difficulty score for each sample
                        Higher score = More difficult sample

        Difficulty Components:
        1. Sample Loss: How wrong is the model's prediction?
        2. Uncertainty: How confident is the model?
        """
        batch_size = features.size(0)

        # ===== FALLBACK: If predictions/targets not provided =====
        if predictions is None or targets is None:
            # Use feature-based heuristic (similar to original rarity)
            norms = torch.norm(features, dim=1)
            mean_norm = norms.mean()
            return torch.abs(norms - mean_norm) / (mean_norm + 1e-8)

        # ===== COMPONENT 1: PER-SAMPLE LOSS =====
        # Compute BCE loss for each sample (not averaged)
        # Shape: [B, num_classes] → [B]
        sample_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction='none'  # Keep per-element loss
        ).mean(dim=1)  # Average across diseases → [B]

        # Normalize loss to [0, 1] range for stability
        # (Optional but helps when combining with uncertainty)
        if sample_loss.max() > 0:
            sample_loss = sample_loss / (sample_loss.max() + 1e-8)

        # ===== COMPONENT 2: PREDICTION UNCERTAINTY =====
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)  # [B, num_classes]

        # Measure distance from decision boundary (0.5)
        # Close to 0.5 → Uncertain (model doesn't know)
        # Close to 0 or 1 → Confident (model is sure)
        confidence = torch.abs(probs - 0.5).mean(dim=1)  # [B]

        # Convert to uncertainty score (higher = more uncertain)
        # confidence in [0, 0.5]: 0.5 = very confident, 0 = very uncertain
        uncertainty = 1 - (2 * confidence)  # Map [0, 0.5] → [1, 0]

        # Clip to ensure uncertainty in [0, 1]
        uncertainty = torch.clamp(uncertainty, 0, 1)

        # ===== COMBINE SIGNALS =====
        # Weighted combination of loss and uncertainty
        difficulty = (self.loss_weight * sample_loss +
                     self.uncertainty_weight * uncertainty)

        return difficulty  # [B]

    def _select_samples_to_store(self, difficulty):
        """
        Select which samples to store based on difficulty scores

        Args:
            difficulty: [B] difficulty scores

        Returns:
            mask: [B] boolean mask, True = store this sample
        """
        batch_size = difficulty.size(0)

        if self.update_mode == 'hard':
            # ===== HARD SAMPLE MINING =====
            # Store only top 30% most difficult samples
            # These are samples where model is struggling the most
            threshold = torch.quantile(difficulty, 0.7)
            mask = difficulty > threshold

            # Explanation:
            # If difficulty = [0.1, 0.8, 1.5, 0.3, 2.1, 0.5, 1.2, 0.9]
            # quantile(0.7) = 1.2
            # mask = [F, F, T, F, T, F, T, F]
            # → Store samples with difficulty > 1.2

        elif self.update_mode == 'easy':
            # ===== EASY SAMPLE MINING =====
            # Store only bottom 30% easiest samples
            # Useful for:
            # - Early training: Learn basics first
            # - Prototypes: Stable, representative features
            threshold = torch.quantile(difficulty, 0.3)
            mask = difficulty < threshold

        elif self.update_mode == 'curriculum':
            # ===== CURRICULUM LEARNING =====
            # Gradually shift from easy to hard samples

            # Update curriculum progress (0 → 1 over training)
            self.curriculum_progress = torch.min(
                self.curriculum_progress + 1.0 / 10000.0,
                torch.tensor(1.0, device=difficulty.device)
            )

            # Early training: low threshold (store easy samples)
            # Late training: high threshold (store hard samples)
            # Linear interpolation: 0.3 → 0.7
            threshold_quantile = (
                self.curriculum_progress * 0.7 +
                (1 - self.curriculum_progress) * 0.3
            )

            threshold = torch.quantile(difficulty, threshold_quantile)
            mask = difficulty > threshold

            # Example progression:
            # Iteration 1000: progress=0.1 → quantile=0.34 → 66% stored (easy bias)
            # Iteration 5000: progress=0.5 → quantile=0.50 → 50% stored (mixed)
            # Iteration 10000: progress=1.0 → quantile=0.70 → 30% stored (hard bias)

        else:
            # ===== STORE ALL =====
            mask = torch.ones(batch_size, dtype=torch.bool, device=difficulty.device)

        return mask

    def update(self, features, predictions=None, targets=None):
        """
        Update memory bank with new samples

        Args:
            features: [B, D] feature vectors to potentially store
            predictions: [B, num_classes] model predictions (logits)
            targets: [B, num_classes] ground truth labels

        Algorithm:
        1. Compute difficulty for each sample
        2. Select samples to store (based on update_mode)
        3. Insert into memory using priority queue
        """
        batch_size = features.size(0)

        # ===== STEP 1: COMPUTE DIFFICULTY =====
        difficulty = self.compute_sample_difficulty(features, predictions, targets)

        # ===== STEP 2: SELECT SAMPLES =====
        mask = self._select_samples_to_store(difficulty)

        selected_features = features[mask]
        selected_scores = difficulty[mask]

        # Early exit if no samples selected
        if selected_features.size(0) == 0:
            return

        # ===== STEP 3: PRIORITY QUEUE INSERTION =====
        for i in range(selected_features.size(0)):
            current_score = selected_scores[i]
            current_feature = selected_features[i]

            if self.index < self.bank_size:
                # ===== MEMORY NOT FULL: APPEND =====
                idx = self.index.item()
                self.memory[idx] = current_feature
                self.importance_scores[idx] = current_score
                self.index += 1

            else:
                # ===== MEMORY FULL: PRIORITY QUEUE REPLACEMENT =====
                # Find slot with minimum importance
                min_score, min_idx = self.importance_scores.min(dim=0)

                # Replace only if new sample is more important
                if current_score > min_score:
                    self.memory[min_idx] = current_feature
                    self.importance_scores[min_idx] = current_score

                # If current_score <= min_score, discard new sample
                # (all existing samples are more important)

        self.num_updates += 1

    def retrieve(self, query, k=3):
        """
        Retrieve relevant features from memory

        Args:
            query: [B, D] query features
            k: number of neighbors to retrieve

        Returns:
            retrieved: [B, D] aggregated retrieved features
        """
        # If memory is empty, return zeros
        if self.index == 0:
            return torch.zeros_like(query)

        # Get valid memory slots (only filled slots)
        valid_size = min(self.index.item(), self.bank_size)
        valid_memory = self.memory[:valid_size]

        # ===== COSINE SIMILARITY =====
        # Normalize query and memory for cosine similarity
        norm_query = F.normalize(query, dim=1)  # [B, D]
        norm_memory = F.normalize(valid_memory, dim=1)  # [M, D]

        # Compute similarity matrix
        similarity = torch.matmul(norm_query, norm_memory.T)  # [B, M]

        # ===== TOP-K RETRIEVAL =====
        k = min(k, valid_size)
        batch_size = query.size(0)
        result = torch.zeros_like(query)

        for i in range(batch_size):
            # Find top-k most similar memory items
            weights, indices = similarity[i].topk(k)  # [k]

            # Retrieve corresponding features
            retrieved_features = valid_memory[indices]  # [k, D]

            # ===== WEIGHTED AGGREGATION =====
            # Normalize weights using softmax
            weights = F.softmax(weights, dim=0).unsqueeze(1)  # [k, 1]

            # Weighted sum
            result[i] = (retrieved_features * weights).sum(dim=0)  # [D]

        return result

    def get_statistics(self):
        """Get memory bank statistics for monitoring"""
        valid_size = min(self.index.item(), self.bank_size)

        if valid_size > 0:
            mean_importance = self.importance_scores[:valid_size].mean().item()
            max_importance = self.importance_scores[:valid_size].max().item()
            min_importance = self.importance_scores[:valid_size].min().item()
        else:
            mean_importance = max_importance = min_importance = 0.0

        return {
            'bank_size': self.bank_size,
            'filled': valid_size,
            'utilization': valid_size / self.bank_size,
            'update_mode': self.update_mode,
            'num_updates': self.num_updates.item(),
            'mean_importance': mean_importance,
            'max_importance': max_importance,
            'min_importance': min_importance,
            'curriculum_progress': self.curriculum_progress.item() if self.update_mode == 'curriculum' else None
        }


# ===== USAGE EXAMPLE =====
if __name__ == '__main__':
    print("Testing Loss-Aware Memory Bank\n")

    # Setup
    batch_size = 8
    feature_dim = 128
    num_classes = 14

    memory = LossAwareMemoryBank(
        feature_dim=feature_dim,
        bank_size=16,
        update_mode='hard'
    )

    # Simulate training batch
    features = torch.randn(batch_size, feature_dim)
    predictions = torch.randn(batch_size, num_classes)  # Logits
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()  # Multi-hot

    # Compute difficulty
    difficulty = memory.compute_sample_difficulty(features, predictions, targets)
    print("Difficulty scores:")
    for i, d in enumerate(difficulty):
        print(f"  Sample {i}: {d.item():.3f}")

    # Update memory
    print(f"\nUpdating memory (mode={memory.update_mode})...")
    memory.update(features, predictions, targets)

    # Stats
    stats = memory.get_statistics()
    print(f"\nMemory statistics:")
    print(f"  Filled: {stats['filled']}/{stats['bank_size']}")
    print(f"  Mean importance: {stats['mean_importance']:.3f}")
    print(f"  Max importance: {stats['max_importance']:.3f}")

    # Retrieve
    query = torch.randn(4, feature_dim)
    retrieved = memory.retrieve(query, k=3)
    print(f"\nRetrieval:")
    print(f"  Query shape: {query.shape}")
    print(f"  Retrieved shape: {retrieved.shape}")
