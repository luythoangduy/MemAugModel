"""
Advanced Memory Bank with improved update and retrieval strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossAwareMemoryBank(nn.Module):
    """
    Memory Bank with loss-aware update and adaptive retrieval

    Improvements:
    1. Update: Prioritize hard samples based on loss/uncertainty
    2. Retrieval: Adaptive k based on confidence
    3. Storage: Priority queue instead of FIFO
    """

    def __init__(self, feature_dim, bank_size=512,
                 update_mode='hard',  # 'hard', 'easy', 'curriculum'
                 adaptive_k=True,
                 min_k=1, max_k=10,
                 momentum=0.9):
        super().__init__()

        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.update_mode = update_mode
        self.adaptive_k = adaptive_k
        self.min_k = min_k
        self.max_k = max_k
        self.momentum = momentum

        # Memory storage
        self.register_buffer('memory', torch.zeros(bank_size, feature_dim))
        self.register_buffer('index', torch.tensor(0))

        # Priority scores for each memory slot
        self.register_buffer('importance_scores', torch.zeros(bank_size))

        # Statistics
        self.register_buffer('num_updates', torch.tensor(0))

    def compute_sample_difficulty(self, features, predictions=None, targets=None):
        """
        Compute difficulty score for each sample

        Args:
            features: [B, D]
            predictions: [B, num_classes] logits (optional)
            targets: [B, num_classes] ground truth (optional)

        Returns:
            difficulty: [B] higher = more difficult
        """
        batch_size = features.size(0)

        if predictions is None or targets is None:
            # Fallback to feature-based scoring
            norms = torch.norm(features, dim=1)
            mean_norm = norms.mean()
            return torch.abs(norms - mean_norm) / (mean_norm + 1e-8)

        # 1. Per-sample loss (primary signal)
        sample_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        ).mean(dim=1)  # [B]

        # 2. Prediction uncertainty (samples near decision boundary)
        probs = torch.sigmoid(predictions)
        # Distance from 0.5 (confident) vs near 0.5 (uncertain)
        confidence = torch.abs(probs - 0.5).mean(dim=1)
        uncertainty = 1 - confidence

        # 3. Combine: High loss + High uncertainty = Hard sample
        difficulty = 0.6 * sample_loss + 0.4 * uncertainty

        return difficulty

    def update(self, features, predictions=None, targets=None):
        """
        Update memory with priority-based selection

        Args:
            features: [B, D] features to potentially store
            predictions: [B, num_classes] model predictions
            targets: [B, num_classes] ground truth labels
        """
        batch_size = features.size(0)

        # Compute difficulty scores
        difficulty = self.compute_sample_difficulty(features, predictions, targets)

        # Select samples based on update mode
        if self.update_mode == 'hard':
            # Store hard samples (high loss)
            threshold = torch.quantile(difficulty, 0.7)
            mask = difficulty > threshold
        elif self.update_mode == 'easy':
            # Store easy samples (low loss)
            threshold = torch.quantile(difficulty, 0.3)
            mask = difficulty < threshold
        elif self.update_mode == 'curriculum':
            # Adaptive based on training progress
            # Early: easy, Late: hard
            progress = min(self.num_updates.item() / 10000.0, 1.0)
            threshold = progress * 0.7 + (1 - progress) * 0.3
            mask = difficulty > torch.quantile(difficulty, threshold)
        else:
            # Store all
            mask = torch.ones(batch_size, dtype=torch.bool, device=features.device)

        selected_features = features[mask]
        selected_scores = difficulty[mask]

        if selected_features.size(0) == 0:
            return

        # Priority-based insertion
        for i in range(selected_features.size(0)):
            current_score = selected_scores[i]
            current_feature = selected_features[i]

            if self.index < self.bank_size:
                # Memory not full, just append
                idx = self.index.item()
                self.memory[idx] = current_feature
                self.importance_scores[idx] = current_score
                self.index += 1
            else:
                # Memory full, replace least important
                min_score, min_idx = self.importance_scores.min(dim=0)

                # Replace if new sample is more important
                if current_score > min_score:
                    self.memory[min_idx] = current_feature
                    self.importance_scores[min_idx] = current_score

        self.num_updates += 1

    def retrieve(self, query, predictions=None, k=None):
        """
        Retrieve relevant memories with adaptive k

        Args:
            query: [B, D] query features
            predictions: [B, num_classes] model predictions (for adaptive k)
            k: fixed k (if None, use adaptive)

        Returns:
            retrieved: [B, D] retrieved features
        """
        if self.index == 0:
            return torch.zeros_like(query)

        valid_memory = self.memory[:self.index] if self.index < self.bank_size else self.memory
        batch_size = query.size(0)

        # Compute similarity
        norm_query = F.normalize(query, dim=1)
        norm_memory = F.normalize(valid_memory, dim=1)
        similarity = torch.matmul(norm_query, norm_memory.T)  # [B, M]

        result = torch.zeros_like(query)

        for i in range(batch_size):
            # Determine k for this sample
            if k is not None:
                current_k = k
            elif self.adaptive_k and predictions is not None:
                # Adaptive k based on confidence
                probs = torch.sigmoid(predictions[i])
                confidence = torch.abs(probs - 0.5).mean()
                # Low confidence → retrieve more neighbors
                current_k = int(self.min_k + (self.max_k - self.min_k) * (1 - confidence))
            else:
                current_k = (self.min_k + self.max_k) // 2

            current_k = min(current_k, valid_memory.size(0))

            # Top-k retrieval
            weights, indices = similarity[i].topk(current_k)
            retrieved_features = valid_memory[indices]

            # Weighted aggregation
            weights = F.softmax(weights, dim=0).unsqueeze(1)
            result[i] = (retrieved_features * weights).sum(dim=0)

        return result


class AttentionMemoryBank(nn.Module):
    """
    Memory Bank with learnable attention-based retrieval
    """

    def __init__(self, feature_dim, bank_size=512, num_heads=4, dropout=0.1):
        super().__init__()

        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Memory storage
        self.register_buffer('memory', torch.zeros(bank_size, feature_dim))
        self.register_buffer('index', torch.tensor(0))
        self.register_buffer('importance_scores', torch.zeros(bank_size))

        # Multi-head attention projections
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)

    def update(self, features, predictions=None, targets=None):
        """Same priority-based update as LossAwareMemoryBank"""
        batch_size = features.size(0)

        if predictions is not None and targets is not None:
            sample_loss = F.binary_cross_entropy_with_logits(
                predictions, targets, reduction='none'
            ).mean(dim=1)
            difficulty = sample_loss
        else:
            norms = torch.norm(features, dim=1)
            difficulty = torch.abs(norms - norms.mean()) / (norms.mean() + 1e-8)

        # Store top difficult samples
        threshold = torch.quantile(difficulty, 0.7)
        mask = difficulty > threshold

        selected_features = features[mask]
        selected_scores = difficulty[mask]

        for i in range(selected_features.size(0)):
            if self.index < self.bank_size:
                idx = self.index.item()
                self.memory[idx] = selected_features[i]
                self.importance_scores[idx] = selected_scores[i]
                self.index += 1
            else:
                min_score, min_idx = self.importance_scores.min(dim=0)
                if selected_scores[i] > min_score:
                    self.memory[min_idx] = selected_features[i]
                    self.importance_scores[min_idx] = selected_scores[i]

    def retrieve(self, query, k=None):
        """
        Multi-head attention retrieval

        Args:
            query: [B, D]
            k: if specified, use top-k; otherwise use full attention

        Returns:
            retrieved: [B, D]
        """
        if self.index == 0:
            return torch.zeros_like(query)

        valid_memory = self.memory[:self.index] if self.index < self.bank_size else self.memory
        batch_size = query.size(0)
        mem_size = valid_memory.size(0)

        # Project to Q, K, V
        Q = self.query_proj(query)  # [B, D]
        K = self.key_proj(valid_memory)  # [M, D]
        V = self.value_proj(valid_memory)  # [M, D]

        # Reshape for multi-head: [B, num_heads, head_dim]
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(mem_size, self.num_heads, self.head_dim)
        V = V.view(mem_size, self.num_heads, self.head_dim)

        # Scaled dot-product attention for each head
        # [B, num_heads, M]
        scores = torch.einsum('bhd,mhd->bhm', Q, K) / (self.head_dim ** 0.5)

        if k is not None and k < mem_size:
            # Top-k attention
            topk_scores, topk_indices = scores.topk(k, dim=2)
            weights = F.softmax(topk_scores, dim=2)  # [B, num_heads, k]

            # Gather top-k values
            # Expand indices for gathering
            topk_indices_expanded = topk_indices.unsqueeze(3).expand(-1, -1, -1, self.head_dim)
            V_expanded = V.unsqueeze(0).expand(batch_size, -1, -1, -1)
            V_topk = torch.gather(V_expanded, 2, topk_indices_expanded)  # [B, num_heads, k, head_dim]

            # Weighted sum
            attended = torch.einsum('bhnk,bhnd->bhd', weights, V_topk)
        else:
            # Full attention
            weights = F.softmax(scores, dim=2)  # [B, num_heads, M]
            attended = torch.einsum('bhm,mhd->bhd', weights, V)

        # Concatenate heads and project
        attended = attended.reshape(batch_size, self.feature_dim)
        output = self.out_proj(attended)
        output = self.dropout(output)

        return output


class PrototypeMemoryBank(nn.Module):
    """
    Memory Bank with prototype-based clustering
    Fast retrieval by clustering memory into prototypes
    """

    def __init__(self, feature_dim, bank_size=512, num_prototypes=64, update_freq=100):
        super().__init__()

        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.num_prototypes = num_prototypes
        self.update_freq = update_freq

        # Raw memory
        self.register_buffer('memory', torch.zeros(bank_size, feature_dim))
        self.register_buffer('index', torch.tensor(0))

        # Prototypes (cluster centers)
        self.register_buffer('prototypes', torch.zeros(num_prototypes, feature_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_prototypes))
        self.register_buffer('prototypes_initialized', torch.tensor(False))

        self.register_buffer('num_updates', torch.tensor(0))

    def update(self, features, predictions=None, targets=None):
        """Store features in raw memory"""
        batch_size = features.size(0)

        for i in range(batch_size):
            idx = self.index.item() % self.bank_size
            self.memory[idx] = features[i]
            self.index = (self.index + 1) % self.bank_size

        self.num_updates += 1

        # Periodically update prototypes
        if self.num_updates % self.update_freq == 0:
            self._update_prototypes()

    def _update_prototypes(self):
        """Cluster memory into prototypes using k-means"""
        valid_size = min(self.index.item(), self.bank_size)

        if valid_size < self.num_prototypes:
            return

        valid_memory = self.memory[:valid_size]

        # Simple k-means clustering
        # Initialize centroids randomly
        if not self.prototypes_initialized:
            indices = torch.randperm(valid_size)[:self.num_prototypes]
            self.prototypes = valid_memory[indices].clone()
            self.prototypes_initialized = torch.tensor(True)

        # K-means iterations
        for _ in range(10):
            # Assign to nearest prototype
            similarity = torch.matmul(
                F.normalize(valid_memory, dim=1),
                F.normalize(self.prototypes, dim=1).T
            )
            assignments = similarity.argmax(dim=1)

            # Update prototypes
            for k in range(self.num_prototypes):
                mask = assignments == k
                if mask.sum() > 0:
                    self.prototypes[k] = valid_memory[mask].mean(dim=0)
                    self.prototype_counts[k] = mask.sum().float()

    def retrieve(self, query, k=3):
        """
        Retrieve from prototypes (fast)

        Args:
            query: [B, D]
            k: number of prototypes to retrieve

        Returns:
            retrieved: [B, D]
        """
        if not self.prototypes_initialized:
            return torch.zeros_like(query)

        # Compute similarity to prototypes
        norm_query = F.normalize(query, dim=1)
        norm_prototypes = F.normalize(self.prototypes, dim=1)
        similarity = torch.matmul(norm_query, norm_prototypes.T)  # [B, num_prototypes]

        # Top-k prototypes
        k = min(k, self.num_prototypes)
        weights, indices = similarity.topk(k, dim=1)  # [B, k]

        # Retrieve prototypes
        retrieved_prototypes = self.prototypes[indices]  # [B, k, D]

        # Weighted aggregation
        weights = F.softmax(weights, dim=1).unsqueeze(2)  # [B, k, 1]
        result = (retrieved_prototypes * weights).sum(dim=1)  # [B, D]

        return result
