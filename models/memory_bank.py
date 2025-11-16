"""
Memory Bank module for storing and retrieving pathological features
Based on the paper: "Mitigating Class Imbalance in Chest X-Ray Classification with Memory-Augmented Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """
    Memory Bank: Store rare/important pathological features and retrieve them during inference

    Update strategies:
    - rarity: Based on L2 norm deviation from batch mean
    - statistical: Uses running mean of norms for outlier detection
    - entropy: Based on prediction entropy (uncertainty)
    - diversity: Select features most different from current memory
    - hybrid: Combine rarity and diversity
    - fifo: First-In-First-Out
    - reservoir: Reservoir sampling (probabilistic)
    """

    def __init__(self, feature_dim, bank_size=512, update_strategy='rarity',
                 rarity_threshold=0.2, diversity_weight=0.5, momentum=0.9,
                 normalize_retrieved=True):
        super(MemoryBank, self).__init__()

        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.update_strategy = update_strategy
        self.rarity_threshold = rarity_threshold
        self.diversity_weight = diversity_weight
        self.momentum = momentum
        self.normalize_retrieved = normalize_retrieved

        # Memory storage (persistent buffers)
        self.register_buffer('memory', torch.zeros(bank_size, feature_dim))
        self.register_buffer('index', torch.tensor(0))
        self.register_buffer('memory_count', torch.zeros(bank_size))

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_var', torch.ones(feature_dim))
        self.register_buffer('num_updates', torch.tensor(0))

        # For statistical strategy - track running mean of norms
        self.register_buffer('running_mean_norm', torch.tensor(0.0))

    def compute_importance_scores(self, features):
        """
        Compute importance scores based on selected strategy

        Args:
            features: Input features [B, D]

        Returns:
            scores: Importance scores [B]
        """
        batch_size = features.size(0)

        if self.update_strategy == 'rarity':
            # Original: Based on L2 norm deviation from mean (Eq. 3 in paper)
            mean_norm = torch.mean(torch.norm(features, dim=1))
            scores = torch.abs(torch.norm(features, dim=1) - mean_norm) / (mean_norm + 1e-8)
            return scores

        elif self.update_strategy == 'statistical':
            # Similar to rarity but uses running mean of norms
            if self.num_updates > 0:
                sample_norms = torch.norm(features, dim=1)
                scores = torch.abs(sample_norms - self.running_mean_norm) / (self.running_mean_norm + 1e-8)
                return scores
            else:
                return torch.zeros(batch_size, device=features.device)

        elif self.update_strategy == 'entropy':
            # Based on prediction entropy (uncertainty)
            norm_features = F.normalize(features, dim=1)
            entropy = -torch.sum(norm_features * torch.log(torch.abs(norm_features) + 1e-8), dim=1)
            return -entropy  # Higher entropy = more uncertain = rarer

        elif self.update_strategy == 'diversity':
            # Select features most different from current memory
            if self.index > 0:
                valid_memory = self.memory[:self.index]
                norm_features = F.normalize(features, dim=1)
                norm_memory = F.normalize(valid_memory, dim=1)
                # Max similarity to any memory item
                max_similarity = torch.matmul(norm_features, norm_memory.T).max(dim=1)[0]
                # Lower similarity = more diverse
                return max_similarity
            else:
                return torch.zeros(batch_size, device=features.device)

        elif self.update_strategy == 'hybrid':
            # Combine rarity and diversity
            # Rarity component
            mean_norm = torch.mean(torch.norm(features, dim=1))
            rarity = torch.abs(torch.norm(features, dim=1) - mean_norm) / (mean_norm + 1e-8)

            # Diversity component
            if self.index > 0:
                valid_memory = self.memory[:self.index]
                norm_features = F.normalize(features, dim=1)
                norm_memory = F.normalize(valid_memory, dim=1)
                max_similarity = torch.matmul(norm_features, norm_memory.T).max(dim=1)[0]
                diversity = 1 - max_similarity
            else:
                diversity = torch.ones(batch_size, device=features.device)

            # Weighted combination
            scores = (1 - self.diversity_weight) * (-rarity) + self.diversity_weight * (-diversity)
            return scores

        elif self.update_strategy == 'fifo':
            # First In First Out - no scoring needed
            return torch.zeros(batch_size, device=features.device)

        elif self.update_strategy == 'reservoir':
            # Reservoir sampling - probabilistic
            return torch.rand(batch_size, device=features.device)

        else:
            raise ValueError(f"Unknown update strategy: {self.update_strategy}")

    def update_statistics(self, features):
        """
        Update running mean and variance

        Args:
            features: Input features [B, D]
        """
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0)

        # For statistical strategy - update running mean of norms
        if self.update_strategy == 'statistical':
            sample_norms = torch.norm(features, dim=1)
            batch_mean_norm = sample_norms.mean()

            if self.num_updates == 0:
                self.running_mean_norm = batch_mean_norm
            else:
                self.running_mean_norm = self.momentum * self.running_mean_norm + (1 - self.momentum) * batch_mean_norm

        if self.num_updates == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

        self.num_updates += 1

    def update(self, features, threshold=None):
        """
        Update memory bank with new features (Eq. 4 in paper)

        Args:
            features: Features to potentially store [B, D]
            threshold: Override default rarity threshold
        """
        batch_size = features.size(0)

        # Update running statistics
        self.update_statistics(features)

        # Compute importance scores
        scores = self.compute_importance_scores(features)

        # Apply threshold if specified
        if threshold is None:
            threshold = self.rarity_threshold

        if self.update_strategy in ['fifo', 'reservoir']:
            # For FIFO and reservoir, store all or based on probability
            if self.update_strategy == 'fifo':
                mask = torch.ones(batch_size, dtype=torch.bool, device=features.device)
            else:  # reservoir
                # Reservoir sampling: probability decreases as memory fills
                total_seen = self.num_updates * batch_size
                probs = torch.minimum(
                    torch.tensor(self.bank_size / (total_seen + 1e-8), device=features.device),
                    torch.ones(batch_size, device=features.device)
                )
                mask = scores < probs  # scores are random in [0,1]
        else:
            # For other strategies, select based on threshold
            if self.update_strategy in ['entropy', 'diversity', 'hybrid']:
                # Lower scores are better (more rare/diverse)
                mask = scores < torch.quantile(scores, threshold)
            else:
                # rarity, statistical
                mask = scores < threshold

        selected_features = features[mask]

        if selected_features.size(0) > 0:
            # If memory is full, replace oldest entries (FIFO)
            if self.index + selected_features.size(0) > self.bank_size:
                # Circular buffer - overwrite from beginning
                remaining = self.bank_size - self.index
                self.memory[self.index:] = selected_features[:remaining]
                overflow = selected_features.size(0) - remaining
                if overflow > 0:
                    self.memory[:overflow] = selected_features[remaining:remaining + overflow]
                    self.index = torch.tensor(overflow)
                else:
                    self.index = torch.tensor(self.bank_size)
            else:
                num_to_add = selected_features.size(0)
                self.memory[self.index:self.index + num_to_add] = selected_features
                self.memory_count[self.index:self.index + num_to_add] += 1
                self.index = (self.index + num_to_add) % self.bank_size

    def retrieve(self, query, k=3, self_match_threshold=0.9999):
        """
        Retrieve relevant memories for query features (Eq. 5-6 in paper)

        Args:
            query: Query features [B, D]
            k: Number of nearest neighbors to retrieve
            self_match_threshold: Threshold to avoid self-matching

        Returns:
            retrieved: Weighted sum of retrieved features [B, D]
        """
        # Get valid memory entries
        if self.index == 0:
            return torch.zeros_like(query)

        valid_memory = self.memory[:self.index] if self.index < self.bank_size else self.memory

        # Compute cosine similarity (Eq. 5)
        norm_query = F.normalize(query, dim=1)
        norm_memory = F.normalize(valid_memory, dim=1)
        similarity = torch.matmul(norm_query, norm_memory.T)

        # Avoid self-similarity (similarity = 1.0)
        mask = similarity < self_match_threshold

        k = min(k, valid_memory.size(0))
        batch_size = query.size(0)
        result = torch.zeros_like(query)

        for i in range(batch_size):
            valid_indices = torch.where(mask[i])[0]

            if len(valid_indices) == 0:
                continue

            valid_similarities = similarity[i, valid_indices]
            k_valid = min(k, valid_similarities.size(0))
            weights, rel_indices = valid_similarities.topk(k_valid)
            abs_indices = valid_indices[rel_indices]

            retrieved = valid_memory[abs_indices]

            # Softmax normalization of weights (Eq. 6)
            weights = F.softmax(weights, dim=0).unsqueeze(1).expand_as(retrieved)
            weighted_features = (retrieved * weights).sum(dim=0)

            result[i] = weighted_features

        # Optionally normalize retrieved features
        if self.normalize_retrieved:
            result = F.normalize(result, dim=1) * torch.norm(query, dim=1, keepdim=True)

        return result

    def get_memory_stats(self):
        """Get statistics about current memory bank"""
        return {
            'size': self.bank_size,
            'filled': min(self.index.item(), self.bank_size),
            'utilization': min(self.index.item() / self.bank_size, 1.0),
            'update_strategy': self.update_strategy,
            'num_updates': self.num_updates.item()
        }
