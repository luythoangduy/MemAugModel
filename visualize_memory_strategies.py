"""
Visualize different memory update strategies
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Simulate difficulty scores for 100 batches
np.random.seed(42)
torch.manual_seed(42)

num_batches = 100
batch_size = 32

# Simulate difficulty scores (some batches have harder samples)
all_difficulties = []
for b in range(num_batches):
    # Early batches: easier (lower difficulty)
    # Late batches: harder (higher difficulty)
    base_difficulty = 0.3 + 0.5 * (b / num_batches)

    # Add noise
    difficulties = torch.randn(batch_size) * 0.2 + base_difficulty
    difficulties = torch.clamp(difficulties, 0, 1)
    all_difficulties.append(difficulties)

# Strategies
def hard_mining(difficulty):
    threshold = torch.quantile(difficulty, 0.7)
    return difficulty > threshold

def easy_mining(difficulty):
    threshold = torch.quantile(difficulty, 0.3)
    return difficulty < threshold

def curriculum_mining(difficulty, progress):
    threshold_quantile = progress * 0.7 + (1 - progress) * 0.3
    threshold = torch.quantile(difficulty, threshold_quantile)
    return difficulty > threshold

# Collect stored samples
hard_samples = []
easy_samples = []
curriculum_samples = []

for b, difficulty in enumerate(all_difficulties):
    progress = b / num_batches

    # Hard mining
    mask_hard = hard_mining(difficulty)
    hard_samples.extend(difficulty[mask_hard].tolist())

    # Easy mining
    mask_easy = easy_mining(difficulty)
    easy_samples.extend(difficulty[mask_easy].tolist())

    # Curriculum
    mask_curr = curriculum_mining(difficulty, progress)
    curriculum_samples.extend(difficulty[mask_curr].tolist())

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Difficulty distribution over batches
ax = axes[0, 0]
all_diff_flat = torch.cat(all_difficulties).numpy()
batch_indices = np.repeat(np.arange(num_batches), batch_size)
scatter = ax.scatter(batch_indices, all_diff_flat, c=all_diff_flat,
                    cmap='viridis', alpha=0.3, s=10)
ax.set_xlabel('Batch Index')
ax.set_ylabel('Difficulty Score')
ax.set_title('Difficulty Distribution Over Training')
plt.colorbar(scatter, ax=ax, label='Difficulty')

# Plot 2: Stored sample distributions
ax = axes[0, 1]
ax.hist([hard_samples, easy_samples, curriculum_samples],
        bins=30, alpha=0.6, label=['Hard', 'Easy', 'Curriculum'])
ax.set_xlabel('Difficulty Score')
ax.set_ylabel('Count')
ax.set_title('Distribution of Stored Samples')
ax.legend()
ax.axvline(np.mean(hard_samples), color='C0', linestyle='--',
          label=f'Hard mean: {np.mean(hard_samples):.2f}')
ax.axvline(np.mean(easy_samples), color='C1', linestyle='--',
          label=f'Easy mean: {np.mean(easy_samples):.2f}')
ax.axvline(np.mean(curriculum_samples), color='C2', linestyle='--',
          label=f'Curriculum mean: {np.mean(curriculum_samples):.2f}')

# Plot 3: Storage rate over time
ax = axes[1, 0]
storage_rates_hard = []
storage_rates_easy = []
storage_rates_curr = []

for b, difficulty in enumerate(all_difficulties):
    progress = b / num_batches

    storage_rates_hard.append(hard_mining(difficulty).float().mean().item())
    storage_rates_easy.append(easy_mining(difficulty).float().mean().item())
    storage_rates_curr.append(curriculum_mining(difficulty, progress).float().mean().item())

ax.plot(storage_rates_hard, label='Hard Mining', linewidth=2)
ax.plot(storage_rates_easy, label='Easy Mining', linewidth=2)
ax.plot(storage_rates_curr, label='Curriculum', linewidth=2)
ax.set_xlabel('Batch Index')
ax.set_ylabel('Storage Rate')
ax.set_title('Storage Rate Over Training')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Statistics table
ax = axes[1, 1]
ax.axis('off')

stats_text = f"""
MEMORY UPDATE STRATEGIES COMPARISON

Hard Sample Mining:
  • Stores top 30% hardest samples
  • Mean difficulty: {np.mean(hard_samples):.3f}
  • Std difficulty: {np.std(hard_samples):.3f}
  • Total stored: {len(hard_samples)}
  • Focus: Samples model struggles with

Easy Sample Mining:
  • Stores bottom 30% easiest samples
  • Mean difficulty: {np.mean(easy_samples):.3f}
  • Std difficulty: {np.std(easy_samples):.3f}
  • Total stored: {len(easy_samples)}
  • Focus: Stable, representative features

Curriculum Learning:
  • Adaptive: Easy → Hard over training
  • Mean difficulty: {np.mean(curriculum_samples):.3f}
  • Std difficulty: {np.std(curriculum_samples):.3f}
  • Total stored: {len(curriculum_samples)}
  • Focus: Progressive difficulty increase

Recommendations:
  • Hard mining: Best for imbalanced datasets
  • Easy mining: Best for prototypes/initialization
  • Curriculum: Best for stable training
"""

ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
       fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('memory_strategies_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: memory_strategies_comparison.png")

# Print summary
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nHard Mining:")
print(f"  Mean difficulty: {np.mean(hard_samples):.3f} (should be HIGH)")
print(f"  Samples stored: {len(hard_samples)}")

print(f"\nEasy Mining:")
print(f"  Mean difficulty: {np.mean(easy_samples):.3f} (should be LOW)")
print(f"  Samples stored: {len(easy_samples)}")

print(f"\nCurriculum:")
print(f"  Mean difficulty: {np.mean(curriculum_samples):.3f} (should be MEDIUM)")
print(f"  Samples stored: {len(curriculum_samples)}")
print(f"  Adapts from {min(storage_rates_curr):.2%} to {max(storage_rates_curr):.2%} storage rate")
