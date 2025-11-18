"""
Test 1-phase no-validation config loading and validation
"""

import yaml
import sys

def test_config(config_path):
    print(f"\nTesting config: {config_path}")
    print("="*60)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check structure
    assert 'experiment_name' in config
    assert 'model' in config
    assert 'memory' in config
    assert 'data' in config
    assert 'phase1' in config

    # Check 1-phase (no phase2)
    is_single_phase = 'phase2' not in config
    assert is_single_phase, "Should be 1-phase config"
    print(f"[OK] Single-phase config: {config['experiment_name']}")

    # Check no validation
    valid_pct = config['data'].get('valid_pct', 0.1)
    assert valid_pct == 0.0, f"Expected valid_pct=0.0, got {valid_pct}"
    print(f"[OK] No validation split (valid_pct={valid_pct})")

    # Check warmup
    warmup_epochs = config['phase1'].get('warmup_epochs')
    assert warmup_epochs is not None, "warmup_epochs not found"
    assert warmup_epochs > 0, f"warmup_epochs should be > 0, got {warmup_epochs}"
    print(f"[OK] Warmup epochs: {warmup_epochs}")

    # Check warmup LR
    warmup_lr = config['phase1'].get('warmup_lr')
    main_lr = config['phase1'].get('lr')
    assert warmup_lr is not None, "warmup_lr not found"
    assert main_lr is not None, "lr not found"
    assert warmup_lr < main_lr, f"warmup_lr should be < lr"
    print(f"[OK] Warmup LR: {warmup_lr} (Main LR: {main_lr})")

    # Check total epochs
    total_epochs = config['phase1'].get('total_epochs')
    assert total_epochs is not None, "total_epochs not found"
    print(f"[OK] Main training epochs: {total_epochs}")

    # Check no early stopping
    early_stopping = config['phase1'].get('early_stopping_patience')
    assert early_stopping is None, f"Expected no early stopping, got patience={early_stopping}"
    print(f"[OK] No early stopping")

    # Check save_best
    save_best = config['phase1'].get('save_best', True)
    assert save_best == False, f"Expected save_best=False, got {save_best}"
    print(f"[OK] Save final model only (save_best={save_best})")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"\nTraining schedule:")
    print(f"  Phase 1 only (no Phase 2)")
    print(f"  Full training set (no validation)")
    print(f"  Warmup: {warmup_epochs} epochs @ LR={warmup_lr}")
    print(f"  Main: {total_epochs} epochs @ LR={main_lr}")
    print(f"  Total: {warmup_epochs + total_epochs} epochs")
    print(f"\nMemory:")
    print(f"  Use memory: {config['memory'].get('use_memory', False)}")
    if config['memory'].get('use_memory'):
        print(f"  Type: {config['memory'].get('memory_type', 'rarity')}")
        print(f"  Bank size: {config['memory'].get('bank_size', 512)}")

    print("\n[PASS] All checks passed!")
    return True


if __name__ == '__main__':
    configs = [
        'configs/experiments/memory_rarity_effv2s_1phase_noval.yaml',
        'configs/experiments/baseline_effv2s_1phase_noval.yaml',
        'configs/experiments/advanced_memory_hard_1phase_noval.yaml',
    ]

    all_passed = True
    for config_path in configs:
        try:
            test_config(config_path)
        except Exception as e:
            print(f"\n[FAIL] FAILED: {config_path}")
            print(f"Error: {e}")
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] ALL CONFIGS VALIDATED")
    else:
        print("[ERROR] SOME CONFIGS FAILED")
        sys.exit(1)
