"""
Test Case 2: Triton RMSNorm
============================
验证 Triton RMSNorm 实现的正确性
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rmsnorm():
    """测试 RMSNorm 实现"""

    print("=" * 60)
    print("Test 2: RMSNorm")
    print("=" * 60)

    try:
        from solutions.rmsnorm import RMSNorm

        dim = 768
        batch_size = 8

        print(f"\nTest Config:")
        print(f"  dim: {dim}")
        print(f"  batch_size: {batch_size}")

        # Create model
        model = RMSNorm(dim)
        print(f"\n✓ Model created successfully")

        # Test forward pass
        x = torch.randn(batch_size, dim)
        output = model(x)

        print(f"\nForward Pass:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        # Compare with PyTorch RMSNorm
        reference = nn.RMSNorm(dim, eps=1e-6)
        reference.weight.data = model.weight.data.clone()
        with torch.no_grad():
            expected = reference(x)

        diff = (output - expected).abs().max()
        print(f"\nAccuracy:")
        print(f"  Max difference from PyTorch RMSNorm: {diff:.6e}")

        if diff < 1e-5:
            print(f"  ✓ Accuracy within tolerance")
        else:
            print(f"  ✗ Accuracy exceeds tolerance 1e-5")
            return False

        # Test gradient flow
        loss = output.sum()
        loss.backward()

        if model.weight.grad is not None:
            print(f"\n  ✓ Gradients computed")
        else:
            print(f"\n  ✗ No gradients computed")
            return False

        print(f"\n{'=' * 60}")
        print(f"Test Result: PASSED")
        print(f"{'=' * 60}")
        return True

    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_rmsnorm()
    sys.exit(0 if success else 1)
