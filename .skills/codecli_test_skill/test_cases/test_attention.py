"""
Test Case 1: Multi-Head Attention
==================================
验证 Multi-Head Attention 实现的正确性
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_multi_head_attention():
    """测试 Multi-Head Attention 实现"""

    print("=" * 60)
    print("Test 1: Multi-Head Attention")
    print("=" * 60)

    try:
        # Import implementation
        from solutions.multi_head_attention import MultiHeadAttention

        # Test parameters
        d_model = 512
        n_heads = 8
        batch_size = 4
        seq_len = 32

        print(f"\nTest Config:")
        print(f"  d_model: {d_model}")
        print(f"  n_heads: {n_heads}")
        print(f"  batch_size: {batch_size}")
        print(f"  seq_len: {seq_len}")

        # Create model
        model = MultiHeadAttention(d_model, n_heads)
        print(f"\n✓ Model created successfully")

        # Test forward pass
        x = torch.randn(batch_size, seq_len, d_model)
        output = model(x, x, x, causal=True)

        print(f"\nForward Pass:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        # Check output shape
        expected_shape = (batch_size, seq_len, d_model)
        if output.shape == expected_shape:
            print(f"  ✓ Output shape correct: {expected_shape}")
        else:
            print(f"  ✗ Output shape incorrect: expected {expected_shape}, got {output.shape}")
            return False

        # Test gradient flow
        loss = output.sum()
        loss.backward()

        # Check if gradients exist
        has_gradients = True
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"  ✗ No gradient for {name}")
                has_gradients = False

        if has_gradients:
            print(f"  ✓ All gradients computed successfully")

        # Check for NaN/Inf gradients
        has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param.grad).any():
                print(f"  ✗ NaN gradient in {name}")
                has_nan = True
            if torch.isinf(param.grad).any():
                print(f"  ✗ Inf gradient in {name}")
                has_nan = True

        if not has_nan:
            print(f"  ✓ No NaN/Inf gradients")

        print(f"\n{'=' * 60}")
        print(f"Test Result: PASSED")
        print(f"{'=' * 60}")
        return True

    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print(f"Please ensure solutions/multi_head_attention.py exists")
        return False
    except Exception as e:
        print(f"\n✗ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_multi_head_attention()
    sys.exit(0 if success else 1)
