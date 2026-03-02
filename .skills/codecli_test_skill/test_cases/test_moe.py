"""
Test Case 3: Mixture of Experts Router
=======================================
验证 MoE Router 实现的正确性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_moe_router():
    """测试 MoE Router 实现"""

    print("=" * 60)
    print("Test 3: MoE Router")
    print("=" * 60)

    try:
        from solutions.moe_router import MoERouter, MoEBlock

        d_model = 256
        d_ff = 512
        num_experts = 4
        top_k = 2
        batch_size = 4
        seq_len = 16

        print(f"\nTest Config:")
        print(f"  d_model: {d_model}")
        print(f"  num_experts: {num_experts}")
        print(f"  top_k: {top_k}")

        # Create router
        router = MoERouter(d_model, num_experts, top_k)
        print(f"\n✓ Router created successfully")

        # Test routing
        x_flat = torch.randn(batch_size * seq_len, d_model)
        dispatch_mask, expert_weights, load_balance_loss = router(x_flat)

        print(f"\nRouting:")
        print(f"  Dispatch mask shape: {dispatch_mask.shape}")
        print(f"  Expert weights shape: {expert_weights.shape}")
        print(f"  Load balance loss: {load_balance_loss.item():.4f}")

        # Check Top-K selection
        selected_per_token = (dispatch_mask > 0).sum(dim=1)
        if (selected_per_token == top_k).all():
            print(f"  ✓ Top-K selection correct")
        else:
            print(f"  ✗ Top-K selection incorrect")
            return False

        # Test full MoE block
        moe = MoEBlock(d_model, d_ff, num_experts, top_k)
        x = torch.randn(batch_size, seq_len, d_model)
        output, aux_loss = moe(x)

        print(f"\nMoE Block:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Auxiliary loss: {aux_loss.item():.4f}")

        if output.shape == x.shape:
            print(f"  ✓ Output shape correct")
        else:
            print(f"  ✗ Output shape incorrect")
            return False

        # Check gradient flow
        loss = output.sum() + aux_loss
        loss.backward()

        grad_count = sum(1 for p in moe.parameters() if p.grad is not None)
        total_params = sum(1 for p in moe.parameters())
        print(f"  Gradients: {grad_count}/{total_params} parameters")

        if grad_count == total_params:
            print(f"  ✓ All gradients computed")
        else:
            print(f"  ✗ Some gradients missing")
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
    success = test_moe_router()
    sys.exit(0 if success else 1)
