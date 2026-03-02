---
name: ai-operator-development
description: Universal AI model verification and operator generator. Parses PyTorch models, generates Triton/CUDA operators, verifies correctness, profiles performance, and generates documentation. Auto-detects model types (Transformer, CNN, MoE, LLaMA, GPT, BERT) and adapts verification accordingly.
license: MIT
---

# AI Operator Development

Universal tool for AI model verification, operator generation, and performance analysis.

## Features

- **Model Verification**: 8-category test suite (syntax, structure, forward pass, gradients, variations, consistency, parameters, edge cases)
- **Operator Generation**: Triton/CUDA kernels for 10+ operator types (RMSNorm, LayerNorm, SiLU, GELU, Attention, MoE, etc.)
- **Performance Profiling**: Timing, memory, FLOPs, throughput analysis
- **Documentation**: Auto-generates model design, operator design, verification reports, performance reports
- **Universal Adapter**: Auto-detects model types and infers initialization parameters

## Usage

### Command Line

```bash
python .skills/ai-operator-development/ai_operator_development.py model.py
```

### Python API

```python
from .skills.ai_operator_development import build

result = build(
    model_file="model2.py",
    vocab_size=1000,
    config={'batch_size': 2, 'seq_len': 16, 'num_runs': 50}
)

# Access results
print(f"Model type: {result['model_type']}")
print(f"Performance: {result['performance']}")
```

## Generated Files

```
operators/<model_name>/
├── rmsnorm.py       # Triton + PyTorch fallback
├── silu.py          # Triton + PyTorch fallback
├── attention.py     # PyTorch implementation
├── moe.py           # MoE router
└── ...

docs/
├── model_design.md
├── operator_design.md
├── verification_report.md
└── performance.md
```

## Supported Operators

| Type | Operators |
|------|-----------|
| Normalization | RMSNorm, LayerNorm, BatchNorm |
| Activation | SiLU, GELU, ReLU, Softmax |
| Attention | Multi-head attention, QKV projection, RoPE |
| Linear | Dense layers, embeddings |
| MoE | Top-K routing, Expert router |
| Convolutional | Conv2D, MaxPool |

## Requirements

**Required**: Python 3.8+, PyTorch 1.10+

**Optional** (auto-detected):
- Triton (for GPU kernels)
- psutil (for memory profiling)
- thop (for FLOPs counting)

## Result Dictionary

```python
{
    'environment': {...},      # Python, packages, GPU info
    'model_type': str,         # 'transformer', 'cnn', 'moe', etc.
    'model_info': {...},       # Classes, operations detected
    'parameter_info': {...},   # Total parameters, model size
    'verification': {...},     # Test results
    'generated_operators': {...},  # Paths to generated files
    'performance': {...},      # Timing, memory, FLOPs, throughput
    'documentation': {...},    # Paths to generated docs
}
```
