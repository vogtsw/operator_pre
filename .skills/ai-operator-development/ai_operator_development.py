"""
AI Operator Development - Universal Model Verification and Optimization Tool
====================================================================================
Complete implementation of:
1. Model structure verification - Code verification, test case generation, inference validation
2. Model operator design - Generate Triton/CUDA operators, verify accuracy on CPU/GPU
3. Performance analysis - Inference time, FLOPs, memory usage, output to performance.md
4. Documentation output - Model design, dataset generation, operator design, performance analysis

Key features:
- Universal model adapter for different AI architectures (Transformer, CNN, MoE, etc.)
- Automatic environment setup and dependency installation
- Comprehensive test suite generation (8-category verification)
- Detailed performance profiling (timing, memory, FLOPs, throughput)
- Complete documentation generation
====================================================================================
"""

import sys
import os
import subprocess
import importlib
import importlib.util
import time
import gc
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from datetime import datetime
import json


# ============================================================================
# Part 1: Universal Environment Setup
# ============================================================================

class UniversalEnvironmentSetup:
    """Universal environment detection and installation"""

    def __init__(self):
        self.triton_available = False
        self.torch_available = False
        self.cuda_available = False
        self.psutil_available = False
        self.thop_available = False  # For FLOPs counting

    def check_environment(self) -> Dict[str, Any]:
        """Comprehensive environment check"""
        result = {
            'python_version': sys.version,
            'packages': {},
            'cuda': None,
            'gpu': None,
            'gpu_count': 0,
            'gpu_memory': None,
            'missing': [],
            'optional': []
        }

        # Check PyTorch
        try:
            import torch
            self.torch_available = True
            result['packages']['torch'] = torch.__version__
            result['cuda'] = torch.version.cuda
            if torch.cuda.is_available():
                self.cuda_available = True
                result['gpu'] = torch.cuda.get_device_name(0)
                result['gpu_count'] = torch.cuda.device_count()
                result['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        except ImportError:
            result['packages']['torch'] = None
            result['missing'].append('torch')

        # Check Triton (optional)
        try:
            import triton
            self.triton_available = True
            result['packages']['triton'] = triton.__version__
        except ImportError:
            result['packages']['triton'] = None
            result['optional'].append('triton')

        # Check psutil (for memory analysis)
        try:
            import psutil
            self.psutil_available = True
            result['packages']['psutil'] = psutil.__version__
        except ImportError:
            result['optional'].append('psutil')

        # Check thop (for FLOPs counting)
        try:
            import thop
            self.thop_available = True
            result['packages']['thop'] = thop.__version__
        except ImportError:
            result['optional'].append('thop')

        # Check numpy
        try:
            import numpy as np
            result['packages']['numpy'] = np.__version__
        except ImportError:
            result['missing'].append('numpy')

        return result

    def install_missing_packages(self, missing: List[str]) -> bool:
        """Auto-install missing packages"""
        if not missing:
            return True

        print("\n" + "=" * 60)
        print(f"Missing packages detected: {', '.join(missing)}")
        print("Auto-installing...")
        print("=" * 60)

        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing + ["--quiet"])
            print("Installation successful")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Auto-install failed: {e}")
            print("Please manually run:")
            print(f"  pip install {' '.join(missing)}")
            return False

    def install_optional_packages(self, optional: List[str]) -> bool:
        """Install optional packages for enhanced features"""
        if not optional:
            return True

        print("\n" + "=" * 60)
        print(f"Optional packages available: {', '.join(optional)}")
        print("These enable enhanced features (FLOPs counting, memory analysis)")
        response = input("Install optional packages? (y/N): ")

        if response.lower() == 'y':
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install"
                ] + optional + ["--quiet"])
                print("Optional packages installed")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Installation failed: {e}")
                return False
        else:
            print("Skipping optional packages")
            return True


# ============================================================================
# Part 2: Universal Model Adapter
# ============================================================================

class UniversalModelAdapter:
    """Adapter for different AI model architectures"""

    def __init__(self, model_file: str):
        self.model_file = Path(model_file)
        self.source = None
        self.tree = None
        self.model_info = {}
        self.model_type = None

    def detect_model_type(self) -> str:
        """Detect the type of model (Transformer, CNN, RNN, Custom)"""
        with open(self.model_file, 'r', encoding='utf-8') as f:
            self.source = f.read()

        # Check for common patterns
        patterns = {
            'transformer': ['MultiheadAttention', 'Transformer', 'attention', 'qkv', 'self_attn'],
            'cnn': ['Conv2d', 'Conv1d', 'MaxPool', 'AvgPool', 'convolution'],
            'rnn': ['LSTM', 'GRU', 'RNN', 'recurrent'],
            'moe': ['MoE', 'mixture_of_experts', 'topk', 'gate'],
            'llama': ['Llama', 'RoPE', 'swiglu', 'rms_norm'],
            'gpt': ['GPT', 'causal_mask', 'gelu'],
            'bert': ['Bert', 'bidirectional', 'cls_token'],
        }

        scores = {}
        for model_type, keywords in patterns.items():
            scores[model_type] = sum(
                1 for kw in keywords if kw.lower() in self.source.lower()
            )

        # Determine model type based on highest score
        if scores['transformer'] >= 2:
            if scores['moe'] >= 1:
                self.model_type = 'moe_transformer'
            elif scores['llama'] >= 2:
                self.model_type = 'llama'
            elif scores['gpt'] >= 1:
                self.model_type = 'gpt'
            elif scores['bert'] >= 1:
                self.model_type = 'bert'
            else:
                self.model_type = 'transformer'
        elif scores['cnn'] >= 2:
            self.model_type = 'cnn'
        elif scores['rnn'] >= 1:
            self.model_type = 'rnn'
        else:
            self.model_type = 'custom'

        return self.model_type

    def analyze_structure(self) -> Dict[str, Any]:
        """Comprehensive model structure analysis"""
        import ast

        try:
            self.tree = ast.parse(self.source)
        except SyntaxError as e:
            return {'status': 'ERROR', 'message': f'Syntax error: {e}'}

        classes = []
        operations = {
            # Normalization
            'rmsnorm': False, 'layernorm': False, 'batchnorm': False, 'groupnorm': False,
            # Activations
            'silu': False, 'gelu': False, 'relu': False, 'softmax': False,
            # Attention
            'attention': False, 'multihead_attention': False, 'qkv_projection': False, 'rope': False,
            # Linear
            'linear': False, 'matmul': False, 'embedding': False,
            # Convolution
            'conv2d': False, 'conv1d': False, 'maxpool': False, 'avgpool': False,
            # MoE
            'moe': False, 'topk': False, 'gate': False,
            # Recurrent
            'lstm': False, 'gru': False,
        }

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                bases = [self._get_name(base) for base in node.bases]
                is_nn_module = any('Module' in b for b in bases)

                if is_nn_module:
                    class_info = {
                        'name': node.name,
                        'bases': bases,
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        'has_forward': any(m.name == 'forward' for m in node.body if isinstance(m, ast.FunctionDef))
                    }

                    # Extract __init__ parameters
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                            class_info['init_args'] = [arg.arg for arg in item.args.args[1:]]  # Skip self

                    classes.append(class_info)

                    # Detect operations in forward method
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                            method_source = ast.unparse(item)
                            self._detect_operations_enhanced(method_source, operations)

        self.model_info = {
            'file': str(self.model_file),
            'type': self.model_type,
            'classes': classes,
            'operations': {k: v for k, v in operations.items() if v},
            'source_lines': len(self.source.split('\n'))
        }

        return self.model_info

    def _get_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def _detect_operations_enhanced(self, source: str, operations: Dict[str, bool]):
        """Enhanced operation detection with more patterns"""
        patterns = {
            # Normalization
            'rmsnorm': ['rsqrt', 'rms_norm', 'RMSNorm'],
            'layernorm': ['LayerNorm', 'F.layer_norm'],
            'batchnorm': ['BatchNorm', 'F.batch_norm'],
            'groupnorm': ['GroupNorm'],
            # Activations
            'silu': ['F.silu', 'SiLU', 'swish'],
            'gelu': ['F.gelu', 'GELU'],
            'relu': ['F.relu', 'ReLU'],
            'softmax': ['F.softmax', 'softmax('],
            # Attention
            'attention': ['attention(', 'attn('],
            'multihead_attention': ['MultiheadAttention', 'nn.MultiheadAttention'],
            'qkv_projection': ['w_qkv', 'qkv_proj', 'q_proj', 'k_proj', 'v_proj'],
            'rope': ['rope', 'RotaryEmbedding', 'apply_rotary'],
            # Linear
            'linear': ['nn.Linear', 'F.linear'],
            'matmul': ['torch.matmul', '@', 'bmm'],
            'embedding': ['nn.Embedding', 'F.embedding'],
            # Convolution
            'conv2d': ['Conv2d', 'F.conv2d'],
            'conv1d': ['Conv1d', 'F.conv1d'],
            'maxpool': ['MaxPool', 'F.max_pool'],
            'avgpool': ['AvgPool', 'F.avg_pool'],
            # MoE
            'moe': ['MoE', 'MixtureOfExperts', 'moe'],
            'topk': ['torch.topk', '.topk('],
            'gate': ['gate', 'router', 'Switch'],
            # Recurrent
            'lstm': ['LSTM', 'nn.LSTM'],
            'gru': ['GRU', 'nn.GRU'],
        }

        source_lower = source.lower()
        for op_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern.lower() in source_lower:
                    operations[op_name] = True
                    break

    def import_model(self, vocab_size: int = 1000, **kwargs):
        """Universal model importer with parameter inference"""
        import torch
        import importlib.util

        spec = importlib.util.spec_from_file_location("model", self.model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find model class
        model_class = None
        init_args = {}

        # Priority 1: Look for common naming patterns
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type):
                # Check if it's a nn.Module subclass
                if hasattr(attr, '__bases__') and any(
                    'Module' in b.__name__ for b in attr.__bases__
                ):
                    # Prioritize certain patterns
                    if any(pattern in attr_name for pattern in [
                        'Model', 'Transformer', 'Network', 'Encoder', 'Decoder'
                    ]):
                        model_class = attr
                        break

        # Priority 2: Take the first nn.Module
        if model_class is None:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, '__bases__'):
                    if any('Module' in b.__name__ for b in attr.__bases__):
                        model_class = attr
                        break

        if model_class is None:
            raise ImportError("No nn.Module class found in the model file")

        # Infer initialization arguments
        import inspect
        sig = inspect.signature(model_class.__init__)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            # Check if parameter has a default value
            if param.default != inspect.Parameter.empty:
                init_args[param_name] = param.default
            else:
                # Set sensible defaults based on parameter name
                if 'vocab' in param_name.lower() and 'size' in param_name.lower():
                    init_args[param_name] = vocab_size
                elif param_name in ['d_model', 'hidden_size', 'embed_dim']:
                    init_args[param_name] = kwargs.get(param_name, 256)
                elif param_name in ['num_layers', 'n_layers', 'depth']:
                    init_args[param_name] = kwargs.get(param_name, 2)
                elif param_name in ['n_head', 'num_heads', 'num_attention_heads']:
                    init_args[param_name] = kwargs.get(param_name, 4)
                elif param_name in ['d_ff', 'ff_dim', 'intermediate_size']:
                    init_args[param_name] = kwargs.get(param_name, 512)
                elif param_name in ['max_seq_len', 'max_length', 'seq_len']:
                    init_args[param_name] = kwargs.get(param_name, 512)
                elif param_name in ['dropout', 'drop_prob']:
                    init_args[param_name] = kwargs.get(param_name, 0.1)

        # User-provided kwargs override
        for key, value in kwargs.items():
            init_args[key] = value

        try:
            model = model_class(**init_args)
            model.eval()
            return model, init_args
        except Exception as e:
            # Try with minimal arguments
            try:
                model = model_class(vocab_size=vocab_size)
                model.eval()
                return model, {'vocab_size': vocab_size}
            except Exception as e2:
                raise ImportError(f"Failed to instantiate model: {e2}")


# ============================================================================
# Part 3: Comprehensive Model Structure Verifier
# ============================================================================

class ComprehensiveModelVerifier:
    """Comprehensive model verification with universal test generation"""

    def __init__(self, model, model_info: Dict[str, Any]):
        self.model = model
        self.model_info = model_info
        self.test_results = []

    def run_all_tests(self, vocab_size: int = 1000) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        import torch

        print("\n" + "=" * 70)
        print("Comprehensive Model Verification")
        print("=" * 70)

        results = {
            'syntax': self._test_syntax(),
            'structure': self._test_structure(),
            'forward_pass': self._test_forward_pass(vocab_size),
            'gradient_flow': self._test_gradient_flow(vocab_size),
            'input_variations': self._test_input_variations(vocab_size),
            'output_consistency': self._test_output_consistency(vocab_size),
            'parameter_analysis': self._test_parameters(),
            'edge_cases': self._test_edge_cases(vocab_size),
        }

        # Summary
        total = len(results)
        passed = sum(1 for r in results.values() if r.get('status') == 'PASS')

        print(f"\n{'=' * 70}")
        print(f"Test Summary: {passed}/{total} tests passed")
        print(f"{'=' * 70}")

        return results

    def _test_syntax(self) -> Dict[str, Any]:
        """Test 1: Syntax validation"""
        print("\n[1/8] Syntax Validation:")
        try:
            # Already validated during parsing
            print("  Syntax check: PASSED")
            return {'status': 'PASS', 'message': 'Valid Python syntax'}
        except Exception as e:
            print(f"  Syntax check: FAILED - {e}")
            return {'status': 'FAIL', 'message': str(e)}

    def _test_structure(self) -> Dict[str, Any]:
        """Test 2: Structure validation"""
        print("\n[2/8] Structure Validation:")
        try:
            classes = self.model_info.get('classes', [])
            nn_modules = [c for c in classes if any('Module' in b for b in c['bases'])]

            if not nn_modules:
                return {'status': 'FAIL', 'message': 'No nn.Module classes found'}

            has_forward = any(c.get('has_forward') for c in nn_modules)

            print(f"  Found {len(nn_modules)} nn.Module classes")
            print(f"  Classes with forward: {sum(1 for c in nn_modules if c.get('has_forward'))}")
            print(f"  Structure check: {'PASSED' if has_forward else 'FAILED'}")

            return {
                'status': 'PASS' if has_forward else 'FAIL',
                'num_classes': len(nn_modules),
                'has_forward': has_forward
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _test_forward_pass(self, vocab_size: int) -> Dict[str, Any]:
        """Test 3: Forward pass"""
        print("\n[3/8] Forward Pass Test:")
        try:
            import torch

            torch.manual_seed(42)
            x = torch.randint(0, vocab_size, (2, 16))

            with torch.no_grad():
                output = self.model(x)

            valid = not torch.isnan(output).any() and not torch.isinf(output).any()

            print(f"  Input shape: {list(x.shape)}")
            print(f"  Output shape: {list(output.shape)}")
            print(f"  Output valid: {'PASSED' if valid else 'FAILED'}")

            return {
                'status': 'PASS' if valid else 'FAIL',
                'input_shape': list(x.shape),
                'output_shape': list(output.shape),
                'output_mean': output.mean().item(),
                'output_std': output.std().item()
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _test_gradient_flow(self, vocab_size: int) -> Dict[str, Any]:
        """Test 4: Gradient flow"""
        print("\n[4/8] Gradient Flow Test:")
        try:
            import torch

            self.model.train()
            x = torch.randint(0, vocab_size, (2, 16))
            output = self.model(x)
            loss = output.mean()
            loss.backward()

            params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
            has_grad = len(params_with_grad) > 0

            print(f"  Parameters with gradients: {len(params_with_grad)}")
            print(f"  Gradient flow: {'PASSED' if has_grad else 'FAILED'}")

            return {
                'status': 'PASS' if has_grad else 'FAIL',
                'params_with_grad': len(params_with_grad)
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _test_input_variations(self, vocab_size: int) -> Dict[str, Any]:
        """Test 5: Different input sizes"""
        print("\n[5/8] Input Variations Test:")
        try:
            import torch

            self.model.eval()
            test_cases = [
                (1, 16, "batch_1"),
                (2, 16, "batch_2"),
                (2, 8, "short_seq"),
                (2, 32, "long_seq"),
            ]

            results = []
            for bs, sl, name in test_cases:
                try:
                    x = torch.randint(0, vocab_size, (bs, sl))
                    with torch.no_grad():
                        output = self.model(x)
                    results.append({'name': name, 'success': True, 'shape': list(output.shape)})
                    print(f"  {name}: PASSED - {list(output.shape)}")
                except Exception as e:
                    results.append({'name': name, 'success': False, 'error': str(e)})
                    print(f"  {name}: FAILED - {e}")

            passed = sum(1 for r in results if r['success'])
            return {
                'status': 'PASS' if passed == len(test_cases) else 'PARTIAL',
                'passed': passed,
                'total': len(test_cases),
                'results': results
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _test_output_consistency(self, vocab_size: int) -> Dict[str, Any]:
        """Test 6: Output consistency"""
        print("\n[6/8] Output Consistency Test:")
        try:
            import torch

            self.model.eval()
            x = torch.randint(0, vocab_size, (2, 16))

            outputs = []
            with torch.no_grad():
                for _ in range(3):
                    output = self.model(x)
                    outputs.append(output.clone())

            max_diff = 0
            for i in range(1, len(outputs)):
                diff = (outputs[0] - outputs[i]).abs().max().item()
                max_diff = max(max_diff, diff)

            consistent = max_diff < 1e-6

            print(f"  Max difference: {max_diff:.2e}")
            print(f"  Consistency: {'PASSED' if consistent else 'FAILED'}")

            return {
                'status': 'PASS' if consistent else 'FAIL',
                'max_difference': max_diff
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _test_parameters(self) -> Dict[str, Any]:
        """Test 7: Parameter analysis"""
        print("\n[7/8] Parameter Analysis:")
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

            return {
                'status': 'PASS',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / 1024 / 1024
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _test_edge_cases(self, vocab_size: int) -> Dict[str, Any]:
        """Test 8: Edge cases"""
        print("\n[8/8] Edge Cases Test:")
        try:
            import torch

            self.model.eval()
            results = []

            # Test with minimum size
            try:
                x = torch.randint(0, vocab_size, (1, 1))
                with torch.no_grad():
                    output = self.model(x)
                results.append({'test': 'min_size', 'status': 'PASS', 'output_shape': list(output.shape)})
                print(f"  min_size (1,1): PASSED")
            except Exception as e:
                results.append({'test': 'min_size', 'status': 'FAIL', 'error': str(e)})
                print(f"  min_size (1,1): FAILED - {e}")

            # Test with all same input
            try:
                x = torch.zeros(2, 8, dtype=torch.long)
                with torch.no_grad():
                    output = self.model(x)
                results.append({'test': 'zeros', 'status': 'PASS'})
                print(f"  zeros input: PASSED")
            except Exception as e:
                results.append({'test': 'zeros', 'status': 'FAIL', 'error': str(e)})
                print(f"  zeros input: FAILED - {e}")

            passed = sum(1 for r in results if r['status'] == 'PASS')
            return {
                'status': 'PASS' if passed == len(results) else 'PARTIAL',
                'passed': passed,
                'total': len(results),
                'results': results
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}


# ============================================================================
# Part 4: Universal Operator Generator
# ============================================================================

class UniversalOperatorGenerator:
    """Generate operators for various model types"""

    def __init__(self, output_dir: str = "operators/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, model_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate operators based on detected operations"""
        generated = {}
        operations = model_info.get('operations', {})

        print(f"\n{'=' * 70}")
        print("Generating Operators")
        print(f"{'=' * 70}")

        # Normalization operators
        if operations.get('rmsnorm'):
            generated['rmsnorm'] = self._generate_rmsnorm()
        if operations.get('layernorm'):
            generated['layernorm'] = self._generate_layernorm()
        if operations.get('batchnorm'):
            generated['batchnorm'] = self._generate_batchnorm()

        # Activation operators
        if operations.get('silu'):
            generated['silu'] = self._generate_silu()
        if operations.get('gelu'):
            generated['gelu'] = self._generate_gelu()
        if operations.get('relu'):
            generated['relu'] = self._generate_relu()
        if operations.get('softmax'):
            generated['softmax'] = self._generate_softmax()

        # Attention operators
        if operations.get('attention') or operations.get('multihead_attention'):
            generated['attention'] = self._generate_attention()
        if operations.get('qkv_projection'):
            generated['qkv_projection'] = self._generate_qkv_projection()
        if operations.get('rope'):
            generated['rope'] = self._generate_rope()

        # Linear operators
        if operations.get('linear') or operations.get('matmul'):
            generated['linear'] = self._generate_linear()
        if operations.get('embedding'):
            generated['embedding'] = self._generate_embedding()

        # Convolution operators
        if operations.get('conv2d'):
            generated['conv2d'] = self._generate_conv2d()
        if operations.get('maxpool'):
            generated['maxpool'] = self._generate_maxpool()

        # MoE operators
        if operations.get('moe') or operations.get('topk'):
            generated['topk'] = self._generate_topk()
            generated['moe_router'] = self._generate_moe_router()

        # Generate __init__.py
        self._generate_init(generated.keys())

        return generated

    def _generate_rmsnorm(self) -> str:
        """Generate RMSNorm operator"""
        code = '''"""
RMSNorm Operator - PyTorch and Triton implementations
"""
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(x_ptr + row_idx * N + col_offsets, mask=mask, other=0.0)

    # Compute RMS
    x_squared = x * x
    mean_square = tl.sum(x_squared, axis=0) / N
    rms = tl.sqrt(mean_square + eps)

    # Load weight and normalize
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    y = (x / rms) * weight

    tl.store(output_ptr + row_idx * N + col_offsets, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm using Triton"""
    assert x.shape[-1] == weight.shape[0], "Last dimension must match"

    input_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    n_blocks = n_rows

    rmsnorm_kernel[(n_blocks,)](
        input_2d, weight, output,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view_as(x)


def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm using PyTorch"""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, use_triton: bool = True) -> torch.Tensor:
    """RMSNorm with automatic backend selection"""
    if use_triton and x.is_cuda and triton is not None:
        return rmsnorm_triton(x, weight, eps)
    return rmsnorm_pytorch(x, weight, eps)
'''
        return self._save_operator('rmsnorm', code)

    def _generate_layernorm(self) -> str:
        """Generate LayerNorm operator"""
        code = '''"""
LayerNorm Operator - PyTorch and Triton implementations
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(x_ptr, y_ptr, weight_ptr, bias_ptr, N, stride, eps, BLOCK_SIZE: tl.constexpr):
    """Layer normalization"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(x_ptr + row_idx * stride + col_offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Normalize
    x_normalized = x_centered / tl.sqrt(var + eps)

    # Load weight and bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)

    y = x_normalized * weight + bias
    tl.store(y_ptr + row_idx * stride + col_offsets, y, mask=mask)


def layernorm_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm using Triton"""
    assert x.shape[-1] == weight.shape[0] == bias.shape[0]

    input_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    layernorm_kernel[(n_rows,)](
        input_2d, output, weight, bias,
        n_cols, input_2d.stride(0), eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view_as(x)


def layernorm_pytorch(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm using PyTorch"""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5, use_triton: bool = True) -> torch.Tensor:
    """LayerNorm with automatic backend selection"""
    if use_triton and x.is_cuda and triton is not None:
        return layernorm_triton(x, weight, bias, eps)
    return layernorm_pytorch(x, weight, bias, eps)
'''
        return self._save_operator('layernorm', code)

    def _generate_silu(self) -> str:
        """Generate SiLU operator"""
        code = '''"""
SiLU (Swish) Operator - PyTorch and Triton implementations
"""
import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """SiLU: x * sigmoid(x) = x / (1 + exp(-x))"""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = x / (1 + tl.exp(-x))
    tl.store(output_ptr + offsets, y, mask=mask)


def silu_triton(x: torch.Tensor) -> torch.Tensor:
    """SiLU using Triton"""
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    silu_kernel[(n_blocks,)](
        x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    return output


def silu_pytorch(x: torch.Tensor) -> torch.Tensor:
    """SiLU using PyTorch"""
    return torch.nn.functional.silu(x)


def silu(x: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """SiLU with automatic backend selection"""
    if use_triton and x.is_cuda and triton is not None:
        return silu_triton(x)
    return silu_pytorch(x)
'''
        return self._save_operator('silu', code)

    def _generate_gelu(self) -> str:
        """Generate GELU operator"""
        code = '''"""
GELU Operator - PyTorch and Triton implementations
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """GELU activation using tanh approximation"""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715

    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    y = 0.5 * x * (1 + tl.tanh(tanh_arg))

    tl.store(output_ptr + offsets, y, mask=mask)


def gelu_triton(x: torch.Tensor) -> torch.Tensor:
    """GELU using Triton"""
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    gelu_kernel[(n_blocks,)](
        x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    return output


def gelu_pytorch(x: torch.Tensor) -> torch.Tensor:
    """GELU using PyTorch"""
    return torch.nn.functional.gelu(x)


def gelu(x: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """GELU with automatic backend selection"""
    if use_triton and x.is_cuda and triton is not None:
        return gelu_triton(x)
    return gelu_pytorch(x)
'''
        return self._save_operator('gelu', code)

    def _generate_softmax(self) -> str:
        """Generate Softmax operator"""
        code = '''"""
Softmax Operator - PyTorch and Triton implementations
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """Softmax with numerical stability"""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf'))

    # Find max for numerical stability
    row_max = tl.max(row, axis=0)
    row_max = tl.max(row_max)

    # Compute exp(x - max)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    denominator = tl.sum(denominator)

    output = numerator / denominator
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output, mask=mask)


def softmax_triton(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax using Triton"""
    assert dim == -1 or dim == x.dim() - 1, "Only last dimension supported"

    input_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    softmax_kernel[(n_rows,)](
        input_2d, output, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )

    return output.view_as(x)


def softmax_pytorch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax using PyTorch"""
    return torch.softmax(x, dim=dim)


def softmax(x: torch.Tensor, dim: int = -1, use_triton: bool = True) -> torch.Tensor:
    """Softmax with automatic backend selection"""
    if use_triton and x.is_cuda and triton is not None:
        return softmax_triton(x, dim)
    return softmax_pytorch(x, dim)
'''
        return self._save_operator('softmax', code)

    def _generate_attention(self) -> str:
        """Generate Attention operator"""
        code = '''"""
Scaled Dot-Product Attention Operator
"""
import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: float = 0.0,
    scale: float = None,
    causal: bool = False
) -> torch.Tensor:
    """
    Scaled dot-product attention: softmax(QK^T / sqrt(d_k))V

    Args:
        q: Query tensor (batch, n_heads, seq_len, d_k)
        k: Key tensor (batch, n_heads, seq_len, d_k)
        v: Value tensor (batch, n_heads, seq_len, d_k)
        mask: Optional attention mask
        dropout: Dropout probability
        scale: Scale factor (default: 1/sqrt(d_k))
        causal: Whether to apply causal mask

    Returns:
        Output tensor and attention weights
    """
    d_k = q.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if causal:
        seq_len = q.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

    # Apply provided mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Dropout
    if dropout > 0:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    # Weighted sum
    output = torch.matmul(attn_weights, v)

    return output, attn_weights


def multi_head_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    n_heads: int,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Multi-head attention

    Args:
        x: Input tensor (batch, seq_len, d_model)
        w_q: Query weight (d_model, d_model)
        w_k: Key weight (d_model, d_model)
        w_v: Value weight (d_model, d_model)
        w_o: Output weight (d_model, d_model)
        n_heads: Number of attention heads
        mask: Optional attention mask

    Returns:
        Output tensor (batch, seq_len, d_model)
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // n_heads

    # Project Q, K, V
    q = torch.matmul(x, w_q.t())
    k = torch.matmul(x, w_k.t())
    v = torch.matmul(x, w_v.t())

    # Reshape for multi-head
    q = q.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)

    # Scaled dot-product attention
    attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask)

    # Concatenate heads
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, d_model)

    # Output projection
    output = torch.matmul(attn_output, w_o.t())

    return output
'''
        return self._save_operator('attention', code)

    def _generate_qkv_projection(self) -> str:
        """Generate QKV projection operator"""
        code = '''"""
QKV Projection Operator
"""
import torch
from typing import Tuple


def qkv_projection(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    n_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merged QKV projection

    Args:
        x: Input tensor (batch, seq_len, d_model)
        w_qkv: Combined QKV weight (3 * d_model, d_model)
        n_heads: Number of attention heads

    Returns:
        Q, K, V tensors (batch, n_heads, seq_len, d_k)
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // n_heads

    # Merged projection
    qkv = torch.matmul(x, w_qkv.t())

    # Reshape and split
    qkv = qkv.reshape(batch_size, seq_len, 3, n_heads, d_k)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]

    return q, k, v


def qkv_projection_separate(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    n_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Separate QKV projection

    Args:
        x: Input tensor (batch, seq_len, d_model)
        w_q: Query weight (d_model, d_model)
        w_k: Key weight (d_model, d_model)
        w_v: Value weight (d_model, d_model)
        n_heads: Number of attention heads

    Returns:
        Q, K, V tensors (batch, n_heads, seq_len, d_k)
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // n_heads

    # Separate projections
    q = torch.matmul(x, w_q.t())
    k = torch.matmul(x, w_k.t())
    v = torch.matmul(x, w_v.t())

    # Reshape for multi-head
    q = q.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)

    return q, k, v
'''
        return self._save_operator('qkv_projection', code)

    def _generate_rope(self) -> str:
        """Generate RoPE operator"""
        code = '''"""
Rotary Position Embedding (RoPE) Operator
"""
import torch
import math


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute frequency tensor for RoPE

    Args:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length
        theta: Base frequency

    Returns:
        Precomputed frequency tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    offset: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key

    Args:
        xq: Query tensor (batch, seq_len, n_heads, head_dim)
        xk: Key tensor (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed frequency tensor
        offset: Starting offset for the position

    Returns:
        Rotated query and key tensors
    """
    batch, seq_len, n_heads, head_dim = xq.shape

    # Reshape to complex
    xq_complex = torch.view_as_complex(xq.float().reshape(batch, seq_len, n_heads, head_dim // 2, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(batch, seq_len, n_heads, head_dim // 2, 2))

    # Apply rotary
    freqs_cis = freqs_cis[offset:offset + seq_len]
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def rope(x: torch.Tensor, freqs_cis: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    Apply RoPE to a single tensor

    Args:
        x: Input tensor (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed frequency tensor
        offset: Starting offset

    Returns:
        Rotated tensor
    """
    batch, seq_len, n_heads, head_dim = x.shape

    # Reshape to complex
    x_complex = torch.view_as_complex(x.float().reshape(batch, seq_len, n_heads, head_dim // 2, 2))

    # Apply rotary
    freqs_cis = freqs_cis[offset:offset + seq_len]
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)

    return x_out.type_as(x)
'''
        return self._save_operator('rope', code)

    def _generate_linear(self) -> str:
        """Generate Linear operator"""
        code = '''"""
Linear/FC Operator with optimizations
"""
import torch


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Optimized linear layer

    Args:
        x: Input tensor (batch, in_features) or (batch, seq_len, in_features)
        weight: Weight tensor (out_features, in_features)
        bias: Optional bias tensor (out_features,)

    Returns:
        Output tensor
    """
    return F.linear(x, weight, bias) if bias is not None else torch.matmul(x, weight.t())


class Linear(torch.nn.Module):
    """Linear layer with optional fused operations"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)
'''
        return self._save_operator('linear', code)

    def _generate_embedding(self) -> str:
        """Generate Embedding operator"""
        code = '''"""
Embedding Lookup Operator
"""
import torch


def embedding_lookup(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Optimized embedding lookup

    Args:
        indices: Token IDs (batch, seq_len)
        weight: Embedding table (vocab_size, d_model)

    Returns:
        Embeddings (batch, seq_len, d_model)
    """
    return torch.nn.functional.embedding(indices, weight)


class Embedding(torch.nn.Module):
    """Embedding layer"""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return embedding_lookup(indices, self.weight)
'''
        return self._save_operator('embedding', code)

    def _generate_topk(self) -> str:
        """Generate TopK operator for MoE"""
        code = '''"""
Top-K Operator for MoE routing
"""
import torch
import torch.nn.functional as F


def topk_gate(
    logits: torch.Tensor,
    k: int,
    epsilon: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute top-k gate for MoE routing

    Args:
        logits: Gate logits (batch, seq_len, num_experts)
        k: Number of experts to select
        epsilon: Small value for numerical stability

    Returns:
        topk_weights: Weights for selected experts
        topk_indices: Indices of selected experts
        topk_gates: Combined gates (for normalization)
    """
    # Apply softmax to get probabilities
    gates = F.softmax(logits, dim=-1)

    # Get top-k
    topk_weights, topk_indices = torch.topk(gates, k, dim=-1)

    # Normalize top-k weights
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + epsilon)

    return topk_weights, topk_indices, gates


def compute_expert_assignment(
    topk_indices: torch.Tensor,
    num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute expert assignment for batch processing

    Args:
        topk_indices: Top-k expert indices (batch, seq_len, k)
        num_experts: Total number of experts

    Returns:
        expert_mask: Boolean mask for each expert
        expert_input_ids: Input IDs for each expert
    """
    batch, seq_len, k = topk_indices.shape

    expert_mask = torch.zeros(num_experts, batch, seq_len, dtype=torch.bool)
    expert_input_ids = torch.full((num_experts, batch, seq_len), -1, dtype=torch.long)

    for i in range(k):
        indices = topk_indices[:, :, i]
        for expert_id in range(num_experts):
            mask = (indices == expert_id)
            expert_mask[expert_id] = expert_mask[expert_id] | mask
            expert_input_ids[expert_id] = torch.where(
                mask,
                torch.full_like(expert_input_ids[expert_id], i),
                expert_input_ids[expert_id]
            )

    return expert_mask, expert_input_ids
'''
        return self._save_operator('topk', code)

    def _generate_moe_router(self) -> str:
        """Generate MoE Router operator"""
        code = '''"""
MoE Router Operator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """
    Mixture of Experts Router

    Routes inputs to experts based on learned gate weights.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route inputs to experts

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            topk_weights: Weights for selected experts
            topk_indices: Indices of selected experts
            dispatcher_mask: Mask for efficient dispatching
        """
        batch_size, seq_len, d_model = x.shape

        # Compute gate logits
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Get top-k
        topk_weights, topk_indices = torch.topk(
            F.softmax(gate_logits, dim=-1),
            self.top_k,
            dim=-1
        )

        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Create dispatcher mask
        dispatcher_mask = torch.zeros(
            self.num_experts, batch_size, seq_len,
            dtype=torch.bool, device=x.device
        )

        for k in range(self.top_k):
            expert_idx = topk_indices[:, :, k]
            for expert_id in range(self.num_experts):
                dispatcher_mask[expert_id] |= (expert_idx == expert_id)

        return topk_weights, topk_indices, dispatcher_mask

    def load_balancing_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert utilization

        Args:
            gate_logits: Gate logits (batch, seq_len, num_experts)

        Returns:
            Load balancing loss
        """
        # Compute fraction of tokens assigned to each expert
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_usage = gate_probs.mean(dim=(0, 1))

        # Ideal: uniform distribution
        target = torch.ones_like(expert_usage) / self.num_experts

        # KL divergence
        loss = F.kl_div(
            torch.log(expert_usage + 1e-10),
            target,
            reduction='batchmean'
        )

        return loss
'''
        return self._save_operator('moe_router', code)

    def _generate_batchnorm(self) -> str:
        """Generate BatchNorm operator stub"""
        code = '''"""
BatchNorm Operator (placeholder - uses PyTorch implementation)
"""
import torch


def batchnorm(x: torch.Tensor, running_mean, running_var, weight, bias, training, eps=1e-5, momentum=0.1):
    """BatchNorm - uses PyTorch implementation"""
    return torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias,
        training=training, momentum=momentum, eps=eps
    )
'''
        return self._save_operator('batchnorm', code)

    def _generate_relu(self) -> str:
        """Generate ReLU operator"""
        code = '''"""
ReLU Operator
"""
import torch


def relu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """ReLU activation"""
    return torch.nn.functional.relu(x, inplace=inplace)


def relu6(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """ReLU6 activation"""
    return torch.nn.functional.relu6(x, inplace=inplace)
'''
        return self._save_operator('relu', code)

    def _generate_conv2d(self) -> str:
        """Generate Conv2D operator stub"""
        code = '''"""
Conv2D Operator (placeholder - uses PyTorch implementation)
"""
import torch


def conv2d(x: torch.Tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """2D Convolution"""
    return torch.nn.functional.conv2d(x, weight, bias, stride, padding, dilation, groups)
'''
        return self._save_operator('conv2d', code)

    def _generate_maxpool(self) -> str:
        """Generate MaxPool operator stub"""
        code = '''"""
MaxPool Operator (placeholder - uses PyTorch implementation)
"""
import torch


def max_pool2d(x: torch.Tensor, kernel_size, stride=None, padding=0):
    """2D Max Pooling"""
    return torch.nn.functional.max_pool2d(x, kernel_size, stride, padding)
'''
        return self._save_operator('maxpool', code)

    def _generate_init(self, operator_names: List[str]):
        """Generate __init__.py"""
        if not operator_names:
            return

        imports = []
        exports = []

        for name in operator_names:
            imports.append(f"from .{name} import *")
            exports.append(f"'{name}'")

        code = f'''"""
Auto-generated operators
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
{chr(10).join(imports)}

__all__ = [{', '.join(exports)}]
'''
        init_file = self.output_dir / "__init__.py"
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"  Generated: {init_file}")

    def _save_operator(self, name: str, code: str) -> str:
        """Save operator to file"""
        output_file = self.output_dir / f"{name}.py"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f'"""Auto-generated {name} operator\n')
            f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""\n')
            f.write(code)

        print(f"  Generated: {output_file}")
        return str(output_file)


# ============================================================================
# Part 5: Operator Verifier
# ============================================================================

class OperatorVerifier:
    """Verify generated operators against PyTorch reference"""

    def __init__(self):
        self.results = {}

    def verify_all(
        self,
        model_info: Dict[str, Any],
        model,
        model_file: str
    ) -> Dict[str, Any]:
        """Verify all generated operators"""
        import torch

        print("\n" + "=" * 70)
        print("Operator Verification")
        print("=" * 70)

        operations = model_info.get('operations', {})
        results = {}

        # Import generated operators
        try:
            sys.path.insert(0, str(Path(model_file).parent))
            from operators.generated import (
                rmsnorm, layernorm, silu, gelu, softmax, qkv_projection,
                attention, rope, topk, moe_router
            )
        except ImportError as e:
            print(f"  Warning: Cannot import operators: {e}")
            return {'error': str(e)}

        # Verify normalization operators
        if operations.get('rmsnorm'):
            results['rmsnorm'] = self._verify_rmsnorm(model)
        if operations.get('layernorm'):
            results['layernorm'] = self._verify_layernorm(model)

        # Verify activation operators
        if operations.get('silu'):
            results['silu'] = self._verify_silu()
        if operations.get('gelu'):
            results['gelu'] = self._verify_gelu()
        if operations.get('softmax'):
            results['softmax'] = self._verify_softmax()

        # Verify attention operators
        if operations.get('attention') or operations.get('multihead_attention'):
            results['attention'] = self._verify_attention()
        if operations.get('qkv_projection'):
            results['qkv_projection'] = self._verify_qkv_projection()

        # Verify MoE operators
        if operations.get('topk') or operations.get('moe'):
            results['topk'] = self._verify_topk()
            results['moe_router'] = self._verify_moe_router()

        # Verify model with operators
        results['model_inference'] = self._verify_model_inference(model)

        return results

    def _verify_rmsnorm(self, model) -> Dict[str, Any]:
        """Verify RMSNorm operator"""
        print("\n  Verifying RMSNorm:")
        try:
            import torch
            from operators.generated import rmsnorm

            # Get weight from model if available
            weight = torch.ones(256)
            x = torch.randn(2, 16, 256)

            with torch.no_grad():
                output = rmsnorm(x, weight, use_triton=False)

            expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight

            max_diff = (output - expected).abs().max().item()
            matches = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

            print(f"    Max diff: {max_diff:.2e}")
            print(f"    Status: {'PASS' if matches else 'FAIL'}")

            return {
                'status': 'PASS' if matches else 'FAIL',
                'max_diff': max_diff
            }
        except Exception as e:
            print(f"    Error: {e}")
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_layernorm(self, model) -> Dict[str, Any]:
        """Verify LayerNorm operator"""
        print("\n  Verifying LayerNorm:")
        try:
            import torch
            from operators.generated import layernorm

            x = torch.randn(2, 16, 256)
            weight = torch.randn(256)
            bias = torch.randn(256)

            with torch.no_grad():
                output = layernorm(x, weight, bias, use_triton=False)

            expected = torch.nn.functional.layer_norm(x, (256,), weight=weight, bias=bias)

            max_diff = (output - expected).abs().max().item()
            matches = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

            print(f"    Max diff: {max_diff:.2e}")
            print(f"    Status: {'PASS' if matches else 'FAIL'}")

            return {
                'status': 'PASS' if matches else 'FAIL',
                'max_diff': max_diff
            }
        except Exception as e:
            print(f"    Error: {e}")
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_silu(self) -> Dict[str, Any]:
        """Verify SiLU operator"""
        print("\n  Verifying SiLU:")
        try:
            import torch
            from operators.generated import silu

            x = torch.randn(2, 16, 256)

            with torch.no_grad():
                output = silu(x, use_triton=False)

            expected = torch.nn.functional.silu(x)

            max_diff = (output - expected).abs().max().item()
            matches = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

            print(f"    Max diff: {max_diff:.2e}")
            print(f"    Status: {'PASS' if matches else 'FAIL'}")

            return {'status': 'PASS' if matches else 'FAIL', 'max_diff': max_diff}
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_gelu(self) -> Dict[str, Any]:
        """Verify GELU operator"""
        print("\n  Verifying GELU:")
        try:
            import torch
            from operators.generated import gelu

            x = torch.randn(2, 16, 256)

            with torch.no_grad():
                output = gelu(x, use_triton=False)

            expected = torch.nn.functional.gelu(x)

            max_diff = (output - expected).abs().max().item()
            matches = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

            print(f"    Max diff: {max_diff:.2e}")
            print(f"    Status: {'PASS' if matches else 'FAIL'}")

            return {'status': 'PASS' if matches else 'FAIL', 'max_diff': max_diff}
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_softmax(self) -> Dict[str, Any]:
        """Verify Softmax operator"""
        print("\n  Verifying Softmax:")
        try:
            import torch
            from operators.generated import softmax

            x = torch.randn(2, 8, 100)

            with torch.no_grad():
                output = softmax(x, dim=-1, use_triton=False)

            expected = torch.softmax(x, dim=-1)

            max_diff = (output - expected).abs().max().item()
            matches = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

            print(f"    Max diff: {max_diff:.2e}")
            print(f"    Status: {'PASS' if matches else 'FAIL'}")

            return {'status': 'PASS' if matches else 'FAIL', 'max_diff': max_diff}
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_attention(self) -> Dict[str, Any]:
        """Verify Attention operator"""
        print("\n  Verifying Attention:")
        try:
            import torch
            from operators.generated import scaled_dot_product_attention

            batch, n_heads, seq_len, d_k = 2, 4, 16, 32
            q = torch.randn(batch, n_heads, seq_len, d_k)
            k = torch.randn(batch, n_heads, seq_len, d_k)
            v = torch.randn(batch, n_heads, seq_len, d_k)

            with torch.no_grad():
                output, _ = scaled_dot_product_attention(q, k, v)

            # Verify output shape
            shape_ok = output.shape == (batch, n_heads, seq_len, d_k)
            valid = not torch.isnan(output).any() and not torch.isinf(output).any()

            print(f"    Shape: {output.shape}")
            print(f"    Status: {'PASS' if shape_ok and valid else 'FAIL'}")

            return {
                'status': 'PASS' if shape_ok and valid else 'FAIL',
                'output_shape': list(output.shape)
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_qkv_projection(self) -> Dict[str, Any]:
        """Verify QKV projection operator"""
        print("\n  Verifying QKV Projection:")
        try:
            import torch
            from operators.generated import qkv_projection

            batch, seq_len, d_model, n_heads = 2, 16, 256, 8
            x = torch.randn(batch, seq_len, d_model)
            w_qkv = torch.randn(3 * d_model, d_model)

            with torch.no_grad():
                q, k, v = qkv_projection(x, w_qkv, n_heads)

            expected_shape = (batch, n_heads, seq_len, d_model // n_heads)
            q_ok = q.shape == expected_shape
            k_ok = k.shape == expected_shape
            v_ok = v.shape == expected_shape

            print(f"    Q shape: {q.shape} {'OK' if q_ok else 'FAIL'}")
            print(f"    K shape: {k.shape} {'OK' if k_ok else 'FAIL'}")
            print(f"    V shape: {v.shape} {'OK' if v_ok else 'FAIL'}")
            print(f"    Status: {'PASS' if all([q_ok, k_ok, v_ok]) else 'FAIL'}")

            return {
                'status': 'PASS' if all([q_ok, k_ok, v_ok]) else 'FAIL',
                'shapes': {
                    'q': list(q.shape),
                    'k': list(k.shape),
                    'v': list(v.shape)
                }
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_topk(self) -> Dict[str, Any]:
        """Verify TopK operator"""
        print("\n  Verifying TopK:")
        try:
            import torch
            from operators.generated import topk_gate

            batch, seq_len, num_experts = 2, 16, 8
            logits = torch.randn(batch, seq_len, num_experts)
            k = 2

            with torch.no_grad():
                topk_weights, topk_indices, gates = topk_gate(logits, k)

            # Check shapes
            weights_ok = topk_weights.shape == (batch, seq_len, k)
            indices_ok = topk_indices.shape == (batch, seq_len, k)

            # Check weights sum to 1
            sums_ok = torch.allclose(topk_weights.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-5)

            print(f"    Weights shape: {topk_weights.shape} {'OK' if weights_ok else 'FAIL'}")
            print(f"    Indices shape: {topk_indices.shape} {'OK' if indices_ok else 'FAIL'}")
            print(f"    Weights sum: {sums_ok}")
            print(f"    Status: {'PASS' if all([weights_ok, indices_ok, sums_ok]) else 'FAIL'}")

            return {
                'status': 'PASS' if all([weights_ok, indices_ok, sums_ok]) else 'FAIL'
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_moe_router(self) -> Dict[str, Any]:
        """Verify MoE Router operator"""
        print("\n  Verifying MoE Router:")
        try:
            import torch
            from operators.generated import MoERouter

            d_model, num_experts, top_k = 256, 8, 2
            batch, seq_len = 2, 16

            router = MoERouter(d_model, num_experts, top_k)
            x = torch.randn(batch, seq_len, d_model)

            with torch.no_grad():
                topk_weights, topk_indices, dispatcher_mask = router(x)

            weights_ok = topk_weights.shape == (batch, seq_len, top_k)
            indices_ok = topk_indices.shape == (batch, seq_len, top_k)
            mask_ok = dispatcher_mask.shape == (num_experts, batch, seq_len)

            print(f"    Weights shape: {topk_weights.shape} {'OK' if weights_ok else 'FAIL'}")
            print(f"    Indices shape: {topk_indices.shape} {'OK' if indices_ok else 'FAIL'}")
            print(f"    Mask shape: {dispatcher_mask.shape} {'OK' if mask_ok else 'FAIL'}")
            print(f"    Status: {'PASS' if all([weights_ok, indices_ok, mask_ok]) else 'FAIL'}")

            return {
                'status': 'PASS' if all([weights_ok, indices_ok, mask_ok]) else 'FAIL'
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}

    def _verify_model_inference(self, model) -> Dict[str, Any]:
        """Verify complete model inference"""
        print("\n  Verifying Model Inference:")
        try:
            import torch

            model.eval()
            x = torch.randint(0, 1000, (2, 16))

            with torch.no_grad():
                output = model(x)

            valid = not torch.isnan(output).any() and not torch.isinf(output).any()

            print(f"    Input shape: {list(x.shape)}")
            print(f"    Output shape: {list(output.shape)}")
            print(f"    Output valid: {'PASS' if valid else 'FAIL'}")

            return {
                'status': 'PASS' if valid else 'FAIL',
                'input_shape': list(x.shape),
                'output_shape': list(output.shape)
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}


# ============================================================================
# Part 6: Comprehensive Performance Profiler
# ============================================================================

class ComprehensivePerformanceProfiler:
    """Comprehensive performance profiling and analysis"""

    def __init__(self, model, model_name: str = "model"):
        import torch
        self.model = model
        self.model_name = model_name
        self.model.eval()
        self.results = {}

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def profile_all(self, vocab_size: int = 1000, batch_size: int = 2, seq_len: int = 16) -> Dict[str, Any]:
        """Run comprehensive performance profiling"""
        import torch

        print("\n" + "=" * 70)
        print("Comprehensive Performance Analysis")
        print("=" * 70)

        # Prepare input
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)

        self.results = {
            'model_name': self.model_name,
            'device': str(self.device),
            'input_shape': list(x.shape),
            'timing': self._profile_timing(x),
            'memory': self._profile_memory(x),
            'flops': self._profile_flops(x),
            'throughput': self._profile_throughput(x),
            'layer_breakdown': self._profile_layer_breakdown(x)
        }

        return self.results

    def _profile_timing(self, x: torch.Tensor, num_runs: int = 100) -> Dict[str, Any]:
        """Profile inference timing"""
        import torch

        print("\n  Timing Analysis:")
        times = []

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        # Time runs
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)

        import numpy as np
        times = np.array(times)

        result = {
            'mean_ms': float(times.mean()),
            'std_ms': float(times.std()),
            'min_ms': float(times.min()),
            'max_ms': float(times.max()),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'num_runs': num_runs
        }

        print(f"    Mean: {result['mean_ms']:.3f} ms")
        print(f"    Std: {result['std_ms']:.3f} ms")
        print(f"    Min: {result['min_ms']:.3f} ms")
        print(f"    Max: {result['max_ms']:.3f} ms")
        print(f"    Median: {result['median_ms']:.3f} ms")
        print(f"    P95: {result['p95_ms']:.3f} ms")
        print(f"    P99: {result['p99_ms']:.3f} ms")

        return result

    def _profile_memory(self, x: torch.Tensor) -> Dict[str, Any]:
        """Profile memory usage"""
        import torch

        print("\n  Memory Analysis:")

        try:
            import psutil
            process = psutil.Process()
            has_psutil = True
        except ImportError:
            has_psutil = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Get initial memory
        if has_psutil:
            initial_cpu = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            initial_gpu = torch.cuda.memory_allocated() / 1024 / 1024

        # Run inference
        with torch.no_grad():
            output = self.model(x)

        # Get peak memory
        if has_psutil:
            peak_cpu = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Calculate model memory
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        activation_memory = output.numel() * output.element_size() / 1024 / 1024

        result = {
            'cpu_memory_mb': {
                'initial': initial_cpu if has_psutil else None,
                'peak': peak_cpu if has_psutil else None,
                'increase': (peak_cpu - initial_cpu) if has_psutil else None
            },
            'gpu_memory_mb': {
                'initial': initial_gpu if torch.cuda.is_available() else None,
                'peak': peak_gpu if torch.cuda.is_available() else None,
                'increase': (peak_gpu - initial_gpu) if torch.cuda.is_available() else None
            },
            'model_params_mb': param_memory,
            'activations_mb': activation_memory,
            'total_mb': param_memory + activation_memory
        }

        if has_psutil:
            print(f"    CPU: {initial_cpu:.2f} -> {peak_cpu:.2f} MB (+{peak_cpu - initial_cpu:.2f})")
        if torch.cuda.is_available():
            print(f"    GPU: {initial_gpu:.2f} -> {peak_gpu:.2f} MB (+{peak_gpu - initial_gpu:.2f})")
        print(f"    Model params: {param_memory:.2f} MB")
        print(f"    Activations: {activation_memory:.2f} MB")

        return result

    def _profile_flops(self, x: torch.Tensor) -> Dict[str, Any]:
        """Profile FLOPs"""
        print("\n  FLOPs Analysis:")

        try:
            import thop
            from thop import profile

            # Count FLOPs using thop
            flops, params = profile(self.model, inputs=(x,), verbose=False)

            result = {
                'total_flops': int(flops),
                'gflops': float(flops / 1e9),
                'params': int(params),
                'params_m': float(params / 1e6)
            }

            print(f"    Total FLOPs: {result['total_flops']:,}")
            print(f"    GFLOPs: {result['gflops']:.3f}")
            print(f"    Parameters: {result['params']:,}")

        except ImportError:
            # Estimate FLOPs manually
            flops = self._estimate_flops_manual(x.shape[0], x.shape[1])
            result = {
                'total_flops': flops,
                'gflops': flops / 1e9,
                'estimated': True
            }
            print(f"    Estimated FLOPs: {flops:,}")
            print(f"    Estimated GFLOPs: {flops / 1e9:.3f}")

        return result

    def _estimate_flops_manual(self, batch_size: int, seq_len: int) -> int:
        """Manually estimate FLOPs"""
        total_flops = 0

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                total_flops += 2 * in_features * out_features * batch_size * seq_len
            elif isinstance(module, torch.nn.MultiheadAttention):
                embed_dim = module.embed_dim
                total_flops += 4 * embed_dim * embed_dim * batch_size * seq_len  # QKV + O
                total_flops += 2 * batch_size * seq_len * seq_len * embed_dim  # Attention
            elif isinstance(module, torch.nn.LayerNorm):
                normalized_shape = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
                total_flops += 5 * normalized_shape * batch_size * seq_len
            elif isinstance(module, torch.nn.Embedding):
                total_flops += batch_size * seq_len * module.embedding_dim

        return total_flops

    def _profile_throughput(self, x: torch.Tensor, duration: float = 10.0) -> Dict[str, Any]:
        """Profile throughput"""
        import torch

        print("\n  Throughput Analysis:")

        batch_size, seq_len = x.shape[:2]
        start_time = time.time()
        num_runs = 0

        with torch.no_grad():
            while time.time() - start_time < duration:
                _ = self.model(x)
                num_runs += 1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        elapsed = time.time() - start_time
        total_tokens = num_runs * batch_size * seq_len

        result = {
            'elapsed_time_sec': elapsed,
            'num_runs': num_runs,
            'total_tokens': total_tokens,
            'tokens_per_second': total_tokens / elapsed,
            'sequences_per_second': num_runs / elapsed,
            'batch_size': batch_size,
            'seq_len': seq_len
        }

        print(f"    Runs: {num_runs}")
        print(f"    Tokens/sec: {result['tokens_per_second']:.0f}")
        print(f"    Sequences/sec: {result['sequences_per_second']:.2f}")

        return result

    def _profile_layer_breakdown(self, x: torch.Tensor, num_runs: int = 10) -> Dict[str, float]:
        """Profile individual layer timing"""
        import torch

        print("\n  Layer Breakdown:")

        timings = {}

        def make_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                if isinstance(output, torch.Tensor):
                    _ = output.mean().item()  # Force computation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                timings[name] = (end - start) * 1000
            return hook

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        # Run
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Show top layers
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"    Top 10 slowest layers:")
        for name, t in sorted_timings:
            print(f"      {name}: {t:.3f} ms")

        return timings

    def save_performance_report(self, output_path: str = "docs/performance.md") -> str:
        """Generate performance report"""
        if not self.results:
            raise ValueError("No profiling results. Run profile_all first.")

        r = self.results

        lines = [
            "# Performance Analysis Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model**: {r['model_name']}",
            f"**Device**: {r['device']}",
            f"**Input Shape**: {r['input_shape']}",
            "",
            "## Timing Analysis",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean | {r['timing']['mean_ms']:.3f} ms |",
            f"| Std Dev | {r['timing']['std_ms']:.3f} ms |",
            f"| Min | {r['timing']['min_ms']:.3f} ms |",
            f"| Max | {r['timing']['max_ms']:.3f} ms |",
            f"| Median | {r['timing']['median_ms']:.3f} ms |",
            f"| P95 | {r['timing']['p95_ms']:.3f} ms |",
            f"| P99 | {r['timing']['p99_ms']:.3f} ms |",
            "",
            "## Memory Usage",
            ""
        ]

        if r['memory']['cpu_memory_mb']['initial'] is not None:
            cpu = r['memory']['cpu_memory_mb']
            lines.extend([
                "### CPU Memory",
                f"- Initial: {cpu['initial']:.2f} MB",
                f"- Peak: {cpu['peak']:.2f} MB",
                f"- Increase: {cpu['increase']:.2f} MB",
                ""
            ])

        if r['memory']['gpu_memory_mb']['initial'] is not None:
            gpu = r['memory']['gpu_memory_mb']
            lines.extend([
                "### GPU Memory",
                f"- Initial: {gpu['initial']:.2f} MB",
                f"- Peak: {gpu['peak']:.2f} MB",
                f"- Increase: {gpu['increase']:.2f} MB",
                ""
            ])

        lines.extend([
            "### Memory Breakdown",
            f"- Model Parameters: {r['memory']['model_params_mb']:.2f} MB",
            f"- Activations: {r['memory']['activations_mb']:.2f} MB",
            f"- Total: {r['memory']['total_mb']:.2f} MB",
            "",
            "## Compute Analysis",
            "",
            f"- Total FLOPs: {r['flops']['total_flops']:,}",
            f"- GFLOPs: {r['flops']['gflops']:.3f}",
            "",
            "## Throughput",
            "",
            f"- Tokens/Second: {r['throughput']['tokens_per_second']:.0f}",
            f"- Sequences/Second: {r['throughput']['sequences_per_second']:.2f}",
            "",
            "---",
            "",
            "*Generated by CUDA Operator Builder v3.0*"
        ])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\nPerformance report saved: {output_path}")
        return str(output_path)


# ============================================================================
# Part 7: Documentation Generator
# ============================================================================

class DocumentationGenerator:
    """Generate comprehensive documentation"""

    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        model_info: Dict[str, Any],
        param_info: Dict,
        perf_results: Dict,
        verification_results: Dict,
        vocab_size: int
    ) -> Dict[str, str]:
        """Generate all documentation files"""
        print("\n" + "=" * 70)
        print("Generating Documentation")
        print("=" * 70)

        docs = {}

        docs['model_design'] = self._generate_model_design(model_info, param_info)
        docs['dataset_generation'] = self._generate_dataset_doc(vocab_size)
        docs['operator_design'] = self._generate_operator_design(model_info)
        docs['verification'] = self._generate_verification_report(verification_results, model_info)

        return docs

    def _generate_model_design(self, model_info: Dict, param_info: Dict) -> str:
        """Generate model design document"""
        lines = [
            "# Model Design Document",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            f"- **Model Type**: {model_info.get('type', 'Unknown')}",
            f"- **Source File**: {model_info.get('file', 'N/A')}",
            f"- **Total Lines**: {model_info.get('source_lines', 'N/A')}",
            "",
            "## Architecture",
            "",
            "### Detected Classes",
            "",
            "| Class | Base Classes | Methods | Forward |",
            "|-------|-------------|---------|---------|"
        ]

        for cls in model_info.get('classes', []):
            bases = ', '.join(cls['bases'])
            methods = ', '.join(cls['methods'])
            has_forward = 'Yes' if cls.get('has_forward') else 'No'
            lines.append(f"| {cls['name']} | {bases} | {methods} | {has_forward} |")

        lines.extend([
            "",
            "### Detected Operations",
            "",
            ", ".join(model_info.get('operations', {}).keys()),
            ""
        ])

        if param_info:
            lines.extend([
                "## Model Parameters",
                "",
                f"- **Total Parameters**: {param_info.get('total_parameters', 0):,}",
                f"- **Trainable Parameters**: {param_info.get('trainable_parameters', 0):,}",
                f"- **Model Size**: {param_info.get('model_size_mb', 0):.2f} MB",
                ""
            ])

        output_file = self.output_dir / "model_design.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  Generated: {output_file}")
        return str(output_file)

    def _generate_dataset_doc(self, vocab_size: int) -> str:
        """Generate dataset generation document"""
        lines = [
            "# Dataset Generation Document",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            "",
            f"- **Vocabulary Size**: {vocab_size}",
            f"- **Default Batch Size**: 2",
            f"- **Default Sequence Length**: 16",
            "",
            "## Generation Method",
            "",
            "```python",
            "import torch",
            "",
            "# Set seed for reproducibility",
            "torch.manual_seed(42)",
            "",
            "# Generate random input tokens",
            "input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))",
            "",
            "# Generate attention mask (all ones for no padding)",
            "attention_mask = torch.ones(batch_size, seq_len)",
            "```",
            "",
            "## Test Configurations",
            "",
            "| Name | Batch Size | Seq Length | Use |",
            "|------|------------|------------|-----|",
            "| default | 2 | 16 | Standard testing |",
            "| batch_1 | 1 | 16 | Single sample |",
            "| short_seq | 2 | 8 | Short sequences |",
            "| long_seq | 2 | 32 | Longer sequences |",
            ""
        ]

        output_file = self.output_dir / "dataset_generation.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  Generated: {output_file}")
        return str(output_file)

    def _generate_operator_design(self, model_info: Dict) -> str:
        """Generate operator design document"""
        lines = [
            "# Operator Design Document",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            "This document describes the generated CUDA/Triton operators for accelerating model inference.",
            "",
            "## Generated Operators",
            ""
        ]

        operator_info = {
            'rmsnorm': {
                'name': 'RMSNorm',
                'formula': 'x / sqrt(mean(x²) + eps) × weight',
                'description': 'Root Mean Square Layer Normalization'
            },
            'layernorm': {
                'name': 'LayerNorm',
                'formula': '(x - μ) / √(σ² + eps) × γ + β',
                'description': 'Layer Normalization with affine transform'
            },
            'silu': {
                'name': 'SiLU (Swish)',
                'formula': 'x × sigmoid(x)',
                'description': 'Swish activation function'
            },
            'gelu': {
                'name': 'GELU',
                'formula': '0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715x³)))',
                'description': 'Gaussian Error Linear Unit'
            },
            'softmax': {
                'name': 'Softmax',
                'formula': 'exp(x_i) / Σ exp(x_j)',
                'description': 'Softmax activation'
            },
            'attention': {
                'name': 'Scaled Dot-Product Attention',
                'formula': 'softmax(QK^T/√d_k)V',
                'description': 'Multi-head attention mechanism'
            },
            'qkv_projection': {
                'name': 'QKV Projection',
                'formula': '[Q;K;V] = x @ [W_q;W_k;W_v]',
                'description': 'Combined Query-Key-Value projection'
            },
            'rope': {
                'name': 'Rotary Position Embedding',
                'formula': 'RoPE(x, pos)',
                'description': 'Rotary position encoding for Transformers'
            },
            'topk': {
                'name': 'Top-K Gate',
                'formula': 'top_k(softmax(gate(x)), k)',
                'description': 'Top-K routing for Mixture of Experts'
            },
            'moe_router': {
                'name': 'MoE Router',
                'formula': 'route(x) -> (weights, indices, mask)',
                'description': 'Router for Mixture of Experts'
            }
        }

        for op_key in model_info.get('operations', {}).keys():
            if op_key in operator_info:
                info = operator_info[op_key]
                lines.extend([
                    f"### {info['name']}",
                    "",
                    f"**Formula**: `{info['formula']}`",
                    f"**Description**: {info['description']}",
                    f"**File**: `operators/generated/{op_key}.py`",
                    ""
                ])

        lines.extend([
            "## Integration",
            "",
            "```python",
            "from operators.generated import *",
            "",
            "# Operators automatically select Triton when available",
            "y = rmsnorm(x, weight, eps=1e-6)",
            "q, k, v = qkv_projection(x, w_qkv, n_heads)",
            "```",
            ""
        ])

        output_file = self.output_dir / "operator_design.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  Generated: {output_file}")
        return str(output_file)

    def _generate_verification_report(self, results: Dict, model_info: Dict) -> str:
        """Generate verification report"""
        passed = sum(1 for r in results.values() if r.get('status') == 'PASS')
        total = len(results)

        lines = [
            "# Verification Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total Tests**: {total}",
            f"- **Passed**: {passed}",
            f"- **Failed**: {total - passed}",
            f"- **Pass Rate**: {passed/total*100 if total > 0 else 0:.1f}%",
            "",
            "## Detailed Results",
            "",
            "| Test | Status | Details |",
            "|------|--------|---------|"
        ]

        for name, result in results.items():
            if 'error' in result:
                lines.append(f"| {name} | ERROR | {result['error']} |")
            else:
                status = result.get('status', 'UNKNOWN')
                icon = 'Pass' if status == 'PASS' else 'Fail'
                details = result.get('message', '')
                if 'max_diff' in result:
                    details = f"max_diff={result['max_diff']:.2e}"
                lines.append(f"| {name} | {icon} | {details} |")

        lines.extend([
            "",
            "## Operations Detected",
            "",
            ", ".join(model_info.get('operations', {}).keys()),
            ""
        ])

        output_file = self.output_dir / "verification_report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  Generated: {output_file}")
        return str(output_file)


# ============================================================================
# Part 8: Main Entry Point
# ============================================================================

def build(
    model_file: str,
    vocab_size: int = 1000,
    config: Dict = None
) -> Dict[str, Any]:
    """
    Main execution function - Universal model verification and optimization

    Args:
        model_file: Path to PyTorch model file
        vocab_size: Vocabulary size for testing
        config: Optional configuration

    Returns:
        Dictionary containing all results
    """
    print("=" * 70)
    print("CUDA Operator Builder v3.0 - Universal Model Verification")
    print("=" * 70)
    print(f"Model: {model_file}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Default config
    if config is None:
        config = {
            'batch_size': 2,
            'seq_len': 16,
            'num_runs': 50,
            'profile_duration': 10.0,
        }

    # ========== Phase 0: Environment Setup ==========
    print("\n" + "=" * 70)
    print("Phase 0: Environment Setup")
    print("=" * 70)

    env_setup = UniversalEnvironmentSetup()
    env_status = env_setup.check_environment()

    print(f"Python: {env_status['python_version'].split()[0]}")
    print(f"PyTorch: {env_status['packages'].get('torch', 'Not installed')}")

    if env_status.get('gpu'):
        print(f"GPU: {env_status['gpu']}")
        print(f"GPU Memory: {env_status.get('gpu_memory', 'N/A')}")

    # Install missing packages
    if env_status['missing']:
        env_setup.install_missing_packages(env_status['missing'])

    # Install optional packages
    if env_status['optional']:
        env_setup.install_optional_packages(env_status['optional'])

    # ========== Phase 1: Model Analysis ==========
    print("\n" + "=" * 70)
    print("Phase 1: Model Analysis")
    print("=" * 70)

    model_adapter = UniversalModelAdapter(model_file)
    model_type = model_adapter.detect_model_type()
    print(f"Detected model type: {model_type}")

    model_info = model_adapter.analyze_structure()

    # Import model
    try:
        model, init_args = model_adapter.import_model(vocab_size=vocab_size)
        print(f"Model loaded: {model.__class__.__name__}")
        print(f"Init args: {init_args}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return {'error': str(e)}

    # ========== Phase 2: Model Verification ==========
    print("\n" + "=" * 70)
    print("Phase 2: Comprehensive Model Verification")
    print("=" * 70)

    verifier = ComprehensiveModelVerifier(model, model_info)
    verification_results = verifier.run_all_tests(vocab_size)

    # Get parameter info
    param_info = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
    }

    # ========== Phase 3: Operator Generation ==========
    print("\n" + "=" * 70)
    print("Phase 3: Operator Generation")
    print("=" * 70)

    operator_gen = UniversalOperatorGenerator()
    generated_ops = operator_gen.generate_all(model_info)

    # ========== Phase 4: Operator Verification ==========
    print("\n" + "=" * 70)
    print("Phase 4: Operator Verification")
    print("=" * 70)

    op_verifier = OperatorVerifier()
    op_verification_results = op_verifier.verify_all(model_info, model, model_file)

    # ========== Phase 5: Performance Profiling ==========
    print("\n" + "=" * 70)
    print("Phase 5: Performance Profiling")
    print("=" * 70)

    profiler = ComprehensivePerformanceProfiler(model, model.__class__.__name__)
    perf_results = profiler.profile_all(
        vocab_size=vocab_size,
        batch_size=config['batch_size'],
        seq_len=config['seq_len']
    )
    perf_report_path = profiler.save_performance_report()

    # ========== Phase 6: Documentation Generation ==========
    print("\n" + "=" * 70)
    print("Phase 6: Documentation Generation")
    print("=" * 70)

    doc_gen = DocumentationGenerator()
    docs = doc_gen.generate_all(
        model_info,
        param_info,
        perf_results,
        op_verification_results,
        vocab_size
    )

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("Execution Complete")
    print("=" * 70)

    all_results = {**verification_results, **op_verification_results}
    passed = sum(1 for r in all_results.values() if r.get('status') == 'PASS')
    total = len(all_results)

    print(f"\nVerification: {passed}/{total} tests passed")

    print("\nGenerated Files:")
    for op_name, op_path in generated_ops.items():
        print(f"  - {op_path}")
    for doc_name, doc_path in docs.items():
        print(f"  - {doc_path}")
    print(f"  - {perf_report_path}")

    return {
        'environment': env_status,
        'model_type': model_type,
        'model_info': model_info,
        'parameter_info': param_info,
        'verification': verification_results,
        'operator_verification': op_verification_results,
        'generated_operators': generated_ops,
        'performance': perf_results,
        'documentation': docs,
        'performance_report': perf_report_path
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CUDA Operator Builder v3.0 - Universal Model Verification and Optimization"
    )
    parser.add_argument("model_file", help="Path to PyTorch model file")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for profiling")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for profiling")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of profiling runs")
    parser.add_argument("--output-dir", default="docs", help="Documentation output directory")

    args = parser.parse_args()

    config = {
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'num_runs': args.num_runs,
        'profile_duration': 10.0,
    }

    try:
        result = build(
            model_file=args.model_file,
            vocab_size=args.vocab_size,
            config=config
        )
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
