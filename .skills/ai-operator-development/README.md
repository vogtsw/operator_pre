# AI Operator Development - Claude Code Skill

Universal AI model verification and operator generator for Claude Code.

## Features

- **Model Verification**: 8-category test suite (syntax, structure, forward pass, gradients, variations, consistency, parameters, edge cases)
- **Operator Generation**: Triton/CUDA kernels for 10+ operator types (RMSNorm, LayerNorm, SiLU, GELU, Attention, MoE, etc.)
- **Performance Profiling**: Timing, memory, FLOPs, throughput analysis with output to `performance.md`
- **Documentation**: Auto-generates model design, operator design, verification reports, performance reports
- **Universal Adapter**: Auto-detects model types (Transformer, CNN, MoE, LLaMA, GPT, BERT, Custom)

## Quick Install

### Windows - Double-click to install
```
install.bat
```

### Windows - PowerShell
```powershell
.\install.ps1
```

### Python (Cross-platform)
```bash
python install.py
```

### Linux/Mac - Shell script
```bash
bash install.sh
```

### Manual Install

**Windows PowerShell**:
```powershell
cd $env:APPDATA\Claude\skills
git clone https://github.com/vogtsw/operator_pre.git ai-operator-development
```

**Linux/Mac**:
```bash
cd ~/.claude/skills
git clone https://github.com/vogtsw/operator_pre.git ai-operator-development
```

## Usage

After installation, use in Claude Code:

```
/ai-operator-development model.py
```

Or with natural language:
```
Use the ai-operator-development skill to analyze my model and generate operators
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- (Optional) Triton for GPU kernels
- (Optional) psutil for memory profiling
- (Optional) thop for FLOPs counting

## Repository

https://github.com/vogtsw/operator_pre

## License

MIT
