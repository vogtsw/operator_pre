# .skills - Claude Skills 使用指南

> 本仓库包含 Claude Code 的自定义 Skills，用于扩展 AI 编程助手的能力。

---

## 目录

- [简介](#简介)
- [Skills 概览](#skills-概览)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [安装方法](#安装方法)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

---

## 简介

`.skills` 目录是 Claude Code 的自定义技能扩展系统。每个 Skill 都是一个独立的功能模块，可以被 Claude Code 调用来完成特定的编程任务。

### 什么是 Claude Code Skills？

Claude Code Skills 是可插拔的功能模块，允许用户：
- 扩展 Claude 的代码生成能力
- 添加特定领域的专业知识
- 自动化复杂的开发流程
- 集成第三方工具和 API

---

## Skills 概览

### 1. ai-operator-development

**AI 模型验证与算子生成工具**

一个通用的 AI 模型验证和算子生成工具，支持自动分析 PyTorch 模型、生成 Triton/CUDA 算子、验证正确性并进行性能分析。

**核心功能：**
- 模型结构验证（8 类测试套件）
- 自动算子生成（RMSNorm、LayerNorm、Attention、MoE 等 10+ 种算子）
- 性能分析（时间、内存、FLOPs、吞吐量）
- 自动文档生成
- 通用模型适配器（自动检测 Transformer、CNN、MoE 等架构）

**使用场景：**
- 深度学习模型验证
- CUDA/Triton 算子开发
- 模型性能优化
- 自动化测试报告生成

**生成的文件：**
```
operators/<model_name>/
├── rmsnorm.py       # Triton + PyTorch 实现
├── silu.py          # Triton + PyTorch 实现
├── attention.py     # PyTorch 实现
├── moe.py           # MoE 路由器
└── ...

docs/
├── model_design.md
├── operator_design.md
├── verification_report.md
└── performance.md
```

**依赖要求：**
- Python 3.8+
- PyTorch 1.10+
- Triton（可选，用于 GPU 内核）
- psutil（可选，用于内存分析）
- thop（可选，用于 FLOPs 计算）

---

### 2. codecli_test_skill

**Code Agent Vibe Coding 评估框架**

一个用于评估 Code Agent 编程能力的七维评估系统，通过实际任务测试 Agent 的代码生成、问题解决和工程能力。

**核心功能：**
- 七维能力评估（代码质量、意图符合度、结果精确度、工程能力、执行效率、Skills 能力、Multi-Agent 能力）
- 多种测试任务（Attention、RMSNorm、MoE、Flash Attention 等）
- 自动化测试和评分
- 详细的评估报告生成

**七维评估体系：**
1. **代码质量** - 结构、风格、文档、类型提示
2. **意图符合度** - 需求匹配、功能完整性
3. **结果精确度** - 输出正确性、测试通过率
4. **工程能力** - 错误处理、模块化设计
5. **执行效率** - 时间遵守、资源使用
6. **Skills 能力** - 工具调用、外部库使用
7. **Multi-Agent 能力** - 任务分解、协调能力

**测试任务列表：**
| ID | 任务 | 类别 | 难度 |
|----|------|------|------|
| 1 | Multi-Head Attention | 模型开发 | ⭐⭐ |
| 2 | Triton RMSNorm | 算子开发 | ⭐⭐⭐ |
| 3 | MoE Router | 模型开发 | ⭐⭐⭐⭐ |
| 4 | Flash Attention | 算子优化 | ⭐⭐⭐⭐ |
| 5 | INT8 量化 | 算法实现 | ⭐⭐⭐ |

**使用场景：**
- Code Agent 能力评估
- 自动化代码审查
- 编程助手性能基准测试
- AI 编程工具对比研究

---

## 快速开始

### 方法一：使用 Claude Code 直接调用

在 Claude Code 中直接使用：

```
请使用 ai-operator-development skill 来验证我的模型
```

```
请使用 codecli_test_skill 来评估这个代码生成任务
```

### 方法二：命令行调用

**ai-operator-development：**
```bash
python .skills/ai-operator-development/ai_operator_development.py model.py
```

**codecli_test_skill：**
```bash
cd .skills/codecli_test_skill
python core/evaluator.py
```

---

## 详细使用说明

### ai-operator-development 使用指南

#### 1. 基本用法

**命令行模式：**
```bash
python .skills/ai-operator-development/ai_operator_development.py <model_file> [options]
```

**Python API 模式：**
```python
from .skills.ai_operator_development import build

result = build(
    model_file="model2.py",
    vocab_size=1000,
    config={
        'batch_size': 2,
        'seq_len': 16,
        'num_runs': 50
    }
)

# 访问结果
print(f"模型类型: {result['model_type']}")
print(f"性能数据: {result['performance']}")
print(f"验证报告: {result['verification']}")
```

#### 2. 输出结果结构

```python
{
    'environment': {...},      # Python、包、GPU 信息
    'model_type': str,         # 'transformer', 'cnn', 'moe' 等
    'model_info': {...},       # 检测到的类和操作
    'parameter_info': {...},   # 总参数量、模型大小
    'verification': {...},     # 测试结果
    'generated_operators': {...},  # 生成的文件路径
    'performance': {...},      # 时间、内存、FLOPs、吞吐量
    'documentation': {...},    # 生成的文档路径
}
```

#### 3. 支持的算子类型

| 类型 | 算子 |
|------|------|
| 归一化 | RMSNorm, LayerNorm, BatchNorm |
| 激活函数 | SiLU, GELU, ReLU, Softmax |
| 注意力机制 | Multi-head attention, QKV projection, RoPE |
| 线性层 | Dense layers, embeddings |
| 混合专家 | Top-K routing, Expert router |
| 卷积 | Conv2D, MaxPool |

---

### codecli_test_skill 使用指南

#### 1. 评估流程

```
Agent 读取 tasks.md → 生成代码 → 运行测试 → 自我评估 → 生成报告
```

#### 2. 文件结构

```
codecli_test_skill/
├── tasks.md                    # 任务定义文件
├── core/
│   └── evaluator.py            # 评估系统
├── test_cases/                 # 测试用例
│   ├── test_attention.py
│   ├── test_rmsnorm.py
│   └── test_moe.py
├── solutions/                  # Agent 生成的代码
├── reports/                    # 评估报告
└── docs/                       # 文档
```

#### 3. 运行评估

```bash
cd .skills/codecli_test_skill
python core/evaluator.py
```

#### 4. 查看报告

评估完成后，查看 `reports/` 目录下的评估报告。

---

## 安装方法

### 自动安装

**Windows (PowerShell)：**
```powershell
.skills\ai-operator-development\install.ps1
```

**Windows (CMD)：**
```cmd
.skills\ai-operator-development\install.bat
```

**Linux/macOS：**
```bash
.skills/ai-operator-development/install.sh
```

**Python 跨平台：**
```bash
python .skills/ai-operator-development/install.py
```

### 手动安装

**ai-operator-development 依赖：**
```bash
pip install torch
pip install triton  # 可选，用于 GPU 内核
pip install psutil  # 可选，用于内存分析
pip install thop    # 可选，用于 FLOPs 计算
```

**codecli_test_skill 依赖：**
```bash
pip install torch
pip install numpy
```

---

## 开发指南

### 创建自定义 Skill

1. **创建 Skill 目录结构：**
```
.skills/
└── your-skill/
    ├── SKILL.md           # Skill 元数据
    ├── README.md          # 使用文档
    ├── install.py         # 安装脚本
    └── your_code.py       # 主要代码
```

2. **编写 SKILL.md：**
```markdown
---
name: your-skill-name
description: A brief description of what your skill does
license: MIT
---

# Your Skill Name

Detailed description of your skill...
```

3. **实现功能代码**
4. **添加安装脚本**
5. **测试和文档**

### Skill 最佳实践

- ✅ 保持代码模块化和可维护
- ✅ 提供清晰的错误消息
- ✅ 编写详细的文档
- ✅ 包含测试用例
- ✅ 支持可选依赖
- ✅ 处理边缘情况

---

## 常见问题

### Q1: 如何确认 Skill 已正确安装？

```bash
# 检查 Skill 目录
ls .skills/

# 查看 Skill 元数据
cat .skills/ai-operator-development/SKILL.md
```

### Q2: Skill 不工作怎么办？

1. 检查 Python 版本（需要 3.8+）
2. 确认所有依赖已安装
3. 查看错误日志
4. 检查 Claude Code 版本

### Q3: 可以创建自己的 Skill 吗？

可以！参考上述"开发指南"部分创建自定义 Skill。

### Q4: 如何更新 Skill？

```bash
cd .skills
git pull
```

或重新下载 Skill 文件并覆盖。

---

## 许可证

本仓库中的 Skills 遵循各自的许可证：
- ai-operator-development: MIT
- codecli_test_skill: MIT

---

## 贡献

欢迎贡献新的 Skills！请确保：
1. 遵循现有的目录结构
2. 包含完整的文档
3. 提供安装脚本
4. 添加测试用例

---

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**最后更新：** 2026-03-03
