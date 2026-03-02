# Code Agent 验证工程

> 被测试的 Code Agent 读取 `tasks.md` → 生成代码 → 运行测试 → 自我评估 → 生成报告

---

## 📁 文件结构

```
task_test/
├── tasks.md                    # 任务定义 MD 文件（Agent 读取）
├── validate.py                 # 验证系统（Agent 执行自我评估）
├── test_cases/                 # 测试用例文件夹
│   ├── test_attention.py
│   ├── test_rmsnorm.py
│   └── test_moe.py
├── solutions/                  # Agent 生成代码存放位置
│   ├── multi_head_attention.py
│   ├── rmsnorm.py
│   └── moe_router.py
└── reports/                    # 输出报告文件夹
    └── code_agent_capability_report.md
```

---

## 🚀 使用方法

### Code Agent 自我评估流程

1. **读取任务**
   ```bash
   # Agent 读取 tasks.md 了解需要完成的任务
   ```

2. **生成解决方案**
   ```bash
   # Agent 在 solutions/ 目录生成代码文件
   ```

3. **运行验证**
   ```bash
   python validate.py
   ```

4. **查看报告**
   ```bash
   # 打开 reports/code_agent_capability_report.md
   ```

---

## 📋 任务列表

| ID | 任务 | 类别 | 难度 | 测试文件 |
|----|------|------|------|----------|
| 1 | Multi-Head Attention | 模型开发 | ⭐⭐ | test_attention.py |
| 2 | Triton RMSNorm | 算子开发 | ⭐⭐⭐ | test_rmsnorm.py |
| 3 | MoE Router | 模型开发 | ⭐⭐⭐⭐ | test_moe.py |
| 4 | Flash Attention | 算子优化 | ⭐⭐⭐⭐ | test_flash_attn.py |
| 5 | INT8 量化 | 算法实现 | ⭐⭐⭐ | test_quantization.py |
| 6 | 调试梯度 | 软件调试 | ⭐⭐ | test_debug_gradient.py |
| 7 | LLaMA 层 | 模型开发 | ⭐⭐⭐⭐ | test_llama.py |
| 8 | 算子融合 | 算子优化 | ⭐⭐⭐ | test_fusion.py |

---

## 🎯 七维评估

1. **代码质量** - 结构、风格、文档、类型提示
2. **意图符合度** - 需求匹配、功能完整性
3. **结果精确度** - 输出正确性、测试通过率
4. **工程能力** - 错误处理、模块化设计
5. **执行效率** - 时间遵守、资源使用
6. **Skills 能力** - 工具调用、外部库使用
7. **Multi-Agent 能力** - 任务分解、协调能力

---

## 📊 报告示例

```markdown
# Code Agent 能力报告

## 📊 总体评估

| 指标 | 数值 |
|------|------|
| 总体得分 | 0.78/1.00 |

### ⭐ 优秀 (Excellent)

该 Code Agent 在大多数维度表现良好...

## 🎯 七维能力分析

### 1️⃣ 代码质量
**得分**: 0.85/1.00

███████████████████░░

**等级**: 优秀

...
```
