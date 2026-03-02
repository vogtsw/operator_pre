# Code Agent 测试任务

> 本文件定义了 Code Agent 需要完成的开发任务
> Agent 读取本文件 → 生成代码 → 运行测试 → 自我评估 → 生成报告

---

## 任务说明

以下是 **8 个 AI 开发任务**，覆盖模型开发、算子开发、软件开发三个领域。

每个任务包含：
- **需求描述**：需要实现的功能
- **验收标准**：完成的标准
- **测试用例**：在 `test_cases/` 文件夹中

---

## Task 1: Multi-Head Attention 实现

**类别**: AI 模型开发
**难度**: ⭐⭐ 中级

### 需求
1. 实现缩放点积注意力机制 (Scaled Dot-Product Attention)
2. 支持多头注意力 (Multi-Head Attention)
3. 包含因果掩码 (Causal Masking) 用于解码器
4. 使用 PyTorch 框架
5. 包含完整的文档字符串和类型提示

### 验收标准
- [ ] 代码无语法错误，可以正常导入
- [ ] 前向传播输出形状正确：(batch, seq_len, d_model)
- [ ] 梯度能够正确回传到所有参数
- [ ] 通过 `test_cases/test_attention.py` 中的测试

### 测试文件
`test_cases/test_attention.py`

---

## Task 2: Triton RMSNorm 算子

**类别**: AI 算子开发
**难度**: ⭐⭐⭐ 高级

### 需求
1. 使用 `@triton.jit` 装饰器实现 GPU kernel
2. 包含前向传播和反向传播
3. 提供 PyTorch 封装类 `RMSNorm`
4. 包含 CPU 降级实现
5. 处理边界情况（eps=0、空输入）

### 验收标准
- [ ] Triton kernel 能够编译成功
- [ ] 结果与 PyTorch nn.RMSNorm 误差 < 1e-5
- [ ] GPU 大规模输入时比 CPU 快
- [ ] 通过 `test_cases/test_rmsnorm.py` 中的测试

### 测试文件
`test_cases/test_rmsnorm.py`

---

## Task 3: Mixture of Experts Router

**类别**: AI 模型开发
**难度**: ⭐⭐⭐⭐ 专家

### 需求
1. 实现 Top-K 专家选择机制
2. 添加负载均衡损失 (Load Balancing Loss)
3. 支持容量因子 (Capacity Factor)
4. 优雅处理专家丢弃情况
5. 包含辅助损失用于均衡专家负载

### 验收标准
- [ ] Top-K 选择结果正确
- [ ] 负载均衡损失能够收敛
- [ ] 没有专家过载 (overload)
- [ ] 通过 `test_cases/test_moe.py` 中的测试

### 测试文件
`test_cases/test_moe.py`

---

## Task 4: Flash Attention 优化

**类别**: AI 算子优化
**难度**: ⭐⭐⭐⭐ 专家

### 需求
1. 实现分块 (Tiling) 策略减少内存
2. 使用原地操作减少内存分配
3. 最小化 HBM (High Bandwidth Memory) 访问
4. 支持因果掩码
5. 处理可变序列长度

### 验收标准
- [ ] 内存复杂度为 O(N) 而非 O(N²)
- [ ] 结果与标准 attention 匹配
- [ ] 长序列 (>2048) 有明显加速
- [ ] 通过 `test_cases/test_flash_attn.py` 中的测试

### 测试文件
`test_cases/test_flash_attn.py`

---

## Task 5: INT8 量化算法

**类别**: AI 算法实现
**难度**: ⭐⭐⭐ 高级

### 需求
1. 计算激活值的动态范围
2. 实现对称量化 (Symmetric Quantization)
3. 添加反量化支持
4. 权重支持逐通道量化
5. 包含量化前后准确率评估

### 验收标准
- [ ] 量化模型能正常运行
- [ ] 准确率下降 < 2%
- [ ] 模型大小减少约 4x
- [ ] 通过 `test_cases/test_quantization.py` 中的测试

### 测试文件
`test_cases/test_quantization.py`

---

## Task 6: 调试 Attention 梯度消失

**类别**: AI 软件调试
**难度**: ⭐⭐ 中级

### 需求
1. 识别并定位梯度消失的位置
2. 修复导致梯度消失的问题
3. 添加梯度裁剪 (Gradient Clipping)
4. 验证梯度正确流动
5. 添加调试用的断言和日志

### 验收标准
- [ ] 梯度能够到达所有参数
- [ ] 无 NaN 或 Inf 梯度
- [ ] 模型能够正常训练
- [ ] 通过 `test_cases/test_debug_gradient.py` 中的测试

### 测试文件
`test_cases/test_debug_gradient.py`

---

## Task 7: LLaMA Transformer 层

**类别**: AI 模型开发
**难度**: ⭐⭐⭐⭐ 专家

### 需求
1. 实现预归一化 (Pre-Norm)：RMSNorm 在 attention/FFN 之前
2. 实现 SwiGLU 激活函数
3. 实现旋转位置编码 (RoPE)
4. 可选：支持分组查询注意力 (GQA)
5. 包含正确的权重初始化

### 验收标准
- [ ] 前向传播正常工作
- [ ] 架构符合 LLaMA 论文描述
- [ ] 梯度正确流动
- [ ] 通过 `test_cases/test_llama.py` 中的测试

### 测试文件
`test_cases/test_llama.py`

---

## Task 8: 算子融合优化

**类别**: AI 算子优化
**难度**: ⭐⭐⭐ 高级

### 需求
1. 将 bias + GeLU + add 融合到一个 kernel
2. 使用 Triton 或 CUDA 实现
3. 处理不同张量大小的边界情况
4. 保持数值精度与未融合版本一致
5. 基准测试并展示性能提升

### 验收标准
- [ ] 融合 kernel 比未融合版本快
- [ ] 结果与未融合版本匹配
- [ ] 内存使用更低
- [ ] 通过 `test_cases/test_fusion.py` 中的测试

### 测试文件
`test_cases/test_fusion.py`

---

## 自我评估维度

完成所有任务后，从以下 **7 个维度** 进行自我评估：

1. **代码质量** - 结构、风格、文档、类型提示、命名规范
2. **意图符合度** - 是否满足所有需求、功能是否完整
3. **结果精确度** - 输出是否正确、测试用例通过率
4. **工程能力** - 错误处理、模块化设计、库使用
5. **执行效率** - 时间遵守、资源使用、算法复杂度
6. **Skills 能力** - 能否调用外部 tools/skills
7. **Multi-Agent 能力** - 能否协调多个 agent/分解子任务

---

## 输出要求

将自我评估报告保存至：
```
reports/code_agent_capability_report.md
```

报告包含：
- 每个任务的完成情况
- 7 个维度的自我评分
- 总体能力评估
- 未完成的任务及原因
- 改进建议
