---
name: strategy-validator
description: >
  通过迭代基准测试验证和优化并行训练策略。生成候选配置、执行分布式训练、提取指标，
  并确定最优并行设置。既可独立使用，也可与 PSA 配合。
---

<a id="strategy-validator"></a>
# Strategy Validator：并行策略验证

**通过实测验证迭代优化策略。** 自动生成并测试多个并行候选，识别瓶颈，并逐步收敛到最优配置。

<a id="when-to-use"></a>
## 使用时机

- **基线训练完成后**，寻找最优并行策略
- 在 `/parallel-strategy-analyzer` 之后验证推荐方案（可选）
- 优化现有配置，例如移除不必要的并行维度
- 为新的模型与硬件组合探索并行策略空间
- **MoE 专用：** 使用专家负载和 All-to-All 指标验证 MoE 策略

**前置条件：**
- 已完成基线训练用于比较，或已知目标吞吐
- 目标硬件可用，例如 8× Ascend 910B 的 NPU/GPU 集群
- 模型检查点可访问，例如 `/path/to/Qwen3-30B-A3B`
- 每次验证运行需 5～10 分钟（50 个训练步）
- 预留迭代探索时间，通常测试 3～5 个候选
- **MoE 模型：** 加载 [references/moe-metrics.md](references/moe-metrics.md) 获取额外指标

<a id="pre-flight-safety-checks"></a>
## 运行前安全检查

**执行验证前，自动进行以下检查：**

<a id="1-device-availability"></a>
### 1. 设备可用性
```bash
# NPU check
npu-smi info

# GPU check
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

**检查要求：** 必须具备所需设备数量，例如 world_size=8 时需要 8 个设备。
**失败处理：** 输出错误并中止。

<a id="2-resource-isolation"></a>
### 2. 资源隔离
```bash
# Check for conflicting training jobs
pgrep -f torch.distributed
```

**检查要求：** 检测到其他分布式作业时发出警告。
**失败处理：** 仅警告，用户可以继续。

<a id="3-runtime-estimation"></a>
### 3. 运行时间估算
```python
# Estimated time = warmup (10 steps × 1.5 × avg_step_time)
#                  + training (40 steps × avg_step_time)
#                  + overhead (60s)
# Typical: 50 steps × 0.7s = ~5-7 minutes
```

**显示内容：** 预计完成时间。
**用户操作：** 首次运行或带 --confirm 标志时确认。

<a id="4-user-confirmation-first-run"></a>
### 4. 用户确认（首次运行）

会话中第一次执行验证时，向用户显示：
```
VALIDATION SAFETY CHECK
======================================================================
Configuration:
  Strategy: TP=1, CP=1, EP=4, DP=2
  Devices: 8
  Steps: 50

Estimated runtime:
  Total: ~6.5 minutes

⚠️  First validation run - approval required

Proceed with validation? (yes/no):
```

**跳过确认：** 自动化流程使用 `--auto-approve`。
**强制确认：** 使用 `--confirm`，每次都询问。

<a id="5-output-format"></a>
### 5. 输出格式

验证将**标准化 JSON 指标**输出到 stdout，供后续解析：
```json
{
  "step": 50,
  "throughput_tokens_per_sec": 4486.2,
  "memory_allocated_gb": 7.12,
  "step_time_ms": 174.5,
  "mfu_percent": 42.1,
  "bubble_ratio": 0.053
}
```

**可用工具：**
- `tests/benchmarks/strategy_validation/safety_checks.py`：运行前检查
- `tests/benchmarks/strategy_validation/metrics_output.py`：标准化输出

<a id="validation-flow-iterative-load-workflow-before-each-phase"></a>
## 验证流程（迭代执行，每个阶段前加载对应工作流）

<a id="mode-1-iterative-optimization-recommended"></a>
### 模式 1：迭代优化（推荐）

```text
Phase 1: Baseline Analysis
  - Extract baseline metrics (throughput, memory, step time)
  - Identify bottlenecks (communication vs computation)
  → workflows/01-baseline-analysis.md

Phase 2: Candidate Generation
  - Generate 3-5 candidates based on:
    * Model size (layers, parameters)
    * Sequence length
    * Hardware constraints
    * Baseline bottlenecks
  - Apply heuristics (e.g., avoid CP for seq<512, avoid TP for <10 layers)
  → workflows/02-candidate-generation.md

Phase 3: Batch Validation
  - Run all candidates (parallel or sequential)
  - Monitor for OOM, FSDP errors, hangs
  - Collect metrics automatically
  → workflows/03-batch-validation.md

Phase 4: Comparison & Iteration
  - Compare all candidates vs baseline
  - Identify winner or iterate (if no improvement)
  - Generate deployment-ready config
  → workflows/04-comparison-iteration.md
```

<a id="mode-2-single-strategy-validation-legacy"></a>
### 模式 2：单策略验证（旧版）

```text
For validating a single PSA recommendation or manual config:
  1. Config generation   → workflows/01-config-generation.md
  2. Benchmark execution → workflows/02-benchmark-execution.md
  3. Metrics extraction  → workflows/03-metrics-extraction.md
  4. PSA comparison      → workflows/04-comparison-report.md
```

**迭代循环：** 基线 → 生成候选 → 验证全部候选 → 选择最佳 → 部署或继续迭代。

<a id="references-on-demand"></a>
## 参考资料（按需加载）

- [references/config-templates.md](references/config-templates.md)：训练配置模板
- [references/benchmark-recipes.md](references/benchmark-recipes.md)：标准工作负载
- [references/metrics-schema.md](references/metrics-schema.md)：指标定义与提取

<a id="output-format"></a>
## 输出格式

<a id="iterative-optimization-report"></a>
### 迭代优化报告

**最终摘要：**

```
Model: Qwen3-30B-A3B (4 layers)
Hardware: 8× Ascend 910B
Sequence Length: 128
Goal: 1.2x throughput improvement

┌───┬─────────────────────┬───────────┬───────────┬──────────┬────────┐
│ # │ Strategy            │ Tokens/s  │ vs Base   │ Memory   │ Status │
├───┼─────────────────────┼───────────┼───────────┼──────────┼────────┤
│ 0 │ Baseline (ref)      │ 1,081     │ 1.00x     │ 6.81 GB  │ Ref    │
│ 1 │ TP=2,CP=1,EP=2,DP=2 │ 2,305     │ 2.13x     │ 6.83 GB  │ ✓ Good │
│ 2 │ TP=2,CP=1,EP=4,DP=1 │ —         │ —         │ —        │ ✗ FSDP │
│ 3 │ TP=4,CP=1,EP=2,DP=1 │ —         │ —         │ —        │ ✗ FSDP │
│ 4 │ TP=1,CP=1,EP=2,DP=4 │ 4,274     │ 3.95x ⭐  │ 6.18 GB  │ ✓ BEST │
└───┴─────────────────────┴───────────┴───────────┴──────────┴────────┘

🏆 Winner: Candidate 4 (TP=1, CP=1, EP=2, DP=4)
  - Throughput: 4274 tokens/s (+295% vs baseline)
  - Training time: -75% (4x faster)
  - Memory: -9% (6.18 GB, more efficient)

Key Insights:
  • CP harmful for short sequences (seq_len < 512)
  • TP harmful for small models (<10 layers)
  • Communication >> Computation (main bottleneck)
  • Simple config (fewer dimensions) often wins

Deployment Config:
  accelerator:
    tp_size: 1  # No tensor parallel
    cp_size: 1  # No context parallel
    ep_size: 2  # Keep expert parallel (MoE requirement)
    pp_size: 1  # No pipeline parallel
  fsdp_config:
    dp_shard_size: 4
    edp_shard_size: 4
```

<a id="single-strategy-validation-legacy"></a>
### 单策略验证（旧版）

```
Strategy: TP=8, CP=1, EP=2, DP=4
Model: Qwen3-30B-A3B (24 layers)
Hardware: 64× Ascend 910B

┌──────────────────┬────────────┬───────────┬────────┐
│ Metric           │ PSA Est.   │ Actual    │ Delta  │
├──────────────────┼────────────┼───────────┼────────┤
│ Memory/device    │ 58.0 GB    │ 61.3 GB   │ +5.7%  │
│ Throughput       │ 125K tok/s │ 118K tok/s│ -5.6%  │
│ MFU              │ 45%        │ 42%       │ -3 pts │
└──────────────────┴────────────┴───────────┴────────┘

Status: ✓ PASS (within ±20% tolerance)
```

<a id="key-heuristics-learned-from-validation-experiments"></a>
## 关键启发式规则（来自验证实验）

<a id="model-size-rules"></a>
### 模型规模规则

- **≤10 层：** TP=1，通信成本大于计算收益
- **10～30 层：** TP=2
- **30～60 层：** TP=4
- **>60 层：** TP=8，或考虑 PP

<a id="sequence-length-rules"></a>
### 序列长度规则

- **seq_len < 512：** CP=1，通信占主导
- **512 ≤ seq_len < 2048：** CP=2，收益有限
- **2048 ≤ seq_len < 4096：** CP=4
- **seq_len ≥ 4096：** CP=8，以摊薄通信成本

<a id="moe-constraints"></a>
### MoE 约束

- **始终保持 EP ≥ 2**，这是框架对融合权重的要求
- EP 并行度应整除专家总数
- 避免 DP=1，它会触发当前框架中的 FSDP 冲突

<a id="failure-patterns"></a>
### 失败模式

- **FSDP 与 no_grad 冲突：** 保持 DP ≥ 2
- **MoE 权重不兼容：** 保持 EP > 1
- **OOM：** 减小 TP/PP，增大 DP，以启用 FSDP 分片

<a id="limitations--safety"></a>
## 限制与安全要求

- **资源隔离：** 每次运行占用全部选定设备 5～10 分钟
- **成本：** 50 个训练步 × 候选数，通常为 3～5 个候选
- **准确性：** 微基准可能遗漏显存碎片、罕见边界情况等长期问题
- **硬件波动：** 吞吐出现 ±5～10% 的变化属于正常情况
- **迭代上限：** 5 个候选失败后停止，避免无限循环

**运行前检查：**
- 通过 `nvidia-smi` / `npu-smi info` 确认设备可用
- 模型检查点存在且可读
- 日志磁盘空间充足，每次运行约需 100 MB
- 目标设备上没有冲突作业

**中止条件：**
- 前 20 步内 OOM：立即拒绝，尝试降低并行度
- 挂起，超过 60 秒无进展：超时并报告通信问题
- Loss 为 NaN/Inf：存在梯度不稳定，检查配置有效性
- FSDP 错误：调整 DP 约束

**成功标准：**
- 吞吐提升达到目标，默认为 1.2 倍
- 显存不超过设备限制
- 连续 50 步没有运行时错误
- 吞吐稳定，标准差小于均值的 10%
