<a id="strategy-validator---usage-guide"></a>
# Strategy Validator 使用指南

<a id="overview"></a>
## 概述

**strategy-validator** Skill 通过迭代实测验证优化并行训练策略。它自动生成候选配置、运行基准测试，并确定最优并行策略。

<a id="quick-start"></a>
## 快速开始

<a id="invocation"></a>
### 调用方式
```
/validate-strategy
```

Skill 将引导你完成：
1. 基线分析，提取当前性能
2. 候选生成，产生 3～5 个优化策略
3. 批量验证，运行全部候选
4. 比较与选择，确定胜出方案

<a id="when-to-use"></a>
## 使用时机

- ✅ 初始训练完成后，优化并行策略
- ✅ 吞吐低于预期时
- ✅ 扩展到更大规模硬件之前，先验证配置
- ✅ 模型或硬件变化时，需要建立新基线

<a id="prerequisites"></a>
## 前置条件

1. **已完成基线训练**
   - 至少 50 步的训练日志
   - 配置文件可访问

2. **硬件可用**
   - 目标集群就绪，例如 8× NPU
   - 没有冲突作业

3. **时间预算**
   - 每个候选约 7 分钟
   - 通常 3～5 个候选，总计 20～35 分钟

4. **磁盘空间**
   - 每次验证运行约需 100 MB

<a id="workflow-modes"></a>
## 工作流模式

<a id="mode-1-iterative-optimization-recommended"></a>
### 模式 1：迭代优化（推荐）

**适用场景：** 希望找到最优配置。

```
Phase 1: Baseline Analysis
  ↓ Extract metrics from existing training

Phase 2: Candidate Generation
  ↓ Generate 3-5 candidates based on heuristics

Phase 3: Batch Validation
  ↓ Run all candidates, collect metrics

Phase 4: Comparison & Iteration
  ↓ Identify winner, decide: deploy or iterate
```

**输出：** 最优配置和部署指南。

<a id="mode-2-single-strategy-validation-legacy"></a>
### 模式 2：单策略验证（旧版）

**适用场景：** 已有明确配置需要验证，例如 PSA 的推荐结果。

```
Phase 1: Config Generation
  ↓ Convert strategy to training config

Phase 2: Benchmark Execution
  ↓ Run 50-step validation

Phase 3: Metrics Extraction
  ↓ Parse logs, extract throughput/memory

Phase 4: PSA Comparison
  ↓ Compare actual vs estimated
```

**输出：** 通过/失败判定及指标差异。

<a id="key-heuristics"></a>
## 关键启发式规则

验证器使用以下从实验中总结的规则：

<a id="model-size"></a>
### 模型规模
- **≤10 层：** TP=1，通信成本超过收益
- **10～30 层：** TP=2
- **30～60 层：** TP=4
- **>60 层：** TP=8 或更高

<a id="sequence-length"></a>
### 序列长度
- **< 512：** CP=1，通信占主导
- **512～2048：** CP=2
- **2048～4096：** CP=4
- **≥ 4096：** CP=8

<a id="moe-models"></a>
### MoE 模型
- **始终保持 EP ≥ 2**，这是框架要求
- 避免 DP=1，以免触发 FSDP 冲突

<a id="bottleneck-patterns"></a>
### 瓶颈模式
- **通信受限** → 减小 TP/CP，增大 DP
- **计算受限** → 可能已经最优
- **显存受限** → 增大 DP，增加分片

<a id="example-session"></a>
## 示例会话

<a id="input"></a>
### 输入
```
Baseline: TP=2, CP=2, EP=2, DP=2
Model: Qwen3-30B-A3B (4 layers)
Hardware: 8× Ascend 910B
Throughput: 1081 tokens/s
Goal: 1.2x improvement
```

<a id="generated-candidates"></a>
### 生成的候选
```
1. TP=2, CP=1, EP=2, DP=2 — Remove CP
2. TP=2, CP=1, EP=4, DP=1 — Increase EP
3. TP=4, CP=1, EP=2, DP=1 — Increase TP
4. TP=1, CP=1, EP=2, DP=4 — Minimal TP
```

<a id="results"></a>
### 结果
```
| # | Strategy | Throughput | vs Base | Status |
|---|----------|------------|---------|--------|
| 1 | Remove CP | 2305 t/s | 2.13x | ✓ Good |
| 2 | Increase EP | — | — | ✗ FSDP |
| 3 | Increase TP | — | — | ✗ FSDP |
| 4 | Minimal TP | 4274 t/s | 3.95x | ✓ BEST |
```

<a id="output"></a>
### 输出
```
🏆 Winner: TP=1, CP=1, EP=2, DP=4
  Throughput: 4274 tokens/s (+295%)
  Training time: -75%
  Memory: -9%

✅ Target exceeded (3.95x > 1.2x)
Status: APPROVED FOR DEPLOYMENT
```

<a id="failure-handling"></a>
## 失败处理

<a id="common-issues"></a>
### 常见问题

**1. 全部候选因 FSDP 错误失败**
```
→ All used DP=1 (framework constraint)
→ Regenerate with DP ≥ 2 constraint
→ Auto-retry once
```

**2. 全部候选均差于基线**
```
→ Baseline already optimal
→ Document findings
→ No deployment change
```

**3. 全部候选均发生 OOM**
```
→ Model too large for hardware
→ Suggest: reduce batch size, increase devices, enable gradient checkpointing
```

**4. 挂起或超时**
```
→ Communication deadlock
→ Check HCCL/NCCL config
→ Reduce parallelism dimensions
```

<a id="output-artifacts"></a>
## 输出产物

验证成功后生成：

<a id="1-production-config"></a>
### 1. 生产配置
```yaml
# validation_experiment/production_config.yaml
accelerator:
  tp_size: 1
  cp_size: 1
  ep_size: 2
  pp_size: 1

fsdp_config:
  dp_shard_size: 4
  edp_shard_size: 4
```

<a id="2-final-report"></a>
### 2. 最终报告
```markdown
# validation_experiment/FINAL_REPORT.md

- Winner configuration
- Performance comparison
- Key insights
- Deployment recommendation
```

<a id="3-deployment-guide"></a>
### 3. 部署指南
```markdown
# validation_experiment/DEPLOYMENT_GUIDE.md

- Step-by-step migration
- Expected impact
- Rollback plan
```

<a id="4-raw-data"></a>
### 4. 原始数据
```
validation_experiment/
├── baseline_metrics.json
├── candidate1_*/
│   ├── config.yaml
│   ├── validation_run.log
│   └── metrics.json
├── candidate2_*/
└── comparison_summary.json
```

<a id="iteration-logic"></a>
## 迭代逻辑

满足以下情况时，验证器可以继续迭代，最多 2 次：
- ✅ 已取得部分收益，值得继续探索
- ✅ 失败模式明确且可以修复，例如全部 DP=1，则增加 DP≥2 约束
- ❌ 尚未达到目标

以下情况停止迭代：
- ✅ 已达到目标，运行成功
- ❌ 达到迭代上限 2 次
- ❌ 无法进一步改进，基线已经最优

<a id="best-practices"></a>
## 最佳实践

1. **在低峰期运行**，会占用设备 20～35 分钟
2. **从基线开始**，始终建立比较参考
3. **使用启发式规则**，这些规则来自真实实验
4. **扩容前先验证**，在 8 个设备上测试比在 64 个设备上成本更低
5. **记录发现**，写入项目记忆供后续参考

<a id="safety"></a>
## 安全要求

- ✅ 只读访问模型检查点
- ✅ 仅写入 `validation_experiment/`
- ✅ 使用独立验证运行，不影响生产
- ✅ OOM、挂起等严重错误会自动中止
- ✅ 最多 2 轮迭代，避免无限循环

<a id="cost-estimation"></a>
## 成本估算

**每个候选：** 50 步 × 训练成本。
**典型会话：** 3～5 个候选。
**总计：** 相当于约 150～250 个训练步。

**投资回报：** 如果找到 2 倍性能提升，约 200 个生产训练步即可抵消验证成本。

<a id="troubleshooting"></a>
## 故障排查

<a id="no-baseline-found"></a>
### “No baseline found”（未找到基线）
```
→ Point to existing training log
→ Or run 50 steps of baseline first
```

<a id="all-candidates-failed"></a>
### “All candidates failed”（全部候选失败）
```
→ Check pre-flight (devices available, no OOM)
→ Review error patterns
→ May need manual intervention
```

<a id="winner-worse-than-baseline"></a>
### “Winner worse than baseline”（胜出候选差于基线）
```
→ Baseline may already be optimal
→ Try different optimization axis (e.g., batch size, learning rate)
```

<a id="references"></a>
## 参考资料

- **真实实验：** `validation_experiment/FINAL_REPORT_CANDIDATE4_WINS.md`
- **工作流：** `.agent/skills/strategy-validator/workflows/`
- **启发式规则来源：** Qwen3-30B-A3B 验证，2026 年 10 月

---

**下一步：** 调用 `/validate-strategy` 开始优化！
