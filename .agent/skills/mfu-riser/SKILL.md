---
name: mfu-riser
description: >
  结合并行策略先验、昇腾性能分析和经过验证的端到端实验，驱动有预算边界、以证据为依据的训练 MFU 优化循环。
  用于通过等价实现、通信、显存、优化器或并行布局改进，以及经过授权的工作负载调优，达到声明的训练 MFU 目标。
---

<a id="mfu-riser"></a>
# MFU Riser：训练 MFU 优化

将模型和硬件先验、实测算子证据以及官方 API 参考转化为可证伪的优化方案。在用户指定的 MFU 目标和预算内执行实验、测量、验证，并更新最佳候选。**Agent 负责规划；Python 控制器负责执行实验和记录结果，不包含额外的 LLM，也不生成方案。**

使用[训练 RSI 协议](../perf-playbook/references/training-rsi.md)固定目标、模型语义、路由和验收标准。配置 `performance_policy`，以支持根据模型架构核算 FLOPs、分级验证和更广的候选范围。仅针对融合的实验可以使用原始 v1 协议。

<a id="entry-requirements-and-fixed-measurement-protocol"></a>
## 前置要求与固定测量协议

实验控制器在 Linux 上运行。它通过建议性文件锁串行更新状态，并在超时时通过进程组清理分布式工作进程。

获取可复现的初始状态、数据集、启动命令、设备集合、完整训练步数、目标和预算。初始状态可以是检查点，也可以是明确声明的随机初始化。复用任务上下文中已有的信息。未指定目标时，先分析并测量基线，再提出有依据的目标。按照 [contracts.md](references/contracts.md) 编写配置，不要用猜测值代替测量结果。

- 在同一比较序列中，固定模型工作量、输入、初始化、路由、优化器、设备、峰值和计时边界。经过许可修改序列长度或批量时，应建立关联序列，并为其独立测量基线。
- 根据实际执行的模型工作量核查 MFU 分子。报告端到端 MFU、注明证据覆盖范围的 HFU，以及有效 tokens/s/device。搜索按 MFU 排名。
- 性能优先时可以使用较短的筛选窗口，但仍须提供实现正确性和目标路径实际生效的正向证据。验收序列应使用完整门禁，测量至少 20 个稳态步，随后在相同源码和工作负载下独立确认。
- 将固定路由视为明确声明的性能工作负载，并保留其中的模型架构、逻辑 Top-K 和 token 语义。训练策略变更应使用该策略对应的验收标准。

`rsi_mode` 默认为 `acceptance`；有要求时可设为 `continuous`。每项任务还须交付 [RSI 改进报告](../autoresearch/references/experience-consolidation.md)。自主复盘有价值的经验，无需用户再次要求。复用已有经验，必要时原位改进；部署专属信息保留在仓库之外。

<a id="iteration-workflow"></a>
## 迭代流程

1. **建立先验。** 使用 [parallel-strategy-analyzer](../parallel-strategy-analyzer/SKILL.md)，结合实际模型架构和硬件拓扑，估算可行布局、显存和成本，将结构化结果保存为 `prior.json`。已有稳定布局可以作为基线。先验不等于测量结果。
2. **建立基线并接入适配器。** 按照 [strategy-validator](../strategy-validator/SKILL.md)，将现有训练入口和可复现初始状态接入控制器的五个 argv 阶段。初始化时固定数值容差、验证器、工作负载和预算，参见 [acceptance.md](references/acceptance.md)。
3. **采集运行证据。** 使用 [profiling](../profiling/SKILL.md)，在预热后至少采集两步。`scripts/collect_state.py` 调用现有解析器，提取算子形状、类型、次数和耗时。使用 [memory-analysis](../memory-analysis/SKILL.md) 分析显存压力；涉及 MoE 路由变更时，使用 [moe-profiler](../moe-profiler/SKILL.md)。无法获得的数据标记为未知。
4. **提出假设。** 按照 [planner.md](references/planner.md)，根据关键路径上的预期节省时间排序方案。`status` 返回 `plan_from_critical_path` 时，先量化与目标的差距，调查最大的未被掩盖开销，再做微优化。依据实测形状和反向支持情况，核查 HyperParallel 实现及版本匹配的[昇腾算子](../perf-playbook/references/ascend-fusion/index.md)。
5. **实现并执行。** 在隔离目录中，或通过显式功能开关实现候选。运行 `trial`：数值与兼容性 `precheck` → 不启用 profiler 的 `benchmark` → 独立的 `profile` → 完整的 `validate` → 任务专属的 `cleanup`。记录前三个阶段的精确命令和环境。
6. **验收与学习。** 仅保留正确、路径生效、证据完整且收益超过噪声阈值的结果。将失败、无效实验和只有局部收益的结果保留在历史中。对比最终追踪与预期中间信号，解释实验结果。将经过审阅、可推广的发现写入 perf-playbook 案例；推测保留为未经验证的假设。
7. **持续循环。** 阅读 `status`，将最佳候选的证据和新的负面结果反馈给规划过程，重复步骤 4～6。筛选结果达标后，进入关联的验收序列；验收通过后执行独立 `confirm`。实验次数或时间预算耗尽时，报告最佳有效候选。每次实验都会消耗预算，包括失败和确认实验；其他 Skill 的默认行为不能使循环变为无界执行。

运行故障或重大性能缺陷按[故障修复流程](../perf-playbook/references/failure-repair.md)处理。保留失败实验，接受受影响路径前必须提供原始工作负载的回归证据。诊断性回退或缩小输入不能作为问题已解决的依据。

<a id="execution-entry-points"></a>
## 执行入口

从仓库根目录运行。Agent 按协议创建 `run.json` 和 `plan.json`，它们可以位于仓库之外。

```bash
python .agent/skills/mfu-riser/scripts/mfu_riser.py init --run-dir /path/to/new-run --config /path/to/run.json
python .agent/skills/mfu-riser/scripts/mfu_riser.py trial --run-dir /path/to/new-run --plan /path/to/plan.json
python .agent/skills/mfu-riser/scripts/mfu_riser.py status --run-dir /path/to/new-run
python .agent/skills/mfu-riser/scripts/mfu_riser.py report --run-dir /path/to/new-run
python .agent/skills/mfu-riser/scripts/mfu_riser.py finalize --run-dir /path/to/new-run
```

`trial` 执行一个方案。**使用本 Skill 的 Agent 驱动完整循环**，用户无需逐轮发出新指令。目标未达成且预算有剩余时继续执行。`scripts/riser_contract.py` 强制执行预算、验收和决策规则。[关联序列协议](references/contracts.md#linked-series)说明如何修改经过授权的容量设置，或重跑筛选胜出的候选。

`scripts/audit_rank_steps.py` 复用 `steps_rankN.jsonl` 和 `meter_steps_rankN.jsonl` 记录，审计所有 rank 的输入、标量轨迹和独立推导的工作量，再导出测量样本。它会明确列出未覆盖的检查，但不替代模块梯度验证或融合路径生效验收。

获准扫描全局批量时，在安排下一次容量探测之前，应根据匹配的已完成实验凭据[估算梯度累积的缩放收益](references/planner.md#decision-process)。

`recover` 执行实验专属清理，将中断实验记录为 `crash` 后，才允许下一次实验。清理失败会使实验保持待处理状态。最佳结果指向工作区、配置和源码快照，不会覆盖用户源码或自动部署。没有 Git 时，应使用完整声明的源码哈希和快照。每个远端 rank 都必须确认其导入和测量的是相同源码。

<a id="outputs"></a>
## 输出

产物包括 `REPORT.md`、`controller.json`、只追加的 `history.jsonl`，以及每次实验的 `context.json`、`source.tar.gz`、阶段日志、`measurement.json`、性能分析 JSON 与 HTML、原始验收证据和 `result.json`。报告应区分局部算子加速、端到端收益、未达成的目标和待完成的数值验收。
`RSI_REPORT.html`、`RSI_REPORT.md` 和 `rsi/report.json` 单独说明仓库改进、复用的经验、测试和暂缓沉淀的经验。
目标确认或预算耗尽后，交付前应完成 `finalize_learning`；仅性能通过不能结束学习任务。

控制器不包含集群专属的 SSH 或 torchrun 启动器。首次接入时，需要复用项目启动器并实现协议适配器。缺少适配支持时，不得报告已完成 NPU 优化。
