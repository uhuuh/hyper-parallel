<a id="configuration-proposal-and-evidence-contracts-v1"></a>
# 配置、方案与证据协议 v1

Agent 为当前实验生成路径和命令。请将下文的 `/absolute/...` 占位符替换为实际文件。控制器执行 argv 列表，不执行 Shell 字符串。SSH、环境准备和 torchrun 编排应放入经过审阅的脚本，再通过 argv 调用。`validator_files` 包含验收实现及其依赖，这些文件不能作为候选调优目标。

## run.json

```json
{
  "schema": "mfu-riser/v1",
  "rsi_mode": "acceptance",
  "workload": {
    "model": "model architecture and config hash",
    "weights_id": "checkpoint manifest hash",
    "dataset_id": "dataset hash plus seed and ordering",
    "optimizer": "full optimizer configuration and hash",
    "precision": "bf16",
    "routing": "native",
    "global_batch_size": 32,
    "world_size": 32,
    "nodes": ["node-a", "node-b", "node-c", "node-d"],
    "peak_tflops_per_device": 376,
    "flops_convention": "PaLM full SxS; pinned counter version",
    "timing_scope": "synchronized complete train step, cross-rank MAX; exact log boundary declared",
    "train_steps": 50,
    "window": [11, 50]
  },
  "target_mfu_percent": 20,
  "max_trials": 6,
  "max_seconds": 7200,
  "command_timeout_seconds": 1200,
  "noise_fraction": 0.02,
  "validator_files": ["/absolute/adapter.py", "/absolute/check_numerics.py"],
  "commands": {
    "precheck": ["/absolute/python", "/absolute/adapter.py", "precheck"],
    "benchmark": ["/absolute/python", "/absolute/adapter.py", "benchmark"],
    "profile": ["/absolute/python", "/absolute/adapter.py", "profile"],
    "validate": ["/absolute/python", "/absolute/adapter.py", "validate"],
    "cleanup": ["/absolute/python", "/absolute/adapter.py", "cleanup"]
  }
}
```

376、20 和 2% 只是示例值，不代表硬件峰值、可实现的目标或实际噪声。使用同节点 A/A 测量或已有的可比记录校准噪声。允许添加工作负载字段；每个字段都会参与比较。

`rsi_mode` 默认为 `acceptance`；若需在已完成实验之间整理经验，可选择 `continuous`。可选的 `learning_repository` 指定接收可移植改进的当前 HP 代码副本；未指定时使用实验工作区。对已冻结源码或验证器的修改应在独立代码副本中进行，参见[经验整理协议](../../autoresearch/references/experience-consolidation.md)。

<a id="performance-policy"></a>
## 性能策略

使用 S4 性能流程时，在运行配置中添加 `performance_policy`。添加 `workload.model_config_sha256`，记录固定模型定义的摘要，并在初始工作负载中包含所有可调字段。下面的示例允许修改序列长度和全局批量。

```json
{
  "validation_level": "screening",
  "low_mfu_threshold_percent": 10,
  "tunable_workload_fields": ["sequence_length", "global_batch_size"],
  "flops_accounting": {"method": "architecture_aware"}
}
```

- `screening` 使用声明的预热后窗口和实现正确性检查。`acceptance` 要求至少 20 个稳态步及完整数值检查。筛选运行可以保留候选，供后续测试。
- 任务未规定低 MFU 策略时，省略该阈值。低于配置的阈值时，`status` 返回 `plan_from_critical_path`，以及当前工作负载下目标吞吐相对最佳吞吐的倍数。规划时应根据实测的未被掩盖耗时估算候选收益。
- 已有经过验证的最佳结果时，`status` 会给出在固定工作量下还需节省的时间（非负值），以及相同 MFU 口径下对应的单步时间预算。
- 只列出任务允许调整的工作负载字段。支持的字段为 `sequence_length`、`micro_batch_size`、`global_batch_size` 和 `gradient_accumulation_steps`。
- 使用 `{"method": "approximation", "max_relative_error": 0.02}` 时，应声明相对误差上限，并提供逐步参考计算。控制器会将近似值与按模型架构分解的结果进行核对。

<a id="linked-series"></a>
## 关联序列

从当前已验证的父序列结果初始化容量序列或验收序列。

```bash
python .agent/skills/mfu-riser/scripts/mfu_riser.py init \
  --run-dir /path/to/new-series --config /path/to/child.json \
  --parent-run /path/to/parent-series
```

容量序列只修改父序列声明为可调的字段，保留模型定义、初始状态、数据身份、路由、硬件和核算口径。为新序列测量基线，后续实验与该序列自身的最佳结果比较。

验收序列将 `validation_level` 从 `screening` 改为 `acceptance`，并可延长 `train_steps` 和 `window`。首轮实验重跑筛选胜出的源码清单，并继承其路径生效检查。验收后还应执行独立确认。两种转换都会记录父运行、父实验、结果摘要和源码摘要；新序列的基线和最佳结果记录初始均为空。

## plan.json

基线使用 `kind=baseline, parent=null, actions=[]`，并指定 `workspace` 和 `source_paths`。下面的候选仅用于举例，并不表示 RMSNorm 是当前模型的热点。

```json
{
  "kind": "candidate",
  "parent": "000-existing-best-id",
  "workspace": "/absolute/isolated-candidate",
  "source_paths": ["hyper_parallel", "train.yaml", "launch.sh", "fusion_adapter.py"],
  "operator_scope": "compute_fusion",
  "actions": ["rms-norm-fusion"],
  "observation": "Operator, rank, step, and shape from state, with the source call site",
  "mechanism": "Reduce intermediate tensors and kernel launches in the normalization chain",
  "expected_effect": "Range justified by the critical path or a synchronized probe; probe first if unknown",
  "falsifier": "Fusion engages but neither the targeted interval nor complete-step time decreases",
  "compatibility": "Verified and pending dtype, layout, epsilon, backward, and version constraints",
  "evidence": ["/absolute/best-trial/profile/state.json"],
  "knowledge": ["/absolute/perf-playbook/references/ascend-fusion/index.md"]
}
```

配置 `performance_policy` 后，`operator_scope` 可以是 `compute_fusion`、`equivalent_implementation`、`parallel_layout`、`communication`、`activation_memory` 或 `optimizer`。等价实现范围包括内核和模块替换。所有范围都必须保留冻结的工作负载和优化器语义。在现有方案字段中说明实测瓶颈、受影响路径和预期的完整训练步收益。

`source_paths` 应包含所有影响被测代码、配置和启动器的文件或目录，路径相对 `workspace`。排除输出目录，否则写入训练日志也会触发源码变更检测。冻结范围不能仅限于编辑的文件。使用 worktree、独立代码副本或复制目录进行隔离。也支持在共享目录中使用显式功能开关，但实验失败后，Agent 必须从 `best` 引用的配置继续。控制器不会恢复用户文件。

确认实验使用 `kind=confirm`，`parent` 指向当前最佳结果，并使用相同的 `workspace`、`source_paths` 和 `actions`。控制器要求源码清单摘要相同；确认实验不能包含新修改。

<a id="runtime-environment"></a>
## 运行环境

五个阶段都会收到 `RISER_TRIAL_DIR`（唯一的绝对目录）、`RISER_TRIAL_TOKEN`、`RISER_CONTEXT`（`context.json`）、`RISER_SOURCE_DIGEST` 和 `RISER_WORKSPACE`。上下文记录配置、计划、源码清单和参考实验。适配器不得将这些预期值冒充为工作进程的实测值。工作进程应独立计算并记录源码哈希和实际工作负载。结果文件必须来自本轮实验。验证应读取原始证据，失败时以非零退出码结束。

## measurement.json

验证器写入 `RISER_TRIAL_DIR/measurement.json`。

```text
schema: "mfu-riser-measurement/v1"
trial_token: current trial identity; raw worker records must carry the same token
workload: actual configuration, exactly matching run.json workload
profiler_during_benchmark: false
runtime: [{rank: 0, source_digest: independently measured worker hash, steps_completed: 50}, ...all ranks]
samples: [{step, time_s, model_tokens, model_flops, mfu_percent}, ...complete fixed window]
checks: {
  real_weights: {passed: true, evidence: "audit/weights.json", sha256: "..."},
  all_ranks_complete: same structure,
  finite_loss_grad: same structure,
  input_equivalence: same structure,
  numerical_equivalence: same structure,
  independent_flops: same structure,
  timing_boundary: same structure,
  fusion_active: same structure, required when actions are present,
  fallback_correct: same structure, required when actions are present
}
profile_state: "profile/state.json"
```

所有证据路径必须位于当前实验目录内，每个 `sha256` 都必须与对应的原始检查报告匹配。基线的输入和数值检查用于验证参考实现是否符合声明的数据；候选检查则与冻结的参考路径比较。仅有非空或有限输出，不能证明 `numerical_equivalence=true`。

`samples` **每步只包含一条全局记录**。不要将每个 rank 重复记录的全局值相加。`time_s` 是所有 rank 中完整训练步耗时的最大值。`model_tokens` 是实际的全局非填充 token 数。`model_flops` 是经过独立验证的全局有效模型工作量。控制器逐步检查 MFU，并要求候选的 token 数和模型 FLOPs 与参考逐步一致。局部内核执行 FLOPs 不等于模型 FLOPs。变长训练步应使用实际长度，不要假定每步长度都是 4096。

```text
Per-step MFU = model_flops / (time_s × world_size × peak_tflops_per_device × 10^12) × 100
Window MFU = sum(model_flops) / (sum(time_s) × world_size × peak × 10^12) × 100
Window throughput = sum(model_tokens) / sum(time_s)
Throughput per device = window throughput / world_size
```

性能策略实验应将 `validation_level` 设为冻结的级别。提供 `model_semantics` 检查，覆盖模型定义、输入与掩码语义、布局、逻辑 Top-K、梯度归约和参数更新。筛选使用 `implementation_correctness`，验收使用 `numerical_equivalence`。融合变更使用 `fusion_active` 和 `fallback_correct`；其他候选范围使用 `candidate_active`，并附上对应运行路径的证据。

`independent_flops` 检查指向本轮实验的 JSON 核算报告。元数据包含 `trial_token`、`source_digest`、`model_config_sha256`，以及与 `workload.flops_convention` 一致的 `convention`。其中的 `steps` 列表与测量窗口匹配，每行包含 `step`、`model_tokens` 和 `components`。分解项将路由专家 GEMM、稀疏注意力等操作类别映射为独立推导的正数有效 FLOPs。对于 `architecture_aware`，控制器对分解项求和；对于 `approximation`，逐行将 `approximate_model_flops` 与该总和比较，并强制执行声明的误差上限。应向冻结的验证器提供架构推导和形状证据。

可选的 `executed_work` 包含 `method`（`hardware_counters` 或 `estimate`）、`coverage`、`evidence`、`sha256`，以及含 `step` 和 `executed_flops` 的 `samples`。每一步都必须属于未启用 profiler 的窗口，且只能出现一次。验证器检查计数是否覆盖声明的设备集合和配置。控制器使用匹配的未启用 profiler 的步耗时计算 HFU，并记录采样步、覆盖范围和方法。它还会在相同步骤和时间范围内计算 `model_mfu_in_hfu_window_percent`。有硬件计数器时，`status` 使用该匹配的 MFU 计算 `target_hfu_percent_at_fixed_work_ratio`。对于估算结果，以及缺少匹配 MFU 的旧凭据，该字段为 null。

可选的 `allocation` 包含 `device_seconds`、`evidence` 和 `sha256`。设备秒数应根据实际作业资源分配计算。实验结果还记录控制器各阶段的墙钟耗时，包括未成功的阶段；阶段耗时包含启动器中的等待时间。

<a id="profiling-state-and-existing-skills"></a>
## 性能分析状态与现有 Skill

运行 `scripts/collect_state.py --prof-dir CAPTURE --out NEW_DIRECTORY`，也可通过 `--prior` 和 `--memory` 提供现有 Skill 输出的 JSON。采集器直接调用 profiling 的 `breakdown.py`，保留其检查和来源信息，并从 `kernel_details.csv` 添加实际形状摘要。它不会修改性能分析公式、从 HTML 推断数值，或编造未知形状和 dtype。

结果为 `valid=false` 时，实验不能被提升为保留结果。对单设备、单节点和多节点运行，都应声明采集的 rank 和阶段。采集器验证提供的报告，但不能证明覆盖了全部全局 rank。适配器必须覆盖该拓扑中有代表性的 rank 和慢 rank，尤其是 PP、EP，并在 `timing_boundary` 证据中记录覆盖情况。

与控制器绑定的采集还要求提供 `--capture-metadata PROFILE_WORKER_METADATA.json`。被测性能采集工作进程将实际实验 token 和源码摘要写入该文件，采集器将它们保存在 `measurement_binding` 中。缺少此绑定的旧追踪仍可用于离线分析，但不能用于提升新的实验结果。冻结的验证适配器检查性能分析与基准测量之间的输入和工作负载一致性。

有计时但缺少形状时，状态保留为 `partial_or_unknown`。选择融合实现前，规划过程必须通过 `record_shapes` 或独立形状探针补齐证据。
