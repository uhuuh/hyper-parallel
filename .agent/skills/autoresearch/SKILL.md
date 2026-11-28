---
name: autoresearch
description: >
  面向内核和计算图优化的有界自主实验循环：性能测量、正确性门禁、keep/discard/crash 决策、
  只追加的实验记录，以及 Git 提交与恢复协议。四条规则由代码强制执行，而非依赖 Agent 自觉遵守。
  适用于自动实验、性能优化循环、autoresearch、算子调优迭代，或任何需要将待提交修改与历史最佳结果比较、
  仅在胜出且通过门禁时保留的工作流。Agent 负责选择想法和修改代码，本 Skill 负责确定性的执行部分。
---

# Autoresearch

有预算边界的自主实验循环。每轮对应一次优化实验：修改声明的单个目标文件、测量性能、通过正确性门禁、记录结果，并决定 keep / discard / crash。循环持续到 Agent 停止。设计参考 torchtitan 的 `graph_trainer/autoresearch`，将文字描述的确定性协议落实为代码。

**本文件提供索引和硬性规则。** 每次运行的目录中会生成操作手册 `manual.md`，需要填写其中的 `[SETUP]`。

<a id="four-rules-enforced-in-code"></a>
## 由代码强制执行的四条规则

这些规则来自实际事故，不依赖 Agent 的记忆：

1. **性能测量只针对已提交代码。** 远端测量按提交同步；未提交修改会使测量静默落在旧版本上，因此 `iterate` 先提交实验。
2. **必须提供正向生效标记**，例如 `engaged=1`。缺失时判为 `crash`，不能静默通过；仅凭“没有警告”，无法区分回退路径和优化路径。
3. **门禁失败使指标作废**，无论数值多好看。
4. **失败保留在历史中。** `discard`/`crash` 通过新的正向提交恢复目标文件，不改写历史；当前最佳状态始终位于 HEAD。

操作手册中的两条禁令用于保护盲测（生成于 `manual.md`）。回顾先前实验时使用本轮运行的产物；诊断门禁时使用脚本的断言输出。

<a id="subcommands"></a>
## 子命令

```bash
python .agent/skills/autoresearch/scripts/autoresearch.py <subcommand> [options]
```

| 意图 | 命令 | 用途 |
|---|---|---|
| 初始化运行目录 | `init --run-dir <d> --target <f> --benchmark-cmd "<c>"` | 创建 run.json 和持续维护的文档 |
| 记录基线 | `baseline --run-dir <d>` | 测量当前提交，记录为首个 keep |
| 评估待提交修改 | `iterate --run-dir <d> --description "<what>"` | 提交→门禁→测量→决策→记录→保留或恢复 |
| 查看进展 | `status --run-dir <d>` | 汇总当前运行状态 |
| 整理经验 | `learn --run-dir <d>` | 应用选定的 RSI 学习策略 |
| 完成交付 | `finalize --run-dir <d>` | 要求完成全部经验复盘，并生成报告 |

`iterate` 要求实验修改在目标文件中保持**未提交**，运行目录之外不能有其他未提交修改。

<a id="runjson-contract"></a>
## run.json 协议

- `benchmark_cmd` / `gate_cmd`：**argv 列表**。Shell 语法只能出现在被调用脚本中，不能写入配置。
- `metric_pattern`：包含 `(?P<metric>...)` 捕获组的正则表达式，取最后一次匹配。
- `metric_lower_is_better`、`noise_fraction`：指标优化方向和丢弃区间。
- `noise_confirm_reruns`（默认 0）：首个样本落入噪声区间时，再运行 N 次，使用中位数决策。
- `aux_patterns`：`{name: regex-with-(?P<value>...)}`，捕获值写入结果行的描述。
- `bench_must_match` / `gate_must_match`：必须出现的字面标记，缺失即 crash。
- `training_protocol`：可选的工作负载 ID 和完整窗口步数，应配合独立 `gate_cmd` 及[结构化训练凭据](../perf-playbook/references/training-rsi.md#structured-training-receipts)使用。
- `rsi_mode`：`acceptance`（默认）或 `continuous`；初始化时通过 `--rsi-mode` 设置。两种模式最终都要求 RSI 改进报告。

<a id="the-experiment-loop-agent-side"></a>
## Agent 侧实验循环

1. 阅读 `status`、`ideas.md`、`learnings.md` 和 `experiment_log.md`。选择无关的新想法前，先检查 `retry_ready` 候选。
2. 修改目标文件，工作区只保留这些文件的未提交修改。
3. 执行 `iterate --run-dir <d> --description "..."`，其余步骤由工具完成。
4. 追加 `experiment_log.md` 条目，更新 `ideas.md` / `learnings.md`，按[自动经验沉淀协议](references/experience-consolidation.md)处理生成的任务。无需等待用户再次要求整理经验。

在声明的预算内继续。记录哪些目标已达成、哪些仍未解决，以及停止原因。最终交付前运行 `finalize`，同时交付验收报告和独立的 `RSI_REPORT.html`。

<a id="candidate-receipts-and-retries"></a>
## 候选凭据与重试

每次 `iterate` 向 `candidates.jsonl` 追加凭据，并在 `results.tsv` 中记录指标历史。凭据包括父提交和候选提交、决策阶段与结果、门禁状态、必需标记和配置的辅助捕获值。如果执行器会输出简短失败原因、路径生效结果或证据标识，可为它配置辅助捕获。使用结构化协议的训练运行还会记录基线及每次门禁、测量调用，包括失败的噪声确认尝试。

- 使用可重复的 `--changes <scope>` 标记修改范围。同一运行内使用稳定名称，如 `optimizer-residency` 或 `operator-path`。
- 声明 `--revisit-on <scope>` 的候选，在后续**被保留**的候选修改该范围时变为 `retry_ready`。`iterate` 和 `status` 都显示符合条件的提交。将触发条件和计划中的重新应用方式写入实验日志。
- 根据本轮笔记或补丁，将想法重新应用到当前保留配置。使用 `--retry-of <commit>` 和适当门禁作为新候选运行，该重试条目随后被消费。后续重试可以声明自己的触发条件。每次重试都遵守本轮正常的资源准入规则和预算。

训练运行遵循[训练 RSI 协议](../perf-playbook/references/training-rsi.md)。

<a id="hard-rules-must-not-violate"></a>
## 硬性规则（不得违反）

- 不得修改引擎 `config.py` / `loop.py` 来使运行通过。
- 性能测量和门禁命令必须使用 argv 列表，不使用 `shell=True` 或内联管道。
- 噪声区间内的结果判为 `discard`，不能强行保留。
- 引擎仅依赖标准库，并保持 pylint 检查通过。

<a id="field-record"></a>
## 实测记录

两次实际运行推动了 DeepSeek-V4.1 融合 Lightning-Indexer 适配：第一次通过三次实验，将四探针指标从 697ms 降至 123ms；第二次以 master 为基线，使用包含长序列的六探针，将 4617ms 降至 261ms，并由门禁发现错误的 bf16 评分优化，记录为 crash。两次运行的产品修改通过 fused-indexer PR 交付；完整日志保存在运行目录 `autoresearch/` 和 `autoresearch2/` 中。

<a id="unit-tests"></a>
## 单元测试

`.agent/tests/tools/test_autoresearch_loop.py` 在临时 Git 仓库中验证 baseline / keep / discard / crash / 脏工作区 / 噪声确认 / 辅助捕获行为，只需 CPU，不需要加速器。测试与入口脚本一样，将引擎目录加入 `sys.path`。
