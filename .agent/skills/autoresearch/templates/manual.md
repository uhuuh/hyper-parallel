<a id="autoresearch-operating-manual"></a>
# Autoresearch 操作手册

本手册用于指导 Agent。每轮在声明的目标文件内实现一个候选，运行门禁和性能测量，并记录 `keep`、`discard` 或 `crash`。在本轮运行预算内持续执行。

<a id="setup-run-configuration"></a>
## [SETUP] 运行配置

- **优化目标：** <FILL: 要优化什么>
- **主指标：** <FILL: benchmark_cmd 的输出字段及优化方向>
- **起点：** <FILL: 当前实现和已知性能>
- **辅助条件：** <FILL: 允许阅读的参考资料，以及网络或历史记录的访问权限>
- **预算和停止条件：** <FILL: 时间或实验次数、相对基线的改进，以及适用时的独立性能目标>
- **测量窗口：** <FILL: 正确性与性能窗口、拓扑、探针、路由和指标口径>
- **变更类别与门禁：** <FILL: 等价实现、数值容差或训练策略变更所需的证据>
- **重试标签：** <FILL: 本轮使用的稳定配置范围名称，例如 optimizer-residency>
- **RSI 学习模式：** 默认为 `acceptance`；用户选择时使用 `continuous`。

<a id="editable-files"></a>
## 可编辑文件

使用本目录 `run.json` 中的 `target_files`。列表以外存在未提交修改时，工具拒绝执行性能测量。

**不得修改：** <FILL: 参考实现、门禁脚本、Trainer，或其他修改后会影响评估的受保护文件>。
可在本轮运行目录的 `scripts/` 下添加分析工具。

<a id="read-scope"></a>
## 阅读范围

**可以阅读：** <FILL: 目标文件、只读参考实现、当前运行文件和自行生成的性能分析结果>
**不得阅读：** <FILL: 无关子系统；扩大范围前先在 experiment_log 中记录原因>

<a id="history-and-gate-source-restrictions"></a>
## 历史记录与门禁源码限制

`[SETUP]` 不能放宽以下要求。

- **不得查看 Git 历史。** 禁止使用 `git log`、`git log -p`、`git show <commit>`、`git diff <commit>` 和 `git blame`。历史提交标题和差异统计可能泄露本轮应独立重新发现的优化。允许使用工作区 `git status`，以及不带提交引用的 `git diff`。通过 `results.tsv`、`candidates.jsonl` 和本轮实验日志读取本轮提交及失败证据。
- **不得阅读门禁脚本源码。** 执行门禁，并根据输出中的断言消息诊断失败。门禁源码中的符号可能泄露先前的优化选择。
- 对阅读授权不确定的材料，按范围外处理。

<a id="benchmark-and-correctness-gate"></a>
## 性能测量与正确性门禁

`run.json` 定义 `benchmark_cmd` 和 `gate_cmd`。缺少 `bench_must_match` 或 `gate_must_match` 要求的正向标记（例如 `engaged=1`）时，结果为 `crash`。没有警告不代表优化路径实际执行。首次测量落入噪声区间时，配置的自动复测会先取中位数，再作决策（`noise_confirm_reruns`）。不要用单次读数覆盖这一结果。

<a id="records"></a>
## 记录

工具向 `results.tsv` 和 `candidates.jsonl` 追加记录，并管理候选提交和恢复。每轮结束后追加 `experiment_log.md`，更新 `ideas.md` 和 `learnings.md`。对于可能重试的候选，保留可复用的改动描述，或本轮生成的补丁。

<a id="experiment-loop"></a>
## 实验循环

在声明的预算内：

1. 运行 `status`，先检查 `retry_ready` 候选，再结合 `ideas.md`、`learnings.md` 和 `experiment_log.md` 选择下一项实验。
2. 在 `target_files` 中实现修改，只保留这些文件的改动。上下文有限时，可将实现和执行委派给子 Agent。应原样传递本手册的限制和命令；主循环接收结果，不为被委派者重写规则。
3. 运行 `autoresearch.py iterate --run-dir <this-directory> --description "<short-description>"`，执行提交、验证和性能测量。用 `--changes` 声明修改范围，用 `--revisit-on` 声明依赖范围。重试时，将改动重新应用到当前最佳配置，并传入 `--retry-of <commit>`。
4. 将返回结果追加到 `experiment_log.md`，并更新想法和经验。

**崩溃：** 找到最先失败的阶段，区分训练失败、验证失败和资源失败。记录修复、重试条件和证据路径。

**停止：** 记录目标是否达到、剩余差距和停止原因。根据生成的检索任务，在 `rsi/review.json` 中复盘全部实验。复用已有规则；通过 `learn` 原位改进规则，或添加合格且可移植的经验。运行 `finalize`，将 `RSI_REPORT.html` 与验收报告一并交付。这是 Agent 应自动完成的工作，无需用户再次提醒。

<a id="environment-requirements"></a>
## 环境要求

- <FILL: 资源占用检查和共享规则；不干扰其他作业，完成后释放资源>
- <FILL: 已知环境约束，例如超时、缓存和并发编译>
