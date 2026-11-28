<a id="automatic-rsi-experience-consolidation"></a>
# 自动沉淀 RSI 经验

每项 RSI 任务都交付两份独立产物：实验验收报告和 **RSI 改进报告**。后者说明框架学到了什么、哪些内容已经存在、修改了哪些仓库文件，以及如何验证。即使没有发现新经验，也必须提供这份报告。无需等待用户再次要求复盘。

<a id="policy"></a>
## 策略

| `rsi_mode` | 仓库更新时机 | Agent 职责 |
|---|---|---|
| `acceptance`（默认） | 最终验收时 | 优化期间收集证据，交付前复盘全部结果 |
| `continuous` | 已完成实验之间，清理结束后 | 合格证据出现时及时处理，验收时再次复盘整个实验序列 |

讨论任务时说明这两种选择。用户没有偏好时，使用 `acceptance`，不因此推迟启动。模式在每轮运行内固定。性能验收通过和经验沉淀完成仍是两个独立状态。

Agent 负责提取经验、解释因果、审阅语义重复项和设计补丁。确定性工具不包含额外的 LLM；它发现当前仓库结构、检索相关规则、检查证据和审阅覆盖、应用修改、运行回归并生成报告。词汇检索辅助语义审阅，不能证明内容具有新颖性。

<a id="required-agent-workflow"></a>
## Agent 必须执行的流程

1. 实验执行框架（Harness）初始化 `rsi/config.json`，并在实验清理后调用共享工具。工具盘点全部结果，包括崩溃、数值失败、噪声区间内的结果和负面实验。默认模式在迭代期间只将证据索引加入待处理队列，不重新扫描全部资料或应用补丁；完整审阅在验收时执行。连续模式则在每次实验完成的边界审阅资料。
2. 阅读 `rsi/review-task.json`、检索到的匹配项，以及当前归属 Skill 的入口。仓库结构和检索结果均从当前代码副本重新生成，不假定固定的 Skill 数量或历史目录映射。
3. 编写 `rsi/review.json`。覆盖每个实验，没有可复用经验时说明原因。声明新内容之前，先检索当前资料。
4. 调用 `learn`。默认模式在优化期间将修改加入待处理队列；连续模式在实验之间验证并应用修改。
5. 最终交付前调用 `finalize`。缺少审阅、证据已变化、仓库结构过期、覆盖不完整或回归失败，都会阻止成功完成。解决问题后重跑，并披露尚未解决的阻塞项。

`autoresearch` 创建只包含已验证经验文件的独立本地提交，以维持目标文件干净的协议。MFU Riser 应用经过验证的工作区修改并记录哈希。两条路径都不会推送、创建 PR 或发布；Git 操作应另行遵循任务已有的授权。
自动学习提交会拒绝目标文件中已有的修改，不会将其他贡献者尚未提交的工作混入学习提交。

<a id="classify-before-writing"></a>
## 写入前先分类

| 决策 | 处理方式 |
|---|---|
| `new` | 添加资料中尚不存在、合格且有用的经验；解释它与检索匹配项的区别 |
| `update` | 原位改进已有指导或实现；指出其当前哈希和局限 |
| `reuse` | 引用已有文件，不另建笔记，也不重复登记旧规则 |
| `defer` | 将未达到资格要求的筛选结果或未测试假设留在本地报告，说明缺少的证据 |
| `reject` | 解释为何该发现只适用于本地、重复、缺乏支持或无法复用 |

已经登记的等价主张会自动复用。措辞不同但语义等价的内容，需要 Agent 审阅检索资料。可以为已有规则补充适用条件、反例、检查或回归覆盖，而不是在其他 Skill 中新增重复段落。

不要发布机器名、pod ID、IP、账户名、挂载路径、仪表盘 URL、凭据或站点专属启动命令。这些信息保留在原始运行证据中。保留有用的技术约束：支持的形状/dtype/布局、API 可用性、硬件能力、分配器行为，以及有证据的软件版本边界。用通信域和带宽层级描述拓扑，不使用主机列表。工具会拦截常见部署标识；Agent 还必须审阅语义可移植性，包括模式检测器无法识别的标识。

<a id="routing-into-the-current-architecture"></a>
## 按当前仓库结构归档

| 可复用结果 | 归档位置 |
|---|---|
| 诊断或调优流程 | 归属 Skill 现有的参考文档；确属新规则时，放入其 `references/experience/`，并从入口链接 |
| 对已有规则的修正 | 直接修改原文件，不新增经验笔记 |
| 确定性的验收或核算不变量 | 现有 Harness 脚本，并配套正向和负向回归 |
| 重复的数据采集或分析操作 | 归属 Skill 的 `scripts/`，并从工作流链接 |
| 运行时缺陷 | 在归属 Skill 中记录诊断和已验证的修复策略；训练模块补丁按正常框架流程审阅 |
| 本地部署绕行方案 | 仅保留在运行目录 |

共享实现为 [rsi_learning.py](../../../scripts/rsi_learning.py)。它发现当前归属，检查补丁基准哈希，拒绝路径逃逸，并在验证失败时回滚仓库写入。可执行修改必须提供回归文件和通过的 argv 命令。工具不能重写本轮被测源码或冻结的验证器。必要时使用 `learning_repository` 将经验沉淀到独立代码副本；改变比较条件需要新建序列。

`.agent/learning/registry.json` 保存可移植主张、归属、形式、版本、代码位置和证据摘要。原始运行路径和部署身份保留在本地报告。新增源码和文档使用仓库的英文术语及代码风格。

<a id="review-contract"></a>
## 审阅协议

编写包含 `id` 和 `keywords` 的经验草稿，再刷新检索：

```bash
python .agent/scripts/rsi_learning.py inspect --repo . --run-dir outputs/example-run
```

填写 `inventory_digest`、每个实验的 ID 和摘要，以及 `related_review`。下面是流程提案示例；声明该示例是新内容之前，先检查仓库：

```json
{
  "schema": "hp-rsi-learning-review/v1",
  "inventory_digest": "<architecture digest>",
  "coverage": [{
    "event_id": "<experiment ID>", "sha256": "<observed digest>",
    "lesson_ids": ["logical-state-identity"], "reason": "Initialization confounder reproduced"
  }],
  "lessons": [{
    "id": "logical-state-identity", "decision": "new", "owner": "precision",
    "form": "procedure", "qualification": "validated",
    "title": "Logical initial state across layouts",
    "keywords": ["logical", "initialization", "layout"],
    "trigger": "Two numerical runs use different sharding layouts.",
    "mechanism": "Rank-local random streams can initialize different logical weights.",
    "claim": "Verify logical initial state before interpreting cross-layout numerical differences.",
    "applicability": "Equivalent models with different shard placement.",
    "limits": "Initial-state identity alone does not prove gradient or convergence equivalence.",
    "reason": "Explain novelty after reviewing current precision procedures.",
    "related_review": [{
      "path": "<retrieved file>", "sha256": "<current digest>",
      "relationship": "different", "reason": "Explain the actual scope difference"
    }],
    "evidence": [{"path": "identity-check.json", "sha256": "<actual evidence digest>"}]
  }]
}
```

已有规则时使用 `reuse` 或 `update`。经过验证的失败，可以用资格值 `negative` 表示有范围限定的反例或检测规则。不合格的筛选结果和未测试假设使用 `defer`，不写入仓库指导。

修改代码或已有文档时，通过 `changes` 提供 `path`、`before_sha256`（新文件为 `null`）和完整 UTF-8 `content`。可执行修改还需要 `regression_tests` 和 `validation: [{"argv": ["python", ".agent/tests/tools/test_example.py"], "timeout_seconds": 120}]`。

经验沉淀对仓库的更新应位于 `.agent/` 下。框架实现修改保留在被测实验代码副本中，记录可复用经验时链接其证据。使用真实的仓库相对测试命令；仅返回零退出码的命令不算验证。保留已经证明的失败机制和一个有效用例。

<a id="commands-and-outputs"></a>
## 命令与输出

```bash
python .agent/skills/autoresearch/scripts/autoresearch.py learn --run-dir outputs/example-run
python .agent/skills/autoresearch/scripts/autoresearch.py finalize --run-dir outputs/example-run
python .agent/skills/mfu-riser/scripts/mfu_riser.py learn --run-dir outputs/example-run
python .agent/skills/mfu-riser/scripts/mfu_riser.py finalize --run-dir outputs/example-run
```

只使用本轮运行所属的入口，不要同时使用两者。外部适配器使用 `rsi_learning.py sync --repo . --run-dir ... --phase acceptance`，并提供 `rsi/experiments.json`，其结构为 `experiments: [{id, path, sha256, decision}]`；路径相对运行目录，必须覆盖整个实验序列。

每次整理都会写入 `RSI_REPORT.html`、`RSI_REPORT.md` 和 `rsi/report.json`，说明经验、决策、形式、代码位置、资格、测试结果及未解决事项。`rsi/applied.json` 记录审阅摘要和文件摘要，以支持幂等完成。验证失败会保留本地日志，并恢复仓库文件。
