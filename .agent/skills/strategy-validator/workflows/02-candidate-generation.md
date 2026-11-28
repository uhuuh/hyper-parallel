<a id="phase-2-candidate-generation"></a>
# 阶段 2：候选生成

<a id="objective"></a>
## 目标
根据基线分析、模型特征和已有启发式规则，生成 3～5 个候选策略。

<a id="inputs"></a>
## 输入
- 阶段 1 的基线指标
- 模型架构：
  - `num_layers`：Transformer 层数
  - `hidden_size`、`num_attention_heads`、`intermediate_size`
  - `is_moe`：模型是否采用 MoE
- 硬件：
  - `device_count`：可用设备总数，例如 8
  - `device_memory_gb`：每设备显存，例如 64 GB
- 训练配置：
  - `seq_len`：序列长度
  - 目标提升倍数，默认为 1.2 倍

<a id="strategy-generation-rules"></a>
## 策略生成规则

<a id="rule-1-model-size--tp"></a>
### 规则 1：由模型规模选择 TP
```python
if num_layers <= 10:
    tp_candidates = [1, 2]  # Prefer TP=1
elif num_layers <= 30:
    tp_candidates = [2, 4]
else:
    tp_candidates = [4, 8]
```

<a id="rule-2-sequence-length--cp"></a>
### 规则 2：由序列长度选择 CP
```python
if seq_len < 512:
    cp = 1  # Always 1 for short sequences
elif seq_len < 2048:
    cp_candidates = [1, 2]
else:
    cp_candidates = [2, 4, 8]
```

<a id="rule-3-moe--ep"></a>
### 规则 3：由 MoE 选择 EP
```python
if is_moe:
    ep >= 2  # Required by framework
    # EP should divide num_experts evenly
else:
    ep = 1
```

<a id="rule-4-fsdp--dp"></a>
### 规则 4：由 FSDP 选择 DP
```python
# Always keep DP >= 2 (avoid FSDP conflicts)
dp_min = 2

# DP = total_devices / (TP × CP × EP × PP)
dp = device_count // (tp * cp * ep * pp)
if dp < dp_min:
    # Reduce other dimensions
    pass
```

<a id="rule-5-baseline-optimization"></a>
### 规则 5：优化基线
根据阶段 1 识别的瓶颈：

**通信受限：**
- 已启用 CP 时将其移除
- 模型较小时减小 TP
- 增大 DP，降低其他维度

**计算受限：**
- 可能已经最优
- 尝试减小 DP，并在 TP 未达上限时增大 TP

**显存受限：**
- 增大 DP，增加 FSDP 分片
- 减小 TP/PP，减少激活复制

<a id="candidate-generation-algorithm"></a>
## 候选生成算法

```python
def generate_candidates(baseline, model_info, hardware, target_improvement=1.2):
    candidates = []

    # Candidate 1: Remove inefficient dimensions
    if baseline.cp > 1 and model_info.seq_len < 512:
        candidates.append({
            "id": "remove_cp",
            "name": "Remove CP (communication reduction)",
            "strategy": {
                "tp": baseline.tp,
                "cp": 1,  # Remove CP
                "ep": baseline.ep,
                "pp": baseline.pp,
            },
            "expected_improvement": "1.5-2.0x (if CP was bottleneck)",
        })

    # Candidate 2: Minimize TP
    if baseline.tp > 1 and model_info.num_layers <= 10:
        dp = hardware.device_count // (1 * 1 * baseline.ep * 1)
        if dp >= 2:  # Satisfy DP constraint
            candidates.append({
                "id": "minimal_tp",
                "name": "TP=1 (eliminate TP communication)",
                "strategy": {
                    "tp": 1,
                    "cp": 1,
                    "ep": baseline.ep,
                    "pp": 1,
                },
                "dp": dp,
                "expected_improvement": "1.5-2.5x (if TP was bottleneck)",
            })

    # Candidate 3: Increase parallelism (if computation-bound)
    if baseline.bottleneck == "computation":
        new_tp = min(baseline.tp * 2, 8)
        dp = hardware.device_count // (new_tp * baseline.cp * baseline.ep * baseline.pp)
        if dp >= 2:
            candidates.append({
                "id": "increase_tp",
                "name": f"TP={new_tp} (more parallelism)",
                "strategy": {
                    "tp": new_tp,
                    "cp": baseline.cp,
                    "ep": baseline.ep,
                    "pp": baseline.pp,
                },
                "expected_improvement": "1.1-1.3x",
            })

    # Candidate 4: Conservative (minor tweaks)
    # ... additional candidates based on specific patterns

    return candidates[:5]  # Limit to top 5
```

<a id="outputs"></a>
## 输出

为每个候选生成：
1. **配置文件：** `validation_experiment/{candidate_id}_config.yaml`
2. **启动脚本：** `validation_experiment/run_{candidate_id}.sh`
3. **元数据：** 预期收益、理由和风险级别

输出示例：
```json
{
  "candidates": [
    {
      "id": "candidate1_remove_cp",
      "name": "Remove CP",
      "strategy": {"tp": 2, "cp": 1, "ep": 2, "pp": 1, "dp": 2},
      "expected_tokens_per_sec": 1500,
      "rationale": "seq_len=128 too short for CP=2, communication overhead dominates",
      "risk": "low"
    },
    {
      "id": "candidate2_minimal_tp",
      "name": "Minimal TP",
      "strategy": {"tp": 1, "cp": 1, "ep": 2, "pp": 1, "dp": 4},
      "expected_tokens_per_sec": 2000,
      "rationale": "4-layer model, TP AllReduce is bottleneck",
      "risk": "medium"
    }
  ]
}
```

<a id="example-from-experiment"></a>
## 实验示例

**基线：** TP=2、CP=2、EP=2、DP=2，吞吐 1081 tokens/s，通信受限。

**生成的候选：**
1. TP=2、CP=1、EP=2、DP=2：移除 CP
2. TP=2、CP=1、EP=4、DP=1：增大 EP，因 DP=1 问题失败
3. TP=4、CP=1、EP=2、DP=1：增大 TP，因 DP=1 问题失败
4. TP=1、CP=1、EP=2、DP=4：使用最小 TP，胜出，吞吐 4274 tokens/s

---

**下一步：** 阶段 3，批量验证。
