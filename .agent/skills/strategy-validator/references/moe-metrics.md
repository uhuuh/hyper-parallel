<a id="moe-specific-metrics-for-strategy-validation"></a>
# MoE-Specific Metrics for Strategy Validation（策略验证的 MoE 专用指标）

对混合专家模型**标准验证指标的扩展**。

验证 MoE 模型训练策略时加载本参考文档。

---

<a id="overview"></a>
## 概述

标准验证指标，如吞吐、显存和 MFU，不足以分析 MoE 模型。需要额外的 MoE 专用指标来诊断：
- 专家负载不均衡
- All-to-All 通信开销
- 路由效率
- 慢 rank 问题

---

<a id="metric-collection-points"></a>
## 指标采集位置

<a id="1-during-training-loop"></a>
### 1. 训练循环中

在 MoE 模块中插入 hook，采集逐步指标：

```python
from hyper_parallel.core.expert_parallel import MoE

def setup_moe_hooks(model):
    """Setup hooks to collect MoE metrics during training"""
    moe_metrics = {
        "tokens_per_expert": [],
        "expert_load_std": [],
        "routing_entropy": [],
        "dropped_tokens": [],
    }

    def moe_forward_hook(module, input, output):
        if hasattr(module, 'tokens_per_expert'):
            # Capture token distribution
            tokens = module.tokens_per_expert.cpu().numpy()
            moe_metrics["tokens_per_expert"].append(tokens)

            # Compute load std
            load_std = compute_load_std(tokens)
            moe_metrics["expert_load_std"].append(load_std)

            # Capture routing probs if available
            if hasattr(module, 'routing_probs'):
                entropy = compute_routing_entropy(module.routing_probs)
                moe_metrics["routing_entropy"].append(entropy)

            # Count dropped tokens (if capacity exceeded)
            if hasattr(module, 'dropped_tokens_count'):
                moe_metrics["dropped_tokens"].append(
                    module.dropped_tokens_count.item()
                )

    # Register hooks for all MoE layers
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            module.register_forward_hook(moe_forward_hook)

    return moe_metrics
```

<a id="2-during-all-to-all-communication"></a>
### 2. All-to-All 通信期间

测量通信开销：

```python
import torch.distributed as dist
import time

def measure_alltoall_overhead(moe_module, step_start_time):
    """Measure All-to-All communication time"""

    # Hook into All-to-All calls
    alltoall_times = []

    original_alltoall = dist.all_to_all_single

    def timed_alltoall(*args, **kwargs):
        start = time.time()
        result = original_alltoall(*args, **kwargs)
        elapsed = time.time() - start
        alltoall_times.append(elapsed * 1000)  # Convert to ms
        return result

    # Monkey-patch temporarily
    dist.all_to_all_single = timed_alltoall

    # ... training step ...

    # Restore original
    dist.all_to_all_single = original_alltoall

    # Compute overhead ratio
    total_step_time = (time.time() - step_start_time) * 1000
    total_alltoall_time = sum(alltoall_times)
    overhead_ratio = total_alltoall_time / total_step_time

    return {
        "alltoall_times": alltoall_times,
        "total_alltoall_time": total_alltoall_time,
        "alltoall_overhead_ratio": overhead_ratio
    }
```

---

<a id="metric-definitions"></a>
## Metric Definitions（指标定义）

<a id="1-expert-load-imbalance"></a>
### 1. 专家负载不均衡

**expert_load_std：** 专家负载的变异系数。

```python
def compute_load_std(tokens_per_expert):
    """Compute normalized standard deviation of expert loads"""
    mean_load = np.mean(tokens_per_expert)
    std_load = np.std(tokens_per_expert)

    # Coefficient of variation (CV)
    cv = std_load / mean_load if mean_load > 0 else 0.0

    return cv

# Interpretation:
# CV < 0.15: Well balanced
# CV 0.15 - 0.30: Moderate imbalance
# CV > 0.30: Severe imbalance
```

**load_imbalance_factor：** 最大负载与平均负载之比。

```python
def compute_load_imbalance_factor(tokens_per_expert):
    """Compute imbalance factor"""
    max_load = np.max(tokens_per_expert)
    avg_load = np.mean(tokens_per_expert)

    factor = max_load / avg_load if avg_load > 0 else 1.0

    return factor

# Interpretation:
# 1.0 - 1.2: Excellent balance
# 1.2 - 1.5: Good balance
# 1.5 - 2.0: Moderate imbalance
# > 2.0: Severe imbalance
```

---

<a id="2-all-to-all-communication-overhead"></a>
### 2. All-to-All 通信开销

**alltoall_overhead_ratio：** All-to-All 耗时占训练步耗时的比例。

```python
def compute_alltoall_overhead(alltoall_times, total_step_time):
    """Compute All-to-All overhead as fraction of step time"""
    total_alltoall = sum(alltoall_times)
    ratio = total_alltoall / total_step_time

    return ratio

# Interpretation:
# < 0.15: Low overhead (compute-bound)
# 0.15 - 0.30: Moderate overhead
# 0.30 - 0.50: High overhead (communication-bound)
# > 0.50: Critical overhead
```

**alltoall_efficiency：** 有效带宽与理论带宽之比。

```python
def compute_alltoall_efficiency(message_size_bytes, time_ms, theoretical_bw_gbps):
    """Compute All-to-All efficiency"""
    # Actual bandwidth achieved
    actual_bw_gbps = (message_size_bytes / (1024**3)) / (time_ms / 1000)

    # Efficiency
    efficiency = actual_bw_gbps / theoretical_bw_gbps

    return efficiency

# Interpretation:
# > 0.7: Excellent
# 0.5 - 0.7: Good
# 0.3 - 0.5: Fair
# < 0.3: Poor (network congestion or overhead)
```

---

<a id="3-routing-efficiency"></a>
### 3. 路由效率

**routing_entropy：** 专家利用率分布的熵。

```python
def compute_routing_entropy(expert_probabilities):
    """Compute entropy of routing distribution"""
    # expert_probabilities: [num_experts]
    # Sum of routing probabilities across all tokens

    # Normalize to get distribution
    probs = expert_probabilities / np.sum(expert_probabilities)

    # Compute entropy: -Σ p_i log(p_i)
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy

    return normalized_entropy

# Interpretation:
# > 0.9: Excellent utilization (uniform)
# 0.7 - 0.9: Good utilization
# 0.5 - 0.7: Moderate skew
# < 0.5: Severe skew (many dead experts)
```

**expert_coverage：** 被激活专家的比例。

```python
def compute_expert_coverage(tokens_per_expert):
    """Compute fraction of experts that received tokens"""
    num_active = np.sum(tokens_per_expert > 0)
    num_total = len(tokens_per_expert)

    coverage = num_active / num_total

    return coverage

# Interpretation:
# 1.0: All experts active
# 0.8 - 1.0: Healthy coverage
# 0.5 - 0.8: Some dead experts
# < 0.5: Many dead experts (over-capacity model)
```

---

<a id="4-straggler-detection"></a>
### 4. 慢 rank 检测

**straggler_factor：** 最大 rank 耗时与平均 rank 耗时之比。

```python
def compute_straggler_factor(rank_times):
    """Compute straggler factor across ranks"""
    max_time = np.max(rank_times)
    avg_time = np.mean(rank_times)

    factor = max_time / avg_time

    return factor

# Interpretation:
# < 1.1: No stragglers
# 1.1 - 1.2: Mild stragglers
# 1.2 - 1.5: Moderate stragglers
# > 1.5: Severe stragglers
```

**rank_time_cv：** 各 rank 耗时的变异系数。

```python
def compute_rank_time_cv(rank_times):
    """Compute CV of rank execution times"""
    std_time = np.std(rank_times)
    mean_time = np.mean(rank_times)

    cv = std_time / mean_time if mean_time > 0 else 0.0

    return cv

# Interpretation:
# < 0.05: Excellent balance
# 0.05 - 0.10: Good balance
# 0.10 - 0.20: Moderate variance
# > 0.20: High variance (stragglers present)
```

---

<a id="5-dropped-tokens"></a>
### 5. 丢弃的 token

**dropped_token_ratio：** 因容量限制而丢弃的 token 比例。

```python
def compute_dropped_token_ratio(dropped_tokens, total_tokens):
    """Compute fraction of tokens dropped"""
    ratio = dropped_tokens / total_tokens if total_tokens > 0 else 0.0

    return ratio

# Interpretation:
# 0.0: No drops (ideal)
# < 0.01: Negligible drops
# 0.01 - 0.05: Moderate drops
# > 0.05: Excessive drops (increase capacity_factor)
```

---

<a id="aggregated-moe-metrics"></a>
## 汇总 MoE 指标

将逐步指标合并到验证摘要：

```python
def aggregate_moe_metrics(moe_metrics_per_step):
    """Aggregate MoE metrics across validation steps"""

    aggregated = {}

    # Load imbalance
    load_stds = [m["expert_load_std"] for m in moe_metrics_per_step]
    aggregated["mean_load_std"] = np.mean(load_stds)
    aggregated["max_load_std"] = np.max(load_stds)

    # Load imbalance factor
    all_tokens = [m["tokens_per_expert"] for m in moe_metrics_per_step]
    imbalance_factors = [
        compute_load_imbalance_factor(tokens) for tokens in all_tokens
    ]
    aggregated["mean_load_imbalance_factor"] = np.mean(imbalance_factors)
    aggregated["max_load_imbalance_factor"] = np.max(imbalance_factors)

    # Communication overhead
    alltoall_ratios = [m["alltoall_overhead_ratio"] for m in moe_metrics_per_step]
    aggregated["mean_alltoall_ratio"] = np.mean(alltoall_ratios)
    aggregated["max_alltoall_ratio"] = np.max(alltoall_ratios)

    # Routing entropy
    if "routing_entropy" in moe_metrics_per_step[0]:
        entropies = [m["routing_entropy"] for m in moe_metrics_per_step]
        aggregated["mean_routing_entropy"] = np.mean(entropies)

    # Expert coverage
    coverages = [
        compute_expert_coverage(m["tokens_per_expert"])
        for m in moe_metrics_per_step
    ]
    aggregated["mean_expert_coverage"] = np.mean(coverages)
    aggregated["min_expert_coverage"] = np.min(coverages)

    # Dropped tokens
    if "dropped_tokens" in moe_metrics_per_step[0]:
        total_dropped = sum(m["dropped_tokens"] for m in moe_metrics_per_step)
        total_tokens = sum(np.sum(m["tokens_per_expert"]) for m in moe_metrics_per_step)
        aggregated["dropped_token_ratio"] = compute_dropped_token_ratio(
            total_dropped, total_tokens
        )

    return aggregated
```

---

<a id="output-format-for-validation-report"></a>
## 验证报告输出格式

为标准验证报告增加 MoE 部分：

```json
{
  "validation_id": "val_20240120_143022",
  "candidate_config": {
    "ep_size": 8,
    "tp_size": 2,
    "dp_size": 4,
    "capacity_factor": 1.5
  },

  "standard_metrics": {
    "throughput": 115000,
    "memory_per_device_gb": 58.3,
    "mfu": 0.42
  },

  "moe_metrics": {
    "mean_load_imbalance_factor": 1.42,
    "max_load_imbalance_factor": 1.68,
    "mean_load_std": 0.24,
    "max_load_std": 0.31,

    "mean_alltoall_ratio": 0.22,
    "max_alltoall_ratio": 0.28,
    "alltoall_efficiency": 0.68,

    "mean_routing_entropy": 0.87,
    "mean_expert_coverage": 0.96,
    "min_expert_coverage": 0.91,

    "dropped_token_ratio": 0.003,

    "straggler_factor": 1.15,
    "rank_time_cv": 0.08
  },

  "interpretation": {
    "load_balance": "Good (factor 1.42)",
    "communication": "Moderate overhead (22%)",
    "routing_efficiency": "Excellent (entropy 0.87)",
    "expert_utilization": "Healthy (96% coverage)",
    "stragglers": "Acceptable (factor 1.15)"
  }
}
```

---

<a id="integration-with-strategy-validator"></a>
## 与 strategy-validator 集成

更新验证流程，采集 MoE 指标：

```python
# In validation script
if model_config.get("is_moe", False):
    # Setup MoE hooks
    moe_metrics = setup_moe_hooks(model)

    # Run validation steps
    for step in range(num_validation_steps):
        # ... training step ...

        # Measure communication
        comm_metrics = measure_alltoall_overhead(model, step_start_time)
        moe_metrics.update(comm_metrics)

    # Aggregate metrics
    aggregated_moe = aggregate_moe_metrics(moe_metrics)

    # Add to validation report
    report["moe_metrics"] = aggregated_moe
```

---

<a id="thresholds-for-passfail"></a>
## 通过与失败阈值

为 MoE 指标定义可接受范围：

| 指标 | 良好 | 可接受 | 较差 |
|--------|------|------------|------|
| 负载不均衡因子 | < 1.3 | 1.3 - 1.8 | > 1.8 |
| AllToAll 占比 | < 0.20 | 0.20 - 0.35 | > 0.35 |
| 路由熵 | > 0.85 | 0.70 - 0.85 | < 0.70 |
| 专家覆盖率 | > 0.95 | 0.85 - 0.95 | < 0.85 |
| token 丢弃比例 | < 0.01 | 0.01 - 0.05 | > 0.05 |
| 慢 rank 因子 | < 1.15 | 1.15 - 1.30 | > 1.30 |

---

<a id="example-diagnosis"></a>
## 诊断示例

<a id="scenario-a-high-communication-overhead"></a>
### 场景 A：通信开销高

```json
{
  "moe_metrics": {
    "mean_alltoall_ratio": 0.38,
    "alltoall_efficiency": 0.52
  },
  "diagnosis": "Communication-bound. Reduce ep_size or improve network."
}
```

<a id="scenario-b-load-imbalance"></a>
### 场景 B：负载不均衡

```json
{
  "moe_metrics": {
    "mean_load_imbalance_factor": 2.15,
    "mean_load_std": 0.42
  },
  "diagnosis": "Severe load imbalance. Increase capacity_factor or enable expert_bias."
}
```

<a id="scenario-c-dead-experts"></a>
### 场景 C：未激活的专家

```json
{
  "moe_metrics": {
    "mean_expert_coverage": 0.73,
    "mean_routing_entropy": 0.58
  },
  "diagnosis": "Many dead experts (27%). Model over-capacity or poor routing."
}
```

---

<a id="references"></a>
## 参考资料

- [moe-profiler/references/moe-metrics.md](../../moe-profiler/references/moe-metrics.md)：指标定义
- [parallel-strategy-analyzer/references/moe-strategy-rules.md](../../parallel-strategy-analyzer/references/moe-strategy-rules.md)：策略规则
