<a id="metrics-schema"></a>
# 指标规范

并行策略验证的标准化指标定义和提取规则。

<a id="metric-taxonomy"></a>
## 指标分类

<a id="p0-core-performance-metrics"></a>
### P0：核心性能指标

所有验证运行都必须提供，可直接与 PSA 估算比较。

<a id="1-throughput"></a>
#### 1. 吞吐

**定义：** 每秒处理的 token 数或样本数，用于衡量训练速度。

```python
{
  "throughput": {
    "tokens_per_sec": float,      # Primary metric: tokens/s
    "samples_per_sec": float,      # Samples/s (= tokens_per_sec / seq_length)
    "mean": float,                 # Average over steps (excluding warmup)
    "std": float,                  # Standard deviation
    "min": float,                  # Minimum observed
    "max": float,                  # Maximum observed
    "num_samples": int,            # Number of steps measured
    "unit": "tokens/s"
  }
}
```

**提取规则：**
- 从日志解析：`tokens/s=(\d+\.?\d*)([KM]?)`
- 排除最前面的 N 个预热步，默认 10 步
- 在剩余步骤上计算统计值
- 转换 K/M 后缀：K=1e3，M=1e6

**容差：** 相对 PSA 估算为 ±15%。

**日志行示例：**
```
[Step 11/50] loss=2.298, throughput=28.5 samples/s, tokens/s=116.7K, step_time=1.23s
```

<a id="2-memory"></a>
#### 2. 显存

**定义：** 每个 GPU/NPU 设备已分配显存的峰值。

```python
{
  "memory": {
    "peak_allocated_gb": float,    # Max memory allocated (primary)
    "peak_reserved_gb": float,     # Max memory reserved by allocator
    "mean_allocated_gb": float,    # Average across ranks
    "std_gb": float,               # Cross-rank standard deviation
    "per_rank": {                  # Per-rank breakdown
      "0": float,
      "1": float,
      ...
    },
    "utilization_percent": float,  # peak / device_capacity × 100
    "headroom_gb": float,          # device_capacity - peak
    "unit": "GB"
  }
}
```

**提取规则：**
- 从日志解析：`max_allocated=(\d+\.?\d*)GB`
- 对每个 rank，取所有已记录步骤中的最大值
- 对各 rank 求平均；同构工作负载的预期差异小于 5%

**容差：** 相对 PSA 估算为 ±20%。

**日志行示例：**
```
[Memory] Rank 0: allocated=58.2GB, reserved=61.4GB, max_allocated=61.1GB
```

<a id="3-model-flops-utilization-mfu"></a>
#### 3. 模型 FLOPs 利用率（MFU）

**定义：** 实际计算 FLOPs 与硬件理论峰值 FLOPs 的比值。

```python
{
  "mfu": {
    "percent": float,              # Primary metric: MFU %
    "actual_tflops": float,        # Actual achieved TFLOPS
    "peak_tflops": float,          # Hardware peak (from known_hardware.md)
    "flops_per_step": float,       # Computed FLOPs per training step
    "step_time_sec": float,        # Mean step time used for calculation
    "unit": "percent"
  }
}
```

**计算：** 优先使用框架内置的 `training.mfu` 遥测，它按 PaLM 风格的模型 FLOP 口径发布 `performance/mfu_percent`，参见 [MFU 核算与目标](../../perf-playbook/references/mfu-targets.md)。

离线重算必须采用相同口径。`6 × N_active` 已包含前向和反向工作；再次乘二会错误地将 MFU 翻倍。保留所支持的注意力项 `12 × L × H × d_head × S`，不要将其省略。

**预测不确定性：** PSA 估算不是验收阈值。应针对模型家族和运行区间校准预测误差，并在报告实测结果时声明核算口径。

**比较：** 使用相对实测基线的目标，或有文档依据的 roofline 分析。不要对某个模型家族强加通用 MFU 范围。使用矩阵优化器的稀疏 MoE 与稠密训练，其有效计算占比可能明显不同。

<a id="4-step-time"></a>
#### 4. 步耗时

**定义：** 每个训练步的墙钟耗时，包含前向、反向和优化器更新。

```python
{
  "step_time": {
    "mean_ms": float,              # Average step time (primary)
    "std_ms": float,               # Standard deviation
    "min_ms": float,
    "max_ms": float,
    "median_ms": float,
    "p95_ms": float,               # 95th percentile
    "unit": "milliseconds"
  }
}
```

**提取规则：**
- 从日志解析：`step_time=(\d+\.?\d*)s`
- 转换为毫秒
- 排除预热步
- 标准差超过均值的 10% 时，标记为不稳定

**容差：** 相对 PSA 估算为 ±15%。

<a id="p1-communication--pipeline-metrics"></a>
### P1：通信与流水线指标

并行维度启用时必须提供，即 TP/CP/EP/PP > 1。

<a id="5-communication-breakdown"></a>
#### 5. 通信分解

**定义：** 从性能追踪中提取的集合通信操作耗时。

```python
{
  "communication": {
    "total_ms": float,             # Total comm time per step
    "percent_of_step": float,      # comm_time / step_time × 100
    "breakdown": {
      "AllReduce": {
        "count": int,              # Number of ops per step
        "total_ms": float,
        "mean_ms": float
      },
      "AllGather": {...},
      "ReduceScatter": {...},
      "Send": {...},
      "Recv": {...},
      "AllToAll": {...}            # EP only
    },
    "by_dimension": {              # Attribution
      "TP": float,                 # TP communication time
      "CP": float,
      "EP": float,
      "DP_FSDP": float
    },
    "overlap_estimate": float,     # % of comm hidden by compute
    "exposed_ms": float,           # Exposed (non-overlapped) comm time
    "unit": "milliseconds"
  }
}
```

**提取规则：**
- 解析 Torch profiler JSON，`*.pt.trace.json`
- 根据名称筛选事件，与集合通信名称进行不区分大小写的匹配
- 按操作类型汇总耗时
- 根据操作模式归属到并行维度：
  - 每层 4 次 AllReduce → TP
  - AllGather + ReduceScatter → FSDP（DP）
  - Send/Recv → PP
  - AllToAll → EP

**容差：** 相对 PSA 估算为 ±30%；通信对网络波动高度敏感。

<a id="6-pipeline-bubble"></a>
#### 6. 流水线气泡

**定义：** 流水线并行中的空闲时间比例，仅适用于 PP > 1。

```python
{
  "bubble": {
    "ratio": float,                # Idle / total time (primary)
    "percent": float,              # ratio × 100
    "idle_time_ms": float,
    "compute_time_ms": float,
    "schedule": str,               # "1f1b" or "interleaved_1f1b"
    "num_microbatches": int,
    "unit": "ratio"
  }
}
```

**提取规则：**
- 从追踪中将事件分类为计算，即前向/反向内核，以及空闲，即等待/接收
- `bubble_ratio = idle_time / (idle_time + compute_time)`
- profiler 不可用时，根据理论公式估算：
  ```python
  bubble_1f1b = (pp - 1) / (num_microbatches + pp - 1)
  ```

**容差：** 相对 PSA 估算为 ±5 个百分点。

**评估：**
- ≤5%：优秀
- 5～10%：良好
- 10～20%：可接受，建议交错调度
- >20%：较差，应增加微批数或减小 PP

<a id="p2-diagnostic-metrics"></a>
### P2：诊断指标

可选，用于深入分析或调试。

<a id="7-loss-trajectory"></a>
#### 7. Loss 轨迹

```python
{
  "loss": {
    "final": float,
    "initial": float,
    "mean": float,
    "trend": str,                  # "decreasing" | "stable" | "diverging"
    "nan_or_inf": bool             # True if any NaN/Inf detected
  }
}
```

**使用场景：** 检测数值不稳定，NaN/Inf 表示配置存在问题。

<a id="8-gradient-norm"></a>
#### 8. 梯度范数

```python
{
  "grad_norm": {
    "mean": float,
    "max": float,
    "clipped_steps": int,          # Steps where grad norm exceeded max_grad_norm
    "overflow_steps": int          # FP16/BF16 overflow events
  }
}
```

**使用场景：** 诊断混合精度问题或梯度爆炸。

<a id="9-device-utilization-from-profiler"></a>
#### 9. 设备利用率（来自 profiler）

```python
{
  "utilization": {
    "compute_percent": float,      # % time in CUDA/NPU kernels
    "memory_bandwidth_percent": float,
    "kernel_efficiency": float     # Achieved / theoretical kernel FLOPs
  }
}
```

**使用场景：** 判断工作负载受计算还是内存限制。

---

<a id="extraction-pipeline"></a>
## 提取流程

<a id="step-1-parse-training-log"></a>
### 步骤 1：解析训练日志

```python
import re
from typing import Dict, List

def parse_training_log(log_path: str, warmup_steps: int = 10) -> Dict:
    """Extract metrics from training log."""

    metrics = {
        "throughput": {"samples": [], "tokens": []},
        "step_time": [],
        "loss": [],
        "memory": {}
    }

    patterns = {
        "throughput": re.compile(
            r'throughput=([\d.]+) samples/s, tokens/s=([\d.]+)([KM]?)'
        ),
        "step_time": re.compile(r'step_time=([\d.]+)s'),
        "loss": re.compile(r'loss=([\d.]+|nan|inf)'),
        "memory": re.compile(
            r'Rank (\d+):.*max_allocated=([\d.]+)GB'
        )
    }

    with open(log_path) as f:
        for line in f:
            # Extract step number
            step_match = re.search(r'Step (\d+)/', line)
            if step_match:
                step = int(step_match.group(1))
                if step <= warmup_steps:
                    continue  # Skip warmup

            # Throughput
            match = patterns["throughput"].search(line)
            if match:
                samples_s, tokens, unit = match.groups()
                tokens_val = float(tokens)
                if unit == 'K':
                    tokens_val *= 1000
                elif unit == 'M':
                    tokens_val *= 1_000_000

                metrics["throughput"]["samples"].append(float(samples_s))
                metrics["throughput"]["tokens"].append(tokens_val)

            # Step time
            match = patterns["step_time"].search(line)
            if match:
                metrics["step_time"].append(float(match.group(1)) * 1000)  # to ms

            # Loss
            match = patterns["loss"].search(line)
            if match:
                loss_str = match.group(1)
                if loss_str not in ["nan", "inf"]:
                    metrics["loss"].append(float(loss_str))
                else:
                    metrics["loss"].append(loss_str)

            # Memory
            match = patterns["memory"].search(line)
            if match:
                rank, mem_gb = match.groups()
                rank = int(rank)
                mem_gb = float(mem_gb)

                if rank not in metrics["memory"]:
                    metrics["memory"][rank] = []
                metrics["memory"][rank].append(mem_gb)

    return metrics
```

<a id="step-2-compute-statistics"></a>
### 步骤 2：计算统计值

```python
import numpy as np

def compute_statistics(metrics: Dict) -> Dict:
    """Compute aggregated statistics."""

    result = {}

    # Throughput
    tokens_per_sec = metrics["throughput"]["tokens"]
    samples_per_sec = metrics["throughput"]["samples"]

    result["throughput"] = {
        "tokens_per_sec": float(np.mean(tokens_per_sec)),
        "samples_per_sec": float(np.mean(samples_per_sec)),
        "std_tokens_per_sec": float(np.std(tokens_per_sec)),
        "min_tokens_per_sec": float(np.min(tokens_per_sec)),
        "max_tokens_per_sec": float(np.max(tokens_per_sec)),
        "num_samples": len(tokens_per_sec)
    }

    # Step time
    step_times = metrics["step_time"]
    result["step_time"] = {
        "mean_ms": float(np.mean(step_times)),
        "std_ms": float(np.std(step_times)),
        "min_ms": float(np.min(step_times)),
        "max_ms": float(np.max(step_times)),
        "median_ms": float(np.median(step_times)),
        "p95_ms": float(np.percentile(step_times, 95))
    }

    # Memory (across ranks)
    memory_per_rank = {
        rank: max(mem_list) for rank, mem_list in metrics["memory"].items()
    }
    peak_memory_values = list(memory_per_rank.values())

    result["memory"] = {
        "peak_allocated_gb": float(np.mean(peak_memory_values)),
        "std_gb": float(np.std(peak_memory_values)),
        "per_rank": memory_per_rank
    }

    # Loss
    losses = [l for l in metrics["loss"] if isinstance(l, float)]
    result["loss"] = {
        "final": float(losses[-1]) if losses else None,
        "initial": float(losses[0]) if losses else None,
        "mean": float(np.mean(losses)) if losses else None,
        "nan_or_inf": any(isinstance(l, str) for l in metrics["loss"])
    }

    return result
```

<a id="step-3-parse-profiler-trace"></a>
### 步骤 3：解析性能追踪

```python
import json

def parse_profiler_trace(trace_path: str) -> Dict:
    """Extract communication metrics from profiler JSON."""

    with open(trace_path) as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])

    comm_ops = {
        "AllReduce": [],
        "AllGather": [],
        "ReduceScatter": [],
        "Send": [],
        "Recv": [],
        "AllToAll": []
    }

    for event in events:
        name = event.get("name", "").lower()
        dur_us = event.get("dur", 0)

        for op_name, op_list in comm_ops.items():
            if op_name.lower() in name:
                op_list.append(dur_us / 1000)  # Convert to ms
                break

    # Aggregate
    breakdown = {}
    total_ms = 0

    for op_name, durations in comm_ops.items():
        if durations:
            breakdown[op_name] = {
                "count": len(durations),
                "total_ms": sum(durations),
                "mean_ms": sum(durations) / len(durations)
            }
            total_ms += sum(durations)

    return {
        "total_ms": total_ms,
        "breakdown": breakdown
    }
```

<a id="step-4-consolidate-output"></a>
### 步骤 4：汇总输出

```python
def generate_metrics_json(log_path, profiler_path, model_config, hardware_config):
    """Generate complete metrics JSON."""

    # Parse raw data
    raw_metrics = parse_training_log(log_path)
    stats = compute_statistics(raw_metrics)

    if profiler_path:
        comm_metrics = parse_profiler_trace(profiler_path)
        stats["communication"] = comm_metrics

    # Compute MFU
    mfu = compute_mfu(
        model_params=model_config["params"],
        seq_length=model_config["seq_length"],
        global_batch_size=model_config["batch_size"],
        step_time_ms=stats["step_time"]["mean_ms"],
        peak_tflops=hardware_config["peak_tflops"]
    )
    stats["mfu"] = mfu

    # Add metadata
    output = {
        "metrics": stats,
        "model": model_config,
        "hardware": hardware_config,
        "run_metadata": {
            "log_file": log_path,
            "profiler_trace": profiler_path,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    return output
```

---

<a id="validation-rules"></a>
## 验证规则

<a id="metric-completeness-check"></a>
### 指标完整性检查

```python
def validate_metrics_completeness(metrics: Dict) -> Dict:
    """Check if all P0 metrics are present."""

    required_p0 = ["throughput", "memory", "mfu", "step_time"]
    missing = [m for m in required_p0 if m not in metrics]

    warnings = []

    if missing:
        return {
            "status": "INCOMPLETE",
            "missing_metrics": missing
        }

    # Check data quality
    if metrics["throughput"]["num_samples"] < 10:
        warnings.append("Low sample count for throughput (<10 steps)")

    if metrics["step_time"]["std_ms"] / metrics["step_time"]["mean_ms"] > 0.2:
        warnings.append("High step time variance (>20%), unstable run")

    if metrics["loss"].get("nan_or_inf"):
        warnings.append("NaN/Inf detected in loss, numerical instability")

    return {
        "status": "COMPLETE" if not warnings else "COMPLETE_WITH_WARNINGS",
        "warnings": warnings
    }
```

<a id="tolerance-assessment"></a>
### 容差评估

```python
def assess_tolerance(actual: float, estimate: float, tolerance_pct: float) -> Dict:
    """Check if metric is within tolerance."""

    delta_pct = (actual - estimate) / estimate * 100
    within_tolerance = abs(delta_pct) <= tolerance_pct

    return {
        "actual": actual,
        "estimate": estimate,
        "delta_percent": round(delta_pct, 1),
        "within_tolerance": within_tolerance,
        "assessment": "✓ PASS" if within_tolerance else "⚠ DEVIATION"
    }
```

---

<a id="output-format"></a>
## 输出格式

最终的 `validation_metrics.json`：

```json
{
  "metrics": {
    "throughput": {...},
    "memory": {...},
    "mfu": {...},
    "step_time": {...},
    "communication": {...},
    "bubble": {...}
  },
  "model": {...},
  "hardware": {...},
  "run_metadata": {...},
  "validation": {
    "status": "COMPLETE",
    "warnings": []
  }
}
```


---

<a id="moe-specific-metrics-moe-models-only"></a>
## MoE-Specific Metrics（MoE 专用指标，仅限 MoE 模型）

**`model.is_moe == True` 时加载。**

详细定义、公式和采集方法参见 [moe-metrics.md](moe-metrics.md)。

<a id="expert-load-metrics"></a>
### 专家负载指标

```json
{
  "moe_metrics": {
    "expert_load": {
      "mean_load_imbalance_factor": 1.42,
      "max_load_imbalance_factor": 1.68,
      "mean_load_std": 0.24,
      "interpretation": "Good (factor 1.42)"
    }
  }
}
```

<a id="communication-metrics"></a>
### 通信指标

```json
{
  "moe_metrics": {
    "communication": {
      "mean_alltoall_ratio": 0.22,
      "max_alltoall_ratio": 0.28,
      "alltoall_efficiency": 0.68,
      "interpretation": "Moderate overhead (22%)"
    }
  }
}
```

<a id="routing-efficiency"></a>
### 路由效率

```json
{
  "moe_metrics": {
    "routing": {
      "mean_routing_entropy": 0.87,
      "mean_expert_coverage": 0.96,
      "dropped_token_ratio": 0.003,
      "interpretation": "Excellent"
    }
  }
}
```

**完整规范：** [moe-metrics.md](moe-metrics.md)
