<a id="phase-3-metrics-extraction"></a>
# 阶段 3：指标提取

解析训练日志和性能追踪，提取标准化性能指标。

<a id="31-primary-metrics"></a>
## 3.1 主要指标

提取与 PSA 估算对应的目标指标：

| 指标 | 单位 | 来源 | 优先级 |
|--------|------|--------|----------|
| **吞吐** | tokens/s、samples/s | 训练日志 | P0 |
| **显存峰值** | 每设备 GB | 训练日志和 profiler | P0 |
| **MFU** | % | 根据 FLOPs 和时间计算 | P0 |
| **气泡比例** | %，仅限 PP | 性能追踪 | P1 |
| **步耗时** | ms | 训练日志 | P1 |
| **通信耗时** | ms | 性能追踪 | P2 |

<a id="32-log-parsing"></a>
## 3.2 日志解析

<a id="321-extract-throughput"></a>
### 3.2.1 提取吞吐

目标日志行，取预热后的第 11～50 步：

```
[Step 11/50] loss=2.298, throughput=28.5 samples/s, tokens/s=116K, step_time=1.23s
[Step 12/50] loss=2.287, throughput=28.7 samples/s, tokens/s=117K, step_time=1.21s
...
[Step 50/50] loss=2.103, throughput=29.2 samples/s, tokens/s=119K, step_time=1.19s
```

**提取脚本：**

```python
import re
import numpy as np

LOG_FILE = "validation_run_20261016_143052.log"
WARMUP_STEPS = 10

def parse_throughput(log_path, warmup=10):
    pattern = re.compile(
        r'\[Step (\d+)/\d+\] '
        r'.*throughput=([\d.]+) samples/s, '
        r'tokens/s=([\d.]+)([KM]?)'
    )

    samples_per_sec = []
    tokens_per_sec = []

    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            step, samples_s, tokens, unit = match.groups()
            step = int(step)

            if step <= warmup:
                continue  # Skip warmup

            samples_per_sec.append(float(samples_s))

            # Parse tokens/s with K/M suffix
            tokens_val = float(tokens)
            if unit == 'K':
                tokens_val *= 1000
            elif unit == 'M':
                tokens_val *= 1_000_000
            tokens_per_sec.append(tokens_val)

    return {
        "mean_samples_per_sec": np.mean(samples_per_sec),
        "std_samples_per_sec": np.std(samples_per_sec),
        "mean_tokens_per_sec": np.mean(tokens_per_sec),
        "std_tokens_per_sec": np.std(tokens_per_sec),
        "num_samples": len(samples_per_sec)
    }

# Result:
# {
#   "mean_samples_per_sec": 28.9,
#   "std_samples_per_sec": 0.4,
#   "mean_tokens_per_sec": 118234,
#   "std_tokens_per_sec": 1632,
#   "num_samples": 40
# }
```

<a id="322-extract-memory"></a>
### 3.2.2 提取显存

目标日志行，每 N 步或在峰值时记录：

```
[Memory] Rank 0: allocated=58.2GB, reserved=61.4GB, max_allocated=61.1GB
[Memory] Rank 1: allocated=58.3GB, reserved=61.5GB, max_allocated=61.2GB
```

**提取：**

```python
def parse_memory(log_path):
    pattern = re.compile(
        r'\[Memory\] Rank (\d+): '
        r'allocated=([\d.]+)GB, '
        r'reserved=([\d.]+)GB, '
        r'max_allocated=([\d.]+)GB'
    )

    max_allocated_per_rank = {}

    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            rank, allocated, reserved, max_alloc = match.groups()
            rank = int(rank)
            max_alloc = float(max_alloc)

            # Track peak across all logs for this rank
            if rank not in max_allocated_per_rank:
                max_allocated_per_rank[rank] = max_alloc
            else:
                max_allocated_per_rank[rank] = max(
                    max_allocated_per_rank[rank], max_alloc
                )

    # Average across ranks (should be similar for data parallel)
    peak_memory_gb = np.mean(list(max_allocated_per_rank.values()))
    std_memory_gb = np.std(list(max_allocated_per_rank.values()))

    return {
        "peak_memory_gb": peak_memory_gb,
        "std_memory_gb": std_memory_gb,
        "per_rank": max_allocated_per_rank
    }

# Result:
# {
#   "peak_memory_gb": 61.3,
#   "std_memory_gb": 0.2,
#   "per_rank": {0: 61.1, 1: 61.2, ...}
# }
```

<a id="323-extract-step-time"></a>
### 3.2.3 提取步耗时

```python
def parse_step_time(log_path, warmup=10):
    pattern = re.compile(r'\[Step (\d+)/\d+\].*step_time=([\d.]+)s')

    step_times_ms = []

    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            step, time_s = match.groups()
            step = int(step)

            if step <= warmup:
                continue

            step_times_ms.append(float(time_s) * 1000)  # Convert to ms

    return {
        "mean_step_time_ms": np.mean(step_times_ms),
        "std_step_time_ms": np.std(step_times_ms)
    }

# Result: {"mean_step_time_ms": 1203.5, "std_step_time_ms": 12.4}
```

<a id="33-profiler-trace-analysis"></a>
## 3.3 性能追踪分析

使用 Torch profiler 的 JSON 输出 `step_15_rank_0.pt.trace.json`。

<a id="331-communication-time-breakdown"></a>
### 3.3.1 通信耗时分解

提取 AllReduce、AllGather、ReduceScatter 等集合通信操作：

```python
import json

def parse_profiler_trace(trace_path):
    with open(trace_path) as f:
        trace = json.load(f)

    # Trace format: {"traceEvents": [...]}
    events = trace.get("traceEvents", [])

    comm_ops = {
        "AllReduce": [],
        "AllGather": [],
        "ReduceScatter": [],
        "Send": [],
        "Recv": []
    }

    for event in events:
        name = event.get("name", "")
        dur = event.get("dur", 0)  # Duration in microseconds

        for op_name in comm_ops.keys():
            if op_name.lower() in name.lower():
                comm_ops[op_name].append(dur / 1000)  # Convert to ms
                break

    # Aggregate
    total_comm_time_ms = sum(sum(times) for times in comm_ops.values())

    breakdown = {
        op: {"count": len(times), "total_ms": sum(times)}
        for op, times in comm_ops.items()
        if times
    }

    return {
        "total_comm_time_ms": total_comm_time_ms,
        "breakdown": breakdown
    }

# Result:
# {
#   "total_comm_time_ms": 142.3,
#   "breakdown": {
#     "AllReduce": {"count": 96, "total_ms": 85.2},
#     "AllGather": {"count": 48, "total_ms": 32.1},
#     "ReduceScatter": {"count": 48, "total_ms": 25.0}
#   }
# }
```

<a id="332-bubble-ratio-pipeline-parallel"></a>
### 3.3.2 气泡比例（流水线并行）

`pp > 1` 时，计算气泡时间：

```python
def compute_bubble_ratio(trace_path, pp_degree):
    if pp_degree == 1:
        return {"bubble_ratio": 0.0, "note": "No PP"}

    with open(trace_path) as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])

    # Find compute vs idle time on each PP rank
    compute_time = 0
    idle_time = 0

    for event in events:
        cat = event.get("cat", "")
        dur = event.get("dur", 0) / 1000  # ms

        if "forward" in cat.lower() or "backward" in cat.lower():
            compute_time += dur
        elif "wait" in cat.lower() or "idle" in cat.lower():
            idle_time += dur

    total_time = compute_time + idle_time
    bubble_ratio = idle_time / total_time if total_time > 0 else 0

    return {
        "bubble_ratio": bubble_ratio,
        "idle_time_ms": idle_time,
        "compute_time_ms": compute_time
    }

# Result: {"bubble_ratio": 0.061, "idle_time_ms": 73.5, "compute_time_ms": 1130.0}
```

<a id="34-mfu-from-framework-telemetry"></a>
## 3.4 从框架遥测获取 MFU

优先使用框架实测输出。通过 `enabled` 和 `peak_tflops_per_device` 启用 `training.mfu`。框架按照 PaLM 风格的模型 FLOP 口径，发布 `performance/mfu_percent`、`performance/model_tflops_per_device` 和 `performance/tokens_per_s_per_device`。MFU、实际执行工作量 HFU 和目标定义，参见 [MFU 核算与目标](../../perf-playbook/references/mfu-targets.md)。

遥测不可用时，使用相同口径离线重算。标准稠密注意力近似如下：

```python
def model_flops_per_token(active_matmul_params, layers, attention_width, seq_len):
    """Include forward/backward work; attention_width is num_heads * head_dim."""
    return 6 * active_matmul_params + 12 * layers * attention_width * seq_len
```

- `6 × N_active` 已包含约 `2N` 的前向工作和约 `4N` 的反向工作，不要再次乘二。
- 保留依赖序列长度的注意力项；长序列下该项可能占比较大。特殊注意力架构需要使用对应的模型 FLOP 实现。
- 从 `N_active` 中排除嵌入查表。对支持的均匀专家架构，按 `top_k / num_experts` 比例计入路由专家矩阵参数。

报告 MFU 时始终说明核算口径和设备峰值。同一次运行在不同口径下，数值可能有明显差异。

<a id="35-consolidated-output"></a>
## 3.5 汇总输出

将全部指标合并到 `validation_metrics.json`：

```json
{
  "strategy": {
    "dp": 4, "tp": 8, "pp": 2, "cp": 1, "ep": 1,
    "fsdp_level": "level2"
  },
  "model": {
    "name": "Qwen3-30B-A3B",
    "params": 30e9,
    "layers": 24,
    "seq_length": 4096
  },
  "hardware": {
    "device_type": "Ascend 910B",
    "device_count": 64,
    "peak_tflops": 320
  },
  "metrics": {
    "throughput": {
      "samples_per_sec": 28.9,
      "tokens_per_sec": 118234,
      "std_tokens_per_sec": 1632
    },
    "memory": {
      "peak_gb": 61.3,
      "std_gb": 0.2
    },
    "mfu": {
      "percent": 42.1,
      "actual_tflops": 134.7
    },
    "step_time": {
      "mean_ms": 1203.5,
      "std_ms": 12.4
    },
    "communication": {
      "total_ms": 142.3,
      "breakdown": {
        "AllReduce": {"count": 96, "total_ms": 85.2},
        "AllGather": {"count": 48, "total_ms": 32.1},
        "ReduceScatter": {"count": 48, "total_ms": 25.0}
      }
    },
    "bubble": {
      "ratio": 0.061,
      "idle_ms": 73.5
    }
  },
  "run_metadata": {
    "total_steps": 50,
    "warmup_steps": 10,
    "runtime_sec": 512,
    "timestamp": "2026-10-16T14:30:52Z",
    "log_file": "validation_run_20261016_143052.log"
  }
}
```

<a id="36-error-handling"></a>
## 3.6 错误处理

**缺失指标：**

```python
def validate_metrics(metrics):
    required = ["throughput", "memory", "mfu", "step_time"]
    missing = [k for k in required if k not in metrics]

    if missing:
        return {
            "status": "INCOMPLETE",
            "missing_metrics": missing,
            "note": "Check log format or profiler output"
        }

    return {"status": "COMPLETE"}
```

**异常值检测：**

```python
# Flag if std_dev > 20% of mean (unstable run)
if metrics["throughput"]["std_tokens_per_sec"] / metrics["throughput"]["tokens_per_sec"] > 0.2:
    warnings.append("High throughput variance (>20%), consider re-running")
```

<a id="37-output-for-phase-4"></a>
## 3.7 交给阶段 4 的输出

**成功：**
```
✓ Metrics extracted from validation_run_20261016_143052.log
✓ Profiler traces analyzed (2 files)
✓ Output: validation_metrics.json
✓ All P0 metrics available
✓ No stability warnings

Ready to execute Phase 4: Comparison Report
```

**带警告：**
```
⚠ Metrics extracted with warnings:
  - Communication breakdown incomplete (no profiler trace for rank 0)
  - High memory variance across ranks (std=2.1GB, investigate load imbalance)
✓ P0 metrics available, proceeding to Phase 4 with caveats
```
