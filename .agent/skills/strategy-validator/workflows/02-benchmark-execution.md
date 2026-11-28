<a id="phase-2-benchmark-execution"></a>
# 阶段 2：基准测试执行

启动分布式训练作业，监测其完成情况，以及 OOM、挂起或其他错误。

<a id="21-device-availability-check"></a>
## 2.1 设备可用性检查

启动前检查硬件状态：

```bash
# For Ascend NPU
npu-smi info | grep "Health Status" | wc -l
# Expected: {{ device_count }} healthy devices

# For NVIDIA GPU
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | wc -l
# Expected: {{ device_count }} devices

# Check for existing processes
pgrep -af "train_lm.py|torchrun" && echo "WARNING: Training processes detected"
```

**以下情况中止：**
- 可用设备数小于所需的 `device_count`
- 检测到已有训练进程，存在资源冲突
- 设备健康检查失败

<a id="22-launch-distributed-job"></a>
## 2.2 启动分布式作业

执行阶段 1 生成的脚本：

```bash
# Set monitoring environment
export VALIDATION_START_TIME=$(date +%s)
export VALIDATION_LOG="validation_run_$(date +%Y%m%d_%H%M%S).log"

# Launch with timeout protection (15 min max for 50 steps)
timeout 900 bash scripts/run_validation_TIMESTAMP.sh 2>&1 | tee "$VALIDATION_LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo "Exit code: $EXIT_CODE" >> "$VALIDATION_LOG"
```

<a id="221-monitoring-signals"></a>
### 2.2.1 监测信号

运行时关注：

**进度指示，预期每 5～10 秒出现一次：**
```
[Step 10/50] loss=2.345, throughput=28.3 samples/s, tokens/s=115K
[Step 11/50] loss=2.298, throughput=28.5 samples/s, tokens/s=116K
```

**前 5 步的显存分配：**
```
[Memory] Rank 0: allocated=58.2GB, reserved=61.4GB, max_allocated=61.1GB
```

**性能采集记录点，第 15、45 步：**
```
[Profiler] Step 15 trace saved to: ./profiler_traces/step_15_rank_*.pt.trace.json
```

<a id="222-failure-modes"></a>
### 2.2.2 失败模式

**OOM 检测：**
```bash
grep -E "CUDA out of memory|NPU out of memory|torch.OutOfMemoryError" "$VALIDATION_LOG"
# If found → immediate abort, report OOM at step X
```

**挂起检测：**
```bash
# No progress for >60s
tail -F "$VALIDATION_LOG" | while read line; do
  if [[ $line =~ "Step" ]]; then
    LAST_PROGRESS=$(date +%s)
  fi
  NOW=$(date +%s)
  if (( NOW - LAST_PROGRESS > 60 )); then
    echo "HANG detected: no progress for 60s"
    pkill -9 -f train_lm.py
    exit 1
  fi
done
```

**Loss 发散：**
```bash
grep -E "loss=nan|loss=inf" "$VALIDATION_LOG"
# If found → gradient instability, report with LR/batch size
```

**通信超时：**
```bash
grep -E "HCCL timeout|NCCL timeout|ProcessGroupNCCL" "$VALIDATION_LOG"
# Common in CP/TP with misconfigured mesh
```

<a id="23-live-metrics-streaming-optional"></a>
## 2.3 实时指标流（可选）

需要实时监测时，解析日志并输出结构化事件：

```python
# monitor_validation.py
import re
import sys

LOG_PATTERN = re.compile(
    r'\[Step (\d+)/(\d+)\] loss=([\d.]+), '
    r'throughput=([\d.]+) samples/s, tokens/s=([\d.]+[KM]?)'
)

for line in sys.stdin:
    match = LOG_PATTERN.search(line)
    if match:
        step, total, loss, samples_s, tokens_s = match.groups()
        print(json.dumps({
            "step": int(step),
            "total_steps": int(total),
            "loss": float(loss),
            "samples_per_sec": float(samples_s),
            "tokens_per_sec": parse_throughput(tokens_s),
            "timestamp": time.time()
        }))
```

与训练同时运行：
```bash
tail -F "$VALIDATION_LOG" | python monitor_validation.py > live_metrics.jsonl
```

<a id="24-completion-criteria"></a>
## 2.4 完成标准

<a id="success"></a>
### 成功
```
✓ All 50 steps completed
✓ Exit code 0
✓ Final checkpoint saved
✓ Profiler traces written
✓ No OOM/hang/timeout errors
```

<a id="partial-success-recoverable"></a>
### 部分成功（可恢复）
```
⚠ Completed 50 steps but with warnings:
  - Gradient overflow (AMP): expected with mixed precision
  - Minor HCCL retries: transient network issue
→ Proceed to metrics extraction with caveat
```

<a id="failure-non-recoverable"></a>
### 失败（不可恢复）
```
✗ OOM at step 12 → Strategy rejected (memory estimate too optimistic)
✗ Hang at step 8 → Communication deadlock (likely CP/PP boundary issue)
✗ Loss NaN at step 3 → Gradient instability (LR too high or FP16 underflow)
✗ Timeout (>15 min) → Throughput far below estimate, abort
```

<a id="25-cleanup"></a>
## 2.5 清理

运行结束后，无论成功或失败都执行：

```bash
# Kill any lingering processes
pkill -9 -f "train_lm.py|torchrun"

# Archive logs
mkdir -p validation_logs/$(date +%Y%m%d)
mv validation_run_*.log validation_logs/$(date +%Y%m%d)/

# Clear HCCL/NCCL cache (if needed for next run)
rm -rf /tmp/hccl_* /tmp/nccl_*

# Device reset (Ascend only, if OOM)
npu-smi reset -d 0-63
```

<a id="26-output-for-phase-3"></a>
## 2.6 交给阶段 3 的输出

**成功产物：**
```
validation_run_20261016_143052.log          # Full training log
live_metrics.jsonl                          # Timestamped step metrics
profiler_traces/step_15_rank_0.pt.trace.json  # Torch profiler data
profiler_traces/step_45_rank_0.pt.trace.json
checkpoint_step_50/                         # Final model state (optional)
```

**失败报告，适用时生成：**
```json
{
  "status": "FAILED",
  "failure_mode": "OOM",
  "failed_at_step": 12,
  "error_message": "torch.cuda.OutOfMemoryError: Tried to allocate 8.00 GiB...",
  "peak_memory_gb": 68.4,
  "estimated_memory_gb": 58.0,
  "delta_percent": 17.9
}
```

**向用户展示的摘要：**

```
✓ Benchmark execution completed successfully
✓ Runtime: 8m 32s (50 steps)
✓ No OOM/hang/timeout errors
✓ Logs: validation_run_20261016_143052.log
✓ Profiler traces: 2 files generated

Ready to execute Phase 3: Metrics Extraction
```

<a id="27-multi-candidate-execution"></a>
## 2.7 多候选执行

验证前三个候选时，按顺序运行，并在每次运行后清理资源：

```bash
for candidate in candidate1 candidate2 candidate3; do
  echo "=== Running $candidate ==="

  # Pre-flight check
  ./check_devices.sh || { echo "Devices not ready"; exit 1; }

  # Execute
  timeout 900 bash scripts/run_validation_${candidate}.sh 2>&1 | \
    tee validation_${candidate}.log

  EXIT_CODE=${PIPESTATUS[0]}

  # Record outcome
  echo "$candidate: exit_code=$EXIT_CODE" >> validation_summary.txt

  # Cleanup before next run
  pkill -9 -f train_lm.py
  sleep 10  # Allow devices to release

  # Early abort on catastrophic failure
  if grep -q "CUDA error: device-side assert" validation_${candidate}.log; then
    echo "Device error detected, aborting remaining candidates"
    break
  fi
done
```

**结果：**
```
candidate1: exit_code=0 ✓
candidate2: exit_code=137 (OOM) ✗
candidate3: exit_code=0 ✓
```

仅对成功运行的候选进入阶段 3。
