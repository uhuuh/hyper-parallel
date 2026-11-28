<a id="phase-3-batch-validation"></a>
# 阶段 3：批量验证

<a id="objective"></a>
## 目标
依次执行全部候选策略，监测错误并收集性能指标。

<a id="inputs"></a>
## 输入
- 阶段 2 生成的候选列表
- 已生成的配置文件和启动脚本
- 目标硬件，使用已授权资源池中的一个节点

<a id="execution-strategy"></a>
## 执行策略

<a id="sequential-validation-recommended"></a>
### 顺序验证（推荐）
一次运行一个候选：
```bash
for candidate in candidate1 candidate2 candidate3; do
    echo "Running $candidate..."
    bash validation_experiment/run_${candidate}.sh

    # Check for errors
    if grep -q "Error\|OOM\|Traceback" validation_experiment/${candidate}/validation_run.log; then
        echo "✗ $candidate failed"
        # Save error for analysis
    else
        echo "✓ $candidate completed"
    fi

    # Short delay between runs
    sleep 5
done
```

**优点：** 简单、便于调试。
**缺点：** 串行执行，3～5 个候选 × 7 分钟，总计 20～35 分钟。

<a id="parallel-validation-advanced"></a>
### 并发验证（高级）
有多个可用节点时：
```bash
# Node 1
bash validation_experiment/run_candidate1.sh &

# Node 2
bash validation_experiment/run_candidate2.sh &

wait
```

**优点：** 更快，全部候选约 7 分钟完成。
**缺点：** 需要多个节点，调试更复杂。

<a id="monitoring"></a>
## 监测

<a id="real-time-progress"></a>
### 实时进度
```bash
# Terminal 1: Run validation
bash validation_experiment/run_candidate1.sh

# Terminal 2: Monitor progress
tail -f validation_experiment/candidate1/validation_run.log | grep "Training:"
```

<a id="error-detection"></a>
### 错误检测
关注以下模式：

**1. FSDP 冲突，DP=1 问题：**
```
RuntimeError: A view was created in no_grad mode...
```
→ 立即拒绝并跳过该候选。

**2. OOM：**
```
NPU out of memory
torch.cuda.OutOfMemoryError
```
→ 记录显存限制，建议降低并行度。

**3. MoE 权重错误：**
```
ValueError: Detected fused MoE expert weights incompatible with TP sharding
```
→ 不允许 EP=1，应增大 EP。

**4. 挂起，无进展：**
```bash
# No log output for >60 seconds
timeout 600 bash validation_experiment/run_candidate.sh
```
→ 通信死锁，中止并分析。

**5. Loss 为 NaN/Inf：**
```
training/foundation_loss=nan
```
→ 梯度不稳定，检查学习率和梯度裁剪。

<a id="success-criteria"></a>
## 成功标准

满足以下条件的候选视为**成功**：
1. ✅ 无错误完成 50 个训练步
2. ✅ 吞吐超过基线，任何提升均可
3. ✅ 显存在设备限制内，小于容量的 90%
4. ✅ 性能稳定，标准差小于均值的 10%

<a id="metrics-extraction"></a>
## 指标提取

每次运行成功后提取指标：
```python
import re, statistics

log_path = f"validation_experiment/{candidate_id}/validation_run.log"
pattern = re.compile(r'tokens/s=([\d.]+),?\s+step_time=([\d.]+)s.*memory/device_max_allocated_gb=([\d.]+)')

tokens_per_sec = []
step_times_ms = []
memory_gb = []

with open(log_path) as f:
    for line in f:
        match = pattern.search(line)
        if match:
            tps, step_time, mem = match.groups()
            tokens_per_sec.append(float(tps))
            step_times_ms.append(float(step_time) * 1000)
            memory_gb.append(float(mem))

# Use last 30 steps (exclude warmup)
tokens_per_sec = tokens_per_sec[-30:]
step_times_ms = step_times_ms[-30:]

metrics = {
    "throughput_tokens_per_sec": statistics.mean(tokens_per_sec),
    "std_tokens_per_sec": statistics.stdev(tokens_per_sec) if len(tokens_per_sec) > 1 else 0,
    "step_time_ms": statistics.mean(step_times_ms),
    "memory_per_device_gb": max(memory_gb),
    "num_samples": len(tokens_per_sec),
}

# Save
import json
with open(f"validation_experiment/{candidate_id}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

<a id="output"></a>
## 输出

每个候选生成：
- `validation_experiment/{candidate_id}/validation_run.log`：完整训练日志
- `validation_experiment/{candidate_id}/metrics.json`：提取的指标
- 状态：✅ 成功，或 ❌ 失败并附错误类型

<a id="example-results"></a>
## 结果示例

```
Candidate 1 (TP=2, CP=1, EP=2, DP=2):
  ✓ Completed 50 steps
  Throughput: 2305 tokens/s (2.13x vs baseline)
  Memory: 6.83 GB
  Status: ✅ SUCCESS

Candidate 2 (TP=2, CP=1, EP=4, DP=1):
  ✗ Failed at initialization
  Error: RuntimeError: A view was created in no_grad mode
  Status: ❌ FSDP CONFLICT

Candidate 3 (TP=4, CP=1, EP=2, DP=1):
  ✗ Failed at initialization (same error)
  Status: ❌ FSDP CONFLICT

Candidate 4 (TP=1, CP=1, EP=2, DP=4):
  ✓ Completed 50 steps
  Throughput: 4274 tokens/s (3.95x vs baseline)
  Memory: 6.18 GB
  Status: ✅ SUCCESS ⭐
```

---

**下一步：** 阶段 4，比较与迭代。
