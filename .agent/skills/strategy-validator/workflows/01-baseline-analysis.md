<a id="phase-1-baseline-analysis"></a>
# 阶段 1：基线分析

<a id="objective"></a>
## 目标
从现有基线训练中提取性能指标，建立比较参考。

<a id="inputs"></a>
## 输入
- 基线训练日志，例如 `output/training_demo/run_parallel_online.log`
- 基线配置，例如 `train_parallel_online.yaml`

<a id="tasks"></a>
## 任务

<a id="1-locate-baseline-artifacts"></a>
### 1. 定位基线产物
```bash
# Find most recent training log
find output/training_demo -name "*.log" -type f -mtime -7

# Confirm config file
ls -lh examples/training_demo/qwen3_moe/train_parallel_online.yaml
```

<a id="2-extract-configuration"></a>
### 2. 提取配置
解析 YAML，识别：
- `accelerator`：tp_size、cp_size、ep_size、pp_size
- `fsdp_config`：dp_shard_size、edp_shard_size
- `training`：global_batch_size、micro_batch_size、train_iters
- `data_transform`：max_seq_len

计算派生指标：
- 总设备数 = TP × CP × EP × PP × DP
- DP 大小 = total_devices / (TP × CP × EP × PP)

<a id="3-extract-performance-metrics"></a>
### 3. 提取性能指标
使用正则表达式解析训练日志：
```python
pattern = r'tokens/s=([\d.]+),?\s+step_time=([\d.]+)s.*memory/device_max_allocated_gb=([\d.]+)'
```

提取：
- `throughput_tokens_per_sec`：预热后最后 30～50 步的均值
- `std_tokens_per_sec`：标准差，用于稳定性检查
- `step_time_ms`：平均步耗时，单位毫秒
- `memory_per_device_gb`：峰值显存占用

<a id="4-identify-bottlenecks"></a>
### 4. 识别瓶颈
分析步耗时的组成：
- **通信受限：** 步耗时远大于预期计算时间
  - 特征：TP/CP 并行度高、序列短、模型小
  - 处理方向：减小 TP/CP

- **计算受限：** 步耗时约等于预期计算时间
  - 特征：模型大、序列长、TP/CP 较低
  - 处理方向：可能已经最优，或增加并行度

- **显存受限：** 接近 OOM，频繁清理缓存
  - 特征：memory_allocated 超过设备容量的 90%
  - 处理方向：增大 DP，增加 FSDP 分片；减小 TP/PP

<a id="5-generate-baseline-report"></a>
### 5. 生成基线报告
```json
{
  "baseline": {
    "config": {
      "tp": 2, "cp": 2, "ep": 2, "pp": 1, "dp": 2,
      "seq_len": 128,
      "global_batch_size": 8,
      "model_layers": 4
    },
    "metrics": {
      "throughput_tokens_per_sec": 1081.4,
      "std_tokens_per_sec": 31.4,
      "step_time_ms": 722.8,
      "memory_per_device_gb": 6.81
    },
    "bottleneck": "communication",
    "evidence": "CP=2 for seq_len=128 introduces ~400ms overhead"
  }
}
```

<a id="outputs"></a>
## 输出
- `validation_experiment/baseline_metrics.json`
- 瓶颈诊断：通信、计算或显存
- 建议的优化方向

<a id="example-from-qwen3-30b-a3b-experiment"></a>
## 示例：Qwen3-30B-A3B 实验

**基线：** TP=2、CP=2、EP=2、DP=2。
- 吞吐：1081.4 tokens/s
- 步耗时：722.8 ms
- 显存：6.81 GB

**瓶颈：** 通信，短序列 seq_len=128 却使用 CP=2。

**建议：** 移除 CP，并考虑减小 TP。

---

**下一步：** 阶段 2，候选生成。
