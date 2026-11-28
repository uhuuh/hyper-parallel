<a id="benchmark-recipes"></a>
# 基准测试 Recipe

用于验证并行策略的标准工作负载。

<a id="design-principles"></a>
## 设计原则

1. **可复现：** 固定种子、合成数据、确定性操作
2. **速度：** 50 步，约 5～10 分钟，足以测量吞吐和显存
3. **覆盖：** 具有真实训练的代表性，包括前向、反向和优化器更新
4. **性能分析：** 在指定记录点启用追踪采集

<a id="standard-recipe-language-model-pre-training"></a>
## 标准 Recipe：语言模型预训练

<a id="target-models"></a>
### 目标模型

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头数 | KV 头数 | FFN 维度 | 验证时间 |
|-------|--------|--------|--------|-------|----------|---------|-----------------|
| Qwen3-0.5B | 0.5B | 24 | 896 | 14 | 2 | 4864 | 约 2 分钟 |
| Qwen3-1.8B | 1.8B | 24 | 2048 | 16 | 2 | 11008 | 约 3 分钟 |
| Qwen3-7B | 7B | 28 | 4096 | 32 | 4 | 11008 | 约 5 分钟 |
| Qwen3-14B | 14B | 40 | 5120 | 40 | 8 | 13824 | 约 7 分钟 |
| Qwen3-30B-A3B（裁剪） | 约 12B | 24 | 8192 | 64 | 8 | 29568 | 约 8 分钟 |
| Qwen3-30B-A3B（完整） | 30B | 60 | 8192 | 64 | 8 | 29568 | 约 15 分钟 |

**通过裁剪加快验证：** 将层数减至原来的 24～40%，在保留显存与通信特征的同时，将运行加快 2～3 倍。

<a id="workload-parameters"></a>
### 工作负载参数

```yaml
benchmark_recipe: lm_pretraining

model:
  model_name_or_path: /path/to/Qwen3-30B-A3B
  crop_layers: 24  # Reduce from 60 for fast validation

data:
  dataset_type: synthetic
  seed: 42

  # Synthetic data config
  vocab_size: 152064  # Qwen3 vocab
  sequence_length: 4096
  num_samples: 1000  # Infinite iterator for benchmark

training:
  max_steps: 50
  warmup_steps: 10

  # Batch sizes
  per_device_train_batch_size: 1  # Micro-batch
  global_batch_size: 128  # Adjust based on dp × num_microbatches

  # Optimizer (full fidelity)
  optimizer_type: adamw
  learning_rate: 1e-4
  lr_scheduler_type: constant_with_warmup
  weight_decay: 0.01

  # No checkpointing (benchmark only)
  save_strategy: no

profiling:
  profile: true
  profile_steps: [15, 45]  # After warmup, near end
  profile_memory: true
  profile_ranks: [0]  # Only rank 0 to reduce overhead

logging:
  logging_steps: 1
  log_memory_steps: 5
  log_level: info
```

<a id="synthetic-data-generator"></a>
### 合成数据生成器

```python
import torch
from torch.utils.data import IterableDataset, DataLoader

class SyntheticLMDataset(IterableDataset):
    """Infinite synthetic data for benchmarking."""

    def __init__(self, vocab_size, seq_length, seed=42):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.seed = seed

    def __iter__(self):
        # Each worker gets different seed
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        generator = torch.Generator().manual_seed(seed)

        while True:
            # Generate random tokens
            input_ids = torch.randint(
                0, self.vocab_size,
                (self.seq_length,),
                generator=generator
            )

            # Labels = shifted input_ids (causal LM)
            labels = input_ids.clone()

            yield {
                "input_ids": input_ids,
                "labels": labels
            }

# Usage in training script
def create_benchmark_dataloader(config, parallel_config):
    dataset = SyntheticLMDataset(
        vocab_size=config.vocab_size,
        seq_length=config.sequence_length,
        seed=config.seed
    )

    # Micro-batch size per device
    batch_size = config.per_device_train_batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # No multiprocessing for synthetic data
        pin_memory=True
    )

    return dataloader
```

<a id="recipe-variants"></a>
## Recipe 变体

<a id="1-memory-limited-validation-fast"></a>
### 1. 显存受限验证（快速）

用于 OOM 检测、基本吞吐等快速检查：

```yaml
training:
  max_steps: 20  # Shorter run
  warmup_steps: 5
  per_device_train_batch_size: 1

profiling:
  profile: false  # Skip profiling for speed

logging:
  logging_steps: 1
```

**使用场景：** 初步筛选前三个候选，每个约需 2～3 分钟。

<a id="2-full-profiling-validation"></a>
### 2. 完整性能分析验证

用于详细的通信与计算分析：

```yaml
training:
  max_steps: 100
  warmup_steps: 20

profiling:
  profile: true
  profile_steps: [25, 50, 75]  # Multiple checkpoints
  profile_memory: true
  profile_with_stack: true  # Capture kernel stacks
  profile_ranks: [0, 1]  # Multiple ranks for comm analysis

  # Advanced profiling
  record_shapes: true
  with_flops: true
  with_modules: true
```

**使用场景：** 生产就绪验证、PSA 校准。

<a id="3-long-context-validation"></a>
### 3. 长上下文验证

用于 CP 或序列并行策略：

```yaml
model:
  crop_layers: 12  # Fewer layers to fit memory with long seq

data:
  sequence_length: 32768  # 32K context

training:
  max_steps: 30  # Shorter due to longer per-step time
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4  # Maintain global batch
```

**使用场景：** CP > 1 的策略、长序列工作负载。

<a id="4-moe-validation"></a>
### 4. MoE 验证

用于 EP，即专家并行：

```yaml
model:
  model_name_or_path: /path/to/mixtral-8x7b  # or Qwen2.5-MoE

training:
  max_steps: 40
  load_balancing_loss_weight: 0.01  # Enable auxiliary loss

profiling:
  profile_steps: [15, 35]
  # Monitor AllToAll communication for EP
```

**使用场景：** EP > 1、MoE 模型。

<a id="expected-outputs"></a>
## 预期输出

<a id="success-metrics-per-recipe"></a>
### 各 Recipe 的成功指标

**根据该工作负载的实测基线制定目标。** 绝对吞吐、显存占用和 MFU 取决于架构、序列长度、并行布局、优化器以及软硬件版本。从其他配置复制的范围不能作为验收标准。例如，使用矩阵优化器的稀疏 MoE 可能报告约 12% 的 PaLM MFU，但这并不意味着它与稠密模型具有相同瓶颈。

| 标准 | 定义 |
|---|---|
| 吞吐 / 步耗时 | 相对本轮实测基线的改进。 |
| 显存 | 设备容量占比，并声明 OOM 余量。 |
| MFU / HFU | 分别声明核算口径，再使用相对基线的目标或有文档依据的 roofline 分析。 |
| 候选比较 | 使用相同节点、设备数和实测步数；排除约定的预热，并连续运行。 |

通过 `training.mfu.target_*` 设置目标，并检查最终验收输出，参见 [MFU 目标](../../perf-playbook/references/mfu-targets.md)。

结果偏离基线预期时，先核查核算方式和配置等价性，再调查硬件或实现问题。

<a id="failure-modes"></a>
### 失败模式

| 症状 | 可能原因 | Recipe 调整 |
|---------|--------------|-------------------|
| 第 1～5 步 OOM | 模型状态过大 | 启用 FSDP level3，减小微批 |
| 第 15 步后 OOM | 激活累积 | 启用激活检查点 |
| 吞吐低于预期的 50% | 通信瓶颈 | 减小 TP/CP，检查网络 |
| Loss = NaN | 数值不稳定 | 降低学习率，检查混合精度 |
| 初始化挂起 | Mesh 配置不匹配 | 检查 dp×tp×pp×cp×ep = device_count |

<a id="recipe-selection-guide"></a>
## Recipe 选择指南

```python
def select_recipe(model_size_b, device_count, validation_depth):
    """
    Args:
        model_size_b: Model size in billions
        device_count: Total devices
        validation_depth: "fast" | "standard" | "full"
    """

    # Determine crop ratio
    if model_size_b > 30:
        crop_ratio = 0.25  # Crop to 25% layers
    elif model_size_b > 10:
        crop_ratio = 0.40
    else:
        crop_ratio = 1.0  # No cropping for small models

    # Select step count
    steps_map = {
        "fast": 20,
        "standard": 50,
        "full": 100
    }
    max_steps = steps_map[validation_depth]

    # Enable profiling?
    profile = validation_depth in ["standard", "full"]

    return {
        "crop_ratio": crop_ratio,
        "max_steps": max_steps,
        "warmup_steps": max_steps // 5,
        "profile": profile,
        "profile_steps": [max_steps // 3, max_steps * 2 // 3] if profile else []
    }
```

<a id="multi-candidate-recipe"></a>
## 多候选 Recipe

依次验证前三个策略时：

```bash
#!/bin/bash
# Run 3 candidates with fast recipe (20 steps each, ~15 min total)

CANDIDATES=("candidate1" "candidate2" "candidate3")
RECIPE="fast"  # Use fast recipe for multi-candidate

for candidate in "${CANDIDATES[@]}"; do
    echo "=== Validating $candidate ==="

    # Generate config with fast recipe
    python generate_config.py \
        --psa_output psa_${candidate}.json \
        --recipe $RECIPE \
        --output config_${candidate}.yaml

    # Launch
    timeout 300 torchrun \
        --nproc_per_node=64 \
        scripts/train_lm.py \
        --config config_${candidate}.yaml \
        2>&1 | tee log_${candidate}.txt

    # Extract quick metrics
    python extract_metrics.py log_${candidate}.txt > metrics_${candidate}.json

    # Cleanup for next run
    pkill -9 -f train_lm.py
    sleep 5
done

# Compare results
python compare_candidates.py metrics_candidate*.json
```

<a id="validation-checklist"></a>
## 验证清单

基准测试前：

- [ ] 模型检查点可访问
- [ ] 通过 `npu-smi` / `nvidia-smi` 确认设备可用
- [ ] 配置与 PSA 策略一致，dp×tp×pp×cp×ep = device_count
- [ ] 磁盘空间充足，日志约需 100 MB
- [ ] 没有冲突进程
- [ ] Recipe 参数与验证目标一致：快速、标准或完整

基准测试后：

- [ ] 所有步骤完成，没有 OOM 或挂起
- [ ] 日志包含吞吐记录
- [ ] 已记录显存指标
- [ ] 启用 profiler 时已写出追踪
- [ ] 退出码为 0
