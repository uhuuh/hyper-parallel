<a id="runnable-validation-configuration"></a>
# 可运行的验证配置

从仓库的 [Qwen3 MoE 训练模板](../../../../examples/training_demo/qwen3_moe/train_parallel_online.yaml)开始。保留其中的 `model._target_`、`optimizer._target_`、数据集、数据加载器和模型资产路径，并在基准测试前将这些路径调整为目标主机上的路径。仅包含模型名称的 YAML 无法运行。

下面是适用于 8 个 rank 的**补丁片段**，不是完整训练配置：

```yaml
accelerator:
  tp_size: 2
  cp_size: 1
  ep_size: 2
  pp_size: 1
fsdp_config:
  dp_shard_size: 4
  edp_shard_size: 4
training:
  train_iters: 50
  train_samples: null
```

Trainer 按 `DP = world_size / (TP * CP * PP)` 推导 DP。EP 共享当前 stage 的 rank，不是 world-size 乘积中的额外因子。每个 stage 的 rank 数必须能被 EP 整除。稠密 FSDP 分片数应整除 `DP * CP`；专家 FSDP 分片数应整除 `(world_size / PP) / EP`。批量设置还必须满足 `micro_batch_num * micro_batch_size * DP == global_batch_size`。每个改变 DP 的邻域候选都要重新检查这一不变量。

使用 `accelerator` 和 `fsdp_config` 设置并行度。顶层 `parallel_strategy` 和 `parallelism` 是搜索或描述格式，不是 Trainer 配置别名。优化器保留模板，并使用官方配置适配器 writer 写入 PSA 结果。基准运行通过 `training.train_iters` 限制步数，并清空 `training.train_samples`。导出的生产配置保留模板的训练长度。

调度器通过 `python -m torch.distributed.run --module
examples.training_demo.train_text CONFIG.yaml --training.train_iters=50 --training.train_samples=null` 启动**单节点**训练。自定义入口必须接受这个位置参数 YAML 及点号覆盖参数。`--devices` 表示本地 rank 数，多节点需要另行提供启动器。

进程成功退出并不足够：日志必须包含不同步骤的记录，其中 `tokens/s=`、`step_time=...s` 和 `memory/device_max_allocated_gb=` 都是有限正数。不同步骤的数量必须达到请求值。计算吞吐和步耗时均值时排除预热。
