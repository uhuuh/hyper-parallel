# 复现已记录的模型适配过程

六个 HF 原生模型使用本目录上一级的通用工具包，按照 `hf-model-integration` Skill 的阶段状态机，从 S0 调研逐步推进到 S5 验收。本目录让其他人能够重新运行各个阶段，并与已有记录比较：

| 文件 | 用途 |
|---|---|
| `reproduce.sh` | 按模型和阶段提供运行命令，保留各次历史运行使用的选项 |
| `parity_check.py` | S2 同进程一致性检查，通过实际训练使用的路径构建模型 |
| `adamw_recipe.py` | 为 S3 归因实验生成仅使用 AdamW 的 Recipe（`_target_` 无法通过覆盖参数修改） |
| `reference_results.md` | 各阶段记录的结果，以及哪些记录产生于相关修复之前 |

请先阅读下方的“比较前须知”：其中两组历史结果使用的代码路径后来已被修复或停用。

## 所需条件

- **原记录使用的硬件：** 每节点 8 个 die、每 die 配备 64 GB HBM 的昇腾 NPU 节点。32B 模型的双节点实验需要两台这样的节点、在两端挂载路径相同的共享存储，以及节点之间的免密 SSH。使用 `training.init_device=cpu` 加载发布权重时，需要关注主机内存：每个 rank 都会先在主机上物化完整模型（原运行使用的 1 TB 内存主机没有成为限制）。
- **原记录使用的软件：** CANN 9.0.0、Python 3.10、torch 2.9.0、torch_npu 2.9.0.post2 和 transformers 5.13.0。如果基础环境中的 transformers 较旧，可将 5.13.0 安装到依赖覆盖目录，并通过 `EXTRA_PYTHONPATH` 指定；transformers 5.13 将依赖版本限制为 tokenizers <= 0.23.0、huggingface_hub < 2.0。图文模型还需要在同一环境中安装 torchvision 和 Pillow。
- **仓库版本：** 使用包含非持久缓冲区修复（`fix(models): keep constructed non-persistent buffers across to_empty`）的提交，或其后续版本。
- **权重：** 每个模型放在 `$MODEL_ROOT/<directory>` 中，其中目录名是 Hub ID 的最后一段：

  | 模型键 | Hub ID | 参数规模与结构 |
  |---|---|---|
  | `glm4-9b` | `THUDM/GLM-4-9B-0414` | 9.4B，稠密模型 |
  | `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M` | 0.36B，稠密模型，共享嵌入权重 |
  | `dsv2-lite` | `deepseek-ai/DeepSeek-V2-Lite` | 15.7B，采用 MLA 的 MoE 模型 |
  | `qwen25-32b` | `Qwen/Qwen2.5-32B` | 32.5B，稠密模型 |
  | `qwen25-vl-3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | 3.8B，图文模型 |
  | `qwen35-9b` | `Qwen/Qwen3.5-9B` | 9B，混合线性注意力模型 |

  原运行没有记录 Hub revision。下载时请记录版本（`huggingface-cli download <id> --local-dir <dir>` 会打印该信息），以便后续区分运行差异和检查点变化。

## 运行一个阶段

```bash
bash .agent/examples/hf_text_bringup/reproduce/reproduce.sh list
DRYRUN=1 MODEL_ROOT=/data/models bash .agent/examples/hf_text_bringup/reproduce/reproduce.sh qwen35-9b s3
MODEL_ROOT=/data/models CANN_SETENV=/path/to/set_env.sh PYTHON_BIN=/path/to/python \
    bash .agent/examples/hf_text_bringup/reproduce/reproduce.sh qwen35-9b s3
```

`DRYRUN=1` 只打印所有命令，不修改任何内容；实际运行前请先阅读打印出的命令。输出位于 `$OUT/<model>/<stage>/`（默认根目录为 `<repo>/output/reproduce`）：每个实验用例有独立日志，相关阶段还会生成 `parity.json`、`judge.txt`、`breakdown.html` 或 `untestable.txt`。

| 阶段 | 执行内容 | 与历史记录比较的项目 |
|---|---|---|
| `s1` | 在主机上检查发布权重布局，并预测共享权重的保存问题 | 退出码为 0、条目数 |
| `s2a` | 单 die 一致性检查：逐位比较参数、缓冲区和 logits | `PARITY PASS` |
| `s2b` | 30 步训练冒烟 | 起止 loss、步耗时、峰值显存 |
| `s3` | 等价性矩阵（重计算、重跑、跨布局）及判定 | `judge.txt` |
| `s4` | 运行 12 步，采集第 5～8 步的追踪，再分解步耗时 | 六项一致性检查、三类耗时占比 |
| `s5` | 按模型规模确定长度的长运行 | 步耗时中位数及尾部耗时、显存、loss 趋势 |

模型专属的额外实验：`smollm2-360m s3-adamw` 仅使用 AdamW 重做跨布局实验；`qwen25-32b s2b-full` 和 `s4-full` 运行完整的 64 层模型，需要在主机上构建模型并启用重计算。

请按顺序运行各阶段；后续阶段假定前面的阶段已通过。双节点实验需要设置 `NODES="host1 host2"`。

## 共享节点

单节点实验在启动前执行 `.agent/skills/hf-model-integration/scripts/node_guard.sh pre`，结束后执行 `post`；双节点启动器自行检查每个节点，只有全部通过才启动。节点繁忙时，该阶段以退出码 3 停止。请稍后重新探测，不要排队等待他人的任务结束。

## S3 证据与判定

将这些预训练权重用于分布式比较之前，请先在当前框架版本上运行 S2 一致性检查。检查不仅比较参数和 logits，也比较 RoPE 频率等非持久缓冲区。如果缓冲区不匹配，需要先修复模型构建问题，再进入 S3。

默认情况下，S3 实验与历史记录一样，依据 Trainer 日志判定：打印值有六位有效数字，并且不检查比较双方是否使用相同输入。设置 `S3_EVIDENCE=structured` 后，所有用例都会切换到运行时结构化证据（`model_integration.mode=runtime`：全精度 loss、梯度范数和逐步输入哈希）；这是当前 Skill 要求的方式。该工具包尚未在设备上运行过此模式；如果某个用例因模型集成检查发现的问题而在构建阶段停止，请保留这些发现，再回退到默认模式。

每项判定独立调用一次比较工具，因此某个用例无法运行时，只影响依赖它的判定。重计算和重跑的比较双方必须精确一致。跨布局比较的前三步必须精确一致；比较工具会报告从哪一步开始分叉，以及差距增长的速度。第一步就出现差异属于缺陷；运行几步后分叉属于归约顺序差异的累积，起点因模型而异。如果某个用例因显存不足而无法运行，会在 `untestable.txt` 中记录原因，不会静默跳过。

## 比较前须知

1. **在多个 rank 上加载发布权重的历史运行早于一项修复。** 修复缓冲区问题之前，预训练路径先在 meta 设备上构建模型，再调用 `to_empty`，随后只恢复检查点包含的内容。这导致所有非持久缓冲区都保留为未初始化内存，包括 RoPE 的 `inv_freq`。受影响的记录包括：`smollm2-360m`、`qwen25-32b` 和 `qwen35-9b` 的全部 S3 用例，以及 `qwen25-vl-3b` 的全部训练阶段。当前运行的绝对 loss 不会与这些记录一致。历史比较的双方具有相同的缓冲区内容，前几步完全一致也说明了这一点，因此“布局之间一致”的结论仍然成立。但分叉起始步和差距增长数据是在位置编码损坏的模型上测得的，需要重新测量。随机初始化的运行不受影响，包括文本模型的 S2b、S4、S5，以及 `glm4-9b` 和 `dsv2-lite` 的所有阶段。
2. **GLM-4 的历史记录使用了后来停用的试验 Recipe：** 对应的模型家族注册和 `examples/training_demo/glm4/` 保存在 `archive/glm4-bringup` 标签下。当前驱动脚本采用预期的通用工具包路径，但尚未通过这一路径在设备上运行过 GLM-4。要精确复现历史数值，请检出该标签并使用其中的 `run_glm4_text.sh`。
3. **DSV2-Lite 的历史 S3 使用旧协议：** 在裁剪为 13 层的模型上随机初始化，并用统计误差带判定。当前标准已替换旧协议，驱动脚本执行的是新协议（发布权重、由相同样本组成的语料）。因此，历史记录没有可直接对应的结果；你的运行将首次执行当前协议。
4. **历史 S2a 一致性检查执行的是其他路径。** 当时延迟加载尚未进入代码库，因此单进程检查实际比较的是 Transformers 自身的加载路径。图文模型的检查还使用了 `force_hf=True`。`parity_check.py` 通过 S3 实际使用的路径构建模型，因此能够发现旧检查未发现的缓冲区缺陷。图文模型使用的 `--image` 选项尚未运行过。
5. **代码库已发生变化。** 历史运行使用的是基于 2026-10-20 上游代码副本的模型适配分支。此后，仓库合入了更新的上游工作，包括通过 DCP 加载 Hugging Face 权重。步耗时略有变化属于预期情况；差异较大时应记录并报告。
6. **按已有记录，长运行从未保存检查点。** S5 的历史记录只覆盖训练步循环的稳定性。
