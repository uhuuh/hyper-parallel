<a id="phase-4-comparison--iteration"></a>
# 阶段 4：比较与迭代

<a id="objective"></a>
## 目标
比较全部验证结果，确定最优策略，并决定部署还是继续迭代。

<a id="inputs"></a>
## 输入
- 阶段 1 的基线指标
- 阶段 3 的全部候选结果
- 目标提升倍数，例如 1.2 倍

<a id="comparison-analysis"></a>
## 比较分析

<a id="1-aggregate-results"></a>
### 1. 汇总结果

```python
def compare_candidates(baseline, candidates):
    results = []

    for candidate in candidates:
        metrics_path = f"validation_experiment/{candidate['id']}/metrics.json"

        if not Path(metrics_path).exists():
            # Failed candidate
            results.append({
                "id": candidate["id"],
                "name": candidate["name"],
                "status": "FAILED",
                "error": extract_error(candidate["id"]),
            })
            continue

        # Successful candidate
        with open(metrics_path) as f:
            metrics = json.load(f)

        improvement = metrics["throughput_tokens_per_sec"] / baseline["throughput_tokens_per_sec"]

        results.append({
            "id": candidate["id"],
            "name": candidate["name"],
            "strategy": candidate["strategy"],
            "throughput": metrics["throughput_tokens_per_sec"],
            "memory": metrics["memory_per_device_gb"],
            "improvement": improvement,
            "status": "SUCCESS",
        })

    # Sort by throughput (descending)
    results.sort(key=lambda x: x.get("throughput", 0), reverse=True)

    return results
```

<a id="2-generate-comparison-table"></a>
### 2. 生成比较表

```markdown
| # | Strategy | Config | Throughput | vs Baseline | Memory | Status |
|---|----------|--------|------------|-------------|--------|--------|
| 0 | Baseline | TP=2,CP=2,EP=2,DP=2 | 1,081 t/s | 1.00x | 6.81 GB | Ref |
| 1 | Remove CP | TP=2,CP=1,EP=2,DP=2 | 2,305 t/s | 2.13x | 6.83 GB | ✓ Good |
| 2 | Increase EP | TP=2,CP=1,EP=4,DP=1 | — | — | — | ✗ FSDP |
| 3 | Increase TP | TP=4,CP=1,EP=2,DP=1 | — | — | — | ✗ FSDP |
| 4 | Minimal TP ⭐ | TP=1,CP=1,EP=2,DP=4 | 4,274 t/s | 3.95x | 6.18 GB | ✓ BEST |
```

<a id="3-identify-winner"></a>
### 3. 确定胜出候选

```python
def identify_winner(results, target_improvement=1.2):
    successful = [r for r in results if r["status"] == "SUCCESS"]

    if not successful:
        return None, "NO_WINNERS"

    # Best = highest throughput
    best = successful[0]

    if best["improvement"] >= target_improvement:
        return best, "TARGET_MET"
    elif best["improvement"] > 1.0:
        return best, "IMPROVED"
    else:
        return None, "NO_IMPROVEMENT"
```

<a id="decision-tree"></a>
## 决策树

<a id="case-1-target-met-"></a>
### 情况 1：达到目标 ✅
```
Winner improvement >= target (e.g., 1.2x)
→ Deploy immediately
→ Generate deployment config
→ Document findings
```

<a id="case-2-improved-but-below-target-"></a>
### 情况 2：有所提升但未达目标 ⚠️
```
Best improvement > 1.0x but < target
→ Option A: Deploy best candidate (partial win)
→ Option B: Iterate with new candidates
  - Analyze what worked
  - Generate refined candidates
  - Run Phase 2-4 again (limit: 2 iterations)
```

<a id="case-3-no-improvement-"></a>
### 情况 3：没有提升 ❌
```
All candidates failed or worse than baseline
→ Analyze failure patterns:
  - All DP=1? → Framework constraint, avoid DP=1
  - All OOM? → Model too large, need more devices or reduce batch
  - All worse? → Baseline already optimal
→ If <2 iterations: Generate new candidates
→ If ≥2 iterations: Accept baseline, report findings
```

<a id="case-4-all-failed-"></a>
### 情况 4：全部失败 ❌
```
All candidates failed (errors, not just worse)
→ Diagnose common failure:
  - FSDP conflict → Adjust DP constraints
  - MoE error → Fix EP requirements
  - OOM → Reduce parallelism, increase sharding
→ Generate safer candidates (more conservative)
→ Re-run Phase 2-4 (max 1 retry)
```

<a id="iteration-logic"></a>
## 迭代逻辑

```python
def should_iterate(results, iteration_count, target_met):
    if target_met:
        return False  # Success, no need to iterate

    if iteration_count >= 2:
        return False  # Max iterations reached

    # Iterate if:
    # - Some improvement found (worth exploring further)
    # - Clear pattern in failures (can address)

    successful = [r for r in results if r["status"] == "SUCCESS"]

    if not successful:
        # All failed - analyze patterns
        failed_reasons = [r["error"] for r in results if r["status"] == "FAILED"]
        if most_common(failed_reasons) == "FSDP":
            return True  # Can fix by adjusting DP

    if successful and successful[0]["improvement"] > 1.0:
        return True  # Some improvement, try to push further

    return False
```

<a id="deployment-artifacts"></a>
## 部署产物

确定胜出候选后，生成：

<a id="1-production-config"></a>
### 1. 生产配置
```yaml
# validation_experiment/production_config.yaml
accelerator:
  tp_size: 1
  cp_size: 1
  ep_size: 2
  pp_size: 1
  sequence_parallel: false  # TP=1, no SP needed
  loss_parallel: true

fsdp_config:
  dp_shard_size: 4
  edp_shard_size: 4
  reshard_after_forward: true
  reshard_after_backward: true
  requires_grad_sync: true
```

<a id="2-migration-guide"></a>
### 2. 迁移指南
```markdown
# Migration to Optimized Config

## Change Summary
- TP: 2 → 1 (remove tensor parallelism)
- CP: 2 → 1 (remove context parallelism)
- DP: 2 → 4 (increase data parallelism)

## Expected Impact
- Throughput: +295% (1081 → 4274 tokens/s)
- Training time: -75%
- Memory: -9% (6.81 → 6.18 GB)

## Deployment Steps
1. Update train_parallel_online.yaml with new config
2. Test with 100 steps on dev data
3. Confirm loss matches baseline trajectory
4. Deploy to production
```

<a id="3-final-report"></a>
### 3. 最终报告
```markdown
# Strategy Validation Report

**Model**: Qwen3-30B-A3B (4 layers)
**Goal**: 1.2x throughput improvement
**Result**: ✓ 3.95x improvement achieved

## Winner
Config: TP=1, CP=1, EP=2, DP=4
Throughput: 4274 tokens/s (+295%)

## Key Insights
- CP harmful for seq_len < 512
- TP harmful for models < 10 layers
- Communication > Computation (main bottleneck)

## Deployment: APPROVED
Risk: Low
Confidence: High (validated over 50 steps)
```

<a id="example-output"></a>
## 输出示例

以下来自 Qwen3-30B-A3B 实验：

```
═══════════════════════════════════════════════
VALIDATION COMPLETE
═══════════════════════════════════════════════

Tested: 4 candidates
Successful: 2 candidates
Failed: 2 candidates (FSDP errors)

🏆 Winner: Candidate 4 (TP=1, CP=1, EP=2, DP=4)
  Throughput: 4274 tokens/s (3.95x vs baseline)
  Training time: -75%
  Memory: -9%

Target: 1.2x → EXCEEDED (3.95x > 1.2x) ✓

Decision: DEPLOY IMMEDIATELY
Status: VALIDATION COMPLETE
Iteration: 0/2 (no iteration needed)

Files:
  - validation_experiment/production_config.yaml
  - validation_experiment/FINAL_REPORT.md
  - validation_experiment/DEPLOYMENT_GUIDE.md
```

---

**结果：** 验证完成，已确定最优策略，可以部署到生产。
