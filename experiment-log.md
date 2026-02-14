# Experiment Log

This document tracks the evolution of kernel-generation improvements, including hypotheses, actions, outcomes, and breakthroughs.

Note: this log captures concise engineering reasoning and decisions (not private chain-of-thought).

## 2026-02-13 - Baseline Reliability Pass

### Goal
Improve run reliability and signal quality before focused GDN optimization.

### Experiment 1 - Resume behavior across multi-task runs
- Hypothesis: Resume logic may fail across multiple tasks because resume path is mutated in-loop.
- Action: Reviewed `agent/main.py` task loop and updated logic to keep an immutable `resume_root`.
- Outcome: Per-task resume path now derives from original root each iteration.
- Files: `agent/main.py`
- Result: Reduced risk of partial/non-deterministic resume behavior.

### Experiment 2 - Temperature control consistency
- Hypothesis: Configured sampling temperature is not reaching model calls.
- Action: Threaded `args.temperature` into proposer and tuner API calls.
- Outcome: Search stochasticity is now controlled by config/CLI as intended.
- Files: `agent/iterative_agent.py`
- Result: Better reproducibility/tuning control.

### Experiment 3 - No-op tuning penalty
- Hypothesis: `str_replace`-based tuning can silently produce no-op iterations, wasting eval budget.
- Action: Counted applied edits; when zero edits apply, mark step as failed with explicit error and skip eval.
- Outcome: No-op rounds are penalized and visible in logs.
- Files: `agent/iterative_agent.py`
- Result: Cleaner optimization signal and fewer wasted evaluations.

### Breakthrough
The key shift was treating ineffective edit rounds as first-class failure signals instead of letting them silently pass through evaluation. Combined with correct resume behavior and temperature plumbing, this makes iteration quality and repeatability significantly better.

## 2026-02-13 - GDN Track Focus Setup

### Goal
Narrow experiments to Track C (Gated Delta Net) and support Modal-only workflows.

### Experiment 4 - Task scoping
- Hypothesis: Running all tracks dilutes search budget and observability for GDN.
- Action: Added dedicated GDN task list.
- Outcome: Runs now target only `gdn` kernels.
- Files: `config/tasks_gdn.txt`
- Result: Budget and analysis concentrated on target track.

### Experiment 5 - Config specialization
- Hypothesis: Dedicated configs reduce run mistakes and improve experiment hygiene.
- Action: Added iterative/evolve GDN configs, plus Modal variants.
- Outcome: One-command runs for local or Modal backend.
- Files:
  - `config/config_iterative_gdn.yaml`
  - `config/config_evolve_gdn.yaml`
  - `config/config_iterative_gdn_modal.yaml`
  - `config/config_evolve_gdn_modal.yaml`
- Result: Faster iteration setup and fewer operator errors.

## 2026-02-13 - GDN Cooking Start (Track C)

### Goal
Start disciplined GDN optimization with isolated decode/prefill experiments on Modal B200.

### Hypothesis
Separating decode and prefill into dedicated runs improves signal quality and helps apply the optimization ladder without cross-kernel noise.

### Action
- Reviewed `skills/gpu-kernel-engineering.md` and aligned strategy to ladder-style optimization.
- Confirmed current GDN definitions in dataset:
  - `gdn_decode_qk4_v8_d128_k_last`
  - `gdn_prefill_qk4_v8_d128_k_last`
- Added dedicated task files and Modal iterative configs:
  - `config/tasks_gdn_decode.txt`
  - `config/tasks_gdn_prefill.txt`
  - `config/config_iterative_gdn_decode_modal.yaml`
  - `config/config_iterative_gdn_prefill_modal.yaml`

### Outcome
We can now run decode-only and prefill-only experiments independently with one command each.

### Breakthrough / Insight
The practical bottleneck is not just kernel quality but experiment hygiene. Isolating kernels by task is a high-leverage change because it tightens feedback loops and makes regression causes obvious.

### Next Step
Run decode-only iterative first, analyze first successful kernel for correctness/perf bottleneck class (memory vs compute vs latency), then apply one structured mutation per run.

## Ongoing Template

Copy this for each new experiment:

```markdown
## YYYY-MM-DD - <experiment name>

### Goal
<what we want to improve>

### Hypothesis
<why this might help>

### Action
<what we changed>

### Outcome
<what happened>

### Breakthrough / Insight
<key learning that changed strategy>

### Next Step
<follow-up action>
```
