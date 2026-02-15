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

## 2026-02-13 - Decode Iteration Extension and Resume Hardening

### Goal
Push decode speedup beyond 22.42 by extending iterative search from 25 to 40 steps.

### Hypothesis
Additional steps with resume should discover better launch/tile combinations after the first valid kernel appears.

### Action
- Ran decode iterative Modal baseline with env-loaded API key and confirmed successful completion at 25 steps.
- Attempted resume from 25->40 and hit a type mismatch in resume scoring (`tuple` vs `float`).
- Fixed `local_best_score` initialization to tuple form in `agent/iterative_agent.py` for resume path correctness.
- Re-ran 40-step resumed decode experiment successfully.

### Outcome
- 40-step resumed run completed cleanly.
- Best score did not improve; global best remained `22.4212` (same as step 17 from the 25-step run).

### Breakthrough / Insight
The current loop is now stable enough for long runs, but tuning efficiency is low: many edits fail to apply due to brittle long-string replacements. This suggests diminishing returns from pure iterative `str_replace` and motivates stronger proposal diversity / targeted template shifts.

### Next Step
Pivot to decode-focused evolve proposals (higher diversity) and/or tighten tuner prompts toward small unique-anchor edits to reduce no-op rounds.

## 2026-02-13 - Decode Evolve Probe (Diversity Test)

### Goal
Test whether proposal diversity (evolve) can beat iterative plateau on decode.

### Hypothesis
Evolve proposals should escape local minima caused by brittle text-edit tuning.

### Action
- Added `config/config_evolve_gdn_decode_modal.yaml` (decode-only, Modal, 30 proposals).
- Ran evolve decode experiment.

### Outcome
- Run hit CLI timeout before full completion, but produced 22 evaluated proposals.
- Best observed proposal so far: `proposal_21_metrics.json` with `correct=True` and `speedup=28.5238`.

### Breakthrough / Insight
Proposal diversity is materially better than iterative-only tuning for this task: partial evolve already exceeds iterative best (`22.4212`) by a large margin.

### Next Step
Complete a full evolve decode run (or restart at 22 proposals for reproducible completion), then port the best proposal into an iterative warm-start track for local refinements.

## 2026-02-13 - Evolve Breakthrough + Iterative Warm Start

### Goal
Convert evolve decode gains into a warm-started iterative refinement loop.

### Hypothesis
Seeding iterative search from the best evolve proposal should either improve beyond evolve peak or at least stabilize a high baseline with lower variance.

### Action
- Completed full decode evolve run with Opus 4.6 (`total_steps=30`) on Modal.
- Best completed speedup reached `31.8910` with correctness preserved.
- Created a warm-start seed from best proposal (`proposal_28`) and mapped it to iterative resume format (`proposal_0_1.*`).
- Added warm-start config: `config/config_iterative_gdn_decode_modal_warmstart.yaml`.
- Ran warm-start iterative decode (`total_steps=20`).

### Outcome
- Warm-start iterative run completed successfully and preserved correctness.
- Best speedup remained `31.8910` (no improvement over evolve-best in this refinement window).

### Breakthrough / Insight
Evolve is currently the major unlock for decode; iterative refinement from an already strong kernel is mostly plateauing due to high no-op edit rate and fragile long-span string replacement.

### Next Step
Use evolve as primary search for decode, then run shorter iterative phases with stricter tuner constraints (unique anchors, minimal-diff edits) to reduce no-op rounds.

## 2026-02-13 - Additional Evolve Pass + Short Warm Refinement

### Goal
Validate whether another full evolve pass can exceed the current decode peak and then refine with a short warm-start iterative run.

### Hypothesis
A second evolve run with the same setup may discover a different high-performing region; a short 12-step warm refine can then polish it.

### Action
- Ran a fresh 30-step decode evolve on Modal (`ap-LG0DrZRhQb8ih2M9TYCxQM`).
- Seeded warm-start from that run's best proposal and executed 12-step iterative refine (`ap-R3o569SMRYuw6tLOOVe5BV`).
- Compared against best-ever evolve result and reset `outputs/warmstart_evolve_decode_latest` to global-best seed (`proposal_28`, speedup `31.8910`).

### Outcome
- New evolve run completed correct but lower best speedup: `28.7158`.
- Short iterative warm refine maintained `28.7158` (no uplift).
- Best overall decode remains previous evolve peak: `31.8910`.

### Breakthrough / Insight
Run-to-run variance in proposal quality is significant; keeping a stable global-best seed is crucial. Short iterative passes are useful for validation but currently do not reliably beat strong evolve peaks.

### Next Step
Introduce stricter tuner edit protocol (small unique anchor edits only) before the next warm-start iterative attempt, while continuing evolve as the main search driver.

## 2026-02-13 - Prompt Hardening (Tuner v2)

### Goal
Reduce wasted iterative steps caused by non-applied `str_replace` edits.

### Hypothesis
Constraining tuner behavior to anchor-based micro-edits and explicit no-edit fallback will lower no-op rates and improve refinement efficiency.

### Action
- Updated `prompt/tuner_prompt.py` with stricter edit protocol:
  - two-stage objective (correctness-first, then perf micro-tuning),
  - max 2 edits per response,
  - required unique anchor line in each `old_str`,
  - explicit `<NO_VALID_EDIT/>` output when safe exact matches are not available,
  - stronger guidance against broad multi-block replacements.

### Outcome
Prompt update completed and syntax-checked.

### Breakthrough / Insight
For this framework, editing protocol quality is as important as model quality. Reliable small edits are higher leverage than aggressive rewrite attempts.

### Next Step
Run a short warm-start iterative pass from global-best seed and compare no-op warning frequency vs prior runs.

## 2026-02-13 - Structured Trace Prompt Integration

### Goal
Integrate structured reasoning trace output without breaking code/edit extraction.

### Hypothesis
If we require a JSON engineering trace before code and then strip it before extraction, we get better auditability without harming execution.

### Action
- Added a structured engineering trace block to proposer prompt (`prompt/proposer_prompt.py`).
- Added output sanitization utility to separate JSON trace from executable text (`agent/utils.py`).
- Wired sanitization into proposer and tuner steps before `extract_first_code` / `extract_edits` (`agent/iterative_agent.py`).
- Persisted cleaned LLM response and extracted trace JSON to step artifacts (`*_response.txt`, `*_trace.json`) for iterative/evolve logs (`agent/iterative_agent.py`, `agent/evolve_agent.py`).

### Outcome
Pipeline now supports trace-first outputs while keeping downstream extraction robust.

### Breakthrough / Insight
Prompt upgrades are safest when paired with deterministic post-processing; extraction robustness must be co-designed with output format.

### Next Step
Run a short decode smoke run to verify trace files are generated and no-op/edit behavior remains stable.

## Ongoing Template

Copy this for each new experiment:

```markdown
## YYYY-MM-DD - <experiment name>

### Goal
<what we want to improve>

### Hypothesis
<why this might help>

### Reasoning
<how did we decide that this is the right optimization step>

### Action
<what we changed>

### Outcome
<what happened>

### Breakthrough / Insight
<key learning that changed strategy>

### Next Step
<follow-up action>
```
