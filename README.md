# MLSys 2026 FlashInfer AI Kernel Generation Contest: Agent Baseline

An LLM agent baseline for the [MLSys 2026 FlashInfer AI Kernel Generation Contest](https://mlsys26.flashinfer.ai/). See the [flashinfer-bench-starter-kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit) to get started.

An LLM agent baseline that iteratively generates and refines Triton kernels for high-performance LLM operations on NVIDIA GPUs, evaluated via [FlashInfer-Bench](https://bench.flashinfer.ai). For the benchmarking framework code, see the [flashinfer-bench repo](https://github.com/flashinfer-ai/flashinfer-bench/).

## Project Structure

```
agent/
  main.py              # Entry point & task orchestration
  iterative_agent.py   # Iterative Agent: propose + refine loop
  evolve_agent.py      # Evolve Agent: elite pool evolution loop
  api.py               # LLM API client (OpenAI / Claude)
  eval.py              # Kernel evaluation via flashinfer-bench API
  utils.py             # Shared utilities & data helpers
prompt/
  proposer_prompt.py   # Kernel proposal prompt
  tuner_prompt.py      # Kernel tuning prompt (str_replace edits)
config/
  config_iterative.yaml   # Iterative agent config
  config_evolve.yaml      # Evolve agent config
  config_mini_test.yaml   # Quick smoke test config
  tasks_default.txt    # Default task list
  tasks_mini.txt       # Minimal task list for smoke test
datasets/              # FlashInfer-Trace / MLSys contest datasets
```

## Quick Start

### 0. Download the Dataset

```bash
mkdir datasets
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest datasets/mlsys26-contest
```

### 1. Set API Key

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
```

### 2. Run the Agent

```bash
python3 -m agent.main --config config/config_mini_test.yaml
```

## Agent Types

| Type | Description |
|------|-------------|
| **iterative** | Proposes an initial kernel, then repeatedly tunes it via str_replace edits |
| **evolve** | Proposes multiple kernels, maintains a recent + elite pool, samples and evolves |

## Config

Example (`config/config_iterative.yaml`):

```yaml
test_source: mlsys26-contest
agent_type: iterative
tasks_path: config/tasks_default.txt
gpu_name: B200
gpu_architecture: Blackwell
api_type: claude
model_name: claude-sonnet-4-5
total_steps: 25
```

Available configs:

| Config | Agent Type |
|--------|------------|
| `config_iterative.yaml` | Iterative Agent |
| `config_evolve.yaml` | Evolve Agent |
| `config_mini_test.yaml` | Quick smoke test |

Key parameters:

- `test_source`: `mlsys26-contest` or `flashinfer-trace`
- `agent_type`: `iterative` or `evolve`
- `tasks_path`: file listing op types / problem IDs to solve
- `total_steps`: number of iterations per task
- `api_type`: `openai` or `claude`
- `model_name`: LLM model to use

### Task List Format

One op type per line. Optionally specify kernel definition IDs after the op type:

```
dsa_paged
gdn
moe
gemm gemm_n128_k2048, gemm_n256_k4096
```

If no kernel definition IDs are given, all kernel definitions under that op type are loaded.

## Output

Results are saved under `outputs/`:

```
outputs/<agent_type>_<test_source>_<steps>_<timestamp>/
  config.yaml
  <op_type>_<problem_id>/
    reference_src.py
    proposal_0_1.py / tune_0_2.py / ...
    global_best_kernel_25.py
    global_best_metrics_25.json
```

## Resume

```bash
python3 -m agent.main \
  --config config/config_iterative.yaml \
  --resume_from outputs/iterative_mlsys26-contest_25_20260208-121400
```

Tasks with existing results are skipped; incomplete tasks continue from where they left off.

## License

See [LICENSE](LICENSE).
