# Iterative Agent: propose initial kernel, then iteratively refine

import argparse
import json
import logging
import os
import shutil

from tqdm import tqdm

from agent.api import query_inference_server
from agent.eval import EvalResult, calculate_score
from agent.utils import extract_edits, extract_first_code, get_dataset_root, str_replace
from prompt.proposer_prompt import generate_pool_prompt, generate_proposer_prompt
from prompt.tuner_prompt import generate_tuner_prompt

logger = logging.getLogger(__name__)


def propose_step(
    ref_arch_src: str,
    inference_server: str,
    kernel_pool: list,
    metrics_pool: list,
    args: argparse.Namespace,
    *,
    context_ids: list[int] | None = None,
    elite_kernel_pool: list | None = None,
    elite_metrics_pool: list | None = None,
    elite_context_ids: list[int] | None = None,
):
    """Generate a new kernel proposal via the Proposer LLM."""
    pool_prompt = generate_pool_prompt(
        kernel_pool=kernel_pool,
        metrics_pool=metrics_pool,
        kernel_pool_ids=context_ids,
        elite_kernel_pool=elite_kernel_pool,
        elite_metrics_pool=elite_metrics_pool,
        elite_pool_ids=elite_context_ids,
    )
    proposer_prompt = generate_proposer_prompt(
        task_params=args.task_params,
        pool_prompt=pool_prompt,
    )
    proposer_output = query_inference_server(
        server=inference_server,
        model_name=args.model_name,
        prompt=proposer_prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    proposal_kernel = extract_first_code(proposer_output, ["python", "cpp"])
    proposal_metrics = args.eval_fn(
        kernel_code=proposal_kernel,
        task_id=getattr(args, "problem_id", None),
        dataset_root=get_dataset_root(args.test_source),
    )

    logs = {
        "proposer_prompt": proposer_prompt,
        "proposal_kernel": proposal_kernel,
        "proposal_metrics": proposal_metrics,
    }
    return proposal_kernel, proposal_metrics, logs


def refine_step(
    ref_arch_src: str,
    inference_server: str,
    previous_kernels: list,
    previous_metrics: list,
    args: argparse.Namespace,
):
    """Refine the latest kernel via the Tuner LLM (str_replace edits)."""
    tuner_prompt = generate_tuner_prompt(
        task_params=args.task_params,
        previous_kernels=previous_kernels,
        previous_metrics=previous_metrics,
        filter_wrong_attempts=getattr(args, "filter_wrong_attempts", False),
    )

    tuner_output = query_inference_server(
        server=inference_server,
        model_name=args.model_name,
        prompt=tuner_prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    tuned_kernel = previous_kernels[-1]
    edits = extract_edits(tuner_output)
    for old_str, new_str in edits:
        tuned_kernel = str_replace(tuned_kernel, old_str, new_str)

    tuned_metrics = args.eval_fn(
        kernel_code=tuned_kernel,
        task_id=getattr(args, "problem_id", None),
        dataset_root=get_dataset_root(args.test_source),
    )

    logs = {
        "tuner_prompt": tuner_prompt,
        "tuned_kernel": tuned_kernel,
        "tuned_metrics": tuned_metrics,
    }
    return tuned_kernel, tuned_metrics, logs


def load_from_logs(log_path: str):
    """
    Load previous kernels and metrics from log files (large_loop_id=0).

    Returns:
        (previous_kernels, previous_metrics, max_step,
         local_best_kernel, local_best_metric, local_best_score)
    """
    empty = ([], [], 0, None, None, -1.0)

    if not os.path.exists(log_path):
        logger.warning(f"Log path {log_path} does not exist")
        return empty

    # Scan for proposal_0_N.py and tune_0_N.py files
    step_files = []
    for f in os.listdir(log_path):
        for prefix in ("proposal_0_", "tune_0_"):
            if f.startswith(prefix) and f.endswith(".py"):
                try:
                    step_idx = int(f.split("_")[2].split(".")[0])
                    step_files.append((prefix.rstrip("_"), step_idx, f))
                except ValueError:
                    pass

    if not step_files:
        logger.warning(f"No step files found in {log_path}")
        return empty

    step_files.sort(key=lambda x: x[1])

    previous_kernels, previous_metrics = [], []
    local_best_kernel, local_best_metric, local_best_score = None, None, -1.0
    max_step = 0

    for step_type, step_idx, kernel_file in step_files:
        kernel_path = os.path.join(log_path, kernel_file)
        metrics_path = os.path.join(
            log_path, kernel_file.replace(".py", "_metrics.json")
        )
        if not os.path.exists(metrics_path):
            logger.warning(f"Metrics file {metrics_path} not found, skipping")
            continue

        with open(kernel_path, "r") as f:
            kernel = f.read()
        with open(metrics_path, "r") as f:
            metrics = EvalResult(**json.load(f))

        previous_kernels.append(kernel)
        previous_metrics.append(metrics)

        score = calculate_score(metrics)
        if score > local_best_score:
            local_best_score = score
            local_best_kernel = kernel
            local_best_metric = metrics

        max_step = max(max_step, step_idx)

    logger.info(
        f"Loaded {len(previous_kernels)} steps from {log_path}, "
        f"max_step={max_step}, local_best_score={local_best_score}"
    )
    return (
        previous_kernels,
        previous_metrics,
        max_step,
        local_best_kernel,
        local_best_metric,
        local_best_score,
    )


def copy_step_files(src_path: str, dst_path: str, max_step: int = 0):
    """Copy step files from source to destination (for resume)."""
    os.makedirs(dst_path, exist_ok=True)

    for step_idx in range(1, max_step + 1):
        for prefix in ["proposal", "tune"]:
            for suffix in [".py", "_metrics.json", "_prompt.txt"]:
                src_file = os.path.join(src_path, f"{prefix}_0_{step_idx}{suffix}")
                if os.path.exists(src_file):
                    dst_file = os.path.join(dst_path, f"{prefix}_0_{step_idx}{suffix}")
                    shutil.copy2(src_file, dst_file)
                    logger.debug(f"Copied {src_file} -> {dst_file}")


def _save_step(log_path, prefix, kernel, metrics, prompt_text):
    """Save kernel, metrics JSON, and prompt for a single step."""
    if log_path is None:
        return
    with open(os.path.join(log_path, f"{prefix}.py"), "w") as f:
        f.write(kernel)
    with open(os.path.join(log_path, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics.model_dump(), f)
    with open(os.path.join(log_path, f"{prefix}_prompt.txt"), "w") as f:
        f.write(prompt_text)


def run_iterative_loop(
    ref_arch_src: str,
    inference_server,
    initial_kernel: str,
    initial_metrics: EvalResult,
    args: argparse.Namespace,
    large_loop_id: int = 0,
    log_path: str = None,
):
    """Run the iterative refine loop: propose once, then refine repeatedly."""
    start_step = 0

    # Resume from existing logs if specified (only for iterative agent)
    if args.agent_type == "iterative" and args.resume_from is not None:
        (
            previous_kernels,
            previous_metrics,
            max_step,
            local_best_kernel,
            local_best_metric,
            local_best_score,
        ) = load_from_logs(args.resume_from)
        loaded_count = len(previous_kernels)
        previous_kernels = previous_kernels[-args.max_memory_round :]
        previous_metrics = previous_metrics[-args.max_memory_round :]
        start_step = max_step
        logger.info(
            f"Resumed from {args.resume_from}, loaded {loaded_count} steps "
            f"(keeping last {len(previous_kernels)} for memory), "
            f"will continue from step {start_step + 1}"
        )

        if log_path and log_path != args.resume_from:
            copy_step_files(args.resume_from, log_path, start_step)
    else:
        previous_kernels = [initial_kernel] if initial_kernel is not None else []
        previous_metrics = [initial_metrics] if initial_metrics is not None else []
        local_best_score = (
            calculate_score(initial_metrics) if initial_metrics is not None else -1.0
        )
        local_best_kernel = initial_kernel
        local_best_metric = initial_metrics

    for i in tqdm(
        range(start_step, args.refine_steps),
        desc=f"Iterative Agent on {args.level}_{args.problem_id}",
        initial=start_step,
        total=args.refine_steps,
    ):
        logger.debug(f"Running kernel {i+1} of {args.refine_steps}")

        if len(previous_kernels) == 0:
            # Propose initial kernel
            proposal_kernel, proposal_metrics, logs = propose_step(
                ref_arch_src, inference_server, [], [], args
            )
            previous_kernels.append(proposal_kernel)
            previous_metrics.append(proposal_metrics)
            local_best_kernel = proposal_kernel
            local_best_metric = proposal_metrics
            local_best_score = calculate_score(proposal_metrics)
            _save_step(
                log_path,
                f"proposal_{large_loop_id}_{i+1}",
                proposal_kernel,
                proposal_metrics,
                logs["proposer_prompt"],
            )
            continue

        # Refine step
        tuned_kernel, tuned_metrics, logs = refine_step(
            ref_arch_src, inference_server, previous_kernels, previous_metrics, args
        )

        _save_step(
            log_path,
            f"tune_{large_loop_id}_{i+1}",
            tuned_kernel,
            tuned_metrics,
            logs["tuner_prompt"],
        )

        previous_kernels.append(tuned_kernel)
        previous_metrics.append(tuned_metrics)

        if len(previous_kernels) > args.max_memory_round:
            previous_kernels.pop(0)
            previous_metrics.pop(0)

        score = calculate_score(tuned_metrics)
        if score > local_best_score:
            local_best_score = score
            local_best_kernel = tuned_kernel
            local_best_metric = tuned_metrics

    logger.debug(
        f"Local Best Score: {local_best_score}, Local Best Metric: {local_best_metric}"
    )
    return local_best_kernel, local_best_metric
