# Proposer prompt: generate a new kernel proposal

import re

PROBLEM_STATEMENT = """## Problem Statement

You write custom kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace.
You may make the decision to replace some operators with custom kernels and leave others unchanged.
You may replace multiple operators with custom implementations,
consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu),
or algorithmic changes (such as online softmax). You are only limited by your imagination.

The goal is to get the best performance for the given architecture.

"""

POOL_PROMPT = """## Kernel Pool

Here is some community developed kernels and their runtime metrics, you can reference them to generate your own kernels.

DO NOT directly copy the kernels, you should generate your own kernels with some major modifications,
such as changing the operator, changing the algorithm, fixing the correctness errors, etc.

<Kernels and their Runtime Metrics>
{pool_kernels_and_metrics}
</Kernels and their Runtime Metrics>

Now, please generate your own kernels with the community developed kernels as reference:
"""


TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

Requirements:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your Triton implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations or markdown formatting

Generate complete, runnable code only - no framework will add device handling wrapper code.

Generate the implementation:
"""


STRUCTURED_ENGINEERING_TRACE_PROMPT = """
## Structured Engineering Trace Protocol

Before code generation, produce an auditable JSON trace that includes:
- optimization ladder rung (1-8) and single optimization focus
- 9-bug prevention audit (masking, dot dims, f32 accumulators, strides, grid, autotune launch args, K alignment, tl vs torch ops, tl ops vs python builtins)
- skills/library trace with section citations
- hardware bottleneck hypothesis and memory/register plan
- contrastive optimization decisions in the format:
  "Compared to [alternative], I chose [decision] because [hardware reason]. Without this, [degradation] due to [bottleneck]."
- numerical stability and verification plan

Output contract:
1) First output a single JSON object trace.
2) Then output ONLY kernel code (no markdown explanation).
3) Do not include any additional prose after code.

All technical claims in the JSON trace must cite sections as `Per Section X.Y`.
"""


def generate_proposer_prompt(pool_prompt: str = None, task_params: dict = None):
    prompt = PROBLEM_STATEMENT

    # Fill template with task_params
    required_keys = set(re.findall(r"\{(\w+)\}", TRITON_PROMPT))
    for key in required_keys:
        if key not in task_params:
            raise ValueError(f"Missing required parameter: {key}")

    prompt += TRITON_PROMPT.format(**task_params)
    prompt += STRUCTURED_ENGINEERING_TRACE_PROMPT
    if pool_prompt is not None:
        prompt += pool_prompt
    return prompt


def generate_pool_prompt_single(
    pool_kernels: list,
    pool_metrics: list,
    *,
    proposal_ids: list[int] | None = None,
):
    if len(pool_kernels) == 0:
        return ""
    pool_kernels_and_metrics = "\n\n".join(
        [
            (
                f"\n### {i}-th kernel"
                + (
                    f" (proposal_id={proposal_ids[i]})"
                    if proposal_ids is not None and i < len(proposal_ids)
                    else ""
                )
                + ":\n\n```python\n"
                + f"{kernel}\n"
                + "```\n\n"
                + f"### {i}-th metrics:\n{metric}"
            )
            for i, (kernel, metric) in enumerate(zip(pool_kernels, pool_metrics))
        ]
    )
    return POOL_PROMPT.format(pool_kernels_and_metrics=pool_kernels_and_metrics)


def generate_pool_prompt(
    *,
    kernel_pool: list,
    metrics_pool: list,
    kernel_pool_ids: list[int] | None = None,
    elite_kernel_pool: list | None = None,
    elite_metrics_pool: list | None = None,
    elite_pool_ids: list[int] | None = None,
):
    """
    Build a single Kernel Pool prompt that can include both a "recent/trajectory" pool
    and an "elite/best" pool. Any pool can be empty/None.
    """
    elite_kernel_pool = elite_kernel_pool or []
    elite_metrics_pool = elite_metrics_pool or []

    parts = []

    elite_part = generate_pool_prompt_single(
        elite_kernel_pool, elite_metrics_pool, proposal_ids=elite_pool_ids
    )
    if elite_part:
        inner = elite_part.split("<Kernels and their Runtime Metrics>")[1].split(
            "</Kernels and their Runtime Metrics>"
        )[0]
        parts.append(("## Context type: elite\n\n" + inner.strip()).strip())

    recent_part = generate_pool_prompt_single(
        kernel_pool, metrics_pool, proposal_ids=kernel_pool_ids
    )
    if recent_part:
        # Strip the outer POOL_PROMPT wrapper so we can merge into one wrapper.
        inner = recent_part.split("<Kernels and their Runtime Metrics>")[1].split(
            "</Kernels and their Runtime Metrics>"
        )[0]
        parts.append(("## Context type: recent\n\n" + inner.strip()).strip())

    if len(parts) == 0:
        return ""

    merged = "\n\n".join(parts)
    return POOL_PROMPT.format(pool_kernels_and_metrics=merged)
