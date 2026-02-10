# Tuner prompt: refine an existing kernel with str_replace edits

import re

from agent.eval import EvalResult

_HARDWARE_INFO = """## Hardware Information

Here is some information about the underlying hardware that you should keep in mind:

- The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

"""

PROBLEM_STATEMENT = """## Problem Statement

You tune the custom Triton kernels in the given architecture to get better performance. The architecture is the reference architecture, and the custom kernels are the previous kernels you have generated.

"""

TASK_INSTRUCTION = """## Task Instruction
You are given the following kernel definition:

```python
{definition}
```

The input shapes can be found in the input of the architecture, and the dtype is {dtype_str}.

The tuning metrics contain the following information:

* **Compiled**: whether the kernel is compiled successfully
* **Error Message**: the compilation or runtime error encountered by the kernel (if any)
* **Correctness**: whether the kernel is correct
* **Runtime**: the runtime of the kernel
* **Fast_p**: compared with the standard PyTorch implementation, how much speedup the customized kernel achieves, calculated as *standard time / custom time*.

### Test Conditions

* **Correctness Test:**
  First, verify the correctness of the custom kernels by running each kernel with the specified input shapes and data types.

* **Warm-up Phase:**
  Warm up the kernel by running it three times with the same input shapes and data types.
  The runtime during the warm-up phase is **not** included in the final runtime, so you may include auto-tuning code as part of this phase.

* **Performance Test:**
  Finally, test the performance of the custom kernels by running each kernel 100 times with the same input shapes and data types.
  The runtime from these test runs **is included** in the final performance measurement.

### Goal

- Perform small, localized updates to code in the last version of the custom kernels with the str_replace command to correct the correctness errors or improve the performance of the custom kernels, but keep the high-level architecture unchanged.
When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation.
- You should ensure the `old_str` matches exactly with the file content, otherwise the str_replace tool will fail.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: You should prefer to send all edits in a single message with multiple calls rather than multiple messages with a single call each.

#### **Output Format**:

You should output all the edits in a single message with multiple call. Each call should be a single edit as follows with id `1` to `n`.
- For each edit, you should provide the reasoning for the edit in the <reasoning_i> block, followed by the old code block in the <old_str_i> block, followed by the new code block in the <new_str_i> block.
- You should ensure the `old_str_i` matches exactly with the file content, otherwise the str_replace tool will fail.

Example output format:

<reasoning_1>
// reasoning for the edit 1
...
</reasoning_1>
<old_str_1>
// old code block (must match exactly)
...
</old_str_1>
<new_str_1>
// new code block
...
</new_str_1>

...

<reasoning_n>
// reasoning for the edit n
...
</reasoning_n>
<old_str_n>
// old code block (must match exactly)
...
</old_str_n>
<new_str_n>
// new code block
...
</new_str_n>

#### **Previous Kernels and Metrics:**

Previously, you have generated the following custom kernels and got the following runtime metrics:

<Previous Kernels and Metrics>
{previous_kernels_and_metrics}
</Previous Kernels and Metrics>

"""


def _extract_format_keys(template: str):
    """Extract format keys from a template string."""
    return set(re.findall(r"\{(\w+)\}", template))


def _is_correct_metric(metric) -> bool:
    """Check if a metric indicates correctness. Handles both EvalResult objects and string representations."""
    if isinstance(metric, EvalResult):
        return metric.correct
    elif isinstance(metric, str):
        # Check string representation for correctness
        return "correctness=True" in metric or '"correctness": true' in metric.lower()
    return False


def generate_tuner_prompt(
    previous_kernels: list[str] = None,
    previous_metrics: list[str] = None,
    filter_wrong_attempts: bool = False,
    task_params: dict = None,
):
    # Filter out wrong attempts if requested
    if filter_wrong_attempts:
        filtered_pairs = [
            (kernel, metric)
            for kernel, metric in zip(previous_kernels, previous_metrics)
            if _is_correct_metric(metric)
        ]
        if filtered_pairs:
            previous_kernels, previous_metrics = zip(*filtered_pairs)
            previous_kernels, previous_metrics = list(previous_kernels), list(
                previous_metrics
            )
        else:
            previous_kernels, previous_metrics = [], []

    previous_kernels_and_metrics_str = "\n".join(
        [
            f"\n### {i}-th attempt: \n\n```python\n{kernel}\n```\n\n### {i}-th Runtime Metrics:\n{metric}"
            for i, (kernel, metric) in enumerate(
                zip(previous_kernels, previous_metrics)
            )
        ]
    )

    # Extract required parameters from task prompt template
    required_keys = _extract_format_keys(TASK_INSTRUCTION)

    # Build format dict: use task_params if provided, otherwise fall back to original parameters
    format_dict = {}
    if task_params is not None:
        format_dict.update(task_params)

    # Fall back to original parameters for missing keys
    for key in required_keys:
        if key not in format_dict:
            if key == "previous_kernels_and_metrics":
                format_dict[key] = previous_kernels_and_metrics_str
            else:
                raise ValueError(f"Missing required parameter: {key}")

    prompt = PROBLEM_STATEMENT
    gpu_name = task_params.get("gpu_name")
    gpu_arch = task_params.get("gpu_architecture")
    if gpu_name and gpu_arch:
        prompt += _HARDWARE_INFO.format(gpu_name=gpu_name, gpu_architecture=gpu_arch)
    prompt += TASK_INSTRUCTION.format(**format_dict)
    return prompt
