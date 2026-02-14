# GPU Kernel Engineering: Complete Knowledge Base

> **Purpose**: This document encodes the accumulated knowledge of high-performance GPU kernel development in Triton and CUDA, distilled from production implementations, competition-winning optimizations, and hard-won debugging experience. Feed this to an LLM-based coding agent to dramatically improve its ability to generate, debug, and optimize GPU kernels.

---

## Table of Contents

1. [How GPU Kernels Actually Work](#1-how-gpu-kernels-actually-work)
2. [Triton Programming Model](#2-triton-programming-model)
3. [The 9 Bugs LLMs Always Make in Triton](#3-the-9-bugs-llms-always-make-in-triton)
4. [Optimization Methodology: Ladder, Not Random Search](#4-optimization-methodology)
5. [Memory System & Bandwidth Optimization](#5-memory-system--bandwidth-optimization)
6. [Tensor Cores & Compute Optimization](#6-tensor-cores--compute-optimization)
7. [Occupancy & Register Pressure](#7-occupancy--register-pressure)
8. [Inline PTX Assembly in Triton](#8-inline-ptx-assembly-in-triton)
9. [Profiling & Bottleneck Diagnosis](#9-profiling--bottleneck-diagnosis)
10. [Advanced Patterns: Persistent Kernels, Warp Specialization, TMA](#10-advanced-patterns)
11. [Autotune Configuration](#11-autotune-configuration)
12. [Numerical Stability](#12-numerical-stability)
13. [Gated Delta Net (GDN) / Linear Attention Kernels](#13-gated-delta-net--linear-attention-kernels)
14. [Reference Implementations](#14-reference-implementations)
15. [Structured Mutation Strategies](#15-structured-mutation-strategies)
16. [Pre-Submission Checklist](#16-pre-submission-checklist)

---

## 1. How GPU Kernels Actually Work

### Execution Hierarchy

```
GPU
 └─ Grid (all thread blocks launched by one kernel)
     └─ Thread Block / CTA (runs on one SM, shares SRAM)
         └─ Warp (32 threads, executes in lockstep SIMT)
             └─ Thread (has its own registers, executes one instruction stream)
```

### Memory Hierarchy (NVIDIA B200 / Blackwell)

| Level | Size | Bandwidth | Latency | Scope |
|-------|------|-----------|---------|-------|
| HBM (Global) | 192 GB | 8 TB/s | ~400 cycles | All SMs |
| L2 Cache | 128 MB | ~12 TB/s | ~200 cycles | All SMs |
| SRAM (Shared) | 256 KB/SM | ~33 TB/s | ~30 cycles | One thread block |
| Registers | 256 KB/SM (65536 × 32-bit) | Immediate | 0 cycles | One thread |

### Key Insight

The single most important thing to understand: **moving data is more expensive than computing on it**. A single HBM load costs ~400 cycles. A fused multiply-add costs 4 cycles. An f32 multiply-add on a tensor core costs effectively 0.25 cycles (amortized across the warp). This means:

- **Minimize HBM traffic** above all else
- **Maximize data reuse** — load once, use many times
- **Fuse operations** — never write intermediate results to HBM if you can keep them in SRAM or registers

### Roofline Model

Every kernel is either:
1. **Memory-bound**: Limited by how fast data can move between HBM and compute units. Optimization = reduce traffic, improve coalescing, increase cache hit rate.
2. **Compute-bound**: Limited by arithmetic throughput. Optimization = use tensor cores, reduce redundant math, increase arithmetic intensity.
3. **Latency-bound**: Neither memory nor compute is saturated. Optimization = increase occupancy, reduce synchronization, hide latency with pipelining.

Calculate arithmetic intensity:
```
Arithmetic Intensity = FLOPs / Bytes Moved
```

For B200: crossover point ≈ 280 FLOPs/byte (bf16 tensor core). Below this → memory-bound. Above → compute-bound.

---

## 2. Triton Programming Model

### Core Concepts

Triton operates on **tiles** (blocks of data), not individual elements. Each kernel instance (called a "program") processes one tile. The programmer specifies the tile shape and the operations; Triton handles the threading, memory coalescing, and instruction scheduling.

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles one tile of BLOCK_SIZE elements
    pid = tl.program_id(0)                          # which tile am I?
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # element offsets
    mask = offs < N                                  # boundary check

    a = tl.load(a_ptr + offs, mask=mask)             # load tile from HBM
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)         # store result to HBM
```

### Triton Compilation Pipeline

```
Python → Triton IR → MLIR (TritonGPU dialect) → LLVM IR → PTX → CUBIN
```

Each stage applies optimizations: tiling, vectorization, CSE, constant folding, coalescing, software pipelining. Understanding this pipeline helps predict what Triton will and won't do automatically.

### Key Triton Operations

| Operation | Purpose | When to Use |
|-----------|---------|-------------|
| `tl.load(ptr, mask, other)` | Load from global memory | Always use mask for variable dims |
| `tl.store(ptr, val, mask)` | Store to global memory | Always use mask for variable dims |
| `tl.dot(a, b)` | Matrix multiply (uses tensor cores) | Any matmul — 10-50× faster than manual |
| `tl.sum(x, axis)` | Reduction | Dot products, norms |
| `tl.max(x, axis)` | Max reduction | Softmax, online normalization |
| `tl.where(cond, a, b)` | Elementwise select | Masking, conditional ops |
| `tl.exp(x)` | Exponential | Softmax, sigmoid, gating |
| `tl.sigmoid(x)` | Logistic sigmoid | Gate computations |
| `tl.arange(0, N)` | Range tensor | Offset computation |
| `tl.program_id(axis)` | Block index | Grid decomposition |
| `tl.atomic_add(ptr, val)` | Atomic accumulation | Reductions across blocks |
| `tl.zeros(shape, dtype)` | Zero-initialized tensor | Accumulators (MUST be f32) |
| `tl.constexpr` | Compile-time constant | Tile sizes, num_warps |
| `.to(dtype)` | Type cast | bf16 ↔ f32 conversion |

### Grid Design

```python
# 1D grid: one dimension of parallelism
grid = (triton.cdiv(N, BLOCK_SIZE),)

# 2D grid: two dimensions (e.g., batch × heads)
grid = (batch_size, num_heads)

# 3D grid with tiles: batch × heads × tiles
grid = lambda meta: (B, H, triton.cdiv(D, meta['BLOCK_D']))
```

**Rule**: Each `tl.program_id(axis)` returns a different grid coordinate. Map your parallelism to grid dimensions.

---

## 3. The 9 Bugs LLMs Always Make in Triton

These bugs account for >80% of kernel failures when LLMs write Triton code. Check for every one of them.

### Bug 1: Missing Mask on `tl.load` / `tl.store`

```python
# ❌ WRONG — crashes on boundary tiles where offs >= N
x = tl.load(ptr + offs)

# ✅ CORRECT — safe on all tiles
x = tl.load(ptr + offs, mask=offs < N, other=0.0)
```

**When masks are needed**: Any time the tile might extend past the end of the tensor. This happens when `N % BLOCK_SIZE != 0`, which is almost always.

**When masks are NOT needed**: When the dimension is `tl.constexpr` and known to divide evenly.

### Bug 2: `tl.dot` with Wrong Dimensions

```python
# ❌ WRONG — tl.dot requires 2D inputs
a = tl.load(...)  # shape [128]
b = tl.load(...)  # shape [128]
c = tl.dot(a, b)  # ERROR: can't dot 1D vectors

# ✅ CORRECT — reshape to 2D
a = tl.load(...)[:, None]  # shape [128, 1]
b = tl.load(...)[None, :]  # shape [1, 128]
c = tl.dot(a, b)           # shape [128, 128] — outer product

# ✅ For inner product, use tl.sum:
dot_product = tl.sum(a * b, axis=0)
```

**Critical rule**: `tl.dot(A, B)` requires `A.shape = [M, K]` and `B.shape = [K, N]`, and **K must be a multiple of 16** for tensor cores.

### Bug 3: Accumulating in bf16

```python
# ❌ WRONG — bf16 accumulation causes catastrophic numerical drift
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16)
for i in range(K // BLOCK_K):
    a = tl.load(...)  # bf16
    b = tl.load(...)  # bf16
    acc += tl.dot(a, b)  # accumulates in bf16 — WRONG

# ✅ CORRECT — always accumulate in f32
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for i in range(K // BLOCK_K):
    a = tl.load(...)  # bf16 input
    b = tl.load(...)  # bf16 input
    acc += tl.dot(a, b)  # tl.dot accumulates in f32 automatically

# Cast to bf16 only at the final store:
tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask)
```

### Bug 4: Wrong Stride Calculation

```python
# ❌ WRONG — assumes contiguous layout
ptr = base_ptr + row * D + col  # only works if tensor is contiguous

# ✅ CORRECT — use actual strides from the tensor
ptr = base_ptr + row * stride_row + col * stride_col
```

**Rule**: Always pass strides as kernel arguments. Never assume contiguity. PyTorch tensors from `.transpose()`, `.reshape()`, `.expand()` are NOT contiguous.

### Bug 5: Grid Size Mismatch

```python
# ❌ WRONG — grid doesn't match kernel's program_id usage
grid = (batch_size,)  # 1D grid
@triton.jit
def kernel(...):
    bid = tl.program_id(0)
    hid = tl.program_id(1)  # ERROR: grid only has 1 dimension

# ✅ CORRECT — grid matches program_id dimensions
grid = (batch_size, num_heads)  # 2D grid
@triton.jit
def kernel(...):
    bid = tl.program_id(0)  # batch
    hid = tl.program_id(1)  # head
```

### Bug 6: Forgetting that `@triton.autotune` Wraps the Function

```python
# ❌ WRONG — the @autotune decorator consumes the config kwargs
@triton.autotune(configs=[triton.Config({'BLOCK': 128})], key=['N'])
@triton.jit
def kernel(ptr, N, BLOCK: tl.constexpr):
    pass

# When calling:
kernel[(grid,)](ptr, N, BLOCK=128)  # ERROR: BLOCK is set by autotune, not caller

# ✅ CORRECT — don't pass autotuned params in the launch
kernel[(grid,)](ptr, N)  # autotune handles BLOCK
```

### Bug 7: `tl.dot` K-dimension Not Multiple of 16

```python
# ❌ WRONG — K=5 can't use tensor cores
a = tl.load(...)  # [M, 5]
b = tl.load(...)  # [5, N]
c = tl.dot(a, b)  # ERROR or falls back to very slow scalar

# ✅ CORRECT — pad K to multiple of 16
BLOCK_K: tl.constexpr = 16  # minimum for tensor cores
# Load with padding (mask handles out-of-bounds)
```

### Bug 8: Using `torch.*` Inside a Triton Kernel

```python
# ❌ WRONG — torch ops don't work inside @triton.jit
@triton.jit
def kernel(...):
    x = tl.load(ptr)
    y = torch.sigmoid(x)  # ERROR: torch is not available in Triton

# ✅ CORRECT — use tl.* equivalents
@triton.jit
def kernel(...):
    x = tl.load(ptr)
    y = tl.sigmoid(x)
```

### Bug 9: NumPy or Python builtins Inside a Triton Kernel

```python
# ❌ WRONG
@triton.jit
def kernel(...):
    x = np.sqrt(val)     # ERROR
    y = math.exp(val)    # ERROR
    z = max(a, b)        # ERROR (Python builtin)

# ✅ CORRECT
@triton.jit
def kernel(...):
    x = tl.sqrt(val)
    y = tl.exp(val)
    z = tl.maximum(a, b)
```

---

## 4. Optimization Methodology

### The Ladder Approach (Not Random Search)

The single most important lesson from successful kernel optimization: **optimize in stages, each with one focused goal**. Don't try to write the perfect kernel in one shot.

```
Rung 1: CORRECTNESS    → Naive but correct baseline
Rung 2: ALGORITHMIC    → Right data decomposition, tiling strategy
Rung 3: TENSOR CORES   → Use tl.dot() for all matrix operations
Rung 4: AUTOTUNE       → Explore tile sizes, num_warps, num_stages
Rung 5: MEMORY ACCESS  → Coalesced loads, minimize HBM traffic
Rung 6: FUSION         → Combine separate passes into one kernel
Rung 7: MICRO-OPT      → Precompute invariants, inline PTX, packed ops
Rung 8: HW-SPECIFIC    → B200 TMA, persistent kernels, warp specialization
```

**Rules**:
- Never advance to the next rung until the current one's success criteria are met
- If a change breaks correctness, **revert immediately**
- Each generation should make ONE focused change, not multiple
- Good changes are kept; bad changes are discarded (monotonic progress)

### Why This Works

Random search (generate 100 kernels, hope one is good) fails because the search space is enormous and most random changes break correctness. The ladder constrains the search to a productive path where each step has a clear goal and measurable success criteria.

Wafer.ai's case study proved this: 7 optimization steps, each building on the last, achieving 9× speedup from naive to hand-tuned. The critical step 4 (DPP instructions) was only possible because steps 1-3 had already established a correct, structured baseline.

---

## 5. Memory System & Bandwidth Optimization

### Memory Coalescing

Warps of 32 threads issue memory requests together. If threads access consecutive addresses, the hardware combines them into one efficient transaction. If threads access scattered addresses, each requires a separate transaction.

```python
# ✅ COALESCED — threads access consecutive elements (column-major for row tiles)
offs = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(ptr + offs)  # All threads load adjacent addresses

# ❌ NON-COALESCED — threads access strided elements
offs = tl.arange(0, BLOCK) * stride  # stride > 1 → gaps between accesses
x = tl.load(ptr + offs)  # Each thread's address is far from neighbors
```

**Rule for 2D tiles**: The fastest-varying dimension should be the one that threads tile across. For a `[M, K]` matrix in row-major layout, tile the K dimension across threads.

### Data Reuse Strategies

```python
# Pattern: Load once, use many times
# If v is needed for all rows of S, load it once:
v = tl.load(v_ptr + offs_v, mask=mask_v)  # load once

for row_start in range(0, D, BLOCK_D):
    s_tile = tl.load(state_ptr + ...)     # load each row tile
    result = tl.dot(s_tile, v[:, None])    # reuse v across all row tiles
```

### Minimizing HBM Traffic

Calculate the minimum bytes that MUST be read/written:
```
Min bytes = sum(input sizes) + sum(output sizes)
Actual bytes = min bytes × (1 + overhead from non-coalesced access, redundant loads, etc.)
```

If actual >> min, you have optimization opportunity. Common fixes:
- Fuse operations to avoid intermediate writes
- Use shared memory to stage data for reuse
- Ensure coalesced access patterns
- Tile dimensions to maximize cache hits

### Vectorized Loads

When loading blocks of data, wider loads are more efficient:

```python
# Triton will automatically vectorize when:
# 1. BLOCK_SIZE is a multiple of 4 (for 128-bit loads)
# 2. Data is contiguous and aligned
# 3. The access pattern is simple (ptr + arange)

# Help Triton vectorize:
BLOCK_SIZE: tl.constexpr = 128  # multiple of 4
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offs, mask=offs < N)  # Triton will use 128-bit loads
```

---

## 6. Tensor Cores & Compute Optimization

### When to Use Tensor Cores

Tensor cores perform matrix multiply-accumulate at 10-50× the rate of regular CUDA cores. Use them for ANY operation that can be expressed as `C = A @ B + C`:

```python
# ✅ Use tl.dot for all matrix multiplications
output = tl.dot(A, B)  # Uses tensor cores automatically

# ❌ Don't manually implement matmul with tl.sum
# This is 10-50× slower:
output = tl.sum(A[:, :, None] * B[None, :, :], axis=1)
```

### Tensor Core Requirements

- `A` shape: `[M, K]` where K is multiple of 16
- `B` shape: `[K, N]` where K is multiple of 16
- Input dtype: bf16, fp16, tf32, fp8, int8
- Accumulator: always f32 (automatic)
- Minimum tile: 16×16

### Converting Operations to `tl.dot`

Many operations can be restructured as matrix multiplications:

```python
# Outer product: v ⊗ k = v @ k^T
outer = tl.dot(v[:, None], k[None, :])  # But v must be [M, 1], k must be [1, K]
# Better: if v=[M] and k=[K], reshape first
outer = tl.dot(v.reshape(M, 1), k.reshape(1, K))  # Only works if M, K are multiples of 16

# Batched dot product: sum(S * q, axis=-1)
# If S=[BV, K] and q=[K], this is S @ q:
output = tl.dot(S_tile, q[:, None])  # [BV, 1]
# Or reshape S to [BV, K] and q to [K, 1]
```

### FLOPS Calculation

For benchmarking, calculate theoretical FLOPS:
```python
# GEMM: 2 * M * N * K (multiply + add per output element)
# Elementwise: num_elements * ops_per_element
# Reduction: num_elements * ops_per_element

# Compare against peak:
# B200 bf16 tensor core: 2,250 TFLOPS
# B200 f32 CUDA cores: ~140 TFLOPS
# H100 bf16 tensor core: 990 TFLOPS
```

---

## 7. Occupancy & Register Pressure

### Why Occupancy Matters

Occupancy = fraction of the SM's warp slots that are filled. Higher occupancy gives the scheduler more warps to switch between, hiding memory latency.

```
Registers per SM: 65,536 (B200) or 65,536 (H100)
Max threads per SM: 2,048 (64 warps)
Max warps per SM: 64

Occupancy = active_warps / max_warps

If your kernel uses 128 registers/thread and 256 threads/block:
- Registers needed: 128 × 256 = 32,768
- Blocks per SM: 65,536 / 32,768 = 2
- Active threads: 2 × 256 = 512
- Active warps: 512 / 32 = 16
- Occupancy: 16 / 64 = 25%
```

### Register Pressure Management

```python
# Check register usage: compile kernel and inspect metadata
compiled = kernel[(grid,)](args, num_warps=8)
print(f"Registers: {compiled.metadata.num_regs}")  # target: < 128

# Strategies to reduce registers:
# 1. Smaller tile sizes (fewer values in flight)
# 2. More warps (same block, more threads → fewer regs/thread available)
# 3. Recompute instead of storing (trade compute for registers)
# 4. Sequential processing instead of parallel
```

### Shared Memory Limits

```
B200: 256 KB per SM (max 228 KB per block)
H100: 228 KB per SM

Shared memory per block = tile_size_bytes × num_tiles_in_flight

Example: 3 pipeline stages of [128, 128] f32 tiles:
= 3 × 128 × 128 × 4 bytes = 192 KB → fits in B200 SRAM
```

### The Occupancy-vs-Tile-Size Tradeoff

Larger tiles → more reuse, fewer HBM accesses, but higher register pressure → lower occupancy.
Smaller tiles → higher occupancy, better latency hiding, but more HBM accesses.

**Rule of thumb**: Start with the largest tile that maintains ≥25% occupancy, then tune.

---

## 8. Inline PTX Assembly in Triton

When Triton's compiler doesn't generate optimal instructions, inject PTX directly using `tl.inline_asm_elementwise()`. This is the escape hatch for specific instruction selection without dropping to full CUDA.

### API

```python
(result,) = tl.inline_asm_elementwise(
    asm="rcp.approx.ftz.f32 $0, $1;",  # PTX instruction string
    constraints="=r,r",                  # output, input register constraints
    args=[input_tensor],                 # Triton tensor arguments
    dtype=[tl.float32],                  # output dtype(s)
    is_pure=True,                        # no side effects
    pack=1,                              # elements per invocation
)
```

### Constraint String

```
"=r"    → output 32-bit register
"r"     → input 32-bit register
"=r,r"  → 1 output, 1 input
"=r,=r,r,r" → 2 outputs, 2 inputs

Placeholders in asm:
$0 = first output
$1 = second output (or first input if only 1 output)
$2 = next argument, etc.
```

### Pack Parameter

Controls how many elements are processed per invocation:

| pack | Element size | Use case |
|------|-------------|----------|
| 1 | 32-bit (f32, i32) | Standard operations |
| 2 | 16-bit (f16, bf16) → packed into 32-bit | Use f16x2/bf16x2 instructions |
| 4 | 8-bit (fp8, int8) → packed into 32-bit | Quantization, byte packing |

### Useful PTX Patterns

#### Fast Approximate Reciprocal
```python
# Standard Triton division generates div.full.f32 (~20 cycles)
# rcp.approx.ftz.f32 is ~4 cycles (loses ~1 ULP precision)
(reciprocal,) = tl.inline_asm_elementwise(
    asm="rcp.approx.ftz.f32 $0, $1;",
    constraints="=r,r",
    args=[denominator],
    dtype=[tl.float32],
    is_pure=True,
    pack=1,
)
result = numerator * reciprocal
```

#### Fast Approximate Exponential
```python
# ex2.approx computes 2^x; for e^x, scale by 1/ln(2) = 1.442695
(exp_val,) = tl.inline_asm_elementwise(
    asm="""
    {
        .reg .f32 scaled;
        mul.f32 scaled, $1, 0f3FB8AA3B;  // x * 1/ln(2) = 1.442695
        ex2.approx.ftz.f32 $0, scaled;
    }
    """,
    constraints="=r,r",
    args=[x],
    dtype=[tl.float32],
    is_pure=True,
    pack=1,
)
```

#### Packed bf16x2 FMA
```python
# Process two bf16 values in one instruction
(result,) = tl.inline_asm_elementwise(
    asm="fma.rn.bf16x2 $0, $1, $2, $3;",
    constraints="=r,r,r,r",
    args=[a_bf16, b_bf16, c_bf16],
    dtype=[tl.bfloat16],
    is_pure=True,
    pack=2,  # two bf16 packed into one 32-bit register
)
```

#### FP4 Quantization (Blackwell)
```python
# Convert two f32 values to fp4 e2m1, packed into 8 bits
x_e2m1x2 = tl.inline_asm_elementwise(
    asm="""
    {
        .reg .b8 tmp<4>;
        cvt.rn.satfinite.e2m1x2.f32 tmp0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 tmp1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 tmp2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 tmp3, $8, $4;
        mov.b32 $0, {tmp0, tmp1, tmp2, tmp3};
    }
    """,
    constraints="=r,r,r,r,r,r,r,r,r",
    args=x_blocks_split,
    dtype=tl.int8,
    is_pure=True,
    pack=4,
)
```

### Caveats

- **Architecture-specific**: PTX may behave differently on different GPU generations
- **Precision loss**: `approx` variants sacrifice ~1 ULP — always verify correctness
- **Compiler boundary**: The compiler can't optimize across inline ASM blocks
- **Elementwise only**: No shared memory ops, no warp-level control, no synchronization
- **Silent failures**: Wrong constraint strings produce wrong results without errors

---

## 9. Profiling & Bottleneck Diagnosis

### Interpreting Performance Data

```
Given metrics from NCU or estimated from ISA analysis:

IF achieved_bandwidth_pct > 60% AND achieved_compute_pct < 30%:
    → MEMORY BOUND
    → Actions: reduce HBM traffic, improve coalescing, fuse ops, use shared memory

IF achieved_compute_pct > 50% AND achieved_bandwidth_pct < 40%:
    → COMPUTE BOUND
    → Actions: use tensor cores, reduce redundant math, increase tile size

IF BOTH < 40%:
    → LATENCY BOUND or LOW OCCUPANCY
    → Actions: check register pressure, increase occupancy, add prefetching

IF register spills detected (local memory loads > 0):
    → REGISTER SPILLS — immediate priority
    → Actions: reduce tile size, use fewer accumulators, simplify inner loop
```

### Hardware Targets (B200)

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| HBM Bandwidth | >5.6 TB/s (70%) | >4 TB/s (50%) | <3.2 TB/s (40%) |
| BF16 Tensor TFLOPS | >1125 (50%) | >675 (30%) | <450 (20%) |
| Occupancy | >50% | >25% | <15% |

### Hardware Targets (H100)

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| HBM Bandwidth | >2.4 TB/s (75%) | >1.6 TB/s (50%) | <1.0 TB/s |
| BF16 Tensor TFLOPS | >500 (50%) | >300 (30%) | <200 (20%) |
| Occupancy | >50% | >25% | <15% |

### ISA-Level Analysis (Static, No GPU Required)

Even without running the kernel, analyze the generated code:

1. **Count instruction types**:
   - HMMA/DMMA: tensor core ops (good — this is the actual work)
   - FFMA: scalar float multiply-add (bad if should be tensor core)
   - LDG/STG: global loads/stores (minimize these)
   - LDS/STS: shared loads/stores (fine — shared memory is fast)
   - BAR: synchronization barriers (minimize — serialization points)

2. **Check for spills**: Any `LOCAL` loads/stores mean register spills to slow local memory

3. **Compute:Memory ratio**: Count compute instructions vs memory instructions
   - Ratio > 10 → compute-bound (optimize for tensor cores)
   - Ratio < 2 → memory-bound (optimize for bandwidth)

4. **Tensor core usage**: If HMMA count = 0 and the kernel does matmul → tl.dot is missing or broken

### Triton Source-Level Static Analysis

Before even compiling, check for these patterns in the source:

```python
# Check for common issues:
has_mask = "mask=" in source or "mask =" in source
has_tl_dot = "tl.dot" in source
has_autotune = "@triton.autotune" in source
bf16_accumulator = "tl.zeros" in source and "bfloat16" in source
torch_in_kernel = "torch." in source  # inside @triton.jit

# Count operations:
load_count = source.count("tl.load")
store_count = source.count("tl.store")
dot_count = source.count("tl.dot")
```

---

## 10. Advanced Patterns

### Persistent Kernels

Instead of launching one kernel per work item, launch a persistent kernel where each SM processes multiple work items sequentially. The advantage: data in SRAM persists across work items.

```python
@triton.jit
def persistent_kernel(work_list_ptr, num_work_items, ...):
    # Each program processes multiple work items
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)  # = num_SMs (or close to it)

    # Work-stealing loop
    for work_idx in range(pid, num_work_items, num_programs):
        # Load work item parameters
        batch_id = tl.load(work_list_ptr + work_idx * 2)
        head_id = tl.load(work_list_ptr + work_idx * 2 + 1)

        # Process — data stays in SRAM between iterations
        ...
```

**When to use**: When there's data that can be reused across work items (e.g., state matrices, weight matrices) and the per-item work is too small to fully utilize an SM.

### Software Pipelining (num_stages)

Overlap data loading with computation by loading the next tile while computing on the current one:

```python
@triton.autotune(configs=[
    triton.Config({...}, num_stages=2),  # 1 compute + 1 prefetch
    triton.Config({...}, num_stages=3),  # 1 compute + 2 prefetch
    triton.Config({...}, num_stages=4),  # 1 compute + 3 prefetch
], key=[...])
```

```
num_stages=1: Load tile → Compute → Load tile → Compute → ...
num_stages=2: Load tile N+1 → Compute tile N → Load tile N+2 → Compute tile N+1
num_stages=3: Load N+2 → Load N+1 → Compute N → Load N+3 → ...
```

More stages = better latency hiding, but more SRAM consumed for staging buffers. Calculate:
```
SRAM per stage = tile_M × tile_K × sizeof(dtype) + tile_K × tile_N × sizeof(dtype)
Total SRAM = num_stages × SRAM_per_stage

# Must fit in: B200 = 256 KB, H100 = 228 KB
```

### Warp Specialization

Different warps within a thread block perform different roles:

```
Thread Block:
┌── TMA/Load Warp ────── MMA Warps (×6) ──── Epilogue Warp ───┐
│   Loads next data      Matrix multiply      Writes results    │
│   from HBM → SRAM      on current data      from SRAM → HBM  │
└──────── Pipeline barriers synchronize the handoffs ──────────┘
```

In Triton, approximate warp specialization via:
- `num_stages` for load/compute overlap
- Separate load and compute phases within the kernel
- Multiple kernel launches with different roles (less efficient)

### TMA (Tensor Memory Accelerator) — Blackwell/Hopper

TMA performs bulk asynchronous copies from HBM to SRAM without using SM compute resources. The SM issues a TMA descriptor and continues computing while data arrives.

In Triton, TMA is implicitly used when:
- `num_stages > 1` (software pipelining)
- Tile accesses are regular and aligned

### Grouped Problem Handling (Variable-Length Sequences)

When processing batches of different-sized problems (e.g., variable-length sequences):

```python
# Precompute: linearize all tiles across all groups
# tile_to_group[i] = which group does tile i belong to?
# tile_to_local[i] = what's the local tile index within that group?

for tile_idx in range(my_start_tile, my_end_tile):
    group_id = tile_to_group[tile_idx]

    if group_id != prev_group_id:
        # Group changed — update pointers, load new parameters
        ptr = group_ptrs[group_id]
        size = group_sizes[group_id]
        prev_group_id = group_id

    # Process tile using current group's pointers
    local_tile = tile_to_local[tile_idx]
    ...
```

Key optimization: only update tensormap/pointers when the group actually changes, not every tile.

---

## 11. Autotune Configuration

### What to Tune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],  # re-tune when these dimensions change
)
```

### Configuration Space Guidelines

| Parameter | Range | Notes |
|-----------|-------|-------|
| BLOCK_M | 32, 64, 128, 256 | Powers of 2. Larger = more reuse, more registers |
| BLOCK_N | 32, 64, 128, 256 | Same. Match to the "wide" dimension |
| BLOCK_K | 16, 32, 64, 128 | Must be ≥16 for tensor cores. Larger = better reuse |
| num_warps | 2, 4, 8, 16 | More warps = more parallelism within block |
| num_stages | 1, 2, 3, 4 | More stages = better pipelining, more SRAM |

### Rules of Thumb

- **Minimum 5 configs** for meaningful exploration
- **Key dimensions**: Always include the dimensions that vary at runtime
- **Start conservative**: Begin with smaller tiles, verify correctness, then increase
- **Memory-bound kernels**: Prefer larger tiles, more stages
- **Compute-bound kernels**: Prefer more warps, balanced tiles
- **Small problems**: Smaller tiles to avoid underutilizing SMs

### For Decode Kernels (Single Token)

```python
# Decode has seq_len=1, so one dimension is tiny
# Tile the state/embedding dimension instead
configs=[
    triton.Config({'BV': 32}, num_warps=4, num_stages=2),
    triton.Config({'BV': 64}, num_warps=4, num_stages=2),
    triton.Config({'BV': 64}, num_warps=8, num_stages=2),
    triton.Config({'BV': 128}, num_warps=8, num_stages=1),
]
```

---

## 12. Numerical Stability

### Golden Rules

1. **All accumulators must be float32**, regardless of input/output dtype
2. **Cast to lower precision only at the final store**
3. **Use `tl.float32` for all intermediate arithmetic** involving sums, products, or recursive updates
4. **Softplus**: Use `softplus(x) = x + log(1 + exp(-|x|))` for numerical stability (avoids overflow for large x)
5. **Sigmoid**: Stable for all x, but `sigmoid(x) ≈ 0` for `x < -10` and `≈ 1` for `x > 10` — test boundary behavior
6. **Log-domain arithmetic**: When multiplying many small numbers, work in log domain to avoid underflow

### Mixed Precision Pattern

```python
# Inputs: bf16 (from PyTorch)
# Internal arithmetic: f32 (for stability)
# Output: bf16 (to PyTorch)

k = tl.load(k_ptr + offs, mask=mask).to(tl.float32)  # bf16 → f32
v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)  # bf16 → f32

# All computation in f32
state = alpha * state + dt * tl.dot(v[:, None], k[None, :])  # f32
output = tl.dot(state, q[:, None])  # f32 accumulation

# Only cast at store
tl.store(out_ptr + offs, output.to(tl.bfloat16), mask=mask)
```

### Softplus Implementation

```python
# ❌ WRONG — exp overflows for large x
def softplus(x):
    return tl.log(1 + tl.exp(x))  # exp(100) = inf

# ✅ CORRECT — numerically stable
def softplus(x):
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    # This is stable for both large positive and negative x
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # simple version

    # Or the more precise version:
    # abs_x = tl.abs(x)
    # return abs_x + tl.log(1.0 + tl.exp(-abs_x))
```

---

## 13. Gated Delta Net / Linear Attention Kernels

### The Recurrence

GDN's decode step (single token update):

```
Inputs per token t:
  q_t ∈ R^K           — query (K = key_dim, typically 128)
  k_t ∈ R^K           — key
  v_t ∈ R^V           — value (V = val_dim, typically 256)
  α ∈ R               — decay gate (sigmoid of learned A_log)
  β_t ∈ R             — erase gate (sigmoid of learned b)
  dt_t ∈ R            — delta step (softplus of learned param + bias)
  scale ∈ R           — 1/sqrt(K)

State: S_t ∈ R^{V×K}  — the recurrent state (float32, NOT bf16)

Update:
  S_t = α · S_{t-1} · (I - β_t · k_t · k_t^T) + dt_t · v_t · k_t^T

Decomposed into 3 operations:
  1. Decay:       S = α · S_{t-1}
  2. Erase:       S = S - β_t · (S @ k_t) ⊗ k_t^T    (Householder-like)
  3. Delta write:  S = S + dt_t · v_t ⊗ k_t^T          (rank-1 update)

Output:
  o_t = scale · S_t @ q_t
```

### GVA (Grouped Value Attention) Head Mapping

GDN uses **more value heads than key/query heads** (e.g., 32 V-heads, 16 Q/K-heads). Multiple V-heads share one Q/K head:

```python
# GVA ratio = num_v_heads / num_q_heads
gva_ratio = H_v // H_q  # typically 2

# Map value head → query/key head
i_hq = i_hv // gva_ratio  # integer division, NOT i_hv * gva_ratio

# In the kernel: use i_hq to index q, k, alpha, beta
# Use i_hv to index v, state, output
```

**This mapping is the #1 source of bugs in LLM-generated GDN kernels.**

### State Layout

The state `S ∈ R^{V×K}` can be stored in two layouts:

```
K-last (recommended):  S[batch, v_head, v_dim, k_dim]
  - K dimension is contiguous → coalesced loads when computing S @ k
  - Matches the outer product v ⊗ k^T (write pattern)

V-last (not recommended): S[batch, v_head, k_dim, v_dim]
  - K dimension is strided → non-coalesced loads
  - But matches S @ q computation pattern
```

**Use K-last layout.** The outer product `v_t ⊗ k_t^T` is the most frequent operation, and K-last makes it coalesced.

### Fused Decode Kernel Design

```python
# Grid: (batch, v_head, v_tiles)
# Each block handles one [BV, K] tile of the state matrix

pid_batch = tl.program_id(0)
pid_vhead = tl.program_id(1)
pid_vtile = tl.program_id(2)

# GVA: map v_head → qk_head
qk_head = pid_vhead // gva_ratio

# Load inputs (small — fit in registers)
q = load(q_ptr, qk_head)      # [K]
k = load(k_ptr, qk_head)      # [K]
v = load(v_ptr, pid_vhead)     # [BV] (tiled)
alpha, beta, dt = load_gates(qk_head)

# Precompute (saves per-element work in the tile loop)
ak = alpha * beta * k          # for erase term
dk = dt * k                    # for write term

# Load state tile [BV, K]
state = load_state(pid_batch, pid_vhead, pid_vtile)  # f32!

# Fused update (all 3 ops in one pass):
sk = sum(state * k, axis=1)             # S @ k → [BV]
state = alpha * state                    # decay
state = state - sk[:, None] * ak[None, :] # erase
state = state + v[:, None] * dk[None, :]  # write

# Output
output = sum(state * q, axis=1) * scale  # S_new @ q → [BV]

# Store
store_state(state)                        # f32
store_output(output.to(bf16))             # bf16
```

### Chunkwise Parallel (Prefill) Design

For processing entire sequences (not just one token), use WY decomposition:

```
Chunk the sequence into blocks of C tokens (C=64 typical).

Within each chunk: the product of Householder matrices
  P = ∏_t (I - β_t · k_t · k_t^T)
can be written in WY form:
  P = I - W · Y^T

where W, Y ∈ R^{K×C} are built incrementally:
  Y[:, t] = k_t
  W[:, t] = β_t · (I - W[:,:t] · Y[:,:t]^T) · k_t

The intra-chunk attention is then a causal matrix multiply with
modified attention scores, which is fully parallel within the chunk.

The inter-chunk state scan is sequential but only O(L/C) steps.
```

---

## 14. Reference Implementations

### GDN Decode (Annotated)

```python
@triton.autotune(
    configs=[
        triton.Config({'BV': 32}, num_warps=4, num_stages=2),
        triton.Config({'BV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BV': 64}, num_warps=8, num_stages=2),
        triton.Config({'BV': 128}, num_warps=8, num_stages=1),
    ],
    key=['V', 'K'],
)
@triton.jit
def gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, output_ptr,
    alpha_ptr, beta_ptr, dt_ptr, scale_ptr,
    stride_sb, stride_sh, stride_sv, stride_sk,  # state strides
    B, H_q, H_v, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr,
):
    # Grid: (batch, v_head, v_tiles)
    i_b = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_tv = tl.program_id(2)

    # CRITICAL: GVA head mapping (v_head → q/k head)
    i_hq = i_hv // (H_v // H_q)

    # Load small inputs (q, k, v, gates) — fit in registers
    offs_k = tl.arange(0, K)
    q = tl.load(q_ptr + i_b * H_q * K + i_hq * K + offs_k).to(tl.float32)
    k = tl.load(k_ptr + i_b * H_q * K + i_hq * K + offs_k).to(tl.float32)

    offs_v = i_tv * BV + tl.arange(0, BV)
    v_mask = offs_v < V
    v = tl.load(v_ptr + i_b * H_v * V + i_hv * V + offs_v, mask=v_mask, other=0.0).to(tl.float32)

    alpha = tl.load(alpha_ptr + i_hq).to(tl.float32)
    beta = tl.load(beta_ptr + i_b * H_q + i_hq).to(tl.float32)
    dt = tl.load(dt_ptr + i_b * H_v + i_hv).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)

    # Precompute for efficiency
    ak = alpha * beta * k   # erase coefficient
    dk = dt * k              # write coefficient

    # Load state tile [BV, K] — ALWAYS float32
    s_ptrs = state_ptr + i_b * stride_sb + i_hv * stride_sh + \
             offs_v[:, None] * stride_sv + offs_k[None, :] * stride_sk
    s_mask = v_mask[:, None]
    state = tl.load(s_ptrs, mask=s_mask, other=0.0)

    # Fused state update: decay + erase + write
    sk = tl.sum(state * k[None, :], axis=1)              # [BV]: S @ k
    state = alpha * state                                  # decay
    state = state - sk[:, None] * ak[None, :]              # erase (Householder)
    state = state + v[:, None] * dk[None, :]               # write (delta rule)

    # Output: S_new @ q
    output = tl.sum(state * q[None, :], axis=1) * scale   # [BV]

    # Store state (f32) and output (bf16)
    tl.store(s_ptrs, state, mask=s_mask)
    tl.store(output_ptr + i_b * H_v * V + i_hv * V + offs_v,
             output.to(tl.bfloat16), mask=v_mask)
```

**Key design choices annotated**:
1. **Grid = (B, H_v, cdiv(V, BV))**: One block per state tile
2. **GVA mapping via integer division**: `i_hq = i_hv // (H_v // H_q)`
3. **Precomputed `ak` and `dk`**: Avoids redundant per-element multiplication
4. **Fused update**: All three operations (decay, erase, write) in one pass over the state tile
5. **f32 state, bf16 I/O**: State arithmetic always in f32, inputs/outputs in bf16
6. **K-last layout**: K dimension contiguous for coalesced access in `S @ k`
7. **Autotune over BV**: Different tile sizes for different state dimensions

---

## 15. Structured Mutation Strategies

When optimizing an existing kernel, apply these strategies ONE AT A TIME (most impactful first):

### 1. tile_size_sweep (Highest Impact)
Try different BLOCK sizes. This is almost always the first thing to try.
```
Current BV=32 → try 64, 128
Current BLOCK_K=32 → try 64, 128
Key: larger tiles = more reuse but more registers
```

### 2. fuse_operations
If the kernel has multiple passes over the same data, combine them.
```
Before: kernel1(load state, compute S@k) → kernel2(load state again, update)
After: one kernel that loads state once and does everything
```

### 3. vectorize_loads
When bandwidth utilization < 60%, loads may not be vectorized.
```
Ensure BLOCK sizes are multiples of 4
Ensure access patterns are contiguous
```

### 4. fuse_output
Compute the output (S@q) during the same pass as the state update, instead of in a separate kernel.

### 5. increase_warps
If memory-bound with low occupancy, more warps help hide latency.
```
Current num_warps=4 → try 8, 16
```

### 6. software_pipelining
Increase num_stages to overlap loads with computation.
```
Current num_stages=1 → try 2, 3
Check: does the increased SRAM usage still fit? (stages × tile_size < SRAM limit)
```

### 7. multi_head_per_block
When per-head work is too small to fill an SM, process multiple heads per block.

### 8. use_tensor_cores
If any matrix operation is done with `tl.sum(a * b, axis=...)`, replace with `tl.dot(a, b)`.

### 9. precompute_invariants
Move loop-invariant computations outside the main loop.
```
Before: for t: ... alpha * beta * k_t ...  (recomputes alpha*beta each iteration)
After: ab = alpha * beta; for t: ... ab * k_t ...
```

### 10. shared_memory_tiling
For data accessed by multiple threads in the block, load into shared memory first.

---

## 16. Pre-Submission Checklist

Before declaring a kernel "done", verify every item:

### Correctness
- [ ] Passes on batch_size=1
- [ ] Passes on non-power-of-2 batch sizes (3, 7, 13)
- [ ] Passes on minimum dimensions (1 head, seq_len=1)
- [ ] Passes on maximum dimensions
- [ ] Passes with non-divisible tile boundaries
- [ ] No NaN or Inf in outputs
- [ ] Max absolute error < tolerance (typically 1e-2 for bf16)
- [ ] Tested with large input values (state ~1000, gates saturated)
- [ ] Tested with small input values (dt near zero)

### Performance
- [ ] Has @triton.autotune with ≥5 configurations
- [ ] Uses tl.dot() for all matrix operations (tensor cores)
- [ ] Accumulates in float32, stores in bf16
- [ ] All tl.load/tl.store have masks for variable dimensions
- [ ] No register spills (check with NCU or ISA analysis)
- [ ] Achieves >50% of memory bandwidth peak (memory-bound) or >30% of compute peak (compute-bound)

### Code Quality
- [ ] No torch.* or numpy inside @triton.jit
- [ ] No .item() calls inside kernel
- [ ] Strides passed as arguments, not hardcoded
- [ ] Grid dimensions match tl.program_id usage
- [ ] Entry point function exists and is correctly named
- [ ] All constexpr parameters are typed as tl.constexpr

### GDN-Specific
- [ ] GVA head mapping: `qk_head = v_head // (H_v // H_q)` — NOT multiply
- [ ] State is float32, never bf16
- [ ] K-last state layout (K dimension contiguous)
- [ ] Softplus computed stably (no exp overflow for large inputs)
- [ ] Gate precomputation: `ak = alpha * beta * k` outside tile loop
- [ ] Both decay AND erase AND write are applied (all three, not just two)

---

## Appendix: Quick Reference Card

### Triton ↔ PTX ↔ SASS Mapping

| Triton | PTX | SASS | Notes |
|--------|-----|------|-------|
| `tl.dot(a, b)` | `mma.sync` | `HMMA/DMMA` | Tensor core |
| `tl.load(ptr)` | `ld.global` | `LDG` | Global load |
| `tl.store(ptr, v)` | `st.global` | `STG` | Global store |
| `tl.exp(x)` | `ex2.approx` + scale | `MUFU.EX2` | Transcendental |
| `tl.sigmoid(x)` | multiple ops | `MUFU.*` + arith | ~6 instructions |
| `a * b + c` | `fma.rn.f32` | `FFMA` | Fused multiply-add |
| `tl.atomic_add` | `atom.global.add` | `ATOMS` | Atomic |
| `barrier` | `bar.sync` | `BAR.SYNC` | Block sync |

### Memory Sizes Quick Reference

```
bf16: 2 bytes    f32: 4 bytes    f64: 8 bytes
int8: 1 byte     int16: 2 bytes  int32: 4 bytes

State [128, 128] f32:  64 KB   (fits in B200 SRAM with room)
State [256, 128] f32: 128 KB   (fits in B200 SRAM, tight)
State [256, 256] f32: 256 KB   (DOES NOT FIT — must tile)

Q/K vector [128] bf16: 256 bytes (fits in registers easily)
V vector [256] bf16:   512 bytes (fits in registers easily)
```

### Common Grid Patterns

```python
# 1D: elements
grid = (triton.cdiv(N, BLOCK),)

# 2D: batch × head
grid = (B, H)

# 3D: batch × head × tile
grid = lambda meta: (B, H, triton.cdiv(D, meta['BLOCK_D']))

# Persistent: num_SMs
grid = (num_SMs,)  # each SM loops over its work
```
