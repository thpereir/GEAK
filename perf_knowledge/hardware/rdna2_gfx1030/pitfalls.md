---
title: RDNA2 / gfx1030 (Radeon PRO V620) — pitfalls
kind: hardware
gens: [gfx1030]
dtypes: []
regimes: [both]
updated: 2026-11-16
sources:
  - ../../../kernel_workflow/knowledge/amd_rdna2.md
  - repro/rdna2_probe.py
---

# RDNA2 / gfx1030 — pitfalls

## Libraries
- **hipBLASLt has no gfx103x kernels, and torch tries it first.** `[measured]` without a
  preference, torch's GEMMs look for `hipblaslt/library/TensileLibrary_lazy_gfx1030.dat` and
  `kernels.*gfx1030*.hsaco`, which do not exist; ROCm's hipBLASLt ships only archs with a matrix core
  (gfx908 / 90a / 942 / 950, gfx110x / 115x / 120x). Set `TORCH_BLAS_PREFER_HIPBLASLT=0` or
  `torch.backends.cuda.preferred_blas_library("cublas")` (rocBLAS on ROCm), which does ship gfx103x
  kernels (`repro/rdna2_probe.py` `env`, `gemm`).
- **No library int8 GEMM through torch.** `[measured]` `torch._int_mm` fails with
  `HIPBLAS_STATUS_INVALID_VALUE` because its only path is hipBLASLt (`repro/rdna2_probe.py` `gemm`).
- **bf16 is the slow dtype.** `[measured]` rocBLAS bf16 GEMM runs at ~10 TFLOP/s against ~35.6 for
  fp16 and 18.8 for fp32 (`repro/rdna2_probe.py` `gemm`).

## Measurement
- **Device numbering differs between tools on a multi-GPU host.** `[measured]` with a V620 and an
  RX 9070 XT installed, torch device 0 was the V620 while `rocminfo`'s first GPU agent and
  `amd-smi -g 0` were the 9070 XT; clock samples taken from index 0 described the idle 9070 XT
  during a V620 load. Match by `gcnArchName` (`repro/rdna2_probe.py` `env`, `clock`).
- **Small transfers measure the launch.** `[measured]` single reads of 1–8 MB per launch take a
  constant ~10 µs per Triton launch, so their "bandwidth" grows with size ([memory.md](memory.md);
  `repro/rdna2_probe.py` `bw`, `launch`).
- **Power limit.** `[measured]` sustained loads run at the 250 W limit with a drifting clock
  ([clocks_power.md](clocks_power.md); `repro/rdna2_probe.py` `clock`).

## Profiling
- **No matrix-utilization counter** exists because there is no matrix unit; an MFMA% or WMMA%
  metric reading empty is expected, not a profiler failure.
- `[measured]` `rocprofv3` 1.3.0 lists 109 counters on the host, including `GL2C_HIT`, `GL2C_MISS`,
  `TCP_REQ`, `TCP_REQ_MISS`, `SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `SQC_LDS_BANK_CONFLICT`, `FETCH_SIZE`
  and `WRITE_SIZE`. The list was taken on the two-GPU host and was not split per agent
  (`repro/rdna2_probe.py` `prof`).
- `rocprof-compute` support on gfx1030 was not determined.

## Environment
- **Triton needs the Python headers.** `[measured]` Triton JIT-builds a C helper against `Python.h`
  on first use; without the `python3-dev` package every Triton kernel fails at its first launch
  (`repro/rdna2_probe.py`, which fetches the headers into its venv).
