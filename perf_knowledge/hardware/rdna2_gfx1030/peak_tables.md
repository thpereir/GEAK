---
title: RDNA2 / gfx1030 (Radeon PRO V620) — peak tables (measured)
kind: hardware
gens: [gfx1030]
dtypes: [fp32, fp16, bf16, int8]
regimes: [both]
updated: 2026-11-16
sources:
  - ../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md
  - repro/rdna2_probe.py
---

# RDNA2 / gfx1030 — peak tables (measured)

> Theoretical denominators are the `gfx1030` section of
> [`roofline/peaks.md`](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md); this card
> does not restate their derivation. It is what was measured on a Radeon PRO V620 (72 CU). Other
> gfx1030 parts (60 or 80 CU) share the ISA and were not measured.

## TL;DR
> rocBLAS GEMM tops out at **~35.6 TFLOP/s fp16, 18.8 fp32 and ~10 bf16**: fp16 is ~75% of the
> 47.4 TFLOP/s max-clock theoretical, bf16 is the slowest dtype on the part. DRAM delivers
> **~505 GB/s** to reads (99% of the 512 GB/s pin rate) and the 128 MB Infinity Cache **~1.5 TB/s**.

## Memory & cache peaks
`value @ gfx1030 (Radeon PRO V620), ROCm 7.13 (pip SDK), torch 2.11.0, Triton 3.6.0, 2026-11-16`,
two runs; ranges cover both:

| Level | Measured | Theoretical | Source |
|---|---:|---:|---|
| DRAM, read | 489.6–504.8 GB/s (256 MB–2 GB, cold) | 512 GB/s | `[measured]` `repro/rdna2_probe.py` `bw`, read-based evict, per-call timed |
| DRAM, read-read-write | 449.1–453.2 GB/s (64 MB–1 GB per buffer) | 512 GB/s | `[measured]` `repro/rdna2_probe.py` `bw` copy rows and `noise` |
| Infinity Cache, read | 1441.6–1580.3 GB/s (64–128 MB, warm) | — | `[measured]` `repro/rdna2_probe.py` `bw`, batched timing |

The read kernel writes one int32 per program, so its bytes are reads only. Below 16 MB the warm
reads are launch-bound (see [clocks_power.md](clocks_power.md)), so the 4 MB L2 has no measured
bandwidth of its own here.

## Sustained reality
Dense square GEMM through `torch.mm` (rocBLAS; torch pinned off hipBLASLt), TFLOP/s, run 1 / run 2,
each a median of 5–10 GPU-event-timed calls ([`repro/rdna2_probe.py`](repro/rdna2_probe.py) `gemm`):

| Precision | 1024³ | 2048³ | 4096³ | 8192³ |
|---|---:|---:|---:|---:|
| fp32 | 10.9 / 11.8 | **18.8** / 18.8 | 18.3 / 18.3 | 17.5 / 17.5 |
| fp16 → fp32 | 5.7 / 5.7 | 32.7 / 32.7 | **35.4** / **35.8** | 33.2 / 33.2 |
| bf16 → fp32 | 7.9 / 9.1 | 9.9 / 9.8 | **10.1** / 10.1 | 9.8 / 9.8 |
| int8 (`torch._int_mm`) | — | — | unsupported | — |

- `[measured]` fp16 is ~75% of 47.4 TFLOP/s and fp32 ~79% of 23.7 at the 2.57 GHz maximum clock.
  Held for 30 s, fp16 4096³ settles at ~31 TFLOP/s with the clock at 2.10 GHz, where the theoretical
  is 38.7 ([clocks_power.md](clocks_power.md); `repro/rdna2_probe.py` `clock`).
- `[measured]` bf16 runs at under a third of fp16 (0.28×) and below fp32: there is no bf16 dot or packed bf16 FMA
  ([isa_notes.md](isa_notes.md); `repro/rdna2_probe.py` `gemm`, `isa`).
- `[measured]` `torch._int_mm` fails (`HIPBLAS_STATUS_INVALID_VALUE` from
  `hipblasLtMatmulAlgoGetHeuristic`): torch's int8 GEMM has only the hipBLASLt path, which ships no
  gfx103x kernels (`repro/rdna2_probe.py` `gemm`).
- `[measured]` 1024³ is too small to fill 72 CUs at these tile sizes; the fp16 kernel rocBLAS picks
  is `MT128x256x16` (`repro/rdna2_probe.py` `gemm`).

## Triton `tl.dot`
fp16 4096³, `tl.dot` with fp32 accumulate, run 1 / run 2 ([`repro/rdna2_probe.py`](repro/rdna2_probe.py) `dot`):

| Tile (BM×BN×BK) | num_warps | TFLOP/s | VGPRs | waves/SIMD |
|---|---:|---:|---:|---:|
| 64×64×32 | 4 | 18.3 / 18.4 | 136 | 7 |
| 128×128×32 | 8 | **25.4** / 25.4 | 201 | 4 |
| 128×64×64 | 4 | 20.2 / 20.2 | 217 | 4 |

`[measured]` every tile lowers to `v_dot2c_f32_f16`, max relative error 3.5e-4 against an fp32
reference; the best tile reaches ~72% of rocBLAS (`repro/rdna2_probe.py` `dot`).

## Not measured
- Vector FMA peak by microbenchmark: the probe's first `valu` kernel was not ALU-bound (identical
  times for different instruction mixes), so no VALU number is quoted; the GEMM rows above are the
  achieved ceilings. `rdna2_probe.py` now measures `valu` by slope with a scaling check.
- int8 through `v_dot4_i32_i8` (no library path to time).
