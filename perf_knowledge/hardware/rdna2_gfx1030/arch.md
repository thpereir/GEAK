---
title: RDNA2 / gfx1030 (Radeon PRO V620) — architecture overview
kind: hardware
gens: [gfx1030]
dtypes: [fp32, fp16, bf16, int8]
regimes: [both]
updated: 2026-11-16
sources:
  - ../../../kernel_workflow/knowledge/amd_rdna2.md
  - ../../expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json
  - ../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md
  - repro/README.md
---

# RDNA2 / gfx1030 (Radeon PRO V620) — architecture overview

> Workflow agents should prefer [`kernel_workflow/knowledge/amd_rdna2.md`](../../../kernel_workflow/knowledge/amd_rdna2.md),
> which is the summary; this directory is the measured detail behind it. The machine model,
> occupancy and theoretical peaks live in `amd_rdna2.md`,
> [`hw_constants.json`](../../expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json)
> and [`roofline/peaks.md`](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md), and are
> not repeated here. Occupancy is in [occupancy.md](occupancy.md), pitfalls in [pitfalls.md](pitfalls.md).

## TL;DR
> gfx1030 ships with 60 to 80 CUs; every number in this directory was measured on a 72-CU Radeon PRO
> V620. RDNA2 has **no matrix core**: fp16 matrix work runs on `v_dot2` and packed FMA, bf16 has no
> arithmetic of its own. Measured, not theoretical: DRAM **~505 GB/s** to reads, Infinity Cache
> **~1.5 TB/s** below 128 MB, rocBLAS fp16 GEMM **~35.6 TFLOP/s** (bf16 only ~10), engine clock
> **2.10–2.16 GHz at the 250 W limit** under sustained load.

## The one-screen cheat sheet
| Fact | Value (Radeon PRO V620) | Detail |
|---|---|---|
| Machine model (wave32, no matrix core, dot ISA, VGPR file, LDS) | see `amd_rdna2.md` §1 | [isa_notes.md](isa_notes.md) |
| L1 / L2 / Infinity Cache / line | 16 KB per CU / 4 MB / 128 MB / 128 B | [memory.md](memory.md) |
| DRAM, measured | 489–505 GB/s read (≥256 MB), ~450 read-read-write (pin 512) | [peak_tables.md](peak_tables.md) |
| Infinity Cache, measured | 1.44–1.58 TB/s read at 64–128 MB, ~466 GB/s at 160 MB | [memory.md](memory.md) |
| GEMM, measured (rocBLAS, burst) | fp32 18.8, fp16 35.4–35.8, bf16 ~10 TFLOP/s (theoretical 23.7 / 47.4 / 23.7) | [peak_tables.md](peak_tables.md) |
| Triton `tl.dot` fp16 | `v_dot2c_f32_f16`, best 25.4 TFLOP/s | [isa_notes.md](isa_notes.md) |
| Engine clock under sustained GEMM | 2.16 → 2.10 GHz, pinned at 250 W | [clocks_power.md](clocks_power.md) |
| Noise floor, quiet box | 2σ ≈ 0.1% | [clocks_power.md](clocks_power.md) |

## Concepts
- **No matrix core.** Peaks follow the vector ALU: fp16 runs at twice fp32 through `v_dot2_f32_f16`
  and `v_pk_fma_f16`; bf16 does not ([isa_notes.md](isa_notes.md)).
- **About 75% of the max-clock theoretical** is the measured fp16 GEMM ceiling in short bursts, and
  the card is power-limited when held ([peak_tables.md](peak_tables.md), [clocks_power.md](clocks_power.md)).
- **A 128 MB Infinity Cache** triples read bandwidth for any working set that fits ([memory.md](memory.md)).

## The levers
1. Compute in fp16 with fp32 accumulation; never pick bf16 on this part ([peak_tables.md](peak_tables.md)).
2. Keep the hot working set under 128 MB ([memory.md](memory.md)).
3. Size register use to the occupancy steps of the 1024-VGPR file ([occupancy.md](occupancy.md)).
4. Time A/Bs interleaved and in batches on a quiet box ([clocks_power.md](clocks_power.md)).

## Pitfalls
- Libraries and tools assume a matrix core or a newer part: hipBLASLt, `torch._int_mm`, mixed-GPU
  device numbering ([pitfalls.md](pitfalls.md)).
