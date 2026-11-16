---
title: RDNA2 / gfx1030 (Radeon PRO V620) — ISA notes
kind: hardware
gens: [gfx1030]
dtypes: [fp32, fp16, bf16, int8]
regimes: [both]
updated: 2026-11-16
sources:
  - ../../../kernel_workflow/knowledge/amd_rdna2.md
  - repro/rdna2_probe.py
---

# RDNA2 / gfx1030 — ISA notes

## TL;DR
> No matrix instructions. fp16 products go through `v_dot2_f32_f16` (two fp16 pairs into an fp32
> accumulator) or `v_pk_fma_f16`; int8 / int4 have `v_dot4_i32_i8` / `v_dot8_i32_i4`; bf16 has
> nothing of its own. Wave32 is the compute default.

## Concepts
- **What assembles.** `[measured]` ROCm LLVM `llvm-mc -mcpu=gfx1030` accepts `v_dot2_f32_f16`,
  `v_dot2c_f32_f16`, `v_dot4_i32_i8`, `v_dot4c_i32_i8`, `v_dot8_i32_i4`, `v_fma_mix_f32` and
  `v_pk_fma_f16`, and rejects `v_dot2_f32_bf16`, `v_wmma_f32_16x16x16_f16` and
  `v_mfma_f32_16x16x16f16` (`repro/rdna2_probe.py` `isa`).
- **Wave size.** `[measured]` `rocminfo` reports a 32-lane wavefront and 32 waves per CU; the
  assembler also accepts `+wavefrontsize64`; Triton kernels compile as wave32
  (`repro/rdna2_probe.py` `env`, `isa`, `dot`).
- **What Triton emits.** `[measured]` `tl.dot` on fp16 lowers to `v_dot2c_f32_f16` (1024–4096 per
  kernel at the probed tiles), with `v_perm_b32` / `v_pack_b32_f16` to pair operands
  (`repro/rdna2_probe.py` `dot`).
- **What rocBLAS runs.** `[measured]` `torch.mm` fp16 on gfx1030 launches a Tensile kernel built for
  the ISA (`Cijk_Ailk_Bljk_HHS_BH_MT128x256x16_…_ISA1030_…`) once torch is pinned to rocBLAS
  (`repro/rdna2_probe.py` `gemm`).

## The levers
1. Shape fp16 inner loops for `v_dot2_f32_f16`: operands in adjacent pairs along K, fp32 accumulate.
2. For int8, write the kernel on `v_dot4_i32_i8` (no library path exists, see
   [pitfalls.md](pitfalls.md)).
3. Convert bf16 inputs to fp16 when the range allows; bf16 math costs as much as fp32 plus the
   conversion ([peak_tables.md](peak_tables.md)).

## Pitfalls
- Any guidance written for MFMA or WMMA (tile tables, `matrix_instr_nonkdim`, `kpack`, matrix-unit
  utilization) does not apply: there is no matrix unit to tune or to count.

## Verify
```bash
echo 'v_dot2_f32_f16 v0, v1, v2, v3' | /opt/rocm/llvm/bin/llvm-mc -arch=amdgcn -mcpu=gfx1030 -show-encoding
python3 repro/rdna2_probe.py --only isa,dot
```

## Sources
- [`repro/README.md`](repro/README.md) — conditions and commands.
