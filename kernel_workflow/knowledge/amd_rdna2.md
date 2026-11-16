# AMD RDNA2 Hardware Reference (gfx103x) — DETECT THE BOX FIRST

Hardware facts for RDNA2 parts. Read this when `rocminfo` reports a `gfx103*` arch.

Currently tabulated: **RDNA2**, `gfx1030` — measured on a Radeon PRO V620 (72 CU). RDNA2 has **no
matrix core**: no WMMA, no MFMA. Matrix work runs on the vector ALU through dot and packed-math
instructions.

## 0. Detect THIS box first

`gfx1030` covers parts with 60 to 80 CUs (the V620 and RX 6800 XT have 72, the W6800 and RX 6800
have 60, the RX 6900 XT has 80). Read the CU count from `rocminfo` rather than from the arch. Scope
the query to the gfx agent; the CPU agent is listed first and its "Compute Unit" line is the core count:

```bash
rocminfo | awk '/Name:.*gfx103/{f=1} f&&/Compute Unit:/{print $3; exit}'
```

On a host with more than one GPU, HIP's device order (what torch and the kernels see) is not
`rocminfo`'s agent order or `amd-smi`'s index: a V620 was torch device 0 while `rocminfo` and
`amd-smi -g 0` described an RX 9070 XT. Match the device by `gcnArchName`, and read clocks and power
from the `amd-smi` index whose `TARGET_GRAPHICS_VERSION` is that arch.

| Part | Arch | CU | Memory | Peak BW | Matrix core |
|---|---|---|---|---|---|
| Radeon PRO V620 | RDNA2 / `gfx1030` | 72 | 32 GB GDDR6, 256-bit | 512 GB/s | none |

## 1. Machine model

| | RDNA2 (`gfx1030`) |
|---|---|
| Wavefront | 32 lanes (the compute default); wave64 is assemblable (`+wavefrontsize64`) |
| Matrix ISA | none: no `v_wmma_*`, no `v_mfma_*` |
| fp16 / int math | `v_dot2_f32_f16` / `v_dot2c_f32_f16` (fp16 pairs into fp32), `v_pk_fma_f16`, `v_fma_mix_f32`, `v_dot4_i32_i8`, `v_dot8_i32_i4` |
| bf16 | no bf16 dot and no packed bf16 FMA (`v_dot2_f32_bf16` does not assemble): bf16 is converted and computed as fp32 |
| Register file | 1024 VGPR/SIMD, 256 addressable per wave, granule 16, up to 16 waves/SIMD; no AGPRs |
| SIMDs | 2 per CU; CUs pair into a WGP (torch reports 36 "multiprocessors" on the 72-CU V620) |
| LDS | 64 KB max per workgroup; 128 KB per WGP |
| Caches (V620) | 16 KB L1 per CU, 128 KB L1 per shader array, 4 MB L2, 128 MB Infinity Cache, 128 B line |
| Memory | GDDR6, 256-bit, 16 Gbps (512 GB/s pin) |

Occupancy steps (compiler-derived, `amd_occupancy.py --compiler-sweep --arch gfx1030`; gfx1031 and
gfx1036 match): ≤64 VGPRs 16 waves/SIMD, ≤80 12, ≤96 10, ≤112 9, ≤128 8, ≤144 7, ≤160 6, ≤192 5,
≤256 4. This is the RDNA3 shape with a smaller file: 128 VGPRs is 8 waves here, 10 on gfx11.

## 2. Applying these numbers

Measured on a Radeon PRO V620 (72 CU), ROCm 7.13 (pip SDK), torch 2.11.0, Triton 3.6.0, 2026-11-16,
two runs within 1% of each other:

- **Memory.** DRAM **~505 GB/s** read (99% of the 512 GB/s pin rate), ~450 GB/s read-read-write. The
  128 MB Infinity Cache serves **~1.44–1.58 TB/s** for a working set of 64–128 MB, and a set of
  160 MB already falls back to ~466 GB/s: keep the hot set under 128 MB.
- **GEMM ceilings (rocBLAS, short bursts).** fp32 **18.8**, fp16 **35.4–35.8**, bf16 **~10** TFLOP/s.
  Use **fp16, not bf16**: bf16 is 3.5x slower than fp16 and slower than fp32 here. Theoretical at
  the 2.57 GHz maximum clock is 23.7 TFLOP/s fp32 and 47.4 fp16.
- **Sustained load is power-limited.** A held fp16 GEMM sits at the 250 W limit from the first
  second; the engine clock falls from 2.16 to 2.10 GHz as the card warms, and throughput settles at
  ~31 TFLOP/s, ~12% under the burst figure.
- **Triton.** `tl.dot` on fp16 lowers to `v_dot2c_f32_f16` and reaches **25.4 TFLOP/s** at best
  (128×128×32 tile, 8 warps, 201 VGPRs = 4 waves/SIMD), ~72% of rocBLAS. MFMA / WMMA tile tables and
  the MFMA-only Triton knobs (`matrix_instr_nonkdim`, `kpack`) do not apply.
- **Libraries.** hipBLASLt ships no gfx103x kernels, and torch tries it first: route torch to rocBLAS
  with `TORCH_BLAS_PREFER_HIPBLASLT=0` or `torch.backends.cuda.preferred_blas_library("cublas")`.
  `torch._int_mm` has only the hipBLASLt path and fails, so int8 GEMM needs a kernel on
  `v_dot4_i32_i8` (its rate was not measured).
- **Launch and noise.** ~3.5 µs per back-to-back small torch op and ~10 µs per Triton launch, so a
  kernel that moves a few MB measures the launch, not the cache. The idle noise floor is 2σ ≈ 0.1%.
- **Profiling.** `rocprofv3` lists the cache, LDS and SQ counters (`GL2C_HIT` / `GL2C_MISS`,
  `TCP_REQ` / `TCP_REQ_MISS`, `SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `SQC_LDS_BANK_CONFLICT`,
  `FETCH_SIZE`, `WRITE_SIZE`, `SQ_WAVES`). There is no matrix unit, so there is no matrix-utilization
  counter to read. `rocprof-compute` was not verified on this part.
- **Roofline denominators** are in `e2e_workflow/knowledge/analysis_skills/roofline/peaks.md`
  (`gfx1030` section), valid only at 72 CU; any other gfx103x part is a hard unknown.
- **Measured long form.** `$SKILL_DIR/../perf_knowledge/hardware/rdna2_gfx1030/` (start at `arch.md`)
  holds the measured tables, their conditions, and the probe that re-takes every number.
