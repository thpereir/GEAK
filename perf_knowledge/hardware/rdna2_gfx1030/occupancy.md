---
title: RDNA2 / gfx1030 (Radeon PRO V620) — occupancy
kind: hardware
gens: [gfx1030]
dtypes: []
regimes: [both]
updated: 2026-11-16
sources:
  - ../../expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json
  - ../../expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py
---

# RDNA2 / gfx1030 — occupancy

The occupancy table is the `gfx1030` entry of
[`hw_constants.json`](../../expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json),
computed by [`amd_occupancy.py`](../../expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py)
(`--arch gfx1030 --vgpr N`). The steps are compiler-derived, not measured on the card:
`amd_occupancy.py --compiler-sweep --arch gfx1030` with ROCm LLVM; gfx1031 and gfx1036 give the same
steps.

## TL;DR
> 1024-VGPR file per SIMD, granule 16, at most 16 waves/SIMD, 256 VGPRs addressable per wave, no
> AGPRs. ≤64 VGPRs 16 waves, ≤80 12, ≤96 10, ≤112 9, ≤128 8, ≤144 7, ≤160 6, ≤192 5, ≤256 4. The
> RDNA3 table (1536 VGPRs, granule 24) does not apply: at 128 VGPRs gfx11 runs 10 waves, gfx1030 8.

- The Triton `tl.dot` tiles in [peak_tables.md](peak_tables.md) sit at 136 VGPRs (7 waves) and
  201–217 VGPRs (4 waves); the fastest one is the 4-wave 128×128×32 tile.
- LDS: 64 KB per workgroup at most; 128 KB per WGP, shared by its two CUs.
