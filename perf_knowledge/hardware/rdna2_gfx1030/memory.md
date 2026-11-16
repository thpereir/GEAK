---
title: RDNA2 / gfx1030 (Radeon PRO V620) — memory hierarchy behaviour
kind: hardware
gens: [gfx1030]
dtypes: [fp32, fp16, bf16, int8]
regimes: [both]
updated: 2026-11-16
sources:
  - ../../../kernel_workflow/knowledge/amd_rdna2.md
  - repro/rdna2_probe.py
---

# RDNA2 / gfx1030 — memory hierarchy behaviour

## TL;DR
> Four cache levels in front of 32 GB of GDDR6: 16 KB L1 per CU, 128 KB L1 per shader array, 4 MB L2
> and a 128 MB Infinity Cache. A read working set up to 128 MB streams at ~1.5 TB/s; at 160 MB it
> drops to the DRAM rate, ~470 GB/s. Plan around the 128 MB edge.

## Concepts
- **Hierarchy.** `[measured]` `rocminfo` reports L1 16 KB, L2 4096 KB, L3 (Infinity Cache) 131072 KB
  and a 128 B cache line; `amd-smi` adds two 128 KB level-1 caches with 4 instances each, one per
  shader array (4 shader engines × 2 arrays) (`repro/rdna2_probe.py` `env`).
- **The Infinity Cache edge.** `[measured]` warm read bandwidth by working set: 1.21–1.33 TB/s at
  16–48 MB, 1.44 TB/s at 64 MB, 1.53–1.58 TB/s at 112–128 MB, then 464–466 GB/s at 160 MB
  (`repro/rdna2_probe.py` `bw`). The drop between 128 and 160 MB is 3.4×.
- **DRAM.** `[measured]` cold reads reach 504.8 GB/s at 2 GB, 99% of the 512 GB/s pin rate of a
  256-bit bus at 16 Gbps; the measured rate rules out the 14 Gbps (448 GB/s) configuration
  (`repro/rdna2_probe.py` `bw`). Mixed traffic is lower: torch.add over two inputs and an output
  sustains 449–453 GB/s.

## The levers
1. Keep the per-kernel hot set (weights + activations re-read in the loop) under 128 MB.
   `[measured]` a 16 MB-per-buffer read-read-write add (48 MB in flight) runs at 1.40–1.43 TB/s, a
   64 MB-per-buffer one at 452 GB/s (`repro/rdna2_probe.py` `bw` copy rows).
2. Batch small kernels: below a few MB the launch, not the memory system, is the limit
   ([clocks_power.md](clocks_power.md)).

## Pitfalls
- Reading a 1–8 MB buffer once per launch measures launch overhead: `[measured]` 1 MB reads at
  ~100 GB/s and 8 MB at ~825 GB/s, i.e. a constant ~10 µs per Triton launch whatever the size
  (a small torch op launches in ~3.5 µs) (`repro/rdna2_probe.py` `bw`, `launch`). L2 bandwidth needs
  a kernel that re-reads in-kernel.
- Evict with a read, not a write, before a cold measurement, as the probe does: a write leaves dirty
  lines whose write-back lands on the timed kernel (same mechanism as on gfx1151).

## Verify
```bash
python3 repro/rdna2_probe.py --only bw,noise
```

## Sources
- [`repro/README.md`](repro/README.md) — conditions and commands.
