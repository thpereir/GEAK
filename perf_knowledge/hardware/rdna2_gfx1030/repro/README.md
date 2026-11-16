# gfx1030 measurement script

The `[measured]` numbers in `../*.md` were taken with this script on a Radeon PRO V620 (gfx1030,
72 CU) on 2026-11-16, ROCm 7.13 from AMD's gfx103X pip index (TheRock), torch 2.11.0, Triton 3.6.0,
Python 3.14. The host also held an RX 9070 XT; the script targets torch device 0 (the V620) and
matches `rocminfo` / `amd-smi` to it by arch. Two full runs agreed within ~1%.

**Measure on a quiet box.** No other GPU work, low CPU load.

| Script | Backs | Page |
|---|---|---|
| `rdna2_probe.py` | sections `env` (identity, caches, clocks), `bw` (read bandwidth vs working set, warm and cold, read-read-write), `gemm` (`torch.mm` per dtype, `torch._int_mm`, which BLAS kernel ran), `valu`, `dot` (Triton `tl.dot` speed, correctness, instructions, VGPRs), `launch`, `noise`, `clock` (clock / power / temperature idle and under a 30 s fp16 GEMM), `isa` (which dot / matrix instructions assemble), `prof` (`rocprofv3` counters) | all pages |

```bash
python3 rdna2_probe.py --setup-venv            # installs ROCm + torch + triton into ~/rdna2-probe-venv, then probes
python3 rdna2_probe.py                         # in an environment that already has them
python3 rdna2_probe.py --only bw,gemm --quick  # a subset
```

It prints one JSON report and saves it as `rdna2_probe_<host>_<time>.json`. `--setup-venv` uses
`https://repo.amd.com/rocm/whl/gfx103X-all/` (override with `--index-url`); only the amdgpu kernel
driver must come from the OS.

## Claims with no script here
- The occupancy steps in `../occupancy.md` come from the compiler sweep in
  `../../../expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py` (`--compiler-sweep`), not
  from this probe.
- The `valu` numbers of the 2026-11-16 runs are not quoted anywhere: that kernel was not ALU-bound.
  The script now measures `valu` by the slope between two loop lengths and reports whether the time
  scales with the work; re-run it before quoting a vector FMA peak.
- 128 KB of LDS per WGP and the 60 / 80-CU members of gfx1030 are architecture facts, not measured.
