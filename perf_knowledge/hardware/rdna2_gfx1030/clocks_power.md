---
title: RDNA2 / gfx1030 (Radeon PRO V620) — clocks, power & measurement noise
kind: hardware
gens: [gfx1030]
dtypes: []
regimes: [both]
updated: 2026-11-16
sources:
  - repro/rdna2_probe.py
---

# RDNA2 / gfx1030 — clocks, power & measurement noise

## TL;DR
> The V620 is power-limited: a held fp16 GEMM reaches the 250 W limit at once and the engine clock
> drifts from 2.16 to 2.10 GHz as the card warms, ~18% under the 2.57 GHz maximum. Sustained GEMM
> throughput is ~12% under the short-burst figure. On a quiet host the noise floor is 2σ ≈ 0.1%.

## Concepts
- **Sustained clock.** `[measured]` `torch.mm` fp16 4096³ held for 30 s: engine clock 2160 MHz at
  3 s, 2100 MHz at 31 s; power 250 W in every 3-second sample (the board limit `amd-smi` reports);
  temperature 44 → 67 °C; throughput median 31.1 TFLOP/s against 35.4–35.8 in a burst
  (`repro/rdna2_probe.py` `clock`, `gemm`).
- **Idle.** `[measured]` 500 MHz engine clock and ~80 W reported at idle (`repro/rdna2_probe.py` `clock`).
- **Maximum clock.** `[measured]` `rocminfo` reports 2570 MHz, the clock the theoretical peaks in
  `peaks.md` assume (`repro/rdna2_probe.py` `env`).
- **Noise floor.** `[measured]` 512 MB-per-buffer torch.add, 10 calls per timed sample, 15 samples:
  CV 0.05–0.06%, 2σ 0.10–0.12% (`repro/rdna2_probe.py` `noise`).
- **Launch cost.** `[measured]` 3.5 µs per back-to-back small kernel (host and GPU-event timing
  agree), 20.7 µs for a launch followed by a synchronize (`repro/rdna2_probe.py` `launch`).

## The levers
1. Compare kernels interleaved (A, B, A, B) rather than in blocks: the clock moves with temperature
   over tens of seconds.
2. Quote burst and sustained throughput separately for anything that runs for seconds.
3. Time kernels shorter than ~100 µs in batches between one pair of events.

## Pitfalls
- Clock and power must be read from the GPU being measured. On a multi-GPU host `amd-smi -g 0` can be
  a different card than torch device 0 ([pitfalls.md](pitfalls.md)).

## Verify
```bash
python3 repro/rdna2_probe.py --only clock,noise,launch
```

## Sources
- [`repro/README.md`](repro/README.md) — conditions and commands.
