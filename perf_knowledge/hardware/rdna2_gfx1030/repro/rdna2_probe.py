#!/usr/bin/env python3
"""rdna2_probe.py — measure an RDNA2 GPU (e.g. Radeon PRO V620, gfx1030) for a GEAK hardware reference.

  python3 rdna2_probe.py              # full run, ~5-8 min; prints progress, then one JSON block
  python3 rdna2_probe.py --quick      # ~1 min smoke test (fewer sizes / reps)
  python3 rdna2_probe.py --skip clock,isa

  python3 rdna2_probe.py --setup-venv [DIR] [--quick]   # no ROCm/torch yet? see below

Needs: ROCm + a ROCm build of PyTorch (`python3 -c "import torch; print(torch.cuda.is_available())"`).
Optional: Triton (ships with torch ROCm wheels), rocminfo, amd-smi or rocm-smi, rocprofv3, llvm-mc.
Every section is independent: a missing tool or an unsupported feature is recorded, not fatal.

--setup-venv [DIR] installs all of that into a Python venv (default ~/rdna2-probe-venv) from AMD's
per-GPU-family pip index (TheRock; default https://repo.amd.com/rocm/whl/gfx103X-all/, override with
--index-url), then re-runs this probe inside it. Only the amdgpu kernel driver must come from the OS
(/dev/kfd and /dev/dri present, your user in the `render` group). Python 3.10-3.14. Re-running with
the same DIR reuses the venv; add --setup-only to stop after the install check.

Run it on a quiet box (no other GPU work, low CPU load), GPU 0 by default (set HIP_VISIBLE_DEVICES to
pick another). It only reads system state and allocates up to ~3 GB of GPU memory; nothing is written
except the JSON report in the current directory.

Sections (all by default):
  env      versions, rocminfo (gfx agent), amd-smi / rocm-smi static info, torch device properties
  bw       read bandwidth vs working set (L2 / Infinity Cache / DRAM, warm and cold), copy bandwidth
  gemm     torch.mm TFLOP/s for fp32 / fp16 / bf16 (+ torch._int_mm int8) and which BLAS kernel ran
  valu     vector ALU peak: fp32 FMA, packed fp16 FMA (Triton; slope over two loop lengths)
  dot      Triton tl.dot on this arch: does it compile, speed, which matrix/dot instructions it emits
  launch   empty-ish kernel launch latency (host and GPU-event timed)
  noise    run-to-run dispersion of a DRAM-bound kernel (the 2-sigma noise floor)
  clock    engine/memory clock, power, temperature: idle and under a sustained fp16 GEMM
  isa      which dot / matrix instructions the assembler accepts for this gfx (llvm-mc)
  prof     rocprofv3 presence and which cache / LDS / SQ counters exist
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument(
    "--quick", action="store_true", help="smoke test: fewer sizes and repetitions"
)
ap.add_argument("--skip", default="", help="comma-separated sections to skip")
ap.add_argument("--only", default="", help="comma-separated sections to run")
ap.add_argument(
    "--out", default="", help="report path (default rdna2_probe_<host>_<time>.json)"
)
ap.add_argument(
    "--setup-venv",
    nargs="?",
    const=os.path.expanduser("~/rdna2-probe-venv"),
    default=None,
    metavar="DIR",
    help="install ROCm + torch + triton into a venv, then run the probe in it",
)
ap.add_argument(
    "--index-url",
    default="https://repo.amd.com/rocm/whl/gfx103X-all/",
    help="AMD pip index for --setup-venv (per GPU family)",
)
ap.add_argument(
    "--setup-only",
    action="store_true",
    help="with --setup-venv: install and verify, do not probe",
)
a = ap.parse_args()


def python_headers(prefix):
    """include/pythonX.Y dir of a uv-fetched CPython under <prefix>/uv-python matching this Python, or None."""
    import glob

    ver = "%d.%d" % sys.version_info[:2]
    hits = sorted(
        glob.glob(
            os.path.join(
                prefix,
                "uv-python",
                f"cpython-{ver}*",
                "include",
                f"python{ver}",
                "Python.h",
            )
        )
    )
    return os.path.dirname(hits[-1]) if hits else None


def setup_venv(venv, index_url):
    """Create/reuse a venv with ROCm (TheRock pip), rocprofv3, torch and triton; -> env for the re-run."""

    def say(msg):
        print(f"[setup] {msg}", file=sys.stderr, flush=True)

    def run(cmd, **kw):
        say("$ " + " ".join(cmd))
        return subprocess.run(cmd, **kw).returncode

    if sys.version_info < (3, 10) or sys.version_info >= (3, 15):
        sys.exit(
            f"[setup] Python {platform.python_version()} has no ROCm wheels; use Python 3.10-3.14"
        )
    for dev in ("/dev/kfd", "/dev/dri"):
        if not os.path.exists(dev):
            sys.exit(
                f"[setup] {dev} is missing: load the amdgpu kernel driver first (this is the one OS-level piece)"
            )
    if not os.access("/dev/kfd", os.R_OK | os.W_OK):
        say(
            "WARNING: no read/write access to /dev/kfd; add your user to the 'render' (and 'video') group "
            "and log in again, or the GPU will not be visible"
        )
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        if run([sys.executable, "-m", "venv", venv]) != 0:
            sys.exit(
                "[setup] venv creation failed (Debian/Ubuntu: install the python3-venv package)"
            )
    pip = [py, "-m", "pip", "install", "--disable-pip-version-check"]
    run(pip + ["-U", "pip"])
    if (
        run(pip + ["--index-url", index_url, "rocm[libraries,devel]", "rocm-profiler"])
        != 0
    ):
        say(
            "rocm[libraries,devel] did not resolve; installing the SDK packages explicitly"
        )
        family = index_url.rstrip("/").rsplit("/", 1)[-1].lower()
        if (
            run(
                pip
                + [
                    "--index-url",
                    index_url,
                    "rocm-sdk-core",
                    "rocm-sdk-devel",
                    f"rocm-sdk-libraries-{family}",
                    "rocm-profiler",
                ]
            )
            != 0
        ):
            sys.exit(f"[setup] ROCm SDK install from {index_url} failed")
    if run(pip + ["--index-url", index_url, "torch", "triton"]) != 0:
        sys.exit(f"[setup] torch/triton install from {index_url} failed")
    bindir = os.path.join(venv, "bin")
    env = dict(
        os.environ,
        VIRTUAL_ENV=venv,
        PATH=bindir + os.pathsep + os.environ.get("PATH", ""),
    )
    root = (
        subprocess.run(
            [os.path.join(bindir, "rocm-sdk"), "path", "--root"],
            capture_output=True,
            text=True,
            env=env,
        )
        if os.path.exists(os.path.join(bindir, "rocm-sdk"))
        else None
    )
    if root is not None and root.returncode == 0 and root.stdout.strip():
        env["ROCM_PATH"] = root.stdout.strip().splitlines()[
            -1
        ]  # probe prefers the venv's tools over /opt/rocm
        rbin = os.path.join(env["ROCM_PATH"], "bin")
        if os.path.isdir(rbin):
            env["PATH"] = rbin + os.pathsep + env["PATH"]
    say(f"ROCM_PATH={env.get('ROCM_PATH', '(not found: rocm-sdk path --root failed)')}")
    # Triton JIT-compiles a C helper against Python.h; a system Python without its -dev package has none.
    has_h = subprocess.run(
        [
            py,
            "-c",
            "import os, sysconfig; print(os.path.exists(os.path.join("
            "sysconfig.get_paths()['include'], 'Python.h')))",
        ],
        capture_output=True,
        text=True,
    )
    if "True" not in has_h.stdout and not python_headers(venv):
        say(
            "Python.h is missing (Triton needs it); fetching CPython headers with uv into the venv (no sudo)"
        )
        ver = subprocess.run(
            [py, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", "uv"])
        run(
            [os.path.join(bindir, "uv"), "python", "install", ver],
            env=dict(env, UV_PYTHON_INSTALL_DIR=os.path.join(venv, "uv-python")),
        )
        if not python_headers(venv):
            say(
                "WARNING: could not get Python headers; Triton sections will fall back or skip. "
                f"Fix: sudo apt install python{ver}-dev  (or: sudo dnf install python3-devel)"
            )
    inc = python_headers(venv)
    if inc:
        env["C_INCLUDE_PATH"] = inc + os.pathsep + env.get("C_INCLUDE_PATH", "")
        say(f"Python headers for Triton: {inc}")
    if os.path.exists(os.path.join(bindir, "rocm-sdk")):
        run([os.path.join(bindir, "rocm-sdk"), "test"], env=env)  # informational
    chk = subprocess.run(
        [
            py,
            "-c",
            "import torch, triton; p = torch.cuda.get_device_properties(0) "
            "if torch.cuda.is_available() else None; print(torch.__version__, triton.__version__, "
            "torch.cuda.is_available(), p.gcnArchName if p else '-')",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    say("torch/triton/GPU check: " + (chk.stdout.strip() or chk.stderr.strip()[-300:]))
    if "True" not in chk.stdout:
        say(
            "WARNING: torch does not see the GPU; GPU sections will be skipped (check driver / render group)"
        )
    for t in ("rocminfo", "amd-smi", "rocprofv3", "hipconfig", "llvm-mc"):
        found = shutil.which(t, path=env["PATH"]) or next(
            (
                p
                for p in (
                    os.path.join(env.get("ROCM_PATH", ""), s, t)
                    for s in ("bin", "llvm/bin", "lib/llvm/bin")
                )
                if os.path.exists(p)
            ),
            None,
        )
        say(f"  {t:10s} {found or 'not found'}")
    return py, env


if a.setup_venv and not os.environ.get("RDNA2_PROBE_IN_VENV"):
    vpy, venv_env = setup_venv(os.path.abspath(a.setup_venv), a.index_url)
    if a.setup_only:
        sys.exit(0)
    argv = (
        ["--quick"] if a.quick else []
    )  # the probe's own options, without the setup flags
    for flag, val in (
        ("--skip", a.skip),
        ("--only", a.only),
        ("--out", os.path.abspath(a.out) if a.out else ""),
    ):
        if val:
            argv += [flag, val]
    venv_env["RDNA2_PROBE_IN_VENV"] = "1"
    os.execve(vpy, [vpy, os.path.abspath(__file__)] + argv, venv_env)
ALL = ["env", "bw", "gemm", "valu", "dot", "launch", "noise", "clock", "isa", "prof"]
SECTIONS = [
    s
    for s in (a.only.split(",") if a.only else ALL)
    if s and s not in a.skip.split(",")
]
Q = a.quick
R = {
    "probe": "rdna2_probe v2",
    "host": socket.gethostname(),
    "quick": Q,
    "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def sh(cmd, timeout=60):
    """-> (rc, stdout+stderr) ; rc None if the tool is missing or timed out."""
    if not shutil.which(cmd[0]) and not os.path.exists(cmd[0]):
        return None, f"{cmd[0]}: not found"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


_ROCM_DIRS = None


def rocm_dirs():
    """ROCM_PATH, /opt/rocm, and whatever hipconfig reports (pip-installed SDKs live elsewhere)."""
    global _ROCM_DIRS
    if _ROCM_DIRS is None:
        _ROCM_DIRS = [os.environ.get("ROCM_PATH", ""), "/opt/rocm"]
        if shutil.which("hipconfig"):
            rc, out = sh(["hipconfig", "--rocmpath"], 30)
            if rc == 0 and out.strip():
                _ROCM_DIRS.append(out.strip().splitlines()[-1])
    return _ROCM_DIRS


def rocm_tool(name):
    for d in rocm_dirs():
        for sub in ("bin", "llvm/bin", "lib/llvm/bin"):
            p = os.path.join(d, sub, name) if d else ""
            if p and os.path.exists(p):
                return p
    return shutil.which(name) or name


def section(name):
    def deco(fn):
        def run():
            if name not in SECTIONS:
                return
            log(f"== {name}")
            t = time.time()
            try:
                R[name] = fn()
            except Exception as e:  # noqa: BLE001
                R[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                log(f"   {name} failed: {R[name]['error']}")
            R.setdefault("_seconds", {})[name] = round(time.time() - t, 1)

        return run

    return deco


# --------------------------------------------------------------------------------------------- env
# Running inside a --setup-venv venv directly: pick up the uv-fetched Python headers Triton compiles against.
_inc = python_headers(sys.prefix)
if _inc and _inc not in os.environ.get("C_INCLUDE_PATH", ""):
    os.environ["C_INCLUDE_PATH"] = (
        _inc + os.pathsep + os.environ.get("C_INCLUDE_PATH", "")
    )
# hipBLASLt ships no RDNA2 (gfx103x) kernels (no TensileLibrary_lazy_gfx1030.dat / kernels.*gfx1030*.hsaco),
# so route torch's GEMMs to rocBLAS, which does support gfx1030. Set before torch is imported.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
try:
    import torch

    HAVE_TORCH = torch.cuda.is_available()
    try:
        torch.backends.cuda.preferred_blas_library(
            "cublas"
        )  # "cublas" is rocBLAS on ROCm builds
    except Exception:  # noqa: BLE001
        pass
except Exception as e:  # noqa: BLE001
    torch, HAVE_TORCH = None, False
    R["torch_error"] = repr(e)[:200]
try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    triton, tl, HAVE_TRITON = None, None, False
DEV = "cuda"


@section("env")
def s_env():
    env = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "hip": getattr(getattr(torch, "version", None), "hip", None),
        "triton": getattr(triton, "__version__", None),
        "cuda_available": HAVE_TORCH,
    }
    for d in rocm_dirs():
        f = os.path.join(d, ".info", "version") if d else ""
        if f and os.path.exists(f):
            env["rocm_version"] = open(f).read().strip()
            break
    rc, out = sh(["hipconfig", "--version"], 30)
    if rc == 0:
        env["hipconfig_version"] = out.strip()[:60]
    if HAVE_TORCH:
        p = torch.cuda.get_device_properties(0)
        env["device"] = {
            k: getattr(p, k, None)
            for k in (
                "name",
                "gcnArchName",
                "total_memory",
                "multi_processor_count",
                "warp_size",
                "max_threads_per_multi_processor",
                "regs_per_multiprocessor",
                "shared_memory_per_block",
                "shared_memory_per_multiprocessor",
                "L2_cache_size",
                "clock_rate",
                "memory_clock_rate",
                "memory_bus_width",
            )
        }
        env["device"]["torch_name_repr"] = repr(torch.cuda.get_device_name(0))
        try:
            env["blas"] = {
                "preferred": str(torch.backends.cuda.preferred_blas_library()),
                "TORCH_BLAS_PREFER_HIPBLASLT": os.environ.get(
                    "TORCH_BLAS_PREFER_HIPBLASLT"
                ),
            }
        except Exception as e:  # noqa: BLE001
            env["blas"] = {"error": str(e)[:120]}
    rc, out = sh([rocm_tool("rocminfo")], 60)
    if rc == 0:
        # keep only the first GPU agent's block
        blocks = re.split(r"\n\*{5,}\s*\nAgent \d+\s*\n\*{5,}", out)
        gfx = torch_gfx()
        gpus = [b for b in blocks if re.search(r"Name:\s+gfx", b)]
        gpu = next(
            (b for b in gpus if gfx and re.search(rf"Name:\s+{gfx}\b", b)),
            gpus[0] if gpus else "",
        )
        env["rocminfo_gpu_agents"] = len(gpus)
        keep = (
            "Name:",
            "Marketing Name:",
            "Compute Unit:",
            "SIMDs per CU:",
            "Shader Engines:",
            "Shader Arrs. per Eng.:",
            "Wavefront Size:",
            "Max Waves Per CU:",
            "Workgroup Max Size:",
            "Max Clock Freq. (MHz):",
            "Cacheline Size:",
            "L1:",
            "L2:",
            "L3:",
            "Chip ID:",
            "Features:",
            "Fast F16 Operation:",
            "ISA 1",
            "Size:",
            "Segment:",
        )
        env["rocminfo_gpu"] = [
            l.strip() for l in gpu.splitlines() if l.strip().startswith(keep)
        ][:60]
    else:
        env["rocminfo_gpu"] = out[:300]
    for tool, args in (
        (
            "amd-smi",
            [
                "static",
                "--asic",
                "--vram",
                "--cache",
                "--limit",
                "--board",
                "-g",
                smi_index(),
            ],
        ),
        (
            "rocm-smi",
            [
                "-d",
                smi_index(),
                "--showproductname",
                "--showmeminfo",
                "vram",
                "--showmaxpower",
                "--showclkfrq",
                "--showvbios",
            ],
        ),
    ):
        rc, out = sh([rocm_tool(tool)] + args, 60)
        env[tool.replace("-", "_")] = (out if rc == 0 else f"rc={rc}: {out}")[:2500]
    return env


_SMI_INDEX = None


def torch_gfx():
    return (
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if HAVE_TORCH
        else None
    )


def json_values(text):
    """Every top-level JSON value in text (tools sometimes print several, or text around them)."""
    dec, i, vals = json.JSONDecoder(), 0, []
    while True:
        starts = [k for k in (text.find("{", i), text.find("[", i)) if k >= 0]
        if not starts:
            return vals
        try:
            v, i = dec.raw_decode(text, min(starts))
            vals.append(v)
        except ValueError:
            i = min(starts) + 1


def smi_index():
    """amd-smi / rocm-smi index of torch's device 0. On multi-GPU boxes the SMI order differs from HIP's
    (e.g. an RX 9070 XT listed first while torch sees the V620 as device 0), so match on gfx arch + name.
    """
    global _SMI_INDEX
    if _SMI_INDEX is None:
        _SMI_INDEX = "0"
        gfx = torch_gfx()
        name = torch.cuda.get_device_name(0) if HAVE_TORCH else ""
        rc, out = sh([rocm_tool("amd-smi"), "static", "--asic", "--json"], 60)
        if rc == 0 and gfx:
            try:
                rows = []
                for j in json_values(
                    out
                ):  # amd-smi may print several JSON values (or text) back to back
                    if isinstance(j, list):
                        rows += [r for r in j if isinstance(r, dict)]
                    elif isinstance(j, dict):
                        rows += (
                            j.get("gpu_data", [j])
                            if isinstance(j.get("gpu_data", [j]), list)
                            else [j]
                        )
                cand = [
                    r
                    for r in rows
                    if str(
                        (r.get("asic") or {}).get("target_graphics_version", "")
                    ).lower()
                    == gfx
                ]
                if len(cand) > 1 and name:
                    cand = [
                        r
                        for r in cand
                        if name.lower()
                        in str((r.get("asic") or {}).get("market_name", "")).lower()
                    ] or cand
                if cand:
                    _SMI_INDEX = str(cand[0].get("gpu", 0))
            except Exception as e:  # noqa: BLE001
                R.setdefault("warnings", []).append(
                    f"amd-smi index match failed: {str(e)[:120]}"
                )
        R["smi_gpu_index"] = _SMI_INDEX
    return _SMI_INDEX


# ---------------------------------------------------------------------------------------- helpers
_EVICT = None


def evict():
    """Read-based eviction (> 128 MB Infinity Cache): a write-based evict leaves dirty lines whose
    write-back would be charged to the timed kernel."""
    global _EVICT
    if _EVICT is None:
        _EVICT = torch.empty(
            768 * 2**20 // 4, dtype=torch.float32, device=DEV
        ).normal_()
    _EVICT.sum()


def bench(fn, iters=20, warmup=3, cold=False, batch=1):
    """Median ms per call (GPU events) and relative spread."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if cold:
            evict()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        s.record()
        for _ in range(batch):
            fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / batch)
    med = statistics.median(ts)
    return med, (max(ts) - min(ts)) / med if med else None


if HAVE_TRITON:

    @triton.jit
    def _read_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        base = pid * BLOCK * ITERS
        for i in range(ITERS):
            offs = base + i * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + offs, mask=offs < n, other=0)
        tl.store(out_ptr + pid, tl.sum(acc, axis=0))

    @triton.jit
    def _fma_kernel(x_ptr, out_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(x_ptr + offs)
        b = a * 0.5 + 0.25
        c = a * 0.25 + 0.5
        d = a + 0.5
        # 4 independent FMA chains per element per iteration. The constants are exact in fp16 and != 1
        # (0.9999 rounds to 1.0 in fp16 and the compiler folds a*1+c into an add); every chain converges
        # to 1.0, so values stay normal.
        for _ in range(ITERS):
            a = a * 0.875 + 0.125
            b = b * 0.75 + 0.25
            c = c * 0.625 + 0.375
            d = d * 0.5 + 0.5
        tl.store(out_ptr + offs, a + b + c + d)

    @triton.jit
    def _dot_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
            b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


_TRITON_STATE = None


def triton_ok():
    """(usable, reason): compile and run one tiny Triton kernel once (needs Python.h and arch support)."""
    global _TRITON_STATE
    if _TRITON_STATE is None:
        if not (HAVE_TORCH and HAVE_TRITON):
            _TRITON_STATE = (False, "torch with a GPU and triton must both import")
        else:
            try:
                x = torch.zeros(1024, dtype=torch.int32, device=DEV)
                o = torch.empty(1, dtype=torch.int32, device=DEV)
                _read_kernel[(1,)](x, o, 1024, BLOCK=1024, ITERS=1)
                torch.cuda.synchronize()
                _TRITON_STATE = (True, "")
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e)[-300:]}"
                if "Python.h" in msg:
                    msg = (
                        "Python.h missing (Triton JIT builds a C helper): rerun with --setup-venv, or install "
                        "python3-dev / python3-devel. " + msg[-150:]
                    )
                _TRITON_STATE = (False, msg)
        R["triton_usable"] = {"ok": _TRITON_STATE[0], "reason": _TRITON_STATE[1]}
        if not _TRITON_STATE[0]:
            log(f"   Triton unusable: {_TRITON_STATE[1][:160]}")
    return _TRITON_STATE


def _read_bw(x, n, cold, batch):
    best = None
    cfgs = ((1024, 8, 4), (2048, 4, 8), (1024, 16, 4), (4096, 4, 8))
    for block, iters, nw in cfgs[:2] if Q else cfgs:
        nprog = triton.cdiv(n, block * iters)
        out = torch.empty(nprog, dtype=torch.int32, device=DEV)
        fn = lambda: _read_kernel[(nprog,)](
            x, out, n, BLOCK=block, ITERS=iters, num_warps=nw
        )  # noqa: E731
        ms, spread = bench(fn, iters=8 if Q else 20, cold=cold, batch=batch)
        gbs = n * 4 / ms / 1e6
        if best is None or gbs > best["GBps"]:
            best = {
                "GBps": round(gbs, 1),
                "ms": round(ms, 4),
                "spread": round(spread, 3),
                "cfg": [block, iters, nw],
            }
    return best


@section("bw")
def s_bw():
    if not HAVE_TORCH:
        return {"skipped": "no torch/ROCm"}
    res = {"read": [], "copy": []}
    sizes = (
        (2, 8, 64, 256, 1024)
        if Q
        else (1, 2, 4, 8, 16, 32, 48, 64, 96, 112, 128, 160, 192, 256, 512, 1024, 2048)
    )
    for mb in sizes:
        n = mb * 2**20 // 4
        x = torch.randint(-(2**30), 2**30, (n,), dtype=torch.int32, device=DEV)
        row = {"MB": mb}
        if triton_ok()[0]:
            batch = max(1, min(50, int(256 / mb)))
            row["warm"] = _read_bw(x, n, cold=False, batch=batch)
            if mb >= 64:
                row["cold"] = _read_bw(x, n, cold=True, batch=1)
        else:  # torch fallback: x.sum() is a read-mostly reduction
            ms, sp = bench(lambda x=x: x.sum(), batch=max(1, min(50, int(256 / mb))))
            row["warm_torch_sum"] = {
                "GBps": round(n * 4 / ms / 1e6, 1),
                "ms": round(ms, 4),
            }
        res["read"].append(row)
        log(
            f"   read {mb:5d} MB  warm {row.get('warm', row.get('warm_torch_sum'))['GBps']} GB/s"
            + (f"  cold {row['cold']['GBps']} GB/s" if "cold" in row else "")
        )
        del x
        torch.cuda.empty_cache()
    for mb in (
        (64, 512) if Q else (16, 64, 128, 256, 512, 1024)
    ):  # torch.add: 2 reads + 1 write
        n = mb * 2**20 // 4
        x, y = torch.rand(n, device=DEV), torch.rand(n, device=DEV)
        o = torch.empty_like(x)
        ms, sp = bench(
            lambda x=x, y=y, o=o: torch.add(x, y, out=o),
            batch=max(1, min(20, int(512 / mb))),
        )
        res["copy"].append(
            {
                "MB_per_buffer": mb,
                "GBps_rrw": round(3 * n * 4 / ms / 1e6, 1),
                "ms": round(ms, 4),
                "spread": round(sp, 3),
            }
        )
        del x, y, o
        torch.cuda.empty_cache()
    return res


@section("gemm")
def s_gemm():
    if not HAVE_TORCH:
        return {"skipped": "no torch/ROCm"}
    out = {"mm": []}
    sizes = (2048, 4096) if Q else (1024, 2048, 4096, 8192)
    for dt_name, dt in (
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ):
        for s in sizes:
            try:
                A = torch.randn(s, s, device=DEV, dtype=dt)
                B = torch.randn(s, s, device=DEV, dtype=dt)
                ms, sp = bench(
                    lambda A=A, B=B: torch.mm(A, B),
                    iters=5 if Q else 10,
                    batch=1 if s >= 4096 else 5,
                )
                out["mm"].append(
                    {
                        "dtype": dt_name,
                        "n": s,
                        "TFLOPs": round(2 * s**3 / ms / 1e9, 2),
                        "ms": round(ms, 3),
                        "spread": round(sp, 3),
                    }
                )
                log(f"   mm {dt_name} {s}: {out['mm'][-1]['TFLOPs']} TFLOP/s")
                del A, B
            except Exception as e:  # noqa: BLE001
                out["mm"].append({"dtype": dt_name, "n": s, "error": str(e)[:160]})
    torch.cuda.empty_cache()
    # int8 GEMM. torch._int_mm only has a hipBLASLt path on ROCm, which has no gfx103x kernels and can
    # abort the process instead of raising, so it runs in a child process.
    child = (
        "import json, torch\n"
        "s = 4096\n"
        "A = torch.randint(-64, 64, (s, s), device='cuda', dtype=torch.int8)\n"
        "B = torch.randint(-64, 64, (s, s), device='cuda', dtype=torch.int8)\n"
        "for _ in range(3): torch._int_mm(A, B)\n"
        "torch.cuda.synchronize()\n"
        "e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)\n"
        "e0.record()\n"
        "for _ in range(10): torch._int_mm(A, B)\n"
        "e1.record(); e1.synchronize(); ms = e0.elapsed_time(e1) / 10\n"
        "print(json.dumps({'n': s, 'TOPs': round(2 * s**3 / ms / 1e9, 2), 'ms': round(ms, 3)}))\n"
    )
    try:
        p = subprocess.run(
            [sys.executable, "-c", child],
            capture_output=True,
            text=True,
            timeout=180,
            env=dict(os.environ),
        )
        line = next(
            (l for l in reversed(p.stdout.splitlines()) if l.startswith("{")), None
        )
        out["int_mm"] = (
            json.loads(line)
            if p.returncode == 0 and line
            else {
                "unsupported": f"rc={p.returncode}",
                "stderr_tail": (p.stderr or p.stdout)[-300:],
            }
        )
    except subprocess.TimeoutExpired:
        out["int_mm"] = {"unsupported": "timed out after 180 s"}
    try:  # which library/kernel torch.mm launches (rocBLAS Tensile "Cijk_*", hipBLASLt, ...)
        from torch.profiler import ProfilerActivity, profile

        A = torch.randn(2048, 2048, device=DEV, dtype=torch.float16)
        torch.mm(A, A)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.mm(A, A)
            torch.cuda.synchronize()
        names = sorted(
            {
                e.name
                for e in prof.events()
                if e.device_type.name in ("CUDA", "PrivateUse1")
            }
        )
        out["mm_fp16_kernels"] = [n[:160] for n in names][:6]
    except Exception as e:  # noqa: BLE001
        out["mm_fp16_kernels"] = f"profiler failed: {str(e)[:160]}"
    return out


@section("valu")
def s_valu():
    """Vector FMA throughput from the SLOPE between two loop lengths, which cancels launch and other
    fixed costs. `scaling` is t(long)/t(short) against the 4x it should be when the ALU is the limit:
    a value well below 4 means something other than the FMAs bounds the kernel, so its number is not
    an ALU peak. Each thread holds 8 elements x 4 independent chains."""
    if not triton_ok()[0]:
        return {"skipped": "Triton unusable: " + triton_ok()[1][:240]}
    res = []
    sms = torch.cuda.get_device_properties(0).multi_processor_count  # WGPs on RDNA
    short, long_ = 128, 512
    for dt_name, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        best = None
        for nw in (4, 8):
            BLOCK = nw * 32 * 8
            nprog = sms * 32
            x = torch.rand(nprog * BLOCK, device=DEV, dtype=dt)
            o = torch.empty_like(x)
            t = {}
            for it in (short, long_):
                t[it], _ = bench(
                    lambda: _fma_kernel[(nprog,)](
                        x, o, ITERS=it, BLOCK=BLOCK, num_warps=nw
                    ),
                    iters=5 if Q else 10,
                )
            dflops = (
                nprog * BLOCK * (long_ - short) * 4 * 2
            )  # 4 FMA chains, 2 flops per FMA
            slope_tf = dflops / max(t[long_] - t[short], 1e-6) / 1e9
            row = {
                "dtype": dt_name,
                "num_warps": nw,
                "elems_per_thread": 8,
                "TFLOPs_slope": round(slope_tf, 2),
                "TFLOPs_long_raw": round(nprog * BLOCK * long_ * 8 / t[long_] / 1e9, 2),
                "ms": {str(k): round(v, 4) for k, v in t.items()},
                "scaling": round(t[long_] / t[short], 2),
                "scaling_expected": long_ // short,
            }
            if best is None or row["TFLOPs_slope"] > best["TFLOPs_slope"]:
                best = row
        try:
            k = _fma_kernel[(1,)](
                x,
                o,
                ITERS=long_,
                BLOCK=best["num_warps"] * 32 * 8,
                num_warps=best["num_warps"],
            )
            asm = k.asm.get("amdgcn", "") if hasattr(k, "asm") else ""
            ops = re.findall(r"^\s+(v_[a-z0-9_]+)", asm, re.M)
            best["top_valu_ops"] = sorted(
                {o_: ops.count(o_) for o_ in set(ops)}.items(), key=lambda t_: -t_[1]
            )[:6]
            m = re.search(r"\.vgpr_count:\s*(\d+)", asm)
            if m:
                best["vgprs"] = int(m.group(1))
        except Exception as e:  # noqa: BLE001
            best["asm_error"] = str(e)[:120]
        best["alu_bound"] = best["scaling"] >= 0.8 * best["scaling_expected"]
        res.append(best)
        log(
            f"   valu {dt_name}: {best['TFLOPs_slope']} TFLOP/s (slope), scaling {best['scaling']}x of "
            f"{best['scaling_expected']}x{'' if best['alu_bound'] else '  <- NOT ALU-bound, do not quote'}"
        )
    return res


@section("dot")
def s_dot():
    if not triton_ok()[0]:
        return {"skipped": "Triton unusable: " + triton_ok()[1][:240]}
    res = []
    M = N = K = 2048 if Q else 4096
    A = torch.randn(M, K, device=DEV, dtype=torch.float16)
    B = torch.randn(K, N, device=DEV, dtype=torch.float16)
    C = torch.empty(M, N, device=DEV, dtype=torch.float16)
    ref = A.float() @ B.float()
    for BM, BN, BK, nw in ((64, 64, 32, 4), (128, 128, 32, 8), (128, 64, 64, 4)):
        row = {"tile": [BM, BN, BK], "num_warps": nw}
        try:
            grid = (M // BM, N // BN)
            fn = lambda: _dot_kernel[grid](
                A, B, C, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=nw
            )  # noqa: E731
            fn()
            torch.cuda.synchronize()
            row["max_rel_err"] = round(
                float((C.float() - ref).abs().max() / ref.abs().max()), 5
            )
            ms, sp = bench(fn, iters=5 if Q else 10)
            row["TFLOPs"] = round(2 * M * N * K / ms / 1e9, 2)
            k = _dot_kernel[grid](A, B, C, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=nw)
            asm = k.asm.get("amdgcn", "") if hasattr(k, "asm") else ""
            ops = re.findall(r"^\s+(v_[a-z0-9_]+)", asm, re.M)
            row["matrix_or_dot_ops"] = sorted(
                {o for o in ops if re.match(r"v_(wmma|mfma|dot|pk_fma|fma_mix)", o)}
            )
            row["top_ops"] = sorted(
                {o: ops.count(o) for o in set(ops)}.items(), key=lambda t: -t[1]
            )[:6]
            m = re.search(r"\.vgpr_count:\s*(\d+)", asm) or re.search(
                r"NumVgprs:\s*(\d+)", asm
            )
            if m:
                row["vgprs"] = int(m.group(1))
            m = re.search(r"\.sgpr_count:\s*(\d+)", asm) or re.search(
                r"NumSgprs:\s*(\d+)", asm
            )
            if m:
                row["sgprs"] = int(m.group(1))
            m = re.search(r"\.wavefront_size:\s*(\d+)", asm)
            if m:
                row["wavefront_size"] = int(m.group(1))
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:220]}"
        res.append(row)
        log(f"   tl.dot fp16 {row['tile']}: {row.get('TFLOPs', row.get('error', '?'))}")
    return res


@section("launch")
def s_launch():
    if not HAVE_TORCH:
        return {"skipped": "no torch/ROCm"}
    x = torch.zeros(256, device=DEV)
    for _ in range(50):
        x.add_(1)
    torch.cuda.synchronize()
    n = 2000
    t = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    torch.cuda.synchronize()
    host_us = (time.perf_counter() - t) / n * 1e6
    ms, _ = bench(lambda: x.add_(1), iters=20, batch=200)
    t = time.perf_counter()
    for _ in range(200):
        x.add_(1)
        torch.cuda.synchronize()
    sync_us = (time.perf_counter() - t) / 200 * 1e6
    return {
        "back_to_back_host_us": round(host_us, 2),
        "back_to_back_gpu_event_us": round(ms * 1e3, 2),
        "launch_plus_sync_us": round(sync_us, 1),
    }


@section("noise")
def s_noise():
    if not HAVE_TORCH:
        return {"skipped": "no torch/ROCm"}
    n = 512 * 2**20 // 4  # 512 MB per buffer: far above the 128 MB Infinity Cache
    x, y = torch.rand(n, device=DEV), torch.rand(n, device=DEV)
    o = torch.empty_like(x)
    samples = []
    for _ in range(6 if Q else 15):
        ms, _ = bench(lambda: torch.add(x, y, out=o), iters=1, warmup=2, batch=10)
        samples.append(3 * n * 4 / ms / 1e6)
    m, sd = statistics.mean(samples), statistics.stdev(samples)
    return {
        "GBps_mean": round(m, 1),
        "cv_pct": round(100 * sd / m, 2),
        "two_sigma_pct": round(200 * sd / m, 2),
        "min": round(min(samples), 1),
        "max": round(max(samples), 1),
    }


def _metrics_snapshot():
    """One reading of clocks / power / temperature from amd-smi (preferred) or rocm-smi."""
    rc, out = sh(
        [
            rocm_tool("amd-smi"),
            "metric",
            "-g",
            smi_index(),
            "--clock",
            "--power",
            "--temperature",
            "--json",
        ],
        20,
    )
    if rc == 0:
        try:
            j = json.loads(
                out[out.index("[") :]
                if out.lstrip().startswith("[") or "[" in out[:5]
                else out
            )
            return {"amd_smi": j[0] if isinstance(j, list) else j}
        except Exception:  # noqa: BLE001
            return {"amd_smi_text": out[:1500]}
    rc, out = sh(
        [
            rocm_tool("rocm-smi"),
            "-d",
            smi_index(),
            "--showclocks",
            "--showpower",
            "--showtemp",
            "--json",
        ],
        20,
    )
    if rc == 0:
        try:
            return {"rocm_smi": json.loads(out[out.index("{") :])}
        except Exception:  # noqa: BLE001
            return {"rocm_smi_text": out[:1500]}
    return {"error": out[:300]}


def _compact(snap):
    """Pull sclk / mclk / power / temp numbers out of whatever the SMI tool returned."""
    txt = json.dumps(snap)
    pick = {}
    for key, pat in (
        (
            "sclk",
            r'"(?:gfx_0|GFX_0|sclk|SCLK)"[^}]*?"clk"\s*:\s*\{?\s*"value"\s*:\s*(\d+)',
        ),
        ("sclk", r'current sclk clock speed:?"\s*:\s*"\((\d+)'),
        (
            "mclk",
            r'"(?:mem_0|MEM_0|mclk|MCLK)"[^}]*?"clk"\s*:\s*\{?\s*"value"\s*:\s*(\d+)',
        ),
        ("mclk", r'current mclk clock speed:?"\s*:\s*"\((\d+)'),
        (
            "power_w",
            r'"(?:socket_power|current_socket_power|average_socket_power)"\s*:\s*\{?\s*"value"\s*:\s*([\d.]+)',
        ),
        ("power_w", r'Average Graphics Package Power \(W\)"\s*:\s*"([\d.]+)'),
        ("power_w", r'Current Socket Graphics Package Power \(W\)"\s*:\s*"([\d.]+)'),
        ("temp_c", r'"(?:hotspot|edge)"\s*:\s*\{?\s*"value"\s*:\s*([\d.]+)'),
        ("temp_c", r'Temperature \(Sensor (?:junction|edge)\) \(C\)"\s*:\s*"([\d.]+)'),
    ):
        if key not in pick:
            m = re.search(pat, txt, re.I)
            if m:
                pick[key] = float(m.group(1))
    return pick


@section("clock")
def s_clock():
    res = {"idle": _metrics_snapshot()}
    res["idle_compact"] = _compact(res["idle"])
    if not HAVE_TORCH:
        return res
    s = 4096
    A = torch.randn(s, s, device=DEV, dtype=torch.float16)
    B = torch.randn(s, s, device=DEV, dtype=torch.float16)
    stop = threading.Event()
    rates = []

    def load():
        while not stop.is_set():
            t = time.perf_counter()
            for _ in range(10):
                torch.mm(A, B)
            torch.cuda.synchronize()
            rates.append(10 * 2 * s**3 / (time.perf_counter() - t) / 1e12)

    th = threading.Thread(target=load, daemon=True)
    th.start()
    dur = 10 if Q else 30
    samples = []
    t0 = time.time()
    while time.time() - t0 < dur:
        time.sleep(3)
        samples.append(
            {"t": round(time.time() - t0, 1), **_compact(_metrics_snapshot())}
        )
    stop.set()
    th.join(timeout=30)
    res["load_fp16_gemm_4096"] = {
        "seconds": dur,
        "samples": samples,
        "TFLOPs_first": round(rates[0], 2) if rates else None,
        "TFLOPs_last": round(rates[-1], 2) if rates else None,
        "TFLOPs_median": round(statistics.median(rates), 2) if rates else None,
    }
    res["load_raw_last"] = str(_metrics_snapshot())[:1200]
    return res


@section("isa")
def s_isa():
    gfx = None
    if HAVE_TORCH:
        gfx = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    gfx = gfx or "gfx1030"
    mc = rocm_tool("llvm-mc")
    probes = {
        "v_dot2_f32_f16": "v_dot2_f32_f16 v0, v1, v2, v3",
        "v_dot2c_f32_f16": "v_dot2c_f32_f16 v0, v1, v2",
        "v_dot4_i32_i8": "v_dot4_i32_i8 v0, v1, v2, v3",
        "v_dot4c_i32_i8": "v_dot4c_i32_i8 v0, v1, v2",
        "v_dot8_i32_i4": "v_dot8_i32_i4 v0, v1, v2, v3",
        "v_dot2_f32_bf16": "v_dot2_f32_bf16 v0, v1, v2, v3",
        "v_fma_mix_f32": "v_fma_mix_f32 v0, v1, v2, v3",
        "v_pk_fma_f16": "v_pk_fma_f16 v0, v1, v2, v3",
        "v_wmma_f32_16x16x16_f16": "v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]",
        "v_mfma_f32_16x16x16f16": "v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[0:3]",
    }
    res = {"gfx": gfx, "llvm_mc": mc, "accepted": [], "rejected": []}
    rc, _ = sh([mc, "--version"], 20)
    if rc is None:
        res["error"] = (
            "llvm-mc not found (set ROCM_PATH or put /opt/rocm/llvm/bin on PATH)"
        )
        return res
    for name, ins in probes.items():
        try:
            p = subprocess.run(
                [mc, "-arch=amdgcn", f"-mcpu={gfx}", "-show-encoding"],
                input=ins,
                capture_output=True,
                text=True,
                timeout=20,
            )
            (
                res["accepted"]
                if p.returncode == 0 and "error" not in p.stderr
                else res["rejected"]
            ).append(name)
        except Exception as e:  # noqa: BLE001
            res["rejected"].append(f"{name} ({e})")
    for mode in (
        "+wavefrontsize64",
    ):  # does the assembler accept wave64 mode for this gfx
        p = subprocess.run(
            [mc, "-arch=amdgcn", f"-mcpu={gfx}", f"-mattr={mode}"],
            input="s_nop 0",
            capture_output=True,
            text=True,
            timeout=20,
        )
        res["wave64_mattr_ok"] = p.returncode == 0
    return res


@section("prof")
def s_prof():
    rp = rocm_tool("rocprofv3")
    res = {}
    rc, out = sh([rp, "--version"], 30)
    res["rocprofv3_version"] = (
        out.strip().splitlines()[0][:120]
        if rc == 0 and out.strip()
        else f"unavailable: {out[:120]}"
    )
    if rc is not None:
        rc, out = sh([rp, "-L"], 120)
        names = sorted(
            set(
                re.findall(
                    r"\b((?:SQ|SQC|TCP|TCC|TA|TD|GL1C|GL2C|GRBM|CPC|CPF|SPI|GL2A|GCEA|"
                    r"LDS|FETCH|WRITE|VALU|SALU|MemUnit|L2Cache)[A-Za-z0-9_]*)\b",
                    out,
                )
            )
        )
        res["counter_count"] = len(names)
        interesting = [
            n
            for n in names
            if re.match(
                r"(GL2C_(HIT|MISS|REQ)|TCC_(HIT|MISS|REQ)|TCP_|GL1C_(HIT|MISS)|"
                r"SQ_INSTS_(LDS|VALU|SALU|VMEM)|SQC_LDS_BANK_CONFLICT|SQ_WAVES|"
                r"SQ_BUSY|GRBM_GUI_ACTIVE|FETCH_SIZE|WRITE_SIZE|SQ_INSTS_VALU_DOT)",
                n,
            )
        ]
        res["interesting_counters"] = interesting[:80]
    rc, out = sh([rocm_tool("rocprof-compute"), "--version"], 30)
    res["rocprof_compute"] = (
        (out.strip().splitlines() or ["?"])[0][:120]
        if rc == 0
        else f"unavailable: {out[:100]}"
    )
    return res


for fn in (
    s_env,
    s_bw,
    s_gemm,
    s_valu,
    s_dot,
    s_launch,
    s_noise,
    s_clock,
    s_isa,
    s_prof,
):
    fn()

out_path = (
    a.out or f"rdna2_probe_{socket.gethostname()}_{time.strftime('%Y%m%d_%H%M')}.json"
)
with open(out_path, "w") as f:
    json.dump(R, f, indent=2, default=str)
log(f"wrote {out_path}")
print("===== RDNA2_PROBE_JSON_BEGIN =====")
print(json.dumps(R, indent=2, default=str))
print("===== RDNA2_PROBE_JSON_END =====")
