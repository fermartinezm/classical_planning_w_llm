#!/usr/bin/env python3
"""
Run all of Axis 1 (Member 1) in one go and pack the results for the shared Drive.

  1. obfuscate.py                 build L1-L3 (+ extra renaming seeds)
  2. pipeline.py gen              all levels in parallel (LLM API)
  3. pipeline.py train + select   per level
  4. pipeline.py test             per level (the long part)
  5. plots                        plots/obfuscation_summary.py + plots/make_plots_and_tables.py
  6. zip                          axis1_results_<date>.zip (upload this to the Drive)

Resumable: if it stops, run the same command again and it continues.

Usage (from the repository root):
  python run_axis1.py --workers 6
  python run_axis1.py --workers 6 --levels L0 L1          # subset
  python run_axis1.py --pack-only                          # just re-plot and re-zip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def sh(*cmd) -> None:
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=ROOT)
    if rc != 0:
        print(f"  (exit code {rc}, continuing)")


def runs(level: str, seeds):
    """(level, seed) pairs: seed 1 for every level, extra seeds only for L1."""
    out = [(level, 1)]
    if level == "L1":
        out += [("L1", s) for s in seeds]
    return out


def pack() -> Path:
    sh(PY, "plots/obfuscation_summary.py")
    sh(PY, "plots/make_plots_and_tables.py", "--inputs", "results/*.jsonl", "--out-dir", "plots/output")
    zpath = ROOT / f"axis1_results_{datetime.now():%Y%m%d_%H%M}.zip"
    patterns = ["results/member1_*", "plots/output/axis1/*", "plots/output/F1_obfuscation.*",
                "generated/axis1_*_selected/*.py"]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for pat in patterns:
            for f in sorted(ROOT.glob(pat)):
                if f.is_file():
                    z.write(f, f.relative_to(ROOT))
    print(f"\nUpload to the Drive: {zpath}")
    return zpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    ap.add_argument("--extra-seeds", nargs="*", type=int, default=[2, 3],
                    help="Extra random renamings of L1 (professor's feedback). Use none to skip.")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (physical cores - 1, ~2.5 GB RAM each)")
    ap.add_argument("--pack-only", action="store_true")
    args, passthrough = ap.parse_known_args()   # e.g. --framework/--model/--tag go to pipeline.py

    if args.pack_only:
        pack()
        return
    import os
    if "openai-compatible" not in passthrough and "gemini" not in passthrough and "openai" not in passthrough \
            and not os.getenv("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY is not set in this terminal")

    # 1. benchmarks
    obf_levels = [l for l in args.levels if l != "L0"]
    if obf_levels:
        sh(PY, "obfuscate.py", "--levels", *obf_levels)
    if "L1" in args.levels and args.extra_seeds:
        sh(PY, "obfuscate.py", "--levels", "L1", "--seeds", *args.extra_seeds)

    if args.workers <= 0:
        import psutil
        cores = psutil.cpu_count(logical=False) or 4
        ram_gb = psutil.virtual_memory().total / 2**30
        args.workers = max(1, min(cores - 1, int(ram_gb / 2.5)))
    print(f"workers = {args.workers}")

    jobs = [r for l in args.levels for r in runs(l, args.extra_seeds)]

    # 2. generation: every level/seed in parallel (network-bound)
    procs = []
    for lvl, seed in jobs:
        cmd = [PY, "pipeline.py", "gen", "--level", lvl, "--seed", str(seed), *passthrough]
        print("$ " + " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=ROOT))
    for p in procs:
        p.wait()

    # 3-4. first the main runs (seed 1, they feed F1), then the extra renamings.
    #      Each phase: train + select for all levels, then test, packing after every level
    #      so partial results can already be uploaded.
    for phase in (1, 0):
        phase_jobs = [j for j in jobs if (j[1] == 1) == bool(phase)]
        for lvl, seed in phase_jobs:
            sh(PY, "pipeline.py", "train", "--level", lvl, "--seed", seed, "--workers", args.workers, *passthrough)
            sh(PY, "pipeline.py", "select", "--level", lvl, "--seed", seed, *passthrough)
        for lvl, seed in phase_jobs:
            sh(PY, "pipeline.py", "test", "--level", lvl, "--seed", seed, "--workers", args.workers, *passthrough)
            pack()


if __name__ == "__main__":
    main()
