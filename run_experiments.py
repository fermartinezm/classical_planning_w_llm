#!/usr/bin/env python3
"""
Unified parallel experiment runner & candidate selector for all 4 team members.

Features:
- Uses canonical splits from `benchmarks/splits.json` (20 train tasks @ 60s, 30 test tasks @ 300s).
- Standardizes on Pyperplan `-s gbf` (or `-s gbf_alt` for open-list alternation).
- Isolates each run's `.soln` file in a temporary directory to prevent parallel race conditions.
- Cross-platform (macOS + Linux) wall-clock timeout and 4 GiB RSS memory cap.
- Exact error taxonomy (`none`, `unsolved`, `timeout`, `oom`, `infinite_h`, `crash_compile`, `crash_runtime`, `invalid_plan`).
- Resumable: skips `{config, domain, task, split, obf_level, seed, candidate_id}` entries already in `--out`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

from src.infra.schema import (
    ARCHIVED_CANDIDATE_DIRS,
    ARCHIVED_SELECTED_HEURISTICS,
    IPC2023_DOMAINS,
    REPO_ROOT,
    RunResult,
    build_canonical_splits,
    compute_agile_score,
    compute_expansion_score,
    save_canonical_splits,
)

PYPERPLAN_SCRIPT = REPO_ROOT / "src" / "pyperplan" / "pyperplan.py"
SPLITS_PATH = REPO_ROOT / "benchmarks" / "splits.json"

RE_EXPANSIONS = re.compile(r"(\d+)\s+Nodes expanded", re.IGNORECASE)
RE_PLAN_LEN = re.compile(r"Plan length:\s*(\d+)")
RE_SEARCH_TIME = re.compile(r"Search time:\s*([0-9.]+)")
RE_TOTAL_TIME = re.compile(r"Total time:\s*([0-9.]+)")
RE_INIT_H = re.compile(r"Initial h value:\s*([^\s\n]+)")
RE_PEAK_MEM = re.compile(r"Peak memory:\s*(\d+)\s*KB")

# Fast Downward regexes
RE_FD_EXPANSIONS = re.compile(r"Expanded\s+(\d+)\s+state\(s\)\.")
RE_FD_PLAN_LEN = re.compile(r"Plan length:\s*(\d+)\s+step\(s\)\.")
RE_FD_SEARCH_TIME = re.compile(r"Search time:\s*([0-9.]+)s")
RE_FD_TOTAL_TIME = re.compile(r"Total time:\s*([0-9.]+)s")
RE_FD_INIT_H = re.compile(r"Initial heuristic value for .*?:\s*([^\s\n]+)")
RE_FD_PEAK_MEM = re.compile(r"Peak memory:\s*(\d+)\s*KB")


def detect_python_interpreter(prefer_pypy: bool = True) -> str:
    if prefer_pypy:
        for cand in ("pypy3", "pypy3.11", "pypy3.10", "pypy"):
            found = shutil.which(cand)
            if found:
                return found
    return sys.executable


def check_heuristic_compiles(heuristic_arg: str) -> Optional[str]:
    """Return error message if any heuristic file has a syntax error or lacks a *Heuristic class."""
    for part in heuristic_arg.split(","):
        part = part.strip()
        if part in ("blind", "hff", "hffpo", "goalcount", "lmcut", "hmax", "hadd"):
            continue
        path = Path(part)
        if not path.is_file():
            return f"File not found: {part}"
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            compile(source, str(path), "exec")
            if "Heuristic" not in source:
                return "No *Heuristic class found"
        except Exception as exc:
            return f"Compile error: {exc}"
    return None


def run_process_with_limits(
    cmd: List[str],
    timeout_s: float,
    mem_limit_mb: int,
    cwd: Optional[Path] = None,
) -> Tuple[int, str, str, float, int, bool, bool]:
    """
    Execute subprocess while polling wall time and RSS memory (cross-platform macOS + Linux).
    Returns: (returncode, stdout, stderr, wall_time_s, peak_rss_kb, timed_out, oom_killed)
    """
    mem_limit_bytes = mem_limit_mb * 1024 * 1024
    start = time.monotonic()
    peak_rss_bytes = 0
    timed_out = False
    oom_killed = False

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    ps_proc = None
    if psutil is not None:
        try:
            ps_proc = psutil.Process(proc.pid)
        except Exception:
            ps_proc = None

    while True:
        ret = proc.poll()
        elapsed = time.monotonic() - start
        if ps_proc is not None:
            try:
                rss = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        pass
                if rss > peak_rss_bytes:
                    peak_rss_bytes = rss
                if rss > mem_limit_bytes:
                    oom_killed = True
                    proc.kill()
                    break
            except Exception:
                pass

        if ret is not None:
            break

        if elapsed > timeout_s:
            timed_out = True
            proc.kill()
            break

        time.sleep(0.05)

    stdout, stderr = proc.communicate()
    wall_time = round(time.monotonic() - start, 4)
    peak_rss_kb = int(peak_rss_bytes // 1024)
    return proc.returncode or 0, stdout, stderr, wall_time, peak_rss_kb, timed_out, oom_killed


def execute_single_job(job: Dict[str, Any]) -> RunResult:
    """Worker entrypoint for a single planner run (Pyperplan or Fast Downward)."""
    domain = job["domain"]
    task_file = Path(job["task_path"]).resolve()
    domain_file = Path(job["domain_path"]).resolve()
    task_name = task_file.name
    split = job["split"]
    config = job["config"]
    model = job["model"]
    obf_level = job["obf_level"]
    seed = int(job["seed"])
    candidate_id = str(job["candidate_id"])
    timeout_s = float(job["timeout_s"])
    mem_limit_mb = int(job["mem_limit_mb"])
    runner_type = job.get("runner_type", "pyperplan")

    if runner_type == "pyperplan":
        heuristic_arg = job["heuristic_arg"]
        compile_err = check_heuristic_compiles(heuristic_arg)
        if compile_err is not None:
            return RunResult(
                domain=domain,
                task=task_name,
                split=split,
                config=config,
                model=model,
                obf_level=obf_level,
                seed=seed,
                candidate_id=candidate_id,
                solved=False,
                time=0.0,
                search_time=0.0,
                expansions=0,
                plan_len=None,
                mem=0,
                init_h=None,
                error="crash_compile",
            )

        with tempfile.TemporaryDirectory(prefix=f"pyp_{domain}_{task_name}_") as tmpdir:
            plan_path = Path(tmpdir) / "plan.soln"
            search_alg = job.get("search", "gbf")
            if "," in heuristic_arg:
                search_alg = "gbf_alt"

            cmd = [
                job["python_bin"],
                str(PYPERPLAN_SCRIPT),
                "-H",
                heuristic_arg,
                "-s",
                search_alg,
                "--plan-file",
                str(plan_path),
                str(domain_file),
                str(task_file),
            ]
            rc, out, err, wall_time, peak_kb, timed_out, oom_killed = run_process_with_limits(
                cmd, timeout_s=timeout_s, mem_limit_mb=mem_limit_mb, cwd=REPO_ROOT
            )
            combined = out + "\n" + err

            m_exp = RE_EXPANSIONS.findall(combined)
            expansions = int(m_exp[-1]) if m_exp else None
            m_plen = RE_PLAN_LEN.search(combined)
            plan_len = int(m_plen.group(1)) if m_plen else None
            m_stime = RE_SEARCH_TIME.search(combined)
            search_time = float(m_stime.group(1)) if m_stime else None
            m_ttime = RE_TOTAL_TIME.search(combined)
            total_time = float(m_ttime.group(1)) if m_ttime else wall_time
            m_inith = RE_INIT_H.search(combined)
            init_h = m_inith.group(1) if m_inith else None
            m_mem = RE_PEAK_MEM.search(combined)
            mem_kb = int(m_mem.group(1)) if m_mem else (peak_kb if peak_kb > 0 else None)

            val_available = shutil.which("validate") is not None
            plan_exists = plan_path.is_file() and plan_path.stat().st_size > 0

            if timed_out:
                return RunResult(
                    domain=domain, task=task_name, split=split, config=config, model=model,
                    obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                    solved=False, time=timeout_s, search_time=search_time, expansions=expansions,
                    plan_len=None, mem=mem_kb, init_h=init_h, error="timeout",
                )
            if oom_killed or "MemoryError" in combined or "out of memory" in combined.lower():
                return RunResult(
                    domain=domain, task=task_name, split=split, config=config, model=model,
                    obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                    solved=False, time=wall_time, search_time=search_time, expansions=expansions,
                    plan_len=None, mem=mem_kb, init_h=init_h, error="oom",
                )
            if rc != 0 or "Traceback (most recent call last)" in combined:
                err_kind = "crash_compile" if ("SyntaxError" in combined or "ImportError" in combined) else "crash_runtime"
                return RunResult(
                    domain=domain, task=task_name, split=split, config=config, model=model,
                    obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                    solved=False, time=wall_time, search_time=search_time, expansions=expansions,
                    plan_len=None, mem=mem_kb, init_h=init_h, error=err_kind,
                )

            if plan_len is not None and plan_exists:
                if "Plan NOT correct" in combined:
                    return RunResult(
                        domain=domain, task=task_name, split=split, config=config, model=model,
                        obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                        solved=False, time=total_time, search_time=search_time, expansions=expansions,
                        plan_len=plan_len, mem=mem_kb, init_h=init_h, error="invalid_plan",
                    )
                if val_available and "Plan correct" not in combined:
                    return RunResult(
                        domain=domain, task=task_name, split=split, config=config, model=model,
                        obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                        solved=False, time=total_time, search_time=search_time, expansions=expansions,
                        plan_len=plan_len, mem=mem_kb, init_h=init_h, error="invalid_plan",
                    )
                return RunResult(
                    domain=domain, task=task_name, split=split, config=config, model=model,
                    obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                    solved=True, time=total_time, search_time=search_time, expansions=expansions,
                    plan_len=plan_len, mem=mem_kb, init_h=init_h, error="none",
                )

            # Unsolved cleanly: check if heuristic returned inf
            if init_h is not None and ("inf" in init_h.lower() or "nan" in init_h.lower()):
                err_type = "infinite_h"
            elif expansions is not None and expansions <= 1:
                err_type = "infinite_h"
            else:
                err_type = "unsolved"

            return RunResult(
                domain=domain, task=task_name, split=split, config=config, model=model,
                obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                solved=False, time=total_time, search_time=search_time, expansions=expansions,
                plan_len=None, mem=mem_kb, init_h=init_h, error=err_type,
            )

    elif runner_type == "fast-downward":
        fd_bin = job["fd_bin"]
        fd_mode = job["fd_mode"]  # "hff" or "lama-first"
        with tempfile.TemporaryDirectory(prefix=f"fd_{domain}_{task_name}_") as tmpdir:
            plan_path = Path(tmpdir) / "sas_plan"
            if fd_mode == "lama-first":
                cmd = [
                    fd_bin,
                    "--alias",
                    "lama-first",
                    "--plan-file",
                    str(plan_path),
                    str(domain_file),
                    str(task_file),
                ]
            else:
                cmd = [
                    fd_bin,
                    "--plan-file",
                    str(plan_path),
                    str(domain_file),
                    str(task_file),
                    "--evaluator",
                    "hff=ff()",
                    "--search",
                    "eager_greedy([hff])",
                ]
            rc, out, err, wall_time, peak_kb, timed_out, oom_killed = run_process_with_limits(
                cmd, timeout_s=timeout_s, mem_limit_mb=mem_limit_mb, cwd=Path(tmpdir)
            )
            combined = out + "\n" + err
            m_exp = RE_FD_EXPANSIONS.findall(combined)
            expansions = int(m_exp[-1]) if m_exp else None
            m_plen = RE_FD_PLAN_LEN.findall(combined)
            plan_len = int(m_plen[-1]) if m_plen else None
            m_stime = RE_FD_SEARCH_TIME.findall(combined)
            search_time = float(m_stime[-1]) if m_stime else None
            m_ttime = RE_FD_TOTAL_TIME.findall(combined)
            total_time = float(m_ttime[-1]) if m_ttime else wall_time
            m_inith = RE_FD_INIT_H.search(combined)
            init_h = m_inith.group(1) if m_inith else None
            m_mem = RE_FD_PEAK_MEM.findall(combined)
            mem_kb = int(m_mem[-1]) if m_mem else (peak_kb if peak_kb > 0 else None)

            if timed_out:
                return RunResult(
                    domain=domain, task=task_name, split=split, config=config, model=model,
                    obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                    solved=False, time=timeout_s, search_time=search_time, expansions=expansions,
                    plan_len=None, mem=mem_kb, init_h=init_h, error="timeout",
                )
            if oom_killed or "Memory limit has been reached" in combined:
                return RunResult(
                    domain=domain, task=task_name, split=split, config=config, model=model,
                    obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                    solved=False, time=wall_time, search_time=search_time, expansions=expansions,
                    plan_len=None, mem=mem_kb, init_h=init_h, error="oom",
                )
            if "Solution found!" in combined and plan_len is not None:
                return RunResult(
                    domain=domain, task=task_name, split=split, config=config, model=model,
                    obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                    solved=True, time=total_time, search_time=search_time, expansions=expansions,
                    plan_len=plan_len, mem=mem_kb, init_h=init_h, error="none",
                )
            return RunResult(
                domain=domain, task=task_name, split=split, config=config, model=model,
                obf_level=obf_level, seed=seed, candidate_id=candidate_id,
                solved=False, time=total_time, search_time=search_time, expansions=expansions,
                plan_len=None, mem=mem_kb, init_h=init_h, error="unsolved",
            )

    raise ValueError(f"Unsupported runner_type: {runner_type}")


def load_completed_keys(out_path: Path) -> set:
    completed = set()
    if not out_path.is_file():
        return completed
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (
                    rec.get("config"),
                    rec.get("domain"),
                    rec.get("task"),
                    rec.get("split"),
                    rec.get("obf_level"),
                    int(rec.get("seed", 1)),
                    str(rec.get("candidate_id")),
                )
                completed.add(key)
            except Exception:
                continue
    return completed


def select_best_candidates(
    jsonl_path: Path,
    rule: str = "paper_rule",
    top_k: int = 1,
) -> Dict[str, List[str]]:
    """
    Compare candidate selection rules over a JSONL file (Axis 1, 2, and 3):
    - `paper_rule`: highest coverage, tie-break by expansion score then agile score
    - `largest_train_only`: highest coverage on the largest 5 tasks per domain (p80..p99)
    - `fewest_expansions`: highest expansion score across solved tasks
    - `oracle`: highest coverage on test tasks (upper bound)
    """
    records_by_dom_cand: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("candidate_id")) == "baseline":
                continue
            records_by_dom_cand[rec["domain"]][str(rec["candidate_id"])].append(rec)

    selected: Dict[str, List[str]] = {}
    for domain, cand_map in sorted(records_by_dom_cand.items()):
        all_tasks = sorted({r["task"] for recs in cand_map.values() for r in recs})
        largest_tasks = set(all_tasks[-5:]) if len(all_tasks) >= 5 else set(all_tasks)

        scored = []
        for cand_id, recs in cand_map.items():
            t_max = 60.0 if any(r.get("split") == "train" for r in recs) else 300.0
            cov = sum(1 for r in recs if r.get("solved"))
            large_cov = sum(1 for r in recs if r.get("solved") and r["task"] in largest_tasks)
            exp_score = sum(
                compute_expansion_score(r.get("expansions"), bool(r.get("solved")))
                for r in recs
            )
            agl_score = sum(
                compute_agile_score(r.get("time"), bool(r.get("solved")), t_max=t_max)
                for r in recs
            )
            if rule == "paper_rule" or rule == "oracle":
                key = (cov, exp_score, agl_score)
            elif rule == "largest_train_only":
                key = (large_cov, cov, exp_score)
            elif rule == "fewest_expansions":
                key = (exp_score, cov, agl_score)
            else:
                raise ValueError(f"Unknown selection rule: {rule}")
            scored.append((key, cand_id))

        scored.sort(reverse=True)
        selected[domain] = [cid for _key, cid in scored[:top_k]]
    return selected


def discover_candidate_files(heur_dir: Path, domain: str, max_candidates: int = 25) -> List[Tuple[str, Path]]:
    """Find `<domain><idx>_10.py` or `<domain>*.py` candidate files in `heur_dir`."""
    matches: List[Tuple[int, str, Path]] = []
    for p in sorted(heur_dir.glob(f"{domain}*.py")):
        stem = p.stem
        suffix = stem[len(domain):].lstrip("-_")
        if suffix.startswith("obfuscated"):
            suffix = suffix[len("obfuscated"):].lstrip("-_")
        m = re.match(r"^(\d+)", suffix)
        order = int(m.group(1)) if m else 999
        cid = suffix or p.stem
        matches.append((order, cid, p))
    matches.sort(key=lambda x: (x[0], x[1]))
    return [(cid, path) for _ord, cid, path in matches[:max_candidates]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared parallel experiment runner (Member 4 infrastructure)")
    parser.add_argument(
        "--preset",
        choices=[
            "custom",
            "baselines",
            "pyperplan-blind",
            "pyperplan-hff",
            "paper-r1",
            "paper-v3",
            "paper-qwen14b",
            "fd-hff",
            "fd-lama-first",
            "archive-r1-candidates",
        ],
        default="custom",
        help="Pre-configured run mode.",
    )
    parser.add_argument("--config", default=None, help="Config label written to JSONL (e.g. 'axis1-L1-v3').")
    parser.add_argument("--model", default="none", help="Model name written to JSONL (e.g. 'deepseek-r1', 'qwen-7b').")
    parser.add_argument("--obf-level", default="L0", choices=["L0", "L1", "L2", "L3"], help="Obfuscation level.")
    parser.add_argument("--seed", type=int, default=1, help="Seed index (default: 1 for fast protocol).")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Benchmark split ('train' 20 tasks @ 60s or 'test' 30 tasks @ 300s).")
    parser.add_argument("--domains", default="all", help="Comma-separated domain names or 'all'.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap on tasks per domain (e.g. 2 for a quick smoke test).")
    parser.add_argument("--heuristic", default=None, help="Built-in heuristic ('blind', 'hff'), path to .py file, or comma-separated .py files for alternation.")
    parser.add_argument("--heuristic-dir", default=None, help="Directory containing candidate heuristic .py files (<domain><id>.py).")
    parser.add_argument("--max-candidates", type=int, default=10, help="Max candidates per domain when using --heuristic-dir (10 for Axis 1/2, 25 for Axis 3).")
    parser.add_argument("--benchmarks-dir", default=None, help="Optional custom benchmark split directory (e.g. benchmarks/obfuscated/L1/testing for Axis 1).")
    parser.add_argument("--time-limit", type=float, default=None, help="Override time limit in seconds (defaults: 60s train, 300s test).")
    parser.add_argument("--mem-limit-mb", type=int, default=4096, help="Memory limit per task in MiB (default: 4096).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes (default: 4).")
    parser.add_argument("--fd-bin", default="fast-downward.py", help="Path to fast-downward.py executable for FD baselines.")
    parser.add_argument("--out", default="results/runs.jsonl", help="Output JSONL path (appends & resumes automatically).")
    parser.add_argument("--select-from", default=None, help="Path to a JSONL file to run candidate selection rules on and exit.")
    parser.add_argument("--select-rule", default="paper_rule", choices=["paper_rule", "largest_train_only", "fewest_expansions", "oracle"])
    parser.add_argument("--select-top-k", type=int, default=1, help="Number of top candidates to select per domain (use 2 or 3 for Axis 3 alternation).")

    args = parser.parse_args()

    if args.select_from:
        sel = select_best_candidates(Path(args.select_from), rule=args.select_rule, top_k=args.select_top_k)
        print(json.dumps({"rule": args.select_rule, "top_k": args.select_top_k, "selected": sel}, indent=2))
        return

    if not SPLITS_PATH.is_file():
        save_canonical_splits(SPLITS_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_manifest = json.load(f)

    domains = IPC2023_DOMAINS if args.domains == "all" else [d.strip() for d in args.domains.split(",") if d.strip()]
    timeout_s = args.time_limit if args.time_limit is not None else (60.0 if args.split == "train" else 300.0)
    python_bin = detect_python_interpreter(prefer_pypy=True)

    default_bench_base = REPO_ROOT / "benchmarks" / "ipc2023-learning" / ("training" if args.split == "train" else "testing")
    bench_base = Path(args.benchmarks_dir).resolve() if args.benchmarks_dir else default_bench_base

    # Expand presets into concrete specifications
    specs: List[Dict[str, Any]] = []
    if args.preset == "baselines":
        specs.append({"mode": "pyperplan-builtin", "heur": "blind", "config": "pyperplan-blind", "model": "none"})
        specs.append({"mode": "pyperplan-builtin", "heur": "hff", "config": "pyperplan-hff", "model": "none"})
        specs.append({"mode": "archived-selected", "arch_key": "deepseek-r1", "config": "paper-r1", "model": "deepseek-r1"})
        if shutil.which(args.fd_bin) or Path(args.fd_bin).is_file():
            specs.append({"mode": "fd", "fd_mode": "hff", "config": "fd-hff", "model": "none"})
            specs.append({"mode": "fd", "fd_mode": "lama-first", "config": "fd-lama-first", "model": "none"})
    elif args.preset in ("pyperplan-blind", "pyperplan-hff"):
        h = "blind" if args.preset.endswith("blind") else "hff"
        specs.append({"mode": "pyperplan-builtin", "heur": h, "config": args.config or args.preset, "model": "none"})
    elif args.preset in ("paper-r1", "paper-v3", "paper-qwen14b"):
        key_map = {"paper-r1": "deepseek-r1", "paper-v3": "deepseek-v3", "paper-qwen14b": "qwen-14b"}
        ak = key_map[args.preset]
        specs.append({"mode": "archived-selected", "arch_key": ak, "config": args.config or args.preset, "model": ak})
    elif args.preset in ("fd-hff", "fd-lama-first"):
        fd_mode = "hff" if args.preset == "fd-hff" else "lama-first"
        specs.append({"mode": "fd", "fd_mode": fd_mode, "config": args.config or args.preset, "model": "none"})
    elif args.preset == "archive-r1-candidates":
        specs.append({
            "mode": "candidate-dir",
            "dir": REPO_ROOT / ARCHIVED_CANDIDATE_DIRS["deepseek-r1"],
            "config": args.config or "axis3-r1-candidates",
            "model": "deepseek-r1",
            "max_cands": 25,
        })
    else:
        if args.heuristic_dir:
            specs.append({
                "mode": "candidate-dir",
                "dir": Path(args.heuristic_dir).resolve(),
                "config": args.config or "custom-candidates",
                "model": args.model,
                "max_cands": args.max_candidates,
            })
        elif args.heuristic:
            specs.append({
                "mode": "single-heur",
                "heur": args.heuristic,
                "config": args.config or "custom",
                "model": args.model,
            })
        else:
            parser.error("Provide --preset, --heuristic, --heuristic-dir, or --select-from.")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(out_path)

    jobs: List[Dict[str, Any]] = []
    for spec in specs:
        for domain in domains:
            task_files = splits_manifest["domains"][domain][args.split]
            if args.max_tasks is not None:
                task_files = task_files[: args.max_tasks]
            domain_dir = bench_base / domain
            domain_path = domain_dir / "domain.pddl"

            if spec["mode"] == "pyperplan-builtin":
                cands = [("baseline", spec["heur"])]
            elif spec["mode"] == "archived-selected":
                info = ARCHIVED_SELECTED_HEURISTICS[spec["arch_key"]]
                hfile = REPO_ROOT / info["dir"] / info["domains"][domain]
                cid = Path(info["domains"][domain]).stem.replace(domain, "")
                cands = [(cid, str(hfile))]
            elif spec["mode"] == "single-heur":
                raw_heur = spec["heur"]
                if "," in raw_heur:
                    resolved_parts = [
                        str(Path(p.strip()).resolve()) if Path(p.strip()).exists() else p.strip()
                        for p in raw_heur.split(",")
                        if p.strip()
                    ]
                    cid = "alt"
                    h_resolved = ",".join(resolved_parts)
                else:
                    cid = Path(raw_heur).stem
                    h_resolved = str(Path(raw_heur).resolve()) if Path(raw_heur).exists() else raw_heur
                cands = [(cid, h_resolved)]
            elif spec["mode"] == "candidate-dir":
                found = discover_candidate_files(spec["dir"], domain, max_candidates=spec["max_cands"])
                cands = [(cid, str(p)) for cid, p in found]
            elif spec["mode"] == "fd":
                cands = [("baseline", spec["fd_mode"])]
            else:
                cands = []

            for cid, h_arg in cands:
                for tfile in task_files:
                    key = (spec["config"], domain, tfile, args.split, args.obf_level, args.seed, str(cid))
                    if key in completed:
                        continue
                    job = {
                        "domain": domain,
                        "task_path": str(domain_dir / tfile),
                        "domain_path": str(domain_path),
                        "split": args.split,
                        "config": spec["config"],
                        "model": spec["model"],
                        "obf_level": args.obf_level,
                        "seed": args.seed,
                        "candidate_id": str(cid),
                        "timeout_s": timeout_s,
                        "mem_limit_mb": args.mem_limit_mb,
                        "python_bin": python_bin,
                    }
                    if spec["mode"] == "fd":
                        job["runner_type"] = "fast-downward"
                        job["fd_bin"] = args.fd_bin
                        job["fd_mode"] = h_arg
                    else:
                        job["runner_type"] = "pyperplan"
                        job["heuristic_arg"] = h_arg
                    jobs.append(job)

    print(
        f"[Runner] Python={python_bin} | Jobs queued={len(jobs)} (skipped {len(completed)} completed) "
        f"| workers={args.workers} | timeout={timeout_s}s | out={out_path}"
    )
    if not jobs:
        return

    done_count = 0
    solved_count = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_job = {pool.submit(execute_single_job, j): j for j in jobs}
            for fut in concurrent.futures.as_completed(future_to_job):
                res = fut.result()
                out_f.write(res.to_json_line() + "\n")
                out_f.flush()
                done_count += 1
                if res.solved:
                    solved_count += 1
                if done_count % 5 == 0 or done_count == len(jobs):
                    print(
                        f"  [{done_count}/{len(jobs)}] last={res.config}:{res.domain}/{res.task} "
                        f"solved={res.solved} time={res.time}s err={res.error} (running solved={solved_count})"
                    )


if __name__ == "__main__":
    main()
