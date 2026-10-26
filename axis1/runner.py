"""Run Pyperplan on many (heuristic, task) pairs in parallel and log one JSON line per run.

Every run follows the shared protocol: GBFS (gbfs_early_goal_test) in the authors'
Pyperplan, a wall-clock time limit, a memory limit (POSIX only), PYTHONHASHSEED=0, one
core per run. Pyperplan writes the plan next to the task, so each run works on its own
copy of the task in a temporary directory.

Solved plans are translated back to the original names and validated on the original
(L0) task, so a plan only counts if it is correct for the real problem. If VAL's
`validate` is on the PATH, Pyperplan also runs it on the obfuscated task.

Output lines follow the team's results format:
  {domain, task, config, model, obf_level, seed, candidate_id, solved, time,
   expansions, plan_len, mem, error}
plus `stage` (train/test/control) and `heuristic` (path). `error` is one of
crash, inf (dead end at the initial state), timeout, memout, unsolved, invalid-plan,
or null. Runs already present in the output file are skipped, so an interrupted
experiment resumes where it stopped.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1.config import OUT, PYPERPLAN, PYTHONHASHSEED, SEARCH, SRC
from axis1.validate_plan import check_plan, load_mappings, translate_plan

KEY_FIELDS = ("stage", "config", "model", "obf_level", "seed", "candidate_id", "task")


def default_python():
    """PyPy as in the paper if available, otherwise the current interpreter."""
    for name in ("pypy3.11", "pypy3.10", "pypy3"):
        exe = shutil.which(name)
        if exe:
            return exe
    return sys.executable


def run_key(job):
    return tuple(job.get(k) for k in KEY_FIELDS)


def done_keys(out_file):
    keys = set()
    if Path(out_file).exists():
        with open(out_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    keys.add(run_key(json.loads(line)))
    return keys


def _limit_memory(gib):
    def apply():
        try:
            import resource
            limit = int(gib * 1024 ** 3)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ImportError, ValueError, OSError):
            pass
    return apply


def run_one(job, time_limit, memory_gib, python, validate=True):
    """job: dict with stage, config, model, obf_level, seed, candidate_id, task, heuristic.

    `task` is relative to the benchmark root (training/<d>/pXX.pddl, testing/<d>/...);
    `heuristic` is a file path or a built-in name such as hff or blind.
    """
    level, rel = job["obf_level"], job["task"]
    domain = rel.split("/")[1]
    task_file = OUT / level / rel
    domain_file = task_file.parent / "domain.pddl"
    result = {k: job.get(k) for k in KEY_FIELDS}
    result.update(domain=domain, heuristic=str(job["heuristic"]), solved=False, time=None,
                  expansions=None, plan_len=None, mem=None, error=None)

    with tempfile.TemporaryDirectory() as tmp:
        d, t = Path(tmp) / "domain.pddl", Path(tmp) / "task.pddl"
        shutil.copy(domain_file, d)
        shutil.copy(task_file, t)
        cmd = [python, str(PYPERPLAN / "pyperplan.py"), "-s", SEARCH, "-H", str(job["heuristic"]), str(d), str(t)]
        env = dict(os.environ, PYTHONHASHSEED=PYTHONHASHSEED)
        preexec = _limit_memory(memory_gib) if os.name == "posix" else None
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit,
                                  env=env, preexec_fn=preexec, cwd=tmp)
            log = proc.stdout + proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            result.update(time=float(time_limit), error="timeout")
            return result
        result["time"] = round(time.time() - start, 3)

        m = re.search(r"(\d+) Nodes expanded", log)
        result["expansions"] = int(m.group(1)) if m else None
        m = re.search(r"Peak memory: (\d+) KB", log)
        result["mem"] = int(m.group(1)) if m else None
        soln = Path(str(t) + ".soln")

        if soln.exists():
            plan = [l for l in soln.read_text().splitlines() if l.strip()]
            result["plan_len"] = len(plan)
            result["solved"] = True
            if validate:
                try:
                    if level != "L0":
                        mapping, tasks = load_mappings(OUT, level, domain)
                        plan = translate_plan(plan, mapping, tasks[rel])
                    orig = SRC / rel
                    ok, _ = check_plan(orig.parent / "domain.pddl", orig, plan)
                except Exception:
                    ok = False
                if not ok:
                    result.update(solved=False, error="invalid-plan")
        elif "MemoryError" in log or "out of memory" in log.lower():
            result["error"] = "memout"
        elif "Initial state is a dead end" in log:
            result["error"] = "inf"
        elif returncode != 0 or "Traceback" in log:
            result["error"] = "crash"
        else:
            result["error"] = "unsolved"
    return result


def run_jobs(jobs, out_file, time_limit, memory_gib, n_jobs, python=None, validate=True):
    """Run all jobs not yet in out_file, appending results as they finish."""
    python = python or default_python()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    seen = done_keys(out_file)
    todo = [j for j in jobs if run_key(j) not in seen]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to go "
          f"(python={python}, {n_jobs} parallel, {time_limit}s, {memory_gib} GiB)", flush=True)
    if not todo:
        return
    solved = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as pool, open(out_file, "a") as out:
        futures = [pool.submit(run_one, j, time_limit, memory_gib, python, validate) for j in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            solved += r["solved"]
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {solved} solved", flush=True)
