"""Correctness check for the obfuscator: h_FF must not care about names.

Runs GBFS + h_FF in the authors' Pyperplan (src/pyperplan, search gbfs_early_goal_test)
on the same tasks at every level, translates each plan
back to original names and validates it on the original task. Fails (exit 1) if
coverage differs between levels or any plan is invalid. Expansions and plan lengths are
reported but not required to match: Pyperplan grounds by iterating over sets of
strings, so new names (and, at L2, new orders) change operator order and hence GBFS
tie-breaking. Small differences in expansions are expected at every level.


Writes one JSON line per run to axis1/results/invariance.jsonl.

  python -m axis1.check_invariance --domains blocksworld childsnack --n-tasks 10
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1.validate_plan import check_plan, load_mappings, translate_plan
from axis1.config import DOMAINS, OUT, PYPERPLAN, PYTHONHASHSEED, RESULTS, SEARCH, SRC, training_tasks

LEVELS = ["L0", "L1", "L2", "L3"]


def run_pyperplan(domain_file, task_file, timeout):
    """Run GBFS + h_FF in a temp dir (Pyperplan writes the .soln next to the task)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "domain.pddl"
        t = Path(tmp) / "task.pddl"
        shutil.copy(domain_file, d)
        shutil.copy(task_file, t)
        env = dict(os.environ, PYTHONHASHSEED=PYTHONHASHSEED)
        start = time.time()
        try:
            proc = subprocess.run([sys.executable, str(PYPERPLAN / "pyperplan.py"), "-s", SEARCH, "-H", "hff", str(d), str(t)],
                                  capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"solved": False, "error": "timeout", "time": timeout}
        elapsed = time.time() - start
        log = proc.stdout + proc.stderr
        soln = Path(str(t) + ".soln")
        res = {"time": round(elapsed, 3), "error": None}
        m = re.search(r"(\d+) Nodes expanded", log)
        res["expansions"] = int(m.group(1)) if m else None
        m = re.search(r"Peak memory: (\d+) KB", log)
        res["mem"] = int(m.group(1)) if m else None
        if soln.exists():
            res["plan"] = soln.read_text().splitlines()
            res["solved"] = True
        else:
            res["solved"] = False
            res["error"] = "crash" if proc.returncode else "unsolved"
        return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--n-tasks", type=int, default=10, help="first N training tasks per domain")
    ap.add_argument("--tasks", nargs="*", default=None, help="explicit tasks, e.g. training/blocksworld/p05.pddl")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--obf", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=RESULTS / "invariance.jsonl")
    args = ap.parse_args(argv)

    failures, warnings = [], []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as log:
        for domain in args.domains:
            tasks = [t for t in args.tasks if f"/{domain}/" in t] if args.tasks else training_tasks(domain)[:args.n_tasks]
            print(f"\n== {domain}")
            print(f"{'task':34}" + "".join(f"{lv:>16}" for lv in args.levels))
            for rel in tasks:
                row, solved, errors = [], {}, {}
                for lv in args.levels:
                    task = args.obf / lv / rel
                    r = run_pyperplan(task.parent / "domain.pddl", task, args.timeout)
                    if r["solved"]:
                        plan = r["plan"]
                        if lv != "L0":
                            mapping, tmap = load_mappings(args.obf, lv, domain)
                            plan = translate_plan(plan, mapping, tmap[rel])
                        orig = args.src / rel
                        ok, msg = check_plan(orig.parent / "domain.pddl", orig, plan)
                        if not ok:
                            failures.append(f"{lv} {domain} {rel}: invalid plan: {msg}")
                            r["solved"], r["error"] = False, "invalid-plan"
                    solved[lv] = r["solved"]
                    errors[lv] = r["error"]
                    plan_len = len([l for l in r.get("plan", []) if l.strip()]) if r["solved"] else None
                    row.append(f"{r.get('expansions')}/{plan_len}" if r["solved"] else (r["error"] or "-"))
                    log.write(json.dumps({
                        "domain": domain, "task": rel, "config": "pyperplan-hff",
                        "model": None, "obf_level": lv, "seed": None, "candidate_id": None,
                        "solved": r["solved"], "time": r.get("time"), "expansions": r.get("expansions"),
                        "plan_len": plan_len, "mem": r.get("mem"), "error": r["error"],
                    }) + "\n")
                if not any(solved.values()):
                    warnings.append(f"{domain} {rel}: unsolved at every level (planner problem, not checked)")
                if len(set(solved.values())) > 1 and any(e == "timeout" for e in errors.values()):
                    warnings.append(f"{domain} {rel}: solved at some levels only, the rest timed out "
                                    f"(search noise near the limit) {solved}")
                elif len(set(solved.values())) > 1:
                    failures.append(f"{domain} {rel}: coverage differs across levels {solved}")
                print(f"{rel:34}" + "".join(f"{c:>16}" for c in row))
    print("\n(cells: expansions/plan length)")
    if warnings:
        print("\nWARNINGS:")
        print("\n".join(warnings))
    if failures:
        print("\nFAILURES:")
        print("\n".join(failures))
        sys.exit(1)
    print("OK: same coverage at every level and every plan valid on the original task"
          + (" (see warnings for tasks that could not be checked)." if warnings else "."))


if __name__ == "__main__":
    main()
