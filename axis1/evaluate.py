"""Evaluate heuristics with Pyperplan (shared protocol). Run from the repository root.

Stages
  train     every candidate on the training tasks of its level, 60 s each
            -> axis1/results/train.jsonl          (input for select_best.py)
  test      the selected candidate per (model, level, seed, domain) on the 240-task
            test subset of its level, 5 min each
            -> axis1/results/test.jsonl
  control   Pyperplan h_FF and blind search on the test subset at every level, 5 min
            -> axis1/results/control.jsonl
            h_FF coverage must be the same at every level (obfuscator check); it is also
            the per-level reference for the LLM heuristics in F1.

Examples
  python -m axis1.evaluate train --jobs 6
  python -m axis1.select_best
  python -m axis1.evaluate test --jobs 6
  python -m axis1.evaluate control --jobs 6
  # smaller runs
  python -m axis1.evaluate train --models deepseek-r1 --levels L1 --domains blocksworld --n-train 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1.config import (DOMAINS, GEN_LEVELS, HEURISTICS_DIR, LEVELS, MEMORY_LIMIT_GIB, MODELS,
                          RESULTS, TEST_TIME_LIMIT_S, TRAIN_TIME_LIMIT_S, test_tasks, training_tasks)
from axis1.runner import run_jobs


def find_candidates(models, levels, seeds, domains):
    """Yield (model, level, seed, domain, cid, path) for every generated candidate."""
    for model in models:
        for level in levels:
            for seed_dir in sorted((HEURISTICS_DIR / model / level).glob("seed*")):
                seed = int(seed_dir.name[4:])
                if seeds and seed not in seeds:
                    continue
                for domain in domains:
                    for path in sorted((seed_dir / domain).glob("h[0-9][0-9].py")):
                        yield model, level, seed, domain, int(path.stem[1:]), path


def train_jobs(args):
    jobs = []
    for model, level, seed, domain, cid, path in find_candidates(args.models, args.levels, args.seeds, args.domains):
        tasks = training_tasks(domain)
        if args.n_train:
            tasks = tasks[:: max(1, len(tasks) // args.n_train)][: args.n_train]
        for rel in tasks:
            jobs.append(dict(stage="train", config="pyperplan-llm", model=model, obf_level=level,
                             seed=seed, candidate_id=cid, task=rel, heuristic=str(path)))
    return jobs


def test_jobs(args):
    sel_file = RESULTS / "selection.json"
    if not sel_file.exists():
        sys.exit("no selection.json: run `python -m axis1.select_best` first")
    jobs = []
    for s in json.loads(sel_file.read_text()):
        if (s["model"] not in args.models or s["obf_level"] not in args.levels
                or s["domain"] not in args.domains or (args.seeds and s["seed"] not in args.seeds)):
            continue
        for rel in test_tasks(s["domain"])[: args.n_test]:
            jobs.append(dict(stage="test", config="pyperplan-llm", model=s["model"], obf_level=s["obf_level"],
                             seed=s["seed"], candidate_id=s["candidate_id"], task=rel, heuristic=s["heuristic"]))
    return jobs


def control_jobs(args):
    jobs = []
    for level in args.levels:
        for domain in args.domains:
            for rel in test_tasks(domain)[: args.n_test]:
                for h in args.baselines:
                    jobs.append(dict(stage="control", config=f"pyperplan-{h}", model=None, obf_level=level,
                                     seed=None, candidate_id=None, task=rel, heuristic=h))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "test", "control"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--levels", nargs="+", default=None,
                    help="default: L0-L3 (generation levels for train/test, all levels for control)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--baselines", nargs="+", default=["hff", "blind"])
    ap.add_argument("--n-train", type=int, default=None, help="subsample training tasks (default: all 99)")
    ap.add_argument("--n-test", type=int, default=None, help="first N of the 30 test tasks per domain")
    ap.add_argument("--time-limit", type=int, default=None)
    ap.add_argument("--memory-gib", type=float, default=MEMORY_LIMIT_GIB)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--python", default=None, help="interpreter for Pyperplan (default: pypy3 if found)")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    args.levels = args.levels or (LEVELS if args.stage == "control" else GEN_LEVELS)

    jobs = {"train": train_jobs, "test": test_jobs, "control": control_jobs}[args.stage](args)
    limit = args.time_limit or (TRAIN_TIME_LIMIT_S if args.stage == "train" else TEST_TIME_LIMIT_S)
    out = args.out or RESULTS / f"{args.stage}.jsonl"
    run_jobs(jobs, out, limit, args.memory_gib, args.jobs, args.python, validate=not args.no_validate)


if __name__ == "__main__":
    main()
