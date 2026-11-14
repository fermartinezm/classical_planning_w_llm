#!/usr/bin/env python3
"""
End-to-end pipeline for LLM-generated heuristics under the shared protocol:
generate candidates -> evaluate on training tasks -> select -> evaluate on test tasks.

Works for any model (DeepSeek, Gemini, OpenAI, or any OpenAI-compatible endpoint such as
Ollama/OpenRouter) and any benchmark version (L0 original, L1-L3 obfuscated, extra
renaming seeds). Defaults are Axis 1's (--name axis1 --owner member1); another axis can
reuse it with e.g. --name axis2 --owner member2.

Every step is resumable: re-running a step skips work that is already done
(existing candidate files, JSONL lines already written).

Steps
-----
  hffcheck  h_FF sanity check on the 20 training tasks/domain (60 s) for L0-L3.
            Renaming must not change h_FF's coverage (small differences in expansions
            are expected from name-based tie-breaking; L2 also reorders actions).
  gen       Generate n candidate heuristics per domain with the LLM (domains in parallel).
  train     Evaluate all candidates on the 20 training tasks/domain (60 s).
  select    Pick the best candidate per domain (paper rule) and copy it to *_selected/.
  test      Evaluate the selected heuristic on the 30 test tasks/domain (300 s).
  all       gen -> train -> select -> test

Examples
--------
  python pipeline.py hffcheck --workers 4
  python pipeline.py gen   --level L1
  python pipeline.py all   --level L1 --workers 4
  python pipeline.py all   --level L1 --seed 2      # second random renaming

Outputs
-------
  (<name>=axis1, <owner>=member1 by default)
  generated/<name>_<L>[s<seed>]_<tag>/                 candidates (<domain><i>_<n>.py)
  generated/<name>_<L>[s<seed>]_<tag>_selected/        selected heuristic per domain
  results/<owner>_<name>_train_<L>[s<seed>]_<tag>.jsonl training runs (one file per level
                                                       and model, so ids never mix)
  results/<owner>_<name>_selection.json                which candidate was selected
  results/<owner>_<name>_test.jsonl                    test runs (all levels, input of F1)
  results/<owner>_hff_check.jsonl                      h_FF sanity check
  logs/<name>/                                         prompts and LLM answers
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a local .env (git-ignored) into os.environ."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


DOMAINS = ["blocksworld", "childsnack", "floortile", "miconic",
           "rovers", "sokoban", "spanner", "transport"]


def level_tag(level: str, seed: int) -> str:
    return level if seed == 1 else f"{level}s{seed}"


def bench_dir(level: str, seed: int, split: str) -> Path:
    sub = "training" if split == "train" else "testing"
    if level == "L0":
        return ROOT / "benchmarks" / "ipc2023-learning" / sub
    lvl = level if seed == 1 else f"{level}-s{seed}"
    return ROOT / "benchmarks" / "obfuscated" / lvl / sub


def cfg_name(args) -> str:
    """Config prefix: extra renaming seeds get their own prefix (axis1s2-...), so plots that
    select 'axis1-*' only see the first renaming and never merge several renamings."""
    return args.name if args.seed == 1 else f"{args.name}s{args.seed}"


def gen_dir(args) -> Path:
    return ROOT / "generated" / f"{args.name}_{level_tag(args.level, args.seed)}_{args.tag}"


def sel_dir(args) -> Path:
    return ROOT / "generated" / f"{args.name}_{level_tag(args.level, args.seed)}_{args.tag}_selected"


def train_out(args) -> Path:
    return ROOT / "results" / f"{args.owner}_{args.name}_train_{level_tag(args.level, args.seed)}_{args.tag}.jsonl"


def test_out(args) -> Path:
    # one test file per model tag, e.g. results/member1_axis1_test_g5mini.jsonl
    return ROOT / "results" / f"{args.owner}_{args.name}_test_{args.tag}.jsonl"


def sel_out(args) -> Path:
    return ROOT / "results" / f"{args.owner}_{args.name}_selection.json"


def hff_out(args) -> Path:
    return ROOT / "results" / f"{args.owner}_hff_check.jsonl"


def run(cmd, log=None) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if log is None:
        return subprocess.call([str(c) for c in cmd], cwd=ROOT)
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.call([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)


def domains_of(args):
    return DOMAINS if args.domains == "all" else args.domains.split(",")


# ----------------------------------------------------------------------------- steps
def step_hffcheck(args):
    """Step 1.2: h_FF must solve the same tasks at every level. --split train (default, cheap)
    or --split test (the 240 test tasks, as in TEAM_GUIDE; slow: use --time-limit to cap it)."""
    levels = ["L0", "L1", "L2", "L3"] if args.level == "all" else [args.level]
    for lvl in levels:
        print(f"[hffcheck] {lvl} ({args.split})")
        cmd = [PY, "run_experiments.py", "--preset", "pyperplan-hff", "--split", args.split,
               "--obf-level", lvl, "--seed", args.seed, "--config", f"{args.name}-hffcheck",
               "--domains", args.domains,
               "--benchmarks-dir", bench_dir(lvl, args.seed, args.split),
               "--workers", args.workers, "--out", hff_out(args)]
        if args.time_limit:
            cmd += ["--time-limit", args.time_limit]
        run(cmd)
    # summary
    cov = {}
    with open(hff_out(args), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != args.split:
                continue
            key = (r["obf_level"], r["domain"])
            cov.setdefault(key, 0)
            cov[key] += int(bool(r["solved"]))
    print(f"\nh_FF coverage on {args.split} tasks:")
    print(f"{'domain':12s} " + " ".join(f"{l:>4s}" for l in ["L0", "L1", "L2", "L3"]))
    for d in DOMAINS:
        print(f"{d:12s} " + " ".join(f"{cov.get((l, d), '-'):>4}" for l in ["L0", "L1", "L2", "L3"]))


def step_gen(args):
    out = gen_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    logdir = ROOT / "logs" / args.name
    logdir.mkdir(parents=True, exist_ok=True)

    def one(domain):
        cmd = [PY, "llm-heuristics.py", "--domain", domain,
               "--framework", args.framework, "--model", args.model,
               "--num-candidates", args.n, "--output-dir", out]
        if args.base_url:
            cmd += ["--base-url", args.base_url]
        if args.level != "L0":
            cmd += ["--suite-dir", bench_dir(args.level, args.seed, "train") / domain]
        log = logdir / f"gen_{level_tag(args.level, args.seed)}_{args.tag}_{domain}.log"
        for attempt in range(1, args.retries + 1):   # retries resume: existing files are skipped
            rc = run(cmd, log)
            have = len(list(out.glob(f"{domain}*_{args.n}.py")))
            if have >= args.n:
                break
            tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
            if "NOT_FOUND" in tail or "API key not valid" in tail or "PERMISSION_DENIED" in tail:
                print(f"  [gen] {domain}: fatal API error (wrong model name or key), see {log}")
                break
            if "per day" in tail or "Insufficient Balance" in tail:
                print(f"  [gen] {domain}: daily quota / balance exhausted, see {log}")
                break
            wait = min(60 * attempt, 300)       # rate limits on free tiers (429)
            print(f"  [gen] {domain}: {have}/{args.n} after attempt {attempt} (rc={rc}), retry in {wait}s")
            time.sleep(wait)
        return domain, len(list(out.glob(f"{domain}*_{args.n}.py")))

    doms = domains_of(args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.gen_parallel) as pool:
        for domain, have in pool.map(one, doms):
            print(f"[gen] {args.level} {domain}: {have}/{args.n} candidates in {out}")


def step_train(args):
    run([PY, "run_experiments.py", "--heuristic-dir", gen_dir(args),
         "--max-candidates", args.n, "--split", "train",
         "--obf-level", args.level, "--seed", args.seed,
         "--config", f"{cfg_name(args)}-{args.level}-{args.tag}-train", "--model", args.model,
         "--domains", args.domains,
         "--benchmarks-dir", bench_dir(args.level, args.seed, "train"),
         "--workers", args.workers, "--out", train_out(args)])


def step_select(args):
    if not train_out(args).is_file():
        print(f"[select] {train_out(args).name} does not exist (no training runs): skipping")
        return
    sys.path.insert(0, str(ROOT))
    from run_experiments import select_best_candidates  # noqa: E402

    sel = select_best_candidates(train_out(args), rule="paper_rule", top_k=1)
    sdir = sel_dir(args)
    sdir.mkdir(parents=True, exist_ok=True)
    record = {}
    for domain, cids in sel.items():
        if not cids:
            continue
        cid = cids[0]
        src = gen_dir(args) / f"{domain}{cid}.py"
        if not src.is_file():
            print(f"  [select] WARNING: {src} not found")
            continue
        shutil.copy2(src, sdir / src.name)
        # training coverage of the selected candidate
        solved = 0
        with open(train_out(args), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["domain"] == domain and str(r["candidate_id"]) == cid and r["solved"]:
                    solved += 1
        record[domain] = {"candidate": cid, "train_solved": solved}
        print(f"[select] {args.level} {domain:12s} -> {src.name} (train solved {solved}/20)")

    allsel = json.loads(sel_out(args).read_text(encoding="utf-8")) if sel_out(args).is_file() else {}
    allsel[f"{args.name}-{level_tag(args.level, args.seed)}-{args.tag}"] = {
        "date": datetime.now().isoformat(timespec="seconds"), "model": args.model,
        "selected": record}
    sel_out(args).write_text(json.dumps(allsel, indent=2), encoding="utf-8")


def step_test(args):
    run([PY, "run_experiments.py", "--heuristic-dir", sel_dir(args),
         "--max-candidates", 1, "--split", "test",
         "--obf-level", args.level, "--seed", args.seed,
         "--config", f"{cfg_name(args)}-{args.level}-{args.tag}", "--model", args.model,
         "--domains", args.domains,
         "--benchmarks-dir", bench_dir(args.level, args.seed, "test"),
         "--workers", args.workers, "--out", test_out(args)])


def main():
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["hffcheck", "gen", "train", "select", "test", "all"])
    ap.add_argument("--level", default="all", help="L0, L1, L2, L3 (hffcheck also accepts 'all')")
    ap.add_argument("--seed", type=int, default=1, help="Renaming seed (1 = benchmarks/obfuscated/<L>)")
    ap.add_argument("--domains", default="all", help="Comma-separated list or 'all'")
    ap.add_argument("--framework", default="deepseek")
    ap.add_argument("--model", default="deepseek-v3")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--tag", default="v3", help="Short model tag used in folder/config names")
    ap.add_argument("--name", default="axis1", help="Experiment name: prefix of configs, folders and files")
    ap.add_argument("--owner", default="member1", help="Prefix of the results files (team convention)")
    ap.add_argument("-n", type=int, default=10, help="Candidates per domain")
    ap.add_argument("--workers", type=int, default=4, help="Parallel planner runs")
    ap.add_argument("--gen-parallel", type=int, default=8, help="Domains generated in parallel")
    ap.add_argument("--split", default="train", choices=["train", "test"], help="hffcheck only")
    ap.add_argument("--time-limit", type=float, default=None, help="hffcheck only: cap per task (s)")
    ap.add_argument("--retries", type=int, default=8, help="Generation attempts per domain (waits between them)")
    args = ap.parse_args()

    if args.step != "hffcheck" and args.level not in ("L0", "L1", "L2", "L3"):
        ap.error("--level must be one of L0, L1, L2, L3 for this step")
    if args.step in ("gen", "all") and args.framework == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        ap.error("DEEPSEEK_API_KEY is not set in this terminal")
    if args.level not in ("L0", "all") and not bench_dir(args.level, args.seed, "train").is_dir():
        ap.error(f"{bench_dir(args.level, args.seed, 'train')} missing: run obfuscate.py first")

    steps = ["gen", "train", "select", "test"] if args.step == "all" else [args.step]
    for s in steps:
        print(f"\n=== {s} ({args.level}, seed {args.seed}) ===", flush=True)
        globals()[f"step_{s}"](args)


if __name__ == "__main__":
    main()
