"""Generate candidate heuristics with an LLM for every (model, level, seed, domain).

Calls the authors' llm-heuristics.py unchanged (same prompt, same answer parsing) on the
suite <domain>-<level> (L0 = the original suite), with a neutral class name so the
heuristic's name does not leak the domain. Each candidate goes to
  axis1/heuristics/<model>/<level>/seed<k>/<domain>/hNN.py
and the full log (prompt, answer, token usage) to axis1/logs/... . Existing candidates
are skipped, so the script can be stopped and restarted. A summary line per call is
appended to axis1/heuristics/generation.jsonl (model id, date, duration, success).

Needs DEEPSEEK_API_KEY (or the key of the chosen framework) and `python -m
axis1.obfuscate` to have been run. Run from the repository root:

  # Axis 1 as planned: R1 x 3 seeds and V3 x 1 seed, levels L0-L3, n = 10
  python -m axis1.generate --jobs 4

  # smoke test: one configuration
  python -m axis1.generate --models deepseek-r1 --levels L1 --seeds 1 --domains blocksworld

  # no API: copy the paper's archived R1 candidates as L0 candidates (pipeline tests)
  python -m axis1.generate --from-archive --models deepseek-r1 --levels L0 --seeds 1
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1.config import (ARCHIVE_R1, DOMAINS, GEN_LEVELS, HEURISTIC_CLASS, HEURISTICS_DIR,
                          LOGS_DIR, MODELS, N_CANDIDATES, OUT, REPO_ROOT, candidate_path,
                          suite_name)

FRAMEWORK = {"deepseek-r1": "deepseek", "deepseek-v3": "deepseek"}
_lock = threading.Lock()


def generate_one(model, level, seed, domain, cid, framework, retries, temperature, top_p):
    target = candidate_path(model, level, seed, domain, cid)
    if target.exists():
        return "skip"
    target.parent.mkdir(parents=True, exist_ok=True)
    log = LOGS_DIR / target.relative_to(HEURISTICS_DIR).with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.part.py")   # llm-heuristics.py wants a .py name
    cmd = [sys.executable, "llm-heuristics.py", "--domain", suite_name(domain, level),
           "--framework", framework, "--model", model,
           "--heuristic-name", HEURISTIC_CLASS, "--heuristic-file", str(tmp),
           "--temperature", str(temperature), "--top-p", str(top_p)]
    status = "failed"
    for attempt in range(1, retries + 2):
        start = time.time()
        with open(log, "w") as f:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(target)
            status = "ok"
            break
        time.sleep(5 * attempt)
    with _lock, open(HEURISTICS_DIR / "generation.jsonl", "a") as f:
        f.write(json.dumps({
            "model": model, "framework": framework, "obf_level": level, "seed": seed,
            "domain": domain, "candidate_id": cid, "status": status, "attempts": attempt,
            "seconds": round(time.time() - start, 1), "temperature": temperature, "top_p": top_p,
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        }) + "\n")
    return status


def copy_from_archive(model, level, seed, domain, cid):
    """Pipeline tests without an API key: use the paper's archived R1 candidates at L0."""
    target = candidate_path(model, level, seed, domain, cid)
    if target.exists():
        return "skip"
    src = ARCHIVE_R1 / f"{domain}{cid}_10.py"
    if not src.exists():
        return "failed"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, target)
    return "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--levels", nargs="+", default=GEN_LEVELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="seed indices (default: 1..k with k from config.MODELS)")
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("-n", "--candidates", type=int, default=N_CANDIDATES)
    ap.add_argument("--framework", default=None, help="default: deepseek for deepseek-* models")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=4, help="parallel API calls")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--from-archive", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list what would be generated")
    args = ap.parse_args(argv)

    for level in args.levels:
        if level != "L0" and not (OUT / level / "mappings").exists():
            sys.exit(f"{OUT / level} missing: run `python -m axis1.obfuscate` first")

    work = []
    for model in args.models:
        seeds = args.seeds or range(1, MODELS.get(model, 1) + 1)
        for level in args.levels:
            for seed in seeds:
                for domain in args.domains:
                    for cid in range(1, args.candidates + 1):
                        if not candidate_path(model, level, seed, domain, cid).exists():
                            work.append((model, level, seed, domain, cid))
    print(f"{len(work)} candidates to generate")
    if args.dry_run:
        for w in work[:20]:
            print("  ", *w)
        return
    if not args.from_archive and not any(os.environ.get(k) for k in
                                         ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")):
        sys.exit("set DEEPSEEK_API_KEY (or the key for your framework) first")

    def task(w):
        model, level, seed, domain, cid = w
        if args.from_archive:
            return copy_from_archive(*w)
        fw = args.framework or FRAMEWORK.get(model, "deepseek")
        return generate_one(model, level, seed, domain, cid, fw, args.retries, args.temperature, args.top_p)

    counts = {}
    with ThreadPoolExecutor(max_workers=1 if args.from_archive else args.jobs) as pool:
        for i, status in enumerate(pool.map(task, work), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 10 == 0 or i == len(work):
                print(f"  {i}/{len(work)} {counts}", flush=True)


if __name__ == "__main__":
    main()
