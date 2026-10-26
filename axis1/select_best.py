"""Pick one heuristic per (model, level, seed, domain) with the paper's rule.

The rule (experiments/get-best-heuristics.py): most training tasks solved; ties broken
by the sum of expansion scores, then by the sum of agile scores, with the scores from
experiments/project.py. Candidates that never solve a training task are not selected;
if a configuration has no working candidate, it is reported and gets no test runs
(it counts as 0 coverage in the summary).

  python -m axis1.select_best            # reads axis1/results/train.jsonl
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1.config import RESULTS


def expansion_score(r):
    if not r["solved"] or not r.get("expansions"):
        return 0.0
    e = r["expansions"]
    if e <= 100:
        return 1.0
    if e > 1e6:
        return 0.0
    return 1 - math.log(e) / math.log(1e6)


def agile_score(r):
    if not r["solved"] or r.get("time") is None:
        return 0.0
    if r["time"] <= 1:
        return 1.0
    return 1 - math.log(r["time"]) / math.log(1800)


def select(rows):
    stats = defaultdict(lambda: [0, 0.0, 0.0, None])   # key -> coverage, exp score, agile, heuristic
    configs = set()
    for r in rows:
        cfg = (r["model"], r["obf_level"], r["seed"], r["domain"])
        configs.add(cfg)
        s = stats[cfg + (r["candidate_id"],)]
        s[0] += r["solved"]
        s[1] += expansion_score(r)
        s[2] += agile_score(r)
        s[3] = r["heuristic"]
    best = {}
    for key, (cov, es, ag, h) in stats.items():
        cfg, cid = key[:4], key[4]
        cand = (cov, es, ag, -cid)   # last: lower id wins remaining ties (deterministic)
        if cov > 0 and (cfg not in best or cand > best[cfg][0]):
            best[cfg] = (cand, cid, h)
    out, missing = [], []
    for cfg in sorted(configs, key=str):
        model, level, seed, domain = cfg
        if cfg not in best:
            missing.append(cfg)
            continue
        (cov, es, ag, _), cid, h = best[cfg]
        out.append(dict(model=model, obf_level=level, seed=seed, domain=domain, candidate_id=cid,
                        heuristic=h, train_coverage=cov, train_expansion_score=round(es, 3),
                        train_agile_score=round(ag, 3)))
    return out, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=RESULTS / "train.jsonl")
    ap.add_argument("--out", type=Path, default=RESULTS / "selection.json")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.train) if l.strip()]
    selection, missing = select(rows)
    args.out.write_text(json.dumps(selection, indent=1))
    print(f"{'model':12} {'level':5} {'seed':4} {'domain':12} {'cand':4} {'train cov':>9}")
    for s in selection:
        print(f"{s['model']:12} {s['obf_level']:5} {s['seed']:<4} {s['domain']:12} {s['candidate_id']:<4} "
              f"{s['train_coverage']:>9}")
    for m in missing:
        print("no working candidate:", *m)
    print(f"wrote {len(selection)} selections to {args.out}")


if __name__ == "__main__":
    main()
