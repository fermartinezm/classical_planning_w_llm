"""Tables for Axis 1: test coverage per domain and level, h_FF control, candidate failures.

Reads axis1/results/{test,control,train}.jsonl and selection.json; writes
axis1/results/summary.csv (one row per model x level x domain, used by plot_f1.py) and
prints Markdown tables for the report.

  python -m axis1.summarize
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from axis1.config import DOMAINS, LEVELS, RESULTS, test_tasks


def read(path):
    path = Path(path)
    return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def coverage_by(rows, key):
    cov = defaultdict(int)
    for r in rows:
        cov[key(r)] += r["solved"]
    return cov


def candidate_status(train_rows):
    """works = solves at least one training task; otherwise the most common failure."""
    runs = defaultdict(list)
    for r in train_rows:
        runs[(r["model"], r["obf_level"], r["seed"], r["domain"], r["candidate_id"])].append(r)
    status = {}
    for key, rs in runs.items():
        if any(r["solved"] for r in rs):
            status[key] = "works"
        else:
            errors = [r["error"] or "unsolved" for r in rs]
            status[key] = max(set(errors), key=errors.count)
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args(argv)

    test = read(args.results / "test.jsonl")
    control = read(args.results / "control.jsonl")
    train = read(args.results / "train.jsonl")
    sel_file = args.results / "selection.json"
    selection = json.loads(sel_file.read_text()) if sel_file.exists() else []
    n_tasks = {d: len(test_tasks(d)) for d in DOMAINS}
    # only domains that appear in the data, so partial runs do not show fake zeros
    present = {r["domain"] for r in train + test + control} | {s["domain"] for s in selection}
    domains = [d for d in DOMAINS if d in present]

    # seeds that were trained count, even without a working candidate (coverage 0)
    seeds = defaultdict(set)
    for r in train:
        seeds[(r["model"], r["obf_level"])].add(r["seed"])
    for s in selection:
        seeds[(s["model"], s["obf_level"])].add(s["seed"])

    llm = coverage_by(test, lambda r: (r["model"], r["obf_level"], r["seed"], r["domain"]))
    ctrl = coverage_by(control, lambda r: (r["config"], r["obf_level"], r["domain"]))
    ctrl_levels = sorted({r["obf_level"] for r in control})

    rows = []
    for (model, level), ss in sorted(seeds.items()):
        for d in domains:
            vals = [llm.get((model, level, s, d), 0) for s in sorted(ss)]
            rows.append(dict(model=model, obf_level=level, domain=d, n_tasks=n_tasks[d],
                             seeds=len(vals), mean=round(statistics.mean(vals), 2),
                             min=min(vals), max=max(vals),
                             hff=ctrl.get(("pyperplan-hff", level, d)),
                             blind=ctrl.get(("pyperplan-blind", level, d))))
    if rows:
        out = args.results / "summary.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}\n")

    # Table 1: LLM coverage (mean over seeds, [min-max]) per level, h_FF in brackets
    for model in sorted({r["model"] for r in rows}):
        levels = [l for l in LEVELS if any(r["model"] == model and r["obf_level"] == l for r in rows)]
        print(f"### Test coverage, {model} (mean over seeds [min-max]); h_FF at that level in ()\n")
        print("| domain | " + " | ".join(levels) + " |")
        print("|---" * (len(levels) + 1) + "|")
        totals = defaultdict(list)
        for d in domains + ["total"]:
            cells = []
            for l in levels:
                if d == "total":
                    ms = [r for r in rows if r["model"] == model and r["obf_level"] == l]
                    mean = sum(r["mean"] for r in ms)
                    hff = sum(r["hff"] or 0 for r in ms) if any(r["hff"] is not None for r in ms) else None
                    cells.append(f"**{mean:g}**" + (f" ({hff})" if hff is not None else ""))
                    continue
                r = next(r for r in rows if r["model"] == model and r["obf_level"] == l and r["domain"] == d)
                rng = f" [{r['min']}-{r['max']}]" if r["seeds"] > 1 else ""
                cells.append(f"{r['mean']:g}{rng}" + (f" ({r['hff']})" if r["hff"] is not None else ""))
            print(f"| {d} | " + " | ".join(cells) + " |")
        print()

    # Table 2: control coverage per level (h_FF must not change)
    if control:
        print("### Control: Pyperplan coverage on the test subset per level\n")
        print("| config | domain | " + " | ".join(ctrl_levels) + " | same? |")
        print("|---" * (len(ctrl_levels) + 3) + "|")
        for cfg in sorted({r["config"] for r in control}):
            for d in domains + ["total"]:
                if d == "total":
                    vals = [sum(ctrl.get((cfg, l, dd), 0) for dd in domains) for l in ctrl_levels]
                else:
                    vals = [ctrl.get((cfg, l, d), 0) for l in ctrl_levels]
                print(f"| {cfg} | {d} | " + " | ".join(map(str, vals)) + f" | {'yes' if len(set(vals)) == 1 else 'NO'} |")
        print()

    # Table 3: candidate failure taxonomy from the training runs
    if train:
        status = candidate_status(train)
        kinds = ["works", "crash", "inf", "timeout", "memout", "unsolved", "invalid-plan"]
        print("### Generated candidates by outcome on the training tasks\n")
        print("| model | level | " + " | ".join(kinds) + " |")
        print("|---" * (len(kinds) + 2) + "|")
        for model, level in sorted({k[:2] for k in status}):
            c = defaultdict(int)
            for k, v in status.items():
                if k[:2] == (model, level):
                    c[v] += 1
            print(f"| {model} | {level} | " + " | ".join(str(c[k]) for k in kinds) + " |")


if __name__ == "__main__":
    main()
