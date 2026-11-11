#!/usr/bin/env python3
"""
Axis 1 (obfuscation) summary tables and figures from the shared JSON-lines results.

Reads (whatever exists):
  results/member1_hff_check.jsonl          h_FF sanity check (training tasks)
  results/member1_axis1_train_*.jsonl      every candidate on the training tasks
  results/member1_axis1_test.jsonl         selected heuristic on the test tasks
  results/member4_baselines.jsonl          Pyperplan h_FF on the test tasks (Member 4)

Writes to plots/output/axis1/:
  summary.md                     all tables in Markdown (paste into the report)
  table_coverage.tex             test coverage per domain and level (LaTeX)
  F1b_candidate_failures.pdf/png share of candidates per failure type and level
  F1c_renaming_seeds.pdf/png     test coverage per renaming seed (if seeds > 1 were run)

Usage (from the repository root):
  python plots/obfuscation_summary.py
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "output" / "axis1"
DOMAINS = ["blocksworld", "childsnack", "floortile", "miconic",
           "rovers", "sokoban", "spanner", "transport"]
LEVELS = ["L0", "L1", "L2", "L3"]
FAILURES = ["crash_compile", "crash_runtime", "infinite_h", "oom", "timeout", "unsolved"]


def load(pattern: str):
    recs = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        with open(path, encoding="utf-8") as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- sections
def hff_section(recs):
    if not recs:
        return "## h_FF sanity check\n\n(no results/member1_hff_check.jsonl yet)\n"
    cov = defaultdict(int)
    n = defaultdict(int)
    exp = {}
    for r in recs:
        k = (r["obf_level"], r["domain"])
        n[k] += 1
        cov[k] += int(bool(r["solved"]))
        if r["solved"]:
            exp[(r["obf_level"], r["domain"], r["task"])] = r["expansions"]
    rows = []
    for d in DOMAINS:
        row = [d] + [f"{cov[(l, d)]}/{n[(l, d)]}" if n[(l, d)] else "-" for l in LEVELS]
        for l in LEVELS[1:]:
            pairs = [(exp[("L0", d, t)], exp[(l, d, t)]) for (lv, dd, t) in exp
                     if lv == l and dd == d and ("L0", d, t) in exp]
            same = sum(1 for a, b in pairs if a == b)
            row.append(f"{same}/{len(pairs)}" if pairs else "-")
        rows.append(row)
    header = ["domain"] + [f"solved {l}" for l in LEVELS] + [f"same exp. {l}" for l in LEVELS[1:]]
    return ("## h_FF sanity check (training tasks, 60 s)\n\n"
            "Renaming must not change what h_FF solves. 'same exp.' counts tasks whose number "
            "of expanded nodes is identical to L0; differences come only from tie-breaking, "
            "which depends on names (L1, L3) and on the order of actions and facts (L2).\n\n"
            + md_table(header, rows) + "\n")


def candidate_section(recs):
    """Per level: how many candidates per domain run at all, and failure taxonomy."""
    if not recs:
        return "## Candidates on training tasks\n\n(no training results yet)\n", {}
    by = defaultdict(lambda: defaultdict(list))  # (level, seed) -> (domain, cand) -> recs
    for r in recs:
        by[(r["obf_level"], int(r.get("seed", 1)))][(r["domain"], str(r["candidate_id"]))].append(r)

    rows_cov, rows_fail, shares = [], [], {}
    for key in sorted(by):
        lvl, seed = key
        label = lvl if seed == 1 else f"{lvl} (seed {seed})"
        cands = by[key]
        # coverage of the best candidate per domain
        best = {}
        for (d, c), rs in cands.items():
            best[d] = max(best.get(d, 0), sum(1 for r in rs if r["solved"]))
        working = defaultdict(int)
        total = defaultdict(int)
        for (d, c), rs in cands.items():
            total[d] += 1
            if any(r["solved"] for r in rs):
                working[d] += 1
        rows_cov.append([label] + [f"{best.get(d, '-')} ({working[d]}/{total[d]})" if total[d] else "-"
                                   for d in DOMAINS])
        # failure taxonomy over all candidate x task runs
        cnt = defaultdict(int)
        for rs in cands.values():
            for r in rs:
                cnt["solved" if r["solved"] else r.get("error", "unsolved")] += 1
        tot = sum(cnt.values())
        shares[label] = {k: cnt[k] / tot for k in ["solved"] + FAILURES}
        rows_fail.append([label, tot] + [f"{100 * cnt[k] / tot:.1f}%" for k in ["solved"] + FAILURES])
    txt = ("## Candidates on training tasks (20 tasks/domain, 60 s)\n\n"
           "Cell = training coverage of the best candidate (working candidates / generated), "
           "where a working candidate solves at least one training task.\n\n"
           + md_table(["level"] + DOMAINS, rows_cov)
           + "\n\nOutcome of every candidate x task run:\n\n"
           + md_table(["level", "runs", "solved"] + FAILURES, rows_fail) + "\n")
    return txt, shares


def test_section(test_recs, base_recs):
    hff = defaultdict(int)
    for r in base_recs:
        if r.get("config") == "pyperplan-hff" and r.get("split") == "test" and r.get("solved"):
            hff[r["domain"]] += 1
    cov = defaultdict(lambda: defaultdict(int))
    exp = defaultdict(list)
    for r in test_recs:
        if not str(r.get("config", "")).startswith("axis1"):
            continue
        seed = int(r.get("seed", 1))
        key = (r["obf_level"], seed)
        cov[key][r["domain"]] += int(bool(r["solved"]))
        cov[key]["_n_" + r["domain"]] += 1
    if not cov:
        return "## Test coverage\n\n(no results/member1_axis1_test.jsonl yet)\n", cov, hff
    rows = []
    for key in sorted(cov):
        lvl, seed = key
        label = lvl if seed == 1 else f"{lvl} (seed {seed})"
        vals = [cov[key][d] if cov[key].get("_n_" + d) else "-" for d in DOMAINS]
        total = sum(v for v in vals if isinstance(v, int))
        rows.append([label] + vals + [total])
    if hff:
        rows.append(["Pyperplan h_FF (L0)"] + [hff.get(d, 0) for d in DOMAINS] + [sum(hff.values())])
    return ("## Test coverage (30 tasks/domain, 300 s)\n\n"
            + md_table(["level"] + DOMAINS + ["total"], rows) + "\n"), cov, hff


def latex_table(cov, hff):
    keys = [k for k in sorted(cov) if k[1] == 1]
    if not keys:
        return ""
    short = {"blocksworld": "Blocks", "childsnack": "Childsn.", "floortile": "Floort.",
             "miconic": "Miconic", "rovers": "Rovers", "sokoban": "Sokoban",
             "spanner": "Spanner", "transport": "Transp."}
    lines = [r"\begin{tabular}{l" + "r" * (len(DOMAINS) + 1) + "}", r"\toprule",
             "Level & " + " & ".join(short[d] for d in DOMAINS) + r" & Total \\", r"\midrule"]
    for k in keys:
        vals = [cov[k].get(d, 0) for d in DOMAINS]
        lines.append(f"{k[0]} & " + " & ".join(map(str, vals)) + f" & {sum(vals)} " + r"\\")
    if hff:
        vals = [hff.get(d, 0) for d in DOMAINS]
        lines.append(r"\midrule")
        lines.append(r"$h^{\mathrm{FF}}$ & " + " & ".join(map(str, vals)) + f" & {sum(vals)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- figures
def fig_failures(shares):
    if not shares:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(shares)
    cats = ["solved"] + FAILURES
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    left = [0.0] * len(labels)
    for c in cats:
        vals = [100 * shares[l][c] for l in labels]
        ax.barh(labels, vals, left=left, label=c.replace("_", " "))
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("Share of candidate x training-task runs (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax.set_title("Outcome of generated candidates per obfuscation level")
    fig.tight_layout()
    fig.savefig(OUT / "F1b_candidate_failures.pdf")
    fig.savefig(OUT / "F1b_candidate_failures.png", dpi=200)
    plt.close(fig)


def fig_seeds(cov):
    by_level = defaultdict(dict)
    for (lvl, seed), c in cov.items():
        by_level[lvl][seed] = sum(c.get(d, 0) for d in DOMAINS)
    multi = {l: s for l, s in by_level.items() if len(s) > 1}
    if not multi:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for i, (lvl, seeds) in enumerate(sorted(multi.items())):
        ys = list(seeds.values())
        ax.scatter([i] * len(ys), ys, zorder=3)
        ax.hlines(mean(ys), i - 0.2, i + 0.2, color="k")
    ax.set_xticks(range(len(multi)))
    ax.set_xticklabels(sorted(multi))
    ax.set_ylabel("Test tasks solved (total)")
    ax.set_title("Different random renamings of the same level")
    fig.tight_layout()
    fig.savefig(OUT / "F1c_renaming_seeds.pdf")
    fig.savefig(OUT / "F1c_renaming_seeds.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hff = load("results/member1_hff_check.jsonl")
    train = load("results/member1_axis1_train_*.jsonl")
    test = load("results/member1_axis1_test.jsonl")
    base = load("results/member4_baselines.jsonl")

    parts = ["# Axis 1 results summary\n"]
    sel = ROOT / "results" / "member1_axis1_selection.json"
    parts.append(hff_section(hff))
    txt, shares = candidate_section(train)
    parts.append(txt)
    if sel.is_file():
        data = json.loads(sel.read_text(encoding="utf-8"))
        rows = []
        for cfg, info in data.items():
            for d, s in info["selected"].items():
                rows.append([cfg, d, s["candidate"], f"{s['train_solved']}/20"])
        parts.append("## Selected candidates\n\n" + md_table(["config", "domain", "candidate", "train"], rows) + "\n")
    txt, cov, hffcov = test_section(test, base)
    parts.append(txt)

    (OUT / "summary.md").write_text("\n".join(parts), encoding="utf-8")
    tex = latex_table(cov, hffcov)
    if tex:
        (OUT / "table_coverage.tex").write_text(tex, encoding="utf-8")
    fig_failures(shares)
    fig_seeds(cov)
    print("\n".join(parts))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
