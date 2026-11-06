#!/usr/bin/env python3
"""
Unified plotting and LaTeX table generator (Member 4 infrastructure).
Reads JSONL result files produced by `run_experiments.py` and generates:
  - F1: Coverage per domain across obfuscation levels L0-L3 (Member 1)
  - F2: Coverage vs. model size + candidate failure taxonomy stacked bar (Member 2)
  - F3: Test coverage of all 25 candidates per domain + selection rules & alternation (Member 3)
  - F4: Log-log scatter of node expansions: LLM heuristic vs. Pyperplan h_FF (Member 4)
  - F5: Cactus plot: cumulative tasks solved vs. time across main configurations (Member 4)
  - summary_table.tex: ICAPS-ready LaTeX table of coverage, agile score, expansion score, and failure rate.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infra.schema import (
    IPC2023_DOMAINS,
    compute_agile_score,
    compute_expansion_score,
)


def load_jsonl_files(paths: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    expanded: List[str] = []
    for p in paths:
        matched = glob.glob(p)
        expanded.extend(matched if matched else [p])
    for fp in expanded:
        path = Path(fp)
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def plot_f4_expansions(records: List[Dict[str, Any]], out_dir: Path, llm_config: str = "paper-r1", hff_config: str = "pyperplan-hff") -> Path:
    """F4 (Member 4): Log-log scatter of node expansions (LLM heuristic vs. Pyperplan h_FF)."""
    by_cfg_task: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = defaultdict(dict)
    for r in records:
        if r.get("split") != "test":
            continue
        by_cfg_task[r["config"]][(r["domain"], r["task"])] = r

    llm_map = by_cfg_task.get(llm_config, {})
    hff_map = by_cfg_task.get(hff_config, {})
    common_keys = sorted(set(llm_map.keys()) & set(hff_map.keys()))

    max_bound = 2e6
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    for dom in IPC2023_DOMAINS:
        xs, ys = [], []
        for d, t in common_keys:
            if d != dom:
                continue
            r_hff = hff_map[(d, t)]
            r_llm = llm_map[(d, t)]
            if not r_hff.get("solved") and not r_llm.get("solved"):
                continue
            x = r_hff.get("expansions") if r_hff.get("solved") and r_hff.get("expansions") else max_bound
            y = r_llm.get("expansions") if r_llm.get("solved") and r_llm.get("expansions") else max_bound
            xs.append(max(1, x))
            ys.append(max(1, y))
        if xs:
            ax.scatter(xs, ys, alpha=0.75, s=28, label=dom)

    ax.plot([1, max_bound], [1, max_bound], "k--", lw=1.0, label="x = y (equal guidance)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, max_bound * 1.3)
    ax.set_ylim(1, max_bound * 1.3)
    ax.set_xlabel(f"Node Expansions ({hff_config})")
    ax.set_ylabel(f"Node Expansions ({llm_config})")
    ax.set_title("F4: Node Expansions — LLM Heuristic vs. Pyperplan h_FF")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    out_file = out_dir / "F4_expansions_scatter.pdf"
    fig.savefig(out_file)
    fig.savefig(out_dir / "F4_expansions_scatter.png", dpi=200)
    plt.close(fig)
    return out_file


def plot_f5_cactus(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    """F5 (Member 4): Cactus plot — cumulative tasks solved vs. wall/total time."""
    by_config: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r.get("split") != "test":
            continue
        if r.get("solved") and r.get("time") is not None:
            by_config[r["config"]].append(max(0.001, float(r["time"])))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for cfg, times in sorted(by_config.items()):
        times_sorted = sorted(times)
        counts = list(range(1, len(times_sorted) + 1))
        ax.step(times_sorted, counts, where="post", lw=2.0, label=f"{cfg} ({len(times_sorted)})")

    ax.set_xscale("log")
    ax.set_xlabel("Time per task (s, log scale)")
    ax.set_ylabel("Cumulative test tasks solved (out of 240)")
    ax.set_title("F5: Cactus Plot — Tasks Solved vs. Time")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    if by_config:
        ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out_file = out_dir / "F5_cactus.pdf"
    fig.savefig(out_file)
    fig.savefig(out_dir / "F5_cactus.png", dpi=200)
    plt.close(fig)
    return out_file


def plot_f1_obfuscation(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    """F1 (Member 1): Coverage per domain across L0-L3 obfuscation levels."""
    levels = ["L0", "L1", "L2", "L3"]
    cov: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hff_cov: Dict[str, int] = defaultdict(int)

    for r in records:
        if r.get("split") != "test" or not r.get("solved"):
            continue
        if r.get("config") == "pyperplan-hff" and r.get("obf_level") == "L0":
            hff_cov[r["domain"]] += 1
        elif r.get("model") != "none":
            cov[r["obf_level"]][r["domain"]] += 1

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = list(range(len(IPC2023_DOMAINS)))
    width = 0.18
    for idx, lvl in enumerate(levels):
        vals = [cov[lvl].get(d, 0) for d in IPC2023_DOMAINS]
        offsets = [pos + (idx - 1.5) * width for pos in x]
        ax.bar(offsets, vals, width=width, label=lvl)

    hff_vals = [hff_cov.get(d, 0) for d in IPC2023_DOMAINS]
    ax.scatter(x, hff_vals, color="black", marker="D", zorder=5, label="Pyperplan h_FF (L0)")

    ax.set_xticks(x)
    ax.set_xticklabels(IPC2023_DOMAINS, rotation=25, ha="right")
    ax.set_ylabel("Tasks Solved (out of 30)")
    ax.set_title("F1: Coverage by Domain Across Obfuscation Levels (L0–L3)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out_file = out_dir / "F1_obfuscation.pdf"
    fig.savefig(out_file)
    fig.savefig(out_dir / "F1_obfuscation.png", dpi=200)
    plt.close(fig)
    return out_file


def plot_f2_scaling(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    """F2 (Member 2): Coverage vs. model + stacked bar of candidate failure taxonomy."""
    models_seen = sorted({r["model"] for r in records if r.get("model") and r["model"] != "none"})
    if not models_seen:
        models_seen = ["deepseek-r1"]

    tax_cats = ["none", "unsolved", "timeout", "oom", "infinite_h", "crash_runtime", "crash_compile"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)

    x = list(range(len(models_seen)))
    solved_counts = []
    for m in models_seen:
        m_recs = [r for r in records if r.get("model") == m and r.get("split") == "test"]
        solved_counts.append(sum(1 for r in m_recs if r.get("solved")))

    ax1.plot(x, solved_counts, marker="o", lw=2.0, label="LLM Heuristic")
    hff_solved = sum(1 for r in records if r.get("config") == "pyperplan-hff" and r.get("split") == "test" and r.get("solved"))
    if hff_solved > 0:
        ax1.axhline(hff_solved, color="k", ls="--", label=f"Pyperplan h_FF ({hff_solved})")
    ax1.set_ylabel("Test Tasks Solved")
    ax1.set_title("F2: Coverage & Candidate Failure Taxonomy by Model")
    ax1.legend(fontsize=8)
    ax1.grid(True, ls=":", alpha=0.5)

    bottoms = [0.0] * len(models_seen)
    for cat in tax_cats:
        pcts = []
        for m in models_seen:
            m_recs = [r for r in records if r.get("model") == m]
            tot = max(1, len(m_recs))
            pcts.append(100.0 * sum(1 for r in m_recs if r.get("error") == cat) / tot)
        ax2.bar(x, pcts, bottom=bottoms, width=0.5, label=cat)
        bottoms = [b + p for b, p in zip(bottoms, pcts)]

    ax2.set_xticks(x)
    ax2.set_xticklabels(models_seen, rotation=20, ha="right")
    ax2.set_ylabel("Outcome Share (%)")
    ax2.legend(fontsize=7, ncol=4, loc="upper right")
    fig.tight_layout()
    out_file = out_dir / "F2_model_scaling.pdf"
    fig.savefig(out_file)
    fig.savefig(out_dir / "F2_model_scaling.png", dpi=200)
    plt.close(fig)
    return out_file


def plot_f3_selection(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    """F3 (Member 3): Per-domain test coverage of candidate heuristics + selection rules."""
    cand_cov: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r.get("split") == "test" and r.get("solved") and r.get("model") != "none":
            cand_cov[r["domain"]][str(r["candidate_id"])] += 1

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, dom in enumerate(IPC2023_DOMAINS):
        c_map = cand_cov.get(dom, {})
        ys = list(c_map.values())
        xs = [i] * len(ys)
        if ys:
            ax.scatter(xs, ys, alpha=0.6, color="steelblue", s=25, label="Candidate" if i == 0 else None)
            ax.scatter([i], [max(ys)], color="crimson", marker="*", s=90, zorder=5, label="Oracle (best test)" if i == 0 else None)

    ax.set_xticks(list(range(len(IPC2023_DOMAINS))))
    ax.set_xticklabels(IPC2023_DOMAINS, rotation=25, ha="right")
    ax.set_ylabel("Test Tasks Solved (out of 30)")
    ax.set_title("F3: Candidate Test Coverage Distribution & Oracle Upper Bound")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out_file = out_dir / "F3_candidate_selection.pdf"
    fig.savefig(out_file)
    fig.savefig(out_dir / "F3_candidate_selection.png", dpi=200)
    plt.close(fig)
    return out_file


def write_latex_summary_table(records: List[Dict[str, Any]], out_dir: Path) -> Path:
    """Generate LaTeX summary table from JSONL test runs (no hand-editing)."""
    by_cfg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("split") == "test":
            by_cfg[r["config"]].append(r)

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Configuration & Solved & Coverage (\%) & Agile Score & Exp.\ Score & Crash/Inf (\%) \\",
        r"\midrule",
    ]
    for cfg, recs in sorted(by_cfg.items()):
        total = len(recs)
        if total == 0:
            continue
        solved = sum(1 for r in recs if r.get("solved"))
        cov_pct = 100.0 * solved / total
        agile = sum(compute_agile_score(r.get("time"), bool(r.get("solved")), 300.0) for r in recs)
        exp_sc = sum(compute_expansion_score(r.get("expansions"), bool(r.get("solved"))) for r in recs)
        bad = sum(1 for r in recs if r.get("error") in ("crash_compile", "crash_runtime", "infinite_h", "invalid_plan"))
        bad_pct = 100.0 * bad / total
        cfg_tex = cfg.replace("_", r"\_")
        lines.append(f"{cfg_tex} & {solved}/{total} & {cov_pct:.1f} & {agile:.1f} & {exp_sc:.1f} & {bad_pct:.1f} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_file = out_dir / "summary_table.tex"
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figures F1-F5 and LaTeX tables from JSONL results.")
    parser.add_argument("--inputs", nargs="+", default=["results/*.jsonl"], help="JSONL file paths or globs.")
    parser.add_argument("--out-dir", default="plots/output", help="Output directory for PDF/PNG figures and .tex table.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl_files(args.inputs)
    if not records:
        print("No JSONL records found.")
        return

    f1 = plot_f1_obfuscation(records, out_dir)
    f2 = plot_f2_scaling(records, out_dir)
    f3 = plot_f3_selection(records, out_dir)
    f4 = plot_f4_expansions(records, out_dir)
    f5 = plot_f5_cactus(records, out_dir)
    tex = write_latex_summary_table(records, out_dir)
    print(f"Generated:\n  - {f1}\n  - {f2}\n  - {f3}\n  - {f4}\n  - {f5}\n  - {tex}")


if __name__ == "__main__":
    main()
