#!/usr/bin/env python3
"""
Shared schema, canonical benchmark splits, error taxonomy, and metric helpers.
Owned by Member 4 — imported by runner, selection, and plotting scripts.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

IPC2023_DOMAINS: List[str] = [
    "blocksworld",
    "childsnack",
    "floortile",
    "miconic",
    "rovers",
    "sokoban",
    "spanner",
    "transport",
]

ERROR_TYPES = {
    "none",              # Solved and validated with VAL (or valid unsolvable termination if applicable)
    "unsolved",          # Search terminated cleanly without finding a plan (non-infinite initial h)
    "timeout",           # Exceeded wall-clock time limit (60s train / 300s test)
    "oom",               # Exceeded 4 GiB memory limit or raised MemoryError
    "infinite_h",        # Heuristic returned inf / float('inf') on initial state or exhausted immediately
    "crash_compile",     # SyntaxError / ImportError / missing Heuristic class in generated Python file
    "crash_runtime",     # Exception raised inside heuristic evaluation during search
    "invalid_plan",      # Planner output a plan that failed VAL validation
}

# Pre-selected best heuristics from Corrêa et al. (NeurIPS 2025) archived in the repo ($0 API cost to reuse)
ARCHIVED_SELECTED_HEURISTICS: Dict[str, Dict[str, str]] = {
    "deepseek-r1": {
        "dir": "archive-heuristics/2025-03-10-deepseek-r1",
        "domains": {
            "blocksworld": "blocksworld9_10.py",
            "childsnack": "childsnack19_10.py",
            "floortile": "floortile15_10.py",
            "miconic": "miconic11_10.py",
            "rovers": "rovers23_10.py",
            "sokoban": "sokoban13_10.py",
            "spanner": "spanner10_10.py",
            "transport": "transport7_10.py",
        },
    },
    "deepseek-v3": {
        "dir": "archive-heuristics/2025-03-25-deepseek-v3-0324",
        "domains": {
            "blocksworld": "blocksworld11_10.py",
            "childsnack": "childsnack22_10.py",
            "floortile": "floortile5_10.py",
            "miconic": "miconic13_10.py",
            "rovers": "rovers23_10.py",
            "sokoban": "sokoban9_10.py",
            "spanner": "spanner17_10.py",
            "transport": "transport20_10.py",
        },
    },
    "qwen-14b": {
        "dir": "archive-heuristics/2025-03-13-deepseek-r1-distill-qwen-14b",
        "domains": {
            "blocksworld": "blocksworld8_10.py",
            "childsnack": "childsnack3_10.py",
            "floortile": "floortile15_10.py",
            "miconic": "miconic8_10.py",
            "rovers": "rovers12_10.py",
            "sokoban": "sokoban11_10.py",
            "spanner": "spanner4_10.py",
            "transport": "transport21_10.py",
        },
    },
}

ARCHIVED_CANDIDATE_DIRS: Dict[str, str] = {
    "deepseek-r1": "archive-heuristics/2025-03-10-deepseek-r1",
    "deepseek-v3": "archive-heuristics/2025-03-25-deepseek-v3-0324",
    "qwen-14b": "archive-heuristics/2025-03-13-deepseek-r1-distill-qwen-14b",
    "gemini-2.0-flash": "archive-heuristics/2025-03-09-ipc2023-temp10",
    "gemini-2.0-flash-thinking": "archive-heuristics/2025-03-14-gemini-thinking",
    "gemini-2.5-pro": "archive-heuristics/2025-03-30-gemini-25-ipc2023",
    "blocksworld-l1-r1": "archive-heuristics/2025-07-23-hidden-blocks-r1",
}


@dataclass
class RunResult:
    domain: str
    task: str
    split: str               # "train" or "test"
    config: str              # e.g. "pyperplan-hff", "paper-r1", "axis1-L1-v3", "axis3-alt3"
    model: str               # e.g. "none", "deepseek-r1", "deepseek-v3", "qwen-7b"
    obf_level: str           # "L0", "L1", "L2", "L3"
    seed: int                # 1 (default single-seed fast protocol)
    candidate_id: str        # e.g. "baseline", "selected", "9_10", "cand_01"
    solved: bool
    time: Optional[float]        # Total time in seconds (wall/process time)
    search_time: Optional[float] # Search time in seconds
    expansions: Optional[int]
    plan_len: Optional[int]
    mem: Optional[int]           # Peak RSS memory in KB
    init_h: Optional[str]        # Initial heuristic value as string (to detect "inf")
    error: str                   # Must be in ERROR_TYPES

    def validate(self) -> None:
        if self.domain not in IPC2023_DOMAINS:
            raise ValueError(f"Unknown domain: {self.domain}")
        if self.split not in ("train", "test"):
            raise ValueError(f"Invalid split: {self.split}")
        if self.obf_level not in ("L0", "L1", "L2", "L3"):
            raise ValueError(f"Invalid obf_level: {self.obf_level}")
        if self.error not in ERROR_TYPES:
            raise ValueError(f"Invalid error type '{self.error}'. Allowed: {sorted(ERROR_TYPES)}")

    def to_json_line(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=False)


def compute_agile_score(time_s: Optional[float], solved: bool, t_max: float = 300.0) -> float:
    """IPC Agile score adjusted for the protocol time limit (default 300s test, 60s train)."""
    if not solved or time_s is None:
        return 0.0
    if time_s <= 1.0:
        return 1.0
    if time_s >= t_max:
        return 0.0
    return max(0.0, 1.0 - (math.log(time_s) / math.log(t_max)))


def compute_expansion_score(expansions: Optional[int], solved: bool, upper_bound: float = 1e6) -> float:
    """IPC-style log expansion score matching experiments/project.py."""
    if not solved or expansions is None or expansions <= 0:
        return 0.0
    if expansions <= 100:
        return 1.0
    if expansions >= upper_bound:
        return 0.0
    return max(0.0, 1.0 - (math.log(expansions) / math.log(upper_bound)))


def build_canonical_splits(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """
    Build the exact 20-train and 30-test (every third task: files[0::3]) split manifest
    across all 8 IPC 2023 Learning Track domains.
    """
    train_base = repo_root / "benchmarks" / "ipc2023-learning" / "training"
    test_base = repo_root / "benchmarks" / "ipc2023-learning" / "testing"

    splits: Dict[str, Any] = {
        "protocol": {
            "search": "gbf",
            "train_time_limit_s": 60,
            "test_time_limit_s": 300,
            "memory_limit_mb": 4096,
            "candidates_per_config": 10,
            "seeds": 1,
            "train_tasks_per_domain": 20,
            "test_tasks_per_domain": 30,
        },
        "domains": {},
    }

    for domain in IPC2023_DOMAINS:
        d_train = train_base / domain
        d_test = test_base / domain

        train_files = sorted(
            f for f in os.listdir(d_train)
            if f.endswith(".pddl") and "domain" not in f
        )[:20]

        all_test_files = sorted(
            f for f in os.listdir(d_test)
            if f.endswith(".pddl") and "domain" not in f
        )
        # Every third task (indices 0, 3, 6, ..., 87) -> exactly 30 tasks covering easy/medium/hard
        test_files = all_test_files[0::3][:30]

        splits["domains"][domain] = {
            "train": train_files,
            "test": test_files,
        }

    return splits


def save_canonical_splits(out_path: Optional[Path] = None) -> Path:
    if out_path is None:
        out_path = REPO_ROOT / "benchmarks" / "splits.json"
    splits = build_canonical_splits(REPO_ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    return out_path


if __name__ == "__main__":
    path = save_canonical_splits()
    print(f"Wrote canonical splits manifest to {path}")

