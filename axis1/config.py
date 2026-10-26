"""Paths and constants for Axis 1. Run every script from the repository root."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "benchmarks" / "ipc2023-learning"       # original IPC 2023 tasks (= L0)
OUT = REPO_ROOT / "benchmarks" / "obfuscated"             # generated, not committed
PYPERPLAN = REPO_ROOT / "src" / "pyperplan"                # the authors' Pyperplan
RESULTS = REPO_ROOT / "axis1" / "results"

DOMAINS = ["blocksworld", "childsnack", "floortile", "miconic",
           "rovers", "sokoban", "spanner", "transport"]
LEVELS = ["L0", "L1", "L2", "L3"]

# Test subset from the team plan: every third of the 90 test tasks (easy, medium, hard).
TEST_DIFFICULTIES = ["easy", "medium", "hard"]
TEST_SUBSET_STEP = 3

# Pyperplan grounds by iterating over sets of strings: fix the hash seed so node
# expansions are reproducible between runs.
PYTHONHASHSEED = "0"
SEARCH = "gbfs_early_goal_test"   # the search the paper uses


def training_tasks(domain):
    return sorted(f"training/{domain}/{p.name}" for p in (SRC / "training" / domain).glob("p*.pddl"))


def test_tasks(domain, subset=True):
    folder = SRC / "testing" / domain
    tasks = []
    for diff in TEST_DIFFICULTIES:
        tasks += sorted(f"testing/{domain}/{p.name}" for p in folder.glob(f"{diff}-p*.pddl"))
    return tasks[::TEST_SUBSET_STEP] if subset else tasks
