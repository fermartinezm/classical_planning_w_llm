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


# ---- shared protocol (team plan) ----
TRAIN_TIME_LIMIT_S = 60       # selection runs on training tasks
TEST_TIME_LIMIT_S = 300       # test runs (paper: 1800)
MEMORY_LIMIT_GIB = 4          # paper: 8
N_CANDIDATES = 10             # candidates per configuration (paper: 25)

HEURISTICS_DIR = REPO_ROOT / "axis1" / "heuristics"   # generated candidates (committed)
LOGS_DIR = REPO_ROOT / "axis1" / "logs"               # full LLM prompts/answers (not committed)

# Axis 1 generation configurations: model key in src/llm_heuristics/models.py -> seeds
MODELS = {"deepseek-r1": 3, "deepseek-v3": 1}
GEN_LEVELS = ["L0", "L1", "L2", "L3"]   # L0 is regenerated with today's API model too
HEURISTIC_CLASS = "CandidateHeuristic"  # neutral class name: the paper's names leak the domain
ARCHIVE_R1 = REPO_ROOT / "archive-heuristics" / "2025-03-10-deepseek-r1"


def suite_name(domain, level):
    """Name of the suite in src/llm_heuristics/suites.py."""
    return domain if level == "L0" else f"{domain}-{level}"


def level_task(level, rel):
    """Path of a task (rel = training/<d>/pXX.pddl or testing/<d>/...) at a level."""
    return OUT / level / rel


def candidate_path(model, level, seed, domain, cid):
    return HEURISTICS_DIR / model / level / f"seed{seed}" / domain / f"h{cid:02d}.py"
