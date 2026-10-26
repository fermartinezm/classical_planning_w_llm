"""Tests for selection and the Pyperplan runner (small, fast)."""

import json

import pytest

from axis1 import obfuscate, runner, select_best
from axis1.config import SRC


def row(cid, solved, exp=None, time=1.0, domain="blocksworld"):
    return dict(model="m", obf_level="L1", seed=1, domain=domain, candidate_id=cid,
                heuristic=f"h{cid:02d}.py", solved=solved, expansions=exp, time=time)


def test_select_prefers_coverage_then_expansions():
    rows = [row(1, True, 50), row(1, False),            # coverage 1
            row(2, True, 5000), row(2, True, 5000),     # coverage 2, many expansions
            row(3, True, 100), row(3, True, 100)]       # coverage 2, few expansions -> best
    sel, missing = select_best.select(rows)
    assert [s["candidate_id"] for s in sel] == [3]
    assert not missing


def test_select_reports_configs_without_working_candidate():
    sel, missing = select_best.select([row(1, False), row(2, False, domain="spanner"),
                                       row(3, True, 10, domain="spanner")])
    assert [s["domain"] for s in sel] == ["spanner"]
    assert missing == [("m", "L1", 1, "blocksworld")]


@pytest.fixture(scope="module")
def obf(tmp_path_factory):
    out = tmp_path_factory.mktemp("obf")
    tasks = ["training/blocksworld/p05.pddl"]
    for level in ["L0", "L2"]:
        obfuscate.obfuscate_domain("blocksworld", level, dict(obfuscate.LEVELS[level]), 0, SRC, out, tasks)
    return out


@pytest.mark.parametrize("level", ["L0", "L2"])
def test_runner_solves_and_validates(obf, monkeypatch, level):
    monkeypatch.setattr(runner, "OUT", obf)
    monkeypatch.setattr("axis1.validate_plan.OUT", obf, raising=False)
    job = dict(stage="control", config="pyperplan-hff", model=None, obf_level=level, seed=None,
               candidate_id=None, task="training/blocksworld/p05.pddl", heuristic="hff")
    r = runner.run_one(job, 60, 4, runner.default_python())
    assert r["solved"] and r["error"] is None and r["plan_len"] and r["expansions"]


def test_runner_reports_crash(obf, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUT", obf)
    bad = tmp_path / "h01.py"
    bad.write_text("from heuristics.heuristic_base import Heuristic\n"
                   "class BrokenHeuristic(Heuristic):\n"
                   "    def __init__(self, task):\n        pass\n"
                   "    def __call__(self, node):\n        raise RuntimeError('boom')\n")
    job = dict(stage="train", config="pyperplan-llm", model="m", obf_level="L0", seed=1,
               candidate_id=1, task="training/blocksworld/p05.pddl", heuristic=str(bad))
    r = runner.run_one(job, 60, 4, runner.default_python())
    assert not r["solved"] and r["error"] == "crash"


def test_run_jobs_resumes(obf, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUT", obf)
    out = tmp_path / "r.jsonl"
    job = dict(stage="control", config="pyperplan-hff", model=None, obf_level="L0", seed=None,
               candidate_id=None, task="training/blocksworld/p05.pddl", heuristic="hff")
    runner.run_jobs([job], out, 60, 4, 1)
    runner.run_jobs([job], out, 60, 4, 1)
    assert len(out.read_text().splitlines()) == 1
    assert json.loads(out.read_text())["solved"]


@pytest.mark.parametrize("domain", ["blocksworld", "childsnack", "floortile", "miconic",
                                    "rovers", "sokoban", "spanner"])
def test_l2_prompt_does_not_name_the_domain(tmp_path, domain, monkeypatch):
    """At L2 (everything renamed) the full prompt must not contain the domain's name.
    (Transport is skipped: the authors' fixed Gripper example says "transport".)"""
    prompt_mod = pytest.importorskip("src.llm_heuristics.prompt")
    from src.llm_heuristics.suites import DomainSuite
    tasks = [f"training/{domain}/p01.pddl", f"training/{domain}/p99.pddl"]
    obfuscate.obfuscate_domain(domain, "L2", dict(obfuscate.LEVELS["L2"]), 0, SRC, tmp_path, tasks)
    base = tmp_path / "L2" / "training" / domain
    name = json.loads((tmp_path / "L2/mappings" / domain / "mapping.json").read_text())["obfuscated_domain_name"]
    suite = DomainSuite(name, str(base / "domain.pddl"), str(base / "p01.pddl"), str(base / "p99.pddl"),
                        str(base / "example-state.out"), str(base / "example-static.out"))
    monkeypatch.chdir(SRC.parent.parent)
    prompt = prompt_mod.create_prompt(suite, "CandidateHeuristic", "neurips", "false").lower()
    assert domain.rstrip("s") not in prompt
