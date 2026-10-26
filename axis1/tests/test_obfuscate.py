"""Fast checks for the obfuscator. Run from the repository root:

  python -m pytest axis1/tests -q
"""

import json

import pytest

from axis1 import obfuscate, sexp
from axis1.config import DOMAINS, SRC

TASKS = ["p01.pddl", "p05.pddl", "p99.pddl"]
LEVELS = ["L1", "L2", "L3"]


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("obf")
    for level in ["L0"] + LEVELS:
        for domain in DOMAINS:
            tasks = [f"training/{domain}/{t}" for t in TASKS] + [f"testing/{domain}/easy-p01.pddl"]
            obfuscate.obfuscate_domain(domain, level, dict(obfuscate.LEVELS[level]), 0, SRC, out, tasks)
    return out


def files(domain):
    return ([f"training/{domain}/domain.pddl", f"testing/{domain}/domain.pddl",
             f"testing/{domain}/easy-p01.pddl"]
            + [f"training/{domain}/{t}" for t in TASKS + obfuscate.EXAMPLE_FILES])


def domain_symbols(domain):
    info = obfuscate.analyse_domain(sexp.parse((SRC / "training" / domain / "domain.pddl").read_text()))
    return ({info["name"]} | set(info["types"]) | {p for p, _ in info["predicates"]}
            | {a for a, _ in info["actions"]} | {c for c, _ in info["constants"]})


def mapping(out_dir, level, domain, name="mapping.json"):
    return json.loads((out_dir / level / "mappings" / domain / name).read_text())


def tokens(path):
    return set(sexp.tokenize(path.read_text().replace("'", " ").replace("frozenset", " ")))


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_no_original_names_left(out_dir, level, domain):
    names = domain_symbols(domain) - obfuscate.KEYWORDS
    new_names = set(mapping(out_dir, level, domain)["symbols"].values())
    for f in files(domain):
        leaked = (names & tokens(out_dir / level / f)) - new_names
        assert not leaked, f"{f}: {leaked}"


@pytest.mark.parametrize("domain", DOMAINS)
def test_l0_is_a_copy(out_dir, domain):
    for f in files(domain):
        assert (out_dir / "L0" / f).read_bytes() == (SRC / f).read_bytes()


@pytest.mark.parametrize("domain", DOMAINS)
def test_l1_keeps_structure(out_dir, domain):
    """L1 only renames: the token sequence must match L0 up to a consistent bijection.
    Variables are renamed per action/predicate scope, so for them we only check that a
    variable stays a variable."""
    orig = sexp.tokenize((SRC / "training" / domain / "domain.pddl").read_text())
    new = sexp.tokenize((out_dir / "L1" / "training" / domain / "domain.pddl").read_text())
    assert len(orig) == len(new)
    fwd, back = {}, {}
    for a, b in zip(orig, new):
        if a.startswith("?"):
            assert b.startswith("?")
            continue
        assert fwd.setdefault(a, b) == b and back.setdefault(b, a) == a


@pytest.mark.parametrize("domain", DOMAINS)
def test_examples_keep_fact_count(out_dir, domain):
    for level in LEVELS:
        for name in obfuscate.EXAMPLE_FILES:
            before = obfuscate._FACT.findall((SRC / "training" / domain / name).read_text())
            after = obfuscate._FACT.findall((out_dir / level / "training" / domain / name).read_text())
            assert len(before) == len(after)


@pytest.mark.parametrize("domain", DOMAINS)
def test_mapping_is_injective(out_dir, domain):
    for level in LEVELS:
        m = mapping(out_dir, level, domain)
        assert len(set(m["symbols"].values())) == len(m["symbols"])


@pytest.mark.parametrize("domain", DOMAINS)
def test_l2_renames_objects_and_l1_does_not(out_dir, domain):
    l1 = mapping(out_dir, "L1", domain, "task_mapping.json")
    l2 = mapping(out_dir, "L2", domain, "task_mapping.json")
    assert all(not t["objects"] for t in l1.values())
    assert any(t["objects"] for t in l2.values())


def test_l3_uses_donor_names(out_dir):
    m = mapping(out_dir, "L3", "blocksworld")
    assert m["obfuscated_domain_name"] == "transport"
    assert m["symbols"]["on"] in {"in", "at", "road", "capacity-predecessor", "capacity"}


def test_deterministic(tmp_path):
    tasks = [f"training/sokoban/{t}" for t in TASKS]
    for out in (tmp_path / "a", tmp_path / "b"):
        obfuscate.obfuscate_domain("sokoban", "L2", dict(obfuscate.LEVELS["L2"]), 7, SRC, out, tasks)
    for f in files("sokoban")[:2] + tasks:
        assert (tmp_path / "a/L2" / f).read_text() == (tmp_path / "b/L2" / f).read_text()
