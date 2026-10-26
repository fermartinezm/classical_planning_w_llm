# Axis 1: semantic cues (obfuscation) — Member 1

**RQ1.** How much of an LLM heuristic's coverage depends on meaningful names in the PDDL?

Everything here runs from the repository root and uses the authors' code unchanged,
except for a few lines at the end of `src/llm_heuristics/suites.py` that register the
obfuscated domains as suites.

## Levels

| Level | Domain name, types, constants, predicates, actions, variables | Objects | Order | Comments |
|-------|------|------|------|------|
| L0 | original | original | original | kept (verbatim copy) |
| L1 | random strings | original | original | removed |
| L2 | random strings | random strings | permuted | removed |
| L3 | misleading words from a donor domain | original | original | removed |

- L2 permutes types, constants, predicates, actions, action parameters, precondition and
  effect conjuncts, objects, initial facts and goal conjuncts, and the facts in the
  example files. Predicate argument positions are not permuted.
- L3 donors (`vocab.py`): blocksworld ↔ transport, childsnack ↔ rovers,
  floortile ↔ miconic, sokoban ↔ spanner, matched by arity where possible. In Blocksworld,
  for example, `on` becomes `capacity-predecessor`, the domain is called `transport`, and
  variables are `?package1`, `?package2`.
- L1 and L3 change exactly the same symbols, so L1 vs L3 isolates *missing* vs *wrong* cues.
- The prompt also contains `example-state.out` and `example-static.out`; they are
  obfuscated with the same mapping.

## Usage

```bash
# 1. generate L0-L3 for the 8 domains: all 99 training tasks + the 30-task test subset
#    (every third of easy, medium, hard) -> benchmarks/obfuscated/ (~240 MB, not committed)
python -m axis1.obfuscate

# 2. generate a heuristic on an obfuscated domain: suites are called <domain>-<level>.
#    Use a neutral heuristic name: the paper's names (blocksworld10Heuristic) leak the domain.
uv run llm-heuristics.py --domain blocksworld-L1 --heuristic-name H1Heuristic \
    --heuristic-file h1.py --framework deepseek --model deepseek-reasoner

# 3. run it on obfuscated tasks
uv run src/pyperplan/pyperplan.py -s gbfs_early_goal_test -H h1.py \
    benchmarks/obfuscated/L1/testing/blocksworld/domain.pddl \
    benchmarks/obfuscated/L1/testing/blocksworld/easy-p01.pddl

# 4. translate the plan back to the original names and validate it on the original task
python -m axis1.validate_plan --level L1 --domain blocksworld \
    --task testing/blocksworld/easy-p01.pddl \
    --plan benchmarks/obfuscated/L1/testing/blocksworld/easy-p01.pddl.soln

# correctness check of the obfuscator (h_FF must solve the same tasks at every level)
python -m axis1.check_invariance --n-tasks 10
python -m pytest -q
```

Same `--seed` → identical files. `--rename-objects yes|no` overrides a level's object
renaming; `--all-test-tasks` obfuscates all 90 test tasks instead of 30.

## Status (Oct 26)

- Obfuscator works for L1–L3 on all 8 domains, including the example files; 66 tests pass
  (no original names left, L1 is a pure renaming, mappings injective, deterministic).
- h_FF check: same coverage at every level and every plan valid on the original task, for
  the first 30 training tasks of Blocksworld, Childsnack and Floortile and the first 20 of
  the other five domains. The only difference is one Floortile task that times out at L1.
- The prompt builds correctly for `blocksworld-L1/L2/L3` and never contains "blocksworld".

## Findings to share with the team

1. **h_FF itself is noisy under renaming.** Expansions change a lot between levels even
   though h_FF ignores names: Floortile p20 needs 21 338 expansions at L0 and 90 545 at
   L1. Pyperplan's grounding order depends on the names, which changes GBFS tie-breaking.
   So we should run h_FF at every level as a control and report coverage differences
   relative to it, not only relative to L0.
2. **Object names leak the domain at L1 and L3.** Childsnack objects are called `child4`,
   `bread5`, `tray2`, `sandw2`, so the domain is easy to recognise even with random
   predicate names. The paper also renamed objects in its Blocksworld test
   (`benchmarks/hidden`). Options: (a) keep the plan's definition and say so in the report,
   or (b) rename objects at L1 too (`--rename-objects yes`; at L3 objects then get
   donor-type names such as `waypoint3`). This needs a team decision before the Nov 2
   smoke test.
3. Variables (`?ob`) are renamed from L1 on, because they carry the same cues as type
   names. The paper's hidden Blocksworld kept them.

## Next steps

1. Smoke test by Nov 2: `blocksworld-L1`, R1, n = 10, one seed, on the test subset.
2. Generation on L1–L3 (R1 × 3 seeds, V3 × 1), tests on the 240-task subset, then the
   F1 figure (coverage per level, with h_FF per level as the control).
