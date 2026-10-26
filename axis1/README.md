# Axis 1: semantic cues (obfuscation) — Member 1

**RQ1.** How much of an LLM heuristic's coverage depends on meaningful names in the PDDL?

Everything in the team plan for this role is here: the obfuscator, heuristic generation
on L0–L3, selection, tests on the 240-task subset, the h_FF invariance check, F1, and drafts
of the report section, the related work on memorisation and the poster panel. It uses the
authors' code unchanged, except for a few lines at the end of
`src/llm_heuristics/suites.py` that register the obfuscated domains as suites.

## Setup (once)

Linux or WSL (Downward Lab and the memory limits need POSIX), from the repository root:

```bash
uv sync && source .venv/bin/activate
pip install pytest                 # for the tests only
# optional, as in the paper: PyPy for Pyperplan (used automatically if found)
sudo apt install pypy3
```

## Run it

```bash
export DEEPSEEK_API_KEY=...        # ~$30-35 for the whole axis (plan: ~$2.50 per R1 generation; check prices)

bash axis1/smoke_test.sh           # Nov 2 gate: Blocksworld, L0+L1, R1, 1 seed (~$0.60)
JOBS=6 bash axis1/run_axis1.sh     # everything below, in order
```

Every step skips work that is already done, so you can stop and rerun. The steps one by
one:

| Step | Command | Output |
|------|---------|--------|
| 1. Obfuscate | `python -m axis1.obfuscate` | `benchmarks/obfuscated/L0-L3` (not committed) |
| 2. Check obfuscator | `python -m axis1.check_invariance --n-tasks 10` | h_FF solves the same tasks at every level |
| 3. Generate | `python -m axis1.generate --jobs 4` | `axis1/heuristics/<model>/<level>/seed<k>/<domain>/hNN.py` |
| 4. Train (60 s) | `python -m axis1.evaluate train` | `axis1/results/train.jsonl` |
| 5. Select | `python -m axis1.select_best` | `axis1/results/selection.json` |
| 6. Test (5 min) | `python -m axis1.evaluate test` | `axis1/results/test.jsonl` |
| 7. Controls | `python -m axis1.evaluate control` | `axis1/results/control.jsonl` (h_FF, blind per level) |
| 8. Tables | `python -m axis1.summarize` | Markdown tables, `axis1/results/summary.csv` |
| 9. F1 | `python -m axis1.plot_f1` | `axis1/figures/f1_<model>.pdf/.png` |

All commands take `--models`, `--levels`, `--seeds`, `--domains`, so the work can be split
across laptops (for example, one level each). Each laptop then commits its own
`axis1/results/*.jsonl` lines. The files are append-only and can be concatenated.
`--n-train` / `--n-test` shrink runs for quick tests.

**Compute.** Generation is 4 levels × (3 R1 seeds + 1 V3 seed) × 8 domains × 10 = 1 280
API calls. Training is 1 280 candidates × 99 tasks × ≤ 60 s. Most runs end in seconds, but
this is the biggest cost (estimate it from the smoke test). If it is too much, select
on fewer training tasks with `--n-train 30` and state it in the report. Tests are
16 configurations × 240 tasks × ≤ 5 min, plus controls of 2 × 4 levels × 240.

**Without an API key** you can test the pipeline with the paper's archived R1 candidates
as L0 candidates: `python -m axis1.generate --from-archive --levels L0 --seeds 1` (then
delete `axis1/heuristics/`).

## Levels

| Level | Domain name, types, constants, predicates, actions, variables | Objects | Order | Comments |
|-------|------|------|------|------|
| L0 | original | original | original | kept (verbatim copy) |
| L1 | random strings | original | original | removed |
| L2 | random strings | random strings | permuted | removed |
| L3 | misleading words from a donor domain | original | original | removed |

- L2 permutes types, constants, predicates, actions, action parameters, precondition and
  effect conjuncts, objects, initial facts, goal conjuncts and example facts.
- L3 donors (`vocab.py`): blocksworld ↔ transport, childsnack ↔ rovers,
  floortile ↔ miconic, sokoban ↔ spanner, matched by arity where possible.
- The prompt's `example-state.out` / `example-static.out` are obfuscated with the same mapping.
- Same `--seed` → identical files. `--rename-objects yes|no` overrides a level's object
  renaming.

## Protocol details (from the team plan, in `axis1/config.py`)

- Search `gbfs_early_goal_test` in the authors' Pyperplan (as in their obfuscation
  experiment), `PYTHONHASHSEED=0`, PyPy if installed.
- Training: all 99 training tasks, 60 s. Test: every third of the 90 test tasks, 5 min, 4 GiB.
- Selection: the paper's rule (`experiments/get-best-heuristics.py`): most training tasks
  solved, then the sum of expansion scores, then the sum of agile scores.
- Models: `deepseek-r1` (3 seeds) and `deepseek-v3` (1 seed), temperature 1.0, top-p 0.5.
  L0 is regenerated with today's API model. The DeepSeek API names now point to newer
  models than the paper's March 2025 ones, and comparisons must stay within one model.
  `axis1/heuristics/generation.jsonl` records the date of every call.
- The heuristic class is called `CandidateHeuristic`. The paper's names
  (`blocksworld10Heuristic`) leak the domain.
- Every solved plan is translated back to the original names and validated on the
  original task; an invalid plan counts as unsolved (`error: invalid-plan`).
- Results use the team's JSON-lines format (`domain, task, config, model, obf_level, seed,
  candidate_id, solved, time, expansions, plan_len, mem, error`) plus `stage` and `heuristic`.

## Report and poster drafts

- `report/axis1_section.md`: method, hypotheses, results skeleton, limitations.
- `report/related_work_memorisation.md`: related work on memorisation, with references.
- `report/poster_panel.md`: the "Do names matter?" panel.

## Findings so far (Oct 26)

1. **h_FF is noisy under renaming.** It solves the same tasks at every level, but
   expansions change a lot (Floortile p20: 21 338 at L0, 90 545 at L1). The reason is that
   Pyperplan's grounding order depends on names, which changes GBFS tie-breaking. So we
   compare each LLM heuristic with h_FF at the same level.
2. **Object names leak the domain at L1 and L3, sometimes literally.** Childsnack objects
   are `child4`, `bread5`, `tray2`. In Rovers and Spanner the objects contain the domain
   name itself (`rover1`, `spanner1`): the L1 prompt for Rovers says "rover" 235 times,
   Spanner's 39 times. At L2 (objects renamed) no prompt contains its domain name. (The word
   "transport" appears in every prompt, but only inside the authors' fixed Gripper example.)
   The paper renamed objects in its Blocksworld test. **The team must decide before
   Nov 2.** My recommendation: rename objects at L1 and L3 too
   (`python -m axis1.obfuscate --rename-objects yes`; at L3 objects then get donor-type
   names such as `waypoint3`). L2 then differs from L1 only by the reordering.
3. Variables (`?ob`) are renamed from L1 on. The paper's hidden Blocksworld kept them.

## Files

`obfuscate.py`, `sexp.py`, `vocab.py`: obfuscator · `validate_plan.py`: plan translation and
validation · `check_invariance.py`: obfuscator check · `generate.py`, `runner.py`,
`evaluate.py`, `select_best.py`: pipeline · `summarize.py`, `plot_f1.py`: tables and F1 ·
`config.py`: paths and protocol · `tests/`: 72 tests (`python -m pytest -q`).
