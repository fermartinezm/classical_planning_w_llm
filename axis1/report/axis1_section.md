# Axis 1 report section (draft, Member 1)

Draft text for Section 5.1 of the report. Numbers in `[[...]]` come from
`axis1/results/summary.md` once the runs are done; delete this note before porting to
LaTeX.

---

## 5.1 Do names matter? Semantic cues and obfuscation

**Question.** Corrêa et al. obfuscate one domain (Blocksworld) for one model (DeepSeek R1)
and find that coverage drops from 66 to 40 tasks, while h_FF in Fast Downward stays at
27–28. We ask how general this is: how much of an LLM heuristic's coverage depends on
meaningful names, across all eight domains, at graded levels of obfuscation, and does a
*wrong* vocabulary hurt more than a *missing* one?

**Obfuscation levels.** We transform every domain consistently in the domain file, the
task files and the two example-state files that the prompt contains:

- **L0, original.**
- **L1, renamed.** The domain name, types, constants, predicates, actions and
  variables are replaced by random lowercase strings. Object names and the order of all
  elements stay unchanged.
- **L2, renamed and shuffled.** L1, plus random object names, plus a random permutation
  of types, constants, predicates, actions, action parameters, precondition and effect
  conjuncts, objects, initial facts and goal conjuncts.
- **L3, misleading.** The same symbols as in L1 are renamed, but with meaningful words
  from a different benchmark domain, matched by arity where possible: Blocksworld ↔
  Transport, Childsnack ↔ Rovers, Floortile ↔ Miconic, Sokoban ↔ Spanner. For example,
  Blocksworld's `on` becomes `capacity-predecessor` and its blocks become packages.

L1 and L3 change exactly the same symbols, so comparing them separates missing cues from
wrong ones. Comments are removed at L1–L3, since they leak names. The renaming is a
bijection per domain and level and is stored, so every plan found on an obfuscated task
is translated back and validated on the original task. [[If the team chose object
renaming at L1, say so here: "Unlike the paper, our L1 keeps object names; L2 renames
them." or the opposite.]]

**Correctness of the obfuscator.** Obfuscation must not change the planning problem. We
check this in three ways. (i) Unit tests: no original name survives, L1 is a pure
renaming of L0, and every mapping is injective. (ii) Every plan found at any level is
valid for the original task. (iii) Pyperplan's h_FF, which ignores names, must solve the
same tasks at every level. Check (iii) holds on the test subset
([[hff coverage per level, e.g. 120/120/120/120]]). Expansion counts do vary between
levels, sometimes by a factor of four, even for h_FF. This is because Pyperplan's
grounding order, and with it GBFS tie-breaking, depends on the symbol names. The paper's
Fast Downward h_FF shows the same effect at a smaller scale (27 vs. 28 tasks). We therefore
compare each LLM heuristic with h_FF *at the same level*, and not only with L0.

**Protocol.** For each level and model we generate n = 10 candidates per domain with the
authors' unchanged prompt and code. We use DeepSeek R1 (`deepseek-reasoner`) with
3 independent generation runs ("seeds") and DeepSeek V3 (`deepseek-chat`) with one,
temperature 1.0 and top-p 0.5 as in the paper. The heuristic class gets a neutral name
(`CandidateHeuristic`), because the paper's names (`blocksworld10Heuristic`) would reveal
the domain. We regenerate L0 with the same API model instead of reusing the archived
heuristics, because the models behind the API names have changed since March 2025
([[record model versions and dates from axis1/heuristics/generation.jsonl]]). The selection
follows the paper's rule on the 99 training tasks, with 60 s per run: most tasks solved,
then the expansion score, then the agile score. The selected heuristic per configuration is
tested on 30 test tasks per domain (every third, 240 in total), with GBFS in Pyperplan,
5 min, 4 GiB and one core per run. The shared protocol (Section 4) lists everything else.

**Hypotheses.** (H1) Coverage falls from L0 to L2. (H2) The fall is largest in famous
domains (Blocksworld, Miconic, Sokoban). (H3) L3 hurts more than L1. (H4) h_FF coverage is
unchanged across levels.

### Results

[[Figure F1: `axis1/figures/f1_deepseek-r1.pdf`. Caption: Test coverage (of 30 tasks)
of the selected DeepSeek R1 heuristic per domain at obfuscation levels L0–L3; bars show
the mean over 3 generation runs, whiskers the range; diamonds show Pyperplan h_FF at the
same level.]]

[[Table: total coverage per level for R1 and V3, with h_FF and blind at each level.]]

Paragraphs to write once the numbers are in, one claim each:

1. **Overall drop (H1).** Total R1 coverage goes from [[L0]] to [[L1]] to [[L2]] out of
   240; h_FF stays at [[x]] (H4). Compare the size of the drop with the paper's Blocksworld
   result (66 → 40 of 90 at 30 min).
2. **Where names matter (H2).** Name the domains with the largest and smallest drops and
   whether the famous domains are among the largest. Note any domain where the LLM
   heuristic stays above h_FF even at L2: there it reasons from the structure.
3. **Missing vs. wrong cues (H3).** Compare L1 with L3 per domain. If L3 < L1, a plausible
   but wrong story actively misleads the model. If L3 ≈ L1, the model ignores the
   vocabulary once it doesn't fit the structure.
4. **Candidate quality.** From the failure table: does the share of candidates that crash,
   return infinity or time out on every training task grow with the level? This separates
   "worse heuristics" from "fewer working heuristics".
5. **R1 vs. V3.** Is the reasoning model more robust to obfuscation than the chat model?

**Limitations.** Shorter limits than the paper (5 min / 4 GiB vs. 30 min / 8 GiB), so
absolute coverage is lower. One model family. Three seeds give a range, not a confidence
interval. L3 uses one donor domain per domain; another pairing could mislead more or less.
