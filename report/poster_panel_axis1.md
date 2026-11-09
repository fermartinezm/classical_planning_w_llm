# Poster panel — "Do names matter?" (Axis 1)

**Question.** Is the LLM reasoning about the PDDL, or recognising a famous domain by its names?

**Setup.** Same 8 domains, four versions each:

- **L0** original · **L1** random names for predicates/actions/types
- **L2** random names everywhere + shuffled order · **L3** names borrowed from another domain
  (Blocksworld written with Transport's words)

Same model, same prompt, 10 candidates per domain, best one on training → 30 test tasks (5 min).

**Sanity check.** h_FF solves the same tasks at every level (160/160 training tasks at L0–L3):
the obfuscated tasks are the same problems.

**Figure.** F1 — coverage per domain at L0–L3, h_FF as a marker. [TBD]

**Take-away.** [TBD, one sentence, e.g. "Names explain X of the gain over h_FF; misleading
names hurt more/less than missing names."]
