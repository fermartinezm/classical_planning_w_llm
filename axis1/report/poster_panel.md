# Poster panel: "Do names matter?" (draft, Member 1)

**Headline (fill in):** Renaming everything costs R1 heuristics [[x]]% of their coverage;
misleading names cost [[y]]%. h_FF does not change.

**Setup (one line per level):**
- L0: original PDDL
- L1: random names (`on` → `nynkyy`)
- L2: random names + random object names + shuffled order
- L3: wrong but meaningful names (Blocksworld written as Transport: `on` → `capacity-predecessor`)

**Figure:** F1, coverage per domain at L0–L3, with h_FF diamonds (`axis1/figures/f1_deepseek-r1.pdf`).

**Three takeaways (fill in after the runs):**
1. Overall: [[L0 → L2 coverage, out of 240]].
2. Most affected: [[domains]]. Least affected: [[domains]].
3. Wrong names vs. no names: [[L3 vs. L1]].

**Sanity check box:** h_FF solves the same [[n]] tasks at every level, and every plan is
validated on the original task.

**What to say at the defense (30 s):** The paper tested one obfuscated domain with one model.
We did all eight domains with two models at four levels, including names that tell the
model a wrong story. This separates how much comes from reasoning about the structure and
how much from recognising a well-known benchmark.
