# Related work: LLM memorisation and contamination (draft, Member 1)

Draft for the memorisation part of Section 3 (Member 2 writes the "LLMs for planning"
part). About 250 words; BibTeX keys are suggestions.

---

**Memorisation and contamination.** Benchmark results for LLMs are hard to interpret when
the test material may be part of the training data. Golchin and Surdeanu
[golchin2024timetravel] show that contamination can be detected by prompting a model to
complete partial benchmark instances, and that it is widespread. A complementary approach
changes the task instead of probing the model. Wu et al. [wu2024reasoning] build
"counterfactual" variants of 11 tasks that break default assumptions, and find that models
do consistently worse on them. They conclude that models rely partly on narrow,
memorised procedures. In planning, the standard tool for this is renaming. PlanBench's
Mystery Blocksworld [valmeekam2023planbench] keeps the problems but renames actions and
facts with unrelated words. GPT-4 drops from about 35% to under 5% plan accuracy there,
and even the reasoning model o1-preview drops from 97.8% to 52.8% [valmeekam2024lrms].
Silver et al. [silver2024generalized] ask GPT-4 to write generalised plans as Python
programs. When all names are replaced by identifiers such as `predicate1`, it solves
almost nothing in most domains.

IPC domains such as Blocksworld have been online for decades, so heuristics written by
LLMs may also be partly recalled rather than derived. Corrêa et al. [correa2025]
address this with three checks: a modified Spanner, an unpublished domain (Rod-Rings), and
an obfuscated Blocksworld. In the latter, DeepSeek R1's heuristic drops from 66 to 40
solved tasks, while h_FF is unaffected. Our Axis 1 extends this single data point to all
eight domains and two models. It also adds graded levels and a misleading vocabulary, a
condition closer to counterfactual tasks than to random renaming.

---

## References (check details before submission)

- [correa2025] A. B. Corrêa, A. G. Pereira, J. Seipp. Classical Planning with
  LLM-Generated Heuristics: Challenging the State of the Art with Python Code. NeurIPS 2025.
  https://arxiv.org/abs/2503.18809
- [golchin2024timetravel] S. Golchin, M. Surdeanu. Time Travel in LLMs: Tracing Data
  Contamination in Large Language Models. ICLR 2024. https://arxiv.org/abs/2308.08493
- [silver2024generalized] T. Silver, S. Dan, K. Srinivas, J. B. Tenenbaum, L. P. Kaelbling,
  M. Katz. Generalized Planning in PDDL Domains with Pretrained Large Language Models.
  AAAI 2024. https://arxiv.org/abs/2305.11014
- [valmeekam2023planbench] K. Valmeekam, M. Marquez, A. Olmo, S. Sreedharan,
  S. Kambhampati. PlanBench: An Extensible Benchmark for Evaluating Large Language Models on
  Planning and Reasoning about Change. NeurIPS 2023 Datasets and Benchmarks.
  https://arxiv.org/abs/2206.10498
- [valmeekam2023critical] K. Valmeekam, M. Marquez, S. Sreedharan, S. Kambhampati. On the
  Planning Abilities of Large Language Models: A Critical Investigation. NeurIPS 2023.
  https://arxiv.org/abs/2305.15771 (the obfuscation the paper says it follows)
- [valmeekam2024lrms] K. Valmeekam, K. Stechly, S. Kambhampati. LLMs Still Can't Plan;
  Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench. arXiv 2409.13373, 2024.
- [wu2024reasoning] Z. Wu et al. Reasoning or Reciting? Exploring the Capabilities and
  Limitations of Language Models Through Counterfactual Tasks. NAACL 2024.
  https://aclanthology.org/2024.naacl-long.102
