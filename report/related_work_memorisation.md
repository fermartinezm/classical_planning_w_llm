# Related work: LLMs, planning and memorisation (draft, Member 1 part)

**LLMs as planners.** Valmeekam et al. [3, 4] showed that LLMs asked to output plans
directly are unreliable on classical planning benchmarks. Their PlanBench benchmark
includes "Mystery Blocksworld", where actions and predicates are renamed so the domain can
no longer be recognised; performance drops sharply there, which suggests that part of the
apparent planning ability comes from recognising familiar domains rather than reasoning
about the given model. Kambhampati et al. [5] argue that LLMs should instead be used inside
frameworks where sound external components check and use their output.

**LLMs writing planning code.** A line of work follows this advice by asking the LLM for
code that a search algorithm uses: generalized plans as Python programs [6], successor and
goal functions [7], and domain-dependent heuristics [1]. Because the planner checks every
step, the plans are always correct; what the LLM contributes is guidance. Corrêa et al. [1]
show that such heuristics, used in GBFS, solve more tasks than $h^{\mathrm{FF}}$, and include a
single-domain obfuscation test. Our Axis 1 asks whether this guidance survives when the
names that identify the domain are removed or replaced.

**Memorisation and data contamination.** Widely used benchmarks are likely part of LLM
training data, which can inflate results [8]. Wu et al. [9] compare models on standard
tasks and on "counterfactual" variants with the same structure but changed conventions,
and find consistent drops, i.e. models partly recite familiar patterns. IPC domains such as
Blocksworld, Sokoban or Miconic, and heuristics written for them, are discussed widely
online. Renaming symbols (our L1/L2) removes the cue that triggers recall; misleading names
(our L3) play the role of a counterfactual variant, where recalled knowledge is wrong.

## References

[1] A. B. Corrêa, A. G. Pereira, J. Seipp. Classical Planning with LLM-Generated Heuristics:
    Challenging the State of the Art with Python Code. NeurIPS 2025.
[3] K. Valmeekam, M. Marquez, S. Sreedharan, S. Kambhampati. On the Planning Abilities of
    Large Language Models: A Critical Investigation. NeurIPS 2023.
[4] K. Valmeekam, M. Marquez, A. Olmo, S. Sreedharan, S. Kambhampati. PlanBench: An Extensible
    Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.
    NeurIPS 2023 (Datasets and Benchmarks).
[5] S. Kambhampati et al. Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo
    Frameworks. ICML 2024.
[6] T. Silver et al. Generalized Planning in PDDL Domains with Pretrained Large Language
    Models. AAAI 2024.
[7] M. Katz et al. Thought of Search: Planning with Language Models Through the Lens of
    Efficiency. NeurIPS 2024.
[8] O. Sainz et al. NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination
    for each Benchmark. Findings of EMNLP 2023.
[9] Z. Wu et al. Reasoning or Reciting? Exploring the Capabilities and Limitations of Language
    Models Through Counterfactual Tasks. NAACL 2024.

(Numbering to be merged with the team bibliography; [2] is reserved for the h_FF reference.)
