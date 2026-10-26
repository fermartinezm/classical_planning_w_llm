from string import Template

IDEAS_PROMPT = Template("""
<problem-description>
You are a highly-skilled professor in AI planning creating a domain-dependent heuristic function for the PDDL domain <domain>$name</domain>. The heuristic function you create will be used to guide a greedy best-first search to solve instances from this domain. Therefore, the heuristic does not need to be admissible. For a given state, the heuristic function should estimate the required number of actions to reach a goal state as accurately as possible. The heuristic should be efficiently computable, and it should minimize the number of expanded nodes during the search. Next, you will receive a sequence of file contents to help you with your task and to show you the definition of the $name domain.
</problem-description>

This is the PDDL domain file of the $name domain, for which you need to create a domain-dependent heuristic:
<domain-file>
$domain
</domain-file>

This is an example of a PDDL instance file of the $name domain:
<instance-file-example-1>
$instance1
</instance-file-example-1>

This is a second example of a PDDL instance file of the $name domain:
<instance-file-example-2>
$instance2
</instance-file-example-2>

Provide five different ideas of domain-dependent heuristics for the <domain>$name</domain> domain that could be implemented later in Python code. Here is a checklist to help you with you:
1) The ideas should be different enough such that they could produce potentially different results while guiding a greedy best-first search
2) Some heuristics could be more efficiently computable, while others could be more accurate but less efficient.
3) Some heuristics should use as much domain-dependent knowledge as possible.
4) The description of the ideas of heuristics should use only natural language.
5) The heuristics should be designed to be used in the <domain>$name</domain> domain, and they should be able to handle the specific characteristics of this domain.
6) Take into account the information that would be available in the PDDL domain and instance files.
7) The heuristic is 0 only for goal states.
8) The heuristic value is finite for solvable states.
9) Provide enough details so someone can implement the idea in Python later.
10) Each heuristic idea heursitic should placed inside <ideia-i> and </ideia-i> tags.
11) Provide domain-dependent heuristic ideas for this specific domain, and for some of the heuristic ideas, do not use general ideas from domain-independent planning such as critical path heuristic and delete relaxation.
12) Some ideas should capture and account for interactions between different problem components.
13) Some ideas should account for the interaction between conflicting subgoals in intermediary parts of the plan.
14) Provide only the heuristic ideas inside <ideia-i> and </ideia-i> tags without any additional text or explanation.
""")
