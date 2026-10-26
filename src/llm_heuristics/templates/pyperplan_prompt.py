from string import Template

PYPERPLAN_PROMPT = Template("""
<problem-descripition>
Suppose you are a Python programmer creating a domain-dependent heuristic function for the domain <domain>$name</domain> for the planner Pyperplan. You will receive the PDDL domain file, two PDDL instances example files, two files with Python code from the planner Pyperplan with domain-independent heuristic functions, one example of a set of facts and how they are represented internally, one example of a set of static facts and how they are represented internally, and one file with Python code from the planner Pyperplan for representing a STRIPS planning task. The domain-dependent heuristic function you will create should take into consideration the initial state and goal condition of the instance, and the operators available in the PDDL domain file. Atoms are internally represented as PDDL atoms (e.g., '(q obj1 ob2)'). The heuristic function you will create will be used in the Pyperplan planner with greedy best-first search to solve other instances of the same domain. The heuristic function should be efficient and accurate and should estimate the number of operator applications required to satisfy the goal condition from a given state. The name of the heuristic should be $heuristic_name. The new domain-dependent heuristic function should minimize as much as possible the number of expanded nodes when guiding greedy best-first search.
</problem-descripition>

<domain-file>
$domain
</domain-file>

<instance-file-example-1>
$instance1
</instance-file-example-1>

<instance-file-example-2>
$instance2
</instance-file-example-2>

<code-file-heuristic-1>
$heuristic1
</code-file-heuristic-1>

<code-file-heuristic-2>
$heuristic2
</code-file-heuristic-2>

<facts>
$state
</facts>

<static>
$static
</static>

<code-file-task>
$task
</code-file-task>

Now, provide only the Python code of the domain-dependent heuristic for the domain. Here is a checklist of things to help you with your code:
1) Check that facts are parsed correctly, and not taking into account the '(' and ')' in the extremes
2) Check that the heuristic is not 0 in non-goal states
3) Check that the heuristic value is finite for solvable states
4) Check if all necessary modules are imported
5) Check that static facts are accessed correctly
""")
