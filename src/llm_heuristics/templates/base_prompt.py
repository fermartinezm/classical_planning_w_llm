from string import Template

BASE_PROMPT = Template("""
<problem-description>
You are a Python programmer creating a domain-dependent heuristic function for the PDDL domain <domain>$name</domain>. You will receive the PDDL domain file, two example PDDL instance files, two Python files with example domain-independent heuristic functions, an example of a set of facts from the domain and how they are represented internally, one example of a set of static facts and how they are represented internally, and one file with Python code for representing a planning task. The domain-dependent heuristic function you create should take into consideration the current state and the goal condition of the instance, and the actions available in the PDDL domain file. PDDL facts are represented as strings (e.g., '(pred obj1 ob2)'). The heuristic function you create will be used to solve other instances of the same domain. The heuristic function should be efficiently computable and accurate, that is, it should minimize the number of expanded nodes during the search. The name of the heuristic should be $heuristic_name.
</problem-description>

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

Provide only the Python code of the domain-dependent heuristic for the domain. Here is a checklist to help you with your code:
1) The code for extracting objects from facts remembers to ignore the surrounding brackets.
2) The heuristic is 0 only for goal states.
3) The heuristic value is finite for solvable states.
4) All used modules are imported.
5) Static facts are accessed correctly.
""")
