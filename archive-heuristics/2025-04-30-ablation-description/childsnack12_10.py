from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number of children who have not yet been served.

    # Assumptions
    - The primary goal is to serve all specified children.
    - Each 'serve' action satisfies the goal for exactly one child.
    - The cost of each action is 1.

    # Heuristic Initialization
    - The heuristic extracts the set of all children that need to be served from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of children that must be served to reach the goal state. This set is extracted during heuristic initialization from the task's goal facts (e.g., `(served child1)`).
    2. For the current state, iterate through the set of children identified in step 1.
    3. For each child, check if the fact `(served <child_name>)` is present in the current state.
    4. Count the number of children for whom the `(served <child_name>)` fact is *not* present in the current state.
    5. The heuristic value is this count. This value represents the minimum number of 'serve' actions required, assuming each action serves a unique unserved child. It ignores the prerequisites for serving (making sandwich, putting on tray, moving tray), making it a simple, admissible, but potentially not very informative heuristic for greedy search compared to one that considers intermediate steps. However, it satisfies the basic requirements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Extract the set of all children that need to be served from the goals.
        # Goals are typically of the form (served childX)
        self.children_to_serve = {
            get_parts(goal)[1] # The child object is the second part of '(served childX)'
            for goal in self.goals
            if match(goal, "served", "*")
        }

        # Static facts are available in task.static, but not strictly needed for this simple heuristic.
        # self.static_facts = task.static

    def __call__(self, node):
        """Estimate the minimum cost to serve all remaining children."""
        state = node.state

        # Count the number of children who are in the goal list but not yet served in the current state.
        unserved_children_count = 0
        for child in self.children_to_serve:
            served_fact = f"(served {child})"
            if served_fact not in state:
                unserved_children_count += 1

        # The heuristic value is the number of unserved children.
        # This is an admissible heuristic as each serve action reduces this count by at most 1 and costs 1.
        # It is 0 if and only if all children in self.children_to_serve are served, which matches the goal.
        return unserved_children_count
