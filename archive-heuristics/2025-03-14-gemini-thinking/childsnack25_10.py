from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all waiting children.
    It counts the number of children who are currently not served and assumes that at least one action is required for each of them to be served.
    This heuristic is a simple count of unserved children.

    # Assumptions:
    - Each unserved child requires at least one action to be served.
    - This heuristic does not consider the complexity of making sandwiches, putting them on trays, or moving trays.
    - It assumes that serving each child is an independent task, and does not account for potential action sharing (e.g., moving a tray to serve multiple children at the same location).

    # Heuristic Initialization
    - The heuristic initialization is minimal. It does not require any pre-computation or extraction of static facts beyond what is readily available in the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children from the goal description.
    2. For the current state, count how many of these children are NOT yet 'served'.
    3. The heuristic value is simply the count of unserved children.
    """

    def __init__(self, task):
        """Initialize the heuristic. Extracts goal conditions."""
        self.goals = task.goals
        self.children_in_goal = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                self.children_in_goal.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state from the current state."""
        state = node.state
        unserved_children_count = 0
        for child in self.children_in_goal:
            served_predicate = f'(served {child})'
            if served_predicate not in state:
                unserved_children_count += 1
        return unserved_children_count
