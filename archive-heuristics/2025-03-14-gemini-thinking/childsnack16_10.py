from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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
    It counts the number of children who are currently waiting but not yet served and
    multiplies this count by a constant factor (estimated actions per child).

    # Assumptions:
    - For each waiting child, a sandwich needs to be made, put on a tray, and served.
    - We assume that there are always enough resources (bread and content) at the kitchen to make sandwiches.
    - Moving trays is considered implicitly in the estimated cost per child.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base Heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are currently in a 'waiting' state.
    2. For each waiting child, check if they are already 'served'.
    3. Count the number of children who are 'waiting' but not 'served'.
    4. Estimate the number of actions required to serve each unserved child.
       A constant value (e.g., 3) can be used as a rough estimate, representing actions like
       making a sandwich, putting it on a tray, and serving it.
    5. The heuristic value is the product of the number of unserved children and the estimated
       actions per child. This provides an estimate of the remaining actions needed to reach the goal.
    """

    def __init__(self, task):
        """Initialize the childsnack heuristic."""
        super().__init__(task)
        self.goal_facts = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """Estimate the number of actions to serve all waiting children."""
        state = node.state
        unserved_children_count = 0

        waiting_children = set()
        served_children = set()

        for fact in state:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                waiting_children.add(parts[1])
            if match(fact, "served", "*"):
                parts = get_parts(fact)
                served_children.add(parts[1])

        for child in waiting_children:
            if child not in served_children:
                unserved_children_count += 1

        # Estimate actions per unserved child (e.g., make, put_on_tray, serve)
        actions_per_child = 3
        heuristic_value = unserved_children_count * actions_per_child

        # If all goals are already achieved, the heuristic value is 0.
        if self.task.goal_reached(state):
            return 0

        return heuristic_value
