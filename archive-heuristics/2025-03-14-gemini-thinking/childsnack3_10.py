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
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of children who are currently waiting and not yet served,
    and multiplies this count by a constant factor representing the estimated actions
    needed per child (making sandwich, putting on tray, serving).

    # Assumptions:
    - For each unserved child, we minimally need to perform actions to make a sandwich,
      put it on a tray, and serve it.
    - Tray movements are not explicitly considered in this simplified heuristic, assuming
      trays are readily available at serving locations or tray movement cost is implicitly
      covered by the constant factor.
    - Sufficient bread and content portions are always available in the kitchen.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - It also extracts static information about children's allergies and waiting places,
      although this simplified version does not directly use allergy information in the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal children: Extract all children that are expected to be 'served' in the goal state.
    2. Identify served children in the current state: Count how many children are already 'served' in the current state.
    3. Calculate unserved children: Subtract the number of served children from the total number of goal children.
    4. Estimate actions per unserved child: Assume a constant number of actions (e.g., 3) are needed for each unserved child (make, put on tray, serve).
    5. Compute heuristic value: Multiply the number of unserved children by the estimated actions per child.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                self.goal_children.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions to serve all children."""
        state = node.state
        served_children_count = 0
        for fact in state:
            if match(fact, "served", "*"):
                served_children_count += 1

        unserved_children_count = len(self.goal_children) - served_children_count
        if unserved_children_count < 0:
            unserved_children_count = 0 # In case of over-achieving goals in intermediate states.

        # Estimate 3 actions per unserved child (make, put on tray, serve)
        heuristic_value = unserved_children_count * 3
        return heuristic_value
