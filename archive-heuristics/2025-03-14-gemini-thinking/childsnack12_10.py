from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robby rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of children that are not yet served.
    It counts the number of children for whom the 'served' predicate is not true in the current state,
    assuming that each unserved child requires at least one serving action.

    # Assumptions
    - Serving each child requires at least one action.
    - The goal is to have all specified children served.

    # Heuristic Initialization
    - Extracts the list of children that need to be served from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that are required to be 'served' according to the goal definition.
    2. For the given state, check for each goal child if the predicate '(served child)' is present.
    3. Count the number of goal children for whom the '(served child)' predicate is NOT present in the current state.
    4. The count obtained in step 3 is the heuristic estimate, representing the number of children yet to be served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal children from the task's goal conditions.
        """
        self.goal_children = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state from the current state.
        This heuristic counts the number of children who are in the goal but are not yet served in the current state.
        """
        state = node.state
        unserved_children_count = 0
        for child in self.goal_children:
            served_predicate = f'(served {child})'
            if served_predicate not in state:
                unserved_children_count += 1
        return unserved_children_count
