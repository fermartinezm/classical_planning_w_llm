from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions in the rovers domain.
    It counts the number of goal predicates that are not yet satisfied in the current state.
    This serves as a simple estimate of the remaining work.

    # Assumptions:
    - Each unsatisfied goal predicate requires at least one action to be achieved.
    - This heuristic does not consider the complexity of achieving each goal predicate,
      such as navigation or resource constraints.
    - It assumes that all goal predicates are independent and do not interfere with each other.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are explicitly used in this simple heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal definition.
    3. For each goal predicate, check if it is present in the current state.
    4. If a goal predicate is NOT present in the current state, increment the heuristic value by 1.
    5. The final heuristic value is the total count of unsatisfied goal predicates.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state.
        This heuristic simply counts the number of goal predicates not yet achieved in the current state.
        """
        state = node.state
        unsatisfied_goals = 0
        for goal in self.goals:
            if goal not in state:
                unsatisfied_goals += 1
        return unsatisfied_goals
