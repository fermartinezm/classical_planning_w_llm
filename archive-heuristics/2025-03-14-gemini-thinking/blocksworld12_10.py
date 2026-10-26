from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of unsatisfied goal predicates in the current state.
    It counts how many goal conditions are not met in the given state, assuming that each unsatisfied goal predicate requires at least one action to be achieved.
    This is a simple and fast heuristic, prioritizing states that satisfy more goal conditions.

    # Assumptions:
    - Each unsatisfied goal predicate requires at least one action to be achieved.
    - The heuristic is non-admissible but aims to guide the search efficiently by quickly evaluating states.

    # Heuristic Initialization
    - The heuristic is initialized with the goal conditions of the planning task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate defined in the task.
    3. For each goal predicate, check if it is present in the current state.
    4. If a goal predicate is NOT present in the current state, increment the heuristic value by 1.
    5. The final heuristic value is the total count of unsatisfied goal predicates.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal conditions from the task.
        """
        self.goals = task.goals

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        The heuristic value is the number of goal predicates that are not satisfied in the current state.
        """
        state = node.state
        heuristic_value = 0
        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1
        return heuristic_value
