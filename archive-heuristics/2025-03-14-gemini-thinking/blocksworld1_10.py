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
    This heuristic estimates the number of blocks that are not in their goal positions.
    It counts the number of unsatisfied goal predicates related to block positions (on, on-table, clear).
    This heuristic is admissible in relaxed planning sense, as it assumes each unsatisfied condition requires at least one action to fix.

    # Assumptions
    - The goal state is defined by a set of `on`, `on-table`, and `clear` predicates.
    - The heuristic focuses on achieving the goal block configuration and does not explicitly consider the `arm-empty` or `holding` predicates.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate.
    3. For each goal predicate, check if it is present in the current state.
    4. If a goal predicate is not present in the current state, increment the heuristic value by 1.
    5. The final heuristic value is the total count of unsatisfied goal predicates.
    """

    def __init__(self, task):
        """Initialize the blocksworldHeuristic by storing the goal predicates."""
        self.goals = task.goals

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state from the current state.
        This is done by counting the number of goal predicates that are not satisfied in the current state.
        """
        state = node.state
        heuristic_value = 0
        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1
        return heuristic_value
