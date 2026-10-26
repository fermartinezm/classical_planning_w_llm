from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of tiles that are not yet painted according to the goal specification.
    It counts the number of goal `painted` predicates that are not satisfied in the current state.
    Each unsatisfied goal predicate contributes 1 to the heuristic value, representing at least one paint action needed.

    # Assumptions:
    - The heuristic assumes that each unsatisfied goal `painted` predicate requires at least one action to be satisfied.
    - It does not consider the cost of moving the robot or changing colors.
    - It is a simple count of unsatisfied goal conditions related to painting.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal conditions from the task.
    - It focuses specifically on the `painted` predicates in the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate.
    3. For each goal predicate, check if it is a `painted` predicate.
    4. If it is a `painted` predicate, check if this predicate is present in the current state.
    5. If the `painted` goal predicate is NOT present in the current state, increment the heuristic value by 1.
    6. After checking all goal predicates, the accumulated heuristic value is returned.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        for goal_fact in self.goals:
            if match(goal_fact, "painted", "*", "*"):
                if goal_fact not in state:
                    heuristic_value += 1
        return heuristic_value
