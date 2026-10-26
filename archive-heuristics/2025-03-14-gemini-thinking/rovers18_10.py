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
    This heuristic estimates the number of actions needed to achieve all goal conditions in the rovers domain.
    It counts the number of goal predicates that are not yet satisfied in the current state.
    This serves as a simple estimate of the remaining work.

    # Assumptions:
    - Each unsatisfied goal predicate requires at least one action to be achieved.
    - This heuristic does not consider the cost of individual actions or the dependencies between goals.
    - It assumes that all goals are achievable.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are pre-processed in this simple version of the heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal definition.
    3. For each goal predicate, check if it is present in the current state.
    4. If a goal predicate is NOT present in the current state, increment the heuristic value by 1.
    5. After checking all goal predicates, the accumulated heuristic value is returned as the estimate of the remaining actions.

    This heuristic is a simple goal-counting heuristic. It is admissible under the very loose assumption that each goal requires at least one action and actions do not interfere negatively with achieving other goals (which is generally not true, but it can still be effective in guiding greedy search).
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state from the current state.
        This is done by counting the number of goal predicates not yet achieved in the current state.
        """
        state = node.state
        heuristic_value = 0
        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1
        return heuristic_value
