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
    This heuristic estimates the number of actions needed to reach the goal state by counting the number of goal predicates that are not satisfied in the current state.
    Each unsatisfied goal predicate contributes a cost of 1 to the heuristic value. This heuristic is admissible in relaxed planning but not necessarily admissible in standard planning. It is efficient to compute and aims to guide the search towards goal states by prioritizing states that satisfy more goal conditions.

    # Assumptions
    - The goal state is defined by a conjunction of predicates involving `on`, `on-table`, and `clear`.
    - Each unsatisfied goal predicate requires at least one action to be satisfied.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are used in this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal definition.
    3. For each goal predicate, check if it is present in the current state.
    4. If a goal predicate is NOT present in the current state, increment the heuristic value by 1.
    5. After checking all goal predicates, the accumulated heuristic value is returned as the estimated cost to reach the goal state.

    This heuristic essentially counts the number of "missing" goal conditions in the current state, providing a simple and fast estimate of the distance to the goal.
    """

    def __init__(self, task):
        """
        Initialize the blocksworld heuristic by storing the goal predicates.
        """
        self.goals = task.goals

    def __call__(self, node):
        """
        Compute the heuristic value for a given state by counting unsatisfied goal predicates.
        """
        state = node.state
        heuristic_value = 0
        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1
        return heuristic_value
