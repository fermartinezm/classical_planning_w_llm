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


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts specified in the goal.
    It counts the number of goal nuts that are not yet tightened in the current state.
    Each untightened goal nut contributes a cost of 1 to the heuristic, representing a lower bound on the number of actions required.

    # Assumptions:
    - The heuristic assumes that for each nut that needs to be tightened, there exists a sequence of actions to tighten it.
    - It does not consider the costs of walking or picking up spanners explicitly, only the tightening action itself as a unit cost.
    - It is a simplification and may underestimate the actual number of actions needed.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal conditions from the task.
    - It identifies the nuts that are required to be 'tightened' in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal nuts: Extract all nuts that must satisfy the '(tightened ?n)' predicate in the goal.
    2. For each goal nut, check the current state: Determine if the nut is already '(tightened ?n)' in the current state.
    3. Count untightened goal nuts: Count how many goal nuts are not '(tightened ?n)' in the current state.
    4. Heuristic value: The heuristic value is the count of untightened goal nuts. This represents a minimum number of 'tighten_nut' actions possibly required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        untightened_goal_nuts_count = 0
        for nut in self.goal_nuts:
            if not f'(tightened {nut})' in state:
                untightened_goal_nuts_count += 1
        return untightened_goal_nuts_count
