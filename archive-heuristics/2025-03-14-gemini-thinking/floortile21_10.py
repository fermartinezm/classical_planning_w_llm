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
    It counts the number of goal `painted` facts that are not present in the current state.
    Each unpainted goal tile is assumed to require at least one paint action.

    # Assumptions
    - Each tile in the goal that is not painted in the current state requires at least one action to be painted.
    - The heuristic does not consider the cost of moving the robot or changing colors.
    - It is a simple count of unsatisfied goal conditions related to painting.

    # Heuristic Initialization
    - Extracts the goal conditions from the task.
    - Stores only the `painted` goal facts for efficient checking during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal fact.
    3. For each goal fact, check if it is a `painted` fact.
    4. If it is a `painted` fact, check if this fact is present in the current state.
    5. If the `painted` goal fact is NOT present in the current state, increment the heuristic value by 1.
    6. Return the total heuristic value, which represents the count of unsatisfied `painted` goal conditions.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic.
        Extracts and stores the goal conditions related to 'painted' tiles.
        """
        self.goal_painted_tiles = set()
        for goal in task.goals:
            if match(goal, "painted", "*", "*"):
                self.goal_painted_tiles.add(goal)

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        Counts the number of goal 'painted' facts that are not in the current state.
        """
        state = node.state
        heuristic_value = 0
        for goal_painted_fact in self.goal_painted_tiles:
            if goal_painted_fact not in state:
                heuristic_value += 1
        return heuristic_value
