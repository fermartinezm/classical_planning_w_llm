from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by counting the number of unsatisfied goal predicates related to tile painting.
    It assumes that for each tile that is not painted as required in the goal, at least one paint action is needed.

    # Assumptions
    - The primary goal is to paint tiles with specific colors.
    - Movement and color change actions are necessary to enable painting, but the heuristic focuses on the painting actions themselves as the main driver towards the goal.
    - It is a simplification and does not explicitly account for the number of move or change_color actions needed.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal definition.
    3. For each goal predicate, check if it is satisfied in the current state.
    4. If a goal predicate is not satisfied in the current state, increment the heuristic value by 1.
    5. The final heuristic value is the total count of unsatisfied goal predicates.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal conditions.
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
