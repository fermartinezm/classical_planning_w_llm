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
    This heuristic estimates the number of actions required to reach the goal state by counting the number of goal predicates that are not satisfied in the current state.
    It is a simple goal-count heuristic, where each unsatisfied goal predicate contributes a cost of 1 to the heuristic value.

    # Assumptions
    - Each unsatisfied goal predicate (painted tile with a specific color) requires at least one action to be achieved.
    - The heuristic does not consider the cost of moving the robot or changing colors explicitly. It assumes that for each unpainted tile in the goal, some action (painting) is necessary.
    - This heuristic is admissible in relaxed planning where we ignore preconditions, but it is not guaranteed to be admissible in general planning. However, it is expected to be efficiently computable and guide the search effectively in many cases.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal definition.
    3. For each goal predicate, check if it is present in the current state.
    4. If a goal predicate is NOT present in the current state, increment the heuristic value by 1.
    5. After checking all goal predicates, the accumulated heuristic value is returned as the estimated cost to reach the goal.

    This heuristic essentially counts the number of tiles that are not yet painted with the desired color according to the goal specification.
    It is a simplification and does not take into account the actions needed to move the robot, change colors, or the adjacency constraints for painting.
    However, it provides a quick and easy way to estimate the distance to the goal state, which can be effective in guiding a greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic by storing the goal predicates.
        """
        self.goals = task.goals

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        The heuristic value is the count of unsatisfied goal predicates in the state.
        """
        state = node.state
        heuristic_value = 0
        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1
        return heuristic_value
