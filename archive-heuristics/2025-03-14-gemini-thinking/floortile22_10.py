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
    This heuristic estimates the number of tiles that are not yet painted according to the goal specification.
    It counts the number of goal `painted` predicates that are not present in the current state.
    This serves as a lower bound on the number of paint actions required.

    # Assumptions
    - Each unpainted tile in the goal requires at least one paint action.
    - Color changes and robot movements are not explicitly considered in this basic heuristic.
    - It assumes that for each goal `(painted tile color)`, at least one `paint_up` or `paint_down` action is needed if the tile is not already painted with the correct color.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - Static facts are not used in this basic heuristic but could be incorporated for more refined versions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate in the task's goal definition.
    3. For each goal predicate that is of the form `(painted ?tile ?color)`:
        a. Check if this exact predicate is present in the current state.
        b. If the predicate is NOT present in the current state, increment the heuristic value by 1.
    4. Return the final heuristic value, which represents the count of unsatisfied `painted` goal conditions.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic.

        - Extracts and stores the goal predicates from the task.
        - (Currently, static facts are not used in this basic heuristic).
        """
        self.goals = task.goals  # Store goal predicates

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        - Counts the number of goal `painted` predicates that are not satisfied in the current state.
        - Returns this count as the heuristic estimate.
        """
        state = node.state
        heuristic_value = 0

        for goal in self.goals:
            if match(goal, "painted", "*", "*"): # Consider only painted goals
                if goal not in state: # Check if the goal fact is in the current state
                    heuristic_value += 1 # Increment for each unsatisfied painted goal

        return heuristic_value
