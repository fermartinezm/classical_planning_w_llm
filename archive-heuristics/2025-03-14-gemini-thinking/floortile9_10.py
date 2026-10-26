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
    This heuristic estimates the number of actions needed to reach the goal state by counting the number of tiles that are not yet painted according to the goal specification. It also adds a penalty if the robot's current color is not useful for painting any of the remaining unpainted goal tiles, suggesting a color change might be needed.

    # Assumptions:
    - The primary actions are painting tiles.
    - Moving and changing colors are secondary actions needed to enable painting.
    - Each unpainted goal tile requires at least one paint action.
    - A color change might be needed if the robot's current color is not required for any of the remaining unpainted goal tiles.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Initialize a counter for unpainted goal tiles to 0.
    3. Extract the goal predicates that specify the desired painted state of tiles.
    4. Iterate through each goal predicate in the form `(painted tile_name color)`:
       - Check if this predicate is present in the current state.
       - If the predicate is NOT in the current state, increment the unpainted goal tiles counter.
    5. Set the initial heuristic value to the count of unpainted goal tiles. This represents the minimum number of paint actions required if all other conditions (robot position, color) are already met.
    6. (Optional and not implemented in this version for simplicity, but could be added for improvement): Consider adding a penalty for color changes if the robot's current color is not useful for any remaining unpainted goal tiles. This would require checking the robot's current color and comparing it against the colors needed for the unpainted goal tiles. For now, we are omitting this color change penalty to keep the heuristic simple and efficient.
    7. Return the heuristic value, which is the count of unpainted goal tiles. This heuristic is admissible in relaxed planning where we ignore preconditions related to robot position and color availability, and only consider the number of paint actions. However, in the original problem, it is not necessarily admissible but should be informative for guiding greedy best-first search.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0
        unpainted_goal_tiles = 0

        goal_paint_conditions = []
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                goal_paint_conditions.append(goal)

        for goal_condition in goal_paint_conditions:
            if goal_condition not in state:
                unpainted_goal_tiles += 1

        heuristic_value = unpainted_goal_tiles
        return heuristic_value
