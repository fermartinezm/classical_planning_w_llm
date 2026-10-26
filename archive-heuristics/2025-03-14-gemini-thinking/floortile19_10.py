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
    This heuristic estimates the number of actions required to reach the goal state in the floortile domain.
    It focuses on the number of tiles that are not yet painted according to the goal and estimates the cost
    based on painting actions, color change actions, and movement actions.

    # Assumptions
    - Each unpainted tile in the goal requires at least one paint action.
    - If the robot does not have the correct color to paint a tile, a color change action is needed.
    - Reaching each unpainted tile requires at least one move action.
    - Available colors are always accessible for color changes if needed.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - Static facts are not explicitly used in this version of the heuristic but could be incorporated for more advanced heuristics (e.g., considering tile layout for movement cost).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1. Initialize the heuristic value to 0.
    2. Identify the robot's current location and the color it is currently holding from the current state.
    3. Iterate through each goal predicate in the task's goal definition.
    4. For each goal predicate of the form `(painted tile_x color_c)`:
        a. Check if this predicate is already true in the current state.
        b. If the predicate is NOT true in the current state, it means the tile `tile_x` needs to be painted with color `color_c`.
           i. Increment the heuristic value by 1. This accounts for the 'paint' action itself.
           ii. Check if the robot currently holds the color `color_c`.
           iii. If the robot does NOT hold the color `color_c`, increment the heuristic value by 1. This accounts for a 'change_color' action.
           iv. Increment the heuristic value by 1. This accounts for a 'move' action to reach the tile to be painted.
    5. Return the accumulated heuristic value.

    This heuristic is a simplification and does not guarantee admissibility, but it aims to provide a reasonable estimate
    of the remaining actions by considering the key action types in the floortile domain: painting, color changing, and movement.
    It prioritizes states that are closer to fulfilling the goal conditions in terms of painted tiles and correct robot color.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        # Get robot's current location and held color
        robot_location = None
        robot_color = None
        for fact in state:
            if match(fact, 'robot-at', '*', '*'):
                robot_location = get_parts(fact)[2]
            if match(fact, 'robot-has', '*', '*'):
                robot_color = get_parts(fact)[2]

        if robot_location is None:
            return float('inf') # No robot in state, should not happen in valid problems but handle for robustness

        for goal in self.goals:
            if match(goal, 'painted', '*', '*'):
                if goal not in state:
                    heuristic_value += 1 # For paint action
                    goal_tile_color = get_parts(goal)[2]
                    if robot_color != goal_tile_color:
                        heuristic_value += 1 # For change_color action
                    heuristic_value += 1 # For move action

        return heuristic_value
