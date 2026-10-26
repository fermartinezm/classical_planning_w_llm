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
    This heuristic estimates the number of actions required to reach the goal state by counting the number of tiles that are not yet painted with the correct color, and adding a potential cost for changing the robot's color if necessary. It prioritizes correcting the paint status of tiles.

    # Assumptions
    - The primary actions to achieve the goal are painting actions.
    - Color changing is a necessary action if the robot does not have the required color to paint the remaining unpainted tiles.
    - Movement actions are implicitly considered necessary to reach tiles to be painted, but their cost is not explicitly accounted for in detail, focusing more on paint and color change operations.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted with which color.
    - No static facts are explicitly used in this heuristic calculation, although the domain definition and problem instance implicitly define the possible tile connections and available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal painted tiles and their target colors from the task goals.
    2. For each goal condition `(painted tile color)`:
       - Check if this condition is already satisfied in the current state.
       - If not satisfied, increment a counter for unpainted tiles.
       - Store the required color for this unpainted tile.
    3. Determine the color the robot currently has from the state.
    4. Check if there are any unpainted tiles. If there are:
       - Check if the robot currently has at least one of the colors required to paint the unpainted tiles.
       - If the robot does not have any of the required colors for the unpainted tiles, increment the heuristic value by 1 to account for a color change action.
    5. The final heuristic value is the sum of the count of unpainted tiles and the potential color change cost (0 or 1).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions related to 'painted' tiles.
        """
        self.goal_painted_tiles = {}
        for goal in task.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_painted_tiles[tile] = color

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        """
        state = node.state
        unpainted_tiles_count = 0
        required_colors_for_unpainted_tiles = set()

        for tile, goal_color in self.goal_painted_tiles.items():
            painted_fact = f'(painted {tile} {goal_color})'
            if painted_fact not in state:
                unpainted_tiles_count += 1
                required_colors_for_unpainted_tiles.add(goal_color)

        if unpainted_tiles_count == 0:
            return 0

        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        color_change_cost = 0
        if required_colors_for_unpainted_tiles:
            if robot_color not in required_colors_for_unpainted_tiles:
                color_change_cost = 1

        return unpainted_tiles_count + color_change_cost
