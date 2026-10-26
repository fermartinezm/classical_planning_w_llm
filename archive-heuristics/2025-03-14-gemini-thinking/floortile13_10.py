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
    This heuristic estimates the number of actions required to paint all tiles according to the goal specification.
    It counts the number of tiles that are not yet painted with the correct color in the current state.
    For each such tile, it adds a cost of 1 (for a paint action). Additionally, if the robot does not have the correct color to paint the tile, it adds another cost of 1 (for a change_color action).
    This heuristic is admissible if we assume that for each unpainted tile, we need at least one paint action and potentially one color change action. However, it is not guaranteed to be admissible due to ignoring movement costs.

    # Assumptions:
    - The primary actions are `paint_up`, `paint_down`. We assume that painting in any direction (up/down/left/right if actions existed) has the same cost.
    - We simplify by assuming that the robot is always in a position from which it can paint the required tiles (ignoring move actions).
    - We only consider the cost of `paint_` and `change_color` actions.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates related to `painted` tiles.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal conditions that are `painted` predicates.
    3. For each goal `(painted tile color)`:
        a. Check if this goal is already achieved in the current state.
        b. If not achieved, increment the heuristic value by 1 (for a `paint_` action).
        c. Check if the robot currently `robot-has` the required `color`.
        d. To do this, get the color from the goal predicate. Then check in the current state if there is a fact `(robot-has robot color)`.
        e. If the robot does not have the correct color, increment the heuristic value by 1 (for a `change_color` action).
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions related to 'painted' predicates.
        """
        self.goal_painted_tiles = []
        for goal in task.goals:
            if match(goal, "painted", "*", "*"):
                self.goal_painted_tiles.append(goal)

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        # Get the robot's current color
        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        for goal_painted in self.goal_painted_tiles:
            if goal_painted not in state:
                heuristic_value += 1  # Cost for painting

                goal_parts = get_parts(goal_painted)
                goal_color = goal_parts[2]

                if robot_color != goal_color:
                    heuristic_value += 1  # Cost for changing color (if needed for this tile)
                    robot_color = goal_color # Assume color is changed after the action for next tile

        return heuristic_value
