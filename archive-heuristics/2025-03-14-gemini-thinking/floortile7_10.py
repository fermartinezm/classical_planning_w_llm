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
    This heuristic estimates the number of actions required to paint all tiles according to the goal specification.
    It counts the number of unsatisfied goal conditions related to tile painting and adds a cost for each.
    The cost considers whether the robot needs to change color and perform a paint action for each unpainted tile.
    Movement cost is not explicitly considered in this simplified heuristic, focusing on paint and color change operations.

    # Assumptions
    - The primary actions are painting tiles and changing robot color.
    - Movement is assumed to be necessary to reach tiles, but its cost is implicitly included in the heuristic.
    - The heuristic is not admissible but aims to be informative and efficient for greedy best-first search.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - Static facts like adjacency relations and available colors are not explicitly used in this simplified heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition `(painted tile_x color_c)`:
    1. Check if the goal condition is already satisfied in the current state. If yes, no cost is added for this goal.
    2. If the goal condition is not satisfied:
        a. Increment the heuristic cost by 1, representing the cost of a 'paint' action.
        b. Determine the required color `color_c` for the tile.
        c. Check the color `color_r` the robot is currently holding in the given state.
        d. If `color_r` is not equal to `color_c`, increment the heuristic cost by 1, representing the cost of a 'change_color' action.
    3. The total heuristic value is the sum of the costs calculated for all unsatisfied goal conditions.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        # Get robot's current color
        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "robot1", "*"): # Assuming robot name is robot1
                robot_color = get_parts(fact)[2]
                break

        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1 # Cost for paint action
                goal_parts = get_parts(goal)
                goal_color = goal_parts[2] # color is the third element in (painted tile color)

                if robot_color is not None and robot_color != goal_color:
                    heuristic_value += 1 # Cost for change_color action

        return heuristic_value
