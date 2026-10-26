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
    This heuristic estimates the number of actions required to achieve the goal state in the floortile domain.
    It focuses on the number of tiles that are not yet painted with the desired color, and considers the necessity
    of changing the robot's color to match the required paint color for each unpainted tile.

    # Assumptions
    - The primary actions are painting tiles and changing robot color.
    - Movement actions are implicitly considered necessary to reach tiles to be painted, but their cost is not explicitly calculated.
    - The heuristic is admissible if we assume that for each unpainted tile, we need at least one paint action and potentially one color change action. However, admissibility is not a requirement.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate of the form `(painted tile_x color_c)` in the task's goal definition.
    3. For each goal predicate, check if it is satisfied in the current state.
    4. If the goal predicate `(painted tile_x color_c)` is NOT satisfied in the current state:
        a. Increment the heuristic value by 1. This accounts for the 'paint' action needed.
        b. Check if the robot currently has the color `color_c`.
        c. To do this, iterate through the current state and look for a predicate `(robot-has robot1 color_current)`.
        d. If the `color_current` is not equal to `color_c`, increment the heuristic value by 1. This accounts for a 'change_color' action if the robot has the wrong color.
    5. Return the total accumulated heuristic value.

    This heuristic estimates the minimum number of 'paint' and 'change_color' actions required to achieve the goal.
    It is a simplification and does not explicitly account for movement actions, but it provides a reasonable estimate
    of the remaining work by focusing on the core tasks of painting and color management.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state.
        The heuristic value is an estimate of the number of actions needed to reach the goal state.
        """
        state = node.state
        heuristic_value = 0

        # Get the robot's current color
        robot_color = None
        for fact in state:
            if match(fact, 'robot-has', 'robot1', '*'):
                robot_color = get_parts(fact)[2]
                break

        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1 # Assume at least one paint action is needed

                goal_parts = get_parts(goal)
                goal_color = goal_parts[2]

                if robot_color != goal_color:
                    heuristic_value += 1 # Assume a color change is needed if robot has wrong color

        return heuristic_value
