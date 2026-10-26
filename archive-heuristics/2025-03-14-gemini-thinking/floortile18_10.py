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
    This heuristic estimates the minimum number of actions required to achieve the goal state in the floortile domain.
    It focuses on the number of tiles that are not yet painted with the desired color in the goal and estimates the cost
    of painting these tiles, including potential color changes. It's a simple heuristic that prioritizes achieving
    the painting goals and assumes movement is less costly in comparison.

    # Assumptions:
    - The primary actions are painting tiles to match the goal specification.
    - Color changes are sometimes necessary and contribute to the cost.
    - Robot movement is implicitly considered necessary to reach tiles but is not explicitly costed in detail.
    - The heuristic is not admissible but aims to be informative and efficiently computable for greedy best-first search.

    # Heuristic Initialization
    - Extracts the goal predicates of the form `(painted ?tile ?color)` to determine which tiles need to be painted with which color.
    - Extracts the available colors from the static facts to check if a color change is possible.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal `painted` predicates from the task definition.
    3. For each goal `painted` predicate `(painted tile_x color_c)`:
        a. Check if the current state already satisfies this goal predicate. If yes, no cost is added for this goal.
        b. If the goal predicate is not satisfied in the current state:
            i. Increment the heuristic value by 1. This accounts for the `paint_up` or `paint_down` action needed to paint the tile.
            ii. Determine the required color `color_c` from the goal predicate.
            iii. Check if the robot currently `robot-has` the required color `color_c`.
            iv. If the robot does not have the required color, increment the heuristic value by 1. This accounts for a `change_color` action, assuming the color is available.
    4. Return the total accumulated heuristic value.

    This heuristic essentially counts the minimum number of paint actions and color change actions needed to satisfy all goal painting conditions,
    ignoring the movement actions required to reach the tiles. It's a simplification that prioritizes the core tasks of painting and color management.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic.

        Extracts goal predicates and available colors from the task definition.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_painted_tiles = []
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                self.goal_painted_tiles.append(goal)
        self.available_colors = set()
        for fact in self.static_facts:
            if match(fact, "available-color", "*"):
                self.available_colors.add(get_parts(fact)[1])


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        Estimates the number of actions needed to reach the goal state from the current state.
        """
        state = node.state
        heuristic_value = 0

        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        for goal_painted_fact in self.goal_painted_tiles:
            if goal_painted_fact not in state:
                heuristic_value += 1  # Cost for painting action
                goal_tile = get_parts(goal_painted_fact)[1]
                goal_color = get_parts(goal_painted_fact)[2]

                if robot_color != goal_color:
                    if goal_color in self.available_colors:
                        heuristic_value += 1 # Cost for color change
                        robot_color = goal_color # Assume color is changed for subsequent checks


        return heuristic_value
