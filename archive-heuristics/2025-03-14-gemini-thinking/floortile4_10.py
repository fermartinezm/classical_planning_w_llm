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
    It counts the number of tiles that are not yet painted with the correct color in the current state and
    estimates the cost to paint each of these tiles.

    # Assumptions:
    - For each incorrectly painted tile, we assume that on average one move action is needed to get adjacent to it.
    - A color change action might be needed if the robot does not have the required color.
    - One paint action is always needed to paint the tile.
    - The heuristic is not admissible but aims to guide the search effectively.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are explicitly used in the current heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal predicates that are of the form `(painted tile color)`.
    3. For each goal predicate `(painted goal_tile goal_color)`:
        a. Check if the current state already satisfies this goal, i.e., if `(painted goal_tile goal_color)` is present in the current state.
        b. If the goal is not satisfied:
            i. Increment the heuristic value by 1, accounting for the 'paint' action.
            ii. Check if the robot currently holds the `goal_color`. To do this, iterate through the state and find a predicate `(robot-has robot color)`.
            iii. If the robot does not have the `goal_color`, increment the heuristic value by 1, accounting for the 'change_color' action.
            iv. Increment the heuristic value by 1, accounting for a 'move' action to position the robot to paint the tile.
    4. Return the total accumulated heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        goal_paint_conditions = []
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                goal_paint_conditions.append(goal)

        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        for goal_paint in goal_paint_conditions:
            if goal_paint not in state:
                heuristic_value += 1  # For paint action
                goal_tile = get_parts(goal_paint)[1]
                goal_color = get_parts(goal_paint)[2]

                if robot_color != goal_color:
                    color_available = False
                    for fact in state:
                        if match(fact, "available-color", goal_color):
                            color_available = True
                            break
                    if color_available:
                        heuristic_value += 1 # For change_color action

                heuristic_value += 1  # For move action (approximation)

        return heuristic_value
