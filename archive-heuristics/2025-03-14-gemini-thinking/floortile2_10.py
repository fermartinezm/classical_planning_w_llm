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
    It focuses on the painting actions and color changes needed to satisfy the goal conditions,
    ignoring the move actions. This provides a lower bound estimate based on the essential painting operations.

    # Assumptions
    - The primary actions are painting tiles to match the goal specification.
    - Color changes might be necessary before painting.
    - Move actions are implicitly assumed to be necessary to reach tiles, but their cost is not explicitly calculated.
    - The heuristic is admissible if only paint and change_color actions are considered as the essential steps towards the goal. However, as move actions are ignored, it is not guaranteed to be admissible in the context of full planning.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates that are of the form `(painted tile color)`.
    - It also identifies the available colors in the domain from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the heuristic value to 0.
    2. Identify the robot's current color from the state.
    3. Iterate through each goal predicate of the form `(painted tile color)` in the task's goal definition.
    4. For each goal predicate, check if it is already satisfied in the current state.
    5. If the goal predicate `(painted tile_g color_g)` is not satisfied in the current state:
        a. Increment the heuristic value by 1. This accounts for the 'paint' action needed for this tile.
        b. Check if the robot's current color is the same as `color_g`.
        c. If the robot's current color is not `color_g`, increment the heuristic value by 1.
           This accounts for a 'change_color' action if the robot needs to switch to the correct color before painting.
    6. Return the accumulated heuristic value.

    This heuristic essentially counts the minimum number of 'paint' actions and 'change_color' actions required to satisfy all painting goals,
    without explicitly considering the 'move' actions. It provides a quick and optimistic estimate of the remaining cost.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic.
        Extract goal predicates of the form (painted tile color) and available colors from static facts.
        """
        self.goals_painted = []
        for goal in task.goals:
            if match(goal, "painted", "*", "*"):
                self.goals_painted.append(goal)
        self.available_colors = set()
        for fact in task.static:
            if match(fact, "available-color", "*"):
                self.available_colors.add(get_parts(fact)[1])


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        The heuristic value is the estimated number of actions needed to reach the goal state.
        """
        state = node.state
        heuristic_value = 0

        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        for goal_painted in self.goals_painted:
            if goal_painted not in state:
                heuristic_value += 1 # Cost for paint action
                goal_color = get_parts(goal_painted)[2]
                if robot_color != goal_color:
                    heuristic_value += 1 # Cost for change_color action

        return heuristic_value
