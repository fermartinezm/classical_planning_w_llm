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
    It primarily focuses on the painting actions and color changes needed, assuming that movement is always possible and not explicitly costing move actions.

    # Assumptions:
    - The heuristic assumes that movement actions are always possible and their cost is implicitly considered less significant compared to painting and color changing actions.
    - It prioritizes achieving the painting goals and accounts for necessary color changes.
    - It does not explicitly calculate the number of move actions needed to reach a tile to be painted.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates, specifically the `painted` goals.
    - It also extracts the available colors from the static facts to determine possible colors the robot can hold.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Identify the goal `painted` facts.
    3. For each goal `(painted tile color)` fact:
        a. Check if this fact is already true in the current state. If yes, no action is needed for this goal, continue to the next goal.
        b. If the goal fact is not true in the current state, it means the tile needs to be painted with the specified color.
        c. Increment the heuristic value by 1, representing the cost of a `paint_up` or `paint_down` action.
        d. Check if the robot currently `robot-has` the required color.
        e. If the robot does not have the required color, increment the heuristic value by 1, representing the cost of a `change_color` action.
    4. Return the total accumulated heuristic value.

    This heuristic essentially counts the minimum number of paint and color change actions needed to satisfy all painting goals, disregarding the movement costs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Specifically, store the goal predicates related to 'painted' and extract available colors.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.available_colors = set()
        for fact in self.static_facts:
            if match(fact, "available-color", "*"):
                self.available_colors.add(get_parts(fact)[1])

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        The heuristic value is an estimate of the number of actions needed to reach the goal state from the current state.
        """
        state = node.state
        heuristic_value = 0

        goal_painted_tiles = []
        for goal_fact in self.goals:
            if match(goal_fact, "painted", "*", "*"):
                goal_painted_tiles.append(goal_fact)

        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "robot*", "*"):
                robot_color = get_parts(fact)[2]
                break

        for goal_fact in goal_painted_tiles:
            if goal_fact not in state:
                heuristic_value += 1  # Cost for painting action
                goal_tile = get_parts(goal_fact)[1]
                goal_color = get_parts(goal_fact)[2]

                if robot_color != goal_color:
                    heuristic_value += 1  # Cost for change_color action


        return heuristic_value
