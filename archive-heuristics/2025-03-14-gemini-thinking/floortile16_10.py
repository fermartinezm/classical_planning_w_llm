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
    This heuristic estimates the number of actions needed to achieve the goal state in the floortile domain.
    It counts the number of tiles that are not yet painted with the desired color as specified in the goal,
    and adds an additional cost of 1 if the robot needs to change its current color to paint any of the unpainted tiles.

    # Assumptions:
    - The primary actions are painting tiles and changing colors. Movement actions are implicitly considered through the count of unpainted tiles.
    - We assume that for each unpainted tile in the goal, at least one 'paint' action is required.
    - We assume that color changes are necessary only once if the robot initially holds an incorrect color for any of the remaining tiles to be painted.

    # Heuristic Initialization
    - Extracts the goal conditions related to 'painted' predicates from the task definition.
    - Identifies the color the robot initially holds.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Identify all goal facts of the form '(painted ?tile ?color)'.
    3. Identify the current 'painted' facts in the given state.
    4. Count the number of goal 'painted' facts that are not present in the current state. This count represents the minimum number of 'paint' actions needed. Add this count to the heuristic value.
    5. Determine the color the robot currently holds in the given state.
    6. Identify the set of colors required for the unsatisfied 'painted' goals (tiles that need to be painted according to the goal but are not yet painted in the current state).
    7. Check if the robot's current color is among the colors required for the unsatisfied goals.
    8. If the robot's current color is NOT among the required colors for the unsatisfied goals (and if there are any unsatisfied goals), increment the heuristic value by 1. This accounts for a potential 'change_color' action needed at some point.
    9. Return the calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and initial robot color.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.goal_painted_tiles = set()
        for goal in self.goals:
            if match(goal, 'painted', '*', '*'):
                self.goal_painted_tiles.add(goal)

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        current_painted_tiles = set()
        for fact in state:
            if match(fact, 'painted', '*', '*'):
                current_painted_tiles.add(fact)

        unsatisfied_goals = self.goal_painted_tiles - current_painted_tiles
        heuristic_value = len(unsatisfied_goals)

        robot_color = None
        for fact in state:
            if match(fact, 'robot-has', 'robot1', '*'): # Assuming robot is always robot1
                robot_color = get_parts(fact)[2]
                break

        required_colors_for_unsatisfied = set()
        for goal_fact in unsatisfied_goals:
            required_colors_for_unsatisfied.add(get_parts(goal_fact)[2])

        if unsatisfied_goals: # Only consider color change if there are still tiles to paint
            if robot_color not in required_colors_for_unsatisfied and required_colors_for_unsatisfied:
                 heuristic_value += 1

        return heuristic_value
