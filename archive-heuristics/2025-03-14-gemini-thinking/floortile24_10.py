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
    This heuristic estimates the number of actions required to reach the goal state in the floortile domain.
    It counts the number of tiles that are not painted according to the goal specification and adds a penalty
    if the robot needs to change color to paint any of the unpainted tiles.

    # Assumptions
    - The heuristic assumes that for each unpainted tile in the goal, a 'paint' action is needed.
    - It also assumes at most one 'change_color' action is needed if the robot initially does not have a color required for any of the unpainted tiles.
    - Movement actions are not explicitly considered in this heuristic, assuming the robot can reach any tile needed for painting.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal conditions that are of the form `(painted tile color)`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal conditions related to painting tiles, i.e., predicates of the form `(painted tile color)`.
    3. For each goal condition `(painted goal_tile goal_color)`:
        a. Check if this condition is already satisfied in the current state.
        b. If the condition is not satisfied, increment the heuristic value by 1 (for a 'paint' action).
        c. Keep track of the `goal_color` required for each unsatisfied goal.
    4. After checking all goal conditions, determine the color the robot currently has.
    5. Check if there is any unsatisfied goal condition for which the required `goal_color` is different from the robot's current color.
    6. If such a mismatch exists for at least one unsatisfied goal, increment the heuristic value by 1 (for a 'change_color' action). This is a simplification and might underestimate the actual color changes needed.
    7. Return the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.painted_goals = []
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                self.painted_goals.append(goal)

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0
        unsatisfied_goals_colors = set()

        for goal in self.painted_goals:
            if goal not in state:
                heuristic_value += 1
                unsatisfied_goals_colors.add(get_parts(goal)[2]) # Extract the color from the goal

        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        needs_color_change = False
        if robot_color is not None:
            for goal_color in unsatisfied_goals_colors:
                if robot_color != goal_color:
                    needs_color_change = True
                    break
        elif unsatisfied_goals_colors: # If robot has no color and there are unpainted goals
            needs_color_change = True

        if needs_color_change and unsatisfied_goals_colors:
            heuristic_value += 1

        return heuristic_value
