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
    This heuristic estimates the number of actions required to reach the goal state
    by counting the number of unsatisfied goal predicates and adding a penalty
    for color changes if the robot is holding an incorrect color to paint a tile.
    It focuses on achieving the painting goals and changing colors, but it does
    not explicitly consider the movement actions of the robot.

    # Assumptions:
    - The primary actions are painting tiles and changing the robot's color.
    - Movement actions are necessary but their cost is implicitly considered
      by assuming the robot can always move to a position to paint if needed.
    - The heuristic is non-admissible but aims to guide the search effectively
      by prioritizing goal achievement and color management.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task.
    - Static facts are not explicitly used in this version of the heuristic,
      although they could be incorporated for more sophisticated estimations
      (e.g., considering adjacency for movement costs).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal predicates from the task definition.
    3. Get the current state of the world.
    4. Identify the robot's current color from the state.
    5. Iterate through each goal predicate in the task's goal definition.
    6. For each goal predicate (painted tile color):
       a. Check if the goal predicate is already satisfied in the current state.
       b. If the goal is not satisfied, increment the heuristic value by 1,
          representing the estimated cost of a 'paint' action.
       c. Check if the robot currently holds the correct color required to
          achieve the current goal predicate.
       d. If the robot does not have the correct color, increment the heuristic
          value by 1, representing the estimated cost of a 'change_color' action.
          Note: This assumes that a color change is always necessary if the robot
          has the wrong color. In reality, the robot might need to change color
          multiple times in a plan, but for a greedy heuristic, counting one
          color change per unsatisfied goal requiring a different color can be
          a reasonable simplification.
    7. Return the total accumulated heuristic value.

    This heuristic is a simplified estimate and does not account for the number
    of move actions required to reach tiles or the optimal sequence of color changes.
    It is designed to be computationally efficient and provide a reasonable
    guidance for greedy best-first search in the floortile domain.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        # Get the robot's current color
        robot_color = None
        for fact in state:
            if match(fact, 'robot-has', '*', '*'):
                robot_color = get_parts(fact)[2]
                break

        for goal in self.goals:
            if goal not in state:
                heuristic_value += 1  # Assume 1 action for painting

                goal_parts = get_parts(goal)
                goal_color = goal_parts[2]

                if robot_color != goal_color:
                    heuristic_value += 1 # Assume 1 action for color change if needed

        return heuristic_value
