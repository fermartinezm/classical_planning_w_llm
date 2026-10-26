from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of tiles that are not yet painted according to the goal specification.
    It counts the number of goal predicates `(painted tile color)` that are not satisfied in the current state.
    This heuristic assumes that each unsatisfied goal predicate requires at least one action to be achieved.

    # Assumptions:
    - Each unsatisfied goal predicate `(painted tile color)` requires at least one action (either paint_up, paint_down, change_color, or move actions to reach the tile).
    - The heuristic does not explicitly consider the cost of moving the robot or changing colors. It focuses on the number of tiles that still need to be painted correctly.
    - It is a simple admissible heuristic in the relaxed problem where move and color change actions are free, and only paint actions have cost 1. In the original problem, it is not necessarily admissible but should be informative for greedy search.

    # Heuristic Initialization
    - Extract the goal `painted` predicates from the task definition during initialization.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter `unsatisfied_goals` to 0.
    2. Iterate through each goal predicate in the task's goal description.
    3. For each goal predicate, check if it is of the form `(painted ?tile ?color)`.
    4. If it is a `painted` predicate, check if this predicate is present in the current state.
    5. If the `painted` predicate is NOT present in the current state, increment `unsatisfied_goals` by 1.
    6. After checking all goal predicates, the value of `unsatisfied_goals` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal painted predicates.
        """
        self.goal_painted_tiles = []
        for goal in task.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_painted_tiles.append((tile, color))

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        The heuristic value is the number of goal painted predicates that are not satisfied in the state.
        """
        state = node.state
        unsatisfied_goals = 0
        for tile, color in self.goal_painted_tiles:
            goal_fact = f'(painted {tile} {color})'
            if goal_fact not in state:
                unsatisfied_goals += 1
        return unsatisfied_goals

    def _get_parts(self, fact):
        """
        Extract the components of a PDDL fact by removing parentheses and splitting the string.
        """
        return fact[1:-1].split()
