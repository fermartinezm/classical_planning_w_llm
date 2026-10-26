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
    This heuristic estimates the number of actions required to reach the goal state by counting the number of tiles that are not yet painted with the desired color according to the goal specification. It assumes that each unpainted tile in the goal requires at least one paint action to be performed.

    # Assumptions
    - The heuristic assumes that for each tile that needs to be painted according to the goal, a painting action is necessary.
    - It does not explicitly consider the cost of moving the robot or changing colors.
    - It is a simple count of unsatisfied goal conditions related to tile painting.

    # Heuristic Initialization
    - The heuristic initialization parses the goal conditions to identify all the tiles that need to be painted and their target colors.
    - It stores these goal painting requirements for efficient access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Goal Painting Conditions:
       - Iterate through the goal predicates of the task.
       - Identify the goal predicates of the form `(painted ?tile ?color)`.
       - Store each goal tile and its required color.

    2. Evaluate Current State:
       - For a given state, iterate through the stored goal painting conditions (tiles and colors).
       - For each goal painting condition `(painted tile_x color_c)`:
         - Check if the fact `(painted tile_x color_c)` is present in the current state.
         - If the fact is NOT present in the current state, it means this goal condition is not yet satisfied.

    3. Count Unsatisfied Goal Conditions:
       - Initialize a counter to 0.
       - For every goal painting condition that is not satisfied in the current state (as determined in step 2), increment the counter by 1.

    4. Return Heuristic Value:
       - The final heuristic value is the total count of unsatisfied goal painting conditions. This value represents a lower bound on the number of paint actions needed to achieve the goal, ignoring movement and color change costs.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic.
        Extracts goal painting conditions from the task definition.
        """
        self.goal_paintings = []
        for goal in task.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_paintings.append((tile, color))

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        Counts the number of goal painting conditions that are not satisfied in the current state.
        """
        state = node.state
        unsatisfied_goals = 0
        for tile, color in self.goal_paintings:
            goal_fact = f'(painted {tile} {color})'
            if goal_fact not in state:
                unsatisfied_goals += 1
        return unsatisfied_goals
