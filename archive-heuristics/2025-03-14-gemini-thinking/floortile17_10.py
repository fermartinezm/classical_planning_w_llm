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
    It counts the number of tiles that are not painted according to the goal and estimates the cost to paint them,
    considering color changes and painting actions. It's a simplification and does not consider movement actions explicitly.

    # Assumptions:
    - The heuristic assumes that for each tile that needs to be painted, the robot can reach a position to paint it in a single move (or moves are already accounted for loosely).
    - It primarily focuses on the painting and color changing actions required.
    - It does not explicitly account for the `clear` predicates or path planning.

    # Heuristic Initialization
    - Extracts the goal predicates of the form `(painted tile color)` to determine which tiles need to be painted with which color.
    - No static facts are explicitly used in this simple heuristic, although adjacency information could be incorporated in a more advanced version.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Identify the goal `painted` predicates.
    3. For each goal `(painted goal_tile goal_color)`:
        a. Check if the current state already satisfies this goal predicate. If yes, no cost is added for this goal.
        b. If the goal is not satisfied, increment the heuristic cost by 1, representing the cost of a `paint_up`, `paint_down`, etc. action.
        c. Check the color the robot is currently holding.
        d. If the robot does not have the `goal_color`, increment the heuristic cost by 1, representing the cost of a `change_color` action.
           (This assumes that changing color is always possible if needed and an available color exists).
    4. Return the total accumulated heuristic cost.

    This heuristic is a simplification and may underestimate the actual cost, as it does not explicitly consider movement actions or the `clear` predicate constraints.
    However, it is efficiently computable and provides a domain-dependent estimate focusing on painting and color management.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic. Extracts goal painted predicates.
        """
        self.goal_painted_tiles = {}
        for goal in task.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_painted_tiles[tile] = color

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        # Get the color the robot is currently holding
        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break # Assuming only one robot and one color at a time

        for goal_tile, goal_color in self.goal_painted_tiles.items():
            goal_fact = f'(painted {goal_tile} {goal_color})'
            if goal_fact not in state:
                heuristic_value += 1 # Cost for painting action

                if robot_color != goal_color:
                    heuristic_value += 1 # Cost for changing color (if needed)
                    robot_color = goal_color # Assume color is changed for subsequent tiles in heuristic calculation

        return heuristic_value
