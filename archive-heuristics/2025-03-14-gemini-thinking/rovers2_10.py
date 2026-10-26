from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions in the Rovers domain.
    It focuses on the main tasks: collecting samples (soil and rock), taking images, and communicating data.
    The heuristic is a simplified estimate and does not guarantee admissibility, but aims for efficiency and informed search guidance.

    # Assumptions:
    - Each goal predicate (communicated data) is considered independently.
    - The cost is estimated based on the remaining actions needed for each goal, assuming a simplified action sequence.
    - Navigation costs are implicitly considered as part of the action costs for sampling, imaging, and communicating.
    - Rovers are assumed to have the necessary equipment if the corresponding action is part of a goal achievement.

    # Heuristic Initialization:
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are explicitly preprocessed in this simplified version for efficiency.

    # Step-By-Step Thinking for Computing Heuristic:
    For each goal predicate in the goal state:
    1. Check if the goal is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If the goal is not achieved, estimate the minimum number of actions required to achieve it based on the goal type:
        - For `communicated_soil_data(waypoint)`: Assume 2 actions are needed: `sample_soil` and `communicate_soil_data`.
        - For `communicated_rock_data(waypoint)`: Assume 2 actions are needed: `sample_rock` and `communicate_rock_data`.
        - For `communicated_image_data(objective, mode)`: Assume 3 actions are needed: `calibrate`, `take_image`, and `communicate_image_data`.
    3. Sum up the estimated action counts for all unachieved goal predicates to get the total heuristic value.
    This simplified heuristic prioritizes achieving each goal with a fixed cost estimate, without detailed planning or pathfinding, to ensure computational efficiency.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        for goal in self.goals:
            if goal not in state:
                goal_parts = get_parts(goal)
                predicate = goal_parts[0]

                if predicate == 'communicated_soil_data':
                    heuristic_value += 2  # Estimate for sample_soil + communicate_soil_data
                elif predicate == 'communicated_rock_data':
                    heuristic_value += 2  # Estimate for sample_rock + communicate_rock_data
                elif predicate == 'communicated_image_data':
                    heuristic_value += 3  # Estimate for calibrate + take_image + communicate_image_data

        return heuristic_value
