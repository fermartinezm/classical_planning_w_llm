from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class rovers1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of objectives that need to be imaged, soil samples to be collected, rock samples to be collected,
    and the communication of the collected data back to the lander. It prioritizes achieving all goals.

    # Assumptions
    - Each rover can perform actions independently.
    - The heuristic assumes that the rover will need to navigate to the required waypoints, sample soil/rock,
      take images, and communicate the data back to the lander.
    - It does not consider the optimal path for navigation or the optimal order of actions.

    # Heuristic Initialization
    - The heuristic initializes by extracting the goal conditions, the locations of soil and rock samples,
      the visibility relationships between waypoints, and the location of the lander.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Goal Check:** If the current state satisfies all goals, return 0.
    2. **Initialize Counts:** Initialize the counts for uncommunicated images, soil data, and rock data.
    3. **Count Uncommunicated Images:** Iterate through the goals to find `communicated_image_data` goals that are not satisfied in the current state.
       For each unsatisfied goal, increment the `uncommunicated_images` count.
    4. **Count Uncommunicated Soil Data:** Iterate through the goals to find `communicated_soil_data` goals that are not satisfied in the current state.
       For each unsatisfied goal, increment the `uncommunicated_soil_data` count.
    5. **Count Uncommunicated Rock Data:** Iterate through the goals to find `communicated_rock_data` goals that are not satisfied in the current state.
       For each unsatisfied goal, increment the `uncommunicated_rock_data` count.
    6. **Calculate Heuristic Value:** The heuristic value is the sum of the counts of uncommunicated images, soil data, and rock data.
       This represents the estimated number of actions required to achieve the goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        if self.goals <= state:
            return 0

        uncommunicated_images = 0
        uncommunicated_soil_data = 0
        uncommunicated_rock_data = 0

        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*") and goal not in state:
                uncommunicated_images += 1
            elif match(goal, "communicated_soil_data", "*") and goal not in state:
                uncommunicated_soil_data += 1
            elif match(goal, "communicated_rock_data", "*") and goal not in state:
                uncommunicated_rock_data += 1

        return uncommunicated_images + uncommunicated_soil_data + uncommunicated_rock_data
