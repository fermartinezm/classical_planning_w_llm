from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_2_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the total Manhattan distance of all boxes to their goal locations.
    It assumes that each box needs to be moved independently to its goal location, ignoring
    potential blockages by other boxes or the robot, and without considering the need for clear locations
    for pushing. It serves as a simple and fast underestimation of the actual cost.

    # Assumptions:
    - The cost of moving a box is proportional to the Manhattan distance to its goal.
    - Boxes can be moved independently of each other.
    - The robot can always reach a position to push a box.
    - Clear locations required for moves and pushes are not explicitly considered in the heuristic.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Parses the static facts to build a location map for easier access to location coordinates
      if location names follow a consistent naming convention (e.g., loc_row_col).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal 'at' predicates to determine the goal location for each box. Store these in a dictionary mapping box names to goal locations.
    3. For each box in the current state:
        a. Get the current location of the box from the state using 'at' predicate.
        b. Retrieve the goal location for this box from the pre-computed goal locations dictionary.
        c. If both current and goal locations are found and follow the 'loc_row_col' naming convention:
            i. Parse the row and column numbers from both location names.
            ii. Calculate the Manhattan distance between the current and goal locations:
                `abs(goal_row - current_row) + abs(goal_col - current_col)`
            iii. Add this Manhattan distance to the total heuristic value.
    4. Return the total accumulated Manhattan distance as the heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations and parsing location information."""
        self.goals = task.goals
        static_facts = task.static

        self.goal_box_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box_name = parts[1]
                location_name = parts[2]
                self.goal_box_locations[box_name] = location_name

    def __call__(self, node):
        """Calculate the Manhattan distance heuristic for the given state."""
        state = node.state
        heuristic_value = 0

        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box_name = parts[1]
                location_name = parts[2]
                box_locations[box_name] = location_name

        for box_name, current_location_name in box_locations.items():
            if box_name in self.goal_box_locations:
                goal_location_name = self.goal_box_locations[box_name]

                try:
                    current_location_parts = current_location_name.split('_')
                    goal_location_parts = goal_location_name.split('_')

                    if len(current_location_parts) == 3 and len(goal_location_parts) == 3:
                        current_row = int(current_location_parts[1])
                        current_col = int(current_location_parts[2])
                        goal_row = int(goal_location_parts[1])
                        goal_col = int(goal_location_parts[2])

                        manhattan_distance = abs(goal_row - current_row) + abs(goal_col - current_col)
                        heuristic_value += manhattan_distance
                except ValueError:
                    # Handle cases where location names are not in the expected format, or parsing fails.
                    # In this simple heuristic, we can ignore these cases or assign a large penalty if needed.
                    pass # Or consider adding a penalty if location parsing fails, if appropriate.


        return heuristic_value
