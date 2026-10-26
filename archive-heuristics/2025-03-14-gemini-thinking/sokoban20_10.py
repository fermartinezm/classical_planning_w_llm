from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to solve Sokoban problems by calculating the sum of Manhattan distances
    for each box from its current location to its goal location. It assumes that each push action moves a box one step closer to its goal,
    ignoring obstacles and potential deadlocks.

    # Assumptions:
    - The heuristic assumes direct paths are always possible and does not account for walls or other boxes blocking the way.
    - It considers only the box positions and goal positions, ignoring the robot's position and the 'clear' predicates in the cost estimation.
    - It is a simple Manhattan distance heuristic and is not admissible, but it is efficiently computable and aims to guide the search effectively.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal 'at' predicates from the task goals to determine the goal location for each box. Store these in a dictionary mapping box names to goal locations.
    3. For the given state, iterate through all 'at' predicates that refer to boxes to find the current location of each box.
    4. For each box, retrieve its goal location from the pre-computed goal locations.
    5. Parse the location names to extract row and column numbers (assuming location names are in the format 'loc_row_column').
       If location names do not follow this format, a different parsing method would be needed based on the specific location naming convention.
    6. Calculate the Manhattan distance between the current location and the goal location for each box.
       Manhattan distance is calculated as the absolute difference in row numbers plus the absolute difference in column numbers.
    7. Sum up the Manhattan distances for all boxes.
    8. Return the total sum as the heuristic value. This value represents the estimated number of moves required to reach the goal state.
       A heuristic value of 0 indicates that all boxes are in their goal locations, which means the goal state is reached (or very close according to this heuristic).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations."""
        self.goals = task.goals
        self.goal_box_locations = {}

        for goal_fact in self.goals:
            if match(goal_fact, "at", "*", "*"):
                parts = get_parts(goal_fact)
                box_name = parts[1]
                goal_location = parts[2]
                self.goal_box_locations[box_name] = goal_location

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        box_current_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] not in ['robot', '-robot']: # To differentiate from at-robot predicate if present in state
                    box_name = parts[1]
                    current_location = parts[2]
                    box_current_locations[box_name] = current_location

        for box_name, goal_location in self.goal_box_locations.items():
            if box_name in box_current_locations:
                current_location = box_current_locations[box_name]

                try:
                    current_loc_parts = current_location.split('_')
                    goal_loc_parts = goal_location.split('_')

                    current_row = int(current_loc_parts[1])
                    current_col = int(current_loc_parts[2])
                    goal_row = int(goal_loc_parts[1])
                    goal_col = int(goal_loc_parts[2])

                    manhattan_distance = abs(current_row - goal_row) + abs(current_col - goal_col)
                    heuristic_value += manhattan_distance
                except: # Handle cases where location names are not in loc_r_c format, default to 0 distance if parsing fails.
                    heuristic_value += 0 # Or handle error as needed, e.g., print warning.

        return heuristic_value
