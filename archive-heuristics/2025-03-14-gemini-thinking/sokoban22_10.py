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
    This heuristic estimates the cost to reach the goal state in the Sokoban domain by calculating the sum of Manhattan distances
    for each box from its current location to its goal location. It assumes that each move or push action has a cost of 1.

    # Assumptions:
    - The heuristic assumes that boxes can be moved independently towards their goal locations, ignoring potential blockages or deadlocks.
    - It considers only the Manhattan distance and does not account for the number of moves required to clear a path or the robot's movements.
    - It is a lower bound on the number of moves only in very simplified scenarios and is generally not admissible for complex Sokoban problems.

    # Heuristic Initialization
    - Extracts the goal box locations from the task goals.
    - Parses location names to extract row and column coordinates for distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract goal locations for each box from the task's goal description. Store these in a dictionary where keys are box names and values are goal location names.
    3. For each box in the goal:
        a. Determine the current location of the box from the current state.
        b. Determine the goal location of the box from the pre-calculated goal box locations.
        c. Parse the row and column numbers from both the current location name and the goal location name. Location names are assumed to be in the format 'loc_row_col'.
        d. Calculate the Manhattan distance between the current location and the goal location using the formula: |row1 - row2| + |col1 - col2|.
        e. Add this Manhattan distance to the total heuristic value.
    4. Return the total heuristic value. This value represents the estimated number of actions needed to move all boxes to their goal locations based on Manhattan distances.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations."""
        self.goals = task.goals
        self.goal_box_locations = {}

        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box_name = parts[1]
                goal_location = parts[2]
                self.goal_box_locations[box_name] = goal_location

    def __call__(self, node):
        """Calculate the heuristic value for the given state."""
        state = node.state
        heuristic_value = 0

        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                box_name = parts[1]
                current_location = parts[2]
                current_box_locations[box_name] = current_location

        for box_name, goal_location in self.goal_box_locations.items():
            if box_name not in current_box_locations: # In case the box is not in the initial state but in goal. Should not happen in Sokoban.
                continue
            current_location = current_box_locations[box_name]

            try:
                current_loc_parts = current_location.split('_')
                goal_loc_parts = goal_location.split('_')

                if len(current_loc_parts) == 3 and len(goal_loc_parts) == 3:
                    current_row = int(current_loc_parts[1])
                    current_col = int(current_loc_parts[2])
                    goal_row = int(goal_loc_parts[1])
                    goal_col = int(goal_loc_parts[2])

                    manhattan_distance = abs(current_row - goal_row) + abs(current_col - goal_col)
                    heuristic_value += manhattan_distance
                else:
                    # Fallback heuristic if location names are not in expected format.
                    # This is a very basic fallback, consider improving it if needed.
                    heuristic_value += 10 # Assign a high cost if location format is unexpected

            except ValueError:
                # Handle cases where location names cannot be parsed to integers.
                # Fallback heuristic, consider improving it if needed.
                heuristic_value += 10 # Assign a high cost if parsing fails

        return heuristic_value
