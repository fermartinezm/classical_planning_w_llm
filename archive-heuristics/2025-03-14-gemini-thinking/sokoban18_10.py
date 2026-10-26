from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

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

def parse_location(location_str):
    """
    Parses a location string in the format 'loc_r_c' and returns a tuple (r, c).
    If the location string does not match the expected format, returns None.
    """
    match_loc = re.match(r'loc_(\d+)_(\d+)', location_str)
    if match_loc:
        return int(match_loc.group(1)), int(match_loc.group(2))
    return None

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to solve Sokoban problems by calculating the sum of Manhattan distances
    for each box from its current location to its goal location. It assumes that moving boxes closer to their goals is a good strategy.

    # Assumptions:
    - The problem is solvable.
    - The location names follow the format 'loc_r_c', representing row and column coordinates.
    - The heuristic focuses on minimizing box movements and does not explicitly consider robot movements or potential deadlocks.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Static facts (like adjacency) are not directly used in this simple Manhattan distance heuristic but could be used for more sophisticated heuristics.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal condition in the task's goal description.
    3. For each goal condition of the form '(at box_name goal_location)':
       a. Extract the box name and the goal location.
       b. Find the current location of the box in the current state.
       c. Parse both the goal location and the current location into (row, column) tuples.
       d. If parsing is successful for both locations, calculate the Manhattan distance between the current and goal locations:
          Manhattan distance = |goal_row - current_row| + |goal_col - current_col|.
       e. Add the calculated Manhattan distance to the total heuristic value.
    4. If a box's current or goal location cannot be parsed into (row, column) format, a distance of 0 is assumed for that box (or handle it as needed, e.g., return a large value if parsing fails, or assume distance 1 if locations are different abstract locations). In this simple heuristic, we assume 0 if parsing fails, effectively ignoring non-grid locations.
    5. Return the total heuristic value. This value represents the estimated number of moves required to reach the goal state.
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
        """Calculate the heuristic value for a given state based on Manhattan distance of boxes to their goal locations."""
        state = node.state
        heuristic_value = 0

        for box_name, goal_location in self.goal_box_locations.items():
            current_location = None
            for fact in state:
                if match(fact, "at", box_name, "*"):
                    current_location = get_parts(fact)[2]
                    break

            if current_location:
                goal_loc_coords = parse_location(goal_location)
                current_loc_coords = parse_location(current_location)

                if goal_loc_coords and current_loc_coords:
                    goal_row, goal_col = goal_loc_coords
                    current_row, current_col = current_loc_coords
                    manhattan_distance = abs(goal_row - current_row) + abs(goal_col - current_col)
                    heuristic_value += manhattan_distance
                # If location parsing fails, we assume distance 0 for simplicity in this heuristic.
                # A more robust heuristic might handle this differently, e.g., by returning infinity or a large constant if locations are not parseable or are abstract.

        return heuristic_value
