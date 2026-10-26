from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances of each box to its goal location and adding the Manhattan distance
    from the robot to the nearest box that is not yet at its goal.

    # Assumptions
    - Locations are named using the format 'loc_row_col', allowing calculation
      of Manhattan distances.
    - Each box has a specific goal location defined in the task goals.
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Does not explicitly process static facts like 'adjacent' as the Manhattan
      distance calculation relies only on the location naming convention.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state facts.
    2. Identify the current location of each box that has a goal location from the state facts.
    3. Initialize the total heuristic cost to 0.
    4. Create a list of boxes that are not currently at their goal locations.
    5. For each box in this list:
       - Calculate the Manhattan distance between its current location and its goal location using the 'loc_row_col' parsing.
       - Add this distance to the total heuristic cost.
    6. If the list of boxes not at their goals is empty, the state is a goal state,
       and the heuristic is 0. Return 0.
    7. If there are boxes not at their goals:
       - If the robot's location was found:
         - Find the minimum Manhattan distance from the robot's current location
           to the location of any box in the list of boxes not at their goals.
         - Add this minimum distance to the total heuristic cost.
       - If the robot's location was not found (should not happen in valid states),
         the robot distance component is effectively skipped.
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for boxes.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.goal_locations = {} # {box_name: goal_location_str}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Static facts like 'adjacent' are not needed for this Manhattan-based heuristic.

    def parse_location(self, loc_str):
        """Parses 'loc_row_col' string into (row, col) tuple."""
        # Assumes loc_str is in the format 'loc_row_col'
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                # Handle cases where row/col are not integers
                # print(f"Warning: Could not parse row/col from location: {loc_str}")
                return (0, 0) # Return a default or indicate error
        else:
            # Handle unexpected location formats
            # print(f"Warning: Unexpected location format: {loc_str}")
            return (0, 0) # Return a default or indicate error


    def manhattan_distance(self, loc1_str, loc2_str):
        """Calculates Manhattan distance between two locations."""
        r1, c1 = self.parse_location(loc1_str)
        r2, c2 = self.parse_location(loc2_str)
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        robot_loc = None
        box_locations = {} # {box_name: location_str}

        # Extract robot and box locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and parts[1] in self.goal_locations:
                 box_locations[parts[1]] = parts[2]

        total_h = 0
        boxes_to_move = [] # List of box names that are not at their goals

        # Calculate sum of Manhattan distances for boxes not at their goals
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # Check if the box exists in the current state and is not at its goal
            if current_loc is not None and current_loc != goal_loc:
                total_h += self.manhattan_distance(current_loc, goal_loc)
                boxes_to_move.append(box)

        # If all boxes are at their goals, the heuristic is 0
        if not boxes_to_move:
            return 0

        # Add robot's distance to the nearest box that needs moving
        min_robot_dist = math.inf # Initialize with infinity
        if robot_loc: # Ensure robot location was found in the state
            for box in boxes_to_move:
                box_loc = box_locations[box]
                dist = self.manhattan_distance(robot_loc, box_loc)
                min_robot_dist = min(min_robot_dist, dist)

            # Add the minimum distance found if it's finite.
            # If robot_loc was None, min_robot_dist remains inf, and we don't add it.
            if min_robot_dist != math.inf:
                 total_h += min_robot_dist
            # else: robot_loc was None, total_h is just sum of box distances.

        # The heuristic value is total_h. It will be > 0 if boxes_to_move is not empty.
        # It can only be 0 if boxes_to_move is empty (goal state).

        return total_h
