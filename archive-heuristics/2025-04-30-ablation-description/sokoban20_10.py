from heuristics.heuristic_base import Heuristic
# No other external modules needed for this simple heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at box1 loc_4_4)" -> ["at", "box1", "loc_4_4"]
    return fact[1:-1].split()

def parse_location(loc_str):
    """Parse a location string like 'loc_R_C' into a tuple (R, C)."""
    # Example: "loc_4_4" -> (4, 4)
    parts = loc_str.split('_')
    # Assuming format is always loc_R_C where R and C are integers
    # Add error handling for robustness if format is not guaranteed
    if len(parts) != 3 or parts[0] != 'loc':
         raise ValueError(f"Unexpected location format: {loc_str}")
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        raise ValueError(f"Could not parse row/col integers from location: {loc_str}")


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the remaining effort by summing the Manhattan
    distances between the current location of each box and its goal location.
    It ignores the robot's position and obstacles, providing a simple,
    non-admissible estimate.

    # Assumptions
    - The goal state is defined by the locations of the boxes.
    - Locations are named in a 'loc_R_C' format, implying a grid structure
      where Manhattan distance is a reasonable metric for the minimum number
      of grid steps a box needs to move.
    - The heuristic assumes a relaxed problem where boxes can move directly
      towards their goals without considering the robot's position, push mechanics,
      or obstacles. This means it is likely not admissible.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Static facts (like 'adjacent') are not explicitly used for distance calculation,
      relying instead on the grid structure implied by location names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the goal location for each box from the pre-calculated `self.goal_locations`.
    3. Iterate through the current state facts to find the current location of each box.
       - For each fact `(at ?box ?location)` in the state:
         - Get the predicate name (`parts[0]`), object name (`parts[1]`), and location name (`parts[2]`).
         - Check if the fact is an `at` predicate for an object that is a box (i.e., has a goal location defined in `self.goal_locations`).
         - If it is, get the box name (`box_name`) and its current location string (`current_loc_str`).
         - Look up the goal location string (`goal_loc_str`) for this box from `self.goal_locations`.
         - Parse the current location string (`current_loc_str`) into a (row, col) tuple using `parse_location`.
         - Parse the goal location string (`goal_loc_str`) into a (row, col) tuple using `parse_location`.
         - Calculate the Manhattan distance between the current (row, col) and the goal (row, col): `abs(current_row - goal_row) + abs(current_col - goal_col)`.
         - Add this distance to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not used for this simple heuristic.
        # static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically like (at box1 loc_2_4)
            parts = get_parts(goal)
            # Check if the goal fact is an 'at' predicate with two arguments
            if parts[0] == "at" and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                # Assuming objects in goal are boxes that need to be moved
                self.goal_locations[obj_name] = loc_name

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        by summing Manhattan distances of boxes to their goals.
        """
        state = node.state  # Current world state.

        # Find current locations of all boxes that have a goal
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check for facts like (at box1 loc_4_4)
            # Ensure it's an 'at' predicate with 3 parts and the object is one we care about (i.e., has a goal)
            if parts[0] == "at" and len(parts) == 3 and parts[1] in self.goal_locations:
                box_name, loc_name = parts[1], parts[2]
                current_box_locations[box_name] = loc_name

        total_distance = 0

        # Calculate sum of Manhattan distances for each box
        for box_name, goal_loc_str in self.goal_locations.items():
            # Ensure the box exists in the current state (it always should in a valid state)
            if box_name in current_box_locations:
                current_loc_str = current_box_locations[box_name]

                # Parse location strings into (row, col) tuples
                try:
                    current_row, current_col = parse_location(current_loc_str)
                    goal_row, goal_col = parse_location(goal_loc_str)

                    # Calculate Manhattan distance
                    distance = abs(current_row - goal_row) + abs(current_col - goal_col)
                    total_distance += distance
                except ValueError:
                    # If location format is unexpected, this state might be invalid
                    # or the heuristic cannot be computed reliably.
                    # Returning infinity indicates a potentially unreachable or problematic state
                    # from the heuristic's perspective.
                    return float('inf')
            else:
                 # This case should ideally not happen in a valid state representation
                 # where all objects from the problem definition are present.
                 # If a box is "missing", it implies the state is invalid or the
                 # heuristic logic needs adjustment (e.g., if objects can be created/destroyed).
                 # For standard PDDL, objects persist.
                 # If it happens, it means a box with a goal isn't located anywhere.
                 # This state is likely unreachable or invalid.
                 return float('inf')


        # The heuristic value is the sum of distances.
        # It is 0 if and only if all boxes are at their goal locations.
        return total_distance

