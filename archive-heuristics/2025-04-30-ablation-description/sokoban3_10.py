from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully, though PDDL facts are usually well-formed.
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to parse location string into row and column
def parse_location(location_str):
    """Parses a location string like 'loc_row_col' into (row, col) integers."""
    parts = location_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers, though unlikely in standard Sokoban PDDL
            return None
    return None # Return None for unexpected formats

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the minimum number of grid steps required to move each box
    from its current location to its goal location, calculated as the sum of Manhattan
    distances for all boxes. It ignores the robot's position and obstacles.

    # Assumptions
    - The locations are arranged in a grid and named using the pattern 'loc_row_col'.
    - The cost of moving a box is related to the grid distance it needs to travel.
    - The heuristic is the sum of individual box-to-goal distances.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Parses the location strings into row and column integers for distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Iterate through the current state facts to find the current location of each box.
       - A fact like `(at ?box ?location)` indicates the current position of `?box`.
    3. For each box found in the current state:
       - Retrieve its current location string.
       - Look up its corresponding goal location string (stored during initialization).
       - Parse both the current and goal location strings into (row, col) integer pairs.
       - Calculate the Manhattan distance between the current (row, col) and goal (row, col): `abs(current_row - goal_row) + abs(current_col - goal_col)`.
       - Add this distance to the total heuristic cost.
    4. The final total heuristic cost is the sum of Manhattan distances for all boxes.
    5. If all boxes are already at their goal locations, their individual distances will be 0, resulting in a total heuristic of 0, which correctly indicates a goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not strictly needed for this simple Manhattan distance heuristic.
        # static_facts = task.static

        # Store goal locations for each box, parsed into (row, col) tuples.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are expected to be of the form (at ?box ?location)
            if parts and parts[0] == "at" and len(parts) == 3:
                box = parts[1]
                location_str = parts[2]
                parsed_location = parse_location(location_str)
                if parsed_location is not None:
                    self.goal_locations[box] = parsed_location
                # else: Log a warning or handle malformed goal fact if necessary

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        based on the sum of Manhattan distances of boxes to their goals.
        """
        state = node.state  # Current world state.

        total_distance = 0  # Initialize the heuristic cost.

        # Find current locations of all boxes in the state
        # We only care about objects that are boxes and have a goal location
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Look for facts of the form (at ?obj ?location)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj = parts[1]
                location_str = parts[2]
                # Check if this object is one of the boxes we care about (i.e., has a goal)
                if obj in self.goal_locations:
                    parsed_location = parse_location(location_str)
                    if parsed_location is not None:
                        current_box_locations[obj] = parsed_location
                    # else: Log a warning or handle malformed location fact if necessary

        # Calculate total Manhattan distance for all boxes whose current location was found
        # We iterate through the boxes we found in the current state
        for box, current_loc_parsed in current_box_locations.items():
             # We already know this box is in self.goal_locations from the loop above
             goal_loc_parsed = self.goal_locations[box]

             # Calculate Manhattan distance
             distance = abs(current_loc_parsed[0] - goal_loc_parsed[0]) + \
                        abs(current_loc_parsed[1] - goal_loc_parsed[1])
             total_distance += distance

        # Note: If a box is in the goal but not in the current state (e.g., carried by robot in gripper domain,
        # but not applicable in Sokoban as boxes are always 'at' a location), it wouldn't be counted.
        # In Sokoban, boxes are always at a location, so current_box_locations should contain all boxes from goals.

        return total_distance
