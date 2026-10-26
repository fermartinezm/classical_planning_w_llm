from heuristics.heuristic_base import Heuristic
# No need for fnmatch if we parse facts directly

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input gracefully, though in a planner context,
        # facts from the state/goals should be well-formed strings.
        # print(f"Warning: Unexpected fact format: {fact}")
        return []
    return fact[1:-1].split()

# Helper function to parse location strings
def parse_location(location_str):
    """Parses a location string like 'loc_r_c' into (row, col) integers."""
    parts = location_str.split('_')
    # Expecting format like 'loc_1_1'
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle unexpected format or non-integer parts
            # print(f"Warning: Could not parse location string {location_str}")
            return None # Return None or raise an error
    else:
        # print(f"Warning: Unexpected location string format: {location_str}")
        return None # Return None or raise an error

# Helper function to calculate Manhattan distance
def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two location strings."""
    coords1 = parse_location(loc1_str)
    coords2 = parse_location(loc2_str)

    if coords1 is None or coords2 is None:
        # Cannot calculate distance if parsing failed.
        # This might indicate an invalid state or unparseable location.
        # A large value pushes this state down in a greedy search.
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances for each box from its current location to its goal location.
    It serves as a simple, non-admissible estimate that prioritizes states
    where boxes are closer to their final destinations. It ignores the robot's
    position and the complexity introduced by obstacles and the push mechanic.

    # Assumptions
    - Location names follow the format 'loc_r_c' where 'r' and 'c' are integers
      representing row and column coordinates on a grid.
    - The grid structure implied by location names is used for distance calculation,
      ignoring the actual adjacency graph and potential obstacles (walls, other boxes).
    - The robot's position and the cost/feasibility of moving the robot to push
      boxes are not explicitly considered in the distance calculation.
    - Each unit of Manhattan distance for a box is assumed to correspond roughly
      to one 'push' action, ignoring the robot's movement cost and the need
      for the robot to maneuver into position.

    # Heuristic Initialization
    - The constructor extracts and stores the goal location for each box
      from the task's goal conditions.
    - Static facts (like 'adjacent' predicates) are not used by this heuristic
      as it relies on the grid coordinates implied by location names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box based on the task's goal conditions.
       This is done once during the heuristic's initialization (`__init__`).
    2. For the given state (provided to the `__call__` method), identify the
       current location of each box.
    3. Initialize a total distance counter to 0.
    4. Iterate through the boxes that have a specified goal location:
       - Retrieve the box's current location from the state information.
       - Retrieve the box's goal location (stored during initialization).
       - If the box is not currently at its goal location:
         - Parse the current and goal location strings (e.g., 'loc_r_c') into
           (row, column) integer coordinates.
         - Calculate the Manhattan distance between the current and goal coordinates:
           `abs(current_row - goal_row) + abs(current_col - goal_col)`.
         - Add this Manhattan distance to the total distance counter.
    5. Return the total distance as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        # Store goals for potential future use or checks (e.g., goal_reached check)
        self.goals = task.goals
        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at boxX loc_Y_Z)
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assume any 'at' goal is for a box we need to move.
                # A more robust check could verify the object type.
                self.goal_locations[obj] = loc

        # Static facts (adjacent predicates) are not used by this heuristic
        # as it relies on the grid coordinates implied by location names.
        # static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
            Returns float('inf') if any required location string cannot be parsed.
        """
        state = node.state

        # Find current locations of all objects (specifically boxes we care about)
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # Only store location if the object is one of the boxes we have a goal for
                 if obj in self.goal_locations:
                     current_locations[obj] = loc

        total_distance = 0

        # Sum Manhattan distances for boxes not at their goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_locations.get(box) # Get current location of the box

            # If the box is not found in the current state, or its location
            # cannot be determined, this state might be problematic.
            # Or, more likely, the box is not at its goal.
            # If current_loc is None, it means the 'at' fact for this box
            # was not found in the state. This shouldn't happen in a valid Sokoban state.
            # We proceed assuming current_loc is found.

            # Check if the box is not already at its goal
            if current_loc != goal_loc:
                distance = manhattan_distance(current_loc, goal_loc)
                # If distance is infinity (parsing failed), this state might be invalid or a dead end.
                # Adding infinity will make the heuristic large, which is appropriate.
                if distance == float('inf'):
                    return float('inf') # Propagate infinity immediately
                total_distance += distance

        return total_distance
