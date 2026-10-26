# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
# If running standalone, you might need to define a dummy base class or remove inheritance.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic must implement __call__")
    # print("Warning: heuristics.heuristic_base.Heuristic not found. Using dummy base class.")


# Helper function to parse location string like 'loc_R_C' into (R, C) tuple
def get_coords(location_str):
    """Parses a location string 'loc_R_C' into a tuple (R, C)."""
    parts = location_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where R or C are not integers, though unlikely in PDDL examples
            return None
    return None # Invalid format

# Helper function to calculate Manhattan distance between two location strings
def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two locations."""
    coords1 = get_coords(loc1_str)
    coords2 = get_coords(loc2_str)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if coordinates are invalid
        # This might indicate a problem with the PDDL instance or parsing
        # Returning infinity or a very large number is appropriate for a heuristic
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

# Helper function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances between each box's current location and its goal location. It
    ignores the robot's position and the presence of obstacles, making it
    admissible but potentially weak.

    # Assumptions
    - Locations are represented as 'loc_R_C' strings, allowing extraction of
      grid coordinates (Row, Column).
    - The goal state specifies the target location for each box using the
      '(at box location)' predicate.
    - The cost of moving a box one step towards its goal (via a push action)
      is at least 1.
    - The grid structure implied by 'loc_R_C' and 'adjacent' predicates
      corresponds to a grid where Manhattan distance is meaningful.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Stores these box-goal mappings in a dictionary `self.box_goals`.
    - Static facts (like 'adjacent') are not directly used in this simple
      Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. For each box that has a specified goal location (identified during initialization):
       - Find the box's current location in the current state by looking for the fact `(at box_name current_location)`.
       - If the box is already at its goal location, its contribution to the
         heuristic is 0.
       - If the box is not at its goal location:
         - Parse the current location string ('loc_R_C') into (Row, Column) coordinates using `get_coords`.
         - Parse the goal location string ('loc_R_C') into (Row, Column) coordinates using `get_coords`.
         - Calculate the Manhattan distance between the current and goal coordinates using `manhattan_distance`.
         - Add this distance to the total heuristic cost.
    3. Return the total heuristic cost. This sum represents the minimum number
       of pushes required across all boxes, ignoring robot movement and obstacles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        # self.goals = task.goals  # Goal conditions are used implicitly by checking against box_goals
        # static_facts = task.static # Static facts are not used in this simple heuristic

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in task.goals:
            # Goal facts are typically (at box_name location_name)
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "at":
                 # We assume any object in an 'at' goal predicate is a box for this domain
                 obj, location = parts[1], parts[2]
                 self.box_goals[obj] = location


    def __call__(self, node):
        """
        Compute the sum of Manhattan distances from each box's current location
        to its goal location.
        """
        state = node.state  # Current world state.

        # Find the current location of each box that has a goal.
        current_box_locations = {}
        # Build a quick lookup for 'at' facts for objects that are goals
        state_at_facts = {parts[1]: parts[2] for fact in state if (parts := get_parts(fact)) and len(parts) == 3 and parts[0] == "at"}


        total_distance = 0

        # Calculate the sum of Manhattan distances for each box to its goal.
        for box, goal_location in self.box_goals.items():
            current_location = state_at_facts.get(box)

            if current_location is None:
                 # This box is not 'at' any location in the state.
                 # This indicates an unexpected state representation or an error.
                 # Assign a large penalty as this state is likely invalid or far from goal.
                 # print(f"Warning: Box {box} location not found in state.")
                 return float('inf') # Cannot reach goal if box location is unknown

            # If the box is not at its goal, add the distance
            if current_location != goal_location:
                dist = manhattan_distance(current_location, goal_location)
                if dist == float('inf'):
                     # Cannot calculate distance, likely invalid location format
                     # This state is problematic
                     return float('inf')
                total_distance += dist

        # The heuristic is the sum of distances. If all boxes are at goals, total_distance is 0.
        return total_distance
