# Required import for the base class
# Assuming heuristics.heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Check for empty string or non-string input
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    content = fact[1:-1].strip()
    if not content:
        return []
    return content.split()

def parse_location(location_str):
    """Parses 'loc_R_C' string into (R, C) tuple."""
    # Example: 'loc_3_5' -> ('loc', '3', '5')
    parts = location_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            # Convert row and column parts to integers
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where R or C are not valid integers
            return None # Indicate parsing failure
    else:
        # Handle cases where the string format is not 'loc_R_C'
        return None # Indicate parsing failure


def manhattan_distance(loc1_str, loc2_str):
    """Calculates Manhattan distance between two locations 'loc_R_C'."""
    coord1 = parse_location(loc1_str)
    coord2 = parse_location(loc2_str)

    # If parsing failed for either location, we cannot compute distance.
    # This might indicate an issue with the input state/goal format.
    # Returning 0 means this box contributes nothing to the heuristic,
    # which is safe but might hide problems.
    if coord1 is None or coord2 is None:
        return 0

    # Calculate Manhattan distance
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to reach a goal state
    by summing the Manhattan distances of each misplaced box to its goal location.
    It serves as a simple and efficiently computable estimate for greedy best-first search.

    # Assumptions:
    - Locations are named in the format 'loc_R_C' where R and C are integers representing row and column.
    - Each box that needs to be moved has a unique goal location specified in the task goals. The mapping is assumed to be based on the box object name (e.g., box1 goes to the goal location specified for box1).
    - The heuristic ignores the robot's position and the constraints imposed by other boxes or walls (except implicitly through the goal location).
    - The cost of moving a box is related to its distance to the goal.

    # Heuristic Initialization
    - Extract the goal location for each box from the task's goal conditions. Store this mapping (box name -> goal location string).
    - Static facts (like 'adjacent') are not directly used in this simple distance calculation, but they define the grid structure assumed by Manhattan distance.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the task definition (`task.goals`). Store this mapping (box name -> goal location string) in `self.box_goals`. This is done during initialization (`__init__`).
    2. For a given state (`node.state`), identify the current location of each box that is present in `self.box_goals`. Store this mapping (box name -> current location string) temporarily.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each box name and its corresponding goal location stored in `self.box_goals`.
       - Get the box's current location from the temporary mapping created in step 2.
       - If the box's current location is the same as its goal location, the box is in place; add 0 to the total cost for this box.
       - If the box's current location is different from its goal location:
         - Parse the row and column numbers from both the current and goal location strings using the `parse_location` helper function (assuming 'loc_R_C' format).
         - Calculate the Manhattan distance between the current location coordinates and the goal location coordinates using the `manhattan_distance` helper function.
         - Add this Manhattan distance to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal locations for boxes."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not used in this simple heuristic.
        # static_facts = task.static

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at ?box ?location)
            parts = get_parts(goal)
            # Check if the fact is an 'at' predicate with two arguments
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 # In Sokoban, 'at' goals usually refer to boxes.
                 # We assume any object in an 'at' goal is a box we need to move.
                 self.box_goals[obj_name] = loc_name
            # Note: This heuristic assumes a 1-to-1 mapping from box name to goal location.
            # If a box appears in multiple goal facts, the last one processed will be used.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Track current locations of objects that are potentially boxes with goals.
        box_current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate with two arguments
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 # Only track locations for objects that we know are boxes with goals
                 if obj_name in self.box_goals:
                     box_current_locations[obj_name] = loc_name
            # We don't need robot location or clear predicates for this simple heuristic.

        total_cost = 0  # Initialize heuristic cost.

        # Calculate cost for each box that has a goal
        for box_name, goal_location in self.box_goals.items():
            # Get the current location of the box from the state.
            # Use .get() with a default value (like None) in case a box
            # listed in goals is somehow not found in the state (unlikely in valid states).
            current_location = box_current_locations.get(box_name)

            # If the box's current location is known and is different from its goal location
            if current_location is not None and current_location != goal_location:
                # Calculate Manhattan distance between current and goal locations
                dist = manhattan_distance(current_location, goal_location)
                total_cost += dist
            # If current_location is None, the box is missing from the state,
            # which shouldn't happen in a valid problem instance.
            # If current_location == goal_location, distance is 0, adds nothing to total_cost.

        return total_cost
