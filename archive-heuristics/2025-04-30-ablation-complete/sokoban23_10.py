# from heuristics.heuristic_base import Heuristic # Assuming this is available

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: '(at box1 loc_4_4)' -> ['at', 'box1', 'loc_4_4']
    # Example: '(at-robot loc_6_4)' -> ['at-robot', 'loc_6_4']
    # Example: '(clear loc_2_4)' -> ['clear', 'loc_2_4']
    return fact[1:-1].split()

# Helper function to parse location string into coordinates
def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    # Assumes location format is always 'loc_R_C' where R and C are integers.
    parts = loc_str.split('_')
    # Basic check for expected format
    if len(parts) == 3 and parts[0] == 'loc' and parts[1].isdigit() and parts[2].isdigit():
        return (int(parts[1]), int(parts[2]))
    else:
        # Fallback for unexpected formats, though not ideal.
        # Returning (0,0) or raising an error are options.
        # Let's return (0,0) as a non-crashing fallback, assuming valid inputs generally.
        # print(f"Warning: Unexpected location format: {loc_str}") # Optional debug print
        return (0, 0) # Default coordinates

# Helper function to calculate Manhattan distance between two location strings
def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two locations given as strings."""
    r1, c1 = parse_location(loc1_str)
    r2, c2 = parse_location(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)


# Inherit from Heuristic in the actual environment
# class sokobanHeuristic(Heuristic):
class sokobanHeuristic: # Replace with 'class sokobanHeuristic(Heuristic):' in the target environment
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances for each box to its goal location and the Manhattan distance
    for the robot to reach each box that needs to be moved.

    # Assumptions
    - Locations are named in the format 'loc_row_col' allowing Manhattan distance calculation.
    - The cost of moving a box one step is 1 (via a push action).
    - The cost of moving the robot one step is 1 (via a move action or as part of a push).
    - The heuristic assumes that the robot needs to reach the box's current location
      before it can start pushing it towards the goal. This is a simplification,
      as the robot needs to be in a specific adjacent cell to push.
    - The goal only specifies the final locations of boxes using the `(at box loc)` predicate.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Static facts (like 'adjacent' or 'clear') are available but not directly used in this
      Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Find the current location of the robot by iterating through the state facts. This is expected to be a fact like `(at-robot loc_X_Y)`.
    3. Find the current location of each box by iterating through the state facts. These are expected to be facts like `(at box_name loc_A_B)`. Store these in a dictionary mapping box name to location string.
    4. For each box that has a specified goal location (extracted during initialization from the task's goals):
       a. Get the box's current location from the dictionary created in step 3.
       b. Get the box's goal location (stored during initialization).
       c. If the box's current location is found in the state and is different from its goal location:
          i. Calculate the Manhattan distance between the box's current location
             and its goal location using the `manhattan_distance` helper function. Add this distance to the total cost. This
             represents the minimum number of push actions required for this box
             if the path were clear and the robot always in position.
          ii. Calculate the Manhattan distance between the robot's current location
              and the box's current location using the `manhattan_distance` helper function. Add this distance to the total cost.
              This is a simplified estimate of the robot movement needed to get
              into a position to push the box.
    5. Return the total heuristic cost. The heuristic value will be 0 if and only if all goal boxes are at their goal locations, assuming the goal only consists of `(at box loc)` facts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are available but not used in this simple heuristic.
        self.static = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are expected to be of the form '(at box_name loc_name)'
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
            # Assuming goal only contains 'at' predicates for boxes.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find the robot's current location and all box locations.
        robot_location = None
        current_box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                obj_name = parts[1] # This is the box name
                location = parts[2]
                current_box_locations[obj_name] = location

        # If robot location wasn't found, something is wrong with the state.
        # Assuming valid states where robot location is always present.
        # If not, returning infinity might be appropriate for search.
        if robot_location is None:
             # print("Warning: Robot location not found in state.") # Debugging
             return float('inf') # Cannot solve without robot

        total_cost = 0  # Initialize action cost counter.

        # Iterate through the boxes that have a goal location defined.
        for box, goal_location in self.goal_locations.items():
            # Get the box's current location.
            current_location = current_box_locations.get(box)

            # If the box exists in the current state and is not at its goal location:
            if current_location and current_location != goal_location:
                # Cost for moving the box itself (minimum pushes)
                box_move_cost = manhattan_distance(current_location, goal_location)
                total_cost += box_move_cost

                # Cost for the robot to reach the box
                # This is a simplification; robot needs to be *behind* the box.
                # Using distance to box location as a proxy.
                robot_reach_cost = manhattan_distance(robot_location, current_location)
                total_cost += robot_reach_cost

        # The heuristic is 0 iff all goal conditions (box locations) are met.
        # Our calculation sums costs only for boxes NOT at their goal.
        # If all goal boxes are at their goal, total_cost will be 0.
        # This matches the requirement that h=0 iff goal is reached (assuming goal
        # only involves box locations).

        return total_cost
