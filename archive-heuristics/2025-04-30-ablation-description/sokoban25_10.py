from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise an error
        # For now, return empty list or handle defensively in calling code
        return []
    return fact[1:-1].split()

# Helper function to match a PDDL fact string against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse location string
def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a (row, col) tuple."""
    parts = loc_str.split('_')
    # Assuming format is always loc_R_C where R and C are integers
    if len(parts) != 3 or parts[0] != 'loc':
         # Handle unexpected location format
         # print(f"Warning: Unexpected location format '{loc_str}'")
         return None # Or raise an error

    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        # Handle cases where R or C are not integers
        # print(f"Warning: Non-integer row/col in location '{loc_str}'")
        return None

# Helper function for Manhattan distance
def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two location strings."""
    coords1 = parse_location(loc1_str)
    coords2 = parse_location(loc2_str)

    if coords1 is None or coords2 is None:
        # Cannot calculate distance for invalid locations
        return float('inf') # Return a large value to indicate impossibility

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan distances
    between each box's current location and its goal location. It is an admissible heuristic
    as it underestimates the true cost by ignoring robot movement and obstacles.

    # Assumptions
    - The goal state specifies the target location for each box using the `(at ?box ?location)` predicate.
    - The locations follow a naming convention like `loc_R_C` allowing extraction of grid coordinates.
    - The grid structure implied by `adjacent` predicates supports movement, and Manhattan distance
      is a valid lower bound on the number of pushes required for a box to move between two points.
    - The heuristic assumes a fixed assignment of boxes to goals as specified in the problem instance.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Stores these box-goal mappings in a dictionary `self.box_goals`.
    - Static facts (like `adjacent`) are not explicitly used in the distance calculation,
      as the location parsing relies on the grid naming convention.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value `total_heuristic` to 0.
    2. Create a dictionary `current_locations` to store the current location of each object (robot and boxes).
    3. Iterate through each fact in the current state (`node.state`):
       a. Parse the fact string to get its predicate and arguments.
       b. If the predicate is "at", it represents a box location: store the mapping from the box name to its location string in `current_locations`.
       c. If the predicate is "at-robot", it represents the robot's location: store the mapping from "robot" (or a similar key) to its location string in `current_locations`.
    4. For each box and its corresponding goal location stored in `self.box_goals` (extracted during initialization):
       a. Get the box's current location string from `current_locations`.
       b. If the box's current location is different from its goal location:
          i. Calculate the Manhattan distance between the box's current location string and its goal location string using the `manhattan_distance` helper function.
          ii. Add this calculated distance to `total_heuristic`.
    5. Return the final `total_heuristic` value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                # Goal is (at box location)
                box, location = args
                self.box_goals[box] = location
            # We ignore other potential goal predicates if any exist

        # Note: Static facts like 'adjacent' are not explicitly used in this
        # simple Manhattan distance heuristic, but the location parsing assumes
        # the grid structure they define.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        based on the sum of Manhattan distances for boxes not at their goals.
        """
        state = node.state  # Current world state.

        # Find the current location of the robot and all boxes.
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip invalid fact strings
                continue

            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                # Fact is (at object location)
                obj_name = parts[1]
                loc_name = parts[2]
                current_locations[obj_name] = loc_name
            elif predicate == "at-robot" and len(parts) == 2:
                 # Fact is (at-robot location)
                 loc_name = parts[1]
                 current_locations["robot"] = loc_name # Use a fixed key for the robot
            # We ignore other predicates like 'clear' or 'adjacent' for this heuristic

        total_heuristic = 0  # Initialize heuristic cost.

        # Calculate the sum of Manhattan distances for boxes not at their goals.
        for box, goal_location in self.box_goals.items():
            current_location = current_locations.get(box) # Get box's current location

            # If the box is not found in the state's 'at' facts, it might be an issue,
            # but we proceed assuming valid states.
            if current_location is None:
                 # This case indicates an inconsistency or unexpected state structure.
                 # In a real planner, you might want to handle this more robustly.
                 # For this heuristic, we assume the box is always located somewhere.
                 # print(f"Warning: Location for box '{box}' not found in state.")
                 continue # Skip this box or assign a high penalty

            if current_location != goal_location:
                # Add the Manhattan distance between current and goal location
                distance = manhattan_distance(current_location, goal_location)
                # If distance is infinity (due to parsing error), the state might be invalid or unreachable
                if distance == float('inf'):
                    return float('inf') # Indicate this state is likely problematic/unsolvable

                total_heuristic += distance

        # The heuristic is the sum of box-goal distances.
        # It does not include robot distance or costs related to clearing paths.
        # This makes it admissible.
        return total_heuristic

