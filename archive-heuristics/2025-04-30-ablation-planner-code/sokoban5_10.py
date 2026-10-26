from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
# No need for deque if not using BFS

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Allow pattern to be shorter than fact parts (e.g., matching predicate and first arg)
    if len(args) > len(parts):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse loc_X_Y into coordinates (X, Y)
def parse_location_coords(location_str):
    """Parses a location string like 'loc_X_Y' into an (X, Y) tuple."""
    try:
        parts = location_str.split('_')
        # Assuming format is loc_row_col (row, column)
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (IndexError, ValueError):
        # If parsing fails, this location is invalid for this heuristic
        return None

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain using Manhattan distance.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing
    the Manhattan distances for each box to its goal location and adding
    the Manhattan distance for the robot to reach the nearest box that
    needs to be moved.

    # Assumptions
    - Location names follow the 'loc_X_Y' format, allowing extraction of grid coordinates.
    - Manhattan distance on these coordinates is a reasonable estimate of path distance.
    - This heuristic is non-admissible.

    # Heuristic Initialization
    - Parses the goal facts to store the target location (as coordinates) for each box.
    - Static facts are not used as the heuristic relies on coordinate geometry.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Convert all relevant location strings ('loc_X_Y') into (X, Y) coordinate tuples. If any conversion fails, the state is considered unsolvable (infinite heuristic).
    3. Identify the goal location (as coordinates) for each box from the pre-parsed goal information.
    4. Initialize the total heuristic value `h` to 0.
    5. Initialize a list to store coordinates of boxes that are not yet at their goals.
    6. For each box:
       - Get its current location coordinates and its goal location coordinates.
       - If the box is not at its goal location:
         - Calculate the Manhattan distance between the box's current coordinates
           and its goal coordinates. Add this distance to `h`.
         - Add the box's current coordinates to the list of coordinates of boxes needing move.
    7. If there are boxes that need to be moved:
       - Get the robot's current location coordinates.
       - Calculate the Manhattan distance from the robot's current coordinates
         to the nearest coordinates among the boxes needing move. Add this
         minimum distance to `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations as coordinates.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not used in this Manhattan distance heuristic.
        # static_facts = task.static

        # Store goal locations as coordinates for each box
        self.goal_locations_coords = {}
        for goal in self.goals:
            # Goal facts are typically (at boxX locY)
            if match(goal, "at", "*", "*"):
                _, box, location_str = get_parts(goal)
                self.goal_locations_coords[box] = parse_location_coords(location_str)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot and box locations in the current state and convert to coordinates
        robot_coords = None
        box_coords = {} # Map box name to its coordinates

        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_location_str = get_parts(fact)
                robot_coords = parse_location_coords(robot_location_str)
                if robot_coords is None: return float('inf') # Invalid robot location format
            elif match(fact, "at", "*", "*"):
                _, box, location_str = get_parts(fact)
                coords = parse_location_coords(location_str)
                if coords is None: return float('inf') # Invalid box location format
                box_coords[box] = coords
            # clear facts are not needed for this heuristic

        total_cost = 0  # Initialize heuristic cost

        boxes_needing_move_coords = []

        # Calculate cost for each box to reach its goal
        for box, goal_coords in self.goal_locations_coords.items():
            current_box_coords = box_coords.get(box)

            if current_box_coords is None:
                 # Box mentioned in goal is not in the current state? Unsolvable.
                 return float('inf')

            if current_box_coords != goal_coords:
                # Box is not at its goal, calculate Manhattan distance
                dist = abs(current_box_coords[0] - goal_coords[0]) + abs(current_box_coords[1] - goal_coords[1])
                total_cost += dist
                boxes_needing_move_coords.append(current_box_coords)

        # Calculate cost for robot to reach a box that needs moving
        if boxes_needing_move_coords:
            if robot_coords is None:
                 # Robot location not found (already handled above, but defensive check)
                 return float('inf')

            min_robot_dist = float('inf')
            for box_c in boxes_needing_move_coords:
                 dist = abs(robot_coords[0] - box_c[0]) + abs(robot_coords[1] - box_c[1])
                 min_robot_dist = min(min_robot_dist, dist)

            # min_robot_dist will be finite here because robot_coords is valid
            total_cost += min_robot_dist

        # Heuristic is 0 only if boxes_needing_move_coords is empty, which means all boxes are at goals.
        # If boxes need moving, total_cost > 0 (assuming coordinates are valid and reachable,
        # which Manhattan distance doesn't guarantee but provides an estimate).

        return total_cost
