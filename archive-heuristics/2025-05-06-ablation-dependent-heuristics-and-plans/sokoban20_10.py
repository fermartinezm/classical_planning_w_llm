# Imports
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by whitespace
    return fact.strip()[1:-1].split()

def get_coords(loc_str):
    """Extract (row, col) from location string like 'loc_row_col'."""
    # Assumes loc_r_c format based on problem examples.
    parts = loc_str.split('_')
    # Basic check for expected format parts
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers
            # This indicates a malformed location string
            raise ValueError(f"Malformed location string (non-integer row/col): {loc_str}")
    else:
        # Handle cases where location string is not in expected 'loc_r_c' format
        raise ValueError(f"Unexpected location string format: {loc_str}")

def manhattan_distance(loc1_str, loc2_str):
    """Calculate Manhattan distance between two locations."""
    r1, c1 = get_coords(loc1_str)
    r2, c2 = get_coords(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)

# Heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach a goal state.
    It consists of two main components:
    1. The estimated robot movement cost to reach a position adjacent to the nearest
       misplaced box.
    2. The sum of Manhattan distances between each misplaced box's current location
       and its goal location, estimating the total number of pushes required.

    # Assumptions
    - The goal state specifies the target location for each box using the `(at ?b ?l)` predicate.
    - There is a one-to-one mapping between boxes and their goal locations specified in the goal.
    - Locations are named in the format `loc_row_col` allowing coordinate extraction for Manhattan distance.
    - The `adjacent` facts define the grid connectivity, used to find adjacent locations.
    - The heuristic ignores obstacles (other boxes, walls) when calculating Manhattan distances,
      which makes it non-admissible but faster.
    - The heuristic ignores potential deadlocks.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an adjacency list representation of the grid from the static `adjacent` facts
      to quickly find locations adjacent to a given location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of the robot from the state.
    3. Identify the current location of each box that has a corresponding goal location.
    4. Create a list of misplaced boxes, including their current and goal locations.
    5. If there are no misplaced boxes, the heuristic is 0 (goal state).
    6. If there are misplaced boxes:
       a. Calculate the minimum Manhattan distance from the robot's current location to any location that is adjacent to *any* of the misplaced boxes. Add this minimum distance to the total cost. This estimates the robot's initial movement to get into a position to start pushing *some* box. If no adjacent locations are found for any misplaced box, the heuristic returns infinity.
       b. For each misplaced box, calculate the Manhattan distance between its current location and its goal location. Sum these distances and add the sum to the total cost. This estimates the total number of push actions required across all boxes.
    7. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {} # Map box name string to goal location string
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                # Goal is (at ?b ?l)
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Build adjacency list from static facts.
        # Maps location string to a list of adjacent location strings.
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.adj:
                    self.adj[loc1] = []
                # Store just the adjacent location.
                self.adj[loc1].append(loc2)
                # Assuming adjacent facts are bidirectional in PDDL,
                # we don't need to explicitly add the reverse here.
                # The PDDL example confirms bidirectional facts are provided.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        robot_loc = None
        box_locations = {} # Map box name string to current location string

        # Extract robot and box locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1] in self.goal_locations:
                 # Only track boxes that are relevant to the goal
                box, location = parts[1], parts[2]
                box_locations[box] = location

        # If robot location is not found (shouldn't happen in valid states), return infinity
        if robot_loc is None:
             # print("Warning: Robot location not found in state.")
             return float('inf')

        misplaced_boxes = [] # List of (box_name, current_loc, goal_loc) for boxes not at goal

        # Identify misplaced boxes
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # If a box expected in the goal is not found in the state, treat as high cost
            if current_box_loc is None:
                 # print(f"Warning: Box {box} not found in state.")
                 # This case might indicate a problem with the state representation or task definition
                 # If a goal box is missing from the state, the problem is likely unsolvable from here.
                 return float('inf')

            # If the box is not at its goal, add it to the list of misplaced boxes
            if current_box_loc != goal_loc:
                misplaced_boxes.append((box, current_box_loc, goal_loc))

        # If no boxes are misplaced, the goal is reached (assuming goals are only box locations)
        if not misplaced_boxes:
            return 0

        total_cost = 0

        # Component 1: Robot movement to the nearest misplaced box's adjacent cell
        min_robot_dist_to_any_box_adj = float('inf')
        found_adjacent_location = False

        for box, current_box_loc, goal_loc in misplaced_boxes:
            adjacent_to_box = self.adj.get(current_box_loc, [])
            if adjacent_to_box:
                found_adjacent_location = True
                for adj_loc in adjacent_to_box:
                    try:
                        dist = manhattan_distance(robot_loc, adj_loc)
                        min_robot_dist_to_any_box_adj = min(min_robot_dist_to_any_box_adj, dist)
                    except ValueError:
                        # Handle cases where location format is unexpected during distance calculation
                        # This indicates a problem with the input data, return infinity
                        # print(f"Error calculating distance involving {robot_loc} or {adj_loc}")
                        return float('inf')

        # If no adjacent locations were found for *any* misplaced box, the problem might be unsolvable
        if not found_adjacent_location:
             # print("Error: No adjacent locations found for any misplaced box.")
             return float('inf')

        # Add the minimum robot distance found
        total_cost += min_robot_dist_to_any_box_adj


        # Component 2: Estimated pushes for all misplaced boxes
        for box, current_box_loc, goal_loc in misplaced_boxes:
            try:
                box_dist = manhattan_distance(current_box_loc, goal_loc)
                total_cost += box_dist
            except ValueError:
                 # Handle cases where location format is unexpected during distance calculation
                 # This indicates a problem with the input data, return infinity
                 # print(f"Error calculating box distance involving {current_box_loc} or {goal_loc}")
                 return float('inf')


        return total_cost
