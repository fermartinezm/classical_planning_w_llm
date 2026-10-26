# from fnmatch import fnmatch # Not used
from heuristics.heuristic_base import Heuristic

# Helper functions (defined outside the class but in the same file)

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         return [] # Return empty list for invalid format
    return fact[1:-1].split()

def parse_loc(loc_str):
    """Parses a location string like 'loc_R_C' into a (row, col) tuple."""
    if not isinstance(loc_str, str):
        return None
    try:
        parts = loc_str.split('_')
        # Assuming format is loc_row_col
        # Check if there are enough parts and the row/col parts are digits
        if len(parts) == 3 and parts[0] == 'loc' and parts[1].isdigit() and parts[2].isdigit():
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
             return None # Unexpected format
    except (ValueError, IndexError):
        return None # Error during conversion or splitting

def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two locations."""
    coords1 = parse_loc(loc1_str)
    coords2 = parse_loc(loc2_str)

    if coords1 is None or coords2 is None:
        # Cannot calculate distance if locations are invalid or unparseable
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach a goal state
     by summing the Manhattan distances for each box to its goal location and
     adding the Manhattan distance from the robot to the nearest box that is
     not yet at its goal. This is a non-admissible heuristic designed for
     greedy best-first search.

    # Assumptions
    - The locations in the domain follow a 'loc_R_C' naming convention,
      representing a grid structure where Manhattan distance is a reasonable
      approximation of shortest path distance.
    - Each box that needs to be moved to a specific location in the goal state
      has a unique target location specified in the goal conjunction.
    - The heuristic ignores obstacles (other boxes, walls) when calculating
      Manhattan distances, treating the grid as freely traversable for
      distance calculation purposes.
    - The heuristic ignores the requirement for the robot to be on a specific side
      of the box to push it.

    # Heuristic Initialization
    - Parses the goal conditions to create a mapping from each box that must
      be at a specific location in the goal state to its target location.
    - Static facts like 'adjacent' are available but not explicitly used
      in the current Manhattan distance calculation, which relies solely
      on the 'loc_R_C' naming convention.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state.
    2. Identify the current location of each box from the state.
    3. Determine the set of boxes that are not currently at their assigned
       goal locations (based on the goal mapping initialized earlier).
    4. If this set of boxes is empty, all relevant boxes are at their goals,
       and the state is a goal state. The heuristic value is 0.
    5. If there are boxes not at their goals:
       a. Calculate the sum of Manhattan distances for each box in this set
          from its current location to its goal location. This provides a
          relaxed estimate of the total 'push' effort needed for all boxes.
       b. Calculate the Manhattan distance from the robot's current location
          to the current location of the *nearest* box that needs to be moved.
          This provides a relaxed estimate of the 'move' effort needed for
          the robot to reach a box it needs to interact with.
       c. The total heuristic value is the sum of the total box-to-goal distance
          and the robot-to-nearest-box distance. If any required location string
          cannot be parsed, the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        # static_facts = task.static # Static facts are available but not used

        # Store goal locations for each box that must be at a specific location.
        # Assuming each such box has exactly one goal location in the goal state.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically like '(at box1 loc_2_4)'
            parts = get_parts(goal)
            # Check for '(at ?box ?location)' facts where ?box is a box object
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Note: The location graph from 'adjacent' facts is not built or used
        # in this Manhattan distance based heuristic for efficiency.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        robot_loc = None
        box_locations = {}

        # Extract robot and box locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            elif parts and parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        # Identify boxes that are not at their goal locations
        # Only consider boxes that have a specified goal location in self.goal_locations
        boxes_to_move = {
            box for box, loc in box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        }

        # If all relevant boxes are at their goals, it's a goal state
        if not boxes_to_move:
            return 0

        total_box_distance = 0
        min_robot_distance = float('inf')

        # Calculate sum of Manhattan distances for boxes to their goals
        for box in boxes_to_move:
            box_loc = box_locations.get(box) # Get current box location
            goal_loc = self.goal_locations.get(box) # Get goal location

            # Should not be None if box is in boxes_to_move, but defensive check
            if box_loc is None or goal_loc is None:
                 return float('inf') # Cannot compute heuristic if locations are missing

            dist = manhattan_distance(box_loc, goal_loc)
            if dist == float('inf'):
                 # This happens if parse_loc failed for box_loc or goal_loc
                 return float('inf') # Cannot compute heuristic

            total_box_distance += dist

        # Calculate distance from robot to nearest box that needs moving
        # Robot location is assumed to be present if boxes_to_move is not empty
        # in a valid problem state.
        if robot_loc is None:
             # Robot location not found in state - invalid state?
             return float('inf') # Cannot compute heuristic

        robot_coords = parse_loc(robot_loc)
        if robot_coords is None:
             # Robot location string is unparseable
             return float('inf') # Cannot compute heuristic

        for box in boxes_to_move:
            box_loc = box_locations.get(box) # Already checked for None above, but defensive
            if box_loc is None: continue

            dist_to_box = manhattan_distance(robot_loc, box_loc)
            # If dist_to_box is inf, it means box_loc is unparseable (robot_loc already checked)
            # If any box_loc is unparseable, min_robot_distance will remain inf if all are.
            # If at least one box_loc is parseable, min_robot_distance will be finite.
            min_robot_distance = min(min_robot_distance, dist_to_box)

        # If min_robot_distance is still inf, it means either boxes_to_move was empty (handled)
        # or robot_loc was None (handled) or all box_locs in boxes_to_move were unparseable.
        # In any case, return inf.
        if min_robot_distance == float('inf'):
             return float('inf')


        # Final heuristic value is the sum of box-to-goal distances and robot-to-nearest-box distance.
        return total_box_distance + min_robot_distance
