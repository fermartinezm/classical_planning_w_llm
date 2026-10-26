from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to parse location string into (row, col)
def parse_location(loc_str):
    """
    Parses a location string like 'loc_R_C' into a (row, col) tuple.
    Returns None if the format is not recognized or parsing fails.
    """
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None
    return None

# Helper function to calculate Manhattan distance between two locations
def manhattan_distance(loc1_str, loc2_str):
    """
    Calculates the Manhattan distance between two location strings.
    Returns float('inf') if either location string cannot be parsed.
    """
    coord1 = parse_location(loc1_str)
    coord2 = parse_location(loc2_str)

    if coord1 is None or coord2 is None:
        return float('inf')

    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the Manhattan distances for each box to its goal location
    and adding the Manhattan distance from the robot to the closest box
    that is not yet at its goal.

    # Assumptions
    - Location names follow the pattern 'loc_R_C' where R and C are integers
      representing row and column, allowing Manhattan distance calculation.
    - Each box specified in the goal has a specific goal location assigned.
    - The grid structure implied by location names is consistent with
      Manhattan distance.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
      Stores this mapping in `self.box_goals`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value to 0.
    2. Identify the robot's current location from the state. If not found or unparsable, return infinity.
    3. Initialize `total_box_to_goal_dist = 0`.
    4. Initialize `min_robot_to_box_dist = float('inf')`.
    5. Initialize `off_goal_boxes_present = False`.
    6. For each box specified in the task's goals (`self.box_goals`):
       a. Find the box's current location in the state. If not found or unparsable, return infinity.
       b. Get the box's goal location from `self.box_goals`.
       c. If the box's current location is not its goal location:
          i. Set `off_goal_boxes_present = True`.
          ii. Calculate the Manhattan distance between the box's current location
              and its goal location (`box_goal_dist`). If infinity, return infinity.
          iii. Add `box_goal_dist` to `total_box_to_goal_dist`.
          iv. Calculate the Manhattan distance between the robot's current location
              and the box's current location (`robot_box_dist`). If infinity, return infinity.
          v. Update `min_robot_to_box_dist = min(min_robot_to_box_dist, robot_box_dist)`.
    7. If `off_goal_boxes_present` is False (meaning all goal boxes are at their goals), return 0.
    8. Otherwise, return `total_box_to_goal_dist + min_robot_to_box_dist`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            # Assuming goals are of the form (at box_name loc_name)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location
            # Note: This heuristic assumes goals are only 'at' predicates for boxes.
            # If other goal predicates exist, they are ignored by this heuristic.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot's current location
        robot_location = None
        # Find current box locations for goal-relevant boxes
        current_box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == "at" and len(parts) == 3 and parts[1] in self.box_goals:
                 # Only track boxes that are relevant to the goal
                box, location = parts[1], parts[2]
                current_box_locations[box] = location

        # Basic validation: Robot location must be found and parsable.
        if robot_location is None or parse_location(robot_location) is None:
             # This indicates an invalid state representation or unparsable robot location.
             return float('inf')

        total_box_to_goal_dist = 0
        min_robot_to_box_dist = float('inf')
        off_goal_boxes_present = False

        for box, goal_loc in self.box_goals.items():
            current_loc = current_box_locations.get(box)

            # If a goal box is not found in the state, something is wrong.
            # Or if its location is unparsable.
            if current_loc is None or parse_location(current_loc) is None:
                 return float('inf') # Indicate error/unparsable state

            if current_loc != goal_loc:
                off_goal_boxes_present = True

                # Calculate box-to-goal distance
                box_goal_dist = manhattan_distance(current_loc, goal_loc)
                if box_goal_dist == float('inf'): return float('inf') # Propagate error

                total_box_to_goal_dist += box_goal_dist

                # Calculate robot-to-box distance
                robot_box_dist = manhattan_distance(robot_location, current_loc)
                if robot_box_dist == float('inf'): return float('inf') # Propagate error

                min_robot_to_box_dist = min(min_robot_to_box_dist, robot_box_dist)

        # If all goal boxes are at their goal, the heuristic is 0.
        if not off_goal_boxes_present:
             return 0

        # If there are off-goal boxes, return the sum of box-goal distances
        # plus the minimum robot-box distance.
        # If min_robot_to_box_dist is still inf, it means off_goal_boxes_present was true
        # but min_robot_to_box_dist was never updated (e.g., all box_goal_dist were inf,
        # which would have returned inf earlier). This check is mostly for safety.
        if min_robot_to_box_dist == float('inf'):
             # This case should ideally not be reached if off_goal_boxes_present is True
             # and no inf was returned earlier.
             return float('inf') # Should indicate a logic error or unhandled state issue

        return total_box_to_goal_dist + min_robot_to_box_dist
