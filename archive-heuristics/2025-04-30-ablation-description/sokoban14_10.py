from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_location(loc_str):
    """Parses a location string like 'loc_r_c' into a tuple (r, c)."""
    # Assumes location names are in the format 'loc_row_col'
    parts = loc_str.split('_')
    # Basic check for expected format
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        except ValueError:
            # Handle cases where row/col are not integers if necessary,
            # but PDDL examples suggest they are.
            pass
    # Return a default or raise an error if format is unexpected
    # For this domain, assuming the format is consistent.
    # If not, this heuristic might fail or return incorrect values.
    # Raising an error is appropriate for unexpected input format.
    raise ValueError(f"Unexpected location format: {loc_str}")


def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two location strings 'loc_r_c'."""
    r1, c1 = parse_location(loc1_str)
    r2, c2 = parse_location(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)

# Heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of Manhattan distances for each box from its current location to its goal location.
    2. The Manhattan distance from the robot's current location to the nearest box that is not yet at its goal location.

    # Assumptions
    - The locations are structured like a grid, and their names follow the pattern 'loc_row_col'.
    - The Manhattan distance on the grid coordinates is a reasonable approximation of the shortest path distance on the location graph defined by 'adjacent' predicates. This approximation ignores obstacles (walls, other boxes) and the need for clear paths.
    - The cost of moving the robot and pushing boxes is related to the distances involved.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the task definition (`task.goals`).
    2. In the current state (`node.state`), find the current location of the robot (`at-robot`) and each box (`at`).
    3. Initialize `total_box_distance` to 0.
    4. Initialize `min_robot_to_box_dist` to infinity.
    5. Initialize `has_boxes_not_at_goal` to False.
    6. Iterate through each box for which a goal location is defined in `self.goal_locations`:
       a. Get the box's current location (`current_loc`) from `current_box_locations` and its goal location (`goal_loc`) from `self.goal_locations`.
       b. If `current_loc` is found and is not equal to `goal_loc`:
          i. Set `has_boxes_not_at_goal` to True.
          ii. Calculate the Manhattan distance between `current_loc` and `goal_loc`. Add this distance to `total_box_distance`.
          iii. If the robot's location (`robot_loc`) was successfully found, calculate the Manhattan distance between `robot_loc` and `current_loc`. Update `min_robot_to_box_dist` if this distance is smaller.
    7. If `has_boxes_not_at_goal` is True (meaning there is at least one box not at its goal):
       a. The total heuristic is `total_box_distance` plus `min_robot_to_box_dist`.
    8. If `has_boxes_not_at_goal` is False (meaning all boxes are at their goals):
       a. The total heuristic is 0.
    9. Return the calculated `total_heuristic`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at box_name loc_name)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1:]
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current locations of robot and boxes.
        robot_loc = None
        current_box_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                box, loc = parts[1:]
                current_box_locations[box] = loc

        total_box_distance = 0
        min_robot_to_box_dist = float('inf')
        has_boxes_not_at_goal = False

        # Calculate sum of distances for boxes not at goal and find nearest such box to robot.
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box) # Get current location, None if box not found (shouldn't happen in valid states)

            # Check if the box exists in the current state and is not at its goal
            if current_loc and current_loc != goal_loc:
                has_boxes_not_at_goal = True
                # Add box-to-goal distance
                total_box_distance += manhattan_distance(current_loc, goal_loc)

                # Update minimum robot-to-box distance
                # Ensure robot location is found before calculating distance
                if robot_loc:
                    dist = manhattan_distance(robot_loc, current_loc)
                    min_robot_to_box_dist = min(min_robot_to_box_dist, dist)
                # Note: If robot_loc is None, min_robot_to_box_dist remains infinity.
                # This case implies an issue with state representation if robot location is missing.
                # Assuming valid states always have robot_loc.

        # The heuristic is the sum of box-to-goal distances plus the robot's distance
        # to the nearest box that needs moving (if any).
        if has_boxes_not_at_goal:
             # If robot_loc was found, min_robot_to_box_dist is finite.
             # If robot_loc was NOT found, min_robot_to_box_dist is infinity.
             # Adding infinity to total_box_distance results in infinity.
             # This is a reasonable behavior for an invalid state (missing robot).
             total_heuristic = total_box_distance + min_robot_to_box_dist
        else:
            # All boxes are at their goal locations.
            total_heuristic = 0

        return total_heuristic
