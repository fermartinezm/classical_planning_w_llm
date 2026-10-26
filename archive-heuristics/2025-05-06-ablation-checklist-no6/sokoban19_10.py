# Assuming the Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If running standalone or base class is not provided, use a dummy:
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_location(location_str):
    """Parses 'loc_r_c' string into (row, col) tuple."""
    parts = location_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where parts[1] or parts[2] are not integers
            # print(f"Warning: Could not parse location string '{location_str}'") # Optional warning
            return None
    # print(f"Warning: Unexpected location string format '{location_str}'") # Optional warning
    return None

def manhattan_distance(loc1_str, loc2_str):
    """Calculates Manhattan distance between two locations."""
    coords1 = parse_location(loc1_str)
    coords2 = parse_location(loc2_str)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if parsing failed
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain using Manhattan distance.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the Manhattan distances of all misplaced boxes to their
    respective goal locations and adding the Manhattan distance from the robot
    to the closest misplaced box.

    # Assumptions
    - Locations are named in the format 'loc_row_col' allowing Manhattan distance calculation.
    - Each box has a unique goal location specified in the task goals.
    - The heuristic ignores obstacles (walls, other boxes) and the specific
      robot positioning required for pushes, providing a relaxed estimate.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current location.
    2. Identify the current location of each box that has a goal.
    3. Find all boxes that are not currently at their goal locations.
    4. If no boxes are misplaced, the heuristic is 0 (goal state).
    5. If boxes are misplaced:
       a. Calculate the sum of Manhattan distances for each misplaced box
          from its current location to its goal location. This estimates the
          minimum number of "box steps" required.
       b. Calculate the Manhattan distance from the robot's current location
          to the location of each misplaced box.
       c. Find the minimum of these robot-to-box distances. This estimates
          the robot's cost to reach a box it needs to move.
       d. The total heuristic value is the sum from step 5a and step 5c.
          This combines the effort to move the boxes with the effort to get
          the robot positioned to start working on the most accessible box.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        super().__init__(task)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at box location)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                 # We assume goals are always (at box_name location_name)
                 box, location = parts[1], parts[2]
                 self.goal_locations[box] = location
            # Add other potential goal types if necessary, but (at box location) is standard

        # Static facts like 'adjacent' are not used by this Manhattan heuristic
        # self.static = task.static # Already stored by super().__init__
        # If we were using graph distance, we would build the graph here.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
                break

        # If robot location is not found, something is wrong with the state
        # or domain representation. Return infinity as it's likely unsolvable
        # from this state.
        if robot_location is None:
             return float('inf')

        # Find current box locations for boxes we care about (those with goals)
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check for (at box location) facts
            if parts[0] == "at" and len(parts) == 3:
                 box, location = parts[1], parts[2]
                 # Only consider facts about boxes that are in our goals
                 if box in self.goal_locations:
                    box_locations[box] = location
            # Note: If a box is not 'at' a location (e.g., carried, though not in Sokoban),
            # it won't be in box_locations. This heuristic assumes boxes are always 'at' a location.

        # Identify misplaced boxes and calculate sum of box-goal distances
        misplaced_boxes = []
        boxes_distance = 0
        for box, goal_l in self.goal_locations.items():
            current_l = box_locations.get(box)
            # If a box from the goals is not found in the state's 'at' facts,
            # it's likely an invalid state or the box is not on the map.
            # We'll treat it as infinitely far for heuristic purposes.
            if current_l is None:
                 return float('inf') # Box from goal not found in state

            if current_l != goal_l:
                misplaced_boxes.append(box)
                dist = manhattan_distance(current_l, goal_l)
                # If any box-goal distance is infinite (due to parsing error),
                # the total heuristic is infinite.
                if dist == float('inf'):
                    return float('inf')
                boxes_distance += dist

        # If all boxes are at their goals, heuristic is 0
        if not misplaced_boxes:
            return 0

        # Calculate minimum robot distance to a misplaced box
        min_robot_box_distance = float('inf')
        for box in misplaced_boxes:
            box_l = box_locations[box]
            dist = manhattan_distance(robot_location, box_l)
            # If any robot-box distance is infinite (due to parsing error),
            # the min distance will be infinite.
            if dist == float('inf'):
                 return float('inf')
            min_robot_box_distance = min(min_robot_box_distance, dist)

        # The heuristic is the sum of box-goal Manhattan distances plus the
        # minimum Manhattan distance from the robot to any misplaced box.
        # This estimates the total box movement needed plus the robot's cost
        # to get to a position to start pushing the "most accessible" box.
        return boxes_distance + min_robot_box_distance
