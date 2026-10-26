from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts and locations

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at box1 loc_3_5)" -> ["at", "box1", "loc_3_5"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Uses fnmatch for wildcard matching.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_3_5)".
    - `args`: The expected pattern (wildcards `*` allowed), e.g., "at", "*", "loc_3_5".
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(location_str):
    """
    Parses a location string like 'loc_R_C' into a (row, col) tuple.
    Assumes the format is always 'loc_R_C'.
    """
    parts = location_str.split('_')
    if len(parts) != 3 or parts[0] != 'loc':
        # This indicates an unexpected location format in the problem definition or state.
        # While this shouldn't happen in valid problems, raising an error or
        # returning a special value could be alternatives to a penalty.
        raise ValueError(f"Unexpected location format: {location_str}")
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        # This indicates non-integer row/col in the location string.
        raise ValueError(f"Could not parse row/col from location: {location_str}")

def manhattan_distance(loc1_str, loc2_str):
    """
    Calculates Manhattan distance between two location strings in 'loc_R_C' format.
    """
    r1, c1 = parse_location(loc1_str)
    r2, c2 = parse_location(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the difficulty of a state by summing the Manhattan
    distances between each box and its assigned goal location, and adding the
    minimum Manhattan distance from the robot to any misplaced box. This
    provides a non-admissible estimate that considers both the box positions
    relative to goals and the robot's proximity to the work area.

    # Assumptions:
    - The goal state specifies the target location for each box using `(at boxX locY)` facts.
    - Location names follow the format `loc_R_C` where R is the row and C is the column.
    - There is a fixed assignment of boxes to goal locations implied by the goal facts.
    - The heuristic does not consider obstacles (like walls or other boxes blocking paths)
      or the specific push mechanic constraints (e.g., deadlocks in corners), beyond
      using Manhattan distance on the grid coordinates.

    # Heuristic Initialization
    - Extracts the mapping from each box object to its required goal location
      from the task's goal conditions. This mapping is stored in `self.goal_locations`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the current location of the robot and every box in the given state by iterating
       through the state facts. Store box locations in `current_box_locations` and the robot
       location in `robot_location`.
    3. Create an empty list `misplaced_boxes_locations` to store the current locations of boxes
       that are not yet at their goal.
    4. Iterate through the `self.goal_locations` mapping (box -> goal_location):
       a. For the current box, get its goal location.
       b. Find the box's current location from `current_box_locations`.
       c. If the box is found and its current location is not the same as its goal location:
          i. Add the box's current location to the `misplaced_boxes_locations` list.
          ii. Calculate the Manhattan distance between the box's current location and its goal location.
          iii. Add this calculated distance to the `total_cost`. Handle potential `ValueError` from
               location parsing by adding a penalty.
    5. If there are any boxes in the `misplaced_boxes_locations` list and the robot's location
       was found:
       a. Calculate the minimum Manhattan distance from the robot's current location
          to the current location of any box in the `misplaced_boxes_locations` list.
       b. Add this minimum distance to the `total_cost`. Handle potential `ValueError` from
          location parsing by adding a penalty.
    6. Return the final `total_cost` as the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Store goal locations for each box.
        # Assuming goal facts are like (at boxX loc_goalY)
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Check for '(at ?o - box ?l - location)' facts in the goal
            if predicate == "at" and len(args) == 2:
                obj, location = args
                # In Sokoban, 'at' goals typically refer to boxes.
                # We assume objects starting with 'box' are the boxes we need to move.
                if obj.startswith('box'):
                     self.goal_locations[obj] = location
            # Other goal predicates (like robot position) are ignored by this heuristic.

        # Static facts (like 'adjacent') are not directly used in this simple heuristic.
        # self.static_facts = task.static

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        Estimates the total Manhattan distance from each box to its goal location
        plus the minimum distance from the robot to a misplaced box.
        """
        state = node.state  # Current world state.

        # Track current locations of boxes and the robot.
        current_box_locations = {}
        robot_location = None

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj, location = args
                # Assuming 'at' facts for boxes start with 'box'
                if obj.startswith('box'):
                    current_box_locations[obj] = location
            elif predicate == "at-robot" and len(args) == 1:
                robot_location = args[0]

        total_cost = 0  # Initialize heuristic cost.
        misplaced_boxes_locations = [] # Collect locations of boxes not at their goal

        # Calculate sum of box-goal distances
        for box, goal_location in self.goal_locations.items():
            # Check if the box exists in the current state (it should)
            if box in current_box_locations:
                current_location = current_box_locations[box]

                # If the box is not already at its goal, add its distance
                if current_location != goal_location:
                    misplaced_boxes_locations.append(current_location)
                    try:
                        # Calculate Manhattan distance between current and goal locations
                        distance = manhattan_distance(current_location, goal_location)
                        total_cost += distance
                    except ValueError as e:
                        # This indicates a problem with the location format. Add a penalty.
                        print(f"Warning: Could not parse location string for box {box} distance: {e}")
                        total_cost += 1000 # Add a penalty

        # Add robot distance to the nearest misplaced box (if any)
        if misplaced_boxes_locations and robot_location:
            min_robot_distance = float('inf')
            try:
                for box_loc in misplaced_boxes_locations:
                    dist = manhattan_distance(robot_location, box_loc)
                    min_robot_distance = min(min_robot_distance, dist)

                # Add the minimum distance found (if finite)
                if min_robot_distance != float('inf'):
                     total_cost += min_robot_distance
            except ValueError as e:
                 # This indicates a problem parsing robot or box location. Add a penalty.
                 print(f"Warning: Could not parse location string for robot distance: {e}")
                 total_cost += 1000 # Add a penalty


        # The heuristic is 0 if and only if all boxes are at their goal locations
        # (assuming the goal covers all relevant boxes).
        # It is finite for any state with valid locations.
        # It is efficiently computable (linear in the number of boxes with goals).
        # It is not admissible as it ignores obstacles and the push mechanics' complexities.

        return total_cost
