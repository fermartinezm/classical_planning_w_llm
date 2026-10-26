from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_4_4)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check: just zip and compare, fnmatch handles wildcards
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """
    Parses a location string like 'loc_row_col' into a (row, col) tuple of integers.
    Assumes the format is always 'loc_R_C' where R and C are integers.
    """
    try:
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format gracefully, maybe return None or raise error
            print(f"Warning: Unexpected location format: {loc_str}")
            return None # Or raise ValueError(f"Unexpected location format: {loc_str}")
    except ValueError:
        print(f"Warning: Could not parse row/col integers from location: {loc_str}")
        return None # Or raise ValueError(f"Could not parse row/col integers from location: {loc_str}")


def manhattan_distance(loc1_str, loc2_str):
    """
    Calculates the Manhattan distance between two locations given as strings.
    Returns infinity if parsing fails for either location.
    """
    coords1 = parse_location(loc1_str)
    coords2 = parse_location(loc2_str)

    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid locations

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing the Manhattan distances of each box to its goal location and
    adding the minimum Manhattan distance from the robot to any box that is
    not yet at its goal.

    # Assumptions:
    - The goal specifies the target location for each box.
    - Locations are named in the format 'loc_row_col'.
    - Manhattan distance is a reasonable approximation of path distance in the grid.
    - The cost of moving the robot to a box is estimated by Manhattan distance.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the task definition.
    2. For the current state:
       a. Find the robot's current location.
       b. Find the current location of each box.
    3. Initialize total heuristic cost to 0.
    4. Initialize minimum robot-to-box distance for non-goal boxes to infinity.
    5. Iterate through each box and its goal location:
       a. Get the box's current location.
       b. If the box is not at its goal location:
          i. Calculate the Manhattan distance between the box's current location and its goal location. Add this to the total heuristic cost.
          ii. Calculate the Manhattan distance between the robot's current location and the box's current location. Update the minimum robot-to-box distance found so far.
    6. If there were any boxes not at their goal (i.e., minimum robot-to-box distance is not infinity), add the minimum robot-to-box distance to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at <box> <location>)
            if match(goal, "at", "*", "*"):
                 # Check if the first argument is a box type (simple check based on naming convention or task objects)
                 # A more robust way would be to check task.objects and their types, but assuming 'box' type for now.
                 # Let's assume any 'at' predicate in the goal refers to a box.
                predicate, obj, location = get_parts(goal)
                # We should verify 'obj' is a box. For simplicity, assume any object in an 'at' goal is a box.
                # A proper parser would give types. Given the example, 'box1' etc. are objects of type 'box'.
                # We can check if the object name starts with 'box' or is listed in task.objects as a box.
                # For this problem, let's assume objects in (at ?obj ?loc) goals are the boxes we care about.
                self.goal_locations[obj] = location

        # print(f"Initialized Sokoban Heuristic. Goal locations: {self.goal_locations}")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        if robot_location is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             # print("Warning: Robot location not found in state.")
             return float('inf') # Cannot solve without robot

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            # Check for (at ?box ?location) facts
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                 # We need to check if parts[1] is actually a box.
                 # A simple way is to see if it's one of the objects in our goal_locations keys.
                 obj_name = parts[1]
                 if obj_name in self.goal_locations:
                    current_box_locations[obj_name] = parts[2]

        total_box_goal_distance = 0
        min_robot_box_distance = float('inf')
        found_box_not_at_goal = False

        # Calculate heuristic components for each box
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box) # Use .get for safety

            if current_location is None:
                 # This box is missing from the state, indicates an invalid state representation
                 # print(f"Warning: Box {box} not found in state.")
                 return float('inf') # Cannot solve if a box is missing

            if current_location != goal_location:
                found_box_not_at_goal = True
                # Add box-to-goal distance
                box_dist = manhattan_distance(current_location, goal_location)
                if box_dist == float('inf'): return float('inf') # Propagate infinity if location parsing failed
                total_box_goal_distance += box_dist

                # Find minimum robot-to-box distance for boxes not at goal
                robot_dist = manhattan_distance(robot_location, current_location)
                if robot_dist == float('inf'): return float('inf') # Propagate infinity
                min_robot_box_distance = min(min_robot_box_distance, robot_dist)

        # If all boxes are at their goal, the heuristic is 0
        if not found_box_not_at_goal:
            return 0

        # The heuristic is the sum of box-to-goal distances plus the cost for the robot
        # to reach the nearest box that needs moving.
        # We add min_robot_box_distance only if there's at least one box not at goal.
        # This prevents adding infinity if all boxes are at goal.
        return total_box_goal_distance + min_robot_box_distance

