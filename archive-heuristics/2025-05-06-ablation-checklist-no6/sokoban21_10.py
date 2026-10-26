from fnmatch import fnmatch
# from heuristics.heuristic_base import Heuristic # Uncomment if inheriting from a base class

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a strict match,
    # as seen in the example usage.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(location_str):
    """
    Parses a location string like 'loc_R_C' into a (row, col) tuple.
    Assumes the format is always 'loc_<row>_<col>'.
    """
    # location_str is like 'loc_R_C'
    parts = location_str.split('_')
    # Assuming format is always loc_R_C
    row = int(parts[1])
    col = int(parts[2])
    return (row, col)

def manhattan_distance(loc1_str, loc2_str):
    """
    Calculates the Manhattan distance between two locations given as 'loc_R_C' strings.
    """
    r1, c1 = parse_location(loc1_str)
    r2, c2 = parse_location(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)

# Define the heuristic class
# If inheriting from a base class, uncomment the line below and add (Heuristic)
class sokobanHeuristic: #(Heuristic)
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost by summing the Manhattan distances of boxes to their goals
    and the estimated robot movement cost to reach the nearest box that needs moving.

    Heuristic value = sum(ManhattanDistance(box_loc, goal_loc) for boxes not at goal)
                      + max(0, ManhattanDistance(robot_loc, nearest_box_loc) - 1)
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # We don't strictly need static facts for this Manhattan distance heuristic,
        # but they are available in task.static if needed for more complex versions
        # (e.g., graph distance, parsing grid structure).

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Find the robot's current location.
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        # Assuming robot location is always present in a valid state.

        # Find current locations of all boxes relevant to the goals.
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                # Only consider objects that are boxes and have a goal location.
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        total_box_distance = 0
        boxes_not_at_goal_locations = [] # Store locations of boxes not at goal

        # Calculate sum of Manhattan distances for boxes not at goal
        # and collect locations of boxes not at goal.
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box) # Get current location

            # If a box is not found in the state, it's an invalid state representation
            # for Sokoban where boxes are always 'at' a location.
            # We assume valid states where all goal boxes are present.
            if current_loc is None:
                 # This case indicates an issue with state representation or problem definition.
                 # For a heuristic, we might return infinity or a large value,
                 # but assuming valid states, this won't happen.
                 continue # Skip this box if its location isn't in the state facts.

            if current_loc != goal_loc:
                total_box_distance += manhattan_distance(current_loc, goal_loc)
                boxes_not_at_goal_locations.append(current_loc)

        # If all boxes are at their goals, the heuristic is 0.
        if not boxes_not_at_goal_locations:
            return 0

        # Find the minimum distance from the robot to any box not at its goal.
        min_dist_robot_to_box = float('inf')
        for box_loc in boxes_not_at_goal_locations:
             dist_robot_to_this_box = manhattan_distance(robot_location, box_loc)
             min_dist_robot_to_box = min(min_dist_robot_to_box, dist_robot_to_this_box)


        # Add the estimated robot movement cost to reach the nearest box.
        # Estimate: distance to get adjacent to the box.
        # Manhattan distance from robot to box location is `d`.
        # To be adjacent, the robot needs to be `d-1` steps away (if d > 0).
        robot_movement_cost = max(0, min_dist_robot_to_box - 1)

        # The total heuristic is the sum of box progress needed and robot positioning needed.
        total_heuristic = total_box_distance + robot_movement_cost

        return total_heuristic
