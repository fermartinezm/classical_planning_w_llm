from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """Parses 'loc_X_Y' string into (X, Y) tuple."""
    parts = loc_str.split('_')
    # Assuming format is always loc_X_Y with X, Y being integers
    # Basic validation for robustness
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return (int(parts[1]), int(parts[2]))
    else:
        # This case indicates an unexpected location format.
        # For this heuristic, we assume valid loc_X_Y format for relevant locations.
        # Returning (0,0) or similar could hide issues. Relying on potential errors
        # in int() conversion if format is truly wrong is acceptable for this context.
        return (int(parts[1]), int(parts[2]))


def manhattan_distance(loc1_str, loc2_str):
    """Calculates Manhattan distance between two locations."""
    x1, y1 = parse_location(loc1_str)
    x2, y2 = parse_location(loc2_str)
    return abs(x1 - x2) + abs(y1 - y2)

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances of each box to its respective goal location and adding the minimum
    Manhattan distance from the robot to a valid push position for any box that
    needs to be moved. A valid push position is adjacent to the box, allows
    pushing towards a location that is currently clear.

    # Assumptions
    - Goal is defined by the final locations of specific boxes.
    - Location names follow the format 'loc_X_Y' where X and Y are integers representing grid coordinates.
    - The grid structure implied by location names corresponds to the adjacency relations defined in static facts.
    - The cost of moving a box one step towards its goal is at least 1 (a push action).
    - The cost of moving the robot is related to Manhattan distance.
    - Adjacency facts define the grid connectivity.
    - The state contains all currently true `clear` facts.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an adjacency map from static facts to determine neighboring locations and their directions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box that needs to be moved (initialized).
    2. Build the adjacency map from static facts (initialized).
    3. For the current state, find the current location of the robot, each box, and the set of clear locations.
    4. Calculate the sum of Manhattan distances for all boxes not at their goals:
       a. Initialize `boxes_distance_sum = 0`.
       b. For each box `b` with goal `l_goal_b`:
          i. Get current location `l_b`.
          ii. If `l_b != l_goal_b`, calculate `manhattan_distance(l_b, l_goal_b)` and add to `boxes_distance_sum`.
    5. Calculate the minimum Manhattan distance from the robot to a valid push position:
       a. Initialize `min_robot_to_push_distance = infinity`.
       b. Get robot's current location `l_robot`.
       c. For each box `b` with goal `l_goal_b`:
          i. Get current location `l_b`.
          ii. If `l_b != l_goal_b`:
              - Iterate through all possible directions `push_dir` from `l_b` according to the adjacency map.
              - Find the potential target location `l_b_target` if `b` is pushed from `l_b` in `push_dir`.
              - Find the required robot location `l_r_push` adjacent to `l_b` in the direction opposite `push_dir`.
              - If `l_r_push` exists in the adjacency map and `l_b_target` is a location listed as `clear` in the current state:
                  - Calculate `manhattan_distance(l_robot, l_r_push)`.
                  - Update `min_robot_to_push_distance` if this distance is smaller.
   d. If no valid push position was found for any box that needs moving (i.e., `min_robot_to_push_distance` remains infinity), it indicates a state where no box can be moved towards a clear spot. This is likely a deadlock or an unhelpful state, so return infinity.
    6. The total heuristic value is `boxes_distance_sum + min_robot_to_push_distance`. If no boxes need moving, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building adjacency map.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static # Static facts

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming goal facts are always of the form (at box_name loc_name)
            if predicate == "at" and len(args) == 2:
                 obj, location = args
                 # Assuming objects in 'at' goals are always boxes in Sokoban.
                 self.goal_locations[obj] = location
            # Ignore other potential goal predicates if any

        # Build adjacency map: location -> {direction -> adjacent_location}
        self.adjacency_map = {}
        # Map direction string to its opposite
        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent" and len(args) == 3:
                loc1, loc2, direction = args
                if loc1 not in self.adjacency_map:
                    self.adjacency_map[loc1] = {}
                self.adjacency_map[loc1][direction] = loc2
                # Also add the reverse direction for easier lookup
                opp_dir = self.opposite_direction.get(direction)
                if opp_dir:
                    if loc2 not in self.adjacency_map:
                        self.adjacency_map[loc2] = {}
                    self.adjacency_map[loc2][opp_dir] = loc1


    def get_adjacent_location(self, loc, direction):
        """Returns the location adjacent to loc in the given direction, or None if none exists."""
        return self.adjacency_map.get(loc, {}).get(direction)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        Sum of box-to-goal Manhattan distances + min robot-to-push position Manhattan distance.
        """
        state = node.state  # Current world state.

        current_locations = {}
        robot_location = None
        clear_locations = set()

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2: # (at box loc)
                obj, location = args
                current_locations[obj] = location
            elif predicate == "at-robot" and len(args) == 1: # (at-robot loc)
                 robot_location = args[0]
            elif predicate == "clear" and len(args) == 1: # (clear loc)
                 clear_locations.add(args[0])


        boxes_distance_sum = 0
        boxes_to_move = [] # Keep track of boxes not at goal

        # Calculate sum of box-to-goal distances and identify boxes to move
        for box, goal_location in self.goal_locations.items():
            current_location = current_locations.get(box)

            # If a box required by the goal is not found in the state, it's an invalid state.
            # Return infinity in this case.
            if current_location is None:
                 return float('inf')

            if current_location != goal_location:
                distance = manhattan_distance(current_location, goal_location)
                boxes_distance_sum += distance
                boxes_to_move.append(box) # Add box name to list

        # If all boxes are at their goals, the heuristic is 0.
        if not boxes_to_move:
            return 0

        # If there are boxes to move but no robot location found, it's an invalid state.
        # Return infinity.
        if robot_location is None:
             return float('inf')

        # Calculate minimum robot-to-push position distance
        min_robot_to_push_distance = float('inf')

        for box in boxes_to_move:
            box_location = current_locations.get(box) # Should exist

            # Iterate through all directions the box *could* be pushed from its current location
            # based on the adjacency map.
            possible_push_dirs = self.adjacency_map.get(box_location, {}).keys()

            for push_dir in possible_push_dirs:
                 target_box_location = self.get_adjacent_location(box_location, push_dir)
                 # The location the robot must be at to push in push_dir is adjacent
                 # to box_location in the *opposite* direction.
                 robot_push_direction = self.opposite_direction.get(push_dir)
                 if robot_push_direction:
                     required_robot_location = self.get_adjacent_location(box_location, robot_push_direction)

                     # A push is possible if the required robot location exists
                     # AND the target location for the box is currently clear.
                     if required_robot_location and target_box_location in clear_locations:
                         # This is a potentially valid push position and target location
                         # Calculate distance from robot to this required position
                         distance = manhattan_distance(robot_location, required_robot_location)
                         min_robot_to_push_distance = min(min_robot_to_push_distance, distance)

        # If no valid push position was found for any box that needs moving,
        # it might indicate a deadlock or an unreachable state. Return infinity.
        if min_robot_to_push_distance == float('inf'):
             # This happens if no box can be pushed (e.g., all adjacent squares blocked or no clear target).
             # This is a good indicator of a problematic state or deadlock.
             return float('inf')


        # The heuristic is the sum of box distances plus the minimum robot distance to a push position.
        return boxes_distance_sum + min_robot_to_push_distance
