from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# --- Helper Functions ---

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_graph(static_facts):
    """Builds an adjacency list graph from 'adjacent' facts."""
    adj_list = {}
    locations = set()

    # First pass: Collect all locations mentioned in adjacent facts
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'adjacent' and len(parts) == 4:
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            locations.add(loc1)
            locations.add(loc2)

    # Initialize adjacency list for all found locations
    for loc in locations:
        adj_list[loc] = set()

    # Second pass: Populate adjacency list
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'adjacent' and len(parts) == 4:
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            adj_list[loc1].add(loc2)

    return adj_list

def shortest_path_distance(start_loc, end_loc, blocked_locs, adj_list):
    """
    Performs BFS to find the shortest path distance between two locations
    on the grid graph, avoiding blocked locations.

    Args:
        start_loc (str): The starting location.
        end_loc (str): The target location.
        blocked_locs (set): A set of locations that cannot be visited during the path.
        adj_list (dict): The adjacency list representation of the location graph.

    Returns:
        int or float('inf'): The shortest distance, or infinity if the end_loc
                             is unreachable from the start_loc given the blocked_locs.
    """
    if start_loc == end_loc:
        return 0

    # Cannot start from a blocked location (should not happen for valid start states)
    if start_loc in blocked_locs:
         return float('inf')

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # Ensure current_loc is a valid key in adj_list
        if current_loc not in adj_list:
             continue # Should not happen if build_graph is correct and locations are from it

        for neighbor in adj_list.get(current_loc, set()): # Use .get for safety
            # A location is blocked if it's in the blocked_locs set AND it's not the target location.
            # We allow reaching the target location even if it's in the blocked set,
            # as the goal might be a location currently occupied by an obstacle that will move.
            is_blocked = neighbor in blocked_locs and neighbor != end_loc

            if not is_blocked and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Target not reachable

# --- Sokoban Heuristic Class ---

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It is calculated as the sum of the shortest path distances for each box
    from its current location to its goal location (considering other objects as obstacles),
    plus the minimum shortest path distance for the robot to reach any location
    adjacent to any box that still needs to be moved (considering other boxes as obstacles).

    # Assumptions
    - The environment is a grid where movement is defined by 'adjacent' facts.
    - Locations occupied by other boxes or the robot are obstacles for box movement.
    - Locations occupied by other boxes are obstacles for robot movement.
    - The heuristic assumes that moving a box one step towards its goal requires
      the robot to be in a specific adjacent position, and subsequent pushes
      in the same direction don't require significant extra robot movement setup.
      The cost of getting the robot into position for the *first* push for *any*
      remaining box is added.
    - The heuristic does not explicitly detect or penalize deadlocks (e.g., boxes
      pushed into corners). Unreachable goals are detected via BFS returning infinity.

    # Heuristic Initialization
    - Builds an adjacency list graph representation of the locations based on
      'adjacent' facts from the static information.
    - Stores the goal locations for each box.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current location of the robot and all boxes from the state.
    2.  Identify the goal location for each box from the task goals.
    3.  Initialize `total_box_distance = 0` and `min_robot_distance_to_push_pos = infinity`.
    4.  Create a set of locations occupied by all boxes (`all_box_locs`).
    5.  For each box specified in the goals:
        a.  Find the box's current location (`current_box_loc`). If the box is not found in the state or is already at its goal location, skip it.
        b.  Add this box to a list of boxes that still need to be moved (`boxes_to_move`).
        c.  Create a set of locations blocked for this specific box's movement: the robot's current location and the locations of all *other* boxes.
        d.  Calculate the shortest path distance from the box's current location to its goal location using BFS on the location graph, avoiding the blocked locations.
        e.  If the goal is unreachable for this box (BFS returns infinity), the state is likely unsolvable; return infinity.
        f.  Add this distance to `total_box_distance`.
    6.  If `total_box_distance` is 0, all boxes are at their goals; return 0.
    7.  Create a set of locations blocked for robot movement: the locations of all boxes (`all_box_locs`).
    8.  For each box in the list of boxes that still need to be moved:
        a.  Find the box's current location (`current_box_loc`).
        b.  For each location adjacent to `current_box_loc` (according to the graph):
            i.  Calculate the shortest path distance from the robot's current location to this adjacent location using BFS, avoiding the blocked locations (other boxes).
            ii. Update `min_robot_distance_to_push_pos` with the minimum distance found so far across all adjacent locations of all boxes that need moving.
    9.  If `min_robot_distance_to_push_pos` is still infinity, the robot cannot reach any location adjacent to any box that needs moving; return infinity.
    10. The total heuristic value is `total_box_distance + min_robot_distance_to_push_pos`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and storing
        goal locations.
        """
        # Assuming task object has attributes: goals, static
        self.goals = task.goals
        static_facts = task.static

        # Build the adjacency list graph from adjacent facts
        self.adj_list = build_graph(static_facts)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                # Goal is (at ?o - box ?l - location)
                box, location = args
                self.goal_locations[box] = location
            # Ignore other types of goal predicates if any exist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # frozenset of fact strings

        # 1. Identify current locations
        robot_loc = None
        box_locations = {}
        # current_locations_set = set() # Not strictly needed for this logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
                # robot_loc is blocked for boxes
            elif predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Check if obj is a box whose goal is specified
                if obj in self.goal_locations:
                    box_locations[obj] = loc
                    # Box locations are blocked for robot and other boxes

        # Ensure robot location is found (should always be the case in a valid state)
        if robot_loc is None:
             return float('inf') # State is malformed or unsolvable

        # 3. Initialize heuristic components
        total_box_distance = 0
        min_robot_distance_to_push_pos = float('inf')
        boxes_to_move = []

        # 4. & 5. Calculate total box distance
        # Locations occupied by other boxes or the robot are blocked for a specific box's movement
        all_box_locs = set(box_locations.values())

        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # If box is not in the state or already at goal, skip
            if current_box_loc is None or current_box_loc == goal_loc:
                continue

            boxes_to_move.append(box)

            # Blocked locations for this specific box: robot + all *other* boxes
            blocked_for_this_box = {robot_loc} | {loc for b, loc in box_locations.items() if b != box}

            # Calculate shortest path for the box
            box_dist = shortest_path_distance(current_box_loc, goal_loc, blocked_for_this_box, self.adj_list)

            # If any box cannot reach its goal, the state is likely unsolvable
            if box_dist == float('inf'):
                return float('inf')

            total_box_distance += box_dist

        # 6. Check if goal is reached
        if total_box_distance == 0:
            return 0

        # 7. & 8. Calculate minimum robot distance to a push position
        # Locations occupied by boxes are blocked for robot movement
        blocked_for_robot = all_box_locs

        # If no boxes need moving, robot distance component is 0 (already handled by total_box_distance == 0)
        if not boxes_to_move:
             return 0 # Should not be reached if total_box_distance > 0

        for box in boxes_to_move:
            current_box_loc = box_locations[box]

            # Find locations adjacent to the current box location
            # Ensure current_box_loc is a valid key in adj_list before accessing
            adjacent_to_box = self.adj_list.get(current_box_loc, set())

            for adj_loc in adjacent_to_box:
                # Calculate shortest path for the robot to reach this adjacent location
                # Robot cannot move into a box location (handled by blocked_for_robot)
                # The shortest_path_distance function allows reaching the end_loc even if blocked,
                # which is correct if adj_loc is the robot's current location.
                robot_dist = shortest_path_distance(robot_loc, adj_loc, blocked_for_robot, self.adj_list)

                min_robot_distance_to_push_pos = min(min_robot_distance_to_push_pos, robot_dist)

        # 9. Check if robot can reach any push position
        if min_robot_distance_to_push_pos == float('inf'):
            return float('inf') # Robot cannot reach any adjacent square of any box to move

        # 10. Total heuristic value
        total_heuristic = total_box_distance + min_robot_distance_to_push_pos

        return total_heuristic
