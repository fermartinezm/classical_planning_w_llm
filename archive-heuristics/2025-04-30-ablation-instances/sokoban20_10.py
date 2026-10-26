# The code should be placed in a file named sokobanHeuristic.py
# It assumes the existence of a base class `Heuristic` with `__init__(self, task)` and `__call__(self, node)`
# If the base class is not provided, the class definition should be `class sokobanHeuristic:`

import collections
from fnmatch import fnmatch

# from heuristics.heuristic_base import Heuristic # Commented out as per instruction to provide only the code

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_distance(start, goal, graph, obstacles):
    """
    Computes the shortest path distance between start and goal locations
    on the grid graph, avoiding obstacles.
    Returns distance or float('inf') if no path.
    """
    if start == goal:
        return 0
    # Cannot start inside an obstacle for pathfinding
    if start in obstacles:
         return float('inf')

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        # Neighbors are locations reachable from current_loc via graph edges
        # graph stores loc -> [(neighbor, direction)]
        neighbors = graph.get(current_loc, [])

        for neighbor_loc, _ in neighbors:
            if neighbor_loc == goal:
                return dist + 1
            if neighbor_loc not in visited and neighbor_loc not in obstacles:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return float('inf') # No path found

def bfs_path(start, goal, graph, obstacles):
    """
    Computes the shortest path between start and goal locations
    on the grid graph, avoiding obstacles.
    Returns a list of locations representing the path (excluding start)
    or None if no path.
    """
    if start == goal:
        return []
    if start in obstacles:
        return None

    queue = collections.deque([(start, [])]) # (current_loc, path_so_far)
    visited = {start}

    while queue:
        current_loc, path = queue.popleft()

        neighbors = graph.get(current_loc, [])

        for neighbor_loc, _ in neighbors:
            if neighbor_loc == goal:
                return path + [neighbor_loc]
            if neighbor_loc not in visited and neighbor_loc not in obstacles:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, path + [neighbor_loc]))

    return None # No path found

def get_direction(loc1, loc2, graph):
    """
    Finds the direction from loc1 to loc2 using the adjacency graph.
    Assumes loc1 and loc2 are adjacent and present in the graph.
    Returns the direction string (e.g., 'right') or None.
    """
    neighbors = graph.get(loc1, [])
    for neighbor_loc, direction in neighbors:
        if neighbor_loc == loc2:
            return direction
    return None # Should not happen if loc1 and loc2 are adjacent and in graph

def get_opposite_direction(direction):
    """Returns the opposite direction string."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen with valid directions

class sokobanHeuristic: # Inherit from Heuristic if available
# class sokobanHeuristic(Heuristic): # Use this line if Heuristic base class is provided
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing contributions
    for each box that is not yet at its goal location. The contribution for a
    misplaced box is estimated as the minimum number of pushes required to move
    it to its goal, plus the cost for the robot to get into position to make
    the first push. Each push is estimated to cost 2 actions (1 push + ~1 robot
    repositioning for the next push), except for the initial setup.

    # Assumptions:
    - The grid structure is defined by `adjacent` facts.
    - Boxes need to be moved to specific goal locations.
    - The robot can only push boxes from one side (the side opposite the push direction).
    - Obstacles for robot movement are walls and other boxes.
    - Obstacles for box movement (in the heuristic calculation) are only walls.
    - The heuristic calculates shortest paths using BFS on the grid graph.

    # Heuristic Initialization
    - Extract the adjacency graph from static facts to enable pathfinding.
    - Extract the goal locations for each box from the task goals.
    - Store a mapping from location and direction to neighbor location.
    - Store a mapping from location and neighbor location to direction.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the robot's current location.
    3. Identify the current location of each box.
    4. Identify locations occupied by boxes (these are obstacles for robot movement).
    5. Pre-calculate shortest path distances for the robot from its current location to all reachable clear locations, avoiding other boxes.
    6. For each box that is not at its assigned goal location:
       a. Calculate the shortest path (`box_path`) from the box's current location (`l_b`) to its goal location (`g_b`) on the grid graph, considering only walls as obstacles for the box itself. If no path exists, the problem is likely unsolvable from this state, return infinity.
       b. The minimum number of pushes for this box is the length of `box_path`.
       c. Determine the location (`l_req`) where the robot must be to make the *first* push along this path. The first step of the box path is `l_b` -> `box_path[0]`. Find the direction (`dir`) of this step. The required robot location `l_req` is adjacent to `l_b` in the *opposite* direction of `dir`.
       d. Calculate the shortest path distance (`robot_reach_dist`) from the robot's current location to `l_req` using the pre-calculated distances. The robot must avoid locations occupied by *other* boxes. If `l_req` is occupied by another box, or unreachable by the robot, return infinity.
       e. Add `box_push_dist * 2 + robot_reach_dist` to the total heuristic cost. The `* 2` accounts for the push action and the estimated robot repositioning needed for the next push along a straight path.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Adjacency graph from static facts.
        - Goal locations for each box.
        - Mapping from location and direction to neighbor location.
        - Mapping from location and neighbor location to direction.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the adjacency graph: location -> [(neighbor_location, direction)]
        # Also build maps for quick lookup: (loc, dir) -> neighbor_loc and (loc, neighbor_loc) -> dir
        self.graph = collections.defaultdict(list)
        self.neighbor_map = {} # (loc, dir) -> neighbor_loc
        self.direction_map = {} # (loc, neighbor_loc) -> dir
        self.locations = set() # Set of all valid locations in the grid

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, d = parts[1], parts[2], parts[3]
                self.graph[l1].append((l2, d))
                self.neighbor_map[(l1, d)] = l2
                self.direction_map[(l1, l2)] = d
                self.locations.add(l1)
                self.locations.add(l2)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        if robot_location is None:
             # Should not happen in valid Sokoban states
             return float('inf')

        # Identify box locations and other obstacles for robot
        box_locations = {}
        robot_obstacles = set() # Locations occupied by boxes
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1].startswith("box"):
                 box, loc = parts[1], parts[2]
                 box_locations[box] = loc
                 robot_obstacles.add(loc)

        # Pre-calculate robot distances from its current location
        # Robot cannot move into locations occupied by boxes
        robot_dist_map = {}
        q = collections.deque([(robot_location, 0)])
        visited_robot = {robot_location}
        # Obstacles for robot BFS are locations occupied by *any* box
        robot_bfs_obstacles = robot_obstacles.copy()

        while q:
            current_loc, dist = q.popleft()
            robot_dist_map[current_loc] = dist

            # Neighbors are locations reachable from current_loc via graph edges
            neighbors = self.graph.get(current_loc, [])
            for neighbor_loc, _ in neighbors:
                if neighbor_loc not in visited_robot and neighbor_loc not in robot_bfs_obstacles:
                    visited_robot.add(neighbor_loc)
                    q.append((neighbor_loc, dist + 1))


        total_heuristic = 0

        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            if current_location is None or current_location == goal_location:
                continue # Box is at goal or not found

            # Calculate shortest path for the box to its goal
            # Obstacles for box pathfinding heuristic are locations not in the graph (walls)
            # We need the path to find the first step direction.
            # BFS for box path: obstacles are just locations not in self.locations (handled by graph)
            # The box cannot move into a location occupied by another box or the robot in reality,
            # but for this non-admissible heuristic, we simplify and only consider walls as obstacles
            # for the box's ideal path calculation.
            box_path = bfs_path(current_location, goal_location, self.graph, set()) # Empty set of obstacles for box path
            if box_path is None:
                # Box cannot reach its goal (e.g., trapped by walls)
                return float('inf') # Problem likely unsolvable

            box_push_dist = len(box_path) # Minimum pushes for this box in isolation

            # Determine the required robot location for the first push
            # First step of the box path is current_location -> box_path[0]
            first_step_loc = box_path[0]
            # Find the direction of the first step (current_location -> first_step_loc)
            push_direction = self.direction_map.get((current_location, first_step_loc))
            if push_direction is None:
                 # Should not happen if box_path is valid and graph is correct
                 return float('inf') # Error state

            # The robot must be adjacent to current_location in the *opposite* direction of the push
            # E.g., to push right from A to B, robot must be at R where (adjacent R A right)
            # This means A is right of R, so R is left of A. R is adjacent to A in direction 'left'.
            # 'left' is the opposite of 'right'.
            opposite_direction = get_opposite_direction(push_direction)
            if opposite_direction is None:
                 return float('inf') # Error state

            required_robot_location = self.neighbor_map.get((current_location, opposite_direction))

            if required_robot_location is None:
                 # This implies there is a wall behind the box in the direction the robot needs to be.
                 # The box cannot actually be pushed from current_location in the required direction.
                 # This state might be a dead end or invalid.
                 return float('inf') # Cannot make the first push

            # Calculate robot reach distance to the required pushing location
            # Use the pre-calculated robot_dist_map
            # The required_robot_location must not be occupied by another box.
            # The robot_dist_map already considers all box locations as obstacles.
            # So, if required_robot_location is in robot_dist_map, it means it's reachable
            # and not occupied by another box.
            robot_reach_dist = robot_dist_map.get(required_robot_location, float('inf'))

            if robot_reach_dist == float('inf'):
                 # Robot cannot reach the required pushing position (blocked by walls or other boxes)
                 return float('inf') # Problem likely unsolvable

            # Add contribution for this box
            # Cost = (pushes * cost_per_push) + cost_to_get_robot_into_position_for_first_push
            # Estimate cost_per_push = 2 (1 push + ~1 robot move to reposition)
            total_heuristic += box_push_dist * 2 + robot_reach_dist

        # Heuristic is 0 if all boxes are at their goals.
        # If total_heuristic is 0, it means the loop didn't add anything, which happens
        # only if all boxes are already at their goal locations.
        return total_heuristic
