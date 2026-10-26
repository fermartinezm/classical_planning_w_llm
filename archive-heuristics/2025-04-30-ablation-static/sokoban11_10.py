from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def build_adjacency_graph(static_facts):
    """Builds an adjacency list representation of the grid from adjacent facts."""
    adj = {}
    all_locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            if loc1 not in adj:
                adj[loc1] = []
            adj[loc1].append((loc2, direction)) # Add loc2 reachable from loc1

            all_locations.add(loc1)
            all_locations.add(loc2)

    # Ensure all locations mentioned have an entry, even if no adjacencies listed
    for loc in all_locations:
        if loc not in adj:
            adj[loc] = []

    return adj, all_locations

def bfs_distance(start_loc, end_loc, obstacles, adjacencies):
    """
    Performs BFS to find the shortest path distance between start_loc and end_loc,
    avoiding locations in the obstacles set for intermediate steps.
    Returns -1 if no path exists.
    Assumes start_loc is not an obstacle.
    """
    if start_loc == end_loc:
        return 0

    if start_loc not in adjacencies or end_loc not in adjacencies:
        return -1 # Cannot reach invalid location

    # If the target itself is an obstacle, we cannot reach it (unless start is target, handled above)
    if end_loc in obstacles:
         return -1

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # Check neighbors based on the adjacency graph
        for neighbor_loc, _ in adjacencies.get(current_loc, []):
            # Neighbor must not be an obstacle for intermediate steps
            if neighbor_loc not in obstacles:
                if neighbor_loc == end_loc:
                    return dist + 1 # Found the end location

                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

    return -1 # No path found

def bfs_distance_multi_target(start_loc, target_locations, obstacles, adjacencies):
    """
    Performs BFS to find the shortest path distance from start_loc to any location
    in target_locations, avoiding locations in the obstacles set for intermediate steps.
    Returns -1 if no path exists.
    Assumes start_loc is not an obstacle.
    """
    if start_loc not in adjacencies:
        return -1

    # Filter target_locations to exclude obstacles - cannot end path in an obstacle
    valid_target_locations = {loc for loc in target_locations if loc not in obstacles}

    # Check if start is already a valid target
    if start_loc in valid_target_locations:
        return 0

    if not valid_target_locations:
        return -1 # No reachable target locations

    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # Check neighbors based on the adjacency graph
        for neighbor_loc, _ in adjacencies.get(current_loc, []):
            # Neighbor must not be an obstacle for intermediate steps
            if neighbor_loc not in obstacles:
                if neighbor_loc in valid_target_locations:
                    return dist + 1 # Found a valid target location

                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

    return -1 # No path found to any valid target location


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated costs for each box
    to reach its goal location. The estimated cost for a single box is the sum of:
    1. The shortest path distance for the box to its goal location, ignoring dynamic obstacles (robot, other boxes). This estimates the minimum number of pushes needed in an ideal scenario.
    2. The shortest path distance for the robot to reach *any* clear location adjacent to the box's current position, avoiding other boxes and permanent obstacles. This estimates the robot movement cost to get into a position to push the box.

    # Assumptions
    - Each box specified in the goal has a unique goal location.
    - The grid structure is defined by `adjacent` facts. Locations not connected by `adjacent` facts are walls/impassable.
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds an adjacency graph of the grid from the static `adjacent` facts.
    - Identifies all valid locations in the grid.
    - Identifies permanent obstacles (locations mentioned in the domain but not in the adjacency graph).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the robot and all boxes from the state.
    2. Get the set of locations that are currently `clear` from the state.
    3. Initialize the total heuristic cost to 0.
    4. For each box specified in the goal:
       a. Get the box's current location and its goal location.
       b. If the box is already at its goal, continue to the next box (cost is 0 for this box).
       c. Calculate the shortest path distance for the box from its current location to its goal location using BFS. For this BFS, consider only the permanent walls (locations not in the adjacency graph) as obstacles. This gives a lower bound on the number of pushes needed, ignoring dynamic obstacles.
       d. Identify all locations adjacent to the box's current location that are currently `clear` in the state. These are potential locations for the robot to stand to push the box.
       e. Calculate the shortest path distance for the robot from its current location to *any* of the clear adjacent locations identified in step 4d. For this BFS, consider locations occupied by *other* boxes and permanent walls as obstacles. The box itself is not an obstacle for the robot trying to reach an adjacent square.
       f. If either BFS returns -1 (no path), it indicates an unsolvable state or a deadlock for this box. Return a large value (e.g., float('inf')) for the total heuristic.
       g. Add the box-to-goal distance (from 4c) and the robot-to-push-spot distance (from 4e) to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the grid graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming goal is always (at box_name loc_name)
            if predicate == "at" and len(args) == 2 and args[0].startswith('box'):
                box, location = args[0], args[1]
                self.goal_locations[box] = location

        # Build the adjacency graph and get all traversable locations
        self.adjacencies, self.traversable_locations = build_adjacency_graph(static_facts)

        # Identify permanent obstacles (locations mentioned but not traversable)
        self.permanent_obstacles = self.get_permanent_obstacles(task.facts)


    def get_permanent_obstacles(self, task_facts):
        """
        Identifies locations that are not part of the traversable grid defined by adjacent facts.
        Assumes any location literal present in any possible ground fact in the domain
        is either traversable via adjacent facts or is a permanent obstacle (wall).
        """
        all_locations_in_domain = set()
        # task_facts is a set of strings representing all possible ground facts
        # We need to parse all these facts to find all location literals
        for fact_str in task_facts:
             parts = get_parts(fact_str)
             for part in parts:
                  # Simple check for location literals starting with 'loc_'
                  if part.startswith('loc_'):
                       all_locations_in_domain.add(part)

        # Locations mentioned in the domain but not in the adjacency graph are permanent obstacles
        permanent_obstacles = all_locations_in_domain - self.traversable_locations
        return permanent_obstacles


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Find robot and box locations and clear locations
        robot_loc = None
        box_locations = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                box_locations[box] = loc
            elif parts[0] == 'clear':
                 clear_locations.add(parts[1])

        # If robot location is unknown or no boxes found (shouldn't happen in valid state)
        if robot_loc is None or not box_locations:
             return float('inf') # Indicate unsolvable

        total_heuristic = 0

        # Calculate heuristic for each box that needs to reach a goal
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # If a box from the goal is not in the state, something is wrong or unsolvable
            if current_box_loc is None:
                 return float('inf')

            # If box is already at goal, it contributes 0 to the heuristic
            if current_box_loc == goal_loc:
                continue

            # --- Calculate box-to-goal distance (min pushes needed) ---
            # BFS for the box path treats permanent obstacles as walls.
            # Dynamic obstacles (other boxes, robot) are ignored for this part,
            # as we estimate the ideal path for the box itself on the static grid.
            box_obstacles_for_grid_dist = self.permanent_obstacles
            box_dist = bfs_distance(current_box_loc, goal_loc, box_obstacles_for_grid_dist, self.adjacencies)

            if box_dist == -1:
                 # Box cannot reach goal even ignoring dynamic obstacles. Likely unsolvable.
                 return float('inf')

            # --- Calculate robot-to-push-spot distance ---
            # Find clear locations adjacent to the box's current location.
            # These are potential spots for the robot to stand to push the box.
            potential_push_spots = {
                neighbor_loc for neighbor_loc, _ in self.adjacencies.get(current_box_loc, [])
                if neighbor_loc in clear_locations # Location must be clear in current state
            }

            # If no clear adjacent spots, the box is currently unpushable. Penalize heavily.
            # This is a strong indicator of a bad state or deadlock for this box.
            if not potential_push_spots:
                 return float('inf')

            # BFS for the robot path to reach any of the potential push spots.
            # Robot avoids permanent obstacles and locations occupied by *other* boxes.
            # Robot can move into clear locations.
            # The set of locations occupied by other boxes:
            other_boxes_locations = {loc for b, loc in box_locations.items() if b != box}
            # Robot obstacles: permanent walls + locations occupied by other boxes.
            robot_obstacles_for_path = self.permanent_obstacles | other_boxes_locations

            min_robot_dist_to_push_spot = bfs_distance_multi_target(robot_loc, potential_push_spots, robot_obstacles_for_path, self.adjacencies)

            if min_robot_dist_to_push_spot == -1:
                 # Robot cannot reach any push spot for this box, avoiding other boxes and walls.
                 return float('inf')

            # Heuristic for this box:
            # (Minimum pushes needed ignoring dynamic obstacles) + (Minimum robot moves to get into a push position now)
            total_heuristic += box_dist + min_robot_dist_to_push_spot

        return total_heuristic
