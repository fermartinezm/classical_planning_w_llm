from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Helper to parse PDDL fact string into a list of parts."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def bfs(start_node_id, adj_list, node_count):
    """
    Performs BFS from a start node on a graph represented by an adjacency list.
    Returns a dictionary mapping reachable node IDs to their distances from the start node.
    """
    distances = {}
    queue = deque([start_node_id])
    distances[start_node_id] = 0

    # Use a set for visited nodes for faster lookup
    visited = {start_node_id}

    while queue:
        u = queue.popleft()
        dist_u = distances[u]

        # Check if node u exists in the adjacency list keys
        if u in adj_list:
            for v in adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    distances[v] = dist_u + 1
                    queue.append(v)
    return distances

def opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen for valid directions

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        two main components for each box that is not yet at its goal location:
        1. The minimum number of push actions required to move the box from its
           current location to its goal location, assuming no obstacles. This is
           calculated as the shortest path distance between the locations on the
           full grid graph.
        2. The minimum number of robot movement actions required to get the robot
           into a position from which it can push the box one step towards its goal.
           This distance is calculated on the graph of currently clear locations.

    Assumptions:
        - The location names follow a grid-like pattern (e.g., loc_row_col),
          although the heuristic relies only on the `adjacent` facts to build
          the graph, not on parsing row/column numbers.
        - The goal state specifies the target location for each box using `(at boxX locY)`.
        - The graph defined by `adjacent` facts is connected, at least for locations
          relevant to boxes and goals.
        - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
        In the constructor (`__init__`), the heuristic precomputes static information:
        - It builds a graph representation of the locations based on `adjacent` facts.
          Locations are mapped to integer IDs.
        - It stores adjacency information, including the direction of adjacency
          (`self.adj_dirs`: l1 -> {l2: dir}) and the reverse mapping
          (`self.adj_locs_by_dir`: l1 -> {dir: l2}).
        - It identifies the goal location for each box from the task's goals.
        - It computes all-pairs shortest path distances between all locations on the
          full static grid graph using BFS. This `self.box_dist_map` is used to estimate
          the minimum number of pushes required for a box.

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
        1. Get the current state, robot's location, and each box's location by parsing
           the state facts.
        2. Identify which locations are currently clear based on the `(clear ?l)` facts
           in the state.
        3. Build a dynamic graph representing locations the robot can move to. This
           graph includes only locations that are currently clear. The robot's current
           location is also considered clear for movement purposes (as the starting point).
        4. Perform a BFS from the robot's current location on the clear graph to
           compute the shortest distance from the robot to all reachable clear locations (`robot_dist_map`).
        5. Initialize the total heuristic value to 0.
        6. Iterate through each box specified in the goal state (`self.box_goal_locations`):
           a. Get the box's current location (`L_b`) from the state.
           b. If the box's location is not found in the state (shouldn't happen in Sokoban)
              or if the box is already at its goal location (`g_b`), its contribution is 0.
              Continue to the next box.
           c. If the box is not at its goal location:
              i. Get the precomputed shortest path distance from `L_b` to `g_b` on the
                 full static graph (`d_box = self.box_dist_map[L_b][g_b]`). If `g_b` is
                 unreachable from `L_b` in the static graph, add a large penalty (e.g., 1000)
                 and continue (the box is likely stuck).
              ii. Add `d_box` to the total heuristic. This estimates the minimum number
                  of pushes needed for this box.
              iii. Find potential robot push positions (`ValidPushPositions`). A location
                   `L_r` is a valid push position for box `b` at `L_b` towards goal `g_b` if:
                   - There is a direction `dir` such that `L_r` is adjacent to `L_b` in `dir`
                     and `L_b` is adjacent to `L_next` in `dir`.
                   - `L_next` (the location the box would move to) exists and is strictly
                     closer to `g_b` than `L_b` (using `self.box_dist_map`).
                   - `L_r` is currently clear (`L_r` in `clear_locations`).
                   - `L_r` is reachable by the robot (`L_r` in `robot_dist_map`).
              iv. Calculate the minimum robot distance to any location in `ValidPushPositions`.
              v. If `ValidPushPositions` is not empty, add this minimum distance to the
                 total heuristic.
              vi. If `ValidPushPositions` is empty (robot cannot currently reach a position
                  to push the box towards the goal along a shortest path step), estimate
                  robot cost as the minimum distance to any clear, reachable location
                  adjacent to the box. If no such adjacent location exists, add 0 for
                  the robot cost component for this box.
        7. Return the total heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Build location graph and mappings
        self.location_names = set()
        self.adj_dirs = {} # l1 -> {l2: dir}
        self.adj_locs_by_dir = {} # l1 -> {dir: l2}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.location_names.add(l1)
                self.location_names.add(l2)

                if l1 not in self.adj_dirs:
                    self.adj_dirs[l1] = {}
                self.adj_dirs[l1][l2] = direction

                if l1 not in self.adj_locs_by_dir:
                    self.adj_locs_by_dir[l1] = {}
                self.adj_locs_by_dir[l1][direction] = l2

        self.location_names = sorted(list(self.location_names)) # Ensure consistent order
        self.name_to_id = {name: i for i, name in enumerate(self.location_names)}
        self.id_to_name = {i: name for i, name in enumerate(self.location_names)}
        self.node_count = len(self.location_names)

        # Build adjacency list using IDs for BFS
        self.adj_list = {self.name_to_id[l1]: [] for l1 in self.location_names}
        for l1, neighbors in self.adj_dirs.items():
             u_id = self.name_to_id[l1]
             for l2 in neighbors:
                 v_id = self.name_to_id[l2]
                 self.adj_list[u_id].append(v_id)

        # 2. Store box goal locations
        self.box_goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.box_goal_locations[box] = location

        # 3. Precompute all-pairs shortest paths for box movement (full graph)
        self.box_dist_map = {}
        for start_name in self.location_names:
            start_id = self.name_to_id[start_name]
            distances = bfs(start_id, self.adj_list, self.node_count)
            self.box_dist_map[start_name] = {self.id_to_name[node_id]: dist for node_id, dist in distances.items()}

    def __call__(self, node):
        state = node.state

        # 1. Get current locations and clear locations
        robot_location = None
        box_locations = {}
        clear_locations = set()

        # Parse state facts efficiently
        for fact in state:
            if fact.startswith('(at-robot '):
                 robot_location = fact[len('(at-robot '):-1]
            elif fact.startswith('(at box'):
                 parts = get_parts(fact)
                 box_locations[parts[1]] = parts[2]
            elif fact.startswith('(clear '):
                 clear_locations.add(fact[len('(clear '):-1])

        if robot_location is None:
             # Should not happen in a valid Sokoban state
             return float('inf') # Indicate unreachable state

        # 2. Build dynamic clear graph for robot movement
        # The robot can only move between clear locations.
        clear_adj_list = {}
        # Initialize clear_adj_list with all clear locations as nodes
        for l1 in clear_locations:
             u_id = self.name_to_id[l1]
             clear_adj_list[u_id] = []

        # Add edges between clear locations
        for l1 in clear_locations:
             u_id = self.name_to_id[l1]
             if l1 in self.adj_dirs: # Check if l1 has neighbors in the static graph
                 for l2 in self.adj_dirs[l1]:
                     if l2 in clear_locations:
                         v_id = self.name_to_id[l2]
                         clear_adj_list[u_id].append(v_id)

        # 3. Compute robot distances on the clear graph
        robot_id = self.name_to_id[robot_location]
        # BFS starts from robot_id and explores reachable nodes in clear_adj_list
        robot_distances_by_id = bfs(robot_id, clear_adj_list, self.node_count)
        # Map distances back to location names for easier lookup
        robot_dist_map = {self.id_to_name[node_id]: dist for node_id, dist in robot_distances_by_id.items()}

        # 4. Calculate total heuristic
        total_heuristic = 0

        for box, g_b in self.box_goal_locations.items():
            L_b = box_locations.get(box)

            # If box location is unknown or box is at goal, skip
            if L_b is None or L_b == g_b:
                continue

            # Box distance component
            # Check if L_b and g_b are reachable from each other in the static graph
            if L_b not in self.box_dist_map or g_b not in self.box_dist_map[L_b]:
                 # Box is in a location unreachable from its goal in the static graph - likely a dead end
                 # Assign a large penalty.
                 total_heuristic += 1000 # Penalty for potentially stuck box
                 continue # Cannot calculate meaningful box or robot distance

            d_box = self.box_dist_map[L_b][g_b]
            total_heuristic += d_box

            # Robot distance component
            valid_push_locs = []
            # Iterate through all possible push directions
            for push_dir in ['up', 'down', 'left', 'right']:
                # L_next is the location the box would move to if pushed in push_dir
                L_next = self.adj_locs_by_dir.get(L_b, {}).get(push_dir)

                # Check if L_next exists and is closer to the goal
                # Also check if L_next is a valid location in the box_dist_map (should be if it exists)
                if L_next and L_b in self.box_dist_map and L_next in self.box_dist_map[L_b] and self.box_dist_map[L_next][g_b] < d_box:
                    # This push direction is towards the goal
                    # L_r is the required robot location to push in this direction
                    required_robot_loc = self.adj_locs_by_dir.get(L_b, {}).get(opposite_direction(push_dir))

                    # Check if the required robot location exists, is clear, and reachable by the robot
                    if required_robot_loc and required_robot_loc in clear_locations and required_robot_loc in robot_dist_map:
                        valid_push_locs.append(required_robot_loc)

            min_robot_dist = float('inf')
            if valid_push_locs:
                min_robot_dist = min(robot_dist_map[L_pc] for L_pc in valid_push_locs)
            else:
                # Fallback: Robot cannot currently reach a position to push this box
                # towards the goal along a shortest path step.
                # Estimate robot cost as the minimum distance to any clear, reachable location adjacent to the box.
                adjacent_clear_reachable_locs = []
                if L_b in self.adj_dirs:
                    for L_adj in self.adj_dirs[L_b]:
                        if L_adj in clear_locations and L_adj in robot_dist_map:
                            adjacent_clear_reachable_locs.append(L_adj)

                if adjacent_clear_reachable_locs:
                     min_robot_dist = min(robot_dist_map[L_adj] for L_adj in adjacent_clear_reachable_locs)
                else:
                     # If no adjacent clear reachable location exists, the robot might be trapped
                     # or the box is completely surrounded by non-clear locations.
                     # Add 0 for the robot cost component for this box.
                     min_robot_dist = 0

            # Add robot cost component
            # Only add if a distance was found (min_robot_dist is not inf, which it won't be with the fallback)
            total_heuristic += min_robot_dist

        return total_heuristic
