import logging
from collections import deque

from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator are available in the execution environment
# from task import Operator, Task


# Helper function to parse location string like 'loc_X_Y' into (X, Y) tuple
def parse_loc_string(loc_str):
    """Parses a location string 'loc_X_Y' into a (row, col) tuple."""
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            logging.error(f"Could not parse location string: {loc_str}")
            return None
    else:
        logging.error(f"Unexpected location string format: {loc_str}")
        return None

# Helper function for opposite directions
def opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen in this domain

# BFS to compute all-pairs shortest paths for robot movement
def compute_all_pairs_shortest_paths(locations, robot_graph):
    """
    Computes shortest path distances between all pairs of locations
    in the robot graph using BFS.
    """
    dist = {}
    for start_loc in locations:
        dist[start_loc] = {}
        q = deque([(start_loc, 0)])
        visited = {loc: False for loc in locations}

        if start_loc in visited: # Handle cases where start_loc might not be in the initial locations list but is added later
             visited[start_loc] = True
             dist[start_loc][start_loc] = 0
        else:
             # This location was added later (e.g., from goals) and might not be in the graph structure keys yet
             # If it's not in the graph keys, it's isolated. Distances to/from it are inf.
             # We can skip BFS from here, distances will remain inf.
             continue


        while q:
            current_loc, current_d = q.popleft()

            if current_loc not in robot_graph:
                 # Location is isolated or not part of the defined graph
                 continue

            for neighbor_loc in robot_graph[current_loc]:
                if neighbor_loc in visited and not visited[neighbor_loc]:
                    visited[neighbor_loc] = True
                    dist[start_loc][neighbor_loc] = current_d + 1
                    q.append((neighbor_loc, current_d + 1))

        # Fill in infinity for unreachable locations
        for loc in locations:
             if loc not in dist[start_loc]:
                 dist[start_loc][loc] = float('inf')

    return dist

# BFS from goal locations on reversed graph to find paths towards goals
def compute_paths_towards_goals(goal_locations, adj_graph, locations):
    """
    Runs BFS from each goal location on the reversed adjacency graph
    to find shortest path distances from any location to the goal,
    the first step location on such a path, and the direction of that step.
    """
    box_dist_to_goal = {l: {} for l in locations} # dist[l][g] = shortest path distance from l to g
    first_step_towards_goal = {l: {} for l in locations} # first_step_towards_goal[l][g] = location adjacent to l on a shortest path from l to g
    direction_from_l_to_first_step = {l: {} for l in locations} # direction_from_l_to_first_step[l][g] = direction from l to first_step_towards_goal[l][g]

    # Build reversed graph for BFS from goals
    rev_adj_graph = {loc: {} for loc in locations}
    for l1, neighbors in adj_graph.items():
        for dir, l2 in neighbors.items():
            if l2 in rev_adj_graph: # Ensure l2 is a known location
                 rev_adj_graph[l2][opposite_direction(dir)] = l1 # Edge from l2 to l1 with opposite direction

    for goal_loc in goal_locations:
        # Initialize distances and path info for this goal
        for loc in locations:
             box_dist_to_goal[loc][goal_loc] = float('inf')
             first_step_towards_goal[loc][goal_loc] = None
             direction_from_l_to_first_step[loc][goal_loc] = None

        q = deque([(goal_loc, 0)])
        if goal_loc in box_dist_to_goal: # Ensure goal_loc is a known location
             box_dist_to_goal[goal_loc][goal_loc] = 0
        else:
             # Goal location is not in the list of known locations, skip BFS from here
             continue


        while q:
            current_loc, current_d = q.popleft()

            if current_loc not in rev_adj_graph:
                 # Location is isolated or not part of the defined graph
                 continue

            # Neighbors in the reversed graph are locations that can reach current_loc
            for dir_from_neighbor, neighbor_loc in rev_adj_graph[current_loc].items():
                 if neighbor_loc in box_dist_to_goal and box_dist_to_goal[neighbor_loc][goal_loc] == float('inf'):
                    # The direction from neighbor_loc to current_loc is the opposite of dir_from_neighbor
                    dir_to_current = opposite_direction(dir_from_neighbor)

                    box_dist_to_goal[neighbor_loc][goal_loc] = current_d + 1
                    # The first step from neighbor_loc towards goal_loc is current_loc
                    first_step_towards_goal[neighbor_loc][goal_loc] = current_loc
                    # The direction from neighbor_loc to the first step (current_loc) is dir_to_current
                    direction_from_l_to_first_step[neighbor_loc][goal_loc] = dir_to_current
                    q.append((neighbor_loc, current_d + 1))

    return box_dist_to_goal, first_step_towards_goal, direction_from_l_to_first_step


class sokobanHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the Sokoban planning domain.
    It estimates the cost to reach the goal state by summing two components:
    1. The estimated number of pushes required for each box not at its goal
       location to reach its goal location. This is calculated as the shortest
       path distance for the box on the grid graph, ignoring obstacles.
    2. The estimated number of robot moves required to get into a position
       to make the *first* push for *any* box that needs moving. This is
       calculated as the minimum shortest path distance from the robot's
       current location to any location adjacent to any box that needs moving,
       from which that box can be pushed towards its goal. This also ignores
       obstacles (other boxes, walls not in graph) on the robot's path.

    Assumptions:
    - The domain uses 'loc_X_Y' format for locations, where X and Y are integers.
    - The static facts define the grid connectivity using 'adjacent' predicates.
    - The goal is defined by '(at boxN loc_X_Y)' facts.
    - The grid defined by 'adjacent' facts is traversable for both the robot
      (moving into clear squares) and boxes (being pushed into clear squares),
      although the heuristic relaxes obstacle constraints.
    - The heuristic assumes a box needs to be pushed along a shortest path
      towards its goal based on the static graph.

    Heuristic Initialization:
    1. Parses all 'adjacent' static facts to build two graph representations:
       - `adj_graph`: A directed graph mapping (location, direction) to the
         neighboring location. Used to find the location resulting from a push
         in a specific direction, and to find the required robot position for
         a push.
       - `robot_graph`: An undirected graph representing traversable paths
         for the robot. Used for computing shortest path distances for robot
         movement.
    2. Collects all unique locations mentioned in 'adjacent' facts.
    3. Parses the goal facts to identify the target location for each box.
       Stores this in `self.box_goals`. Adds goal locations to the set of
       known locations if they are not already present from static facts.
    4. Precomputes all-pairs shortest path distances for robot movement using
       BFS on the `robot_graph`. Stores this in `self.robot_dist`.
    5. For each goal location, runs BFS on the *reversed* `adj_graph` to
       precompute:
       - `self.box_dist_to_goal`: Shortest path distance from any location
         to that goal location (representing estimated pushes).
       - `self.first_step_towards_goal`: The next location on a shortest path
         from a given location towards that goal.
       - `self.direction_from_l_to_first_step`: The direction from a given
         location to the `first_step_towards_goal`.
    6. Precomputes `self.push_from_map`: Maps (box_location, push_direction) to
       the required robot location to perform that push.

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
    1. Get the current state from the node.
    2. Parse the state to find the robot's current location (`l_r`) and the
       current location of each box (`box_locs`). Convert string locations
       to the internal tuple representation.
    3. Identify which boxes are not currently at their goal locations, considering
       only boxes that have a defined goal.
    4. If no boxes are not at their goals, the state is a goal state, return 0.
    5. Calculate the first component: Sum the precomputed shortest path
       distances (`self.box_dist_to_goal`) from the current location of each
       box that needs moving to its respective goal location. If any box's
       goal is unreachable from its current location, the heuristic is infinity.
    6. Calculate the second component: Find the minimum shortest path distance
       from the robot's current location (`l_r`) to a valid push location
       for any box that needs moving.
       - For each box `b` not at its goal:
         - Get its current location `l_b` and goal location `g_b`.
         - Use `self.first_step_towards_goal[l_b][g_b]` to find the next location
           `l_next` on a shortest path from `l_b` to `g_b`. If `l_next` is None
           (meaning `l_b` cannot reach `g_b`), skip this box (already handled
           by the infinity check in step 5).
         - Use `self.direction_from_l_to_first_step[l_b][g_b]` to find the
           direction `dir` from `l_b` to `l_next`.
         - Use `self.push_from_map[l_b][dir]` to find the required robot
           location `l_prev` to push box `b` from `l_b` in direction `dir`.
           If no such `l_prev` exists, this push is impossible based on static
           adjacencies; skip this push position.
         - Calculate `self.robot_dist[l_r][l_prev]`. If `l_prev` is unreachable
           from `l_r`, this distance is infinity.
         - Update the minimum robot distance found so far.
       - If no reachable push position was found for any box that needs moving,
         it means the robot cannot get into position to start pushing any
         required box towards its goal based on the static graph. The heuristic
         is infinity.
    7. The total heuristic value is the sum of the box distance component and
       the minimum robot positioning cost component.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals

        # --- Precomputation ---

        # 1. Parse static facts to build graphs and location maps
        self.locations = set()
        self.adj_graph = {} # (r, c) -> {direction: (nr, nc)}
        self.robot_graph = {} # (r, c) -> set((nr, nc))
        self.push_from_map = {} # (r, c) -> {push_dir: (robot_r, robot_c)} # Maps box_loc -> push_dir -> robot_loc

        for fact_str in task.static:
            if fact_str.startswith('(adjacent'):
                parts = fact_str.strip(')').split()
                if len(parts) == 4:
                    loc1_str = parts[1]
                    loc2_str = parts[2]
                    dir = parts[3]

                    loc1 = parse_loc_string(loc1_str)
                    loc2 = parse_loc_string(loc2_str)

                    if loc1 is None or loc2 is None:
                        logging.warning(f"Skipping invalid adjacent fact: {fact_str}")
                        continue

                    self.locations.add(loc1)
                    self.locations.add(loc2)

                    if loc1 not in self.adj_graph:
                        self.adj_graph[loc1] = {}
                    self.adj_graph[loc1][dir] = loc2

                    if loc1 not in self.robot_graph:
                        self.robot_graph[loc1] = set()
                    self.robot_graph[loc1].add(loc2)

                    if loc2 not in self.robot_graph:
                        self.robot_graph[loc2] = set()
                    self.robot_graph[loc2].add(loc1) # Robot graph is undirected
                else:
                    logging.warning(f"Skipping invalid adjacent fact format: {fact_str}")


        # Ensure all locations mentioned in adj_graph are keys in graph structures
        for loc in list(self.locations): # Iterate over a copy as we might add
             if loc not in self.robot_graph:
                  self.robot_graph[loc] = set()
             if loc not in self.adj_graph:
                  self.adj_graph[loc] = {} # Location might be mentioned but have no outgoing edges

        # Build push_from_map: Iterate through adj_graph (l_prev -> l_b in dir)
        # This means robot at l_prev can push box at l_b in dir
        for l_prev, neighbors in self.adj_graph.items():
            for dir, l_b in neighbors.items():
                if l_b not in self.push_from_map:
                    self.push_from_map[l_b] = {}
                self.push_from_map[l_b][dir] = l_prev


        # 3. Parse goal facts to get box goals and goal locations
        self.box_goals = {} # box_name -> (gr, gc)
        self.goal_locations = set()
        for goal_fact in self.goals:
            if goal_fact.startswith('(at box'):
                parts = goal_fact.strip(')').split()
                if len(parts) == 3:
                    box_name = parts[1]
                    goal_loc_str = parts[2]
                    goal_loc = parse_loc_string(goal_loc_str)
                    if goal_loc:
                        self.box_goals[box_name] = goal_loc
                        self.goal_locations.add(goal_loc)
                else:
                    logging.warning(f"Skipping invalid goal fact format: {goal_fact}")

        # Add goal locations to self.locations if they are not already there
        all_relevant_locations = list(self.locations | self.goal_locations)
        self.locations = all_relevant_locations # Use this comprehensive list for BFS

        # Ensure goal locations are in graph structures if they weren't from static facts
        for g_loc in self.goal_locations:
             if g_loc not in self.adj_graph: self.adj_graph[g_loc] = {}
             if g_loc not in self.robot_graph: self.robot_graph[g_loc] = set()
             if g_loc not in self.push_from_map: self.push_from_map[g_loc] = {}


        # 2. Precompute robot distances (using the comprehensive list of locations)
        self.robot_dist = compute_all_pairs_shortest_paths(self.locations, self.robot_graph)

        # 4. Precompute box path info towards goals (BFS from goals on reversed graph)
        self.box_dist_to_goal, self.first_step_towards_goal, self.direction_from_l_to_first_step = \
            compute_paths_towards_goals(list(self.goal_locations), self.adj_graph, self.locations)


    def __call__(self, node):
        """
        Computes the Sokoban heuristic for the given state.
        """
        state = node.state

        # 1. Parse current state
        l_r = None
        box_locs = {} # box_name -> (br, bc)
        # clear_locs = set() # Not used in this heuristic calculation

        for fact_str in state:
            if fact_str.startswith('(at-robot'):
                parts = fact_str.strip(')').split()
                if len(parts) == 2:
                    l_r_str = parts[1]
                    l_r = parse_loc_string(l_r_str)
                else:
                    logging.warning(f"Skipping invalid at-robot fact format: {fact_str}")
            elif fact_str.startswith('(at box'):
                parts = fact_str.strip(')').split()
                if len(parts) == 3:
                    box_name = parts[1]
                    loc_str = parts[2]
                    loc = parse_loc_string(loc_str)
                    if loc:
                        box_locs[box_name] = loc
                else:
                    logging.warning(f"Skipping invalid at box fact format: {fact_str}")
            # elif fact_str.startswith('(clear'):
            #      parts = fact_str.strip(')').split()
            #      if len(parts) == 2:
            #         loc_str = parts[1]
            #         loc = parse_loc_string(loc_str)
            #         if loc:
            #             clear_locs.add(loc)
            #      else:
            #         logging.warning(f"Skipping invalid clear fact format: {fact_str}")


        # Ensure robot location is found and is a valid parsed location in the graph
        if l_r is None or l_r not in self.locations:
             logging.error(f"Robot location {l_r} not found or invalid in state/graph.")
             return float('inf')


        # 3. Identify boxes not at goals (only considering boxes with defined goals)
        boxes_to_move = [b for b, l_b in box_locs.items() if b in self.box_goals and l_b != self.box_goals[b]]

        # 4. If no boxes are not at goals, return 0
        if not boxes_to_move:
            return 0

        # 5. Calculate box-goal distance sum
        box_dist_sum = 0
        for b in boxes_to_move:
            l_b = box_locs[b]
            g_b = self.box_goals[b]

            # Ensure box location and goal location are in the known locations
            if l_b not in self.locations or g_b not in self.locations:
                 logging.warning(f"Box location {l_b} or goal location {g_b} not in graph locations.")
                 return float('inf')

            dist = self.box_dist_to_goal[l_b][g_b]
            if dist == float('inf'):
                 # Box cannot reach its goal based on the static graph
                 return float('inf')
            box_dist_sum += dist

        # 6. Calculate robot positioning cost
        min_robot_dist = float('inf')
        robot_reachable_push_pos_found = False

        for b in boxes_to_move:
            l_b = box_locs[b]
            g_b = self.box_goals[b]

            # Find the required robot location to push box b from l_b towards g_b
            # Need the first step location l_next on a shortest path from l_b to g_b
            # Check if l_b can reach g_b (dist != inf handled above, but first_step might be None)
            if l_b not in self.first_step_towards_goal or g_b not in self.first_step_towards_goal[l_b]:
                 # Should not happen if box_dist_sum is finite, but defensive check
                 continue

            l_next = self.first_step_towards_goal[l_b][g_b]

            # If l_next is None, it means l_b cannot reach g_b (already handled by box_dist_sum check)
            if l_next is None:
                 continue

            # Find the direction from l_b to l_next
            if l_b not in self.direction_from_l_to_first_step or g_b not in self.direction_from_l_to_first_step[l_b]:
                 # Should not happen if l_next is not None
                 continue

            dir = self.direction_from_l_to_first_step[l_b][g_b]

            # Find the required robot location l_prev to push box b from l_b in direction dir
            if l_b in self.push_from_map and dir in self.push_from_map[l_b]:
                l_prev = self.push_from_map[l_b][dir]

                # Calculate robot distance to this required push location
                if l_r in self.robot_dist and l_prev in self.robot_dist[l_r]:
                    robot_dist_to_l_prev = self.robot_dist[l_r][l_prev]
                    if robot_dist_to_l_prev != float('inf'):
                         min_robot_dist = min(min_robot_dist, robot_dist_to_l_prev)
                         robot_reachable_push_pos_found = True
                # else: l_r or l_prev is not in the robot graph (unreachable)
            # else: l_b cannot be pushed in that direction (no adjacent l_prev in static facts)

        # If no reachable push position was found for any box that needs moving,
        # it might indicate a dead end or an issue with graph connectivity.
        # Return infinity in this case.
        if not robot_reachable_push_pos_found:
             return float('inf')


        # 7. Total heuristic value
        h = box_dist_sum + min_robot_dist

        return h

