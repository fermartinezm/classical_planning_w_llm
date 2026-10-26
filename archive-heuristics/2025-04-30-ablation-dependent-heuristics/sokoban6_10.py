import re
import collections
import math

# Helper function to parse a PDDL fact string
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into its predicate and arguments.
    e.g., '(at-robot loc_6_4)' -> ('at-robot', ['loc_6_4'])
    e.g., '(adjacent loc_4_2 loc_4_3 right)' -> ('adjacent', ['loc_4_2', 'loc_4_3', 'right'])
    """
    # Remove parentheses and split by space
    parts = fact_string.strip('()').split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function to build adjacency graph and get all locations
def build_adjacency_graph(static_facts):
    """
    Builds an undirected graph from 'adjacent' facts.
    Nodes are locations, edges connect adjacent locations.
    Returns the graph (dict: loc -> list of neighbor_loc) and a set of all locations.
    """
    adj_graph = collections.defaultdict(list) # Store list of neighbors
    all_locations = set()
    for fact_string in static_facts:
        predicate, args = parse_fact(fact_string)
        if predicate == 'adjacent':
            loc1, loc2, direction = args
            # Add edge in both directions for undirected graph traversal
            adj_graph[loc1].append(loc2)
            adj_graph[loc2].append(loc1)
            all_locations.add(loc1)
            all_locations.add(loc2)
    # Remove duplicates from adjacency lists (optional, BFS handles cycles)
    for loc in adj_graph:
        adj_graph[loc] = list(set(adj_graph[loc]))
    return adj_graph, all_locations

# Helper function to compute all-pairs shortest paths using BFS
def compute_all_pairs_shortest_paths(locations, adj_graph):
    """
    Computes shortest path distances between all pairs of locations
    in the adjacency graph using BFS.
    Returns a dictionary mapping (start_loc, end_loc) -> distance.
    """
    dist_map = {}
    for start_loc in locations:
        q = collections.deque([(start_loc, 0)])
        visited = {start_loc}
        dist_map[(start_loc, start_loc)] = 0

        while q:
            current_loc, dist = q.popleft()

            # Iterate through all neighbors
            for neighbor_loc in adj_graph.get(current_loc, []): # Use [] for safety if loc not in graph
                 # Check if neighbor_loc is a valid location and not visited in this BFS run
                 if neighbor_loc in locations and neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    dist_map[(start_loc, neighbor_loc)] = dist + 1
                    q.append((neighbor_loc, dist + 1))

    return dist_map

# Helper function to get adjacent locations from the graph
def get_adjacent_locations(location, adj_graph):
    """
    Returns a list of locations adjacent to the given location based on the graph.
    """
    return adj_graph.get(location, []) # Use [] for safety if loc not in graph


class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing two components:
    1.  The estimated cost to move each box to its goal location. This is approximated
        by the sum of the shortest path distances (in the grid graph) from each box's
        current location to its assigned goal location.
    2.  The estimated cost for the robot to get into a position to push a box. This
        is approximated by the minimum shortest path distance (in the grid graph)
        from the robot's current location to any location adjacent to any box that
        is not yet at its goal.

    Assumptions:
    -   The problem involves moving boxes to specific goal locations defined by '(at boxX loc_Y_Z)' facts in the goal state.
    -   Location names are consistent and represent nodes in a grid-like structure
        defined by 'adjacent' facts.
    -   The 'adjacent' facts define an undirected graph (if loc1 is adjacent to loc2,
        then loc2 is adjacent to loc1, possibly with a reverse direction). The
        implementation assumes this symmetry for distance calculations.
    -   Each box that needs to be moved has a unique goal location specified in the task goals.

    Heuristic Initialization:
    The heuristic is initialized with the planning task object. During initialization,
    it performs the following steps:
    1.  Parses 'adjacent' facts from the task's static information to build an
        adjacency graph representing the grid connectivity.
    2.  Identifies all unique locations present in the 'adjacent' facts.
    3.  Computes the shortest path distance between all pairs of locations in the
        grid graph using Breadth-First Search (BFS). These distances are stored
        in a dictionary for quick lookup during heuristic computation.
    4.  Parses the task's goal facts to identify the target location for each box.
        This mapping from box name to goal location is stored.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1.  Identify the current location of the robot and each box by parsing the state facts.
        Only boxes that have a goal location defined in the task are considered.
    2.  Identify which of these boxes are not yet at their assigned goal locations. These are the 'active' boxes.
    3.  If there are no active boxes, the state is a goal state for boxes, and the heuristic value is 0.
    4.  If there are active boxes, calculate the box-to-goal component:
        a.  For each active box, find the pre-calculated shortest path distance from its current location to its specific goal location using the `dist_map`.
        b.  Sum these distances. This sum is the box-to-goal heuristic component. If any box goal is unreachable in the static graph (distance is infinity), the sum will be infinity.
    5.  Calculate the robot-to-box component:
        a.  Find the minimum shortest path distance from the robot's current location to *any* location that is adjacent to *any* of the active boxes. This uses the `dist_map` and `get_adjacent_locations`. This estimates the cost for the robot to reach a position from which it can start pushing an un-goboxed box. If no such adjacent location is reachable in the static graph, this distance will be infinity.
    6.  The total heuristic value is the sum of the box-to-goal component and the robot-to-box component. If either component is infinity, the total heuristic is infinity, indicating a likely unsolvable state (in the static grid).
    """
    def __init__(self, task):
        # 1. Build adjacency graph and get all locations
        self.adj_graph, self.all_locations = build_adjacency_graph(task.static)

        # 2. Compute all-pairs shortest paths
        self.dist_map = compute_all_pairs_shortest_paths(self.all_locations, self.adj_graph)

        # 3. Identify box goal locations
        self.box_goals = {} # box -> goal_loc
        for fact_string in task.goals:
            predicate, args = parse_fact(fact_string)
            if predicate == 'at' and len(args) == 2:
                box, goal_loc = args
                self.box_goals[box] = goal_loc

        # Store the list of boxes we care about (those with goals)
        self.goal_boxes = list(self.box_goals.keys())


    def __call__(self, state):
        # 1. Identify robot and box locations in the current state
        robot_loc = None
        box_locations = {} # box -> loc

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'at-robot' and len(args) == 1:
                robot_loc = args[0]
            elif predicate == 'at' and len(args) == 2:
                box, loc = args
                if box in self.goal_boxes: # Only track boxes that have goals
                    box_locations[box] = loc

        # 2. Identify active boxes (not at goal)
        active_boxes = []
        active_box_current_locs = []
        active_box_goal_locs = []

        # Ensure we only consider boxes that exist in the current state AND have a goal
        for box in self.goal_boxes:
             current_loc = box_locations.get(box)
             goal_loc = self.box_goals.get(box)

             # A box is 'active' if it exists in the state, has a goal, and is not at its goal
             if current_loc is not None and goal_loc is not None and current_loc != goal_loc:
                 active_boxes.append(box)
                 active_box_current_locs.append(current_loc)
                 active_box_goal_locs.append(goal_loc)


        # 3. If no active boxes, it's a goal state (for boxes)
        if not active_boxes:
            return 0

        # 4. Calculate box-to-goal component
        total_box_dist = 0
        for i in range(len(active_boxes)):
             current_loc = active_box_current_locs[i]
             goal_loc = active_box_goal_locs[i]
             # Use pre-calculated grid distance. Get returns None if key not found.
             # Use math.inf if path doesn't exist in the static graph.
             # dist_map keys are (start, end). Ensure both start and end are in the map.
             dist = self.dist_map.get((current_loc, goal_loc), math.inf)
             total_box_dist += dist

        # If any box goal is unreachable in the static graph, the state is likely unsolvable
        if total_box_dist >= math.inf: # Check >= because sum of infs is inf
             return math.inf

        # 5. Calculate robot-to-box component
        min_robot_dist = math.inf
        # Robot needs to get adjacent to *any* active box
        for box in active_boxes:
            box_loc = box_locations[box]
            # Find locations adjacent to the box's current location in the static graph
            for adj_loc in get_adjacent_locations(box_loc, self.adj_graph):
                 # Check if robot can reach this adjacent location in the static graph
                 # dist_map keys are (start, end). Ensure both start and end are in the map.
                 dist = self.dist_map.get((robot_loc, adj_loc), math.inf)
                 min_robot_dist = min(min_robot_dist, dist)

        # If robot cannot reach any location adjacent to any active box in the static graph
        if min_robot_dist == math.inf:
             # This state is likely unsolvable in the static grid.
             return math.inf

        # 6. Total heuristic
        total_heuristic = total_box_dist + min_robot_dist

        # Ensure heuristic is 0 only for goal states.
        # We already returned 0 if active_boxes is empty.
        # If active_boxes is not empty, total_box_dist > 0 (since dist >= 1 for different locations).
        # min_robot_dist >= 0.
        # So total_heuristic > 0 for non-goal states (where boxes need moving).

        return total_heuristic
