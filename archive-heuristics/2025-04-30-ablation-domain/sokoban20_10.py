from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def get_opposite_direction(direction):
    """Returns the opposite direction string."""
    if direction == 'right': return 'left'
    if direction == 'left': return 'right'
    if direction == 'down': return 'up'
    if direction == 'up': return 'down'
    return None # Should not happen

def get_neighbor_in_direction(loc_str, direction, graph):
    """Returns the neighbor location string in the given direction, or None."""
    neighbors = graph.get(loc_str, [])
    for neighbor_loc, dir in neighbors:
        if dir == direction:
            return neighbor_loc
    return None

def build_graph(static_facts):
    """Builds adjacency list graph from adjacent facts."""
    graph = {}
    reverse_graph = {}
    all_locs = set()

    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            all_locs.add(loc1)
            all_locs.add(loc2)

            if loc1 not in graph:
                graph[loc1] = []
            graph[loc1].append((loc2, direction))

            # Build reverse graph
            opp_dir = get_opposite_direction(direction)
            if loc2 not in reverse_graph:
                reverse_graph[loc2] = []
            reverse_graph[loc2].append((loc1, opp_dir))

    # Ensure all locations mentioned are keys in both graphs, even if they have no neighbors listed
    for loc in all_locs:
        if loc not in graph: graph[loc] = []
        if loc not in reverse_graph: reverse_graph[loc] = []

    return graph, reverse_graph

def bfs_distances(start_loc_str, graph, blocked_locs):
    """
    Performs BFS from start_loc_str on the graph, avoiding blocked_locs.
    Returns a dictionary of distances to all reachable locations.
    """
    distances = {}
    if start_loc_str in blocked_locs:
         return distances # Start is blocked

    queue = collections.deque([(start_loc_str, 0)])
    visited = {start_loc_str}
    distances[start_loc_str] = 0

    while queue:
        current_loc, dist = queue.popleft()

        # Check neighbors from the graph
        for neighbor_loc, _ in graph.get(current_loc, []):
            if neighbor_loc not in visited and neighbor_loc not in blocked_locs:
                visited.add(neighbor_loc)
                distances[neighbor_loc] = dist + 1
                queue.append((neighbor_loc, dist + 1))

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing:
    1. The minimum number of pushes required for each misplaced box to reach its goal location (calculated as shortest path distance on the grid graph ignoring obstacles).
    2. The minimum distance the robot needs to travel from its current location to a position from which it can make the *first* push for any of the misplaced boxes.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Locations are named `loc_R_C`.
    - A box can be pushed one step if the robot is in the adjacent location behind it and the target location is clear.
    - The cost of moving the robot one step is 1. The cost of pushing a box one step (which also moves the robot) is 1.
    - Deadlocks are detected if a box cannot reach its goal or the robot cannot reach a pushing position, returning infinity.

    # Heuristic Initialization
    - Parses the `adjacent` facts to build a graph representation of the grid and its reverse.
    - Identifies the goal location for each box from the task goals.
    - Pre-calculates the shortest path distance from every location to every box's goal location on the grid graph, ignoring obstacles (using BFS on the reverse graph from the goal).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes by parsing the state facts.
    2. Identify locations that are currently clear.
    3. Determine which locations are blocked for robot movement (those not clear, not the robot's current location, and locations occupied by boxes).
    4. Identify the goal location for each box from the pre-calculated goals.
    5. Determine which boxes are not currently at their goal locations (misplaced boxes).
    6. If no boxes are misplaced, the heuristic value is 0.
    7. Initialize the total heuristic value to 0.
    8. Calculate the shortest path distances for the robot from its current location to all reachable locations, considering blocked locations.
    9. Initialize the minimum robot distance to any potential first pushing position for any misplaced box to infinity.
    10. For each misplaced box:
        a. Get its current location (`l_b`) and goal location (`g_b`).
        b. Retrieve the pre-calculated shortest path distance for the box from `l_b` to `g_b` (minimum pushes). If this distance is infinity, the box cannot reach its goal, so the state is likely unsolvable; return infinity. Add this distance to the total heuristic.
        c. Find potential initial pushing positions for this box: Iterate through neighbors (`l'_b`) of `l_b`. If `l'_b` is on a shortest path from `l_b` to `g_b` (checked using pre-calculated distances to the goal), then the required robot pushing position (`p`) is the neighbor of `l_b` in the opposite direction of `l_b` -> `l'_b`.
        d. For each valid pushing position `p` found in step 10c: If `p` is reachable by the robot (i.e., `p` is in the robot's distance map calculated in step 8), update the minimum robot distance found so far (step 9) with the robot's distance to `p`.
    11. After processing all misplaced boxes, if the minimum robot distance found (step 9) is still infinity, it means the robot cannot reach any position to start pushing any misplaced box; return infinity.
    12. Add the minimum robot distance (step 9) to the total heuristic (step 7).
    13. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Graph representation of the grid from static facts.
        - Pre-calculated box distances to goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph and reverse graph from adjacent facts
        self.graph, self.reverse_graph = build_graph(static_facts)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Pre-calculate box distances from all locations to all goal locations
        # This assumes boxes can move freely (no obstacles)
        self.box_dist_to_goals = {}
        # We only need distances to the specific goal locations for boxes
        for goal_box, goal_loc in self.goal_locations.items():
             # BFS from the goal location on the reverse graph to get distances *to* the goal
             self.box_dist_to_goals[goal_box] = bfs_distances(goal_loc, self.reverse_graph, set())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify robot and box locations
        robot_loc = None
        box_locs = {}
        clear_locs = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box_locs[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                clear_locs.add(parts[1])

        # 3. Determine locations blocked for robot movement
        # Robot cannot move into locations that are not clear, unless it's its own location (which is handled by BFS start)
        # Also, robot cannot move into locations occupied by boxes.
        robot_blocked_locs = set(loc for loc in self.graph.keys() if loc not in clear_locs)
        if robot_loc in robot_blocked_locs:
             robot_blocked_locs.remove(robot_loc) # Robot's current location is not blocked for starting BFS

        for box, loc in box_locs.items():
             if loc != robot_loc: # Robot is not blocked by a box it's standing on (shouldn't happen)
                 robot_blocked_locs.add(loc)


        # 4. & 5. Identify misplaced boxes
        misplaced_boxes = [
            box for box, loc in box_locs.items()
            if self.goal_locations.get(box) != loc
        ]

        # 6. If no boxes are misplaced, the heuristic is 0
        if not misplaced_boxes:
            return 0

        total_heuristic = 0
        min_robot_dist_to_any_push_pos = float('inf')

        # 8. Calculate robot distances from current location considering obstacles
        robot_dist_map = bfs_distances(robot_loc, self.graph, robot_blocked_locs)

        # 10. Calculate box pushes and find potential first push positions
        for box in misplaced_boxes:
            l_b = box_locs[box]
            g_b = self.goal_locations[box]

            # 10a. Calculate box distance (minimum pushes)
            # Use pre-calculated distances to the goal
            box_dist_to_goal_map = self.box_dist_to_goals.get(box, {})
            box_dist = box_dist_to_goal_map.get(l_b, float('inf'))

            if box_dist == float('inf'):
                 # Box cannot reach its goal - likely a deadlock or unsolvable
                 return float('inf') # Return infinity for unsolvable states

            total_heuristic += box_dist

            # 10c. Find potential first push positions for this box
            # Iterate through neighbors of l_b to find potential next steps l'_b
            for l_prime_b, dir_l_b_to_l_prime_b in self.graph.get(l_b, []):
                 # Check if l'_b is on a shortest path from l_b to g_b for the box
                 # Distance from l_b to g_b should be 1 + distance from l'_b to g_b
                 dist_l_b_to_g_b = box_dist_to_goal_map.get(l_b, float('inf'))
                 dist_l_prime_b_to_g_b = box_dist_to_goal_map.get(l_prime_b, float('inf'))

                 if dist_l_b_to_g_b == dist_l_prime_b_to_g_b + 1:
                     # l'_b is on a shortest path from l_b to g_b
                     # The required robot pushing position p is adjacent to l_b
                     # in the opposite direction of l_b -> l'_b
                     required_robot_dir = get_opposite_direction(dir_l_b_to_l_prime_b)
                     push_pos = get_neighbor_in_direction(l_b, required_robot_dir, self.graph)

                     if push_pos:
                         # 10d. Calculate robot distance to this potential push_pos
                         if push_pos in robot_dist_map:
                             min_robot_dist_to_any_push_pos = min(min_robot_dist_to_any_push_pos, robot_dist_map[push_pos])
                         # else: push_pos is unreachable by robot due to obstacles, ignore this push_pos

        # 11. Add the minimum robot distance to the total heuristic
        if min_robot_dist_to_any_push_pos == float('inf'):
             # Robot cannot reach any pushing position for any misplaced box
             return float('inf') # Likely unsolvable

        total_heuristic += min_robot_dist_to_any_push_pos

        # 13. Return the total estimated cost
        return total_heuristic
