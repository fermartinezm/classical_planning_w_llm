from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions (kept outside the class)

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def get_opposite_dir(dir):
    """Get the opposite direction."""
    if dir == 'up': return 'down'
    if dir == 'down': return 'up'
    if dir == 'left': return 'right'
    if dir == 'right': return 'left'
    return None # Should not happen

def shortest_path(graph, start, end, obstacles):
    """
    Find the shortest path distance from start to end in the graph, avoiding obstacles.
    Returns distance or infinity if no path exists.
    """
    if start == end:
        return 0
    # Check if start/end are valid nodes in the graph being searched
    if start not in graph or end not in graph:
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        # Check if current_loc is in graph before iterating neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # No path found

def _find_shortest_path_list(graph, start, end, obstacles):
    """
    Find the shortest path (list of locations) from start to end in the graph, avoiding obstacles.
    Returns path list or None if no path exists.
    """
    if start == end:
        return [start]
    # Check if start/end are valid nodes in the graph being searched
    if start not in graph or end not in graph:
        return None

    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        current_loc, path = queue.popleft()
        if current_loc == end:
            return path
        # Check if current_loc is in graph before iterating neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    return None # No path found

def get_push_pos_name(box_loc_name, next_box_loc_name, static_graph_with_dirs):
    """
    Given box_loc and the next location it should move to, find the robot's
    required position to push it.
    static_graph_with_dirs: dict {loc1: {loc2: dir, ...}, ...}
    """
    # Find the direction from box_loc_name to next_box_loc_name
    direction_of_push = None
    # Iterate through neighbors of box_loc_name to find next_box_loc_name and its direction
    # Use .get(..., {}) to safely handle cases where box_loc_name might not be a key
    for neighbor, dir in static_graph_with_dirs.get(box_loc_name, {}).items():
         if neighbor == next_box_loc_name:
              direction_of_push = dir
              break

    if direction_of_push is None:
        # next_box_loc_name is not a direct neighbor of box_loc_name in the graph.
        # This indicates an invalid path step was provided or graph is incomplete.
        return None

    opposite_dir = get_opposite_dir(direction_of_push)

    # Find the location adjacent to box_loc_name in the opposite direction
    # Use .get(..., {}) to safely handle cases where box_loc_name might not be a key
    for neighbor, dir in static_graph_with_dirs.get(box_loc_name, {}).items():
        if dir == opposite_dir:
            return neighbor

    return None # Should not happen if the grid is complete around box_loc_name


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated
    cost for each box that is not yet at its goal location. The cost for a single
    box is estimated as the minimum number of pushes required to move it to its
    goal location (shortest path distance in the static grid graph) plus the
    estimated cost for the robot to reach the position required for the *first*
    push of that box (shortest path distance in the current state graph, avoiding
    obstacles).

    # Assumptions
    - The grid structure and adjacency relationships are defined by the 'adjacent' facts.
    - Locations not connected by 'adjacent' facts are walls or unreachable.
    - Boxes need to be pushed to specific goal locations.
    - The robot must be in a specific adjacent location to push a box.
    - The shortest path in the static grid graph represents the minimum number of pushes.
    - The shortest path in the dynamic state graph represents the minimum robot moves.
    - The required push position for a box is the location adjacent to the box's
      current location in the opposite direction of the intended push. This push
      position must be clear for the robot to move there.

    # Heuristic Initialization
    - Build a static graph of locations based on 'adjacent' facts. Store both the simple adjacency list and a version including directions.
    - Identify all valid location names.
    - Map each box to its goal location from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Identify the current location of the robot.
    3. Identify the current location of each box that has a goal location.
    4. Identify all locations that are currently 'clear'.
    5. Build the dynamic graph for robot movement: nodes are locations, edges exist between adjacent locations if the destination is 'clear'.
    6. Initialize total heuristic value `h = 0`.
    7. For each box `b` that is not at its goal location `g_loc`:
        a. Identify the box's current location `b_loc`.
        b. Identify locations occupied by *other* boxes. These are obstacles for the current box's movement path in the static graph.
        c. Find the shortest path (list of locations) from `b_loc` to `g_loc` in the *static graph* (ignoring dynamic obstacles except other boxes). Use BFS. If no path exists, return infinity.
        d. The length of this path minus 1 is the minimum number of pushes required for this box. Add this number of pushes to `h`.
        e. If pushes are required (path length > 1), identify the first step in the path: `b_loc` -> `next_b_loc`.
        f. Determine the required robot position `push_pos` to push the box from `b_loc` to `next_b_loc`. This is the location adjacent to `b_loc` in the opposite direction of `b_loc` to `next_b_loc`.
        g. Calculate the shortest path distance from the robot's current location to `push_pos` in the *dynamic robot graph* (only moving through 'clear' locations). Use BFS. If no path exists (robot cannot reach the required clear push position), return infinity.
        h. Add this robot movement distance to `h`.
    8. Return the total heuristic value `h`.

    Note: This heuristic sums costs for individual boxes, ignoring potential interactions (positive or negative) between boxes and robot movements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Build the static graph from adjacent facts
        self.static_graph = {} # simple adjacency list {loc1: [loc2, loc3], ...}
        self.static_graph_with_dirs = {} # {loc1: {loc2: dir, ...}, ...}
        self.all_locations = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
                if loc1 not in self.static_graph:
                    self.static_graph[loc1] = []
                if loc2 not in self.static_graph:
                    self.static_graph[loc2] = []
                # Add edge in both directions if not already present
                if loc2 not in self.static_graph[loc1]:
                     self.static_graph[loc1].append(loc2)
                if loc1 not in self.static_graph[loc2]:
                     self.static_graph[loc2].append(loc1)

                if loc1 not in self.static_graph_with_dirs:
                    self.static_graph_with_dirs[loc1] = {}
                if loc2 not in self.static_graph_with_dirs:
                    self.static_graph_with_dirs[loc2] = {}
                self.static_graph_with_dirs[loc1][loc2] = direction
                self.static_graph_with_dirs[loc2][loc1] = get_opposite_dir(direction) # Store opposite direction

        # Store goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.box_goals[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Extract current state information
        robot_loc = None
        box_locations = {} # {box_name: location_name}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1] in self.box_goals:
                 box, loc = parts[1], parts[2]
                 box_locations[box] = loc

        # 4. Identify locations that are currently clear
        clear_locations = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'clear'}

        # 5. Build the dynamic graph for robot movement
        # Robot can move to adjacent locations that are clear
        robot_graph = {}
        for loc in self.all_locations:
             robot_graph[loc] = []
             if loc in self.static_graph: # Ensure loc is a valid node in the static graph
                 for neighbor in self.static_graph[loc]:
                     if neighbor in clear_locations:
                         robot_graph[loc].append(neighbor)

        # 6. Initialize total heuristic value
        total_heuristic = 0

        # 7. For each box not at its goal
        for box, goal_loc in self.box_goals.items():
            current_box_loc = box_locations.get(box)

            if current_box_loc is None or current_box_loc == goal_loc:
                continue # Box is already at goal or missing

            # 7b. Obstacles for the box path in the static graph are locations occupied by *other* boxes.
            other_box_locations = {loc for b, loc in box_locations.items() if b != box}

            # 7c. Find the shortest path for the box in the static graph
            # The box path cannot go through locations occupied by other boxes.
            box_path = _find_shortest_path_list(self.static_graph, current_box_loc, goal_loc, other_box_locations)

            if box_path is None:
                # Box cannot reach the goal (e.g., trapped by other boxes or walls)
                return float('inf')

            # 7d. Minimum pushes = path length - 1
            min_pushes = len(box_path) - 1
            total_heuristic += min_pushes

            # 7e, 7f, 7g, 7h. Add robot cost for the first push
            if min_pushes > 0:
                # Identify the first step
                next_box_loc = box_path[1]

                # Determine required robot push position
                push_pos = get_push_pos_name(current_box_loc, next_box_loc, self.static_graph_with_dirs)

                if push_pos is None:
                     # Should not happen if static_graph_with_dirs is built correctly and box_path is valid.
                     # This implies the required push_pos doesn't exist or is not adjacent correctly.
                     return float('inf')

                # Calculate robot movement distance to push_pos in the dynamic robot graph
                # Robot can only move through clear locations to reach push_pos.
                # The push_pos itself must be clear *before* the push action.
                # The robot_graph already only includes edges to clear locations.
                # So, calculate shortest path in robot_graph to push_pos.
                # No extra obstacles set is needed for shortest_path here, as robot_graph defines valid moves.
                robot_dist_to_push_pos = shortest_path(robot_graph, robot_loc, push_pos, set())

                if robot_dist_to_push_pos == float('inf'):
                    # Robot cannot reach the required push position through clear locations.
                    return float('inf')

                total_heuristic += robot_dist_to_push_pos

        # 8. Return the total heuristic value
        return total_heuristic
