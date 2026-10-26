import collections

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
    Estimates the cost to reach the goal state by summing the estimated costs
    for each box that is not yet at its goal location. The estimated cost for
    a single box is the sum of two components:
    1. The minimum number of pushes required to move the box from its current
       location to its goal location, assuming no obstacles other than walls
       (derived from the grid graph). This is calculated as the shortest path
       distance on the grid graph.
    2. The minimum number of robot moves required to get the robot from its
       current location to a position adjacent to the box, from which it can
       make the first push towards the box's goal. This is calculated as the
       shortest path distance on the grid graph for the robot.

    Assumptions:
    - The problem locations form a grid structure implicitly defined by the
      `adjacent` predicates in the static facts.
    - The goal state specifies the target location for each box using `(at box_name loc_name)` facts.
    - Each box present in the initial state (and thus potentially in subsequent states)
      that is relevant to the goal has a corresponding target location specified
      in the goal state.
    - Locations not connected by `adjacent` predicates are impassable (walls).
    - The heuristic calculates distances on the full grid graph, ignoring
      dynamic obstacles (other boxes, robot, clear status) for simplicity and
      efficiency. This makes the heuristic non-admissible but potentially
      more informative than simpler alternatives.

    Heuristic Initialization:
    The heuristic constructor processes the static facts from the task definition
    to build a graph representation of the Sokoban grid. This graph stores
    adjacency information based on the `adjacent` predicates, mapping a location
    and a direction to the resulting adjacent location. Goal locations for each
    box are also stored.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the current location of the robot from the state facts.
    2. Identify the current location of each box that needs to reach a goal
       location (by checking against the stored goal facts).
    3. Initialize the total heuristic value `h` to 0.
    4. For each box that is not yet at its goal location:
       a. Find the box's current location and its target goal location.
       b. Calculate a shortest path for the box from its current location
          to its goal location on the pre-built grid graph using Breadth-First Search (BFS).
          If no path exists (goal is unreachable on the grid), the state is
          likely a dead end, and the heuristic returns infinity. The number
          of pushes required is the length of this path minus one.
       c. Determine the first step location on the shortest box path and the
          direction of this first push.
       d. Calculate the required location for the robot to be able to make
          this first push. According to the PDDL `push` action, if the box
          moves from `?bloc` to `?floc` in direction `?dir`, the robot must
          be at `?rloc` such that `(adjacent ?rloc ?bloc ?dir)`. This means
          `?bloc` is adjacent to `?rloc` in the reverse direction of `?dir`.
          So, the required robot location is the neighbor of the box's current
          location in the direction opposite to the push direction. If such a
          location does not exist (e.g., box is against a wall), the box cannot
          be pushed in that direction, indicating a dead end for this box's
          goal path. The heuristic returns infinity.
       e. Calculate the shortest path for the robot from its current location
          to the required push location on the pre-built grid graph using BFS.
          If the required push location is unreachable by the robot on the grid,
          the heuristic returns infinity. The number of robot moves required
          is the length of this path minus one.
       f. Add the estimated number of box pushes and the estimated number of
          robot moves for this box to the total heuristic value `h`.
    5. Return the total heuristic value `h`. If `h` is 0, it means all relevant
       boxes are at their goal locations, indicating a goal state.
    """
    def __init__(self, task, static_info):
        # Build graph from static_info
        self.graph = collections.defaultdict(dict)
        self.locations = set()
        reverse_dir_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        self.reverse_dir_map = reverse_dir_map # Store for later use

        for fact in static_info:
            if fact.startswith('(adjacent'):
                parts = fact.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                self.graph[loc1][direction] = loc2
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Store goal locations for boxes
        self.goal_locs = {}
        for goal in task.goals:
            if goal.startswith('(at '):
                parts = goal.strip('()').split()
                box_name = parts[1]
                loc_name = parts[2]
                self.goal_locs[box_name] = loc_name

    def bfs_path(self, start_loc, end_loc):
        """
        Finds a shortest path between two locations on the grid graph using BFS.
        Returns the path as a list of location names, or None if unreachable.
        """
        if start_loc == end_loc:
            return [start_loc]
        queue = collections.deque([(start_loc, [start_loc])])
        visited = {start_loc}
        while queue:
            current_loc, path = queue.popleft()
            # Iterate through neighbors in the graph
            for direction, neighbor_loc in self.graph.get(current_loc, {}).items():
                 if neighbor_loc not in visited:
                    if neighbor_loc == end_loc:
                        return path + [neighbor_loc]
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, path + [neighbor_loc]))
        return None # Unreachable

    def bfs_distance(self, start_loc, end_loc):
        """
        Finds the shortest distance between two locations on the grid graph using BFS.
        Returns the distance, or float('inf') if unreachable.
        """
        if start_loc == end_loc:
            return 0
        queue = collections.deque([(start_loc, 0)])
        visited = {start_loc}
        while queue:
            current_loc, dist = queue.popleft()
            for direction, neighbor_loc in self.graph.get(current_loc, {}).items():
                 if neighbor_loc not in visited:
                    if neighbor_loc == end_loc:
                        return dist + 1
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))
        return float('inf') # Unreachable


    def __call__(self, state, task):
        # Extract robot location
        robot_loc = None
        for fact in state:
            if fact.startswith('(at-robot '):
                robot_loc = fact.strip('()').split()[1]
                break
        if robot_loc is None:
             # Robot location not found, invalid state?
             return float('inf')

        # Extract current box locations that have a goal
        current_box_locs = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                obj_name = parts[1]
                loc_name = parts[2]
                # Only consider objects that are listed as goals
                if obj_name in self.goal_locs:
                     current_box_locs[obj_name] = loc_name

        h = 0
        for box_name, current_loc in current_box_locs.items():
            goal_loc = self.goal_locs.get(box_name)

            # Should always find goal_loc if box_name was in self.goal_locs keys
            if goal_loc is None:
                 # This case should ideally not happen in a well-formed task
                 continue # Or return float('inf')? Let's assume valid task.

            if current_loc == goal_loc:
                continue # Box is already at goal

            # Calculate box distance (pushes needed)
            box_path = self.bfs_path(current_loc, goal_loc)
            if box_path is None:
                # Box goal is unreachable from its current location on the grid
                return float('inf')

            box_dist = len(box_path) - 1 # Number of pushes

            # Find the required robot location for the first push
            # First step of box path is current_loc -> next_loc
            next_loc = box_path[1]

            # Find direction from current_loc to next_loc
            push_dir = None
            # Iterate through possible directions from current_loc
            for direction, neighbor_loc in self.graph.get(current_loc, {}).items():
                if neighbor_loc == next_loc:
                    push_dir = direction
                    break

            if push_dir is None:
                 # This should not happen if next_loc is a valid neighbor from current_loc
                 # and box_path was found by BFS on self.graph
                 return float('inf') # Defensive check

            # Find location robot needs to be at to push current_loc in push_dir
            # This is the location loc_r such that (adjacent loc_r current_loc push_dir)
            # which means current_loc is adjacent to loc_r in reverse_push_dir
            # i.e., self.graph[current_loc][reverse_push_dir] == loc_r
            reverse_push_dir = self.reverse_dir_map[push_dir]
            robot_push_loc = self.graph.get(current_loc, {}).get(reverse_push_dir)

            if robot_push_loc is None:
                 # The box cannot be pushed from this location in the required direction
                 # (e.g., against a wall/edge). This state is likely a dead end for this box.
                 return float('inf')

            # Calculate robot distance to the push location
            robot_dist = self.bfs_distance(robot_loc, robot_push_loc)
            if robot_dist == float('inf'):
                 # Robot cannot reach the required push location on the grid
                 return float('inf')

            h += box_dist + robot_dist

        # If the loop finishes, all relevant boxes are at their goals, or the sum is finite.
        # If h is 0, it means the loop was skipped entirely (all boxes at goal).
        return h
