from heuristics.heuristic_base import Heuristic
from collections import deque

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Sokoban.

    Summary:
    Estimates the cost to reach the goal by summing the minimum number of pushes
    required for each box to reach its goal location (ignoring obstacles for
    box movement) and adding the minimum number of robot moves required to
    reach a location from which it can make the first push towards the goal
    for any box that needs moving (considering other boxes as obstacles).

    Assumptions:
    - The PDDL defines a grid-like structure using 'adjacent' facts.
    - Locations are represented as strings like 'loc_X_Y'.
    - Goals specify the final location for each box using '(at box_name loc_name)'.
    - The grid defined by 'adjacent' facts is connected.
    - 'adjacent' facts are provided for both directions (e.g., l1->l2 right, l2->l1 left).

    Heuristic Initialization:
    - Parses 'adjacent' facts from the static information to build both a directed
      and an undirected graph representing the grid connectivity.
    - Creates a mapping for opposite directions (e.g., up <-> down).
    - Stores the goal locations for each box.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current location of the robot and each box from the state.
    2. Identify the goal location for each box from the task goals.
    3. Initialize total box distance heuristic component to 0.
    4. Create a list of boxes that are not yet at their goal locations.
    5. Create a set of potential robot push locations (locations adjacent to a box
       where the robot needs to be to push the box towards its goal).
    6. For each box not at its goal:
       a. Compute the shortest path distance between the box's current location
          and its goal location on the *undirected* grid graph, ignoring all obstacles.
          This distance represents the minimum number of pushes required for this box
          in an empty grid.
       b. Add this distance to the total box distance.
       c. If any box cannot reach its goal (distance is infinity), the state is
          likely a dead end, return infinity.
       d. Find a shortest path for the box from its current location to its goal
          on the *undirected* grid graph.
       e. If a path exists and has more than one location, determine the direction
          of the first step (from current location to the next location on the path).
       f. Find the location adjacent to the box's current location in the *opposite*
          of the direction found in step 6e using the *directed* grid graph. This is
          a required robot push location for this box's next step. Add it to the set
          of potential robot push locations.
    7. If all boxes are at their goals (total box distance is 0), return 0.
    8. Identify obstacle locations for the robot: these are the current locations
       of all boxes.
    9. Compute the minimum shortest path distance for the robot from its current
       location to any location in the set of potential robot push locations,
       considering the obstacle locations, using BFS on the *undirected* grid graph.
    10. If the robot cannot reach any potential push location, return infinity.
    11. The final heuristic value is the total box distance plus the minimum robot distance.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Build grid graphs from adjacent facts
        self.grid_graph_directed = self._build_grid_graph_directed(self.static)
        self.grid_graph_undirected = self._build_grid_graph_undirected(self.static)

        # Map opposite directions
        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # Store box goal locations
        self.box_goal_map = self._parse_goal_locations(self.goals)

    def _build_grid_graph_directed(self, static_facts):
        """Builds a directed adjacency list graph from adjacent facts."""
        graph = {}
        for fact_str in static_facts:
            if fact_str.startswith('(adjacent '):
                parts = fact_str.strip('()').split()
                # Ensure correct number of parts for (adjacent l1 l2 dir)
                if len(parts) == 4:
                    l1 = parts[1]
                    l2 = parts[2]
                    direction = parts[3]
                    if l1 not in graph:
                        graph[l1] = []
                    graph[l1].append((l2, direction))
        return graph

    def _build_grid_graph_undirected(self, static_facts):
        """Builds an undirected adjacency list graph from adjacent facts."""
        graph = {}
        for fact_str in static_facts:
            if fact_str.startswith('(adjacent '):
                parts = fact_str.strip('()').split()
                 # Ensure correct number of parts for (adjacent l1 l2 dir)
                if len(parts) == 4:
                    l1 = parts[1]
                    l2 = parts[2]
                    if l1 not in graph:
                        graph[l1] = []
                    if l2 not in graph[l1]: # Avoid duplicates
                        graph[l1].append(l2)
                    if l2 not in graph:
                        graph[l2] = []
                    if l1 not in graph[l2]: # Avoid duplicates
                        graph[l2].append(l1)
        return graph

    def _parse_goal_locations(self, goals):
        """Parses goal facts to map boxes to their goal locations."""
        box_goal_map = {}
        for goal_str in goals:
            parts = goal_str.strip('()').split()
            if parts[0] == 'at' and len(parts) == 3: # Ensure it's an (at box loc) fact
                box_name = parts[1]
                goal_loc = parts[2]
                box_goal_map[box_name] = goal_loc
        return box_goal_map

    def _bfs_distance(self, start_loc, end_loc, graph, obstacle_locs=None):
        """
        Performs BFS to find the shortest path distance between start_loc and end_loc.
        Optionally takes a set of obstacle locations to avoid.
        Returns distance or float('inf').
        """
        if start_loc == end_loc:
            return 0
        # Check if locations exist in the graph before starting BFS
        if start_loc not in graph or end_loc not in graph:
             # If start or end is not in the graph, it's unreachable in the grid
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        if obstacle_locs is None:
            obstacle_locs = set()

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            for neighbor_loc in graph.get(current_loc, []):
                # Check if neighbor is in graph (should be if graph is built correctly)
                # and not visited and not an obstacle
                if neighbor_loc in graph and neighbor_loc not in visited and neighbor_loc not in obstacle_locs:
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

        return float('inf') # No path found

    def _bfs_path(self, start_loc, end_loc, graph):
        """
        Performs BFS to find a shortest path between start_loc and end_loc.
        Returns a list of locations representing the path, or None if no path.
        """
        if start_loc == end_loc:
            return [start_loc]
        # Check if locations exist in the graph before starting BFS
        if start_loc not in graph or end_loc not in graph:
             return None

        queue = deque([(start_loc, [start_loc])])
        visited = {start_loc}

        while queue:
            current_loc, path = queue.popleft()

            if current_loc == end_loc:
                return path

            for neighbor_loc in graph.get(current_loc, []):
                # Check if neighbor is in graph (should be if graph is built correctly)
                # and not visited
                if neighbor_loc in graph and neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, path + [neighbor_loc]))

        return None # No path found

    def _bfs_min_distance_to_set(self, start_loc, target_locs, graph, obstacle_locs=None):
        """
        Performs BFS to find the minimum shortest path distance from start_loc
        to any location in the target_locs set.
        Optionally takes a set of obstacle locations to avoid.
        Returns distance or float('inf').
        """
        if not target_locs:
            return float('inf') # No targets

        if start_loc in target_locs:
            return 0 # Robot is already at a target location

        # Check if start location exists in the graph
        if start_loc not in graph:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        if obstacle_locs is None:
            obstacle_locs = set()

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc in target_locs:
                return dist

            for neighbor_loc in graph.get(current_loc, []):
                # Check if neighbor is in graph (should be if graph is built correctly)
                # and not visited and not an obstacle
                if neighbor_loc in graph and neighbor_loc not in visited and neighbor_loc not in obstacle_locs:
                    visited.add(neighbor_loc)
                    queue.append((neighbor_loc, dist + 1))

        return float('inf') # No path found to any target

    def __call__(self, node):
        state = node.state

        robot_location = None
        box_locations = {} # Map box name to location string
        current_obstacle_locs = set() # Locations occupied by boxes

        # Extract current locations from state
        for fact_str in state:
            parts = fact_str.strip('()').split()
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                box_loc = parts[2]
                box_locations[box_name] = box_loc
                current_obstacle_locs.add(box_loc) # Boxes are obstacles for the robot

        total_box_distance = 0
        boxes_to_move = [] # List of (box_name, current_loc, goal_loc)
        potential_robot_push_locs = set() # Specific locations robot needs to reach

        # Identify boxes needing movement and calculate their independent distances
        for box_name, current_loc in box_locations.items():
            goal_loc = self.box_goal_map.get(box_name)
            if goal_loc and current_loc != goal_loc:
                boxes_to_move.append((box_name, current_loc, goal_loc))

                # Calculate box distance (undirected graph, ignoring obstacles)
                distance = self._bfs_distance(current_loc, goal_loc, self.grid_graph_undirected)
                if distance == float('inf'):
                    # A box cannot reach its goal on the grid graph
                    return float('inf')
                total_box_distance += distance

                # Find potential robot push locations for this box's next step
                # Find a shortest path for the box on the undirected graph
                path = self._bfs_path(current_loc, goal_loc, self.grid_graph_undirected)

                if path and len(path) > 1:
                    next_box_loc = path[1]
                    # Find direction from current_loc to next_box_loc using directed graph
                    direction_to_next = None
                    # Iterate through neighbors of current_loc in the directed graph
                    for neighbor, direction in self.grid_graph_directed.get(current_loc, []):
                        if neighbor == next_box_loc:
                            direction_to_next = direction
                            break # Found the direction of the first step

                    if direction_to_next:
                        required_robot_dir = self.opposite_direction.get(direction_to_next)
                        if required_robot_dir:
                            # Find the location adjacent to current_loc in the required_robot_dir
                            # using the directed graph to get the specific location
                            for neighbor, direction in self.grid_graph_directed.get(current_loc, []):
                                if direction == required_robot_dir:
                                    potential_robot_push_locs.add(neighbor)
                                    break # Found the push location for this box's next step

        # If all boxes are at their goals, the heuristic is 0
        if not boxes_to_move:
            return 0

        # Calculate robot cost to get to any required push location
        min_robot_distance = float('inf')
        if robot_location:
            # Obstacles for the robot are the locations of *all* boxes.
            # Robot moves on the undirected grid graph.
            min_robot_distance = self._bfs_min_distance_to_set(
                robot_location,
                potential_robot_push_locs,
                self.grid_graph_undirected,
                obstacle_locs=current_obstacle_locs
            )

        # If robot cannot reach any required push location
        if min_robot_distance == float('inf'):
             # This can happen if the robot is trapped or the required push locations are blocked
             return float('inf')

        # Total heuristic is sum of box distances + robot distance to get into position
        return total_box_distance + min_robot_distance
