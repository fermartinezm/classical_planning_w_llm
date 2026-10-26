from collections import deque

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        two components:
        1. The sum of the shortest path distances for each box from its current
           location to its assigned goal location. This estimates the minimum
           number of push actions required for the boxes.
        2. The minimum shortest path distance from the robot's current location
           to any location adjacent to any box that is not yet at its goal.
           This estimates the minimum number of move actions required for the
           robot to get into a position to interact with a misplaced box.
        Shortest paths are calculated using BFS on the graph of locations
        derived from the 'adjacent' facts in the PDDL domain.

    Assumptions:
        - The locations form a graph defined by 'adjacent' facts.
        - The goal state specifies a unique target location for each box.
        - Location names follow a pattern (e.g., loc_R_C), although the
          heuristic primarily relies on the graph structure, not the grid
          coordinates derived from names.

    Heuristic Initialization:
        The heuristic is initialized with the planning task object. During
        initialization:
        - The graph of locations is built from the 'adjacent' facts in the
          static information. Edges are added in both directions based on
          the provided adjacent facts.
        - The goal location for each box is extracted from the task's goal
          state.
        - For each box goal location, a Breadth-First Search (BFS) is performed
          on the *reverse* graph to precompute the shortest path distances
          from all reachable locations *to* that goal location. This allows
          efficient lookup of box-goal distances during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Extract the current location of the robot and each box from the state facts.
        2. Initialize the heuristic value `h` to 0.
        3. Identify all boxes that are not currently at their assigned goal location.
        4. For each misplaced box:
           a. Get the box's current location and its goal location.
           b. Look up the precomputed shortest path distance from the box's
              current location to its goal location using the distance map
              computed during initialization (BFS on the reverse graph from the goal).
           c. If the goal is unreachable from the box's current location, the state
              is likely a deadlock or unsolvable. Return infinity.
           d. Add this distance to `h`.
        5. If there are no misplaced boxes, the state is a goal state, and the
           heuristic is 0. Return `h`.
        6. If there are misplaced boxes, calculate the robot's contribution:
           a. Perform a BFS starting from the robot's current location to find
              shortest path distances to all reachable locations.
           b. Initialize a variable `min_robot_dist_to_adjacent` to infinity.
           c. For each misplaced box:
              i. Get the box's current location.
              ii. For each location adjacent to the box's current location (according
                  to the graph built during initialization):
                  - Look up the shortest path distance from the robot's current
                    location to this adjacent location using the BFS result from step 6a.
                  - Update `min_robot_dist_to_adjacent` with the minimum distance found.
           d. If `min_robot_dist_to_adjacent` is still infinity, the robot cannot
              reach any location adjacent to any misplaced box. The state is likely
              unsolvable. Return infinity.
           e. Add `min_robot_dist_to_adjacent` to `h`.
        7. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by building the location graph and precomputing
        distances to goal locations.

        Args:
            task: The planning task object (instance of the Task class).
        """
        self.task = task
        self.graph = self._build_graph(task.static)
        self.goal_locations = self._parse_goal_locations(task.goals)
        self.goal_dist_maps = self._precompute_goal_distances()

    def _build_graph(self, static_facts):
        """
        Builds the location graph from adjacent facts.

        Args:
            static_facts: A frozenset of static facts from the task.

        Returns:
            A dictionary representing the graph where keys are locations
            and values are lists of adjacent locations.
        """
        graph = {}
        # Collect all locations first to ensure all nodes are in the graph dict
        all_locations = set()
        for fact in static_facts:
            if fact.startswith('(adjacent'):
                parts = fact.strip('()').split()
                l1 = parts[1]
                l2 = parts[2]
                all_locations.add(l1)
                all_locations.add(l2)

        for loc in all_locations:
            graph[loc] = []

        # Add edges based on adjacent facts
        for fact in static_facts:
            if fact.startswith('(adjacent'):
                parts = fact.strip('()').split()
                l1 = parts[1]
                l2 = parts[2]
                # Add directed edge l1 -> l2
                graph[l1].append(l2)
                # Assuming bidirectionality is explicitly stated for all connections
                # in the PDDL instance by providing adjacent facts for both directions.
                # If not, we would need to infer the reverse direction here.
                # Based on example files, both directions are explicitly listed.

        return graph

    def _parse_goal_locations(self, goals):
        """
        Parses the goal facts to find the target location for each box.

        Args:
            goals: A frozenset of goal facts from the task.

        Returns:
            A dictionary mapping box names to their goal locations.
        """
        goal_locations = {}
        for fact in goals:
            if fact.startswith('(at'):
                parts = fact.strip('()').split()
                # Fact is '(at box_name location)'
                box_name = parts[1]
                goal_location = parts[2]
                goal_locations[box_name] = goal_location
        return goal_locations

    def _precompute_goal_distances(self):
        """
        Computes shortest path distances from all locations to each goal location.
        Uses BFS on the reverse graph.

        Returns:
            A dictionary mapping goal locations to their distance maps
            (dictionaries mapping locations to distances).
        """
        goal_dist_maps = {}
        reverse_graph = {}

        # Build reverse graph
        # Ensure all nodes from the forward graph are keys in the reverse graph
        for loc in self.graph:
             if loc not in reverse_graph:
                 reverse_graph[loc] = []

        for l1, neighbors in self.graph.items():
            for l2 in neighbors:
                if l2 not in reverse_graph:
                    reverse_graph[l2] = [] # Should already exist from previous loop
                reverse_graph[l2].append(l1)


        # Run BFS from each goal location on the reverse graph
        for goal_loc in self.goal_locations.values():
            if goal_loc in reverse_graph: # Ensure goal location is a valid node in the graph
                 goal_dist_maps[goal_loc] = self._bfs(goal_loc, reverse_graph)
            else:
                 # This should not happen in valid PDDL instances where goals are reachable nodes
                 print(f"Warning: Goal location {goal_loc} not found in graph nodes.")
                 goal_dist_maps[goal_loc] = {} # Empty map, distances will be inf

        return goal_dist_maps

    def _bfs(self, start_node, graph):
        """
        Performs Breadth-First Search to find shortest distances from a start node.

        Args:
            start_node: The node to start the BFS from.
            graph: The graph (adjacency list).

        Returns:
            A dictionary mapping reachable nodes to their shortest distance
            from the start node. Nodes not reachable have distance infinity.
        """
        distances = {node: float('inf') for node in graph}
        if start_node not in graph:
             # Start node is not in the graph, no paths possible
             return distances

        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The estimated cost (heuristic value) or float('inf') if likely unsolvable.
        """
        robot_location = None
        box_locations = {} # {box_name: location}

        # Extract robot and box locations from the current state
        for fact in state:
            if fact.startswith('(at-robot'):
                robot_location = fact.strip('()').split()[1]
            elif fact.startswith('(at'):
                parts = fact.strip('()').split()
                box_name = parts[1]
                location = parts[2]
                box_locations[box_name] = location

        h = 0
        misplaced_boxes = []

        # Calculate box-goal distances
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # Only consider boxes that exist in the current state and are misplaced
            if current_loc and current_loc != goal_loc:
                misplaced_boxes.append(box)
                # Get precomputed distance from current_loc to goal_loc
                # Ensure goal_loc is in precomputed maps (handled in init)
                if goal_loc not in self.goal_dist_maps:
                     # This indicates an issue during initialization or an invalid goal
                     print(f"Error: Goal location {goal_loc} not in precomputed maps.")
                     return float('inf')

                dist_box_goal = self.goal_dist_maps[goal_loc].get(current_loc, float('inf'))

                if dist_box_goal == float('inf'):
                     # Box cannot reach its goal from its current location
                     return float('inf')

                h += dist_box_goal

        # If all boxes are at goals, heuristic is 0
        if not misplaced_boxes:
            return 0

        # Calculate robot distance to closest location adjacent to any misplaced box
        min_robot_dist_to_adjacent = float('inf')

        # Compute distances from the robot's current location
        # Ensure robot_location is a valid node in the graph
        if robot_location not in self.graph:
             # Robot is in a location not part of the traversable graph
             return float('inf') # Likely unsolvable

        robot_dist_map = self._bfs(robot_location, self.graph)

        for box in misplaced_boxes:
            box_loc = box_locations[box]
            # Check if box_loc is a valid node in the graph and has neighbors
            if box_loc in self.graph and self.graph[box_loc]:
                 for adjacent_loc in self.graph[box_loc]:
                     dist_robot_to_adjacent = robot_dist_map.get(adjacent_loc, float('inf'))
                     min_robot_dist_to_adjacent = min(min_robot_dist_to_adjacent, dist_robot_to_adjacent)
            # If box_loc is not in graph or has no neighbors, robot cannot reach adjacent
            # This case is covered if min_robot_dist_to_adjacent remains inf

        # If robot cannot reach any location adjacent to any misplaced box
        if min_robot_dist_to_adjacent == float('inf'):
             return float('inf')

        h += min_robot_dist_to_adjacent

        return h
