# Standard library imports
import collections

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the shortest path distances for each box from its current location
        to its goal location. The shortest path is computed on the static
        grid graph defined by the 'adjacent' predicates, ignoring the robot's
        position and other dynamic obstacles. This heuristic is not admissible
        as it does not account for the robot's movement cost or the cost of
        moving blocking boxes out of the way. It serves as an estimate of the
        minimum number of *pushes* required if the path was clear and the
        robot was always in position.

    Assumptions:
        - The locations form a connected graph (or at least, each box's
          current location and its goal location are in the same connected
          component). If a goal is unreachable from a box's location, the
          distance will be infinity, resulting in an infinite heuristic value.
        - The 'adjacent' predicates define a symmetric relationship, meaning
          if A is adjacent to B in direction D, then B is adjacent to A
          in the opposite direction. The heuristic builds an undirected graph
          based on these adjacencies.
        - Location names start with 'loc_'.
        - Box names start with 'box'.
        - Robot location fact is '(at-robot ...)'.
        - Box location facts are '(at box... ...)'.
        - Goal facts are '(at box... ...)'.
        - Static facts include 'adjacent' predicates.

    Heuristic Initialization:
        1. Parse all location names from the initial state, goals, and static facts.
        2. Parse the goal location for each box from the task's goals. Store this
           mapping (box name -> goal location name).
        3. Build an adjacency list representation of the grid graph from the
           static 'adjacent' predicates. Since movement is possible in both
           directions between adjacent locations, add edges for both directions
           (l1 -> l2 and l2 -> l1) for each '(adjacent l1 l2 dir)' fact.
        4. Compute all-pairs shortest paths on this grid graph using Breadth-First
           Search (BFS) starting from every location. Store the distances in a
           dictionary where `distances[loc1][loc2]` is the shortest path distance
           between `loc1` and `loc2`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Given a state (a frozenset of facts), identify the current location
           of each box by parsing the '(at box... ...)' facts in the state.
        2. Initialize the total heuristic value `h` to 0.
        3. For each box that has a specified goal location:
            a. Get the box's current location from the parsed state information.
            b. Get the box's goal location from the precomputed goal mapping.
            c. If the box is already at its goal location, its contribution to
               the heuristic is 0.
            d. If the box is not at its goal location, look up the shortest path
               distance between its current location and its goal location in the
               precomputed distances table.
            e. Add this distance to the total heuristic value `h`. If the goal
               location is unreachable from the box's current location, the
               distance will be infinity, and the total heuristic will become
               infinity.
        4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing box goals and shortest paths.

        Args:
            task: The planning task object containing initial_state, goals, and static facts.
        """
        self.box_goals = {}
        self.adj_list = collections.defaultdict(list)
        self.all_locations = set()

        # 1. Collect all locations from initial state, goals, and static facts
        for fact in task.initial_state:
            self.all_locations.update(self._extract_locations(fact))
        for fact in task.goals:
            self.all_locations.update(self._extract_locations(fact))
        for fact in task.static:
            self.all_locations.update(self._extract_locations(fact))

        # 2. Parse box goals from task goals
        for goal_fact in task.goals:
            parts = self._parse_fact(goal_fact)
            # Goal facts are expected to be of the form '(at box_name loc_name)'
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                self.box_goals[parts[1]] = parts[2]

        # 3. Build adjacency list graph from static adjacent facts
        for static_fact in task.static:
            parts = self._parse_fact(static_fact)
            # Adjacent facts are expected to be of the form '(adjacent loc1 loc2 dir)'
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                # Assuming adjacency is symmetric for movement, add both directions
                self.adj_list[loc1].append(loc2)
                self.adj_list[loc2].append(loc1)

        # Ensure all locations from all_locations are keys in adj_list, even if isolated
        # This prevents errors in BFS if a location has no adjacencies listed
        for loc in self.all_locations:
             if loc not in self.adj_list:
                 self.adj_list[loc] = []

        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_locations:
            self.distances[start_node] = self._bfs(start_node, self.adj_list, self.all_locations)

    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove surrounding brackets and split by space
        if fact_string.startswith('(') and fact_string.endswith(')'):
            return fact_string[1:-1].split()
        return [] # Return empty list for malformed facts

    def _extract_locations(self, fact_string):
        """Helper to extract location names from a fact string."""
        parts = self._parse_fact(fact_string)
        # Locations are assumed to start with 'loc_'
        return [p for p in parts if p.startswith('loc_')]

    def _bfs(self, start_node, adj_list, all_locations):
        """Computes shortest path distances from start_node to all other nodes."""
        q = collections.deque([start_node])
        # Initialize distances for all known locations
        dists = {node: float('inf') for node in all_locations}

        # If the start node isn't in our list of known locations, something is wrong.
        # Return infinite distances as a safe default.
        if start_node not in all_locations:
             return dists

        dists[start_node] = 0

        while q:
            curr = q.popleft()
            # Iterate over neighbors using adj_list.get(curr, []) to handle potential missing keys safely,
            # although we've added all_locations as keys in __init__.
            for neighbor in adj_list.get(curr, []):
                # neighbor is guaranteed to be in all_locations and thus in dists
                if dists[neighbor] == float('inf'):
                    dists[neighbor] = dists[curr] + 1
                    q.append(neighbor)
        return dists

    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        Args:
            state: A frozenset of facts representing the current state.

        Returns:
            The estimated cost (integer or float('inf')) to reach a goal state.
            Returns 0 if the state is a goal state. Returns float('inf') if
            any box goal is unreachable from its current location.
        """
        current_box_locations = {}

        # 1. Find current box locations from the state facts
        for fact in state:
            parts = self._parse_fact(fact)
            # Current box locations are facts like '(at box_name loc_name)'
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                if obj_name.startswith('box'):
                    current_box_locations[obj_name] = loc_name
            # We don't need the robot location for this heuristic

        # 2. Calculate sum of box-to-goal distances
        h = 0
        for box, goal_l in self.box_goals.items():
            current_l = current_box_locations.get(box)

            # Check if the box's current location is known and valid within our graph
            # and if the goal location is also known.
            # current_l should be in self.all_locations if state is valid.
            # goal_l should be in self.all_locations if task is valid.
            # self.distances[current_l] contains distances to all locations in self.all_locations.
            if current_l is None or current_l not in self.distances or goal_l not in self.distances[current_l]:
                 # This indicates an unreachable goal or a state inconsistency
                 # (e.g., a box is at a location not part of the known graph).
                 # Treat this as infinite cost.
                 return float('inf')

            dist = self.distances[current_l][goal_l]

            # If the shortest path distance is infinity, the goal is unreachable for this box
            if dist == float('inf'):
                return float('inf')

            h += dist

        return h
