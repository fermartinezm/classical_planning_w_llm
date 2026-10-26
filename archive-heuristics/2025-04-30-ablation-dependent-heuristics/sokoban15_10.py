# Import necessary modules
import collections
import math

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        contributions from each box that is not yet at its goal location.
        The contribution for a single box is the sum of:
        1. The shortest path distance (number of pushes) required for the box
           to reach its goal location in the grid graph (ignoring other boxes
           as obstacles for the box's path itself).
        2. The minimum shortest path distance (number of robot moves) required
           for the robot to reach a location from which it can perform the
           first push of the box towards its goal, considering other boxes
           and walls as obstacles for the robot's movement.

    Assumptions:
        - The domain defines a grid-like structure using `adjacent` facts.
        - Each box has a unique goal location specified in the task goals.
        - Locations are named in a way that allows parsing (though not strictly
          used for grid coordinates in this implementation, only as identifiers).
        - The `adjacent` facts define an undirected graph (if `(adjacent l1 l2 d)`
          is true, then `(adjacent l2 l1 opposite(d))` is also true).

    Heuristic Initialization:
        The `__init__` method processes the static information from the task:
        - It builds an adjacency list (`self.adj_list`) representing the grid
          connectivity based on `adjacent` facts. This is used for calculating
          box-goal distances.
        - It builds a second adjacency list (`self.adj_list_with_dirs`) that
          also stores the direction of movement, used to determine the required
          robot position for a push.
        - It extracts the goal location for each box from the task's goal facts
          and stores it in `self.box_goals`.
        - It creates a mapping for opposite directions.

    Step-By-Step Thinking for Computing Heuristic:
        The `__call__` method computes the heuristic for a given state:
        1.  Parse the input `state` to identify the robot's current location,
            the current location of each box, and the set of clear locations.
        2.  Initialize the `total_heuristic` to 0.
        3.  Perform a Breadth-First Search (BFS) starting from the robot's
            current location to calculate the shortest distance for the robot
            to reach all other reachable locations that are currently clear.
            Store these distances in `robot_distances`.
        4.  Pre-calculate distances from all locations to each unique goal location
            by performing BFS from each unique goal location on the full grid graph.
            Store these in `all_distances_to_goals`.
        5.  Iterate through each box and its corresponding goal location stored
            during initialization.
        6.  If a box is already at its goal location, its contribution to the
            heuristic is 0, so continue to the next box.
        7.  If the box is not at its goal:
            a.  Retrieve the pre-calculated distances to the box's goal location
                (`distances_to_goal`).
            b.  Get the box's shortest path distance (`box_dist`) from its current
                location to its goal using `distances_to_goal`. If the goal is
                unreachable for the box in the grid, the state is likely unsolvable,
                return `math.inf`.
            c.  Identify potential locations (`next_box_loc`) adjacent to the
                current box location that lie on a shortest path from the box's
                current location to its goal. A neighbor `next_box_loc` is on
                a shortest path if its distance *to* the goal (`distances_to_goal[next_box_loc]`)
                is one less than the box's distance *to* the goal (`box_dist - 1`).
            d.  For each such `next_box_loc`, determine the required location
                (`push_loc`) where the robot must be positioned to push the box
                from its current location to `next_box_loc`. This `push_loc` is
                adjacent to the box's current location in the direction opposite
                to the push direction.
            e.  Find the minimum distance (`min_robot_dist`) from the robot's
                current location to any of the identified `push_loc` candidates,
                using the pre-calculated `robot_distances`. If the robot cannot
                reach any valid `push_loc` (e.g., blocked by other objects),
                the state is likely unsolvable, return `math.inf`.
            f.  Add `box_dist + min_robot_dist` to the `total_heuristic`.
        8.  After processing all boxes, return the `total_heuristic`.
    """
    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        Args:
            task: The planning task object.
        """
        self.adj_list = collections.defaultdict(list)
        self.adj_list_with_dirs = collections.defaultdict(list)
        self.box_goals = {}
        self.locations = set() # Store all location names

        # Parse static adjacent facts to build graph
        for fact in task.static:
            if fact.startswith('(adjacent '):
                parts = fact.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                self.adj_list[loc1].append(loc2)
                self.adj_list_with_dirs[loc1].append((loc2, direction))
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Parse goal facts to map boxes to goal locations
        # Assuming goals are conjunctions of (at box loc)
        goal_facts = task.goals
        for fact in goal_facts:
             if fact.startswith('(at '):
                 parts = fact.strip('()').split()
                 box = parts[1]
                 loc = parts[2]
                 self.box_goals[box] = loc

        self.opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


    def _bfs(self, start, goal, adj_list):
        """
        Performs BFS to find the shortest distance between two locations
        in the given adjacency list graph.
        Returns distance or math.inf if unreachable.
        """
        if start == goal:
            return 0
        queue = collections.deque([(start, 0)])
        visited = {start}
        while queue:
            current_loc, dist = queue.popleft()
            if current_loc == goal:
                return dist
            for neighbor in adj_list.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return math.inf # Goal unreachable

    def _bfs_to_goal(self, goal, adj_list):
        """
        Performs BFS from the goal location to find distances from all
        reachable locations to the goal in the given adjacency list graph.
        Returns a dictionary {location: distance_to_goal}.
        """
        distances = {goal: 0}
        queue = collections.deque([goal])
        visited = {goal}

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # In an undirected graph (like Sokoban grid), distance from A to B
            # is the same as distance from B to A. So BFS from goal finds
            # distances *from* goal, which are distances *to* goal.
            for neighbor in adj_list.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances


    def _robot_bfs_all(self, start, adj_list, clear_locations):
        """
        Performs BFS from start to find distances to all reachable clear locations
        for the robot.
        Returns a dictionary {location: distance}.
        """
        distances = {start: 0}
        queue = collections.deque([start])
        visited = {start}

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            for neighbor in adj_list.get(current_loc, []):
                # Robot can move to a neighbor if the neighbor is clear
                if neighbor not in visited and neighbor in clear_locations:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances


    def __call__(self, state):
        """
        Computes the domain-dependent heuristic for a given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The heuristic value (estimated cost to reach the goal).
            Returns math.inf if the state is estimated to be unsolvable.
        """
        robot_loc = None
        box_locations = {} # box_name -> location
        clear_locations = set() # set of location names

        # Parse state facts
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at':
                box = parts[1]
                loc = parts[2]
                box_locations[box] = loc
            elif parts[0] == 'clear':
                loc = parts[1]
                clear_locations.add(loc)

        total_heuristic = 0

        # Calculate robot distances to all reachable clear locations once per state
        robot_distances = self._robot_bfs_all(robot_loc, self.adj_list, clear_locations)

        # Pre-calculate distances from all locations to each unique goal location
        # This is done once per state for all boxes.
        # Mapping: goal_loc -> {location: distance_to_goal}
        all_distances_to_goals = {}
        for goal_loc in set(self.box_goals.values()): # Get unique goal locations
             all_distances_to_goals[goal_loc] = self._bfs_to_goal(goal_loc, self.adj_list)


        for box, goal_loc in self.box_goals.items():
            current_box_loc = box_locations.get(box)

            if current_box_loc is None:
                 # This state is malformed, box is missing. Treat as unsolvable.
                 return math.inf

            if current_box_loc == goal_loc:
                continue # Box is already at goal

            # Get pre-calculated distances to this goal
            distances_to_goal = all_distances_to_goals.get(goal_loc)
            if distances_to_goal is None:
                 # Should not happen if goal_loc is in self.box_goals.values()
                 # but adding check for robustness
                 return math.inf # Error state

            # 1. Calculate box distance to goal (minimum pushes)
            # We can get this from the pre-calculated distances_to_goal
            box_dist = distances_to_goal.get(current_box_loc, math.inf)

            if box_dist == math.inf:
                # Box cannot reach its goal location in the grid
                return math.inf

            # 2. Find potential robot push locations for the first step towards goal
            robot_push_locs = set()
            # Find neighbors of current_box_loc that are on a shortest path to goal_loc
            potential_next_box_locs = []
            for next_box_loc in self.adj_list.get(current_box_loc, []):
                 # Check if next_box_loc is on a shortest path from current_box_loc to goal_loc
                 # This is true if dist(next_box_loc, goal_loc) == dist(current_box_loc, goal_loc) - 1
                 dist_next = distances_to_goal.get(next_box_loc, math.inf)
                 if dist_next == box_dist - 1:
                     potential_next_box_locs.append(next_box_loc)

            # If no neighbor is on a shortest path (and box is not at goal)
            # This check might be too strict. Maybe any push that reduces distance is okay?
            # Let's stick to shortest path neighbors for a more informed heuristic.
            if not potential_next_box_locs and box_dist > 0:
                 # Box cannot move along a shortest path from its current position.
                 # This state might be a deadlock or require complex maneuvers.
                 # For a non-admissible heuristic, returning infinity is reasonable.
                 return math.inf


            # Find the required robot push locations for these potential next box locations
            for next_box_loc in potential_next_box_locs:
                 # Find the direction from current_box_loc to next_box_loc
                 direction_to_next = None
                 for neighbor_loc, direction in self.adj_list_with_dirs.get(current_box_loc, []):
                     if neighbor_loc == next_box_loc:
                         direction_to_next = direction
                         break

                 if direction_to_next:
                     # Find the location robot needs to be at to push from current_box_loc to next_box_loc
                     # Robot must be adjacent to current_box_loc in the opposite direction
                     required_robot_dir = self.opposite_direction.get(direction_to_next)
                     if required_robot_dir:
                         for neighbor_loc, direction in self.adj_list_with_dirs.get(current_box_loc, []):
                             if direction == required_robot_dir:
                                 robot_push_locs.add(neighbor_loc)
                                 break # Found the push location for this direction

            # If no valid push locations found (e.g., adjacent spot for robot is a wall)
            if not robot_push_locs and box_dist > 0:
                 return math.inf # Box cannot be pushed towards its goal

            # 3. Calculate minimum robot distance to reach any of the push locations
            min_robot_dist = math.inf
            reachable_push_locs = False
            for push_loc in robot_push_locs:
                if push_loc in robot_distances:
                    min_robot_dist = min(min_robot_dist, robot_distances[push_loc])
                    reachable_push_locs = True

            if not reachable_push_locs and box_dist > 0:
                 # Robot cannot reach any valid push location for this box
                 return math.inf

            # Heuristic contribution for this box: box_distance + robot_distance_to_push_position
            total_heuristic += box_dist + min_robot_dist

        # Heuristic is 0 if all boxes are at their goals (handled by the loop skipping)
        return total_heuristic
