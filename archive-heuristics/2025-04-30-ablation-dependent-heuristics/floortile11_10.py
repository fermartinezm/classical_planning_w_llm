import math
from collections import defaultdict, deque

# Assume the Task and Operator classes are available from code-file-task
# from task import Task, Operator # This line is commented out as per instructions

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing, for each goal tile that is
    not yet painted with the correct color, the minimum estimated cost for any
    robot to paint that tile. The estimated cost for a single robot to paint
    a specific tile includes the cost to move to an adjacent tile, the cost
    to change color if necessary, and the cost of the paint action itself.
    Movement cost is estimated using shortest path distances on the static
    tile grid graph, ignoring dynamic state changes like tiles becoming
    unclear.

    Assumptions:
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates
      is static and can be represented as a graph where tiles are nodes and
      these predicates define edges.
    - Movement is possible between any two adjacent tiles in the static grid
      graph, ignoring the 'clear' predicate for distance calculation.
    - Robots always possess a color in any reachable state (as implied by
      the domain structure where 'robot-has' is in the initial state and
      only changed via 'change_color').
    - 'available-color' facts are static.
    - The goal only requires painting tiles that are initially clear.
    - The problem is solvable (or unreachable goal tiles incur a large penalty).

    Heuristic Initialization:
    The heuristic is initialized with the planning task object.
    1. The goal facts (specifically, the target painted states) are stored in `self.goals`.
    2. A graph representing the tile grid connectivity is built from the
       static 'up', 'down', 'left', 'right' facts. This graph is stored as
       an adjacency list `self.tile_graph` where keys are tile names and
       values are lists of adjacent tile names. The graph is treated as
       undirected (edges added in both directions).
    3. Available colors are stored in `self.available_colors` from static
       'available-color' facts (though not directly used in the current
       heuristic calculation logic).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the current state satisfies all goal conditions (`self.goals <= state`). If yes, the
       heuristic value is 0.
    2. Identify all goal facts of the form '(painted tile color)' that are
       not present in the current state. These represent the tiles that still
       need to be painted with a specific color. Store them as a list of
       (tile, color) tuples in `unpainted_goals`.
    3. Determine the current location and the color held by each robot from
       the facts in the current state. Store these in `robot_locations` and
       `robot_colors` dictionaries.
    4. For each robot whose location is a valid tile in the graph, compute the
       shortest path distance from its current location to every other reachable
       tile in the static tile grid graph using Breadth-First Search (BFS).
       Store these distances in `robot_distances`, a dictionary mapping robot
       names to their respective distance dictionaries.
    5. Initialize the total heuristic value `h` to 0.
    6. For each `(target_tile, target_color)` tuple in the `unpainted_goals` list:
        a. Find all tiles `adj_t` that are directly adjacent to `target_tile`
           based on the static tile graph. Store them in `adj_tiles`.
        b. If `target_tile` has no adjacent tiles, it cannot be painted. Add a
           large penalty (`self.unreachable_penalty`) to `h` and continue to the
           next unpainted goal tile.
        c. Initialize `min_cost_for_tile` to infinity (`float('inf')`). This
           variable will store the minimum estimated cost for *any* robot to
           paint tile `target_tile`.
        d. For each robot `r` present in `robot_locations`:
            i. Get the robot's current location `loc_r` and color `color_r`.
            ii. Calculate the estimated cost for the robot to acquire the
                target color `target_color`: 0 if `color_r` is already
                `target_color`, and 1 otherwise. Store this in `color_cost`.
            iii. If BFS distances are available for robot `r` (i.e., `r` is in
                 `robot_distances`), calculate the minimum movement cost for
                 robot `r` to reach *any* tile in the set of adjacent tiles
                 `adj_tiles`. This is the minimum distance from `loc_r` to any
                 tile in `adj_tiles` found in the BFS results (`dists_from_robot`).
                 Initialize to infinity if no adjacent tile is reachable from the robot.
            iv. If `min_move_cost_for_robot` is not infinity, calculate the total
                estimated cost for robot `r` to paint tile `target_tile`:
                `min_move_cost_for_robot + color_cost + 1` (where +1 is for the
                paint action itself).
            v. Update `min_cost_for_tile` with the minimum of its current
               value and the cost calculated for robot `r`.
        e. Add `min_cost_for_tile` to the total heuristic value `h`. If
           `min_cost_for_tile` remained infinity (meaning no robot can reach
           an adjacent tile), add a large penalty (`self.unreachable_penalty`) to `h`.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        self.tile_graph = defaultdict(list)
        self.available_colors = set()

        # Build tile graph and get available colors from static facts
        for fact_str in task.static:
            predicate, args = self._parse_fact(fact_str)
            if predicate in ('up', 'down', 'left', 'right'):
                # Assuming these predicates define bidirectional connections
                tile1, tile2 = args
                self.tile_graph[tile1].append(tile2)
                self.tile_graph[tile2].append(tile1)
            elif predicate == 'available-color':
                self.available_colors.add(args[0])

        # Remove duplicates from adjacency lists (optional, BFS handles cycles)
        # for tile in self.tile_graph:
        #     self.tile_graph[tile] = list(set(self.tile_graph[tile]))

        # Determine a large penalty value based on graph size
        num_tiles = len(self.tile_graph)
        # Max possible finite cost for one tile: max_dist + color_change + paint
        # Max dist is num_tiles - 1. So max cost is (num_tiles - 1) + 1 + 1 = num_tiles + 1.
        # Penalty should be larger than this.
        self.unreachable_penalty = num_tiles * 2 + 2 if num_tiles > 0 else 1000


    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The estimated number of actions to reach the goal.
        """
        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify unpainted goal tiles
        unpainted_goals = []
        for goal_fact in self.goals:
            if goal_fact.startswith('(painted '):
                if goal_fact not in state:
                    _, tile, color = self._parse_fact(goal_fact)
                    unpainted_goals.append((tile, color))

        # If no unpainted goals, but not goal state, something is wrong or
        # goals include non-painted facts. Assuming goals are only painted facts.
        # This check should ideally be redundant due to the initial goal check.
        if not unpainted_goals:
             return 0


        # 3. Identify robot info
        robot_locations = {}
        robot_colors = {}
        for fact_str in state:
            if fact_str.startswith('(robot-at '):
                _, robot, tile = self._parse_fact(fact_str)
                robot_locations[robot] = tile
            elif fact_str.startswith('(robot-has '):
                _, robot, color = self._parse_fact(fact_str)
                robot_colors[robot] = color

        # 4. Compute distances from each robot using BFS
        robot_distances = {}
        for robot, loc in robot_locations.items():
             # Only run BFS if the robot's location is a valid tile in the graph
            if loc in self.tile_graph:
                robot_distances[robot] = self._bfs(loc)
            # else: Robot is in an unknown location, cannot compute distances
            # This robot will be skipped in the next step.

        # 5. Calculate heuristic sum
        h = 0

        for target_tile, target_color in unpainted_goals:
            # 6a. Find adjacent tiles
            adj_tiles = set(self.tile_graph.get(target_tile, []))

            # 6b. If target_tile has no adjacent tiles, it's impossible to paint
            if not adj_tiles:
                 h += self.unreachable_penalty
                 continue

            # 6c. Initialize min cost for this tile
            min_cost_for_tile = float('inf')

            # 6d. Find min cost over all robots for this tile
            for robot in robot_locations:
                loc_r = robot_locations[robot]
                color_r = robot_colors.get(robot) # Get robot's current color

                # Check if robot_distances contains info for this robot
                if robot not in robot_distances:
                    continue # Skip robot if BFS failed (e.g., starting in unknown location)

                dists_from_robot = robot_distances[robot]

                # 6d.ii. Calculate color cost
                # Assuming robot always has a color based on domain analysis
                # If target_color is not available, this path is impossible, but
                # the domain implies goal colors are available.
                color_cost = 0 if color_r == target_color else 1

                # 6d.iii. Calculate min movement cost for this robot to any adjacent tile
                min_move_cost_for_robot = float('inf')
                for adj_t in adj_tiles:
                    if adj_t in dists_from_robot:
                        min_move_cost_for_robot = min(min_move_cost_for_robot, dists_from_robot[adj_t])

                # 6d.iv. Calculate total cost for this robot to paint this tile
                if min_move_cost_for_robot != float('inf'):
                    cost_for_robot = min_move_cost_for_robot + color_cost + 1 # +1 for paint action
                    # 6d.v. Update min cost for this tile
                    min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # 6e. Add min cost for this tile to total heuristic
            h += min_cost_for_tile if min_cost_for_tile != float('inf') else self.unreachable_penalty

        # 7. Return total heuristic value
        return h

    def _parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string."""
        # Remove surrounding brackets and split by spaces
        parts = fact_str.strip('()').split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _bfs(self, start_node):
        """
        Performs BFS from start_node to find distances to all reachable nodes.

        Args:
            start_node: The tile name to start BFS from.

        Returns:
            A dictionary mapping reachable tile names to their shortest distance
            from start_node. Returns an empty dictionary if start_node is not
            in the graph.
        """
        if start_node not in self.tile_graph:
            return {} # Cannot start BFS from a node not in the graph

        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            curr_tile = queue.popleft()
            dist = distances[curr_tile]

            # Get neighbors from the pre-built graph
            neighbors = self.tile_graph.get(curr_tile, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)

        return distances
