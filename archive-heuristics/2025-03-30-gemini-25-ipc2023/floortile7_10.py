import itertools
from collections import defaultdict, deque
from fnmatch import fnmatch
import math # For infinity

# Try to import the base class provided by the planner environment.
# If it fails, define a dummy class for standalone execution/testing.
try:
    # This assumes the planner framework provides a base class
    # in the specified path.
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base not found. Using dummy Heuristic class.")
    # Define a dummy base class if the actual one is not available
    # This allows the code to be syntactically correct and potentially runnable
    # in isolation for basic checks, though it won't integrate with a planner.
    class Heuristic:
        def __init__(self, task):
            # Store the task object, which might contain goals, initial state, etc.
            self.task = task
        def __call__(self, node):
            # Base class method should be overridden by subclasses.
            # node typically contains the current state.
            raise NotImplementedError("Heuristic.__call__ must be implemented by subclasses.")


def get_parts(fact):
    """
    Extracts the components of a PDDL fact string.
    Removes the surrounding parentheses and splits the string by spaces.
    Example: "(predicate obj1 obj2)" -> ["predicate", "obj1", "obj2"]
    Returns an empty list if the fact is not a valid string, is empty,
    or does not follow the expected '(...) 'format.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        # Strip parentheses and split by whitespace
        return fact[1:-1].split()
    # Return empty list for invalid or non-matching input
    return []


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain floortile.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to paint
    all required tiles according to the goal state. It calculates the cost for
    each unsatisfied goal predicate `(painted ?tile ?color)` and sums these costs.
    The cost for a single goal is estimated as the minimum cost for any available
    robot to achieve it. This cost includes the estimated movement cost for the
    robot to reach a valid painting position adjacent to the target tile, the
    cost of changing color if necessary, and the cost of the painting action itself.
    It is designed for use with Greedy Best-First Search and does not need to be
    admissible.

    # Assumptions
    - The cost of moving between tiles is based on the shortest path distance in the
      tile connectivity graph (derived from static `up`, `down`, `left`, `right`
      predicates). This distance calculation ignores the dynamic `clear` status
      of intermediate tiles, acting as a relaxation. Distances are precomputed
      using Breadth-First Search (BFS).
    - Each goal (painting a specific tile with a specific color) is considered
      independently. The heuristic sums the minimum estimated cost for achieving
      each unsatisfied goal, assuming the most efficient robot (based on this
      heuristic's estimate) performs the task. This does not account for potential
      conflicts (e.g., two robots needing the same tile) or synergies (e.g., one
      robot painting multiple nearby tiles efficiently).
    - All actions (`move_*`, `paint_*`, `change_color`) have a uniform cost of 1.
    - The `clear` precondition for the tile *being painted* (`?y` in `paint_up`/
      `paint_down`) is ignored in the heuristic calculation. This simplifies the
      estimation, as determining the cost to make a tile clear can be complex and
      state-dependent.
    - The `clear` precondition for movement actions is also ignored when calculating
      shortest path distances (handled by the BFS relaxation on the static graph).

    # Heuristic Initialization (`__init__`)
    - Stores the goal predicates `(painted ?tile ?color)` from `task.goals`.
    - Parses static facts (`up`, `down`, `left`, `right`) to build an adjacency
      list representing the tile connectivity graph.
    - Identifies all unique tile objects mentioned in the static connectivity facts
      and potentially in the initial/goal states.
    - Determines the valid locations (`adj_tile`) from which each `goal_tile`
      can be painted based on the `up`/`down` static predicates. Stores this mapping
      in `self.paint_locations`. A robot must be at one of these `adj_tile`
      locations to paint the `goal_tile`.
    - Precomputes all-pairs shortest path distances between all identified tiles
      using BFS on the static connectivity graph. Stores distances in the
      `self.distances` dictionary. Unreachable pairs are implicitly assigned
      infinite distance (i.e., they won't have an entry or `get` will return default).
    - Identifies all robot objects defined in the task, checking `task.objects_by_type`
      first (if available in the task representation), then falling back to parsing
      the initial state predicates (`robot-at`, `robot-has`) if necessary.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  Parse the current state (`node.state`, which is a set of fact strings)
        to extract:
        - The current location of each robot: `robot_locations = {robot: tile}`.
        - The current color held by each robot: `robot_colors = {robot: color}`.
        - The set of currently painted tiles: `current_painted = {(tile, color)}`.
    2.  Identify the set of unsatisfied goals (`unsatisfied_goals`) by taking the
        set difference between the required goal predicates (`self.goal_predicates`)
        and the `current_painted` set from the state.
    3.  If `unsatisfied_goals` is empty, the painting goals are met, so the
        heuristic estimate is 0. Return 0.
    4.  Initialize `total_heuristic_value = 0`.
    5.  Check if any robots were identified during initialization (`self.robots`).
        If not, and there are unsatisfied goals, return infinity (`float('inf')`)
        because the goals cannot be achieved without robots.
    6.  Iterate through each unsatisfied goal `G = (goal_tile, goal_color)` in
        `unsatisfied_goals`:
        a.  Retrieve the precomputed list of adjacent tiles (`adjacent_paint_locations`)
            from `self.paint_locations` for the `goal_tile`. These are the tiles
            a robot must stand on to paint `goal_tile`.
        b.  If `adjacent_paint_locations` is empty (meaning `goal_tile` cannot be
            painted according to the domain's static `up`/`down` rules, or the
            tile is isolated), consider the goal unreachable from any state.
            Return infinity.
        c.  Initialize `min_cost_for_goal = float('inf')`. This variable will track
            the minimum estimated cost found so far among all robots for achieving
            this specific goal `G`.
        d.  Iterate through each robot `R` in `self.robots`:
            i.   Check if the current state provides location (`robot_locations`)
                 and color (`robot_colors`) information for robot `R`. If either
                 is missing, this robot's state is unknown in the current node;
                 skip this robot as it cannot be assessed.
            ii.  Get `R`'s current tile `robot_current_tile` and current color
                 `robot_current_color`.
            iii. Find the minimum movement cost (`min_move_dist`) required for
                 robot `R` to reach *any* of the `adjacent_paint_locations`. This
                 is done by looking up the precomputed distances in `self.distances`
                 from `robot_current_tile` to each `target_tile` in
                 `adjacent_paint_locations` and taking the minimum. Initialize
                 `min_move_dist` to infinity before checking.
            iv.  If `min_move_dist` remains infinity after checking all target tiles,
                 it means robot `R` cannot reach any valid painting location for
                 this goal from its current position (based on static connectivity).
                 Set the cost for this robot (`robot_cost`) to infinity.
            v.   If `min_move_dist` is finite (a path exists):
                 - Calculate the color change cost: `color_cost = 1` if
                   `robot_current_color` is different from `goal_color`, else `0`.
                 - Define the painting action cost: `paint_cost = 1`.
                 - Calculate the total estimated cost for robot `R` to achieve goal `G`:
                   `robot_cost = min_move_dist + color_cost + paint_cost`.
            vi.  Update the minimum cost found so far for achieving goal `G`:
                 `min_cost_for_goal = min(min_cost_for_goal, robot_cost)`.
        e.  After checking all robots, if `min_cost_for_goal` is still infinity,
            it implies that no robot can achieve this specific goal `G` from the
            current state (according to the heuristic's relaxed model). The state
            is likely a dead end regarding this goal. Return infinity.
        f.  Add the calculated `min_cost_for_goal` (which is finite at this point)
            to the `total_heuristic_value`.
    7.  After iterating through all unsatisfied goals, return the final computed
        `total_heuristic_value`.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and goals."""
        super().__init__(task) # Call base class constructor if needed
        self.goals = task.goals
        static_facts = task.static

        # Store goal predicates: {(tile, color)}
        self.goal_predicates = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure it's a valid 'painted' goal predicate
            if len(parts) == 3 and parts[0] == 'painted':
                self.goal_predicates.add((parts[1], parts[2]))

        # Extract tiles, build adjacency list, and paint locations
        self.tiles = set() # Set of all known tile names
        self.adj = defaultdict(list) # Adjacency list for tile graph
        # paint_locations[tile_to_paint] = [list of tiles robot must be at]
        self.paint_locations = defaultdict(list)

        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts

             pred = parts[0]
             # Process connectivity predicates (up, down, left, right)
             if pred in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                 tile1, tile2 = parts[1], parts[2]
                 # Add tiles to the set of known tiles
                 self.tiles.add(tile1)
                 self.tiles.add(tile2)
                 # Add edges for movement. Assuming connectivity is bidirectional
                 # for movement, but check domain specifics if needed.
                 # BFS naturally handles directed graphs if adj list is built accordingly.
                 # For distance, usually treat as undirected unless specified otherwise.
                 self.adj[tile1].append(tile2)
                 self.adj[tile2].append(tile1) # Assume symmetric movement possibility

                 # Store paint locations based on up/down predicates
                 # (up target source) means robot AT source paints target
                 if pred == 'up':
                     self.paint_locations[tile1].append(tile2)
                 # (down target source) means robot AT source paints target
                 elif pred == 'down':
                     self.paint_locations[tile1].append(tile2)

        # Identify robots
        self.robots = set()
        # Try to get robots from task object types if available (preferred method)
        # This depends on the planner's Task object structure.
        if hasattr(task, 'objects_by_type') and isinstance(task.objects_by_type, dict):
             # Assumes task.objects_by_type is a dict like {'robot': ['r1', 'r2'], 'tile': [...]}
             self.robots = set(task.objects_by_type.get('robot', []))

        # Fallback: Infer robots from initial state predicates if not found above
        # or if the task object doesn't provide typed objects.
        if not self.robots:
            # print("Info: Inferring robots from initial state predicates.") # Optional debug info
            for fact in task.initial_state:
                parts = get_parts(fact)
                if not parts: continue
                # Identify robots from predicates like (robot-at r t) or (robot-has r c)
                if len(parts) >= 2 and parts[0] in ('robot-at', 'robot-has'):
                    self.robots.add(parts[1]) # Add the robot name (second element)

        if not self.robots:
             print("Warning: No robots identified in the task. Heuristic might return infinity if goals exist.")


        # Precompute all-pairs shortest paths using BFS
        self.distances = {} # Stores distances: (tile1, tile2) -> distance

        # Ensure all tiles identified are processed, even if isolated or only in init/goal
        all_known_tiles = self.tiles.copy()
        # Add tiles mentioned in initial state but not in connectivity facts
        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            if len(parts) >= 2 and parts[0] in ['robot-at', 'clear', 'painted']:
                 tile_name = parts[1]
                 # Add if it's a tile and not already known via connectivity
                 # (We might need a way to confirm parts[1] is a tile if types aren't available)
                 # For now, assume if it appears here, it's a tile.
                 all_known_tiles.add(tile_name)
        # Add tiles mentioned in goals but not elsewhere
        for tile, _ in self.goal_predicates:
             all_known_tiles.add(tile)


        # Run BFS from each known tile to find shortest paths to all reachable tiles
        for start_node in all_known_tiles:
            # Skip if start_node is somehow invalid (e.g., empty string)
            if not start_node: continue

            # Initialize BFS queue and visited dictionary for this start_node
            queue = deque([(start_node, 0)]) # Store (tile, distance_from_start)
            # visited_dist keeps track of visited nodes and their shortest distance found so far
            visited_dist = {start_node: 0}
            # Store distance from start_node to itself (0)
            self.distances[(start_node, start_node)] = 0

            while queue:
                curr_tile, dist = queue.popleft()

                # Explore neighbors using the prebuilt adjacency list
                # Use .get(curr_tile, []) to handle tiles with no outgoing connections gracefully
                for neighbor in self.adj.get(curr_tile, []):
                    # Process neighbor only if it's a known tile and hasn't been visited yet
                    # from this start_node
                    if neighbor in all_known_tiles and neighbor not in visited_dist:
                        # Mark neighbor as visited and record distance
                        visited_dist[neighbor] = dist + 1
                        self.distances[(start_node, neighbor)] = dist + 1
                        # Add neighbor to the queue for further exploration
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given node."""
        state = node.state # The current state is a frozenset of PDDL fact strings

        # Parse current state efficiently to get dynamic information
        robot_locations = {} # robot -> tile
        robot_colors = {}    # robot -> color
        # Store currently painted tiles as {(tile, color)} for quick comparison with goals
        current_painted = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            # Extract robot locations
            if predicate == 'robot-at' and len(parts) == 3:
                robot_locations[parts[1]] = parts[2] # parts[1]=robot, parts[2]=tile
            # Extract robot colors
            elif predicate == 'robot-has' and len(parts) == 3:
                robot_colors[parts[1]] = parts[2] # parts[1]=robot, parts[2]=color
            # Extract painted tiles
            elif predicate == 'painted' and len(parts) == 3:
                current_painted.add((parts[1], parts[2])) # parts[1]=tile, parts[2]=color
            # Note: 'clear' predicate is not used in this heuristic version

        # Identify unsatisfied goals by set difference
        unsatisfied_goals = self.goal_predicates - current_painted

        # If there are no unsatisfied goals, the goal state (regarding painting) is reached
        if not unsatisfied_goals:
            return 0

        total_heuristic_value = 0

        # Check if robots were identified during initialization
        if not self.robots:
             # If there are goals but no robots, goals are unreachable
             return float('inf') if unsatisfied_goals else 0

        # Calculate estimated cost for each unsatisfied goal
        for goal_tile, goal_color in unsatisfied_goals:
            # Find the precomputed list of locations from where goal_tile can be painted
            adjacent_paint_locations = self.paint_locations.get(goal_tile, [])

            # If a goal tile has no adjacent locations from which it can be painted
            # (based on static up/down facts), it's impossible to achieve this goal.
            if not adjacent_paint_locations:
                # print(f"Warning: No adjacent paint locations found for tile {goal_tile}") # Optional debug
                return float('inf') # Goal is statically unreachable

            min_cost_for_goal = float('inf') # Min cost found so far for this goal

            # Find the minimum cost among all robots to achieve this goal
            for robot in self.robots:
                # Ensure this robot's current location and color are known from the state
                if robot not in robot_locations or robot not in robot_colors:
                    # If state info is missing for this robot, skip it.
                    # print(f"Warning: State for robot {robot} not found in current state.") # Optional debug
                    continue

                robot_current_tile = robot_locations[robot]
                robot_current_color = robot_colors[robot]

                # Find the minimum movement distance from the robot's current tile
                # to any of the required adjacent painting locations.
                min_move_dist = float('inf')
                for target_tile in adjacent_paint_locations:
                    # Look up precomputed distance. Use .get() with a default of infinity
                    # to handle cases where the target is unreachable from the current tile.
                    dist = self.distances.get((robot_current_tile, target_tile), float('inf'))
                    min_move_dist = min(min_move_dist, dist)

                # Calculate the total cost for this robot if a path exists
                if min_move_dist != float('inf'):
                    # Cost to change color (1 if needed, 0 otherwise)
                    color_change_cost = 1 if robot_current_color != goal_color else 0
                    # Cost of the paint action itself
                    paint_action_cost = 1
                    # Total estimated cost for this robot to achieve this goal
                    cost_for_this_robot = min_move_dist + color_change_cost + paint_action_cost
                else:
                    # If no path exists for this robot to reach a painting location
                    cost_for_this_robot = float('inf')

                # Update the minimum cost found for this goal across all robots
                min_cost_for_goal = min(min_cost_for_goal, cost_for_this_robot)

            # After checking all robots, if min_cost_for_goal is still infinity,
            # it means no robot can reach a position to paint this goal tile
            # from the current state (based on the heuristic's model).
            if min_cost_for_goal == float('inf'):
                # print(f"Warning: Goal ({goal_tile} {goal_color}) deemed unreachable by any robot.") # Optional debug
                return float('inf') # Indicate state is likely a dead end

            # Add the minimum cost found for this goal to the total heuristic value
            total_heuristic_value += min_cost_for_goal

        # Return the final summed heuristic value.
        # It should be non-negative by construction.
        return total_heuristic_value
