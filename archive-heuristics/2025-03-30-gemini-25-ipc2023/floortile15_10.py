import itertools
import collections
from fnmatch import fnmatch
# Assuming the planner infrastructure provides the Heuristic base class
# If not, define a simple base class:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError
try:
    # Attempt to import the standard Heuristic base class
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        """A base class for heuristics, compatible with the expected interface."""
        def __init__(self, task):
            self.task = task # Store the task description
        def __call__(self, node):
            """Evaluates the heuristic function for a given state node."""
            # A node typically has attributes like node.state
            raise NotImplementedError("Heuristic evaluation not implemented.")


# Helper functions
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(at robot1 tile1)" -> ["at", "robot1", "tile1"]
    """
    # Handles potential extra spaces inside, removes outer parentheses
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """Checks if a fact string matches a pattern with optional wildcards ('*').
    Compares the predicate and arguments.
    Example: match("(at robot1 tile1)", "at", "*", "tile1") -> True
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the pattern length
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element (allowing wildcards)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile PDDL domain.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to reach the goal
    state, where specific tiles must be painted with specific colors. It decomposes the
    problem by calculating the estimated cost for each unsatisfied goal predicate
    `(painted tile color)` individually and summing these costs. The cost for a single
    goal is estimated as the minimum cost required for *any* available robot to achieve
    that painting task. This minimum cost calculation considers three components:
    1.  The cost of changing color (1 action), if the robot doesn't currently hold the required color.
    2.  The cost of moving the robot (N actions) from its current location to a tile
        from which it can paint the target tile (i.e., a tile directly below or above the target tile,
        based on `up`/`down` predicates). The minimum number of moves across all valid painting
        locations is used.
    3.  The cost of the painting action itself (1 action).
    The heuristic aims for informativeness for Greedy Best-First Search, not necessarily admissibility.

    # Assumptions
    - The tile layout is defined by static `up`, `down`, `left`, `right` predicates, forming a graph
      where tiles are nodes and adjacencies are edges. This graph might be disconnected.
    - All colors specified in the goal state are available colors in the domain (i.e., an
      `available-color` predicate exists for them), otherwise `change_color` would be impossible.
    - Robots always hold exactly one color. The `change_color` action swaps the current color for another.
    - All actions (`move_*`, `paint_*`, `change_color`) have a uniform cost of 1.
    - The heuristic ignores potential conflicts and interactions between robots. For example, it doesn't
      account for the fact that a tile must be `clear` to be moved onto or painted, nor does it model
      robots blocking each other's paths. It calculates costs assuming robots can act independently.
    - Tiles, once painted, cannot change color. If a tile is painted with the wrong color relative
      to a goal, that goal becomes permanently unreachable.

    # Heuristic Initialization
    - The constructor (`__init__`) performs preprocessing based on the task definition (`task`).
    - It parses static facts (`up`, `down`, `left`, `right`) to build an adjacency map (`self.adj`)
      for tile connectivity, used for calculating movement distances.
    - It determines, for each tile, the set of adjacent locations a robot must occupy to paint it
      (based only on `up` and `down` facts, as per `paint_up`/`paint_down` actions) and stores
      this in `self.paint_locations`.
    - It computes all-pairs shortest path distances between all known tiles using Breadth-First Search (BFS)
      run from each tile. These distances (number of moves) are stored in `self.dist`. Unreachable pairs
      are implicitly handled (lookup returns infinity).
    - It extracts the target state conditions, specifically the `(painted tile color)` facts,
      into a set `self.goal_tuples`.
    - It identifies all unique `robot`, `tile`, and `color` objects from the static facts, initial state,
      and goals to ensure all relevant entities are known.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** In the `__call__` method, process the input `node.state` (a set of facts)
        to determine the current location (`robot_loc`) and held color (`robot_col`) for each robot,
        and the current color of each painted tile (`painted_state`).
    2.  **Check for Unsolvable State (Wrong Color):** Iterate through the required goal paintings
        (`self.goal_tuples`). For each `(goal_tile, goal_color)`, check `painted_state`. If `goal_tile`
        is painted but `painted_state[goal_tile] != goal_color`, the goal is impossible. Return `float('inf')`.
    3.  **Identify Unsatisfied Goals:** Create a set `unpainted_goals` containing all `(tile, color)` pairs
        from `self.goal_tuples` for which the tile is not currently painted with the correct color.
    4.  **Handle Goal State:** If `unpainted_goals` is empty, the current state satisfies all goals. Return 0.
    5.  **Calculate Cost per Unsatisfied Goal:** Initialize `total_heuristic_value = 0`. For each
        `(goal_tile, goal_color)` in `unpainted_goals`:
        a. Initialize `min_cost_for_this_goal` to `float('inf')`.
        b. **Find Painting Locations:** Get the set of tiles (`possible_paint_locs`) from which `goal_tile`
           can be painted using `self.paint_locations.get(goal_tile, set())`. If this set is empty,
           `goal_tile` cannot be painted according to the domain's static structure. Return `float('inf')`.
        c. **Evaluate Each Robot:** Iterate through all known robots `r`:
            i.   **Get Robot State:** Retrieve the robot's location `r_loc` and color `r_col` from the parsed state.
                 If the robot's state is missing (e.g., robot not mentioned in `robot-at` or `robot-has`), skip this robot.
            ii.  **Calculate Color Cost (`cost_c`):** If `r_col != goal_color`, set `cost_c = 1` (for `change_color`). Else, `cost_c = 0`.
            iii. **Calculate Min Movement Cost (`dist_m`):** Initialize `min_dist_to_paint_loc = float('inf')`.
                 For each `paint_loc` in `possible_paint_locs`, find the shortest path distance `d = self.dist.get((r_loc, paint_loc), float('inf'))`.
                 Update `min_dist_to_paint_loc = min(min_dist_to_paint_loc, d)`.
                 If `min_dist_to_paint_loc` remains `float('inf')` after checking all `paint_loc`s, this robot cannot reach any valid
                 position to paint the target tile. Continue to the next robot. Otherwise, set `dist_m = min_dist_to_paint_loc`.
            iv.  **Calculate Paint Cost (`cost_p`):** Set `cost_p = 1` (for `paint_up` or `paint_down`).
            v.   **Total Cost for Robot `r`:** Calculate `cost_r = cost_c + dist_m + cost_p`.
            vi.  **Update Minimum Goal Cost:** `min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)`. Keep track if at least one robot could potentially achieve the goal (`possible_robots_found = True`).
        d. **Check Goal Reachability:** After evaluating all robots, if `min_cost_for_this_goal` is still `float('inf')`
           (or if `possible_robots_found` is false), it means no robot can achieve this specific goal from the current state.
           Return `float('inf')`.
        e. **Accumulate Total Cost:** Add the calculated `min_cost_for_this_goal` to `total_heuristic_value`.
    6.  **Return Heuristic Value:** Return the final `total_heuristic_value`.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.robots = set()
        self.tiles = set()
        self.colors = set()
        self.goal_tuples = set() # Stores (tile, color) pairs for goals
        # tile -> {adjacent tiles for movement}
        self.adj = collections.defaultdict(set)
        # tile_to_paint -> {locations robot must be AT to paint it}
        self.paint_locations = collections.defaultdict(set)
        # (tile1, tile2) -> shortest distance (number of moves)
        self.dist = {}

        # --- Parse Static Facts ---
        # Robustly parse static facts, identifying tiles, colors, adjacency, and paint locations
        for fact in static_facts:
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip empty or malformed facts
                predicate = parts[0]
                args = parts[1:]

                # Process adjacency predicates (up, down, left, right)
                if predicate in ["up", "down", "left", "right"] and len(args) == 2:
                    t1, t2 = args
                    self.tiles.add(t1)
                    self.tiles.add(t2)
                    # Add edges for movement graph
                    self.adj[t1].add(t2)
                    self.adj[t2].add(t1)

                    # Determine locations for painting based on up/down
                    # If (up y x), robot at x paints y.
                    if predicate == "up":
                        tile_to_paint = t1 # y
                        robot_loc_to_paint_from = t2 # x
                        self.paint_locations[tile_to_paint].add(robot_loc_to_paint_from)
                    # If (down y x), robot at x paints y.
                    elif predicate == "down":
                        tile_to_paint = t1 # y
                        robot_loc_to_paint_from = t2 # x
                        self.paint_locations[tile_to_paint].add(robot_loc_to_paint_from)

                # Process available colors
                elif predicate == "available-color" and len(args) == 1:
                    self.colors.add(args[0])
            except IndexError:
                # Log warning for malformed facts if necessary
                # print(f"Warning: Skipping malformed static fact: {fact}")
                continue


        # --- Parse Goals ---
        # Extract (painted tile color) goals
        for goal in self.goals:
             try:
                if match(goal, "painted", "*", "*"):
                    _, tile, color = get_parts(goal)
                    self.goal_tuples.add((tile, color))
                    # Ensure goal tiles and colors are tracked
                    self.tiles.add(tile)
                    self.colors.add(color)
             except IndexError:
                 # print(f"Warning: Skipping malformed goal fact: {goal}")
                 continue

        # --- Identify Robots and Initial State Info ---
        # Use initial state to find robots and their starting colors/locations
        for fact in task.initial_state:
             try:
                parts = get_parts(fact)
                if not parts: continue
                predicate = parts[0]
                args = parts[1:]
                # Identify robots from robot-at or robot-has predicates
                if predicate == "robot-at" and len(args) == 2:
                    self.robots.add(args[0])
                    self.tiles.add(args[1]) # Track robot starting tiles
                elif predicate == "robot-has" and len(args) == 2:
                    self.robots.add(args[0])
                    self.colors.add(args[1]) # Track robot starting colors
             except IndexError:
                 # print(f"Warning: Skipping malformed init fact: {fact}")
                 continue

        # --- Precompute Distances (BFS from each tile) ---
        # Convert tiles set to list for stable iteration order
        all_tiles_list = list(self.tiles)
        for start_node in all_tiles_list:
            # Handle isolated tiles (not present as key in self.adj)
            if start_node not in self.adj and start_node in self.tiles:
                 self.dist[(start_node, start_node)] = 0
                 continue # No BFS needed for isolated tile

            # Initialize BFS queue and visited dictionary
            queue = collections.deque([(start_node, 0)]) # (node, distance)
            # Tracks visited nodes and their shortest distance from start_node
            visited_dist = {start_node: 0}
            self.dist[(start_node, start_node)] = 0

            # Perform BFS
            while queue:
                current_node, distance = queue.popleft()

                # Explore neighbors
                # Use get() for safe access in case a node has no outgoing edges recorded
                for neighbor in self.adj.get(current_node, set()):
                    if neighbor not in visited_dist:
                        visited_dist[neighbor] = distance + 1
                        # Store the computed shortest distance
                        self.dist[(start_node, neighbor)] = distance + 1
                        queue.append((neighbor, distance + 1))

        # After BFS, self.dist contains shortest path lengths between reachable tile pairs.
        # Unreachable pairs will not have an entry.

    def __call__(self, node):
        """Calculate the heuristic value for the given state node."""
        state = node.state

        # --- Get Current State Info ---
        robot_loc = {} # robot -> tile
        robot_col = {} # robot -> color
        painted_state = {} # tile -> color

        # Parse the current state facts
        for fact in state:
            try:
                parts = get_parts(fact)
                if not parts: continue
                predicate = parts[0]
                args = parts[1:]
                if predicate == "robot-at" and len(args) == 2:
                    robot_loc[args[0]] = args[1]
                elif predicate == "robot-has" and len(args) == 2:
                    robot_col[args[0]] = args[1]
                elif predicate == "painted" and len(args) == 2:
                    painted_state[args[0]] = args[1]
            except IndexError:
                 # Silently ignore malformed facts in the state representation
                 # print(f"Warning: Skipping malformed state fact: {fact}")
                 continue

        # --- Identify Unsatisfied Goals & Check for Unreachable Goals ---
        unpainted_goals = set()
        for goal_tile, goal_color in self.goal_tuples:
            current_color = painted_state.get(goal_tile)
            if current_color is None:
                # Tile is not painted yet, needs to be painted goal_color
                unpainted_goals.add((goal_tile, goal_color))
            elif current_color != goal_color:
                # Tile is painted, but with the WRONG color. Goal is impossible.
                return float('inf')

        # If there are no unpainted goals, the goal state is reached
        if not unpainted_goals:
            return 0

        # --- Calculate Heuristic Value ---
        total_heuristic_value = 0
        # Iterate through each goal that still needs to be achieved
        for goal_tile, goal_color in unpainted_goals:
            min_cost_for_this_goal = float('inf')

            # Find the locations from where this goal_tile can be painted
            possible_paint_locs = self.paint_locations.get(goal_tile, set())

            # If a goal tile has no associated painting locations, it's impossible
            if not possible_paint_locs:
                return float('inf')

            # Track if any robot could potentially achieve this goal
            possible_robots_found = False
            # Evaluate the cost for each robot to achieve this goal
            for r in self.robots:
                # Ensure the robot's current state (location and color) is known
                if r not in robot_loc or r not in robot_col:
                     continue # Skip robot if its state is incomplete

                r_loc = robot_loc[r]
                r_col = robot_col[r]
                cost_r = 0 # Cost for this robot 'r' to achieve the current goal

                # 1. Calculate Color Cost
                if r_col != goal_color:
                    cost_r += 1 # Cost of one 'change_color' action

                # 2. Calculate Minimum Movement Cost
                min_dist_to_paint_loc = float('inf')
                # Find the minimum distance from robot's location to any valid painting location
                for paint_loc in possible_paint_locs:
                    # Use precomputed distances, default to infinity if unreachable
                    distance = self.dist.get((r_loc, paint_loc), float('inf'))
                    min_dist_to_paint_loc = min(min_dist_to_paint_loc, distance)

                # If the robot cannot reach any location to paint this tile, skip it
                if min_dist_to_paint_loc == float('inf'):
                    continue # Try the next robot

                # Add the movement cost (number of move actions)
                cost_r += min_dist_to_paint_loc

                # 3. Calculate Paint Cost
                cost_r += 1 # Cost of one 'paint' action

                # Update the minimum cost found for achieving this goal
                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)
                possible_robots_found = True # Mark that at least one robot path was evaluated

            # After checking all robots:
            # If no robot could possibly reach a painting spot for this goal, state is dead-end
            if not possible_robots_found:
                 return float('inf')
            # Also handle the case where all possible robots had infinite distance (should be covered by above)
            if min_cost_for_this_goal == float('inf'):
                 return float('inf')

            # Add the minimum cost for achieving this goal to the total heuristic value
            total_heuristic_value += min_cost_for_this_goal

        # Return the total estimated cost for all unpainted goals
        return total_heuristic_value
