import sys
from collections import deque

# Ensure the heuristic base class is available.
# If the planner framework places heuristic implementations in a specific structure (e.g., a 'heuristics' directory),
# this import should work. Otherwise, sys.path might need adjustment depending on execution context.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a dummy base class if the import fails (e.g., for standalone testing)
    print("Warning: heuristics.heuristic_base not found. Using dummy Heuristic class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# Helper function to parse PDDL fact strings
def get_parts(fact_string):
    """Removes parentheses and splits PDDL fact string into parts.
    Returns an empty list if the format is invalid."""
    if isinstance(fact_string, str) and len(fact_string) > 2 and fact_string.startswith('(') and fact_string.endswith(')'):
        return fact_string[1:-1].split()
    return []


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain floortile.

    Summary:
    This heuristic estimates the remaining cost (number of actions) to paint all required tiles
    with their specified colors, starting from a given state. It employs a greedy strategy:
    in each step, it finds the "cheapest" possible next painting task (considering robot
    movement to an adjacent tile, potential color change, and the paint action itself)
    and assigns it to the best-suited robot. The total heuristic value is the sum of the
    costs of these greedily chosen assignments until all goals are accounted for.
    This heuristic is designed for use with Greedy Best-First Search and is likely
    non-admissible (it might overestimate the true cost).

    Assumptions:
    - The static predicates (`up`, `down`, `left`, `right`, `available-color`) correctly define the
      environment's connectivity and available resources.
    - All tiles specified in goal `(painted tile color)` predicates are actually paintable
      (i.e., there exists at least one adjacent tile from which a robot can perform the paint action,
      according to the static directional predicates).
    - Robot movement cost is estimated using shortest path distances on the static tile grid,
      ignoring the dynamic `(clear tile)` predicate for efficiency. This means the heuristic assumes
      robots can eventually reach required locations, even if temporarily blocked.
    - The cost associated with ensuring a target tile is `clear` before painting is ignored
      for simplicity (cost assumed to be 0 or handled by movement).
    - Robots always hold exactly one color at any time.
    - The `change_color` action always costs 1 unit and is assumed possible whenever a robot
      needs a different color that is listed as `available-color`.

    Heuristic Initialization (`__init__`):
    - Extracts the goal conditions (`(painted tile color)`) from the task definition.
    - Identifies all unique robots, tiles, and colors present in the problem instance by parsing
      the initial state, static facts, and goals.
    - Constructs an undirected adjacency graph representing tile connectivity based on the static
      `up`, `down`, `left`, `right` predicates. This graph is used for pathfinding.
    - Precomputes all-pairs shortest path distances between tiles using Breadth-First Search (BFS)
      on the static adjacency graph. Stores these distances for efficient lookup. Unreachable
      tile pairs are assigned an infinite distance.
    - Determines, for each tile, which adjacent tiles can serve as a 'source' location for a robot
      to paint it, based on the directional predicates (e.g., if `(up tileY tileX)` exists,
      a robot at `tileX` can potentially execute `paint_up` on `tileY`).

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
    1.  **Identify Unsatisfied Goals**: Compare the set of `(painted tile color)` facts in the
        current `state` with the required `goal_tuples` derived during initialization. Create a set
        of `unsatisfied_goals`. If this set is empty, the state is a goal state, return 0.
    2.  **Get Current Robot State**: Parse the `state` to find the current location (`robot_loc`)
        and held color (`robot_col`) for each robot present in the state.
    3.  **Initialize Simulation**: Create mutable copies of `robot_loc` and `robot_col`. These copies
        will be updated within the heuristic calculation to simulate the effect of assigned tasks
        on robot positions and colors for subsequent cost estimations. Initialize `heuristic_value = 0`.
        Identify robots that are actually available and defined in the current state. If no robots
        are available, return infinity if goals remain, 0 otherwise.
    4.  **Greedy Assignment Loop**: Repeat as long as `unsatisfied_goals` is not empty:
        a.  **Find Cheapest Task**: Iterate through every combination of an available robot (`r`)
            and a remaining unsatisfied goal (`g = (goal_tile, goal_color)`).
            i.  **Calculate Cost for (r, g)**:
                - Determine the set of possible source tiles (`adj_tile`) from which `goal_tile` can be painted.
                - Find the `adj_tile` in this set that is closest to `r`'s current simulated location
                  (`current_robot_loc[r]`), using the precomputed `distances`. Let this minimum
                  distance be `move_cost`. If no such `adj_tile` is reachable (distance is infinity),
                  this assignment is impossible (`move_cost = inf`).
                - Calculate the color change cost: `color_cost = 1` if `r`'s simulated color
                  (`current_robot_col[r]`) is different from `goal_color`, otherwise `color_cost = 0`.
                - The painting action itself costs 1: `paint_cost = 1`.
                - The total estimated cost for this specific assignment `(r, g)` is
                  `move_cost + color_cost + paint_cost`. If `move_cost` was infinity, the total cost
                  is infinity.
            ii. **Track Minimum**: Keep track of the assignment `(robot, goal, adj_tile, cost)` that
                has the lowest finite total cost found so far across all robots and goals in this iteration.
        b.  **Select Best Assignment**: After checking all combinations, identify the best assignment
            `(chosen_robot, chosen_goal, target_adj_tile, min_total_cost)` found.
        c.  **Handle Unsolvable State**: If no assignment with a finite cost could be found in step 4a
            (i.e., `min_total_cost` remains infinity), it implies the remaining goals cannot be achieved
            from the current simulated state according to the heuristic's estimation. Return `float('inf')`.
        d.  **Update Heuristic Value**: Add the cost of the best assignment (`min_total_cost`) to the
            running `heuristic_value`.
        e.  **Update Simulated State**: Modify the simulation state to reflect the completion of the
            chosen task: update `current_robot_loc[chosen_robot]` to `target_adj_tile` (the location
            from which the paint action occurred) and update `current_robot_col[chosen_robot]` to
            `chosen_goal[1]` (the color used for painting).
        f.  **Remove Assigned Goal**: Remove `chosen_goal` from the set of `unsatisfied_goals`.
    5.  **Return Value**: Once the loop terminates (because `unsatisfied_goals` is empty), return the
        accumulated `heuristic_value`.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Extract objects (robots, tiles, colors) ---
        self.robots = set()
        self.tiles = set()
        self.colors = set()

        all_facts_to_parse = static_facts | initial_state | self.goals
        for fact in all_facts_to_parse:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            try:
                if pred in ["up", "down", "left", "right"] and len(parts) == 3:
                    self.tiles.add(parts[1])
                    self.tiles.add(parts[2])
                elif pred == "available-color" and len(parts) == 2:
                    self.colors.add(parts[1])
                elif pred == "robot-at" and len(parts) == 3:
                    self.robots.add(parts[1])
                    self.tiles.add(parts[2])
                elif pred == "painted" and len(parts) == 3:
                    self.tiles.add(parts[1])
                    self.colors.add(parts[2])
                elif pred == "robot-has" and len(parts) == 3:
                    self.robots.add(parts[1])
                    self.colors.add(parts[2])
                elif pred == "clear" and len(parts) == 2:
                     self.tiles.add(parts[1])
            except IndexError:
                 print(f"Warning: Malformed fact encountered during object parsing: {fact}", file=sys.stderr)

        # Convert to lists for consistent ordering if needed, though sets are fine for membership checks
        self.robots = list(self.robots)
        self.tiles = list(self.tiles)
        self.colors = list(self.colors)

        # --- Build adjacency graph and paint sources ---
        # Adjacency for pathfinding (undirected)
        self.adj = {tile: [] for tile in self.tiles}
        # Paint sources: paint_sources[dest_tile] = list of src_tiles from which dest can be painted
        self.paint_sources = {tile: [] for tile in self.tiles}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts or len(parts) != 3: continue
            pred = parts[0]
            if pred in ["up", "down", "left", "right"]:
                dest_tile, src_tile = parts[1], parts[2]
                # Ensure tiles exist in our parsed set before adding graph edges
                if dest_tile in self.tiles and src_tile in self.tiles:
                    # Add undirected edge for pathfinding distance calculation
                    self.adj[src_tile].append(dest_tile)
                    self.adj[dest_tile].append(src_tile)
                    # Record that a robot at src_tile can paint dest_tile
                    self.paint_sources[dest_tile].append(src_tile)

        # --- Precompute all-pairs shortest paths ---
        self.distances = self._compute_all_pairs_shortest_paths()

        # --- Parse goal conditions into (tile, color) tuples ---
        self.goal_tuples = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts or len(parts) != 3: continue
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                # Only add goals involving known objects
                if tile in self.tiles and color in self.colors:
                    self.goal_tuples.add((tile, color))
                else:
                    print(f"Warning: Goal fact {goal} involves unknown tile '{tile}' or color '{color}'. Ignoring.", file=sys.stderr)


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of tiles using BFS."""
        distances = {tile: {other_tile: float('inf') for other_tile in self.tiles} for tile in self.tiles}
        if not self.tiles:
            return distances # Return empty distances if no tiles

        for start_node in self.tiles:
            # Check if start_node is valid (should be if self.tiles is correct)
            if start_node not in distances: continue
            distances[start_node][start_node] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node} # Use visited set for efficiency

            while queue:
                current_node, dist = queue.popleft()
                # Iterate through neighbors from the pre-built adjacency list
                for neighbor in self.adj.get(current_node, []):
                    if neighbor not in visited:
                         visited.add(neighbor)
                         # Check if neighbor is a valid key before assignment (should be)
                         if neighbor in distances[start_node]:
                            distances[start_node][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
        return distances

    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state

        # --- Get current state info ---
        robot_loc = {}
        robot_col = {}
        current_painted_tuples = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            try:
                # Check predicate and arity before accessing parts
                if pred == "robot-at" and len(parts) == 3 and parts[1] in self.robots:
                    robot_loc[parts[1]] = parts[2]
                elif pred == "robot-has" and len(parts) == 3 and parts[1] in self.robots:
                    robot_col[parts[1]] = parts[2]
                elif pred == "painted" and len(parts) == 3:
                     # Store all painted facts found in state for goal checking
                     current_painted_tuples.add((parts[1], parts[2]))
            except IndexError:
                 # This shouldn't happen if get_parts checks length, but as safety
                 print(f"Warning: Malformed fact encountered during state parsing: {fact}", file=sys.stderr)

        # --- Identify unsatisfied goals ---
        # Filter goal tuples to only those involving known objects (already done in init)
        unsatisfied_goals = self.goal_tuples - current_painted_tuples

        if not unsatisfied_goals:
            return 0 # Goal state reached

        # --- Initialize simulation ---
        heuristic_value = 0
        # Filter robots to those actually present and defined in the current state
        available_robots_in_state = [r for r in self.robots if r in robot_loc and r in robot_col]

        if not available_robots_in_state:
             # No robots available to achieve remaining goals
             return float('inf') if unsatisfied_goals else 0

        # Create copies for simulation state
        current_robot_loc = {r: robot_loc[r] for r in available_robots_in_state}
        current_robot_col = {r: robot_col[r] for r in available_robots_in_state}

        # --- Greedy assignment loop ---
        remaining_goals = set(unsatisfied_goals) # Operate on a copy

        while remaining_goals:
            min_total_cost = float('inf')
            best_assignment = None # Stores (robot, goal_tuple, best_adj_tile, cost)

            # Iterate through available robots and remaining goals
            for robot in available_robots_in_state:
                robot_tile = current_robot_loc[robot]
                robot_color = current_robot_col[robot]

                for goal in remaining_goals:
                    goal_tile, goal_color = goal
                    # Get potential locations from where the robot can paint the goal tile
                    possible_paint_sources = self.paint_sources.get(goal_tile, [])

                    if not possible_paint_sources:
                        # This goal tile cannot be painted based on static facts. Skip.
                        continue

                    min_dist_to_adj = float('inf')
                    best_adj_tile_for_goal = None

                    # Find the closest reachable paint source tile
                    for adj_tile in possible_paint_sources:
                        # Safely access precomputed distance
                        dist = self.distances.get(robot_tile, {}).get(adj_tile, float('inf'))

                        if dist < min_dist_to_adj:
                            min_dist_to_adj = dist
                            best_adj_tile_for_goal = adj_tile

                    # Calculate cost only if a reachable adjacent tile was found
                    if best_adj_tile_for_goal is not None and min_dist_to_adj != float('inf'):
                        move_cost = min_dist_to_adj
                        color_cost = 1 if robot_color != goal_color else 0
                        paint_cost = 1
                        total_cost_for_assignment = move_cost + color_cost + paint_cost
                    else:
                        # Cannot reach any adjacent tile to paint this goal
                        total_cost_for_assignment = float('inf')

                    # Check if this assignment is the best one found so far in this iteration
                    if total_cost_for_assignment < min_total_cost:
                        min_total_cost = total_cost_for_assignment
                        best_assignment = (robot, goal, best_adj_tile_for_goal, min_total_cost)

            # --- End of finding the best assignment for this iteration ---

            if best_assignment is None or min_total_cost == float('inf'):
                # If no assignment with finite cost was found, implies state is unsolvable
                # according to this heuristic's estimation (e.g., remaining goals unpaintable/unreachable)
                return float('inf')

            # --- Process the best assignment found ---
            chosen_robot, chosen_goal, target_adj_tile, cost = best_assignment

            heuristic_value += cost
            remaining_goals.remove(chosen_goal)

            # --- Update simulated state for the chosen robot ---
            # Robot is now simulated to be at the tile from which it painted, holding the new color
            current_robot_loc[chosen_robot] = target_adj_tile
            current_robot_col[chosen_robot] = chosen_goal[1] # chosen_goal is (tile, color)

        # --- End of greedy assignment loop ---
        return heuristic_value
