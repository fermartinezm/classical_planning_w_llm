import heapq
from collections import deque
from fnmatch import fnmatch
import itertools

# Try to import the base class, but define a placeholder if it fails
# (useful for standalone testing, though the final environment will provide it)
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a placeholder base class if the import fails
    class Heuristic:
        def __init__(self, task):
            """Initialize the heuristic with the planning task."""
            self.task = task
        def __call__(self, node):
            """
            Evaluate the heuristic function for a given node.
            Must be implemented by subclasses.
            """
            raise NotImplementedError("Heuristic evaluation not implemented.")


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL floortile domain.

    # Summary
    This heuristic estimates the remaining cost to reach the goal state by summing
    the estimated costs for achieving each unsatisfied 'painted' goal predicate.
    The cost for each goal includes the paint action itself, the minimum number
    of moves required for any robot to reach an adjacent painting position, and
    the cost of a color change if the robot estimated to be best for the task
    (minimum move + color change cost) does not currently hold the required color.

    # Assumptions
    - The primary goal is to satisfy all `(painted ?tile ?color)` predicates specified in the goal.
    - The grid structure (tiles and their connections: up, down, left, right) is static.
    - All colors specified in goal predicates are available (`available-color` static facts).
    - Robots can move between connected tiles. The heuristic implicitly assumes target tiles for movement/painting will become clear when needed.
    - The heuristic ignores the `clear` precondition for tiles to be painted or moved onto. This simplifies the calculation and makes the heuristic non-admissible, which is acceptable for Greedy Best-First Search, aiming for informativeness over admissibility.
    - The heuristic ignores potential negative interactions, such as multiple robots interfering with each other or actions invalidating preconditions for other required actions.
    - Each unsatisfied goal's cost is estimated independently based on the minimum cost achievable by *any* robot, and these costs are summed. This might over or under-count resource usage (e.g., a single robot trip might satisfy multiple goals, or multiple color changes might be needed even if counted once per goal).

    # Heuristic Initialization
    - Parses the task's static facts (`task.static`) to build the tile grid connectivity. It creates two adjacency maps: `adj` for general movement (up, down, left, right) and `paint_adj` storing which tiles a robot must be on to paint a target tile (based on `up`/`down` facts).
    - Identifies all unique tiles, robots (extracted from the initial state), and available colors (from static facts).
    - Precomputes all-pairs shortest path distances between all tiles using Breadth-First Search (BFS) on the `adj` graph. The distance represents the minimum number of `move_*` actions. This is stored in `self.distances`.
    - Stores the goal conditions `(painted ?tile ?color)` in a dictionary `self.goal_dict` mapping `tile` to its required `color` for efficient lookup during heuristic evaluation.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Extract Current State:** Get the current facts from the input `node.state`, identifying robot locations (`robot-at`), robot colors (`robot-has`), and currently painted tiles (`painted`).
    2.  **Identify Unsatisfied Goals:** Compare the `painted_tiles` in the current state with the `self.goal_dict`. Create a list `unsatisfied_goals` of `(tile, color)` pairs that are required by the goal but not currently satisfied.
    3.  **Handle Goal State:** If `unsatisfied_goals` is empty, the current state is a goal state, return 0.
    4.  **Check Robot Availability:** If there are no robots defined (`self.robots` is empty), the goal is unreachable, return infinity.
    5.  **Iterate Through Unsatisfied Goals:** For each `(target_tile, target_color)` in `unsatisfied_goals`:
        a.  **Find Painting Positions:** Look up `target_tile` in `self.paint_adj` to get the set of adjacent tiles `possible_paint_locations` a robot must occupy to paint `target_tile`. If `target_tile` is not in `self.paint_adj` or the set is empty, this specific goal is statically impossible, return infinity.
        b.  **Estimate Minimum Robot Cost for this Goal:** Initialize `min_robot_cost_for_goal` to infinity. Iterate through each `robot` in `self.robots`:
            i.  Get the robot's current location `current_robot_loc` and color `current_robot_col`. Handle cases where robot info might be missing (though unlikely in valid states).
            ii. Calculate the minimum move cost `robot_min_move` for this robot to reach any tile in `possible_paint_locations` using the precomputed `self.distances`. If the robot cannot reach any of these locations (distance is infinity), skip this robot for this goal.
            iii.Calculate the color change cost `color_cost`. If `current_robot_col` is different from `target_color`, the cost is 1 (assuming `target_color` is available, checked during initialization implicitly). Otherwise, the cost is 0.
            iv. Calculate the combined cost for this robot: `combined_cost = robot_min_move + color_cost`.
            v.  Update `min_robot_cost_for_goal = min(min_robot_cost_for_goal, combined_cost)`.
        c.  **Check Goal Reachability:** If after checking all robots, `min_robot_cost_for_goal` is still infinity, it means no robot can achieve this goal. Return infinity.
        d.  **Add Goal Cost to Total:** Add `1 + min_robot_cost_for_goal` to the running `heuristic_value`. The `1` represents the cost of the `paint_up`/`paint_down` action itself.
    6.  **Return Total Value:** After iterating through all unsatisfied goals, return the total calculated `heuristic_value`.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class if needed
        self.goals = task.goals
        static_facts = task.static

        # --- Preprocessing: Extract objects and relationships ---
        self.tiles = set()
        self.robots = set()
        self.colors = set()
        # Adjacency for movement (all directions)
        self.adj = {}
        # Adjacency for painting: paint_adj[target_tile] = {robot_pos1, robot_pos2}
        self.paint_adj = {}

        # Parse static facts
        for fact in static_facts:
            parts = self._get_parts(fact)
            pred = parts[0]
            if self._match(parts, "up", "*", "*"): # (up target_tile robot_pos)
                target_tile, robot_pos = parts[1], parts[2]
                self.tiles.add(target_tile)
                self.tiles.add(robot_pos)
                self.adj.setdefault(target_tile, set()).add(robot_pos)
                self.adj.setdefault(robot_pos, set()).add(target_tile)
                self.paint_adj.setdefault(target_tile, set()).add(robot_pos)
            elif self._match(parts, "down", "*", "*"): # (down target_tile robot_pos)
                target_tile, robot_pos = parts[1], parts[2]
                self.tiles.add(target_tile)
                self.tiles.add(robot_pos)
                self.adj.setdefault(target_tile, set()).add(robot_pos)
                self.adj.setdefault(robot_pos, set()).add(target_tile)
                self.paint_adj.setdefault(target_tile, set()).add(robot_pos)
            elif self._match(parts, "left", "*", "*"): # (left tile_a tile_b) -> can move left from b to a
                tile_a, tile_b = parts[1], parts[2]
                self.tiles.add(tile_a)
                self.tiles.add(tile_b)
                self.adj.setdefault(tile_a, set()).add(tile_b)
                self.adj.setdefault(tile_b, set()).add(tile_a)
            elif self._match(parts, "right", "*", "*"): # (right tile_a tile_b) -> can move right from b to a
                tile_a, tile_b = parts[1], parts[2]
                self.tiles.add(tile_a)
                self.tiles.add(tile_b)
                self.adj.setdefault(tile_a, set()).add(tile_b)
                self.adj.setdefault(tile_b, set()).add(tile_a)
            elif self._match(parts, "available-color", "*"):
                self.colors.add(parts[1])

        # Extract robots from the initial state
        # Robots might not be mentioned in static facts but exist in the initial state.
        for fact in task.initial_state:
             parts = self._get_parts(fact)
             if self._match(parts, "robot-at", "*", "*"):
                 self.robots.add(parts[1])
             elif self._match(parts, "robot-has", "*", "*"): # Also capture robots from here
                 self.robots.add(parts[1])

        # Compute all-pairs shortest paths using BFS
        self.distances = self._compute_distances()

        # Store goal predicates in a more usable format: {tile: color}
        self.goal_dict = {}
        for goal in self.goals:
            parts = self._get_parts(goal)
            if self._match(parts, "painted", "*", "*"):
                _, tile, color = parts
                # Ensure goal tile and color are known entities
                if tile not in self.tiles:
                    print(f"Warning: Goal tile '{tile}' not found in static facts.")
                if color not in self.colors:
                    print(f"Warning: Goal color '{color}' not found in available colors.")
                self.goal_dict[tile] = color

    def _get_parts(self, fact_string):
        """Extracts predicate and arguments from a PDDL fact string."""
        # Removes parentheses and splits by space
        return fact_string[1:-1].split()

    def _match(self, fact_parts, *pattern):
        """Checks if fact parts (already split) match a pattern tuple."""
        if len(fact_parts) != len(pattern):
            return False
        # fnmatch allows wildcard matching, '*' matches anything
        return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

    def _compute_distances(self):
        """Computes shortest path distances between all pairs of tiles using BFS."""
        distances = {t: {t2: float('inf') for t2 in self.tiles} for t in self.tiles}
        tile_list = list(self.tiles) # Use a list for consistent iteration order

        for start_node in tile_list:
            # Check if start_node has any connections defined
            if start_node not in self.adj and start_node in self.tiles:
                 distances[start_node][start_node] = 0 # Distance to self is 0
                 continue # No outgoing edges from this tile

            distances[start_node][start_node] = 0
            queue = deque([start_node])
            # Keep track of visited nodes and their distances from start_node
            visited_dist = {start_node: 0}

            while queue:
                current_node = queue.popleft()
                current_dist = visited_dist[current_node]

                # Explore neighbors based on the movement adjacency graph
                for neighbor in self.adj.get(current_node, set()):
                    if neighbor not in visited_dist:
                         # Ensure neighbor is a valid tile we initialized
                         if neighbor in self.tiles:
                             visited_dist[neighbor] = current_dist + 1
                             distances[start_node][neighbor] = current_dist + 1
                             queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state node.
        """
        state = node.state

        # --- Extract current state info ---
        robot_locs = {}     # {robot_name: tile_name}
        robot_colors = {}   # {robot_name: color_name}
        painted_tiles = {}  # {tile_name: color_name}

        for fact in state:
            parts = self._get_parts(fact)
            if self._match(parts, "robot-at", "*", "*"):
                robot_locs[parts[1]] = parts[2]
            elif self._match(parts, "robot-has", "*", "*"):
                robot_colors[parts[1]] = parts[2]
            elif self._match(parts, "painted", "*", "*"):
                painted_tiles[parts[1]] = parts[2]
            # 'clear' predicate is ignored by this heuristic

        # --- Calculate heuristic ---
        heuristic_value = 0
        unsatisfied_goals = []

        # Identify unsatisfied goals
        for tile, goal_color in self.goal_dict.items():
            if painted_tiles.get(tile) != goal_color:
                unsatisfied_goals.append((tile, goal_color))

        # If no unsatisfied goals, we are at the goal state
        if not unsatisfied_goals:
            return 0

        # If there are goals but no robots, the goal is unreachable
        if not self.robots:
             return float('inf')

        # Calculate cost for each unsatisfied goal
        for target_tile, target_color in unsatisfied_goals:

            # Find adjacent tiles from where painting is possible for this target_tile
            possible_paint_locations = self.paint_adj.get(target_tile)

            # If a goal tile has no associated painting locations from static facts, it's impossible
            if not possible_paint_locations:
                 # print(f"Warning: Goal tile {target_tile} has no valid painting positions based on static facts.")
                 return float('inf')

            min_robot_cost_for_goal = float('inf')

            # Find the minimum cost (move + color_change) for any robot to achieve this goal
            for robot in self.robots:
                # Ensure robot's current state is known
                if robot not in robot_locs or robot not in robot_colors:
                     # This might happen if the problem definition is unusual, skip this robot
                     # print(f"Warning: State information missing for robot {robot}")
                     continue

                current_robot_loc = robot_locs[robot]
                current_robot_col = robot_colors[robot]

                # Ensure the robot's location is valid in our distance map
                if current_robot_loc not in self.distances:
                    # Robot is on a tile disconnected or not in static facts? Problematic.
                    # print(f"Warning: Robot {robot} location {current_robot_loc} not found in distance map.")
                    continue # This robot cannot reach any goal position

                # Find minimum move cost from this robot to a valid painting position
                robot_min_move = float('inf')
                for adj_tile in possible_paint_locations:
                    # Ensure the adjacent tile is a valid key in the distance matrix
                    if adj_tile in self.distances[current_robot_loc]:
                         dist = self.distances[current_robot_loc][adj_tile]
                         robot_min_move = min(robot_min_move, dist)
                    # else: distance remains inf, path doesn't exist or adj_tile is invalid

                # If robot cannot reach any painting position for this goal, skip it
                if robot_min_move == float('inf'):
                    continue

                # Calculate color change cost for this robot
                color_cost = 0
                if current_robot_col != target_color:
                    # We assume target_color is valid/available based on initialization checks
                    color_cost = 1
                    # If target_color wasn't in self.colors, we'd have warned at init or could return inf here

                # Combined cost (move + color) for this robot for this goal
                combined_cost = robot_min_move + color_cost
                min_robot_cost_for_goal = min(min_robot_cost_for_goal, combined_cost)

            # After checking all robots for this goal:
            if min_robot_cost_for_goal == float('inf'):
                 # No robot can achieve this goal (cannot reach or needed color unavailable implicitly handled).
                 # print(f"Warning: No robot can achieve goal ({target_tile}, {target_color})")
                 return float('inf')

            # Add costs for this goal: 1 (for paint action) + min combined (move + color) cost
            heuristic_value += (1 + min_robot_cost_for_goal)

        # Ensure heuristic is non-negative
        return max(0, heuristic_value)

