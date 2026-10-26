import itertools
from collections import deque
# Assuming the heuristic base class is accessible as heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact_str):
    """
    Safely parses a PDDL fact string "(predicate arg1 arg2 ...)" into a list
    [predicate, arg1, arg2, ...]. Handles potential errors gracefully.

    Args:
        fact_str: The PDDL fact string.

    Returns:
        A list containing the predicate name and arguments, or an empty list
        if the input format is invalid.
    """
    if isinstance(fact_str, str) and len(fact_str) > 1 and fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    return [] # Return empty list for invalid format


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    for the floortile domain. The estimation is based on the sum of minimum costs
    required to achieve each unsatisfied 'painted' goal predicate. For each
    unsatisfied goal `(painted target_tile target_color)`, it calculates the
    minimum cost considering all available robots. This cost includes the travel
    distance for a robot to reach an adjacent tile (from which painting `target_tile`
    is possible) and the cost of changing to `target_color` if necessary. The final
    heuristic value is the sum of these minimum costs over all unmet goals, plus
    one paint action cost for each unmet goal.

    # Assumptions
    - The cost of each action (move_*, paint_*, change_color) is 1.
    - The heuristic ignores the 'clear' status of tiles when calculating movement
      distances between tiles. It uses the static grid connectivity defined by
      'up', 'down', 'left', 'right' predicates. This simplification is suitable
      for a greedy best-first search heuristic where admissibility is not required.
    - The heuristic assumes any robot can eventually reach any tile if the grid
      is connected according to the static predicates.
    - The cost for achieving each goal is calculated independently based on the
      minimum cost achievable by *any* single robot, and these costs are summed up.
      This might overestimate the total plan cost if one robot can efficiently
      achieve multiple goals sequentially, but serves as an informative estimate
      to guide the greedy search.

    # Heuristic Initialization
    - Extracts all tile objects by parsing static connectivity predicates
      ('up', 'down', 'left', 'right').
    - Builds an adjacency map (`self.adj`) representing the static grid where
      robots can move (bi-directional connections).
    - Parses 'up' and 'down' predicates to determine, for each paintable tile `y`,
      the set of adjacent tiles `x` from which `y` can be painted. Stores this
      mapping in `self.paint_adj`: `target_tile -> set(adjacent_painter_tile)`.
    - Precomputes all-pairs shortest path distances between all known tiles using
      Breadth-First Search (BFS) on the static grid defined by `self.adj`.
      Distances are stored in `self.all_pairs_distances`.
    - Stores the set of goal predicates `(painted tile color)` as tuples `(tile, color)`
      in `self.goal_predicates`.
    - Identifies all robot objects by looking for `robot-at` or `robot-has`
      predicates in the initial state provided by the `task` object.
    - Collects all known colors from `available-color` static facts, the initial
      state (`robot-has`, `painted`), and the goal state (`painted`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse Current State: Extract the current location (`robot_locations`) and
       held color (`robot_colors`) for each robot, and the set of currently
       painted tiles (`painted_tiles`) as `(tile, color)` tuples from the input
       `node.state`.
    2. Identify Unsatisfied Goals: Calculate the set difference between the required
       goal predicates (`self.goal_predicates`) and the currently `painted_tiles`.
       This gives the set of `unmet_goals`. If `unmet_goals` is empty, the goal
       state is reached, and the heuristic value is 0.
    3. Calculate Cost Per Unmet Goal: Initialize `total_heuristic_value = 0`. Iterate
       through each `(target_tile, target_color)` in `unmet_goals`:
       a. Base Cost: Add 1 to `total_heuristic_value` for the required `paint_up`
          or `paint_down` action.
       b. Find Painting Positions: Retrieve the set of tiles (`adjacent_painter_tiles`)
          from `self.paint_adj.get(target_tile, set())`. These are the locations a
          robot must be *at* to paint `target_tile`.
       c. Check Paintability: If `adjacent_painter_tiles` is empty, it implies
          `target_tile` cannot be painted according to the domain's static connections.
          The goal is fundamentally unreachable; return `math.inf`.
       d. Calculate Minimum Robot Cost: Initialize `min_cost_for_this_goal` to `math.inf`.
          If there are no robots (`self.robots` is empty), the goal is unreachable; return `math.inf`.
          Iterate through each `robot` in `self.robots`:
          i.   Get Robot State: Retrieve the robot's current position `robot_pos` and
               color `robot_col` from the parsed state. If the state information is
               missing for a robot, skip it (it cannot contribute).
          ii.  Calculate Minimum Distance: Find the minimum shortest path distance
               from `robot_pos` to *any* tile in `adjacent_painter_tiles` using the
               precomputed `self._get_distance` method. Let this be `min_dist_to_adj`.
               If the robot cannot reach any required painting position, `min_dist_to_adj`
               will be `math.inf`.
          iii. Handle Unreachable Painting Position: If `min_dist_to_adj` is `math.inf`,
               this robot cannot fulfill this task. Set its contribution cost
               (`current_robot_cost`) to `math.inf`.
          iv.  Calculate Color Change Cost: If `min_dist_to_adj` is finite, check if
               `robot_col` matches `target_color`. The `color_change_cost` is 1 if
               they differ, and 0 otherwise.
          v.   Total Robot Cost: `current_robot_cost = min_dist_to_adj + color_change_cost`.
       e. Update Minimum Goal Cost: Update `min_cost_for_this_goal` by taking the
          minimum of its current value and `current_robot_cost`. This finds the
          cheapest way *any* robot can achieve this specific goal (movement + color change).
       f. Check Goal Reachability: After checking all robots, if `min_cost_for_this_goal`
          is still `math.inf`, it means no robot can reach a suitable position and/or
          acquire the correct color to paint this tile from the current state. The goal
          is unreachable; return `math.inf`.
       g. Add Robot Cost to Total: Add the finite `min_cost_for_this_goal` to the
          `total_heuristic_value`.
    4. Return Total Value: Return the final `total_heuristic_value`. If `math.inf` was
       returned at any point, that value indicates unreachability. Ensure the final
       finite value is returned as an integer, as all action costs are integers.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        self.robots = set()
        self.tiles = set()
        # Initialize colors from static 'available-color' facts
        self.colors = {p[1] for f in static_facts if (p := get_parts(f)) and p[0] == "available-color"}

        self.adj = {} # Adjacency list for movement: tile -> set(neighbor_tile)
        self.paint_adj = {} # Tiles from which a target tile can be painted: target_tile -> set(adjacent_painter_tile)

        # Parse static facts for connectivity and tiles
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            if pred in ["up", "down", "left", "right"]:
                t1, t2 = args
                self.tiles.add(t1)
                self.tiles.add(t2)
                # Add bi-directional adjacency for movement using sets for uniqueness
                self.adj.setdefault(t1, set()).add(t2)
                self.adj.setdefault(t2, set()).add(t1)
                # Store adjacency for painting: if (up y x) or (down y x), robot is at x to paint y
                if pred == "up" or pred == "down":
                     # y = t1 (target), x = t2 (robot location)
                     self.paint_adj.setdefault(t1, set()).add(t2)

        # Parse initial state for robots and potentially missing tiles/colors
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             args = parts[1:]
             if pred == "robot-at":
                 self.robots.add(args[0])
                 self.tiles.add(args[1]) # Ensure robot start tiles are included
             elif pred == "robot-has":
                 self.robots.add(args[0])
                 self.colors.add(args[1]) # Ensure robot start colors are known
             elif pred == "painted":
                 self.tiles.add(args[0])
                 self.colors.add(args[1]) # Ensure initially painted tiles/colors are known
             elif pred == "clear":
                 self.tiles.add(args[0]) # Ensure clear tiles are known

        # Process goals to ensure all relevant objects are known and store goal predicates
        self.goal_predicates = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            if pred == "painted":
                tile, color = args
                self.tiles.add(tile) # Ensure goal tiles are known
                self.colors.add(color) # Ensure goal colors are known
                self.goal_predicates.add((tile, color)) # Store as tuple (tile, color)

        # Precompute all-pairs shortest paths using BFS
        self.all_pairs_distances = {}
        # Create adjacency map suitable for BFS (list or tuple of neighbors)
        adj_for_bfs = {tile: list(neighbors) for tile, neighbors in self.adj.items()}
        # Ensure all tiles have an entry in adj_for_bfs, even if disconnected
        for tile in self.tiles:
            if tile not in adj_for_bfs:
                 adj_for_bfs[tile] = []

        for start_tile in self.tiles:
            self.all_pairs_distances[start_tile] = self._bfs(start_tile, adj_for_bfs)

    def _bfs(self, start_node, adj_map):
        """Performs BFS from start_node on the static tile grid."""
        distances = {tile: float('inf') for tile in self.tiles}
        if start_node not in self.tiles:
             # Should not happen if tiles are collected correctly
             return {}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            for neighbor in adj_map.get(current_node, []):
                # Check if neighbor is known and not visited with shorter path
                if neighbor in distances and distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def _get_distance(self, tile1, tile2):
        """Returns the precomputed shortest distance between two tiles."""
        if tile1 not in self.all_pairs_distances:
            # Start tile was not found during precomputation (should not happen)
            return float('inf')
        # Use .get for safety, returning inf if tile2 is not reachable from tile1
        return self.all_pairs_distances[tile1].get(tile2, float('inf'))


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.

        Args:
            node: A search node containing the current state (`node.state`).

        Returns:
            An integer estimate of the remaining actions, or math.inf if the
            goal is determined to be unreachable from this state.
        """
        state = node.state

        # --- 1. Parse current state ---
        robot_locations = {}
        robot_colors = {}
        painted_tiles = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            if pred == "robot-at":
                robot_locations[args[0]] = args[1]
            elif pred == "robot-has":
                robot_colors[args[0]] = args[1]
            elif pred == "painted":
                painted_tiles.add((args[0], args[1])) # Store as tuple

        # --- 2. Identify unsatisfied goals ---
        unmet_goals = self.goal_predicates - painted_tiles

        # If all goals are met, heuristic is 0
        if not unmet_goals:
            return 0

        # --- 3. Calculate cost for each unmet goal ---
        total_heuristic_value = 0
        for target_tile, target_color in unmet_goals:
            # Cost for the paint action itself
            paint_action_cost = 1

            # Find tiles from which the target_tile can be painted
            adjacent_painter_tiles = self.paint_adj.get(target_tile, set())

            if not adjacent_painter_tiles:
                # This goal tile cannot be painted according to domain physics
                return math.inf # Goal is fundamentally unreachable

            min_cost_for_this_goal = float('inf')

            # If there are no robots, goals are unreachable
            if not self.robots:
                 return math.inf

            # Calculate minimum cost across all robots
            for robot in self.robots:
                # Ensure robot current state is known
                if robot not in robot_locations or robot not in robot_colors:
                     # Assume robot cannot contribute if its state is unknown in this node
                     continue

                robot_pos = robot_locations[robot]
                robot_col = robot_colors[robot]

                # Find min distance from robot_pos to any required adjacent_painter_tile
                min_dist_to_adj = float('inf')
                for adj_tile in adjacent_painter_tiles:
                    dist = self._get_distance(robot_pos, adj_tile)
                    min_dist_to_adj = min(min_dist_to_adj, dist)

                # If min_dist_to_adj is still inf, this robot cannot reach any required painting position
                if min_dist_to_adj == float('inf'):
                    current_robot_cost = float('inf')
                else:
                    # Calculate color change cost
                    color_change_cost = 0 if robot_col == target_color else 1
                    # Total cost for this robot to achieve this goal
                    current_robot_cost = min_dist_to_adj + color_change_cost

                # Update minimum cost found so far for this goal across all robots
                min_cost_for_this_goal = min(min_cost_for_this_goal, current_robot_cost)

            # Check if the goal was found to be reachable by at least one robot
            if min_cost_for_this_goal == float('inf'):
                 # No robot can reach a position to paint this tile with the correct color.
                 return math.inf # State leads to a dead end regarding this goal

            # Add the cost for this goal (paint action + min robot cost)
            total_heuristic_value += (paint_action_cost + min_cost_for_this_goal)

        # --- 4. Return total estimated cost ---
        # The value should be an integer since all costs are integers
        # Check for overflow if total_heuristic_value could exceed standard int limits, though unlikely here.
        return int(total_heuristic_value)
