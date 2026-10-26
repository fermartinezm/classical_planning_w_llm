import collections
import math
from heuristics.heuristic_base import Heuristic # Ensure this import path is correct for the environment

# Helper function to parse PDDL facts like '(predicate obj1 obj2)'
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(robot-at robot1 tile_0_1)" -> ["robot-at", "robot1", "tile_0_1"]
    Returns an empty list if the fact is malformed (e.g., not starting/ending with parentheses).
    """
    if fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return []

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'floortile'.

    # Summary
    This heuristic estimates the remaining cost to achieve the goal state,
    which involves painting specific tiles with designated colors. It calculates
    the cost for each unsatisfied painting goal individually and sums these costs.
    The cost for a single goal is estimated as the minimum cost for any robot
    to achieve it, considering movement, potential color change, and the painting action itself.
    It is designed for Greedy Best-First Search and is not necessarily admissible, aiming for
    informativeness over admissibility.

    # Assumptions
    - The goal is solely defined by `(painted ?tile ?color)` predicates.
    - Each unsatisfied goal `(painted target_tile target_color)` requires:
        1. A robot moving to a tile adjacent (up/down) to `target_tile`.
        2. The robot holding the `target_color`.
        3. The robot performing a `paint_up` or `paint_down` action (cost 1).
    - The heuristic relaxes the problem by:
        - Ignoring conflicts between robots (e.g., needing the same tile, blocking paths).
        - Assuming the `clear` precondition for painting and movement can always be satisfied eventually (the cost of clearing is not explicitly modeled, but implicitly included in movement/painting actions of the plan).
        - Summing the minimum cost for each goal independently, potentially overcounting shared actions (like movement or color changes). This is acceptable for a non-admissible heuristic guiding GBFS.
    - Distances between tiles are precomputed using Breadth-First Search (BFS) on the static tile connectivity graph defined by `up`, `down`, `left`, `right` facts. Movement cost is the shortest path length (number of move actions).
    - Color change cost (`change_color` action) is 1 if needed, 0 otherwise.

    # Heuristic Initialization
    - Parses the task's goals (`task.goals`) to identify target `(painted tile color)` states (`self.goal_paint_map`).
    - Parses static facts (`task.static`) and initial state (`task.initial_state`) to:
        - Identify all robots, tiles, and colors involved in the problem.
        - Build a tile connectivity graph for movement based on `up`, `down`, `left`, `right` facts (`self.adj`).
        - Identify for each tile, which adjacent tiles a robot must be on to paint it, based on `up` and `down` facts (`self.paint_adj`).
    - Precomputes all-pairs shortest path distances between all identified tiles using BFS (`self.dist`). Stores distances as integers (number of steps).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract current robot locations (`robot_locs`), robot colors (`robot_colors`), and currently painted tiles (`painted_tiles`) from the input `node.state`.
    2.  **Identify Unsatisfied Goals:** Compare `painted_tiles` with the required `self.goal_paint_map`. Collect a list of `(tile, color)` pairs that are goals but not yet satisfied correctly in the state.
    3.  **Handle Goal State:** If there are no unsatisfied goals, the state is a goal state, return 0.
    4.  **Check Robot Availability:** If there are unsatisfied goals but no robots were identified during initialization (e.g., an instance with goals but no robots), the goal is unreachable, return infinity (`math.inf`).
    5.  **Calculate Cost per Goal:** Initialize `total_heuristic_cost = 0`. Iterate through each unsatisfied goal `g = (target_tile, target_color)`:
        a.  Initialize `min_cost_for_goal = math.inf`.
        b.  **Find Painting Locations:** Determine the set `paintable_locations` (tiles adjacent to `target_tile` from which painting is possible via `paint_up`/`paint_down`) using the precomputed `self.paint_adj`. If this set is empty for a required goal tile (meaning no static `up` or `down` fact allows painting it), the goal is impossible, return infinity for the state.
        c.  **Evaluate Each Robot:** For each robot `r` identified during initialization:
            i.  Get the robot's current location `current_loc` and held color `current_color` from the parsed state. If the robot's state is missing (e.g., not `at` a tile or `has` a color), skip this robot for this goal calculation.
            ii. **Color Cost:** Calculate `cost_change = 1` if `current_color != target_color`, else `0`.
            iii. **Movement Cost:** Find the minimum distance from `current_loc` to any tile in `paintable_locations` using the precomputed `self.dist` map. Let this be `min_dist_to_adj`. If no path exists (`min_dist_to_adj` remains infinity), this robot cannot reach a painting position for this goal. Set `cost_move = min_dist_to_adj`.
            iv. **Paint Cost:** Set `cost_paint = 1` (representing the `paint_up` or `paint_down` action).
            v.  **Total Robot Cost:** Calculate the total estimated cost for this robot to achieve this goal: `cost_for_robot_r = cost_change + cost_move + cost_paint`. Handle potential infinity if `cost_move` is infinite.
            vi. **Update Minimum:** Update `min_cost_for_goal = min(min_cost_for_goal, cost_for_robot_r)`.
        d.  **Check Goal Reachability:** If `min_cost_for_goal` remains infinity after checking all robots, this specific goal is deemed unreachable by the heuristic (no robot can reach a position to paint it with the correct color), return infinity for the state.
        e.  **Accumulate Cost:** Add the calculated `min_cost_for_goal` to `total_heuristic_cost`.
    6.  **Return Total Cost:** Return the final `total_heuristic_cost`. If it reached infinity at any point, return `math.inf`. Otherwise, return the calculated sum as an integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task information: goals, static facts,
        and initial state to build necessary data structures like the connectivity graph,
        painting locations, and precomputed distances.
        """
        self.goals = task.goals
        static_facts = task.static
        # Combine initial state and static facts for comprehensive object discovery
        all_facts = task.initial_state.union(static_facts)

        self.robots = set()
        self.tiles = set()
        self.colors = set()

        # Extract objects (robots, tiles, colors) from all known facts
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty or malformed facts
             pred = parts[0]
             try:
                 # Identify objects based on predicate structure
                 if pred == 'robot-at' and len(parts) == 3:
                     self.robots.add(parts[1]); self.tiles.add(parts[2])
                 elif pred == 'robot-has' and len(parts) == 3:
                     self.robots.add(parts[1]); self.colors.add(parts[2])
                 elif pred == 'available-color' and len(parts) == 2:
                     self.colors.add(parts[1])
                 elif pred == 'clear' and len(parts) == 2:
                     self.tiles.add(parts[1])
                 elif pred == 'painted' and len(parts) == 3:
                     self.tiles.add(parts[1]); self.colors.add(parts[2])
                 elif pred in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                     self.tiles.add(parts[1]); self.tiles.add(parts[2])
             except IndexError:
                 # Log a warning if a fact doesn't match expected structure
                 print(f"Warning: Malformed fact encountered during object extraction: {fact}")
                 continue

        # Parse goals and ensure goal objects are registered
        self.goal_paint_map = {} # Stores {tile_name: target_color}
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paint_map[tile] = color
                # Ensure tile and color from goal are known
                self.tiles.add(tile)
                self.colors.add(color)

        # Build adjacency graph for movement (adj) and painting locations (paint_adj)
        self.adj = {tile: set() for tile in self.tiles}
        # paint_adj: Key = tile to be painted; Value = set of tiles robot must be AT to paint Key
        self.paint_adj = {tile: set() for tile in self.tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) != 3: continue # Adjacency/painting facts have 3 parts
            pred = parts[0]
            t1, t2 = parts[1], parts[2]

            # Ensure tiles involved in static relations are known
            if t1 not in self.tiles: self.tiles.add(t1)
            if t2 not in self.tiles: self.tiles.add(t2)

            # Store movement adjacency (bi-directional)
            if pred in ["up", "down", "left", "right"]:
                self.adj.setdefault(t1, set()).add(t2)
                self.adj.setdefault(t2, set()).add(t1)

            # Store painting adjacency based on paint_up/paint_down actions
            # (paint_up ?r ?y ?x ?c) requires (up ?y ?x) and (robot-at ?r ?x) -> paint y from x
            if pred == "up":
                # If fact is (up tileY tileX), robot at tileX can paint tileY
                self.paint_adj.setdefault(t1, set()).add(t2)
            # (paint_down ?r ?y ?x ?c) requires (down ?y ?x) and (robot-at ?r ?x) -> paint y from x
            elif pred == "down":
                 # If fact is (down tileY tileX), robot at tileX can paint tileY
                self.paint_adj.setdefault(t1, set()).add(t2)

        # Precompute All-Pairs Shortest Paths (APSP) using BFS for all known tiles
        self.dist = {t1: {t2: math.inf for t2 in self.tiles} for t1 in self.tiles}
        for start_node in self.tiles:
            # Check if start_node is valid before starting BFS
            if start_node not in self.tiles: continue

            self.dist[start_node][start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                current_node = queue.popleft()
                current_d = self.dist[start_node][current_node]

                # Explore neighbors based on movement adjacency graph
                for neighbor in self.adj.get(current_node, set()):
                    # Check if neighbor is a valid tile (should be if adj is built correctly)
                    if neighbor not in self.tiles: continue
                    # If we found the first path or a shorter path (BFS guarantees shortest in unweighted graph)
                    if self.dist[start_node][neighbor] == math.inf:
                        self.dist[start_node][neighbor] = current_d + 1
                        queue.append(neighbor)


    def __call__(self, node):
        """
        Calculates the heuristic value (estimated cost to goal) for the given state node.
        Returns the sum of minimum estimated costs for achieving each unsatisfied goal,
        or math.inf if any goal is deemed unreachable.
        """
        state = node.state

        # --- 1. Parse Current State ---
        robot_locs = {}   # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        painted_tiles = {} # {tile_name: color_name}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            try:
                if pred == "robot-at" and len(parts) == 3:
                    robot_locs[parts[1]] = parts[2]
                elif pred == "robot-has" and len(parts) == 3:
                    robot_colors[parts[1]] = parts[2]
                elif pred == "painted" and len(parts) == 3:
                    painted_tiles[parts[1]] = parts[2]
            except IndexError:
                 # Ignore malformed facts during state parsing
                 continue

        # --- 2. Identify Unsatisfied Goals ---
        unsatisfied_goals = []
        for tile, target_color in self.goal_paint_map.items():
            current_color = painted_tiles.get(tile)
            # Add to list if tile is not painted or painted with the wrong color
            if current_color != target_color:
                unsatisfied_goals.append({'tile': tile, 'color': target_color})

        # --- 3. Handle Goal State ---
        if not unsatisfied_goals:
            return 0 # Goal reached

        # --- 4. Check Robot Availability ---
        if not self.robots:
             # If goals remain but no robots exist in the problem
             return math.inf

        # --- 5. Calculate Cost per Goal ---
        total_heuristic_cost = 0
        for goal in unsatisfied_goals:
            target_tile = goal['tile']
            target_color = goal['color']

            min_cost_for_goal = math.inf

            # --- 5b. Find Painting Locations ---
            # Get the set of tiles where a robot must be to paint the target_tile
            paintable_locations = self.paint_adj.get(target_tile)

            # If target_tile exists but no static facts allow painting it, it's impossible
            if paintable_locations is None or not paintable_locations:
                # This goal cannot be achieved according to domain definition
                return math.inf

            # --- 5c. Evaluate Each Robot ---
            found_evaluatable_robot = False # Track if at least one robot has valid state info
            for robot in self.robots:
                current_loc = robot_locs.get(robot)
                current_color = robot_colors.get(robot)

                # Cannot use this robot if its location or color is unknown in the current state
                if current_loc is None or current_color is None:
                    continue

                # Also check if the robot's current location is a valid tile known to the heuristic
                if current_loc not in self.tiles:
                    # Robot is at an unexpected location, cannot calculate distance
                    continue

                found_evaluatable_robot = True

                # --- 5c.ii. Color Cost ---
                cost_change = 0 if current_color == target_color else 1

                # --- 5c.iii. Movement Cost ---
                min_dist_to_adj = math.inf
                # Find shortest distance from current location to any required painting location
                for adj_tile in paintable_locations:
                     # Ensure the adjacent tile is valid before looking up distance
                     if adj_tile not in self.tiles: continue
                     # Get precomputed distance (will be inf if unreachable)
                     dist = self.dist.get(current_loc, {}).get(adj_tile, math.inf)
                     min_dist_to_adj = min(min_dist_to_adj, dist)

                cost_move = min_dist_to_adj # This can be math.inf

                # --- 5c.iv. Paint Cost ---
                cost_paint = 1 # Cost of the paint action itself

                # --- 5c.v. Total Robot Cost ---
                # If movement cost is infinite, total cost is infinite
                if cost_move == math.inf:
                    cost_for_robot_r = math.inf
                else:
                    # Sum the costs: change color (if needed) + move + paint
                    cost_for_robot_r = cost_change + cost_move + cost_paint

                # --- 5c.vi. Update Minimum cost for this goal across all robots ---
                min_cost_for_goal = min(min_cost_for_goal, cost_for_robot_r)

            # If no robot could be evaluated (e.g., all robots lacked state info)
            # and there are still goals, something is wrong -> treat as unreachable.
            if not found_evaluatable_robot and unsatisfied_goals:
                 return math.inf

            # --- 5d. Check Goal Reachability ---
            if min_cost_for_goal == math.inf:
                # This specific goal is deemed unreachable by any robot in the current state
                return math.inf

            # --- 5e. Accumulate Cost ---
            # Add the minimum cost found for this goal to the total heuristic value
            total_heuristic_cost += min_cost_for_goal

        # --- 6. Return Total Cost ---
        # Final check for infinity before returning
        if total_heuristic_cost == math.inf:
             return math.inf
        else:
             # Return the calculated cost as an integer
             # Use round() before int() to handle potential floating point nuances if any were introduced,
             # though all costs here (1 or distance) should be integers.
             return int(round(total_heuristic_cost))
