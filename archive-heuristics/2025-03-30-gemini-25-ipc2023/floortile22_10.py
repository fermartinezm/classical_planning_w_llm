import sys
from collections import deque
from fnmatch import fnmatch
# Assuming the base class is available at this path relative to the execution context
# If the heuristic file is in a 'heuristics' directory, and the base class is in the same directory:
# from .heuristic_base import Heuristic
# If the base class is in a different location, adjust the import path accordingly.
# For the given context, let's assume it's available as:
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string.
    Example: "(pred arg1 arg2)" -> ["pred", "arg1", "arg2"]
    Returns empty list if input is not a valid fact string format.
    """
    if isinstance(fact, str) and fact.startswith("(") and fact.endswith(")"):
        # Basic split, assumes arguments don't contain spaces
        return fact[1:-1].split()
    return []

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    for the floortile domain. The primary goal is to have specific tiles painted
    with specific colors. The heuristic calculates the estimated cost for achieving
    each unsatisfied paint goal independently and sums these costs. The cost for a
    single goal is estimated as the minimum cost for *any* available robot to
    achieve it. This cost includes the minimum movement actions to reach a tile
    adjacent to the target tile (from where painting is possible), the cost of
    changing color if necessary (1 action), and the cost of the painting action
    itself (1 action).

    # Assumptions
    - The cost of each action (move_up/down/left/right, paint_up/down, change_color) is 1.
    - The heuristic calculates the cost for each unsatisfied goal independently,
      assigning it to the robot that can achieve it with the minimum estimated cost.
      It ignores potential conflicts (e.g., two robots needing the same tile) or
      synergies (e.g., one robot painting multiple nearby tiles efficiently).
    - It assumes that paths between tiles computed via the static connectivity
      (`up`, `down`, `left`, `right` facts) can eventually be traversed. It does not
      dynamically consider if tiles are blocked by other robots, only the shortest
      path distance based on the grid structure.
    - It assumes target tiles for painting are initially `clear` or can be made `clear`
      as part of the plan. The `paint` actions require the target tile `?y` to be
      `clear`, but the heuristic focuses on the cost to get a robot to the adjacent
      tile `?x` with the right color.

    # Heuristic Initialization
    - The constructor (`__init__`) preprocesses information from the task definition.
    - It parses static facts, initial state, and goals to identify all relevant objects:
      `tiles`, `robots`, and `colors`.
    - It parses static connectivity facts (`up`, `down`, `left`, `right`) to build
      an adjacency list (`self.adj`) representing possible movements between tiles.
    - It identifies, for each tile `t`, the set of neighboring tiles (`self.paintable_neighbors[t]`)
      from which `t` can be painted. This is derived from `up(t, x)` and `down(t, x)` facts,
      meaning a robot at `x` can paint `t`.
    - It computes all-pairs shortest path distances between all tiles using Breadth-First Search (BFS)
      on the movement adjacency graph (`self.adj`). These distances are stored in `self.distances`.
    - It stores the set of goal predicates `(painted tile color)` in a dictionary
      `self.goal_painted` mapping each goal tile to its required color.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State**: In the `__call__` method, extract the current state information:
        - `robot_positions`: Dictionary mapping each robot to its current tile.
        - `robot_colors`: Dictionary mapping each robot to the color it currently holds.
        - `current_painted`: Dictionary mapping currently painted tiles to their colors.
    2.  **Identify Unsatisfied Goals**: Compare `current_painted` with `self.goal_painted`.
        Create a list `unsatisfied_goals` containing `(tile, color)` pairs for goals
        not yet met in the current state.
    3.  **Handle Goal State**: If `unsatisfied_goals` is empty, the current state is a goal state,
        so the heuristic value is 0.
    4.  **Estimate Cost per Unsatisfied Goal**: Iterate through each `(target_tile, target_color)`
        in `unsatisfied_goals`:
        a.  **Determine Possible Painting Locations**: Find the set `possible_paint_locs` =
            `self.paintable_neighbors[target_tile]`. These are the tiles a robot must be on
            to paint `target_tile`. If this set is empty (based on static facts), the goal
            is structurally impossible to achieve; return infinity.
        b.  **Calculate Minimum Robot Cost**: For this specific goal, find the minimum cost
            required for *any* robot to achieve it. Iterate through each `robot` `r`:
            i.   Get the robot's `current_tile` and `current_color` from the parsed state.
                 If information for this robot is missing, skip it.
            ii.  Calculate `move_cost`: Find the minimum shortest path distance from
                 `current_tile` to *any* tile in `possible_paint_locs`. Use the precomputed
                 `self.distances`. If `current_tile` or `paint_loc` is invalid, or if no path
                 exists from `current_tile` to any `paint_loc`, the `move_cost` for this robot
                 is effectively infinity.
            iii. Calculate `color_cost`: This is 1 if `current_color != target_color`, else 0.
            iv.  Set `paint_cost = 1` (for the `paint_up` or `paint_down` action).
            v.   The total cost for robot `r` to achieve this goal is `move_cost + color_cost + paint_cost`.
                 If `move_cost` was infinity, the total cost is infinity.
        c.  **Find Minimum Goal Cost**: The estimated cost for this single goal (`min_cost_for_this_goal`)
            is the minimum total cost found across all robots in step 4b.
        d.  **Check Reachability**: If `min_cost_for_this_goal` remains infinity after checking all
            robots, it means this specific goal is currently unreachable by any robot. The overall
            state is likely unsolvable from this point; return infinity.
    5.  **Sum Costs**: The total heuristic value is the sum of `min_cost_for_this_goal` for all
        `unsatisfied_goals`.
    6.  **Return Value**: Return the calculated total heuristic value. It will be 0 for goal states,
        a positive integer for reachable non-goal states, and infinity for states from which
        at least one goal is determined to be unreachable.
    """

    def __init__(self, task):
        # Initialize base class if necessary (depends on Heuristic base class implementation)
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- 1. Collect Objects (Tiles, Robots, Colors) ---
        self.tiles = set()
        self.robots = set()
        self.colors = set()

        # Combine all known facts (static, init, goal) to ensure all objects are found
        all_facts = static_facts.union(task.initial_state).union(self.goals)

        for fact_str in all_facts:
            parts = get_parts(fact_str)
            if not parts: continue # Skip if parsing failed (e.g., empty string)

            pred = parts[0]
            args = parts[1:]

            try:
                if pred in ["up", "down", "left", "right"]:
                    # Assumes args[0] and args[1] are tiles
                    self.tiles.add(args[0])
                    self.tiles.add(args[1])
                elif pred == "clear":
                    # Assumes args[0] is a tile
                     self.tiles.add(args[0])
                elif pred == "painted":
                    # Assumes args[0] is tile, args[1] is color
                     self.tiles.add(args[0])
                     self.colors.add(args[1])
                elif pred == "available-color":
                    # Assumes args[0] is a color
                    self.colors.add(args[0])
                elif pred == "robot-at":
                    # Assumes args[0] is robot, args[1] is tile
                    self.robots.add(args[0])
                    self.tiles.add(args[1])
                elif pred == "robot-has":
                    # Assumes args[0] is robot, args[1] is color
                    self.robots.add(args[0])
                    self.colors.add(args[1])
                # Ignore unknown predicates or predicates not defining objects of interest
            except IndexError:
                # Handles cases where a predicate expects args but none are present in parts[1:]
                # print(f"Warning: Skipping fact due to unexpected args: {fact_str}")
                pass


        # --- 2. Build Adjacency Structures (Movement and Painting) ---
        # adj: tile -> set of tiles reachable by one move action
        self.adj = {tile: set() for tile in self.tiles}
        # paintable_neighbors: tile_to_paint -> set of tiles robot must be at to paint it
        self.paintable_neighbors = {tile: set() for tile in self.tiles}

        for fact_str in static_facts:
            parts = get_parts(fact_str)
            if not parts: continue

            pred = parts[0]
            args = parts[1:]

            try:
                # For connectivity predicates, expect 2 arguments
                tile1, tile2 = args[0], args[1]

                # Ensure both arguments are known tiles before processing
                if tile1 not in self.tiles or tile2 not in self.tiles:
                    continue # Skip if involves unknown/invalid objects

                # Note: PDDL defines (pred tile_y tile_x) where action is from x to y
                # Example: (up tile_1_1 tile_0_1) means move UP from tile_0_1 to tile_1_1
                # Example: paint_up(r, y, x, c) requires (up y x), robot at x paints y.

                if pred == "up": # (up y x)
                    self.adj[tile2].add(tile1) # Can move from x to y
                    self.paintable_neighbors[tile1].add(tile2) # Robot at x can paint_up y
                elif pred == "down": # (down y x)
                    self.adj[tile2].add(tile1) # Can move from x to y
                    self.paintable_neighbors[tile1].add(tile2) # Robot at x can paint_down y
                elif pred == "left": # (left y x)
                    self.adj[tile2].add(tile1) # Can move from x to y
                elif pred == "right": # (right y x)
                    self.adj[tile2].add(tile1) # Can move from x to y
            except IndexError:
                 # print(f"Warning: Skipping static fact due to unexpected args: {fact_str}")
                 pass

        # --- 3. Compute All-Pairs Shortest Paths ---
        self.distances = self._compute_shortest_paths()

        # --- 4. Store Goal Painted Information ---
        self.goal_painted = {} # tile -> color
        for goal_str in self.goals:
            parts = get_parts(goal_str)
            if not parts: continue

            try:
                if parts[0] == "painted":
                    tile, color = parts[1], parts[2]
                    # Ensure tile and color are known objects before storing goal
                    if tile in self.tiles and color in self.colors:
                        self.goal_painted[tile] = color
                    # else: print(f"Warning: Goal '{goal_str}' involves unknown objects.")
            except IndexError:
                 # print(f"Warning: Skipping goal due to unexpected args: {goal_str}")
                 pass


    def _compute_shortest_paths(self):
        """
        Computes all-pairs shortest path distances using BFS for each tile as a source.
        Returns a dictionary distances[start_tile][end_tile] = shortest_distance.
        Distance is float('inf') if unreachable.
        """
        distances = {t: {t2: float('inf') for t2 in self.tiles} for t in self.tiles}
        for start_node in self.tiles:
            # Basic check if start_node is valid (should always be true)
            if start_node not in distances: continue

            distances[start_node][start_node] = 0
            queue = deque([start_node])

            # Keep track of visited nodes *within this specific BFS run*
            # Alternatively, checking distance == inf achieves the same for unweighted graphs
            visited_in_bfs = {start_node}

            while queue:
                current_node = queue.popleft()
                current_dist = distances[start_node][current_node]

                # Explore neighbors based on the movement adjacency list
                for neighbor in self.adj.get(current_node, set()):
                    # Check if neighbor is a valid tile and hasn't been visited in this BFS
                    if neighbor in self.tiles and neighbor not in visited_in_bfs:
                        visited_in_bfs.add(neighbor)
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
                    # If using distance check instead of visited set:
                    # if neighbor in distances[start_node] and distances[start_node][neighbor] == float('inf'):
                    #    distances[start_node][neighbor] = current_dist + 1
                    #    queue.append(neighbor)

        return distances


    def __call__(self, node):
        """
        Calculate the heuristic value for the given state node.
        Input `node` is expected to have a `state` attribute which is a frozenset of facts.
        """
        state = node.state

        # --- 1. Parse Current State ---
        robot_positions = {} # robot -> tile
        robot_colors = {}    # robot -> color
        current_painted = {} # tile -> color

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue

            pred = parts[0]
            args = parts[1:]
            try:
                # Check if the objects involved are known objects from initialization
                if pred == "robot-at" and args[0] in self.robots and args[1] in self.tiles:
                    robot_positions[args[0]] = args[1]
                elif pred == "robot-has" and args[0] in self.robots and args[1] in self.colors:
                     robot_colors[args[0]] = args[1]
                elif pred == "painted" and args[0] in self.tiles and args[1] in self.colors:
                    current_painted[args[0]] = args[1]
            except IndexError:
                 # print(f"Warning: Skipping state fact due to unexpected args: {fact_str}")
                 pass

        # --- 2. Identify Unsatisfied Goals ---
        unsatisfied_goals = []
        for tile, goal_color in self.goal_painted.items():
            # A goal is unsatisfied if the tile is not painted, or painted with the wrong color
            if tile not in current_painted or current_painted[tile] != goal_color:
                unsatisfied_goals.append((tile, goal_color))

        # --- 3. Handle Goal State ---
        if not unsatisfied_goals:
            return 0 # All goals are satisfied

        # --- 4 & 5. Calculate Heuristic Value (Sum of Min Costs per Goal) ---
        total_heuristic_value = 0
        for target_tile, target_color in unsatisfied_goals:
            min_cost_for_this_goal = float('inf')

            # Check if target tile is valid before accessing neighbors map
            if target_tile not in self.paintable_neighbors:
                 # This should not happen if initialization was correct and goals are valid
                 # print(f"Error: Target goal tile {target_tile} not found in paintable neighbors map.")
                 return float('inf') # Cannot achieve goal if tile doesn't exist in map

            possible_paint_locs = self.paintable_neighbors.get(target_tile, set())

            if not possible_paint_locs:
                # If static analysis shows no tile can paint the target_tile
                # print(f"Warning: Goal tile {target_tile} has no paintable neighbors based on static facts.")
                return float('inf') # Goal is structurally impossible

            # Find the minimum cost for *any* robot to achieve this goal
            for robot in self.robots:
                # Ensure robot has current position and color information in this state
                if robot not in robot_positions or robot not in robot_colors:
                    # If a robot is missing state info, it cannot contribute to this goal now
                    continue

                current_tile = robot_positions[robot]
                current_color = robot_colors[robot]

                # Check if current_tile is valid before accessing distances map
                if current_tile not in self.distances:
                     # Should not happen if initialization included all tiles robots can be on
                     # print(f"Error: Robot {robot}'s current tile {current_tile} not in distance map.")
                     continue # Cannot calculate distance for this robot

                # Find min distance from robot's current_tile to any possible paint_loc
                min_dist_to_paint_loc = float('inf')
                for paint_loc in possible_paint_locs:
                    # Check if paint_loc is valid before accessing distances map
                    if paint_loc not in self.distances[current_tile]:
                        # Should not happen if distance map is complete for all tiles
                        # print(f"Error: Paint location {paint_loc} not in distance map from {current_tile}.")
                        continue # Cannot reach this specific paint_loc

                    dist = self.distances[current_tile][paint_loc]
                    # Update minimum distance if this path is shorter (and reachable)
                    if dist != float('inf'):
                        min_dist_to_paint_loc = min(min_dist_to_paint_loc, dist)

                # Calculate total cost for this robot if a path to a paint location exists
                if min_dist_to_paint_loc == float('inf'):
                    cost_for_robot_r = float('inf') # No path found for this robot to paint this tile
                else:
                    move_cost = min_dist_to_paint_loc
                    # Cost to change color is 1 if needed, 0 otherwise
                    color_cost = 0 if current_color == target_color else 1
                    # Cost of the paint action itself is 1
                    paint_cost = 1
                    cost_for_robot_r = move_cost + color_cost + paint_cost

                # Update the minimum cost found so far for *this specific goal*
                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_for_robot_r)

            # --- Check Goal Reachability ---
            # If after checking all robots, the min cost is still infinity, this goal is unreachable
            if min_cost_for_this_goal == float('inf'):
                # print(f"Warning: Unsatisfied goal {(target_tile, target_color)} is unreachable by any robot in the current state.")
                return float('inf') # Return infinity to prune this state/path in search

            # --- Accumulate Cost ---
            # Add the minimum cost found for this goal to the total heuristic value
            total_heuristic_value += min_cost_for_this_goal

        # --- 6. Return Final Heuristic Value ---
        # The total value is the sum of minimum costs for each unsatisfied goal
        return total_heuristic_value

