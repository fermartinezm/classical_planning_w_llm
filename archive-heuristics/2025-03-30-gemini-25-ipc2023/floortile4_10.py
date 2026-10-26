import heapq
from collections import deque
from fnmatch import fnmatch # Keep fnmatch in case needed later, though split() is used now
from heuristics.heuristic_base import Heuristic # Assuming this base class exists and is imported correctly

def get_parts(fact: str):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(robot-at robot1 tile_0_1)" -> ["robot-at", "robot1", "tile_0_1"]
    """
    return fact[1:-1].split()

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    Estimates the cost to reach the goal state by summing estimates for achieving
    each unsatisfied `(painted tile color)` goal. The cost for each goal considers
    the minimum actions (move, change color, paint) required by any single robot
    to achieve that goal individually. The heuristic aims to be informative for
    Greedy Best-First Search and does not need to be admissible.

    # Assumptions
    - The tile layout (adjacency defined by up/down/left/right) is static.
    - All tiles forming the grid are reachable from each other if the grid is connected.
    - Each robot always holds exactly one color at any time relevant for painting.
    - The cost ignores potential conflicts between robots (e.g., needing the same tile)
      and negative interactions (e.g., one robot needing to paint a tile currently
      occupied by another robot, requiring the second robot to move first).
    - The heuristic sums the costs calculated for the *best* robot assigned to each
      *individual* goal. This might overestimate if one robot can efficiently handle
      multiple nearby goals or underestimate if conflicts/interference are significant.

    # Heuristic Initialization
    - Parses static facts (`task.static`) to build a tile adjacency graph (`self.adj`)
      representing possible movements between tiles.
    - Computes All-Pairs Shortest Paths (APSP) between all tiles using Breadth-First Search
      (BFS) starting from each tile. Distances are stored in `self.distances`.
    - Identifies all unique tiles, robots, and colors present in the problem instance.
    - Determines the set of adjacent tiles (`self.paint_adj`) from which each target tile
      can be painted based on the static `up/down/left/right` predicates.
    - Stores the set of goal predicates `(painted tile color)` from `task.goals`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unsatisfied Goals**: Determine the set of `(painted target_tile target_color)`
        goal predicates that are not currently true in the `node.state`. If this set is empty,
        the state is a goal state, and the heuristic value is 0.
    2.  **Parse Current State**: Extract the current location (`robot_locations`) and
        held color (`robot_colors`) for each robot from the `node.state`. Also, identify
        all tiles currently painted (`painted_in_state`).
    3.  **Calculate Cost per Unsatisfied Goal**: Iterate through each unsatisfied goal
        `g = (target_tile, target_color)`:
        a.  Find the set of adjacent tiles (`paint_locations`) from which `target_tile`
            can be painted, using the precomputed `self.paint_adj`. If no such locations
            exist, the goal is statically impossible (return infinity).
        b.  Initialize `min_cost_for_g = infinity` to track the minimum cost found so far
            to achieve this specific goal `g`. Also, store the movement (`best_move_cost_for_goal`)
            and color change (`best_color_cost_for_goal`) components associated with that minimum cost.
        c.  Iterate through each robot `r`:
            i.  Retrieve the robot's current tile `robot_tile` and color `robot_color`.
                If the robot's location or color isn't defined in the state, skip it
                (or handle as an error, though this shouldn't typically happen in valid states).
            ii. Calculate the color change cost (`cost_c`): It's 1 if `robot_color` is not
                `target_color`, otherwise 0. A robot must hold *some* color to be able
                to change it or paint.
            iii. Calculate the minimum movement cost (`cost_m`): Find the shortest path distance
                 from `robot_tile` to any tile in `paint_locations`. This uses the precomputed
                 APSP stored in `self.distances`. If no path exists (distance is infinity),
                 this robot cannot reach a position to paint the target tile.
            iv. The cost of the paint action itself (`cost_p`) is always 1.
            v.  If the robot can reach a painting location (`cost_m` is not infinity) and holds a color,
                calculate the total cost for this robot `r` to achieve goal `g`:
                `total_cost_r = cost_c + cost_m + cost_p`.
            vi. Update `min_cost_for_g` if `total_cost_r` is lower than the current minimum.
                Simultaneously, update `best_move_cost_for_goal` to `cost_m` and
                `best_color_cost_for_goal` to `cost_c`.
        d.  After checking all robots, if `min_cost_for_g` is still infinity, it means no robot
            can achieve this goal from the current state (return infinity, indicating a dead end).
        e.  Add the `best_move_cost_for_goal` and `best_color_cost_for_goal` (corresponding to the
            cheapest robot for this goal) to running totals `total_movement_cost` and
            `total_color_change_cost`, respectively.
    4.  **Aggregate Costs**:
        a.  The total paint cost (`total_paint_cost`) is the number of unsatisfied goals.
        b.  The total movement cost is the sum of the minimum movement costs calculated for each goal.
        c.  The total color change cost is the sum of the minimum color change costs for each goal.
    5.  **Final Heuristic Value**: The heuristic estimate `h` is the sum of these three components:
        `h = total_paint_cost + total_movement_cost + total_color_change_cost`. Return `max(0, h)`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Extract objects and static relations ---
        self.tiles = set()
        self.robots = set()
        self.colors = set()
        self.adj = {} # Adjacency for movement: tile -> {neighbor1, neighbor2,...}
        self.paint_adj = {} # Adjacency for painting: target_tile -> {source_tile1, ...}

        # Parse static facts to find tiles, colors, and adjacency
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate in ["up", "down", "left", "right"]:
                # args[0] is the tile that can be painted (target)
                # args[1] is the tile the robot must be at (source)
                target_tile, source_tile = args
                self.tiles.add(target_tile)
                self.tiles.add(source_tile)
                # Add bi-directional adjacency for movement graph
                self.adj.setdefault(target_tile, set()).add(source_tile)
                self.adj.setdefault(source_tile, set()).add(target_tile)
                # Add paint adjacency: robot at source_tile can paint target_tile
                self.paint_adj.setdefault(target_tile, set()).add(source_tile)
            elif predicate == "available-color":
                self.colors.add(args[0])

        # Parse initial state to find robots and potentially missed objects
        for fact in task.initial_state:
             parts = get_parts(fact)
             predicate = parts[0]
             args = parts[1:]
             if predicate == "robot-at":
                 self.robots.add(args[0])
                 self.tiles.add(args[1]) # Ensure robot's starting tile is known
             elif predicate == "robot-has":
                 self.robots.add(args[0])
                 self.colors.add(args[1]) # Ensure robot's starting color is known
             elif predicate == "painted":
                 self.tiles.add(args[0])
                 self.colors.add(args[1])
             elif predicate == "clear":
                 self.tiles.add(args[0])

        # Ensure all tiles have entries in adjacency dicts, even if empty
        for tile in self.tiles:
            self.paint_adj.setdefault(tile, set())
            self.adj.setdefault(tile, set())

        # --- Compute All-Pairs Shortest Paths (APSP) ---
        self.distances = self._compute_apsp()

        # --- Store goal predicates ---
        self.goal_predicates = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                # Store as tuple: (tile, color)
                self.goal_predicates.add(tuple(parts[1:]))

    def _compute_apsp(self):
        """Computes all-pairs shortest paths using BFS from each tile."""
        distances = {t: {t2: float('inf') for t2 in self.tiles} for t in self.tiles}
        for start_node in self.tiles:
            # Check if start_node is actually in the graph described by adj
            if start_node not in self.adj and start_node not in distances: continue

            distances[start_node][start_node] = 0
            queue = deque([start_node])
            # Visited set needs to be reset for each BFS starting from a different node
            visited = {start_node}

            while queue:
                current_node = queue.popleft()
                # Ensure current_node exists in the distance map for start_node
                if current_node not in distances[start_node]: continue
                current_dist = distances[start_node][current_node]

                # Iterate through neighbors from the adjacency list
                for neighbor in self.adj.get(current_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # Ensure neighbor exists in the distance map before assignment
                        if neighbor in distances[start_node]:
                           distances[start_node][neighbor] = current_dist + 1
                           queue.append(neighbor)
                        # else: handle potential inconsistency if neighbor wasn't in self.tiles initially

        return distances

    def __call__(self, node):
        """Calculate the heuristic value for the given state node."""
        state = node.state

        # --- Parse current state ---
        robot_locations = {} # robot -> tile
        robot_colors = {r: None for r in self.robots} # robot -> color, default None
        painted_in_state = set() # set of (tile, color) tuples

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == "robot-at" and args[0] in self.robots:
                robot_locations[args[0]] = args[1]
            elif predicate == "robot-has" and args[0] in self.robots:
                robot_colors[args[0]] = args[1]
            elif predicate == "painted":
                painted_in_state.add(tuple(args)) # (tile, color)

        # --- Find unsatisfied goals ---
        unsatisfied_goals = self.goal_predicates - painted_in_state

        if not unsatisfied_goals:
            return 0 # Goal state reached

        # --- Calculate costs per goal ---
        total_paint_cost = len(unsatisfied_goals) # Each needs one paint action
        total_movement_cost = 0
        total_color_change_cost = 0

        for target_tile, target_color in unsatisfied_goals:
            min_cost_for_goal = float('inf')
            best_move_cost_for_goal = float('inf')
            best_color_cost_for_goal = float('inf')

            # Tiles from where target_tile can be painted
            paint_locations = self.paint_adj.get(target_tile, set())

            # If a tile has a goal but no adjacent tile allows painting it, it's impossible
            if not paint_locations:
                 # print(f"Warning: Goal tile {target_tile} cannot be painted from any adjacent tile.")
                 return float('inf')

            possible_to_achieve_this_goal = False
            for robot in self.robots:
                # Ensure robot's current state is known
                if robot not in robot_locations:
                    # print(f"Warning: Robot {robot} location unknown in state.")
                    continue
                robot_tile = robot_locations[robot]
                robot_color = robot_colors[robot] # Can be None

                # Robot needs a color to paint or change color
                if robot_color is None:
                    continue

                # Cost component 1: Color change
                cost_c = 1 if robot_color != target_color else 0

                # Cost component 2: Movement
                min_dist_r = float('inf')
                # Check reachability from robot_tile to any required paint_location
                if robot_tile not in self.distances:
                    # Should not happen if APSP computed correctly for all tiles
                    # print(f"Warning: Robot tile {robot_tile} not in distance matrix.")
                    continue

                for adj_tile in paint_locations:
                    # Ensure adj_tile is valid before lookup
                    if adj_tile in self.distances[robot_tile]:
                        dist = self.distances[robot_tile][adj_tile]
                        min_dist_r = min(min_dist_r, dist)
                    # else: print(f"Warning: Paint location {adj_tile} not in distance matrix for {robot_tile}.")


                # If robot cannot reach any required adjacent tile, it cannot perform this task
                if min_dist_r == float('inf'):
                    continue

                cost_m = min_dist_r

                # Cost component 3: Paint action
                cost_p = 1

                # Total cost for this robot to achieve this goal
                total_cost_r = cost_c + cost_m + cost_p
                possible_to_achieve_this_goal = True # Mark that at least one robot could do it

                # Check if this robot is the best one found so far for this goal
                if total_cost_r < min_cost_for_goal:
                    min_cost_for_goal = total_cost_r
                    # Store the component costs associated with this best robot
                    best_move_cost_for_goal = cost_m
                    best_color_cost_for_goal = cost_c

            # After checking all robots for this goal:
            if not possible_to_achieve_this_goal:
                # If no robot could possibly achieve this goal (e.g., all unreachable)
                # print(f"Warning: Unsatisfied goal {(target_tile, target_color)} seems unreachable by any robot.")
                return float('inf') # State is a dead end

            # Add the costs associated with the best robot found for this goal
            total_movement_cost += best_move_cost_for_goal
            total_color_change_cost += best_color_cost_for_goal

        # --- Final heuristic value ---
        # Sum of:
        # 1. Number of paint actions needed (one per unsatisfied goal)
        # 2. Sum of minimum movement costs for each goal (by its best robot)
        # 3. Sum of minimum color change costs for each goal (by its best robot)
        h_value = total_paint_cost + total_movement_cost + total_color_change_cost

        # Ensure heuristic is non-negative
        return max(0, h_value)

