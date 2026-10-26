import heapq
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this path is correct for the planner framework

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string.
    Example: "(robot-at r1 t1)" -> ["robot-at", "r1", "t1"]
    """
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    Estimates the cost to paint all required tiles by greedily assigning
    each unpainted goal tile to the robot that can paint it with the
    minimum estimated cost (movement + color change + paint action).
    The heuristic simulates the state changes (robot position and color)
    after each assignment to provide a potentially more accurate estimate
    than simple summation, aiming to guide a Greedy Best-First Search effectively.

    # Assumptions
    - All actions (move, paint, change_color) have a cost of 1.
    - The heuristic ignores the `clear` predicate constraints for movement
      and painting path calculations, focusing on connectivity and basic action costs.
      This is a relaxation common in heuristics.
    - Multiple robots can work in parallel, but the heuristic assigns goals sequentially
      based on the minimum immediate estimated cost to achieve one remaining goal.
    - The tile connectivity graph is static and derived from `up`, `down`, `left`, `right` facts.
    - Problems are well-formed (e.g., goal tiles are paintable via adjacent tiles).

    # Heuristic Initialization
    - Parses goal facts `(painted tile color)` to identify target tiles and colors.
    - Parses static facts (`up`, `down`, `left`, `right`, `available-color`) to:
        - Build a tile connectivity graph (adjacency list).
        - Identify the "painting tiles" for each target tile: the adjacent tiles a robot
          must occupy to paint the target tile using `paint_up` or `paint_down`.
        - Collect sets of all known tiles and colors.
    - Precomputes all-pairs shortest paths (APSP) between all known tiles using Breadth-First Search (BFS),
      storing the distances. This allows for O(1) lookup of movement cost estimates.
    - Stores the set of goal tuples `(tile, color)`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the current location (`robot-at`) and held color (`robot-has`)
        for each robot, and the set of currently `painted` tiles from the input state. Identify all robots present.
    2.  **Identify Unsatisfied Goals:** Determine the set of `(painted tile color)` goals that are
        required by the task but are not yet true in the current state.
    3.  **Check Goal Achievement:** If there are no unsatisfied goals, the current state is a goal state,
        and the heuristic value is 0.
    4.  **Initialize Simulation:** Create temporary copies of the current robot locations and colors.
        These will be updated during the heuristic calculation to simulate the progression towards the goal.
        Initialize the total heuristic cost `h = 0`.
    5.  **Greedy Assignment Loop:** Repeat the following steps until all unsatisfied goals are assigned:
        a.  **Calculate Minimum Cost Assignment:** Iterate through every robot `r` and every remaining
            unsatisfied goal `g = (target_tile, target_color)`. For each such pair:
            i.   Identify the set of `painting_tiles` (`PT`) associated with `target_tile` (precomputed in `__init__`).
                 These are the tiles the robot `r` must be at to paint `target_tile`.
            ii.  For each `pt` in `PT`:
                 - Calculate movement cost: `move_cost = distance(sim_robot_locs[r], pt)` using the precomputed APSP distances. If `pt` is unreachable from the robot's current simulated location, this cost is infinity.
                 - Calculate color change cost: `cc_cost = 1` if the robot's current simulated color `sim_robot_colors[r]` is different from the `target_color`, otherwise `cc_cost = 0`.
                 - Calculate the total estimated cost for this specific option (`r` paints `g` from `pt`): `cost = move_cost + cc_cost + 1` (the final +1 represents the `paint` action itself).
            iii. Keep track of the minimum `cost` found across all combinations of `r`, `g`, and `pt` considered in this iteration. Store the corresponding `best_r`, `best_g` (goal tuple), `best_pt` (painting tile), and `min_cost`.
        b.  **Handle Unsolvable States:** If after checking all combinations, no valid assignment was found (e.g., all remaining goals are unreachable), it suggests a dead end. Return `float('inf')`.
        c.  **Update Heuristic Value:** Add the `min_cost` of the best assignment found to the total heuristic cost `h`.
        d.  **Update Simulated State:** Modify the simulation state to reflect the chosen assignment:
            i.   Update the simulated location of the chosen robot: `sim_robot_locs[best_r] = best_pt`.
            ii.  Update the simulated color of the chosen robot: `sim_robot_colors[best_r] = best_g[1]` (the target color).
        e.  **Remove Achieved Goal:** Remove `best_g` from the set of remaining unsatisfied goals.
    6.  **Return Total Cost:** Once the loop finishes (all goals have been assigned), return the total accumulated heuristic value `h`.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.tiles = set()
        self.robots = set() # Will be populated in __call__ based on state
        self.colors = set()
        self.adj = {}  # Adjacency list for tiles: tile -> {neighbor1, ...}
        # Stores tiles from which a target tile can be painted: target_tile -> {painting_tile1, ...}
        self.painting_tiles = {}
        self.goal_tuples = set() # Set of (tile, color) goals
        self.distances = {} # Stores APSP: tile1 -> {tile2: distance}

        # --- Parse Static Facts and Identify Objects ---
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            # Build connectivity graph (undirected for distance calculation)
            if pred in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                self.tiles.add(t1)
                self.tiles.add(t2)
                self.adj.setdefault(t1, set()).add(t2)
                self.adj.setdefault(t2, set()).add(t1) # Add edge in both directions

                # Store painting tile relationship based on paint actions
                # paint_up(?r, ?y, ?x, ?c) requires (up ?y ?x) and robot at ?x to paint ?y
                # paint_down(?r, ?y, ?x, ?c) requires (down ?y ?x) and robot at ?x to paint ?y
                if pred == "up": # (up target_tile painting_tile)
                    target, painter_loc = t1, t2
                    self.painting_tiles.setdefault(target, set()).add(painter_loc)
                elif pred == "down": # (down target_tile painting_tile)
                    target, painter_loc = t1, t2
                    self.painting_tiles.setdefault(target, set()).add(painter_loc)

            elif pred == "available-color":
                self.colors.add(parts[1])

        # Parse goals to get target (tile, color) pairs
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tuples.add((tile, color))
                self.tiles.add(tile) # Ensure goal tiles are known
                self.colors.add(color) # Ensure goal colors are known

        # --- Precompute All-Pairs Shortest Paths (APSP) using BFS ---
        # Initialize distances to infinity
        for t1 in self.tiles:
            self.distances[t1] = {t2: float('inf') for t2 in self.tiles}

        # Run BFS from each tile
        for start_node in self.tiles:
            if start_node not in self.distances: continue # Should not happen if tiles populated correctly

            self.distances[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node} # Keep track of visited nodes per BFS run

            while queue:
                current_tile = queue.popleft()
                current_dist = self.distances[start_node][current_tile]

                # Explore neighbors
                for neighbor in self.adj.get(current_tile, set()):
                    # Check if distance can be improved (or first time visited)
                    # In standard BFS from start_node, visited check is sufficient
                    if neighbor not in visited:
                         visited.add(neighbor)
                         self.distances[start_node][neighbor] = current_dist + 1
                         queue.append(neighbor)
                    # If graph had weights, we'd check:
                    # if self.distances[start_node][neighbor] > current_dist + 1:
                    #    self.distances[start_node][neighbor] = current_dist + 1
                    #    queue.append(neighbor) # Re-add for potential further updates if using Dijkstra-like approach


    def __call__(self, node):
        state = node.state

        # --- Parse Current State ---
        robot_locs = {}
        robot_colors = {}
        painted_now = set()
        current_robots = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locs[robot] = tile
                current_robots.add(robot)
            elif pred == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                # Ensure robot is added even if only 'robot-has' is seen first
                current_robots.add(robot)
            elif pred == "painted":
                tile, color = parts[1], parts[2]
                painted_now.add((tile, color))

        # Store the set of robots found in this state if not already stored
        # (Useful if __init__ couldn't determine all robots)
        if not self.robots:
             self.robots = current_robots


        # --- Identify Unsatisfied Goals ---
        unsatisfied_goals = self.goal_tuples - painted_now

        # --- Check Goal Achievement ---
        if not unsatisfied_goals:
            return 0

        # --- Greedy Assignment Simulation ---
        h_cost = 0
        # Create copies for simulation within this heuristic call
        sim_robot_locs = robot_locs.copy()
        sim_robot_colors = robot_colors.copy()
        remaining_goals = unsatisfied_goals.copy()

        # Check if all robots have location and color info (basic state validity)
        for r in current_robots:
            if r not in sim_robot_locs or r not in sim_robot_colors:
                 # This implies an inconsistent or unexpected state format.
                 # Return infinity as the state seems invalid for planning.
                 # print(f"Warning: Robot {r} missing location or color info in state.")
                 return float('inf')


        # --- Greedy Assignment Loop ---
        while remaining_goals:
            min_cost = float('inf')
            best_r, best_g, best_pt = None, None, None

            # Find the globally cheapest (robot, goal, painting_tile) assignment
            # among all current robots and remaining goals.
            for r in current_robots:
                current_r_loc = sim_robot_locs.get(r)
                current_r_color = sim_robot_colors.get(r)

                # Should not happen due to check above, but defensive check
                if current_r_loc is None or current_r_color is None:
                    continue

                for g in remaining_goals:
                    target_tile, target_color = g

                    # Get the tiles from where target_tile can be painted
                    possible_painting_tiles = self.painting_tiles.get(target_tile)

                    # If a goal tile has no associated painting tiles in static facts, it's unachievable.
                    if not possible_painting_tiles:
                        # This goal cannot be satisfied according to the domain rules parsed.
                        # The overall problem might be unsolvable. Return inf.
                        # print(f"Warning: Goal {g} seems unachievable (no painting tiles found).")
                        return float('inf')


                    for pt in possible_painting_tiles:
                        # --- Cost Calculation ---
                        # Movement cost from robot's current simulated location to the painting tile
                        move_cost = self.distances.get(current_r_loc, {}).get(pt, float('inf'))

                        # If the painting tile is unreachable, skip this option
                        if move_cost == float('inf'):
                             continue

                        # Color change cost
                        cc_cost = 0 if current_r_color == target_color else 1

                        # Total cost: move + change_color (if needed) + paint
                        cost = move_cost + cc_cost + 1

                        # --- Update Best Assignment Found So Far ---
                        if cost < min_cost:
                            min_cost = cost
                            best_r, best_g, best_pt = r, g, pt

            # --- Handle Unsolvable States ---
            # If no assignment could be found in this iteration (min_cost is still inf),
            # it means remaining goals are unreachable from current simulated robot states.
            if best_r is None:
                # print(f"Warning: No reachable assignment found for remaining goals: {remaining_goals}")
                return float('inf')

            # --- Update Heuristic Value ---
            h_cost += min_cost

            # --- Update Simulated State ---
            # The chosen robot 'best_r' is now simulated to be at 'best_pt'
            # and holding the required color 'best_g[1]'.
            sim_robot_locs[best_r] = best_pt
            sim_robot_colors[best_r] = best_g[1] # best_g is (tile, color)

            # --- Remove Achieved Goal ---
            remaining_goals.remove(best_g)

        # --- Return Total Cost ---
        return h_cost

