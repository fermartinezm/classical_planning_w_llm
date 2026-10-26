import itertools
import heapq
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts against patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the pattern length
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL floortile domain.

    # Summary
    This heuristic estimates the remaining cost to achieve all goal `(painted tile color)` facts.
    It calculates the cost for each unsatisfied goal individually and sums these costs.
    The cost for a single goal considers the minimum effort required by any available robot
    to reach a suitable position, potentially change to the required color, and perform the paint action.
    Movement costs are based on precomputed shortest path distances between all tiles.

    # Assumptions
    - All actions have a cost of 1.
    - Robots can only paint tiles directly above (`paint_up`) or below (`paint_down`) their current location.
    - The `tile_X_Y` naming convention implies a grid, but the heuristic relies solely on the explicit `up/down/left/right` predicates for connectivity.
    - Robots always hold exactly one color (as per `change_color` and initial states).
    - The grid structure is static.

    # Heuristic Initialization
    - Extracts all tiles, robots, and available colors.
    - Parses static connectivity predicates (`up`, `down`, `left`, `right`) to build an adjacency graph of tiles.
    - Precomputes all-pairs shortest path distances between tiles using Breadth-First Search (BFS). This represents the minimum move actions required between any two tiles.
    - Identifies the required robot positions for painting each specific tile based on `up` and `down` predicates.
    - Stores the set of goal predicates `(painted ?tile ?color)`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unsatisfied Goals:** Determine which `(painted ?tile ?color)` goal predicates are not currently true in the given state. If all goals are satisfied, the heuristic value is 0.
    2.  **Iterate Through Unsatisfied Goals:** For each unsatisfied goal `(painted goal_tile goal_color)`:
        a.  **Find Required Painting Positions:** Retrieve the precomputed set of tiles `P` from which `goal_tile` can be painted (i.e., tiles `t` such that `(up goal_tile t)` or `(down goal_tile t)` is a static fact). If no such tile exists, the goal is impossible (return infinity).
        b.  **Calculate Minimum Cost Per Robot:** For each robot `r`:
            i.   Get the robot's current tile `current_tile` and current color `current_color`.
            ii.  **Movement Cost:** Find the minimum precomputed shortest path distance from `current_tile` to any tile `p` in the set `P`. This is the minimum number of `move_*` actions required. If no path exists (distance is infinity), this robot cannot reach a painting position for this goal.
            iii. **Color Change Cost:** If `current_color` is not the `goal_color`, add a cost of 1 for the `change_color` action. Otherwise, the cost is 0.
            iv.  **Painting Cost:** Add a cost of 1 for the required `paint_up` or `paint_down` action.
            v.   **Total Cost for Robot:** Sum the movement, color change, and painting costs for robot `r` to achieve this specific goal.
        c.  **Minimum Cost for Goal:** Determine the minimum total cost calculated across all robots for this goal. This represents the estimated cheapest way to satisfy this single goal predicate, ignoring interactions with other goals. If no robot can achieve this goal (all costs are infinity), the state is considered a dead end (return infinity).
    3.  **Sum Costs:** Sum the minimum costs calculated for each individual unsatisfied goal. This sum is the final heuristic estimate. This summation might overestimate the true cost because one robot might efficiently achieve multiple goals, but it provides an informative, non-admissible estimate suitable for Greedy Best-First Search.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Extract objects
        self.tiles = set()
        self.robots = set()
        self.colors = set()
        # Assuming objects are defined implicitly by their use in static facts or goals/init
        # A more robust parser would get them from (:objects) in the PDDL instance
        for fact in static_facts.union(task.initial_state).union(self.goals):
            parts = get_parts(fact)
            pred_name = parts[0]
            args = parts[1:]
            # This is a simplified way to find objects; relies on naming conventions or predicate signatures
            if pred_name in ["up", "down", "left", "right", "clear", "painted", "robot-at"]:
                for arg in args:
                    if arg.startswith("tile"): self.tiles.add(arg)
                    if arg.startswith("robot"): self.robots.add(arg)
            if pred_name in ["painted", "robot-has", "available-color", "change_color"]:
                 for arg in args:
                    # Basic check for colors - might need refinement based on actual color names
                    if not arg.startswith("tile") and not arg.startswith("robot"):
                         # Check if it's listed as an available color
                         if f"(available-color {arg})" in static_facts:
                            self.colors.add(arg)

        # Build adjacency list for tile graph (undirected)
        self.adj = {tile: set() for tile in self.tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                if t1 in self.tiles and t2 in self.tiles:
                    self.adj[t1].add(t2)
                    self.adj[t2].add(t1)

        # Precompute all-pairs shortest paths using BFS
        self.dist_map = self._compute_distances()

        # Precompute possible painting positions for each tile
        # paint_positions[goal_tile] = {set of tiles robot must be AT to paint goal_tile}
        self.paint_positions = {tile: set() for tile in self.tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            # If (up goal_tile base_tile), robot at base_tile can paint_up goal_tile
            if pred == "up":
                goal_tile, base_tile = parts[1], parts[2]
                if goal_tile in self.tiles and base_tile in self.tiles:
                    self.paint_positions[goal_tile].add(base_tile)
            # If (down goal_tile base_tile), robot at base_tile can paint_down goal_tile
            elif pred == "down":
                goal_tile, base_tile = parts[1], parts[2]
                if goal_tile in self.tiles and base_tile in self.tiles:
                    self.paint_positions[goal_tile].add(base_tile)

        # Store goal predicates for easy lookup
        self.goal_predicates = {goal for goal in self.goals if match(goal, "painted", "*", "*")}


    def _compute_distances(self):
        """Computes all-pairs shortest paths using BFS."""
        dist_map = {t1: {t2: float('inf') for t2 in self.tiles} for t1 in self.tiles}
        for start_node in self.tiles:
            if start_node not in self.adj: continue # Skip disconnected tiles if any
            dist_map[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node} # Keep track per BFS run

            while queue:
                current_tile = queue.popleft()
                current_dist = dist_map[start_node][current_tile]

                for neighbor in self.adj.get(current_tile, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist_map[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
        return dist_map

    def __call__(self, node):
        """Estimate the cost to reach a goal state from the given state node."""
        state = node.state

        # --- Parse current state ---
        robot_at = {}
        robot_has = {}
        painted_facts = set()

        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, r, t = get_parts(fact)
                robot_at[r] = t
            elif match(fact, "robot-has", "*", "*"):
                _, r, c = get_parts(fact)
                robot_has[r] = c
            elif match(fact, "painted", "*", "*"):
                painted_facts.add(fact)

        # --- Identify unsatisfied goals ---
        unsatisfied_goals = self.goal_predicates - painted_facts

        # If already in a goal state, cost is 0
        if not unsatisfied_goals:
            return 0

        total_heuristic_cost = 0

        # --- Calculate cost for each unsatisfied goal ---
        for goal_fact in unsatisfied_goals:
            _, goal_tile, goal_color = get_parts(goal_fact)

            # Find required painting positions for this goal tile
            possible_paint_positions = self.paint_positions.get(goal_tile)

            # If a tile needs painting but there's no adjacent tile from which to paint it
            if not possible_paint_positions:
                return float('inf') # Goal is impossible

            min_cost_for_this_goal = float('inf')

            # Calculate cost for each robot to achieve this goal
            for robot in self.robots:
                current_tile = robot_at.get(robot)
                current_color = robot_has.get(robot)

                # If robot location or color is unknown (should not happen in valid states)
                if current_tile is None or current_color is None:
                    continue # Skip this robot

                # Calculate minimum movement cost for this robot to reach a painting position
                move_cost = float('inf')
                for paint_pos in possible_paint_positions:
                    # Check reachability using precomputed distances
                    dist = self.dist_map.get(current_tile, {}).get(paint_pos, float('inf'))
                    move_cost = min(move_cost, dist)

                # If no painting position is reachable by this robot
                if move_cost == float('inf'):
                    continue # This robot cannot achieve this goal

                # Calculate color change cost
                color_cost = 0 if current_color == goal_color else 1

                # Calculate painting cost
                paint_action_cost = 1

                # Total cost for this robot to achieve this goal
                cost_r = move_cost + color_cost + paint_action_cost
                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)

            # If no robot can achieve this goal
            if min_cost_for_this_goal == float('inf'):
                return float('inf') # State is a dead end

            # Add the minimum cost for this goal to the total heuristic value
            total_heuristic_cost += min_cost_for_this_goal

        # Return the sum of minimum costs for all unsatisfied goals
        # Handle the case where there are no robots defined
        if not self.robots and unsatisfied_goals:
             return float('inf')

        return total_heuristic_cost

