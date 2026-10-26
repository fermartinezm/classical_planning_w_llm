import collections
import itertools
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists and is importable

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Handle potential length mismatch if pattern has different arity
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'floortile'.

    Summary:
    Estimates the cost to reach the goal state by summing the minimum estimated costs
    to achieve each individual unsatisfied 'painted' goal. The cost for a single goal
    considers the cost for a robot to potentially change color, move to a tile adjacent
    to the target tile, and perform the paint action. The minimum cost across all
    available robots is chosen for each goal. This heuristic is not admissible but aims
    to be informative for greedy best-first search.

    Assumptions:
    - The tile connectivity is defined by the static up/down/left/right predicates.
    - Robots can move between adjacent tiles if the destination tile is clear. The pathfinding
      for the heuristic uses precomputed shortest paths on the static grid, assuming tiles
      can eventually become clear.
    - Any available color can be obtained by a robot with one 'change_color' action if it holds another color.
    - The primary goals are of the form (painted tile color). Other goal types are ignored by this heuristic.
    - If a tile is painted with the wrong color in the current state, the goal is considered unreachable
      from that state, as there is no unpaint action.

    Heuristic Initialization:
    - Parses static facts (`up`, `down`, `left`, `right`) to build an adjacency map of the tiles.
    - Identifies the set of all tiles and available colors from static facts.
    - Determines, for each tile `y` that can be painted, the set of adjacent tiles `x` a robot must be at to paint `y`.
    - Precomputes all-pairs shortest paths (APSP) between all tiles using Breadth-First Search (BFS)
      starting from each tile. Stores distances in a dictionary `self.dist`.
    - Stores the goal predicates, filtering for `(painted ...)` goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic cost `h = 0`.
    2. Parse the current state `node.state` to find:
       - Position of each robot `(robot-at r t)`.
       - Color held by each robot `(robot-has r c)`.
       - Currently painted tiles `(painted t c)`.
    3. Identify the set of robots available in the current state. If no robots exist and goals are present, return infinity.
    4. Iterate through each goal predicate `g = (painted target_tile target_color)` stored during initialization.
    5. Check if goal `g` is already satisfied in the current state (i.e., `target_tile` is painted with `target_color`). If yes, continue to the next goal.
    6. Check if `target_tile` is painted but with a *different* color. If yes, this state cannot lead to the goal. Return infinity.
    7. If goal `g` is not satisfied and the tile is either clear or unpainted:
       a. Find the set `X_set` of all tiles `x` from which `target_tile` can be painted (these are the adjacent tiles stored during initialization).
       b. If `X_set` is empty (e.g., isolated tile), the goal is unreachable. Return infinity.
       c. Initialize `min_cost_for_goal = infinity`.
       d. For each robot `r` present in the state:
          i.   Get the robot's current position `pos_r` and current color `col_r`.
          ii.  Calculate the color change cost `cost_c`:
               - 0 if `col_r == target_color`.
               - 1 if `col_r != target_color` and `target_color` is an available color (checked during init).
               - Infinity otherwise (robot cannot get the required color, or doesn't hold a color).
          iii. If `cost_c` is infinity, this robot cannot fulfill the goal; continue to the next robot.
          iv.  Calculate the minimum movement cost `cost_m` for robot `r` to reach *any* tile `x` in `X_set` from its current position `pos_r`. This uses the precomputed shortest path distances: `cost_m = min(distance(pos_r, x))` over all `x` in `X_set`. If no `x` in `X_set` is reachable from `pos_r`, `cost_m` is infinity.
          v.   If `cost_m` is infinity, this robot cannot reach a position to paint the tile; continue to the next robot.
          vi.  Calculate the total cost for robot `r` to achieve this goal: `cost_r = cost_c + cost_m + 1` (where +1 is for the `paint` action).
          vii. Update `min_cost_for_goal = min(min_cost_for_goal, cost_r)`.
       e. If `min_cost_for_goal` is still infinity after checking all robots, it means this goal is currently unreachable by any robot. Return infinity.
       f. Add `min_cost_for_goal` to the total heuristic cost `h`.
    8. After iterating through all goals, return the total cost `h`. If `h` is 0, it implies all `(painted ...)` goals are satisfied.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Heuristic Initialization ---

        # 1. Parse static facts: Build adjacency list, identify tiles, colors, and painting positions
        self.tiles = set()
        self.adj = collections.defaultdict(set) # Adjacency for movement: tile -> {neighbor1, neighbor2, ...}
        self.available_colors = set()
        # Map: tile_to_be_painted -> {set of positions robot must be at}
        self.painting_positions = collections.defaultdict(set)

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'available-color':
                self.available_colors.add(parts[1])
            elif predicate in ['up', 'down', 'right', 'left']:
                # Example: (up tile_y tile_x) means tile_y is 'up' relative to tile_x.
                # A robot at tile_x can paint tile_y using paint_up/down/etc.
                tile_y, tile_x = parts[1], parts[2]
                self.tiles.add(tile_x)
                self.tiles.add(tile_y)

                # Add adjacency for movement (symmetric)
                self.adj[tile_x].add(tile_y)
                self.adj[tile_y].add(tile_x)

                # Store the required robot position 'tile_x' to paint 'tile_y'
                self.painting_positions[tile_y].add(tile_x)

        # 2. Precompute All-Pairs Shortest Paths (APSP) using BFS from each tile
        self.dist = {}
        for start_node in self.tiles:
            # Ensure we only compute distances between tiles
            if not start_node.startswith("tile"): continue

            self.dist[(start_node, start_node)] = 0
            queue = collections.deque([(start_node, 0)])
            # Keep track of visited nodes and their distances from start_node
            visited_dist = {start_node: 0}

            while queue:
                curr_tile, d = queue.popleft()

                for neighbor in self.adj.get(curr_tile, set()):
                    # Only consider neighbors that are tiles
                    if not neighbor.startswith("tile"): continue

                    if neighbor not in visited_dist:
                        visited_dist[neighbor] = d + 1
                        self.dist[(start_node, neighbor)] = d + 1
                        queue.append((neighbor, d + 1))
                    # Optimization: If found shorter path (unlikely with BFS on unweighted graph, but safe)
                    # elif visited_dist[neighbor] > d + 1:
                    #     visited_dist[neighbor] = d + 1
                    #     self.dist[(start_node, neighbor)] = d + 1
                    #     # Re-add neighbor to queue if its distance improved? Not needed for standard BFS.


        # 3. Store parsed 'painted' goals
        self.parsed_goals = []
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'painted' and len(parts) == 3:
                 self.parsed_goals.append({'tile': parts[1], 'color': parts[2]})
             # Note: Ignores other potential goal types

    def get_dist(self, tile1, tile2):
        """Helper function to get precomputed distance between two tiles."""
        # Return infinity if tiles are identical but distance wasn't set (should be 0)
        if tile1 == tile2:
            return 0
        # Return precomputed distance, or infinity if unreachable
        return self.dist.get((tile1, tile2), float('inf'))

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state node."""
        state = node.state
        h_value = 0

        # --- Parse current state ---
        robot_positions = {} # robot -> tile
        robot_colors = {}    # robot -> color
        painted_tiles = {}   # tile -> color

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'robot-at' and len(parts) == 3:
                robot_positions[parts[1]] = parts[2]
            elif predicate == 'robot-has' and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]
            elif predicate == 'painted' and len(parts) == 3:
                painted_tiles[parts[1]] = parts[2]

        robots = list(robot_positions.keys())

        # If there are goals but no robots, goals are unreachable
        if not robots and self.parsed_goals:
             return float('inf')

        # --- Evaluate Goals ---
        num_unsatisfied_goals = 0
        for goal in self.parsed_goals:
            target_tile = goal['tile']
            target_color = goal['color']

            # 1. Check if goal is satisfied
            current_paint = painted_tiles.get(target_tile)
            if current_paint == target_color:
                continue # Goal satisfied

            num_unsatisfied_goals += 1

            # 2. Check if tile is wrongly painted (unsolvable)
            if current_paint is not None and current_paint != target_color:
                # Wrong color painted, and no way to undo it.
                return float('inf')

            # --- Goal is not satisfied, calculate min cost to achieve it ---
            min_cost_for_goal = float('inf')

            # 3. Find required robot positions to paint the target tile
            required_positions = self.painting_positions.get(target_tile)
            if not required_positions:
                 # Target tile cannot be painted (misconfigured PDDL?)
                 return float('inf') # Goal unreachable

            # 4. Calculate cost for each robot
            possible = False # Can any robot achieve this goal?
            for r in robots:
                robot_pos = robot_positions.get(r)
                current_robot_color = robot_colors.get(r)

                # Robot needs a position and a color to be useful
                if not robot_pos or not current_robot_color:
                    continue # Should not happen based on domain logic, but safe check

                # a. Color change cost
                cost_c = 0
                if current_robot_color != target_color:
                    # Check if the target color is available in the domain
                    if target_color in self.available_colors:
                        cost_c = 1 # Assumes 1 action to change color
                    else:
                        # Target color doesn't exist or isn't available
                        cost_c = float('inf')

                if cost_c == float('inf'):
                    continue # This robot cannot get the required color

                # b. Movement cost
                min_move_cost_r = float('inf')
                for req_pos in required_positions:
                    # Ensure the required position is a valid tile
                    if not req_pos.startswith("tile"): continue
                    dist = self.get_dist(robot_pos, req_pos)
                    min_move_cost_r = min(min_move_cost_r, dist)

                if min_move_cost_r == float('inf'):
                    # This robot cannot reach any position to paint the tile
                    continue # Try next robot

                # c. Total cost for this robot
                cost_r = cost_c + min_move_cost_r + 1 # +1 for paint action
                min_cost_for_goal = min(min_cost_for_goal, cost_r)
                possible = True # At least one robot could potentially do it

            # 5. Check if goal is achievable by any robot
            if not possible:
                # No robot can achieve this goal from the current state configuration
                return float('inf') # State is a dead end for this goal

            # 6. Add cost for this goal to total heuristic value
            h_value += min_cost_for_goal

        # Final check: if h_value is 0, all goals must be satisfied.
        # If h_value is 0 but num_unsatisfied_goals > 0, something is wrong.
        # This can happen if min_cost_for_goal was calculated as 0 for an unsatisfied goal,
        # which should only occur if robot is at the right place with the right color (cost=0+0+1=1).
        # So, h_value should only be 0 if num_unsatisfied_goals is 0.
        if h_value == 0 and num_unsatisfied_goals > 0:
             # This case indicates a potential issue in cost calculation,
             # maybe a goal requires 0 cost but isn't met. Let's return 1 as a minimum
             # if goals are unmet but calculated cost is 0.
             # A more robust approach might involve returning num_unsatisfied_goals
             # if h_value is less informative.
             # For now, let's assume cost calculation is correct (min cost is >= 1 for unmet goal).
             pass

        # Return the total estimated cost
        return h_value
