import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists in the environment

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact (string) by removing parentheses
    and splitting by space."""
    # Ensure fact is a string and has parentheses before stripping
    if isinstance(fact, str) and len(fact) > 1 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Return empty list for invalid format

def match(fact, *args):
    """
    Check if a PDDL fact (string) matches a given pattern.
    Uses fnmatch for wildcard support in args.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the pattern length
    if len(parts) != len(args):
        return False
    # Check if all parts match the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'floortile'.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to paint all
    required tiles according to the goal state. It calculates the cost for each
    unpainted goal tile by finding the minimum cost for *any* available robot to
    reach an adjacent tile, potentially change to the required color, and perform
    the paint action. The total heuristic value is the sum of these minimum costs
    over all unsatisfied 'painted' goals. It aims for informativeness to guide
    greedy best-first search, not admissibility.

    # Assumptions
    - All actions (move_*, paint_*, change_color) have a uniform cost of 1.
    - The heuristic ignores the `clear` predicate constraints for movement and painting.
      This simplifies the calculation and makes it faster, but potentially less accurate
      and non-admissible, as `clear` preconditions might block actions in reality.
    - It assumes any robot can change to any available color with a cost of 1 if
      it doesn't already hold that color. The `change_color` action requires the
      robot to hold *some* color initially.
    - The cost for each goal tile is calculated independently by finding the cheapest
      robot to achieve it. This doesn't explicitly model complex interactions or
      resource contention (e.g., multiple robots needing the same tile path or color
      simultaneously) but provides a reasonable estimate suitable for greedy search guidance.

    # Heuristic Initialization
    - Stores the set of goal `(painted ?tile ?color)` conditions.
    - Parses static facts (`up`, `down`, `left`, `right`) to build an adjacency graph
      representing possible robot movements between tiles.
    - Computes all-pairs shortest paths (APSP) using Breadth-First Search (BFS) on the
      tile graph. The results (minimum movement costs between any two tiles) are stored
      in a nested dictionary `self.distances`.
    - Identifies all robot objects defined in the task by checking initial state predicates
      like `robot-at` and `robot-has`.
    - Precomputes the set of adjacent tiles (`paint_adj`) from which each target tile
      can be painted. This is based on `up` and `down` static facts, as the domain
      only defines `paint_up` and `paint_down` actions, requiring the robot to be
      at the tile below or above the target tile, respectively.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unsatisfied Goals:** Determine the set of `(painted ?tile ?color)`
        facts specified in the goal that are *not* present in the current state's facts.
    2.  **Check for Goal State:** If the set of unsatisfied goals is empty, the
        current state satisfies the painting goals, and the heuristic value is 0.
    3.  **Parse Current State:** Extract the current location `loc(r)` for each robot `r`
        from `(robot-at ?r ?loc)` facts, and the currently held color `col(r)` from
        `(robot-has ?r ?col)` facts found in the current state `node.state`.
    4.  **Calculate Cost per Unsatisfied Goal:** For each unsatisfied goal `(painted t c)`:
        a.  **Find Painting Locations:** Identify the set of tiles `Adj(t)` from which
            tile `t` can be painted using the precomputed `paint_adj[t]` mapping. These are
            the tiles a robot must be *at* to execute `paint_up` or `paint_down` on `t`.
        b.  **Calculate Minimum Robot Cost:** Iterate through all identified robots `r`:
            i.   Retrieve the robot's current location `loc(r)` and color `col(r)` from the parsed state. If information is missing for a known robot, skip it (should not happen in valid states).
            ii.  **Color Cost:** `cost_color = 0` if `col(r) == c`, otherwise `cost_color = 1`
                 (representing the cost of one `change_color` action).
            iii. **Movement Cost:** Find the minimum path cost from the robot's current location `loc(r)` to any tile `adj` in the set `Adj(t)`. This is calculated as `min_move_cost = min(self.distances[loc(r)][adj])` over all `adj` in `Adj(t)`, using the precomputed APSP distances. If `loc(r)` is already one of the `adj` tiles, the distance to that specific tile is 0. If any required `adj` tile is unreachable from `loc(r)`, its distance is treated as infinity.
            iv.  **Painting Cost:** The cost of the `paint_up` or `paint_down` action itself is 1.
            v.   **Total Robot Cost:** The total estimated cost for robot `r` to achieve this goal is `robot_cost = cost_color + min_move_cost + 1` (paint cost). If `min_move_cost` is infinity (meaning the robot cannot reach *any* required adjacent position), then `robot_cost` is infinity.
        c.  **Determine Minimum Goal Cost:** The estimated cost for achieving this specific
            goal `(painted t c)` is the minimum `robot_cost` found across all robots capable of potentially reaching a painting position. `min_goal_cost = min(robot_cost for r in robots)`.
        d.  **Handle Unreachability:** If `min_goal_cost` remains infinity after checking all robots, it implies this goal cannot be achieved by any robot from the current state (e.g., due to disconnected graph components for all robots relative to the goal). The heuristic should return infinity for the entire state in this case, signaling a dead-end.
    5.  **Sum Costs:** The total heuristic value for the state is the sum of the
        `min_goal_cost` calculated for each unsatisfied goal. The value is capped at 0 minimum (it should naturally be non-negative).
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Needed to find all objects initially

        # Extract goal painted tiles
        self.goal_painted = set()
        for fact in self.goals:
            if match(fact, "painted", "*", "*"):
                self.goal_painted.add(fact)

        # --- Object and Graph Initialization ---
        self.tiles = set()
        self.robots = set()
        self.colors = set()
        # Adjacency list for robot movement: adj[from_tile] = {to_tile1, to_tile2, ...}
        self.adj = collections.defaultdict(set)
        # Adjacency list for painting: paint_adj[target_tile] = {tile_robot_must_be_at_1, ...}
        self.paint_adj = collections.defaultdict(set)

        # Process static facts for connectivity and available colors
        for fact in self.static_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts
             pred = parts[0]

             if pred in ["up", "down", "left", "right"]:
                 # parts[1] is the destination tile (y), parts[2] is the source tile (x)
                 tile_y, tile_x = parts[1], parts[2]
                 self.tiles.add(tile_x)
                 self.tiles.add(tile_y)

                 # Based on action definitions:
                 # move_up(r, x, y) requires (up y x), moves r from x to y.
                 # move_down(r, x, y) requires (down y x), moves r from x to y.
                 # move_left(r, x, y) requires (left y x), moves r from x to y.
                 # move_right(r, x, y) requires (right y x), moves r from x to y.
                 # So, the predicate defines the target 'y' relative to the current 'x'.
                 # The adjacency list should map 'x' to 'y'.
                 self.adj[tile_x].add(tile_y)

                 # paint_up(r, y, x, c) requires (up y x), robot at x paints y.
                 # paint_down(r, y, x, c) requires (down y x), robot at x paints y.
                 # So, to paint 'y', the robot must be at 'x'.
                 if pred == "up" or pred == "down":
                     self.paint_adj[tile_y].add(tile_x)

             elif pred == "available-color":
                 self.colors.add(parts[1])

        # Process initial state facts to find robots and potentially more objects
        # This helps ensure all relevant objects are known from the start.
        for fact in self.initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts
             pred = parts[0]
             if pred == "robot-at":
                 self.robots.add(parts[1])
                 self.tiles.add(parts[2])
             elif pred == "robot-has":
                 self.robots.add(parts[1])
                 self.colors.add(parts[2])
             elif pred == "clear":
                 self.tiles.add(parts[1])
             elif pred == "painted":
                 self.tiles.add(parts[1])
                 self.colors.add(parts[2])

        # --- Compute All-Pairs Shortest Paths (APSP) using BFS ---
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))
        if not self.tiles: # Handle case with no tiles defined
            print("Warning: No tiles found in the problem definition.")
            return

        for start_node in self.tiles:
            # Ensure the start node itself is in the distances map
            if start_node not in self.distances:
                 self.distances[start_node] = collections.defaultdict(lambda: float('inf'))

            self.distances[start_node][start_node] = 0
            queue = collections.deque([(start_node, 0)])
            visited = {start_node} # Keep track per BFS run

            while queue:
                current_node, dist = queue.popleft()
                # Iterate over neighbors reachable via move actions defined in self.adj
                for neighbor in self.adj.get(current_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # Check if neighbor exists in the outer dict, though defaultdict handles this
                        if start_node not in self.distances:
                             self.distances[start_node] = collections.defaultdict(lambda: float('inf'))
                             self.distances[start_node][start_node] = 0 # Re-ensure start dist is 0
                        self.distances[start_node][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state node.
        """
        state = node.state
        infinity = float('inf')

        # Find unsatisfied goals by comparing current painted facts with goal painted facts
        current_painted = {fact for fact in state if match(fact, "painted", "*", "*")}
        unsatisfied_goals = self.goal_painted - current_painted

        # If all painting goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        # Get current robot locations and colors from the current state
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                if robot in self.robots: # Only track known robots
                    robot_locations[robot] = tile
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                if robot in self.robots: # Only track known robots
                    robot_colors[robot] = color

        total_heuristic_cost = 0

        # Calculate cost for each unsatisfied goal
        for goal_fact in unsatisfied_goals:
            parts = get_parts(goal_fact)
            if not parts or len(parts) != 3: continue # Skip malformed goal facts
            target_tile, target_color = parts[1], parts[2]

            min_cost_for_goal = infinity
            # Get the set of positions a robot needs to be at to paint the target tile
            possible_robot_positions = self.paint_adj.get(target_tile)

            # If a tile is a goal but cannot be painted (no adjacent tile allows painting action)
            if not possible_robot_positions:
                # print(f"Warning: Goal {goal_fact} seems unachievable - no adjacent tile allows painting.")
                return infinity # This goal is impossible, state is likely a dead-end

            # Find the minimum cost using any robot
            for robot in self.robots:
                # Ensure we have current location and color for this robot in the state
                if robot not in robot_locations or robot not in robot_colors:
                     # print(f"Warning: State information missing for robot {robot} in current node.")
                     continue # Skip robot if its state is unknown

                current_robot_loc = robot_locations[robot]
                current_robot_col = robot_colors[robot]

                # 1. Color Change Cost: 1 if robot needs to change color, 0 otherwise
                cost_color = 0 if current_robot_col == target_color else 1

                # 2. Minimum Movement Cost: Find the min distance to any required painting position
                min_move_cost = infinity
                # Check distance from current location to each possible painting position
                for adjacent_tile in possible_robot_positions:
                    # Fetch precomputed distance; defaults to infinity if path doesn't exist or tiles are unknown
                    dist = self.distances.get(current_robot_loc, {}).get(adjacent_tile, infinity)
                    min_move_cost = min(min_move_cost, dist)

                # If no adjacent painting position is reachable by this robot
                if min_move_cost == infinity:
                    cost_robot = infinity
                else:
                    # 3. Painting Action Cost: Always 1 for the paint action itself
                    cost_paint = 1
                    # Total cost for this robot to achieve this goal
                    cost_robot = cost_color + min_move_cost + cost_paint

                # Update the minimum cost found so far for achieving this goal (across all robots)
                min_cost_for_goal = min(min_cost_for_goal, cost_robot)

            # If after checking all robots, the goal remains unreachable
            if min_cost_for_goal == infinity:
                 # print(f"Warning: Goal {goal_fact} is unreachable by any robot from current state.")
                 return infinity # If any goal is unreachable, the state is effectively a dead-end

            # Add the minimum cost for achieving this goal to the total heuristic value
            total_heuristic_cost += min_cost_for_goal

        # The heuristic value is the sum of minimum costs for all unsatisfied goals
        # Ensure the result is non-negative (it should be, but max(0,...) is safe)
        return max(0, total_heuristic_cost)

