import collections
from fnmatch import fnmatch
import math # For infinity

# Try to import the base class, provide a dummy if not found
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails
    # This allows the code to be syntactically correct,
    # assuming the actual execution environment provides it.
    class Heuristic:
        def __init__(self, task):
            self.task = task

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extract the components of a PDDL fact string (e.g., "(pred obj1 obj2)").
    Removes parentheses and splits by space. Returns a list of strings.
    Returns an empty list if the fact format is unexpected.
    """
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'floortile'.

    # Summary
    This heuristic estimates the remaining cost to achieve all painting goals.
    It calculates the cost for each unsatisfied painting goal `(painted tile_g color_g)`
    individually and sums these costs. The cost for a single goal is estimated as the
    minimum cost for any available robot to achieve it. This minimum cost includes:
    1. The shortest path distance for the robot to move from its current location to a
       tile adjacent to `tile_g` from which `tile_g` can be painted (using `paint_up`
       or `paint_down` actions).
    2. The cost of changing color (action `change_color`, cost 1) if the robot does
       not currently hold the required `color_g`.
    3. The cost of the paint action itself (e.g., `paint_up`, cost 1).

    # Assumptions
    - The heuristic calculates shortest path distances based on the static tile
      connectivity (`up`, `down`, `left`, `right` facts), assuming movement is
      bidirectional between connected tiles. It ignores dynamic constraints like
      the `clear` status of tiles or blockage by other robots (relaxation).
    - The cost of changing color is assumed to be 1 if the robot needs a different
      color than it currently holds. It assumes the target color is always available
      via the `change_color` action (precondition `available-color ?c2`).
    - The heuristic sums the estimated costs for each unsatisfied goal independently.
      It does not account for potential positive or negative interactions between
      achieving different goals (e.g., one robot efficiently painting multiple nearby
      tiles, or robots needing to move out of the way for each other).
    - Painting is only possible via `paint_up` and `paint_down` actions, requiring the
      robot to be at a specific adjacent tile defined by the `up` and `down` static
      predicates.

    # Heuristic Initialization
    - The constructor (`__init__`) pre-processes information from the task definition.
    - It stores the set of goal predicates of the form `(painted ?tile ?color)`.
    - It extracts all unique tile, robot, and color objects by parsing static facts,
      initial state facts, and goal facts.
    - It builds an adjacency list (`self.adj`) representing the tile grid connectivity
      based on `up`, `down`, `left`, `right` static facts, used for pathfinding.
    - It computes All-Pairs Shortest Paths (APSP) between all tiles using Breadth-First
      Search (BFS) on the static grid graph and stores the distances in `self.distances`.
      This allows for efficient lookup of movement costs later.
    - It identifies and stores the valid "painting positions" for each tile in
      `self.paint_adj`. For a tile `t` to be painted, this dictionary stores the set
      of adjacent tiles `adj_t` where a robot must be located, based on `(up t adj_t)`
      or `(down t adj_t)` static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Get Current State:** Obtain the current set of facts (`state`) from the search node.
    2.  **Extract State Properties:** Parse the `state` to find the current location (`robot_locations`) and held color (`robot_colors`) for each robot, and the set of already `painted_tiles`.
    3.  **Identify Unsatisfied Goals:** Determine the set of `(painted tile color)` goals that are present in `self.goal_predicates` but not in the current `painted_tiles`.
    4.  **Handle Goal State:** If there are no unsatisfied goals, the state is a goal state, return 0.
    5.  **Handle No Robots Case:** If the problem defines no robots, and there are unsatisfied goals, return infinity as the goals cannot be achieved.
    6.  **Initialize Heuristic Value:** Set `total_heuristic_value = 0`.
    7.  **Iterate Through Unsatisfied Goals:** For each unsatisfied goal `G = (painted tile_g color_g)`:
        a.  **Find Painting Positions:** Look up the precomputed set `PaintPos(tile_g)` from `self.paint_adj`. If `PaintPos(tile_g)` is empty (meaning static facts do not define a way to paint `tile_g`), the goal is considered structurally unreachable; return infinity.
        b.  **Find Minimum Robot Cost for Goal G:** Initialize `min_cost_for_G = infinity`.
        c.  **Iterate Through Robots:** For each robot `r`:
            i.  **Get Robot State:** Retrieve `loc_r` and `color_r`. Skip if the robot's state is missing.
            ii. **Calculate Movement Cost:** Find the minimum shortest path distance from `loc_r` to any tile `p` in `PaintPos(tile_g)` using the precomputed `self.distances`. `move_dist = min(self.distances[loc_r].get(p, infinity) for p in PaintPos(tile_g))`. If no path exists from `loc_r` to any `p`, `move_dist` will be infinity.
            iii.**Skip if Unreachable:** If `move_dist` is infinity, this robot cannot reach a position to paint this tile; continue to the next robot.
            iv. **Calculate Color Change Cost:** `color_cost = 1` if `color_r != color_g`, else `0`.
            v.  **Calculate Paint Action Cost:** `paint_action_cost = 1`.
            vi. **Calculate Total Robot Cost:** `robot_cost = move_dist + color_cost + paint_action_cost`.
            vii.**Update Minimum Cost for Goal:** `min_cost_for_G = min(min_cost_for_G, robot_cost)`.
        d.  **Check Goal Reachability:** If `min_cost_for_G` remains infinity after checking all robots, it means no robot can currently achieve this goal (due to reachability constraints); return infinity.
        e.  **Add Goal Cost to Total:** Add the calculated `min_cost_for_G` to `total_heuristic_value`.
    8.  **Return Heuristic Value:** Return the final `total_heuristic_value`.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class
        self.goals = task.goals
        static_facts = task.static
        init_facts = task.initial_state

        # Extract objects (tiles, robots, colors)
        self.tiles = set()
        self.robots = set()
        self.colors = set()
        all_facts = static_facts.union(init_facts).union(self.goals)
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip if parsing failed
             pred_name = parts[0]
             try:
                 # Identify objects based on predicates they appear in
                 if pred_name in ["up", "down", "left", "right"]:
                     self.tiles.add(parts[1])
                     self.tiles.add(parts[2])
                 elif pred_name == "clear":
                     self.tiles.add(parts[1])
                 elif pred_name == "painted":
                     self.tiles.add(parts[1])
                     self.colors.add(parts[2])
                 elif pred_name == "robot-at":
                     self.robots.add(parts[1])
                     self.tiles.add(parts[2])
                 elif pred_name == "robot-has":
                     self.robots.add(parts[1])
                     self.colors.add(parts[2])
                 elif pred_name == "available-color":
                     self.colors.add(parts[1])
             except IndexError:
                 # Ignore facts with unexpected number of arguments for known predicates
                 pass

        # Build adjacency list for movement (undirected graph)
        self.adj = collections.defaultdict(set)
        # Build adjacency list for painting: tile_to_paint -> {tile_to_be_at}
        self.paint_adj = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            try:
                if pred in ["up", "down", "left", "right"]:
                    t1, t2 = parts[1], parts[2]
                    # Add edges for movement graph (assume symmetric)
                    self.adj[t1].add(t2)
                    self.adj[t2].add(t1)
                    # Add edges for painting relationship based on domain actions
                    # If (up t1 t2), robot at t2 can paint_up t1
                    if pred == "up":
                        self.paint_adj[t1].add(t2)
                    # If (down t1 t2), robot at t2 can paint_down t1
                    elif pred == "down":
                        self.paint_adj[t1].add(t2)
            except IndexError:
                 pass # Ignore static facts with unexpected structure

        # Compute All-Pairs Shortest Paths (APSP) using BFS
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))

        for start_node in self.tiles:
            # Check if start_node is actually part of the connected graph components
            # Nodes might be listed as objects but not appear in connectivity facts
            if start_node not in self.adj and not any(start_node in neighbors for neighbors in self.adj.values()):
                 self.distances[start_node][start_node] = 0
                 continue # Isolated node

            self.distances[start_node][start_node] = 0
            queue = collections.deque([start_node])
            # Use visited dict to store distances to prevent cycles and redundant work in BFS
            visited_dist = {start_node: 0}

            while queue:
                current_node = queue.popleft()
                current_dist = visited_dist[current_node]

                # Iterate through neighbors from the adjacency list
                for neighbor in self.adj.get(current_node, set()):
                    if neighbor not in visited_dist:
                        visited_dist[neighbor] = current_dist + 1
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

        # Store goal predicates `(painted tile color)` for quick lookup
        self.goal_predicates = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             # Ensure the goal is a 'painted' predicate with correct arity
             if parts[0] == "painted" and len(parts) == 3:
                 self.goal_predicates.add(goal)


    def __call__(self, node):
        """
        Calculates the heuristic value for a given search node.
        """
        state = node.state

        # Find current state properties: robot locations, robot colors, painted tiles
        robot_locations = {}
        robot_colors = {}
        painted_tiles = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            try:
                if pred == "robot-at" and len(parts) == 3:
                    robot_locations[parts[1]] = parts[2] # robot -> tile
                elif pred == "robot-has" and len(parts) == 3:
                    robot_colors[parts[1]] = parts[2] # robot -> color
                elif pred == "painted" and len(parts) == 3:
                    painted_tiles.add(fact) # Store the full fact "(painted tile color)"
            except IndexError:
                # Ignore malformed facts in the current state
                pass

        # Identify unsatisfied goals by comparing with precomputed goal set
        unsatisfied_goals = self.goal_predicates - painted_tiles

        # If all goals are satisfied, heuristic estimate is 0
        if not unsatisfied_goals:
            return 0

        # If there are no robots defined, remaining goals are unreachable
        if not self.robots:
             return float('inf')

        total_heuristic_value = 0

        # Calculate cost for each unsatisfied goal
        for goal_fact in unsatisfied_goals:
            parts = get_parts(goal_fact)
            # Goal structure was validated in __init__
            tile_g, color_g = parts[1], parts[2]

            # Find the set of positions from where tile_g can be painted
            possible_paint_positions = self.paint_adj.get(tile_g, set())

            # If static facts define no way to paint this goal tile, it's unreachable
            if not possible_paint_positions:
                 return float('inf')

            min_cost_for_goal = float('inf')

            # Find the minimum cost among all robots to achieve this goal
            for robot in self.robots:
                # Ensure robot's current location and color are known from the state
                if robot not in robot_locations or robot not in robot_colors:
                     continue # Skip robots with incomplete state information

                loc_r = robot_locations[robot]
                color_r = robot_colors[robot]

                # Calculate minimum movement distance to a valid painting position
                current_min_move_dist = float('inf')
                for paint_pos in possible_paint_positions:
                    # Look up pre-calculated distance, default to infinity if no path exists
                    dist = self.distances[loc_r].get(paint_pos, float('inf'))
                    current_min_move_dist = min(current_min_move_dist, dist)

                # If this robot cannot reach any required painting position, skip it
                if current_min_move_dist == float('inf'):
                    continue

                # Calculate color change cost (1 if colors differ, 0 otherwise)
                color_cost = 1 if color_r != color_g else 0

                # Calculate paint action cost (always 1 for the paint action itself)
                paint_action_cost = 1

                # Total estimated cost for this robot for this specific goal
                robot_cost = current_min_move_dist + color_cost + paint_action_cost
                # Update the minimum cost found so far for this goal
                min_cost_for_goal = min(min_cost_for_goal, robot_cost)

            # If no robot could possibly achieve this goal (e.g., all unreachable)
            if min_cost_for_goal == float('inf'):
                 # This implies the goal is currently unreachable by any robot
                 return float('inf')

            # Add the minimum cost for achieving this goal to the total heuristic value
            total_heuristic_value += min_cost_for_goal

        # Return the total estimated cost for all unsatisfied goals
        return total_heuristic_value
