import math
from collections import deque
import itertools # Not strictly required by final code, but useful for planning concepts

# Try to import the base class, provide a dummy if not found (e.g., for standalone testing)
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails
    class Heuristic:
        def __init__(self, task):
            """Dummy initializer."""
            pass
        def __call__(self, node):
            """Dummy call method."""
            raise NotImplementedError("Heuristic base class not found.")

def get_parts(fact: str):
    """
    Extracts predicate and arguments from a PDDL fact string like '(predicate arg1 arg2)'.

    Args:
        fact: The PDDL fact string.

    Returns:
        A list containing the predicate name followed by its arguments,
        or an empty list if the fact is malformed.
    """
    # Remove leading/trailing whitespace, then check for parentheses
    cleaned_fact = fact.strip()
    if not cleaned_fact.startswith("(") or not cleaned_fact.endswith(")"):
        return [] # Malformed fact
    # Remove parentheses and split by whitespace
    return cleaned_fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Floortile domain.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to reach the
    goal state from a given state. It works by summing the estimated minimum costs
    required to achieve each individual unsatisfied goal predicate of the form
    `(painted ?tile ?color)`. The cost for satisfying a single goal predicate
    is estimated by finding the minimum cost for *any* available robot to perform
    the task. This minimum cost considers the robot's movement to an adjacent tile,
    a potential color change action, and the final paint action.

    # Assumptions
    - The grid of tiles where robots move is potentially disconnected, but the
      heuristic correctly handles reachability using precomputed distances (assigning
      infinite cost if a required location is unreachable).
    - All actions defined in the domain (`move_*`, `paint_*`, `change_color`)
      have a uniform cost of 1.
    - The heuristic uses precomputed all-pairs shortest path distances based on the
      *static* grid structure (`up`, `down`, `left`, `right` facts). It deliberately
      ignores the dynamic `(clear ?tile)` status for pathfinding to ensure efficient
      computation. This means the movement cost is a lower bound on the actual moves
      needed, making the heuristic non-admissible but fast.
    - It assumes that if a robot needs a specific color to paint, it can always
      obtain it using the `change_color` action (cost 1), provided the color is
      available in the domain (which is implicitly assumed by the action's definition).
    - The heuristic does not model resource contention (e.g., multiple robots needing
      to occupy the same tile or interference from painting actions making tiles
      non-clear). It calculates the cost for each goal independently and sums them up.
      This summation of minimum individual costs typically overestimates the true
      optimal plan cost, which is acceptable for a non-admissible heuristic guiding
      a greedy search.

    # Heuristic Initialization (`__init__`)
    - The constructor receives the planning task object.
    - It parses the static facts (`task.static`) to:
        - Identify all unique tile names (`self.tiles`).
        - Build an adjacency list representation of the tile grid (`self.adjacencies`),
          where `self.adjacencies[tile]` gives a set of neighboring tiles.
        - Build an auxiliary graph structure suitable for BFS.
    - It runs Breadth-First Search (BFS) starting from each tile to compute the
      all-pairs shortest path distances on the static grid. These distances are
      stored in `self.distances`, where `self.distances[t1][t2]` is the minimum
      number of move actions to get from tile `t1` to `t2`. Infinite distance
      (`float('inf')`) indicates unreachability.
    - It parses the goal facts (`task.goals`) to extract the set of target
      `(painted ?tile ?color)` predicates, storing them as tuples `(tile, color)`
      in `self.goal_preds`.
    - It ensures the distance matrix covers all tiles mentioned in static facts or goals,
      correctly handling potentially isolated tiles.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse Current State:** Extract current robot positions (`robot_positions`),
        the color each robot holds (`robot_colors`), and the set of already
        painted tiles (`painted_tiles`) from the input state `node.state`. Also,
        identify all unique robot names (`robots`).
    2.  **Identify Unsatisfied Goals:** Calculate the set difference between the
        required goal predicates (`self.goal_preds`) and the currently satisfied
        ones (`painted_tiles`). This gives `unmet_goals`.
    3.  **Handle Goal State:** If `unmet_goals` is empty, the current state is a goal
        state, return 0.
    4.  **Handle No Robots:** If `unmet_goals` is not empty but there are no `robots`
        in the state, the goals are unreachable, return `float('inf')`.
    5.  **Iterate Through Unsatisfied Goals:** For each `(goal_tile, goal_color)` in `unmet_goals`:
        a.  Initialize `min_cost_for_this_goal` to `float('inf')`.
        b.  **Check Goal Tile Validity:** Ensure `goal_tile` is a known tile present in
            the precomputed `self.distances`. If not, the goal is impossible; set
            `min_cost_for_this_goal` to `inf`.
        c.  **Find Adjacent Tiles:** Get the set of tiles `adjacent_tiles_for_painting`
            that are adjacent to `goal_tile` using `self.adjacencies`. A robot must
            be positioned on one of these tiles to paint `goal_tile`.
        d.  **Check Reachability:** If `goal_tile` exists but has no adjacent tiles
            (e.g., an isolated tile in the grid definition), painting is impossible;
            set `min_cost_for_this_goal` to `inf`.
        e.  **Iterate Through Robots:** For each robot `r`:
            i.  Get robot's current position `robot_pos` and color `robot_col`. Skip if
                information is missing for this robot.
            ii. **Check Robot Position Validity:** If `robot_pos` is not in `self.distances`
                (e.g., robot is somehow off-grid), this robot cannot contribute; its cost
                is `inf`.
            iii. **Calculate Color Change Cost:** `color_change_cost` is 1.0 if `robot_col`
                 is different from `goal_color`, otherwise 0.0.
            iv. **Calculate Minimum Movement Cost:** Initialize `min_robot_move_cost` to `inf`.
                 Iterate through each `adj_tile` in `adjacent_tiles_for_painting`. Find the
                 precomputed distance `dist = self.distances[robot_pos][adj_tile]`. Update
                 `min_robot_move_cost = min(min_robot_move_cost, dist)` if `dist` is finite.
            v.  **Calculate Total Robot Cost:** If `min_robot_move_cost` remains `inf` (meaning
                 the robot cannot reach *any* required adjacent tile), the `robot_total_cost`
                 is `inf`. Otherwise, `robot_total_cost = min_robot_move_cost + color_change_cost + 1.0`
                 (where 1.0 is the cost of the `paint` action).
            vi. **Update Minimum Goal Cost:** Update `min_cost_for_this_goal = min(min_cost_for_this_goal, robot_total_cost)`.
        f.  **Accumulate Heuristic Value:** Add `min_cost_for_this_goal` to the total `heuristic_value`.
            If `min_cost_for_this_goal` is `inf` at this point, it means *no* robot can achieve
            this specific goal. The overall goal is thus unreachable from this state. Set
            `heuristic_value` to `inf` and break the loop over goals.
    6.  **Return Value:** If `heuristic_value` is `inf`, return `float('inf')`. Otherwise,
        return `math.ceil(heuristic_value)` to provide a non-negative integer estimate,
        rounding up the sum of costs.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information and goals.
        Precomputes tile adjacencies and all-pairs shortest paths.
        """
        super().__init__(task) # Call base class constructor if needed
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.tiles = set()
        self.adjacencies = {} # tile -> {neighbor1, neighbor2, ...}
        self.goal_preds = set() # {(tile, color), ...}

        # --- Parse Static Facts and Build Graph ---
        tile_graph = {} # For BFS: tile -> {neighbor1, ...}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]
            args = parts[1:]

            # Assumes predicates up, down, left, right define symmetric adjacency
            if predicate in ["up", "down", "left", "right"] and len(args) == 2:
                t1, t2 = args
                self.tiles.add(t1)
                self.tiles.add(t2)
                # Add edges for graph used in BFS
                tile_graph.setdefault(t1, set()).add(t2)
                tile_graph.setdefault(t2, set()).add(t1)
                # Store adjacencies for later lookup
                self.adjacencies.setdefault(t1, set()).add(t2)
                self.adjacencies.setdefault(t2, set()).add(t1)

        # --- Parse Goals ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                # Add the tile from the goal to the set of all known tiles
                self.tiles.add(tile)
                self.goal_preds.add((tile, color))

        # --- Precompute Distances ---
        # Compute distances based on the graph derived from connectivity predicates
        self.distances = self._compute_all_pairs_shortest_paths(tile_graph)
        # Ensure the distance matrix covers all tiles, including isolated ones
        self._ensure_distances_for_all_tiles()


    def _compute_all_pairs_shortest_paths(self, graph):
        """Computes shortest path distances between all pairs of known tiles using BFS."""
        distances = {}
        # Use all tiles known from static facts and goals for initialization
        all_known_tiles = list(self.tiles)

        # Initialize distance matrix structure for all known tiles
        for t1 in all_known_tiles:
            distances[t1] = {t2: float('inf') for t2 in all_known_tiles}
            # Distance from a tile to itself is 0
            if t1 in distances: # Check required as t1 might be isolated and not a key yet if graph empty
                 distances[t1][t1] = 0

        # Run BFS starting from each tile that has connections in the graph
        for start_node in graph.keys():
            # Ensure start_node is in the distances structure (should be if self.tiles is correct)
            if start_node not in distances: continue

            # Queue for BFS, starting with the source node
            queue = deque([start_node])
            # Keep track of nodes visited *during this specific BFS run* to avoid cycles
            # and redundant processing within this run.
            processed_in_bfs = {start_node}

            while queue:
                current_node = queue.popleft()
                # If current_node somehow isn't in the matrix, skip (shouldn't happen)
                if current_node not in distances[start_node]: continue

                current_dist = distances[start_node][current_node]

                # Explore neighbors
                for neighbor in graph.get(current_node, set()):
                    # Check if neighbor is a known tile and if we found the first path to it
                    if neighbor in distances[start_node] and distances[start_node][neighbor] == float('inf'):
                        distances[start_node][neighbor] = current_dist + 1
                        # Add neighbor to queue only if it hasn't been processed in this BFS run
                        if neighbor not in processed_in_bfs:
                             queue.append(neighbor)
                             processed_in_bfs.add(neighbor)
        return distances

    def _ensure_distances_for_all_tiles(self):
        """Ensures the distance matrix includes entries for all known tiles,
           handling tiles that might be isolated (not present in the graph keys)."""
        all_tiles_list = list(self.tiles)
        for t1 in all_tiles_list:
            if t1 not in self.distances:
                # If t1 wasn't a key in the graph, it's isolated. Add its entry.
                self.distances[t1] = {t2: float('inf') for t2 in all_tiles_list}
                # Distance to self is 0, ensure this entry exists
                if t1 in self.distances: # Should be true now
                    self.distances[t1][t1] = 0
            else:
                # If t1 was in the graph, ensure its distance dict contains all other tiles
                for t2 in all_tiles_list:
                    if t2 not in self.distances[t1]:
                        # If t2 wasn't reached from t1, distance is inf
                        self.distances[t1][t2] = float('inf')


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns the estimated cost as ceil(value) or float('inf').
        """
        state = node.state
        heuristic_value = 0.0 # Use float for calculations involving infinity

        # --- Parse Current State ---
        robot_positions = {} # r -> t
        robot_colors = {} # r -> c
        painted_tiles = set() # {(t, c), ...}
        robots = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == "robot-at" and len(args) == 2:
                robot, tile = args
                robot_positions[robot] = tile
                robots.add(robot)
            elif predicate == "robot-has" and len(args) == 2:
                robot, color = args
                robot_colors[robot] = color
                robots.add(robot) # Also capture robots from here
            elif predicate == "painted" and len(args) == 2:
                tile, color = args
                painted_tiles.add((tile, color))

        # --- Calculate Heuristic ---
        unmet_goals = self.goal_preds - painted_tiles

        # If no unmet goals, we are at the goal state.
        if not unmet_goals:
            return 0

        # If there are goals but no robots, goals are unreachable.
        if not robots:
             return float('inf')

        # Calculate cost for each unmet goal
        for goal_tile, goal_color in unmet_goals:
            min_cost_for_goal = float('inf')

            # Check if the goal tile itself is valid and known
            if goal_tile not in self.distances:
                 # Goal involves a tile completely unknown or undefined in connectivity.
                 min_cost_for_goal = float('inf')
            else:
                # Find tiles adjacent to the goal tile based on static connections
                adjacent_tiles_for_painting = self.adjacencies.get(goal_tile, set())

                # If the goal tile exists but has no adjacent tiles defined, painting is impossible.
                # Check self.adjacencies which directly reflects static facts.
                # A tile in self.tiles might not be in self.adjacencies if it had no connections.
                if goal_tile in self.tiles and goal_tile not in self.adjacencies:
                     min_cost_for_goal = float('inf') # Isolated tile

                # If adjacency entry exists but is empty (shouldn't happen if built correctly), also impossible.
                elif not adjacent_tiles_for_painting and goal_tile in self.adjacencies:
                     min_cost_for_goal = float('inf')

                else:
                    # Find the minimum cost among all robots to achieve this goal
                    for r in robots:
                        # Ensure robot has position and color information from the current state
                        if r not in robot_positions or r not in robot_colors:
                            continue # Skip robot if state info is incomplete for it

                        robot_pos = robot_positions[r]
                        robot_col = robot_colors[r]

                        # Check if the robot's current position is valid
                        if robot_pos not in self.distances:
                             # Robot is on an unknown tile, cannot contribute.
                             robot_total_cost = float('inf')
                        else:
                            # 1. Cost for potential color change
                            color_change_cost = 0.0 if robot_col == goal_color else 1.0

                            # 2. Minimum movement cost to reach a tile adjacent to the goal tile
                            min_robot_move_cost = float('inf')
                            for adj_tile in adjacent_tiles_for_painting:
                                # Ensure the adjacent tile is valid before checking distance
                                if adj_tile in self.distances[robot_pos]:
                                    dist = self.distances[robot_pos][adj_tile]
                                    # Update minimum if distance is finite
                                    if dist != float('inf'):
                                         min_robot_move_cost = min(min_robot_move_cost, dist)

                            # 3. Calculate total cost for this robot
                            if min_robot_move_cost == float('inf'):
                                # This robot cannot reach any required adjacent tile
                                robot_total_cost = float('inf')
                            else:
                                # Cost = Movement + Color Change + Paint Action
                                paint_cost = 1.0
                                robot_total_cost = min_robot_move_cost + color_change_cost + paint_cost

                        # Update the minimum cost found so far for achieving this goal
                        min_cost_for_goal = min(min_cost_for_goal, robot_total_cost)

            # Add the minimum cost for this goal to the total heuristic value
            # If any goal is found to be impossible (cost is inf), the total becomes inf
            if min_cost_for_goal == float('inf'):
                 heuristic_value = float('inf')
                 # If one goal is impossible, no need to check others
                 break
            else:
                 # Ensure we don't add to infinity
                 if heuristic_value != float('inf'):
                     heuristic_value += min_cost_for_goal

        # --- Return Final Value ---
        if heuristic_value == float('inf'):
            # Return infinity representation used by the planner
            return float('inf')
        else:
            # Return the ceiling of the sum as a non-negative integer estimate
            # Using ceil because the heuristic is non-admissible anyway
            return max(0, math.ceil(heuristic_value))

