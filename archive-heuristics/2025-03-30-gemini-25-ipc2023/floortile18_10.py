import heapq
from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available at this path
# Make sure this import path is correct for the target environment.
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(robot-at robot1 tile_0_1)" -> ["robot-at", "robot1", "tile_0_1"]
    """
    # Remove parentheses and split by space
    if len(fact) < 2 or not fact.startswith("(") or not fact.endswith(")"):
        # Return empty list for malformed facts to avoid errors downstream
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Checks if a PDDL fact matches a pattern with optional wildcards ('*').
    Example: match("(painted tile_1_1 white)", "painted", "*", "white") -> True
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the pattern length
    if not parts or len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element (using fnmatch for wildcard support)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    Estimates the cost to paint all required tiles by summing the estimated minimum cost for each unpainted tile. The cost for a single tile considers the painting action itself (cost 1), potential color change needed by the 'best' robot (cost 1 if needed, 0 otherwise), and the movement cost for that robot to reach a tile adjacent to the target tile (minimum path length based on static connectivity).

    # Assumptions
    - Tile connectivity is defined solely by the explicit `up`, `down`, `left`, `right` predicates found in static facts. The heuristic does not infer grid structure from tile names.
    - The cost ignores potential conflicts between robots (e.g., needing the same tile clear, blocking paths). It assumes robots can operate independently to achieve their assigned sub-goals.
    - The cost ignores positive interactions (e.g., one move sequence helping multiple goals, one color change serving multiple tiles of the same color). Each goal's cost is calculated independently.
    - The cost assumes required tiles can be cleared eventually; it does not model the cost or feasibility of clearing tiles dynamically (which depends on the current state and other robots' positions).
    - The heuristic uses precomputed shortest path distances on the tile graph based on static connectivity. It ignores the dynamic `clear` status of tiles during pathfinding for computational efficiency, as is common for heuristics.

    # Heuristic Initialization
    - Stores the target goal predicates `(painted tile color)`.
    - Parses static facts (`up`, `down`, `left`, `right`) to build a symmetric adjacency list representing tile connectivity.
    - Identifies all unique tiles involved in connectivity, initial state, or goals.
    - Computes all-pairs shortest paths (APSP) between all identified tiles using Breadth-First Search (BFS) on the static graph structure. Stores distances (`float('inf')` for unreachable pairs).
    - Identifies all robots by checking `robot-at` or `robot-has` predicates in the initial state.
    - Stores a reference to the task object (`self.task`) for goal checking using `task.goal_reached()`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current state (`node.state`) is already a goal state using `self.task.goal_reached()`. If yes, return 0 immediately.
    2.  Parse the current state to find robot locations (`robot-at`), robot colors (`robot-has`), and currently painted tiles (`painted`).
    3.  Initialize the total heuristic value `h = 0`.
    4.  If no robots were identified during initialization (`self.robots` is empty), return `float('inf')` because painting goals cannot be achieved (assuming the state is not already the goal, checked in step 1).
    5.  Iterate through each required goal `G = (painted target_tile target_color)` stored in `self.goal_painted`.
    6.  If goal `G` is already satisfied in the current state (check `painted` map), skip to the next goal.
    7.  Find the set `AdjacentTiles` for `target_tile` using the precomputed `self.adj`. If `target_tile` has no adjacent tiles defined in the static facts (`AdjacentTiles` is empty), then this goal is impossible to achieve via the `paint_up`/`paint_down` actions. Return `float('inf')` as the heuristic value for this state (it's a dead end).
    8.  Calculate the minimum cost `min_cost_for_G` required to achieve this single goal `G`, considering all available robots. Initialize `min_cost_for_G = float('inf')`.
    9.  For each robot `r` in `self.robots`:
        a. Retrieve the robot's current tile `robot_tile` and color `robot_color` from the parsed state. If the robot's state is incomplete or its `robot_tile` is not in the precomputed distance map (i.e., it's on an unknown or isolated tile), skip this robot for this goal.
        b. Calculate the color change cost: `cost_color = 1` if `robot_color != target_color`, else `0`.
        c. Calculate the minimum movement cost (`cost_move`): Initialize `cost_move = float('inf')`. Iterate through all `adj_tile` in `AdjacentTiles`. Find the shortest path distance `dist = self.distances[robot_tile].get(adj_tile, float('inf'))` using the precomputed APSP table. Update `cost_move = min(cost_move, dist)`.
        d. If `cost_move` remains `inf` after checking all adjacent tiles, this robot cannot reach any position to paint the target tile. Its total cost `cost_robot` is effectively infinite for this goal.
        e. Otherwise, the total estimated cost for this robot to achieve this goal is `cost_robot = 1 (for the paint action) + cost_color + cost_move`.
        f. Update the minimum cost found so far for the goal: `min_cost_for_G = min(min_cost_for_G, cost_robot)`.
    10. If `min_cost_for_G` is still `inf` after checking all robots, it means no robot can achieve this goal from the current state. The state is a dead end. Return `float('inf')`.
    11. Add the calculated `min_cost_for_G` (the cost associated with the best robot for this specific goal) to the total heuristic value `h`.
    12. After checking all goals, return the total heuristic value `h`. Since the cost for each unsatisfied goal is at least 1 (for the paint action), `h` will be greater than 0 if any goals were unsatisfied. If `h` is 0, it implies all `painted` goals were satisfied, and the initial check confirmed it's a goal state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information, goals,
        and precomputing distances.
        """
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.adj = {} # Adjacency list: tile -> set of adjacent tiles
        self.tiles = set()
        self.robots = set()

        # --- Parse Static Facts (Connectivity) ---
        # Use a set of tuples to store unique, undirected edges
        tile_pairs = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            # Check for connectivity predicates (up, down, left, right)
            if predicate in ["up", "down", "left", "right"] and len(args) == 2:
                t1, t2 = args[0], args[1]
                self.tiles.add(t1)
                self.tiles.add(t2)
                # Add edge as a sorted tuple to handle symmetry and avoid duplicates
                tile_pairs.add(tuple(sorted((t1, t2))))

        # Build the adjacency list from the unique pairs
        for t1, t2 in tile_pairs:
             self.adj.setdefault(t1, set()).add(t2)
             self.adj.setdefault(t2, set()).add(t1)

        # --- Identify Robots and Initial Tiles ---
        # Robots are identified from initial state predicates like robot-at or robot-has
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             args = parts[1:]
             if predicate == "robot-at" and len(args) == 2: # e.g., (robot-at robot1 tile_0_1)
                 self.robots.add(args[0])
                 self.tiles.add(args[1]) # Ensure robot's initial tile is in the set
             elif predicate == "robot-has" and len(args) == 2: # e.g., (robot-has robot1 black)
                 self.robots.add(args[0])

        # --- Store Goal Information & Ensure Goal Tiles are Known ---
        self.goal_painted = {} # target_tile -> target_color
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            # Check if it's a 'painted' goal predicate
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color
                self.tiles.add(tile) # Ensure goal tiles are included in the set of all tiles

        # --- Compute All-Pairs Shortest Paths (APSP) using BFS ---
        # self.distances[tile1][tile2] will store the shortest path length
        self.distances = {}
        for start_node in self.tiles:
            # Initialize distances from start_node to all other known tiles as infinity
            self.distances[start_node] = {tile: float('inf') for tile in self.tiles}

            # Distance from a node to itself is 0
            if start_node in self.tiles: # Check if start_node is valid
                 self.distances[start_node][start_node] = 0
            else:
                 continue # Skip if start_node isn't a known tile

            # BFS queue initialized with the starting node
            queue = deque([start_node])
            # Visited set for the current BFS run
            visited = {start_node}

            while queue:
                current_node = queue.popleft()
                current_dist = self.distances[start_node][current_node]

                # Explore neighbors using the adjacency list
                # Use .get(current_node, set()) to handle nodes with no outgoing edges
                for neighbor in self.adj.get(current_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        """
        state = node.state

        # Optimization: If the state is already a goal state, heuristic value is 0.
        if self.task.goal_reached(state):
            return 0

        # If there are no robots defined, goals cannot be achieved unless already met.
        if not self.robots:
             # Since the goal wasn't met initially, return infinity.
             return float('inf')

        # --- Parse Current State Information ---
        robot_at = {} # robot -> current_tile
        robot_has = {} # robot -> current_color
        painted = {} # tile -> current_color

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == "robot-at" and len(args) == 2:
                robot_at[args[0]] = args[1]
            elif predicate == "robot-has" and len(args) == 2:
                robot_has[args[0]] = args[1]
            elif predicate == "painted" and len(args) == 2:
                painted[args[0]] = args[1]

        # --- Calculate Heuristic: Sum of minimum costs for each unsatisfied goal ---
        total_heuristic = 0

        for target_tile, target_color in self.goal_painted.items():
            # Check if this specific goal is already satisfied
            if painted.get(target_tile) == target_color:
                continue

            # --- This goal is unsatisfied, calculate the minimum cost to achieve it ---

            # Find tiles adjacent to the target tile (needed for painting actions)
            adjacent_tiles = self.adj.get(target_tile, set())

            # If the target tile has no adjacent tiles, it cannot be painted.
            if not adjacent_tiles:
                 # This goal is unreachable, making the overall state unsolvable from here.
                 return float('inf')

            min_cost_for_goal = float('inf')

            # Find the minimum cost across all robots to achieve this goal
            for robot in self.robots:
                # Check if the robot's current state is known and valid
                if robot not in robot_at or robot not in robot_has:
                    continue # Skip robot if its state is not fully defined
                robot_tile = robot_at[robot]
                if robot_tile not in self.distances:
                    # Robot is on a tile not recognized in static connectivity map.
                    continue # Cannot calculate movement cost

                robot_color = robot_has[robot]

                # 1. Calculate Color Change Cost
                cost_color = 0 if robot_color == target_color else 1

                # 2. Calculate Minimum Movement Cost to an Adjacent Tile
                min_move_cost = float('inf')
                # Check distance from robot's current tile to each adjacent tile
                for adj_tile in adjacent_tiles:
                    # Use .get() for safety, defaulting to infinity if adj_tile isn't reachable
                    dist = self.distances[robot_tile].get(adj_tile, float('inf'))
                    min_move_cost = min(min_move_cost, dist)

                # If min_move_cost is still infinity, the robot cannot reach any adjacent tile
                if min_move_cost == float('inf'):
                     cost_for_robot = float('inf')
                else:
                     # 3. Calculate Painting Cost (always 1 action)
                     cost_paint = 1
                     # Total cost for this robot to achieve this goal
                     cost_for_robot = cost_paint + cost_color + min_move_cost

                # Update the minimum cost found so far for this goal
                min_cost_for_goal = min(min_cost_for_goal, cost_for_robot)

            # If min_cost_for_goal is still infinity after checking all robots,
            # then this specific goal is unreachable from the current state.
            if min_cost_for_goal == float('inf'):
                # The overall state is a dead end.
                return float('inf')

            # Add the minimum cost for achieving this goal to the total heuristic value
            total_heuristic += min_cost_for_goal

        # Return the total estimated cost.
        # If total_heuristic is 0, it means all 'painted' goals were satisfied,
        # and the initial check confirmed the state is a goal state.
        # If total_heuristic > 0, it represents the sum of minimum costs for unsatisfied goals.
        return total_heuristic
