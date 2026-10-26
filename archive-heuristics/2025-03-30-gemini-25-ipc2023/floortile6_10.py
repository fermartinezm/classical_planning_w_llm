import sys
from collections import deque
import math # Using math.inf for infinity

# Try to import the base class, define a placeholder if not found
try:
    # Ensure the path to 'heuristics' directory is discoverable by Python
    # Example: export PYTHONPATH=$PYTHONPATH:/path/to/project/root
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Using a placeholder.", file=sys.stderr)
    # Define a placeholder if the import fails (e.g., for standalone testing)
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# Helper function to parse PDDL fact string
def get_parts(fact: str):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: '(pred obj1 obj2)' -> ['pred', 'obj1', 'obj2']
    Returns an empty list if the fact is malformed (e.g., not enclosed in parentheses).
    """
    fact = fact.strip()
    if len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL floortile domain.

    # Summary
    Estimates the cost to reach the goal state by summing the estimated costs
    for achieving each unsatisfied 'painted' goal individually. For each
    unsatisfied goal (painting a specific tile with a specific color), it
    calculates the minimum cost required by any available robot to achieve it.
    This cost includes moving the robot to an adjacent tile, potentially
    changing the robot's held color, and performing the paint action. It is
    designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - The primary goal is to satisfy all `(painted tile color)` predicates specified in the goal.
    - Robots can move between adjacent tiles according to `up/down/left/right` predicates.
      The heuristic calculates shortest path based on connectivity, assuming target tiles for movement
      will eventually become clear (it doesn't check the `clear` predicate).
    - Robots paint adjacent tiles based on `paint_up`/`paint_down` actions, requiring the robot
      to be at the correct adjacent tile (`x`) to paint the target tile (`y`).
    - The cost of each action (move_*, paint_*, change_color) is 1.
    - The heuristic aims for informativeness for Greedy Best-First Search.
    - Tile connectivity defines a graph where shortest paths represent minimum move actions.

    # Heuristic Initialization
    - Parses static facts (`up`, `down`, `left`, `right`) to build a tile connectivity graph (`self.adj`)
      representing possible moves (tile -> list of reachable neighbors).
    - Identifies locations (`x`) from which each target tile (`y`) can be painted (`self.paint_locations`),
      based on `up`/`down` static predicates linking `y` and `x`.
    - Precomputes all-pairs shortest paths distances between tiles using Breadth-First Search (BFS) (`self.distances`).
      Unreachable pairs have infinite distance.
    - Stores the set of goal predicates (`self.goals`) and extracts the target (tile, color) pairs
      that need to be painted (`self.goal_tuples`).
    - Identifies all unique tile names mentioned in static connectivity facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unsatisfied Goals**: Compare the `(painted tile color)` facts in the current `state`
        against the required `self.goal_tuples`. Collect all goal tuples not present in the state.
    2.  **Handle Goal State**: If there are no unsatisfied goals, the state is a goal state, return 0.
    3.  **Iterate Through Unsatisfied Goals**: For each unsatisfied goal `(tile_goal, color_goal)`:
        a.  **Find Potential Painting Locations**: Look up `tile_goal` in `self.paint_locations` to get the list
            of adjacent tiles (`paint_locs`) a robot must be at to paint `tile_goal`.
        b.  **Check Goal Achievability Primitives**: If `tile_goal` has no associated painting locations (e.g., it's not
            paintable via up/down actions as defined in static facts) or if there are no robots in the current state,
            this specific goal is considered unachievable from the current state configuration. Assign a large penalty
            (`self.large_penalty`) to this goal's cost and proceed to the next goal.
        c.  **Calculate Minimum Robot Cost**: If the goal seems potentially achievable, iterate through each
            robot `r` present in the state:
            i.   *Get Robot State*: Find the robot's current location `tile_r` and held color `color_r`.
            ii.  *Calculate Color Change Cost* (`cost_color`): 0 if `color_r == color_goal`, else 1 (for `change_color`).
            iii. *Calculate Minimum Movement Cost* (`min_dist_to_paint_loc`): Find the minimum shortest path
                 distance from `tile_r` to any tile in `paint_locs` using precomputed `self.distances`.
                 If `tile_r` cannot reach any `paint_loc`, this distance is infinity.
            iv.  *Calculate Total Cost for Robot* (`cost_r`): Sum `cost_color`, `min_dist_to_paint_loc`,
                 and 1 (for the `paint` action). If `min_dist_to_paint_loc` is infinity, `cost_r` is infinity.
        d.  **Determine Minimum Cost for Goal** (`min_cost_for_goal`): Find the minimum finite `cost_r` among
            all robots calculated in the previous step. Initialize this search with infinity.
        e.  **Handle Unreachable Goal**: If after checking all robots, `min_cost_for_goal` remains infinity,
            it means no robot can currently reach a position to paint it. Assign the large penalty
            (`self.large_penalty`) to `min_cost_for_goal`.
    4.  **Sum Costs**: Add the `min_cost_for_goal` (which might be the penalty) for the current unsatisfied goal
        to the total heuristic value `h`.
    5.  **Final Value**: Return the total sum `h`. If `h` happens to be 0 but the state is not a goal state
        (which theoretically shouldn't happen with cost >= 1 per goal), return 1 as a safeguard.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        Builds adjacency graph, finds painting locations, computes all-pairs shortest paths,
        and extracts goal tuples.
        """
        self.goals = task.goals
        static_facts = task.static

        self.adj = {}  # tile -> list of adjacent tiles reachable FROM tile
        self.paint_locations = {} # tile_to_paint -> list of tiles robot must be AT
        all_tiles = set()
        self.goal_tuples = set()
        # Using a large number as penalty for potentially unreachable goals
        self.large_penalty = 10000

        print("Initializing floortileHeuristic...", file=sys.stderr)
        # Parse static facts to build graph, identify paint locations, and collect tiles
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            # Process connectivity predicates (up, down, left, right)
            if pred in ["up", "down", "left", "right"]:
                if len(parts) != 3: continue # Ensure format (pred t1 t2)
                t1, t2 = parts[1], parts[2]
                all_tiles.add(t1)
                all_tiles.add(t2)
                # Adjacency: If (pred t1 t2), a robot at t2 can potentially move to t1
                # Example: (up tile_1_1 tile_0_1) allows move_up from tile_0_1 to tile_1_1
                self.adj.setdefault(t2, []).append(t1)

                # Painting Locations: Only up/down relations allow painting actions
                # Example: (up t1 t2) -> paint_up(r, y=t1, x=t2, c) needs robot at t2 to paint t1
                if pred in ["up", "down"]:
                    self.paint_locations.setdefault(t1, []).append(t2)

            # Process explicit tile type declarations if present
            elif pred == "tile":
                 if len(parts) > 1:
                     all_tiles.add(parts[1])

        self.all_tiles = list(all_tiles)
        print(f"Found {len(self.all_tiles)} unique tiles.", file=sys.stderr)

        # Precompute all-pairs shortest paths using BFS
        self.distances = self._compute_all_pairs_shortest_paths()

        # Extract goal tuples (tile, color) that need painting
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Ensure the goal is a 'painted' predicate with correct arity
            if parts and parts[0] == "painted" and len(parts) == 3:
                self.goal_tuples.add((parts[1], parts[2])) # (tile, color)
        print(f"Found {len(self.goal_tuples)} painting goals.", file=sys.stderr)
        print("Initialization complete.", file=sys.stderr)


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of tiles using BFS.
        Returns a dictionary mapping (start_tile, end_tile) to distance (int or math.inf).
        """
        distances = {}
        if not self.all_tiles:
            print("Warning: No tiles found for distance calculation.", file=sys.stderr)
            return distances

        num_tiles = len(self.all_tiles)
        print(f"Computing all-pairs shortest paths for {num_tiles} tiles...", file=sys.stderr)

        for i, start_node in enumerate(self.all_tiles):
            # Initialize distances from start_node to infinity for all tiles
            for tile in self.all_tiles:
                distances[(start_node, tile)] = math.inf
            # Distance to self is 0
            distances[(start_node, start_node)] = 0

            # BFS queue stores (node, distance)
            queue = deque([(start_node, 0)])

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors based on the precomputed adjacency list `self.adj`
                for neighbor in self.adj.get(current_node, []):
                    # If we found a shorter path to the neighbor
                    if distances[(start_node, neighbor)] > dist + 1:
                        distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

            # Optional: Provide progress update periodically
            # if (i + 1) % max(1, num_tiles // 10) == 0:
            #     print(f"  BFS progress: {i+1}/{num_tiles}", file=sys.stderr)

        print("Finished computing distances.", file=sys.stderr)
        # Optional: Check for unreachable pairs
        inf_count = sum(1 for d in distances.values() if d == math.inf)
        # Subtract self-loops which should be 0
        unreachable_pairs = inf_count - num_tiles if inf_count >= num_tiles else inf_count
        if unreachable_pairs > 0:
             print(f"Warning: Found {unreachable_pairs} unreachable tile pairs.", file=sys.stderr)

        return distances

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state node.
        Estimates the remaining cost to reach the goal state.
        """
        state = node.state
        h = 0 # Initialize heuristic value

        # --- Parse current state information ---
        robot_locations = {} # Map: robot_id -> current_tile
        robot_colors = {}    # Map: robot_id -> current_color
        current_painted = set() # Set of (tile, color) tuples currently painted

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            pred = parts[0]
            # Extract robot locations
            if pred == "robot-at" and len(parts) == 3:
                robot_locations[parts[1]] = parts[2]
            # Extract robot colors
            elif pred == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]
            # Extract currently painted tiles
            elif pred == "painted" and len(parts) == 3:
                current_painted.add((parts[1], parts[2]))

        # --- Identify unsatisfied goals ---
        unsatisfied_goals = self.goal_tuples - current_painted

        # If all goals are satisfied, heuristic value is 0
        if not unsatisfied_goals:
            return 0

        # --- Calculate cost for each unsatisfied goal ---
        for tile_goal, color_goal in unsatisfied_goals:
            min_cost_for_goal = math.inf # Initialize minimum cost for this goal

            # Find locations from where this tile can be painted
            possible_paint_locations = self.paint_locations.get(tile_goal, [])

            # Check for primitive unachievability conditions
            if not possible_paint_locations:
                # This tile cannot be painted according to static facts (no up/down link)
                min_cost_for_goal = self.large_penalty
            elif not robot_locations:
                 # No robots exist in the current state to achieve any goal
                 min_cost_for_goal = self.large_penalty
            else:
                # Calculate cost for each robot to achieve this goal
                achievable_by_any_robot = False
                current_min_robot_cost = math.inf

                for robot, robot_tile in robot_locations.items():
                    robot_color = robot_colors.get(robot)
                    # Assume robot always has a color if it exists in robot-at
                    if robot_color is None: continue # Skip if robot state is inconsistent

                    # Cost Component 1: Color Change Cost
                    cost_color = 0 if robot_color == color_goal else 1

                    # Cost Component 2: Minimum Movement Cost
                    min_dist_to_paint_loc = math.inf
                    for paint_loc in possible_paint_locations:
                        # Retrieve precomputed distance, default to infinity if pair missing (should not happen with full BFS)
                        dist = self.distances.get((robot_tile, paint_loc), math.inf)
                        min_dist_to_paint_loc = min(min_dist_to_paint_loc, dist)

                    # Cost Component 3: Painting Action Cost
                    cost_paint = 1

                    # Calculate total cost for this robot for this goal
                    if min_dist_to_paint_loc == math.inf:
                        # This robot cannot reach any location to paint this tile
                        cost_r = math.inf
                    else:
                        cost_r = cost_color + min_dist_to_paint_loc + cost_paint
                        achievable_by_any_robot = True # Mark that at least one robot path exists

                    # Update the minimum cost found so far across all robots for this goal
                    current_min_robot_cost = min(current_min_robot_cost, cost_r)

                # If no robot could find a path (all costs were inf), assign penalty
                if not achievable_by_any_robot:
                     min_cost_for_goal = self.large_penalty
                else:
                     min_cost_for_goal = current_min_robot_cost

            # --- Add cost for this goal to total heuristic value ---
            # If cost is inf (truly unreachable), add the penalty instead
            if min_cost_for_goal == math.inf:
                 h += self.large_penalty
            else:
                 h += min_cost_for_goal

        # --- Final Safeguard and Return ---
        # If heuristic calculated to 0 but goals are not met, return 1.
        # This indicates a potential flaw, ensures non-zero cost for non-goal states.
        if h == 0 and unsatisfied_goals:
             return 1

        # Return the calculated heuristic value. Can be large if penalties were added.
        # Check for potential overflow if penalties sum up excessively? Assume standard float precision is sufficient.
        return h

