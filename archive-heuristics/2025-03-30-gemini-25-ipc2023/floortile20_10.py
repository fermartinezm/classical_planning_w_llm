import collections
import heapq # Not strictly needed for BFS, but can be useful
from fnmatch import fnmatch
# Import the base class for heuristics
from heuristics.heuristic_base import Heuristic 

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes parentheses and splits a PDDL fact string into parts.
    Handles potential extra spaces and removes leading/trailing whitespace.
    """
    return fact.strip()[1:-1].split()

# Helper function to match facts against patterns
def match(fact, *pattern):
    """Checks if a fact string matches a given pattern tuple.
    Uses fnmatch for basic wildcard support if needed in patterns.
    """
    parts = get_parts(fact)
    if len(parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    Estimates the cost to reach the goal state by summing the estimated minimum
    costs for achieving each unsatisfied 'painted' goal individually. The cost for
    a single goal considers the minimum cost for any available robot to paint the
    target tile with the target color. This cost includes the shortest path
    movement cost for the robot to reach an adjacent tile from which it can paint,
    the cost of changing color if necessary, and the cost of the paint action itself.
    This heuristic is designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - All actions (move_up/down/left/right, paint_up/down, change_color) have a uniform cost of 1.
    - The heuristic ignores potential negative interactions or resource conflicts between robots 
      (e.g., two robots needing the same tile clear simultaneously). It assumes robots can move
      along their shortest paths without interference.
    - It assumes that any tile specified in a `(painted ?tile ?color)` goal is
      paintable according to the domain rules (i.e., it is adjacent to some other tile `x` via 
      an `(up ?tile x)` or `(down ?tile x)` relationship defined in the static facts, allowing 
      `paint_up` or `paint_down` from `x`).
    - It assumes that the target tile for painting (`?y` in paint actions) is 'clear' when the 
      paint action needs to be applied. If the tile is not clear in the current state (e.g.,
      occupied by another robot), the heuristic does not add extra cost for clearing it, 
      potentially underestimating the true cost in such specific cases.
    - The heuristic calculates the minimum cost over all robots for each goal independently and 
      sums these minimum costs. This sum serves as a non-admissible estimate guiding greedy search, 
      as it doesn't solve the optimal assignment of goals to robots.

    # Heuristic Initialization
    - Extracts all robot, tile, and color objects from the task definition (static facts, init, goals).
    - Parses static facts (`up`, `down`, `left`, `right`) to build an adjacency
      graph representation of the tile grid for movement.
    - Determines, for each tile `y`, which adjacent tiles `x` a robot must be AT to paint `y`
      (based on `(up y x)` or `(down y x)` facts). Stores this mapping in `paintable_from[y] -> {x1, x2,...}`.
    - Computes all-pairs shortest paths (APSP) between all tiles using Breadth-First
      Search (BFS) starting from each tile. Stores distances in `dist[start][end]`.
    - Stores the set of goal facts that are of the form `(painted ?tile ?color)`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unsatisfied Goals**: Filter the set of goal `(painted ...)` facts
        to find those not present in the current `state`. If this set is empty,
        the goal is reached, and the heuristic value is 0.
    2.  **Get Current Robot States**: Parse the current `state` to find the location
        (`robot-at`) and held color (`robot-has`) for each robot. Store these in dictionaries.
    3.  **Calculate Cost Per Unsatisfied Goal**: Initialize total heuristic `h = 0`. Iterate
        through each unsatisfied goal `g = (painted target_tile target_color)`:
        a. Initialize `min_cost_for_this_goal` to infinity. This will store the minimum cost found
           across all robots to achieve this specific goal.
        b. Determine the set of `possible_robot_locations` (tiles `x` such that `(up target_tile x)`
           or `(down target_tile x)` is true) using the precomputed `paintable_from` map.
           If this set is empty for `target_tile`, the goal is considered intrinsically
           unreachable by the domain's actions (no tile `x` exists to paint from); return infinity 
           immediately as the state is likely unsolvable.
        c. Iterate through each known `robot` identified during initialization:
           i.   Get the robot's current location `robot_loc` and current color `robot_col`
                from the state information gathered in step 2. If the robot isn't in the current
                state's `robot_locs` (e.g., if robots could be removed, though unlikely), skip it.
           ii.  **Color Change Cost**: If the robot currently holds a color (`robot_col` is not None)
                and `robot_col` is different from `target_color`, the `color_change_cost` is 1.
                Otherwise (if color matches or robot holds no color), it's 0.
           iii. **Movement Cost**: Find the minimum shortest path distance from `robot_loc`
                to any tile `adj_tile` in `possible_robot_locations`. This is calculated as
                `min(self.dist[robot_loc].get(adj_tile, float('inf')))`. Store this as `move_cost`.
           iv.  **Paint Cost**: The cost of the `paint_up` or `paint_down` action is always 1.
           v.   **Validity Check & Total Cost**: A robot can only achieve the goal if it can reach the
                required location (`move_cost != inf`) AND it currently holds *some* color (`robot_col` 
                is not None), because both `paint` and `change_color` actions require `(robot-has ?r ?c)`.
                If these conditions aren't met, this robot cannot achieve the goal in this estimated sequence.
                Set `cost_for_robot_r` to infinity. Otherwise, calculate the total estimated cost for this robot:
                `cost_for_robot_r = move_cost + color_change_cost + paint_cost`.
           vi.  **Update Minimum**: Update `min_cost_for_this_goal = min(min_cost_for_this_goal, cost_for_robot_r)`.
        d. **Check Goal Reachability**: After checking all robots, if `min_cost_for_this_goal`
           is still infinity, it means no robot can achieve this specific goal from the current
           state according to the heuristic's estimation. The state might be a dead end or unsolvable.
           Return infinity.
        e. **Accumulate Cost**: Add the finite `min_cost_for_this_goal` to the total heuristic value `h`.
    4.  **Return Total Heuristic Value**: Return the final sum `h`. Ensure it's non-negative (it should be
        by construction, but `max(0, h)` is safe).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        - Extracts objects (robots, tiles, colors).
        - Builds tile adjacency graph.
        - Determines where robots must be to paint each tile.
        - Computes all-pairs shortest paths between tiles.
        - Stores the painting goals.
        """
        self.goals = task.goals
        static_facts = task.static
        # Combine all known facts to ensure all objects are captured reliably
        all_known_facts = static_facts | task.initial_state | self.goals

        # --- Extract Objects ---
        self.robots = set()
        self.tiles = set()
        self.colors = set()
        for fact in all_known_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty or malformed facts
             pred = parts[0]
             args = parts[1:]
             try:
                 # Based on predicate definitions and argument types
                 if pred == 'robot-at' and len(args) == 2: self.robots.add(args[0]); self.tiles.add(args[1])
                 elif pred in ['up', 'down', 'left', 'right'] and len(args) == 2: self.tiles.add(args[0]); self.tiles.add(args[1])
                 elif pred == 'clear' and len(args) == 1: self.tiles.add(args[0])
                 elif pred == 'painted' and len(args) == 2: self.tiles.add(args[0]); self.colors.add(args[1])
                 elif pred == 'robot-has' and len(args) == 2: self.robots.add(args[0]); self.colors.add(args[1])
                 elif pred == 'available-color' and len(args) == 1: self.colors.add(args[0])
                 # Predicate 'free-color' is ignored as it seems unused in actions/goals
             except IndexError:
                 # Fallback for unexpected argument counts
                 # print(f"Warning: Skipping fact due to unexpected format: {fact}")
                 pass

        # --- Build Grid Adjacency and Paintability Info ---
        # self.adj: tile -> {adjacent tiles for movement}
        self.adj = collections.defaultdict(set)
        # self.paintable_from: tile_to_paint (y) -> {set of tiles robot must be AT (x) to paint y}
        self.paintable_from = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            # Process adjacency relations for movement
            if pred in ['up', 'down', 'left', 'right'] and len(args) == 2:
                t1, t2 = args[0], args[1]
                # Add bidirectional adjacency for movement actions
                self.adj[t1].add(t2)
                self.adj[t2].add(t1)
                
                # Process relations relevant for painting actions
                # paint_up(?r ?y ?x ?c): requires (up ?y ?x), robot at ?x paints ?y
                # paint_down(?r ?y ?x ?c): requires (down ?y ?x), robot at ?x paints ?y
                if pred == 'up': # Fact is (up y x)
                    y, x = args[0], args[1]
                    self.paintable_from[y].add(x)
                elif pred == 'down': # Fact is (down y x)
                    y, x = args[0], args[1]
                    self.paintable_from[y].add(x)

        # --- Compute All-Pairs Shortest Paths (APSP) using BFS ---
        # self.dist: start_tile -> end_tile -> shortest_distance
        self.dist = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))
        tile_list = list(self.tiles) # Use a fixed list for iteration stability
        
        for start_node in tile_list:
            if start_node not in self.tiles: continue # Should not happen, but safety check
            
            self.dist[start_node][start_node] = 0
            queue = collections.deque([(start_node, 0)]) # Store (node, distance)
            visited = {start_node} # Keep track of visited nodes to prevent cycles and redundant work

            while queue:
                u, d = queue.popleft()

                # Explore neighbors based on the adjacency graph
                for v in self.adj.get(u, set()):
                    if v not in visited:
                        visited.add(v)
                        self.dist[start_node][v] = d + 1
                        queue.append((v, d + 1))

        # --- Store Goal Painted Facts ---
        # self.goal_painted: set of goals like '(painted tile_1_1 white)'
        self.goal_painted = set()
        for goal in self.goals:
             parts = get_parts(goal)
             # Check if it's a 'painted' goal with correct number of args
             if parts and parts[0] == "painted" and len(parts) == 3:
                 self.goal_painted.add(goal)


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # --- Identify Unsatisfied Goals ---
        unsatisfied_goals = {goal for goal in self.goal_painted if goal not in state}

        if not unsatisfied_goals:
            return 0 # Goal state reached

        # --- Get Current Robot States ---
        robot_locs = {} # robot -> location
        robot_colors = {} # robot -> color (or None if not holding one)
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            if pred == "robot-at" and len(args) == 2:
                robot_locs[args[0]] = args[1]
            elif pred == "robot-has" and len(args) == 2:
                robot_colors[args[0]] = args[1]

        # --- Calculate Heuristic Value ---
        h_value = 0
        # Iterate through robots known to exist in the problem instance
        active_robots = list(self.robots) 
        if not active_robots: # If there are no robots defined, goals are unreachable
             if unsatisfied_goals: return float('inf')
             else: return 0 # Should not happen if goals exist

        for goal in unsatisfied_goals:
            _, target_tile, target_color = get_parts(goal)

            min_cost_for_goal = float('inf')

            # Find locations robot needs to be AT to paint the target_tile
            possible_robot_locs = self.paintable_from.get(target_tile)

            if not possible_robot_locs:
                # This goal tile cannot be painted based on domain's up/down actions
                # This indicates an unsolvable goal or problem definition issue.
                # print(f"Warning: Goal tile {target_tile} is unpaintable (no adjacent 'x' found).")
                return float('inf') # Unsolvable state

            for robot in active_robots:
                # Check if this robot is currently in the state (has a location)
                if robot not in robot_locs:
                    continue # Skip robots not present or without location in this state

                robot_loc = robot_locs[robot]
                # Get robot's color, defaults to None if (robot-has) fact is missing
                robot_col = robot_colors.get(robot) 

                # Cost 1: Color Change
                color_change_cost = 0
                # A robot needs to *have* a color to perform change_color or paint
                can_perform_actions = (robot_col is not None)
                
                if can_perform_actions and robot_col != target_color:
                    color_change_cost = 1

                # Cost 2: Movement
                move_cost = float('inf')
                for adj_tile in possible_robot_locs:
                    # Get precomputed distance; default to infinity if no path or tile unknown
                    dist = self.dist[robot_loc].get(adj_tile, float('inf'))
                    move_cost = min(move_cost, dist)

                # Cost 3: Painting
                paint_cost = 1

                # Validity Check & Total Cost Calculation
                if move_cost == float('inf') or not can_perform_actions:
                    # Robot cannot reach a painting position, OR
                    # Robot doesn't hold a color, so cannot paint or change color.
                    cost_for_robot_r = float('inf')
                else:
                    cost_for_robot_r = move_cost + color_change_cost + paint_cost

                # Update the minimum cost found so far for achieving this goal
                min_cost_for_goal = min(min_cost_for_goal, cost_for_robot_r)

            # After checking all robots, if min_cost is still infinity, the goal is unreachable
            if min_cost_for_goal == float('inf'):
                 # print(f"Warning: Goal {goal} seems unreachable by any robot in state {state}.")
                 return float('inf') # Indicate unsolvable state from here

            # Add the minimum cost for achieving this goal to the total heuristic value
            h_value += min_cost_for_goal

        # Ensure heuristic is non-negative (should be true by construction)
        return max(0, h_value)

