import collections
import itertools
from fnmatch import fnmatch
# The planner framework provides the Heuristic base class.
# If running standalone, you might need to define a placeholder like this:
# class Heuristic:
#     """Base class for heuristics."""
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError("Heuristic evaluation not implemented.")
from heuristics.heuristic_base import Heuristic # Ensure this import path is correct for your planner setup


# Helper functions
def get_parts(fact_str):
    """Removes parentheses and splits a PDDL fact string."""
    # Handles potential leading/trailing whitespace and removes parentheses
    return fact_str.strip()[1:-1].split()

def match(fact_parts, *pattern):
    """Checks if the parts of a fact match a pattern (with '*' wildcards)."""
    if len(fact_parts) != len(pattern):
        return False
    # Use fnmatch for wildcard matching ('*')
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their specified colors. It calculates the cost for each unsatisfied goal
    by finding the minimum estimated cost for any single robot to achieve that goal.
    The total heuristic value is the sum of these minimum costs over all unsatisfied goals.
    The cost for a single robot to achieve a single goal includes estimated movement
    actions, a potential color change action, and the final painting action.

    # Assumptions
    - The heuristic ignores the `clear` predicate constraint for movement and painting.
      It assumes that robots can move to any required tile and that target tiles
      can be painted when needed (relaxation). This means paths are calculated on the
      static grid structure, ignoring dynamic obstacles (other robots or already painted tiles).
    - Robots are assumed to operate independently when calculating the cost for each goal.
      Potential interference, blocking, or synergies between robots are ignored.
    - Each goal `(painted tile color)` is evaluated independently, and the minimum
      costs are summed. This might overestimate the total actions if a single robot
      action (like moving or changing color) can contribute to achieving multiple goals.
    - The cost of the `change_color` action is always 1.
    - The cost of `move_*` actions is always 1.
    - The cost of `paint_up`/`paint_down` actions is always 1.

    # Heuristic Initialization
    - Extracts the target `(painted tile color)` facts from the task goals.
    - Identifies all tiles, robots, and available colors from the task objects and initial state/static facts.
    - Builds two adjacency maps from static facts:
        - `move_adj`: Represents possible moves between tiles (`up`, `down`, `left`, `right`).
        - `paint_adj`: Maps each tile `t` to the set of tiles from which `t` can be painted (i.e., tiles `p` where `(up t p)` or `(down t p)` holds).
    - Precomputes all-pairs shortest path distances (minimum number of `move_*` actions)
      between all tiles using Breadth-First Search (BFS) on the `move_adj` map. Stores
      these distances in `self.distances`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current state (`node.state`) to find:
        - The set of currently satisfied `(painted tile color)` facts (as tuples).
        - The current location `tile_robot` for each robot `r`.
        - The current color `color_robot` held by each robot `r`.
    2.  Identify the set of unsatisfied goals by comparing the task goals (stored as tuples)
        with the currently painted tiles (parsed into tuples).
    3.  If there are no unsatisfied goals, the state is a goal state, and the
        heuristic value is 0.
    4.  Initialize `total_heuristic_value = 0`.
    5.  For each unsatisfied goal `(painted tile_goal color_goal)`:
        a.  Find the set of `painting_positions` for `tile_goal` using the
            precomputed `paint_adj` map. These are the tiles a robot must be at
            to paint `tile_goal`.
        b.  Initialize a minimum cost `min_cost_for_this_goal = float('inf')`.
        c.  If `painting_positions` is empty, the goal tile cannot be painted according
            to the static connections, so the goal is considered unreachable (cost is infinity).
        d.  For each robot `r`:
            i.   Get the robot's current location `tile_robot` and held color `color_robot`.
                 Handle cases where robot state information might be missing in the current state.
            ii.  Calculate the movement cost (`move_cost`): Find the minimum shortest
                 path distance from `tile_robot` to any tile `p` in `painting_positions`.
                 This uses the precomputed `self.distances`. If no path exists from the
                 robot's current location to any valid painting position, this cost is infinity.
                 `move_cost = min(finite_costs)` where finite_costs are distances != inf.
            iii. Calculate the color change cost (`color_cost`): 1 if `color_robot` is
                 not equal to `color_goal`, otherwise 0.
            iv.  Calculate the painting cost (`paint_cost`): Always 1 for the
                 `paint_up` or `paint_down` action.
            v.   Sum these costs: `cost_r = move_cost + color_cost + paint_cost`. If `move_cost`
                 was infinity, `cost_r` will be infinity.
            vi.  Update the minimum cost found so far for this goal across all robots:
                 `min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)`.
        e.  Add `min_cost_for_this_goal` to `total_heuristic_value`. If `min_cost_for_this_goal`
            remains infinity after checking all robots, it means the goal is unreachable
            from the current state by any robot. In this case, the total heuristic value
            becomes infinity, signaling a potential dead end.
    6.  Return `total_heuristic_value`.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class
        self.goals = task.goals
        self.static_facts = task.static

        # Extract objects
        self.tiles = set()
        self.robots = set()
        self.colors = set()

        # Infer objects from goal facts
        self.goal_tuples = set() # Store goals as (tile, color) tuples
        for fact in self.goals:
            try:
                parts = get_parts(fact)
                # Ensure the fact is a 'painted' predicate with 3 parts
                if len(parts) == 3 and parts[0] == "painted":
                    tile, color = parts[1], parts[2]
                    self.goal_tuples.add((tile, color))
                    self.tiles.add(tile)
                    self.colors.add(color)
            except IndexError:
                # Skip malformed goal facts
                print(f"Warning: Could not parse goal fact: {fact}")
                continue

        # Infer objects and build adjacency maps from static facts
        self.move_adj = collections.defaultdict(set) # tile -> {neighbor_tile} for movement
        self.paint_adj = collections.defaultdict(set) # target_tile -> {robot_pos_tile} for painting

        for fact in self.static_facts:
            try:
                parts = get_parts(fact)
                # Adjacency predicates define tiles and connectivity
                if len(parts) == 3 and parts[0] in {"up", "down", "left", "right"}:
                    t1, t2 = parts[1], parts[2]
                    self.tiles.add(t1)
                    self.tiles.add(t2)
                    # Movement graph (assume symmetric connection based on domain structure)
                    self.move_adj[t1].add(t2)
                    self.move_adj[t2].add(t1)

                    # Painting adjacency: robot at t2 can paint t1 via up/down
                    if parts[0] == "up" or parts[0] == "down":
                         self.paint_adj[t1].add(t2)

                # Available colors
                elif len(parts) == 2 and parts[0] == "available-color":
                    self.colors.add(parts[1])
            except IndexError:
                # Skip malformed static facts
                print(f"Warning: Could not parse static fact: {fact}")
                continue

        # Infer robots from initial state facts (more reliable than just objects list)
        for fact in task.initial_state:
             try:
                 parts = get_parts(fact)
                 if len(parts) == 3 and parts[0] == "robot-at":
                     self.robots.add(parts[1])
                     self.tiles.add(parts[2]) # Ensure tile is known
                 elif len(parts) == 3 and parts[0] == "robot-has":
                     self.robots.add(parts[1])
                     self.colors.add(parts[2]) # Ensure color is known
             except IndexError:
                 # Skip malformed init facts
                 print(f"Warning: Could not parse initial state fact: {fact}")
                 continue

        # Precompute all-pairs shortest path distances for movement
        self.distances = self._compute_distances()


    def _compute_distances(self):
        """Computes all-pairs shortest paths using BFS on the movement graph."""
        distances = {t: {t2: float('inf') for t2 in self.tiles} for t in self.tiles}
        if not self.tiles:
             return distances # Return empty dict if no tiles

        for start_node in self.tiles:
            # Check if start_node exists in the graph keys, handle isolated nodes
            if start_node not in self.move_adj and len(self.tiles) > 1:
                 distances[start_node][start_node] = 0
                 continue # Isolated node, only reachable from itself

            distances[start_node][start_node] = 0
            queue = collections.deque([start_node])
            visited = {start_node} # Visited set per BFS run

            while queue:
                current_node = queue.popleft()
                current_dist = distances[start_node][current_node]

                # Iterate through neighbors using the move_adj map
                for neighbor in self.move_adj.get(current_node, set()):
                    # Ensure neighbor is a known tile before processing
                    if neighbor in self.tiles and neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state

        # --- 1. Parse current state ---
        robot_locations = {} # robot -> tile
        robot_colors = {}    # robot -> color
        current_painted_tuples = set() # (tile, color)

        for fact in state:
            try:
                parts = get_parts(fact)
                if len(parts) == 3 and parts[0] == "robot-at":
                    robot_locations[parts[1]] = parts[2]
                elif len(parts) == 3 and parts[0] == "robot-has":
                    robot_colors[parts[1]] = parts[2]
                elif len(parts) == 3 and parts[0] == "painted":
                    current_painted_tuples.add((parts[1], parts[2]))
            except IndexError:
                 # Ignore malformed facts in the state
                 continue

        # --- 2. Identify unsatisfied goals ---
        unsatisfied_goals = self.goal_tuples - current_painted_tuples

        # --- 3. Check for goal state ---
        if not unsatisfied_goals:
            return 0

        # --- 4. Initialize heuristic value ---
        total_heuristic_value = 0

        # --- 5. Calculate cost for each unsatisfied goal ---
        # If there are goals but no robots, the task is impossible
        if not self.robots and unsatisfied_goals:
            return float('inf')

        for goal_tile, goal_color in unsatisfied_goals:
            # 5a. Find positions from where goal_tile can be painted
            painting_positions = self.paint_adj.get(goal_tile, set())

            # 5b. Initialize min cost for achieving this goal
            min_cost_for_this_goal = float('inf')

            # 5c. If no positions exist to paint from, goal is unreachable
            if not painting_positions:
                 # This implies the goal tile has no up/down neighbours defined,
                 # or wasn't properly parsed. Assume unreachable.
                 return float('inf')

            # 5d. Calculate cost for each robot to achieve this goal
            for robot in self.robots:
                # Ensure robot's current state is known
                if robot not in robot_locations or robot not in robot_colors:
                     # Robot state unknown, cannot contribute. Cost is effectively infinite.
                     cost_r = float('inf')
                else:
                    tile_robot = robot_locations[robot]
                    color_robot = robot_colors[robot]

                    # 5d.ii. Movement Cost
                    move_cost = float('inf')
                    # Check if robot's location is valid and in distance matrix
                    if tile_robot in self.distances:
                        # Calculate min distance to any required painting position p
                        # Ensure p is reachable from tile_robot (dist is not inf)
                        possible_costs = [self.distances[tile_robot].get(p, float('inf'))
                                          for p in painting_positions if p in self.distances[tile_robot]]
                        # Filter out infinities before taking min, handle case of no reachable positions
                        finite_costs = [c for c in possible_costs if c != float('inf')]
                        if finite_costs:
                             move_cost = min(finite_costs)
                        # else: move_cost remains infinity if no painting position is reachable

                    # 5d.iii. Color Cost
                    color_cost = 0
                    if color_robot != goal_color:
                        color_cost = 1 # Assumes change_color action costs 1

                    # 5d.iv. Paint Cost
                    paint_cost = 1 # Assumes paint action costs 1

                    # 5d.v. Total cost for this robot and goal
                    # If move_cost is inf, cost_r will be inf
                    if move_cost == float('inf'):
                        cost_r = float('inf')
                    else:
                        # Ensure costs are integers before summing if possible, though float('inf') requires float
                        cost_r = float(move_cost) + float(color_cost) + float(paint_cost)

                # 5d.vi. Update minimum cost found for this goal across all robots
                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)

            # 5e. Add minimum cost for this goal to total heuristic value
            # If min_cost_for_this_goal is still infinity, the goal is deemed unreachable
            if min_cost_for_this_goal == float('inf'):
                # This state leads to a dead end regarding this goal
                return float('inf')

            total_heuristic_value += min_cost_for_this_goal

        # --- 6. Return total heuristic value ---
        # Can be 0, positive float (due to inf intermediate), or float('inf')
        # Return as int if not infinity, for consistency if preferred by planner
        if total_heuristic_value == float('inf'):
            return float('inf')
        else:
            # Ensure non-negative integer value
            return max(0, int(round(total_heuristic_value)))
