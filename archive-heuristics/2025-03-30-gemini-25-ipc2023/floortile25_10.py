import collections
import itertools
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'floortile'.

    # Summary
    This heuristic estimates the remaining cost to reach the goal state by summing
    the estimated costs for achieving each unsatisfied 'painted' goal predicate independently.
    For each unsatisfied goal `(painted target_tile target_color)`, it calculates the minimum
    cost required for any single robot to achieve it. This cost includes the movement cost
    for the robot to reach an adjacent tile from which it can paint, the cost of changing
    to the `target_color` if necessary, and the cost of the paint action itself (cost 1).
    The total heuristic value is the sum of these minimum costs over all unsatisfied goals.

    # Assumptions
    - The cost of each action (move, paint, change_color) is 1.
    - The heuristic ignores the `clear` precondition for movement and painting actions.
      It assumes robots can move freely to adjacent tiles based on shortest path distance,
      even if those tiles are currently occupied or already painted.
    - It ignores potential interactions or resource contention between robots. The cost
      for each goal is calculated independently, assuming a dedicated (cheapest) robot
      for that specific goal, without considering if that robot might be needed elsewhere
      or if its path/action might interfere with others.
    - The movement cost is based on the precomputed shortest path distance between tiles,
      ignoring any dynamic obstacles (other robots).
    - A robot needs one `change_color` action if it doesn't currently hold the required color.

    # Heuristic Initialization
    - The constructor (`__init__`) performs preprocessing based on the task's static information
      and goal specification.
    - It identifies all tiles, robots, and available colors defined in the problem.
    - It parses static connectivity predicates (`up`, `down`, `left`, `right`) to build an
      adjacency graph representation of the tile layout.
    - It determines, for each tile, the set of adjacent tiles from which it can be painted
      using `paint_up` or `paint_down` actions (based on static `up`/`down` predicates).
    - It computes all-pairs shortest path distances between all tiles using Breadth-First Search (BFS)
      starting from each tile. These distances are stored for efficient lookup during heuristic evaluation.
    - It extracts the set of goal conditions, specifically the `(painted tile color)` facts required
      in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Get Current State:** Extract the current locations of all robots (`robot-at`), the
        color each robot holds (`robot-has`), and the set of already painted tiles (`painted`)
        from the input state `node.state`.
    2.  **Identify Unsatisfied Goals:** Compare the set of `(painted tile color)` facts in the
        current state with the set of required goal conditions (precomputed in `__init__`).
        Create a list of unsatisfied goals `G_unsat`.
    3.  **Check for Goal State:** If `G_unsat` is empty, the current state is a goal state,
        return 0.
    4.  **Calculate Cost per Goal:** Initialize total heuristic value `h = 0`. Iterate through
        each unsatisfied goal `g = (target_tile, target_color)` in `G_unsat`:
        a.  Initialize the cost for this specific goal `cost_g = 1` (representing the final `paint` action).
        b.  Retrieve the precomputed set `Adj` of adjacent tiles from which `target_tile` can be painted.
            If `Adj` is empty (meaning the tile cannot be painted according to static facts),
            the goal is unreachable; return `float('inf')`.
        c.  Initialize `min_cost_for_robot = float('inf')` to track the minimum cost found so far
            across all robots to achieve this goal `g`.
        d.  Iterate through each robot `r` defined in the task:
            i.   Get the robot's current location `loc = robot_locs[r]` and current held
                 color `color = robot_colors.get(r, None)`.
            ii.  Calculate the color change cost: `color_change_cost = 1` if `color != target_color`, else 0.
                 If the `target_color` is not an available color in the domain, return `float('inf')`.
            iii. Calculate the minimum movement cost: `move_cost = min(dist[loc][adj])` for all `adj` in `Adj`.
                 The `dist[loc][adj]` values are the precomputed shortest path distances. If `loc` cannot
                 reach any `adj` tile (distance is infinity), set `move_cost` to `float('inf')`.
            iv.  Calculate the total cost for this robot `r` to achieve goal `g`:
                 `robot_total_cost = color_change_cost + move_cost`. If `move_cost` was infinity,
                 this total cost will also be infinity.
            v.   Update the minimum cost found for this goal:
                 `min_cost_for_robot = min(min_cost_for_robot, robot_total_cost)`.
        e.  After checking all robots, if `min_cost_for_robot` is still `float('inf')`, it means
            no robot can achieve this goal; return `float('inf')`.
        f.  Add the minimum cost found for this goal to its base paint cost: `cost_g += min_cost_for_robot`.
        g.  Add the total cost for this goal to the overall heuristic value: `h += cost_g`.
    5.  **Return Total Heuristic Value:** After iterating through all unsatisfied goals, return the
        accumulated heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Preprocessing: Parse static info and compute distances
        self._parse_static_info()
        self._compute_distances()

        # Store goal conditions for quick lookup
        self.goal_conditions = set()
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                # Ensure parts has the expected structure before accessing indices
                if len(parts) == 3:
                    self.goal_conditions.add((parts[1], parts[2])) # (tile, color) tuple

    def _parse_static_info(self):
        """Parses static facts to build adjacency map, identify objects, and find paint locations."""
        self.tiles = set()
        self.robots = set()
        self.available_colors = set()
        # Adjacency list for distance calculation (undirected)
        self.adj = collections.defaultdict(list)
        # Map: target_tile -> set of adjacent tiles from which it can be painted
        self.paintable_from = collections.defaultdict(set)

        # Extract objects and connectivity from static facts
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ["up", "down", "left", "right"] and len(parts) == 3:
                t1, t2 = parts[1], parts[2]
                self.tiles.add(t1)
                self.tiles.add(t2)
                # Add edges for BFS distance calculation (assume movement is possible in both directions between adjacent tiles)
                self.adj[t1].append(t2)
                self.adj[t2].append(t1)
                # Store specific locations for painting actions based on domain definition
                # (paint_up ?r ?y ?x ?c) requires (up ?y ?x), robot at ?x paints ?y
                if pred == "up":
                    self.paintable_from[t1].add(t2) # t1 is ?y, t2 is ?x
                # (paint_down ?r ?y ?x ?c) requires (down ?y ?x), robot at ?x paints ?y
                elif pred == "down":
                    self.paintable_from[t1].add(t2) # t1 is ?y, t2 is ?x
            elif pred == "available-color" and len(parts) == 2:
                self.available_colors.add(parts[1])

        # Ensure all objects mentioned in init/goal are captured, especially robots
        all_facts = self.task.initial_state.union(self.goals)
        for fact in all_facts:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == "robot-at" and len(parts) == 3:
                 self.robots.add(parts[1])
                 self.tiles.add(parts[2])
             elif pred == "robot-has" and len(parts) == 3:
                 self.robots.add(parts[1])
                 # Color might not be in available_colors if robot starts with it? Add it.
                 self.available_colors.add(parts[2])
             elif pred == "painted" and len(parts) == 3:
                 self.tiles.add(parts[1])
                 self.available_colors.add(parts[2]) # Ensure goal colors are known
             elif pred == "clear" and len(parts) == 2:
                 self.tiles.add(parts[1])

        # If no robots are defined in the problem, the heuristic logic might fail later.
        # This case should ideally be handled based on whether goals require painting.
        if not self.robots and any(match(g, "painted", "*", "*") for g in self.goals):
            print("Warning: No robots defined in the problem, but goals require painting.")
            # Heuristic will likely return infinity if called.


    def _compute_distances(self):
        """Computes all-pairs shortest paths between tiles using BFS."""
        self.dist = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))

        # Run BFS from each tile to find shortest paths to all other reachable tiles
        for start_node in self.tiles:
            # Distance from start_node to itself is 0
            self.dist[start_node][start_node] = 0
            queue = collections.deque([start_node])
            # Keep track of visited nodes for the current BFS run
            visited_in_bfs = {start_node}

            while queue:
                u = queue.popleft()
                current_dist = self.dist[start_node][u]

                # Explore neighbors
                for v in self.adj[u]:
                    if v not in visited_in_bfs:
                        visited_in_bfs.add(v)
                        self.dist[start_node][v] = current_dist + 1
                        queue.append(v)

    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # 1. Get current state information
        robot_locs = {}
        robot_colors = {}
        painted_tiles_state = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "robot-at" and len(parts) == 3:
                robot_locs[parts[1]] = parts[2]
            elif pred == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]
            elif pred == "painted" and len(parts) == 3:
                painted_tiles_state.add((parts[1], parts[2]))

        # 2. Identify unsatisfied goals
        unsatisfied_goals = self.goal_conditions - painted_tiles_state

        # 3. Check for Goal State
        if not unsatisfied_goals:
            return 0 # Goal reached

        # Handle case where goals exist but no robots are defined or present in the state
        if not self.robots or not robot_locs:
             if unsatisfied_goals:
                 return float('inf') # Cannot achieve painting goals without robots
             else:
                 return 0 # Should have been caught above, but safe check

        # 4. Calculate heuristic value by summing costs for each unsatisfied goal
        h_value = 0
        for target_tile, target_color in unsatisfied_goals:
            # Cost for the paint action itself
            cost_g = 1

            # Find adjacent tiles from where painting is possible
            paint_locations = self.paintable_from.get(target_tile)

            # If a goal tile cannot be painted from anywhere, the goal is unreachable
            if not paint_locations:
                # print(f"Warning: Goal tile {target_tile} has no adjacent painting locations.")
                return float('inf')

            min_cost_for_robot = float('inf')

            # Check if the target color is valid
            if target_color not in self.available_colors:
                 # print(f"Warning: Target color {target_color} for tile {target_tile} is not available.")
                 return float('inf') # Cannot achieve goal if color doesn't exist

            # Find the minimum cost (move + color_change) for any robot to paint this tile
            for robot in self.robots:
                # Robot must exist and have a location in the current state
                if robot not in robot_locs:
                    # This robot isn't present or trackable in this state, skip it.
                    continue

                current_loc = robot_locs[robot]
                current_color = robot_colors.get(robot, None) # Robot might not hold a color

                # Cost to change color (if needed)
                color_change_cost = 0
                if current_color != target_color:
                    color_change_cost = 1

                # Cost to move to the closest valid painting location
                move_cost = float('inf')
                for adj_tile in paint_locations:
                    # Retrieve precomputed distance. Default is 'inf' if not reachable.
                    distance = self.dist[current_loc].get(adj_tile, float('inf'))
                    move_cost = min(move_cost, distance)

                # If move_cost is still infinity, this robot cannot reach any painting spot for this tile
                if move_cost == float('inf'):
                    robot_total_cost = float('inf')
                else:
                    robot_total_cost = color_change_cost + move_cost

                min_cost_for_robot = min(min_cost_for_robot, robot_total_cost)

            # If min_cost_for_robot is still infinity after checking all robots, no robot can achieve this goal
            if min_cost_for_robot == float('inf'):
                # print(f"Warning: No robot can reach a painting location for goal ({target_tile}, {target_color}).")
                return float('inf') # Goal is unreachable from this state

            # Add the minimum robot-specific cost (move + color) to the paint action cost
            cost_g += min_cost_for_robot
            # Add the total cost for achieving this goal to the overall heuristic value
            h_value += cost_g

        # 5. Return Total Heuristic Value
        # Ensure heuristic is non-negative (guaranteed by calculation)
        # Ensure heuristic is 0 iff goal state (handled by initial check)
        # Ensure heuristic is finite for solvable states (returns inf for detected unsolvable states)
        return h_value

