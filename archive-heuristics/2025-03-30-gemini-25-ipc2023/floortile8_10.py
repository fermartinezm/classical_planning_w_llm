import itertools
from collections import deque
from fnmatch import fnmatch
# Assuming the planner provides the Heuristic base class and task structure
# from heuristics.heuristic_base import Heuristic 
# If the base class is not automatically available, you might need to define a placeholder:
class Heuristic:
    def __init__(self, task):
        self.task = task
    def __call__(self, node):
        raise NotImplementedError

# Helper functions
def get_parts(fact: str) -> list[str]:
    """Extract the components of a PDDL fact string (e.g., "(pred a b)")."""
    # Removes parentheses and splits by space
    return fact[1:-1].split()

def match(fact: str, *args: str) -> bool:
    """
    Check if a PDDL fact string matches a given pattern.
    Uses fnmatch for wildcard support in args.
    Example: match("(at robot1 locA)", "at", "robot*", "*") -> True
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the pattern length
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain floortile.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing estimates for three components: painting actions, color change actions,
    and movement actions. It targets informativeness for Greedy Best-First Search
    and is not necessarily admissible (it might overestimate the true cost).

    # Assumptions
    - The goal is solely defined by `(painted tile color)` predicates.
    - Each robot always holds exactly one color (the `free-color` predicate seems unused
      in the provided domain actions).
    - The tile layout forms a grid where movement cost is 1 per step between adjacent tiles.
      Adjacency is defined by `up`, `down`, `left`, `right` static predicates.
    - Painting a tile `y` requires the robot `r` to be on an adjacent tile `x` such that
      `(up y x)` or `(down y x)` holds, and `y` must be `clear`.
    - All actions (move, paint, change_color) have a cost of 1.

    # Heuristic Initialization (`__init__`)
    - Parses the task's static facts (`up`, `down`, `left`, `right`, `available-color`)
      to build the tile grid connectivity for movement (`self.adj`) and identify potential
      painting positions (`self.paint_adj`).
    - Identifies all unique tiles, robots (inferred from initial state predicates like
      `robot-at`), and available colors (from `available-color` and initial `robot-has`).
    - Precomputes all-pairs shortest paths (APSP) between all tiles using Breadth-First Search (BFS)
      on the movement graph (`self.adj`) and stores them in `self.distances` for quick lookup
      of minimum movement costs.
    - Stores the goal `(painted tile color)` conditions as a set of tuples in `self.goal_predicates`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse State:** Extract current robot locations (`robot_loc`), the color held by each
        robot (`robot_color`), and the set of currently painted tiles with their colors
        (`painted_tiles`) from the input state node.
    2.  **Identify Unsatisfied Goals:** Determine the set `unsatisfied_goals` containing
        `(goal_tile, goal_color)` pairs from `self.goal_predicates` that are not present
        in the current `painted_tiles`. If this set is empty, the goal state is reached,
        and the heuristic value is 0.
    3.  **Estimate Painting Cost (`h_paint`):** Each unsatisfied goal requires at least one
        `paint_up` or `paint_down` action. The estimated cost is simply the number of
        unsatisfied goals: `h_paint = len(unsatisfied_goals)`.
    4.  **Estimate Color Change Cost (`h_color`):**
        a. Find `RequiredColors`: the set of unique colors needed for the `unsatisfied_goals`.
        b. Find `HeldColors`: the set of unique colors currently held by any robot in the state.
        c. Estimate the minimum necessary `change_color` actions as the number of required colors
           that are not currently held by any robot: `h_color = len(RequiredColors - HeldColors)`.
           This assumes that if a needed color isn't held, at least one robot must change to it.
    5.  **Estimate Movement Cost (`h_move`):**
        a. Initialize `h_move = 0`.
        b. For each `(goal_tile, goal_color)` in `unsatisfied_goals`:
           i. Identify the set of adjacent tiles (`PaintPositions`) from which `goal_tile` can be
              painted, using the precomputed `self.paint_adj` map. If no such positions exist
              (based on static `up`/`down` facts), the goal is impossible; return infinity.
           ii. Calculate `min_dist_for_goal`: the minimum movement distance from *any* robot's
              current location (`r_tile`) to *any* of the required `PaintPositions`. This uses
              the precomputed `self.distances[(r_tile, paint_pos)]`.
           iii. If no robot can reach any `PaintPosition` for this goal (i.e., all distances are
               infinity), the state is a dead end regarding this goal; return infinity.
           iv. Add this minimum distance (`min_dist_for_goal`) to the total movement estimate:
               `h_move += min_dist_for_goal`. This summation likely overestimates the true total
               movement, as one robot might move efficiently between multiple goals, but serves
               as an informative estimate for greedy search.
    6.  **Combine Costs:** The final heuristic value is the sum of the estimated costs for
        painting, color changes, and movement: `h = h_paint + h_color + h_move`. Return this
        value as a float.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class if necessary
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.adj = {} # tile -> {neighbor1, ...} for movement adjacency
        self.paint_adj = {} # tile_to_paint -> {robot_pos1, ...} for painting adjacency
        self.tiles = set()
        self.robots = set()
        self.available_colors = set()
        self.goal_predicates = set() # Stores (tile, color) tuples representing goals

        # --- Parse Static Facts & Infer Objects ---
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            # Process adjacency predicates (up, down, left, right)
            if pred in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                self.tiles.add(t1)
                self.tiles.add(t2)
                # Build movement adjacency graph (assuming symmetric relations are provided)
                self.adj.setdefault(t1, set()).add(t2)
                self.adj.setdefault(t2, set()).add(t1)
                # Build painting adjacency map: if robot is at t2, it can paint t1
                # Only 'up' and 'down' allow painting in this domain
                if pred == "up" or pred == "down":
                    # t1 is the tile being painted, t2 is the robot's required position
                    self.paint_adj.setdefault(t1, set()).add(t2)
            # Process available colors
            elif pred == "available-color":
                self.available_colors.add(parts[1])

        # --- Infer Robots, Tiles, Colors from Initial State and Goals ---
        # Use initial state to find robots and ensure all objects involved are known
        for fact in task.initial_state:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == "robot-at":
                 self.robots.add(parts[1]) # Identify robot
                 self.tiles.add(parts[2])  # Identify tile
             elif pred == "robot-has":
                 self.robots.add(parts[1]) # Identify robot
                 self.available_colors.add(parts[2]) # Colors held initially are effectively available
             elif pred == "clear":
                 self.tiles.add(parts[1]) # Identify tile
             elif pred == "painted":
                 self.tiles.add(parts[1]) # Identify tile
                 self.available_colors.add(parts[2]) # Colors used initially are relevant

        # Parse goals and ensure goal objects (tiles, colors) are known
        for goal in self.goals:
            # Assuming goals are simple predicates like (painted tile color)
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile, color = parts[1], parts[2]
                self.goal_predicates.add((tile, color))
                self.tiles.add(tile) # Ensure goal tile is in the set of tiles
                self.available_colors.add(color) # Ensure goal color is known

        # --- Precompute All-Pairs Shortest Paths (APSP) using BFS ---
        self.distances = {} # Stores (tile1, tile2) -> distance
        if self.tiles: # Proceed only if tiles were identified
            for tile1 in self.tiles:
                # Compute shortest distances from tile1 to all reachable tiles
                tile_distances = self._bfs(tile1, self.adj)
                for tile2, dist in tile_distances.items():
                    # Store distance, ensuring symmetry for undirected graph
                    if (tile1, tile2) not in self.distances:
                         self.distances[(tile1, tile2)] = dist
                    if (tile2, tile1) not in self.distances:
                         self.distances[(tile2, tile1)] = dist
        # Using self.distances.get((t1, t2), float('inf')) later handles unreachable pairs

    def _bfs(self, start_node: str, adj: dict[str, set[str]]) -> dict[str, int]:
        """
        Performs Breadth-First Search on the tile grid graph defined by 'adj'
        starting from 'start_node'.
        Returns a dictionary mapping reachable tiles to their shortest distance (number of moves).
        """
        distances = {start_node: 0} # Distance from start_node to itself is 0
        queue = deque([start_node]) # Queue for BFS nodes to visit
        visited = {start_node}      # Set of visited nodes to avoid cycles

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Explore neighbors of the current node
            # Use .get(node, set()) to handle nodes that might have no outgoing edges listed
            for neighbor in adj.get(current_node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1 # Distance increases by 1 move
                    queue.append(neighbor)
        return distances

    def __call__(self, node) -> float:
        """
        Calculate the heuristic value (estimated cost to goal) for the given state node.
        Returns float (to accommodate infinity for dead ends).
        """
        state = node.state # The current state is a frozenset of PDDL fact strings

        # --- Parse Current State ---
        robot_loc = {} # Maps robot name -> current tile location
        robot_color = {} # Maps robot name -> current held color
        painted_tiles = set() # Set of (tile, color) tuples currently painted
        for fact in state:
            # Efficiently parse common predicates
            if fact.startswith("(robot-at"):
                parts = get_parts(fact)
                robot_loc[parts[1]] = parts[2]
            elif fact.startswith("(robot-has"):
                parts = get_parts(fact)
                robot_color[parts[1]] = parts[2]
            elif fact.startswith("(painted"):
                parts = get_parts(fact)
                painted_tiles.add((parts[1], parts[2]))

        # --- Identify Unsatisfied Goals ---
        # Find goal predicates that are not currently true in the state
        unsatisfied_goals = set()
        for goal_tile, goal_color in self.goal_predicates:
            if (goal_tile, goal_color) not in painted_tiles:
                 unsatisfied_goals.add((goal_tile, goal_color))
                 # Note: This simple check assumes the problem is valid. It doesn't check
                 # if a tile is goal-painted with color C1 but already painted with C2.
                 # Such states should ideally be pruned by the planner's goal check or
                 # action applicability checks (cannot paint an already painted tile).

        # If all goals are satisfied, heuristic cost is 0
        if not unsatisfied_goals:
            return 0.0

        # --- Calculate Heuristic Components ---

        # 1. Painting Cost (`h_paint`): Minimum 1 paint action per unsatisfied goal.
        h_paint = len(unsatisfied_goals)

        # 2. Color Change Cost (`h_color`): Minimum changes needed to get required colors.
        required_colors = {color for _, color in unsatisfied_goals}
        held_colors = set(robot_color.values())
        # Cost is the number of required colors not currently held by *any* robot.
        h_color = len(required_colors - held_colors)

        # 3. Movement Cost (`h_move`): Sum of minimum moves for each goal.
        h_move = 0
        # Check if robots exist; if not, goals requiring robots are unreachable.
        if not self.robots:
             # If there are unsatisfied goals but no robots, return infinity.
             return float('inf')

        for goal_tile, goal_color in unsatisfied_goals:
            min_dist_for_goal = float('inf')
            # Find positions from which this goal_tile can be painted
            possible_paint_positions = self.paint_adj.get(goal_tile, set())

            # If a goal tile has no adjacent position allowing painting (based on static facts)
            if not possible_paint_positions:
                # This indicates an impossible goal based on the grid structure.
                return float('inf')

            found_reachable_robot = False
            # Find the minimum distance from *any* robot to *any* required painting position
            for r in self.robots:
                r_tile = robot_loc.get(r)
                # If a robot's location isn't defined in the state (shouldn't happen)
                if r_tile is None: continue

                # Find the minimum distance from this robot 'r' to any valid painting position
                dist_r_to_paint_pos = float('inf')
                for paint_pos in possible_paint_positions:
                    # Lookup precomputed distance, default to infinity if path doesn't exist
                    dist = self.distances.get((r_tile, paint_pos), float('inf'))
                    dist_r_to_paint_pos = min(dist_r_to_paint_pos, dist)

                # If this robot 'r' can reach at least one painting position
                if dist_r_to_paint_pos != float('inf'):
                    found_reachable_robot = True
                    # Update the overall minimum distance needed for this goal (across all robots)
                    min_dist_for_goal = min(min_dist_for_goal, dist_r_to_paint_pos)

            # If no robot could reach any painting position for this specific goal
            if not found_reachable_robot:
                 # This state is a dead end regarding achieving this goal.
                 return float('inf')

            # Add the minimum distance found for this goal to the total move estimate
            # If min_dist_for_goal remained infinity, the previous check returned already.
            h_move += min_dist_for_goal

        # --- Combine Costs ---
        # Total estimated cost is the sum of costs for painting, color changes, and movement.
        heuristic_value = h_paint + h_color + h_move
        # Return as a float, ensuring non-negativity (though components should be non-negative)
        return float(max(0, heuristic_value))

