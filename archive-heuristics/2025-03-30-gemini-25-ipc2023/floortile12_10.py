import itertools
from collections import deque
from fnmatch import fnmatch
# Assuming the planner infrastructure provides the base class
# and that it is available in the execution environment.
# If not, uncomment the following placeholder:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError
from heuristics.heuristic_base import Heuristic # Adjust import path if necessary


# Helper functions (defined globally or as static methods)
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: '(pred obj1 obj2)' -> ['pred', 'obj1', 'obj2']
    Returns an empty list if the fact is malformed.
    """
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle potential malformed facts gracefully
        return []
    return fact[1:-1].split()

def match(fact, *pattern):
    """
    Checks if a PDDL fact string matches a given pattern.
    The pattern consists of strings, where '*' can be used as a wildcard.
    Example: match('(at robot1 locA)', 'at', '*', 'locA') -> True
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the pattern length
    if len(parts) != len(pattern):
        return False
    # Check if each part matches the corresponding pattern element (using fnmatch for wildcards)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL floortile domain.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to paint
    all goal tiles with their specified colors. It calculates the minimum
    estimated cost required for each *individual* unpainted goal tile. This
    cost considers the shortest path movement for a robot, a potential color
    change action, and the painting action itself, minimized over all available
    robots. The heuristic value is the sum of these minimum costs for all
    unpainted goals. It is designed to guide greedy best-first search and is
    likely non-admissible (it might overestimate the true cost).

    # Assumptions
    - All actions defined in the domain (move_*, paint_*, change_color) have
      a uniform cost of 1.
    - The tile grid structure, defined by the static `up`, `down`, `left`,
      `right` predicates, does not change during planning.
    - For pathfinding calculations, robots are assumed to be able to move
      through any tile. The heuristic ignores potential temporary blockages
      by other robots (this is a common relaxation in heuristics).
    - Painting a target tile `y` requires the robot to be positioned on an
      adjacent tile `x` such that `up(y, x)` or `down(y, x)` holds, according
      to the `paint_up` and `paint_down` action definitions.
    - The primary objective is to satisfy all `painted(tile, color)` facts
      specified in the task's goal section.

    # Heuristic Initialization
    - The constructor (`__init__`) performs pre-computation based on the task's
      static information and goals.
    - It parses the `task.goals` to identify the set of target `(tile, color)` pairs.
    - It parses `task.static_facts` (containing `up`, `down`, `left`, `right`)
      to build an adjacency map representing the tile grid connectivity.
    - It identifies all unique tile names involved in the grid.
    - It pre-computes all-pairs shortest path distances between all tiles using
      Breadth-First Search (BFS) and stores them in `self.dist`.
    - It pre-computes a mapping `self.paintable_from[target_tile]` which stores
      the set of tiles `{x}` from which `target_tile` can be painted (derived
      from `up(target_tile, x)` and `down(target_tile, x)` facts).
    - It identifies all robot names present in the problem, typically inferred
      from predicates like `robot-at` or `robot-has` in the `task.initial_state`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State**: In the `__call__` method, extract the current
        locations of all robots (`robot-at` facts), the color each robot holds
        (`robot-has` facts), and the set of currently painted tiles
        (`painted` facts) from the input `node.state`.
    2.  **Identify Unsatisfied Goals**: Determine the set of `unpainted_goals`
        by taking the difference between the required goal `(tile, color)` pairs
        (stored in `self.parsed_goals`) and the `painted_in_state` set.
    3.  **Handle Goal State**: If the `unpainted_goals` set is empty, the
        current state satisfies the painting goals, so the heuristic value is 0.
    4.  **Initialize Heuristic Value**: Set `total_heuristic_value = 0`.
    5.  **Check Robot Availability**: If there are `unpainted_goals` but no
        robots are currently present in the state (`robot_locations` is empty),
        the goals cannot be achieved. Return infinity (`float('inf')`).
    6.  **Iterate Through Unsatisfied Goals**: For each goal
        `(target_tile, target_color)` in the `unpainted_goals` set:
        a.  **Find Potential Painting Locations**: Retrieve the precomputed set
            of tiles `possible_paint_spots = self.paintable_from[target_tile]`
            from which `target_tile` can be painted. If this set is empty (due
            to domain structure), the goal is statically impossible; return infinity.
        b.  **Calculate Minimum Cost Per Robot**: Initialize the minimum cost
            for achieving this specific goal to infinity
            (`min_cost_for_this_goal = float('inf')`). Then, iterate through
            each `robot` currently active in the state:
            i.   **Get Robot State**: Find the robot's current location
                 `robot_loc` and the color it holds `robot_col`. If the robot
                 currently holds no color (which might indicate an unusual state
                 or domain variation), skip this robot for this painting task.
            ii.  **Calculate Minimum Movement Cost**: Find the minimum shortest
                 path distance from `robot_loc` to any tile `spot` in the
                 `possible_paint_spots`. Use the precomputed `self.dist` table.
                 If no `spot` is reachable from `robot_loc` (all distances are
                 infinity), this robot cannot currently paint this tile; continue
                 to the next robot. Keep track if at least one robot *can* reach
                 a suitable spot (`achievable_by_any_robot`).
            iii. **Calculate Color Change Cost**: Set `color_cost = 1` if the
                 robot's current color `robot_col` is different from the
                 `target_color`, otherwise set `color_cost = 0`.
            iv.  **Calculate Painting Cost**: The cost of the `paint` action
                 itself is always 1 (`paint_action_cost = 1`).
            v.   **Total Cost for Robot r**: Sum the costs:
                 `cost_r = move_cost + color_cost + paint_action_cost`.
            vi.  **Update Minimum for Goal**: Update the minimum cost found so far
                 for this goal:
                 `min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)`.
        c.  **Check Goal Reachability**: After checking all active robots, if
            none of them could reach a valid painting spot for this goal (i.e.,
            `achievable_by_any_robot` is still false), then this goal is
            dynamically unreachable in the current state; return infinity.
        d.  **Accumulate Cost**: Add the calculated `min_cost_for_this_goal`
            (the minimum cost found across all robots for *this* goal) to the
            `total_heuristic_value`.
    7.  **Return Total Heuristic Value**: After iterating through all unpainted
        goals, return the final `total_heuristic_value`. If any step determined
        that the goals are unreachable, infinity will have been returned earlier.
        The value is returned as an integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-computing static information.
        - Parses goals.
        - Builds tile adjacency map from static connectivity facts.
        - Computes all-pairs shortest paths between tiles.
        - Determines potential painting locations for each tile.
        - Identifies robots from the initial state.
        """
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static
        # Store initial state to help identify robots if needed
        self.initial_state = task.initial_state

        # 1. Parse Goals into a set of (tile, color) tuples
        self.parsed_goals = set()
        for fact in self.goals:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                # Ensure fact has the expected structure (painted tile color)
                if len(parts) == 3:
                    self.parsed_goals.add((parts[1], parts[2]))

        # 2. Build Adjacency Map, Identify Tiles, and Precompute Paintable Spots
        self.adj = {} # Adjacency list for pathfinding: tile -> {neighbor1, neighbor2,...}
        self.tiles = set() # Set of all unique tile names
        # Map: target_tile -> {set of tiles robot can be on to paint target_tile}
        self.paintable_from = {}

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]

            # Process connectivity facts (up, down, left, right)
            if pred in ["up", "down", "left", "right"] and len(parts) == 3:
                t1, t2 = parts[1], parts[2]
                # Add tiles to the set of known tiles
                self.tiles.add(t1)
                self.tiles.add(t2)
                # Add adjacency links (assuming movement is bidirectional for pathfinding)
                self.adj.setdefault(t1, set()).add(t2)
                self.adj.setdefault(t2, set()).add(t1)

                # Determine painting spots based on 'up' and 'down' relations
                # If up(y, x) or down(y, x), a robot at tile x can paint tile y.
                if pred == "up" or pred == "down":
                     target_tile, robot_pos = t1, t2
                     self.paintable_from.setdefault(target_tile, set()).add(robot_pos)

        # Ensure all tiles mentioned in goals or initial state are known,
        # even if they are isolated according to connectivity facts.
        involved_tiles = set()
        for fact in self.initial_state.union(self.goals):
             parts = get_parts(fact)
             if not parts: continue
             # Add tiles involved in relevant predicates
             if parts[0] in ["robot-at", "clear", "painted"] and len(parts) > 1:
                 involved_tiles.add(parts[1])
             if parts[0] == "robot-at" and len(parts) == 3:
                 involved_tiles.add(parts[2]) # Add the tile the robot is at
        self.tiles.update(involved_tiles) # Add these tiles to the main set

        # 3. Compute All-Pairs Shortest Paths using BFS
        self.dist = self._compute_all_pairs_shortest_paths(self.tiles, self.adj)

        # 4. Identify Robots (Infer from initial state predicates)
        self.robots = set()
        for fact in self.initial_state:
             # Robots are typically the first argument in robot-at or robot-has
             if match(fact, "robot-at", "*", "*") or match(fact, "robot-has", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) > 1:
                     self.robots.add(parts[1]) # Add the robot name

        # Optional: Add a warning if no robots are found, as the heuristic might not work.
        if not self.robots:
             print(f"Warning: floortileHeuristic initialized for task {task.name} but found no robots in the initial state.")


    def _compute_all_pairs_shortest_paths(self, nodes, adj):
        """
        Computes shortest path distances between all pairs of nodes using BFS.
        Assumes an unweighted graph (all moves cost 1).
        Returns a dictionary distances[start_node][end_node] = distance.
        Distance is float('inf') if nodes are disconnected.
        """
        distances = {n: {} for n in nodes}
        for start_node in nodes:
            # Initialize distances from start_node to all other nodes as infinity
            for node in nodes:
                distances[start_node][node] = float('inf')
            # Distance to self is 0
            distances[start_node][start_node] = 0

            # Queue for BFS: stores (node, distance_from_start)
            queue = deque([(start_node, 0)])
            # Keep track of visited nodes and their shortest distance found so far
            visited_dist = {start_node: 0}

            while queue:
                current_node, d = queue.popleft()

                # Explore neighbors of the current node
                for neighbor in adj.get(current_node, set()):
                    # If neighbor hasn't been visited yet
                    if neighbor not in visited_dist:
                        visited_dist[neighbor] = d + 1
                        distances[start_node][neighbor] = d + 1
                        queue.append((neighbor, d + 1))
                    # Optional optimization for weighted graphs (not needed here):
                    # If a shorter path is found to an already visited node.
                    # elif d + 1 < visited_dist[neighbor]:
                    #    visited_dist[neighbor] = d + 1
                    #    distances[start_node][neighbor] = d + 1
                    #    queue.append((neighbor, d + 1)) # Re-add to update paths
        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value (estimated cost to goal) for the given state node.
        """
        state = node.state

        # 1. Parse current state information
        robot_locations = {} # robot -> tile
        robot_colors = {}    # robot -> color
        painted_in_state = set() # set of (tile, color) tuples

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            # Extract robot location
            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            # Extract robot's current color
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            # Extract currently painted tiles
            elif parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_in_state.add((tile, color))

        # 2. Identify unpainted goals
        unpainted_goals = self.parsed_goals - painted_in_state

        # 3. Handle Goal State: If no goals remain unpainted, cost is 0
        if not unpainted_goals:
            return 0

        # 4. Initialize heuristic value
        total_heuristic_value = 0
        # Get the list of robots currently active (i.e., have a location)
        active_robots = list(robot_locations.keys())

        # 5. Check Robot Availability: If goals remain but no robots are active
        if not active_robots and unpainted_goals:
             # Goals cannot be achieved without robots
             return float('inf')

        # 6. Calculate estimated cost for each unpainted goal
        for target_tile, target_color in unpainted_goals:
            min_cost_for_this_goal = float('inf')

            # 6a. Find Potential Painting Locations for this target tile
            possible_paint_spots = self.paintable_from.get(target_tile, set())

            # If a goal tile has no precomputed painting spots, it's statically impossible
            if not possible_paint_spots:
                 # This indicates an issue with the domain/problem definition or unreachable goal
                 # print(f"Warning: Goal tile {target_tile} has no adjacent painting spots defined by up/down facts.")
                 return float('inf') # Goal is statically unreachable

            # Flag to track if *any* robot can potentially reach a spot to paint this goal
            achievable_by_any_robot = False

            # 6b. Calculate Minimum Cost Per Robot for this goal
            for r in active_robots:
                robot_loc = robot_locations[r]
                # Get the robot's current color; needed for color change cost
                robot_col = robot_colors.get(r)

                # A robot needs a color to either paint or change color.
                # If a robot somehow has no color in the state, it can't contribute now.
                if robot_col is None:
                    continue # Skip this robot for this task

                # 6b.ii. Calculate Minimum Movement Cost for this robot to reach a paint spot
                min_dist_to_paint_spot = float('inf')
                robot_can_reach_spot = False # Can *this* robot reach *any* spot?
                for spot in possible_paint_spots:
                    # Look up precomputed shortest path distance
                    # Default to infinity if robot_loc or spot is unknown, or path doesn't exist
                    dist = self.dist.get(robot_loc, {}).get(spot, float('inf'))
                    if dist != float('inf'):
                        min_dist_to_paint_spot = min(min_dist_to_paint_spot, dist)
                        robot_can_reach_spot = True # Found at least one reachable spot

                # If this specific robot cannot reach any suitable painting spot
                if not robot_can_reach_spot:
                    continue # Try the next robot

                # If we reach here, this robot *can* reach a spot. Mark goal as potentially achievable.
                achievable_by_any_robot = True

                # 6b.iii. Calculate Color Change Cost
                color_cost = 0 if robot_col == target_color else 1

                # 6b.iv. Calculate Painting Action Cost
                paint_action_cost = 1

                # 6b.v. Total Estimated Cost for this Robot to achieve this goal
                move_cost = min_dist_to_paint_spot # Shortest distance found
                cost_r = move_cost + color_cost + paint_action_cost

                # 6b.vi. Update Minimum Cost for this Goal (across all robots)
                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_r)

            # 6c. Check Goal Reachability after checking all robots
            # If no robot could reach a painting spot for this specific goal
            if not achievable_by_any_robot:
                 # This means the goal is currently unreachable by any active robot
                 # print(f"Warning: Goal ({target_tile}, {target_color}) is unreachable by any active robot in the current state.")
                 return float('inf') # State leads to a dead end for this goal

            # If min_cost is still infinity here, it might mean all reachable robots
            # lacked color (if possible), or another unexpected issue. Treat as unreachable.
            if min_cost_for_this_goal == float('inf'):
                 return float('inf')

            # 6d. Accumulate the minimum cost found for this goal
            total_heuristic_value += min_cost_for_this_goal

        # 7. Return Total Heuristic Value
        # Ensure infinity propagates correctly if encountered
        if total_heuristic_value == float('inf'):
            return float('inf')

        # Return the final sum as an integer estimate
        # Use round() before int() to handle potential floating point inaccuracies if any step used floats.
        # Although all costs here are integers, this is safer.
        return int(round(total_heuristic_value))
