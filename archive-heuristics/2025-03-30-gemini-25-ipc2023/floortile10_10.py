import itertools
from collections import deque
from fnmatch import fnmatch
import math # Import math for infinity

# Try to import the base class provided by the planner framework
try:
    from heuristics.heuristic_base import Heuristic 
except ImportError:
    # Define a dummy base class if the import fails 
    # (e.g., for standalone testing or different environments)
    class Heuristic:
        """Placeholder base class for heuristics."""
        def __init__(self, task):
            """Initialize heuristic with task information."""
            self.task = task
        def __call__(self, node):
            """Evaluate the heuristic for a given node."""
            raise NotImplementedError("Heuristic evaluation not implemented.")

# Helper functions for parsing PDDL facts more robustly
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Expects facts in the format '(predicate arg1 arg2 ...)'.
    Returns a list of strings: [predicate, arg1, arg2, ...].
    Returns an empty list if the input is not a valid fact string.
    """
    if isinstance(fact, str) and len(fact) > 1 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Return empty list for non-conforming strings or non-strings

def fact_match(fact_parts, *args):
    """
    Checks if a list of fact parts matches a pattern.
    Wildcards (*) are allowed in the pattern arguments.
    
    Args:
        fact_parts (list): The list of strings from get_parts().
        *args: A sequence of strings representing the pattern (e.g., "predicate", "*", "object").
        
    Returns:
        bool: True if the parts match the pattern, False otherwise.
    """
    if len(fact_parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Floortile domain.

    # Summary
    Estimates the cost to reach the goal state by summing the estimated costs
    for achieving each individual unsatisfied goal predicate `(painted ?tile ?color)`.
    The cost for a single goal includes the paint action (cost 1), potential cost for a robot
    to move away from the target tile if it's occupied (cost 1), the minimum cost for any robot 
    to move to an adjacent position suitable for painting, and the cost for that robot 
    to change to the required color (cost 1 if needed).

    # Assumptions
    - The grid structure (up, down, left, right predicates) is static.
    - All actions defined in the domain have a uniform cost of 1.
    - The heuristic ignores potential negative interactions between robots (e.g., blocking paths,
      resource contention for colors). It assumes goals can be achieved somewhat independently.
    - Movement cost is calculated using the shortest path distance on the static grid graph, 
      ignoring the dynamic `clear` status of tiles. This is a relaxation necessary for 
      efficient computation.
    - If a target tile (that needs painting) is currently occupied by a robot, an 
      estimated cost of 1 is added, assuming the occupying robot must move away. This is a simplification.
    - The input PDDL problems are assumed to be solvable, and the static connections allow
      reaching the necessary locations to paint the goal tiles.

    # Heuristic Initialization
    - Parses the task's goal conditions to identify target `(painted ?tile ?color)` states.
    - Extracts all relevant objects (robots, tiles, colors) from the task definition 
      (static facts, initial state, goals).
    - Builds an adjacency list representing the static grid connectivity based on `up`, `down`, 
      `left`, `right` predicates, treating it as an undirected graph for movement.
    - Precomputes all-pairs shortest path distances between all tiles on this static grid 
      using Breadth-First Search (BFS). Stores these distances for efficient lookup.
    - Determines, for each tile, the set of adjacent tiles a robot must occupy to be able
      to paint it (based on `up` and `down` predicates relevant to `paint_up`/`paint_down` actions). 
      Stores this in `paint_adj`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse Current State: For the given state node, extract the current location of each robot 
       (`robot_loc`), the color each robot holds (`robot_col`), the set of currently painted tiles 
       and their colors (`painted`), and the set of tiles currently occupied by robots (`occupied_tiles`).
    2. Identify Unsatisfied Goals: Compare the `painted` tiles in the current state with the 
       `goal_dict` (derived during initialization). Count the number of goals not yet met.
    3. Check for Goal State: If all goals are satisfied, the heuristic value is 0.
    4. Initialize Total Cost: Set the heuristic estimate `h = 0`.
    5. Iterate Through Unsatisfied Goals: For each goal `(painted target_tile target_color)` that is not satisfied:
        a. Calculate Base Goal Cost: Initialize `goal_cost = 1` (representing the mandatory `paint` action).
        b. Add Occupancy Cost: If the `target_tile` is in the `occupied_tiles` set, increment `goal_cost` by 1.
        c. Find Minimum Robot Cost to Paint: Determine the minimum cost for *any* available robot to perform this specific painting task.
           i. Identify Required Painting Locations: Get the set `adj_tiles` from `paint_adj[target_tile]`. These are the locations a robot needs to be at. If this set is empty, the goal is statically impossible (return a large value).
           ii. Check Robot Availability: Ensure there are robots defined in the problem. If not, the goal is impossible (return large value).
           iii. Iterate Through Robots: For each robot `r`:
               - Verify Robot State: Check if the current state provides location (`r_loc`) and color (`r_col`) for this robot. Skip if incomplete.
               - Calculate Minimum Move Cost: Find the minimum shortest path distance from `r_loc` to any tile in `adj_tiles` using the precomputed `dist` table. Handle unreachable cases (distance is infinity). Let this be `move_cost`.
               - Calculate Color Change Cost: If `r_col` is not the `target_color`, set `color_cost = 1`, otherwise `color_cost = 0`.
               - Calculate Total Robot Cost: `robot_cost = move_cost + color_cost`. This might be infinity if `move_cost` is infinity.
           iv. Determine Minimum Cost Across Robots: Find the minimum `robot_cost` among all valid robots considered. Let this be `min_robot_cost`.
        d. Aggregate Goal Cost: If `min_robot_cost` is infinity (meaning no robot can perform the task due to reachability or color issues), return a large finite value (e.g., 1,000,000) for the heuristic, signaling a likely dead end. Otherwise, add the calculated `goal_cost` plus the integer value of `min_robot_cost` to the total heuristic estimate `h`.
    6. Return Total Heuristic Value: Return the final calculated integer value `h`.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class if necessary
        self.goals = task.goals
        static_facts = task.static
        # Combine initial state and static facts for robust object extraction
        all_known_facts = task.initial_state.union(static_facts)

        # Extract objects (robots, tiles, colors)
        self.robots = set()
        self.tiles = set()
        self.colors = set() 
        
        adj_relations = {"up", "down", "left", "right"}

        for fact in all_known_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed or non-string facts

            pred = parts[0]
            # Use fact_match for predicate checking and argument count validation
            if fact_match(parts, "robot-at", "*", "*"): 
                self.robots.add(parts[1])
                self.tiles.add(parts[2])
            elif fact_match(parts, "clear", "*"): 
                self.tiles.add(parts[1])
            elif fact_match(parts, "painted", "*", "*"): 
                self.tiles.add(parts[1])
                self.colors.add(parts[2])
            elif fact_match(parts, "robot-has", "*", "*"): 
                 self.robots.add(parts[1])
                 self.colors.add(parts[2])
            elif fact_match(parts, "available-color", "*"): 
                 self.colors.add(parts[1])
            elif pred in adj_relations and len(parts) == 3: # Check predicate is known relation
                 self.tiles.add(parts[1])
                 self.tiles.add(parts[2])

        # Add any objects mentioned only in goals (if not already found)
        for goal in self.goals:
             parts = get_parts(goal)
             if fact_match(parts, "painted", "*", "*"):
                 self.tiles.add(parts[1]) # Ensure tile exists
                 self.colors.add(parts[2]) # Ensure color exists
        
        # Build adjacency list for movement (undirected graph)
        self.adj = {tile: set() for tile in self.tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred in adj_relations and len(parts) == 3:
                t1, t2 = parts[1], parts[2]
                # Add edges only if both tiles are known
                if t1 in self.tiles and t2 in self.tiles:
                    self.adj[t1].add(t2)
                    self.adj[t2].add(t1)

        # Build paint adjacency: target_tile -> {locations robot must be at to paint it}
        self.paint_adj = {tile: set() for tile in self.tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            # paint_up(r, y, x, c) requires (up y x) -> robot at x paints y
            # paint_down(r, y, x, c) requires (down y x) -> robot at x paints y
            if fact_match(parts, "up", "*", "*"): # (up target_tile adjacent_tile)
                target_tile, adjacent_tile = parts[1], parts[2]
                if target_tile in self.paint_adj and adjacent_tile in self.tiles:
                    self.paint_adj[target_tile].add(adjacent_tile)
            elif fact_match(parts, "down", "*", "*"): # (down target_tile adjacent_tile)
                target_tile, adjacent_tile = parts[1], parts[2]
                if target_tile in self.paint_adj and adjacent_tile in self.tiles:
                    self.paint_adj[target_tile].add(adjacent_tile)

        # Precompute all-pairs shortest paths using BFS
        self.dist = {}
        # If no tiles were found, self.dist remains empty. BFS handles start_node not in nodes.
        for start_tile in self.tiles:
             self.dist[start_tile] = self._bfs(start_tile, self.adj, self.tiles)

        # Store goal dictionary: {tile: color}
        self.goal_dict = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if fact_match(parts, "painted", "*", "*"):
                # Ensure the tile mentioned in the goal is known
                if parts[1] in self.tiles:
                    self.goal_dict[parts[1]] = parts[2]
                # else: Goal mentions an unknown tile - problem might be malformed.

    def _bfs(self, start_node, adj, nodes):
        """
        Performs Breadth-First Search to find shortest path distances from start_node.
        
        Args:
            start_node: The starting tile name.
            adj (dict): The adjacency list for the grid graph.
            nodes (set): A set of all valid tile names.
            
        Returns:
            dict: A dictionary mapping tile names to their shortest distance (integer) 
                  from start_node, or math.inf if unreachable.
        """
        # Initialize distances to infinity for all known nodes
        distances = {node: math.inf for node in nodes}
        
        # If start_node isn't a valid node (e.g., isolated tile not in adj keys), return infinities
        if start_node not in distances:
             return distances 

        distances[start_node] = 0
        queue = deque([start_node])
        # Keep track of nodes added to the queue to avoid redundant processing
        processed = {start_node} 

        while queue:
            current_node = queue.popleft()
            # Use adj.get(node, set()) for safety in case a node has no neighbors listed
            for neighbor in adj.get(current_node, set()):
                # Process neighbor only if it's a valid node and hasn't been reached yet
                if neighbor in distances and neighbor not in processed:
                    distances[neighbor] = distances[current_node] + 1
                    processed.add(neighbor)
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.

        Args:
            node: A state node object containing the current state (node.state).
                  node.state is expected to be a frozenset of fact strings.

        Returns:
            int: The estimated cost (number of actions) to reach a goal state.
                 Returns 0 for a goal state. Returns a large integer (1,000,000) 
                 if the state appears to be a dead end based on the heuristic's logic.
        """
        state = node.state

        # Parse current state efficiently
        robot_loc = {}
        robot_col = {}
        painted = {}
        occupied_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip potential non-facts or malformed strings

            pred = parts[0]
            # Use fact_match for robust checking
            if fact_match(parts, "robot-at", "*", "*"): 
                robot, tile = parts[1], parts[2]
                robot_loc[robot] = tile
                occupied_tiles.add(tile)
            elif fact_match(parts, "robot-has", "*", "*"): 
                robot, color = parts[1], parts[2]
                robot_col[robot] = color
            elif fact_match(parts, "painted", "*", "*"): 
                tile, color = parts[1], parts[2]
                painted[tile] = color

        h = 0 # Use integer for heuristic value, as all action costs are 1

        # Iterate through goals stored during initialization
        unsatisfied_goal_count = 0
        for target_tile, target_color in self.goal_dict.items():
            # Check if this goal is satisfied in the current state
            if painted.get(target_tile) == target_color:
                continue

            # --- Goal is unsatisfied ---
            unsatisfied_goal_count += 1
            goal_cost = 1  # Base cost for the 'paint' action

            # Add cost if the target tile itself is occupied
            if target_tile in occupied_tiles:
                goal_cost += 1

            # Find the minimum cost for *any* robot to achieve this specific goal
            min_robot_total_cost = math.inf
            
            # Get the locations from where this tile can be painted
            possible_paint_locations = self.paint_adj.get(target_tile, set())

            # If statically impossible to paint this tile, return large value
            if not possible_paint_locations:
                return 1_000_000 
            
            # If there are no robots defined, goals cannot be achieved
            if not self.robots:
                 return 1_000_000

            # Check costs for each robot
            valid_robot_found_for_goal = False
            for robot in self.robots:
                # Ensure robot's state (location, color) is known in the current state
                if robot not in robot_loc or robot not in robot_col:
                     continue # Skip robots with incomplete state info in this node

                valid_robot_found_for_goal = True
                r_loc = robot_loc[robot]
                r_col = robot_col[robot]

                # Calculate min move cost from robot's current location to any required adjacent tile
                min_move_cost = math.inf
                # Check if robot's location is valid and has precomputed distance info
                if r_loc in self.dist:
                    current_robot_distances = self.dist[r_loc]
                    for adj_tile in possible_paint_locations:
                        # Use precomputed distance, default to infinity if adj_tile not reachable
                        distance = current_robot_distances.get(adj_tile, math.inf)
                        min_move_cost = min(min_move_cost, distance)
                # else: Robot location not found in distance map (e.g., isolated tile), move_cost remains inf

                # Calculate color change cost (1 if needed, 0 otherwise)
                color_cost = 0 if r_col == target_color else 1

                # Total cost for this robot (can be inf if move_cost is inf)
                # Add costs only if move_cost is finite
                if min_move_cost != math.inf:
                    robot_total_cost = min_move_cost + color_cost
                    min_robot_total_cost = min(min_robot_total_cost, robot_total_cost)
                # else: This robot cannot reach any suitable painting spot, cost remains inf

            # If no robot had complete state info for this goal evaluation
            if not valid_robot_found_for_goal and self.robots:
                 # This implies robots exist but their state isn't fully defined in 'state'
                 # Treat as potentially unreachable for safety.
                 return 1_000_000 

            # If no robot can reach/paint this tile (all costs were inf)
            if min_robot_total_cost == math.inf:
                # Goal seems unreachable by any robot from current state
                return 1_000_000 # Indicate likely dead end

            # Add the integer cost for this goal to the total heuristic value
            # min_robot_total_cost should be finite here
            h += goal_cost + int(min_robot_total_cost) 

        # If h is 0 and we started with unsatisfied goals, something is wrong.
        # However, the logic ensures h=0 only if unsatisfied_goal_count is 0 initially.
        return h

