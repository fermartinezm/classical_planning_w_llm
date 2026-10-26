import collections
import itertools # Not used directly, but potentially useful for graph algorithms

# Try to import the base class, provide a dummy if not found
try:
    # Assuming the heuristic base class is available in the environment
    # where this code will be executed.
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    # (e.g., for standalone testing or if the import path differs)
    class Heuristic:
        """Dummy base class for Heuristic if import fails."""
        def __init__(self, task):
            """Dummy initializer."""
            pass
        def __call__(self, node):
            """Dummy call method."""
            raise NotImplementedError("Heuristic base class not found.")

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: '(predicate arg1 arg2)' -> ['predicate', 'arg1', 'arg2']
    Removes leading/trailing whitespace and parentheses before splitting.
    """
    # Remove parentheses and split by space
    return fact.strip()[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban problems.
    It calculates the sum of the shortest path distances for each box from its current
    location to its target goal location (estimating the minimum number of pushes).
    It adds an estimate of the number of moves the robot needs to make to reach the
    closest box that is not yet in its goal location. The heuristic incorporates
    static dead-end detection: if any box is located in a square from which its goal
    is determined to be permanently unreachable (based on map geometry), the state
    is assigned an infinite cost, effectively pruning this path in the search.

    # Assumptions
    - The cost of both 'move' and 'push' actions is 1.
    - The heuristic prioritizes minimizing the total number of pushes and the
      initial robot positioning moves required to start pushing.
    - Dynamic obstacles (other boxes blocking paths) are ignored for computational
      efficiency; only the static map layout is considered for distances and dead ends.
    - Dead ends are identified based on a static analysis of the map: locations from
      which no sequence of pushes can lead to a goal square.

    # Heuristic Initialization (`__init__`)
    1.  Stores the task definition, including goals (`task.goals`) and static facts (`task.static`).
    2.  Parses static `adjacent` facts (`task.static`) to build graph representations:
        a. `self.adj`: An undirected adjacency list representing navigable space for calculating
           shortest path distances (ignoring push mechanics).
        b. `self.adj_directed`: A directed adjacency dictionary (`{from: {to: direction}}`)
           capturing the specific directions of movement between adjacent locations, used
           for simulating push possibilities during dead-end analysis.
        c. `self.locations`: A set of all unique location identifiers found in the map.
    3.  Computes All-Pairs Shortest Paths (APSP) using Breadth-First Search (BFS) starting
        from each location on the undirected graph (`self.adj`). Distances are stored in
        `self.distances[loc1][loc2]`. Unreachable pairs have a distance of `float('inf')`.
    4.  Extracts the target goal location for each box from `task.goals` into the
        `self.goal_locs` dictionary (`{box_name: goal_location}`).
    5.  Computes dead squares (`self.dead_squares`):
        a. Initializes a set `live_squares` containing all unique goal locations found on the map.
        b. Performs a backward reachability analysis starting from these `live_squares`:
           - Uses a queue to explore squares from which a box could be pushed *into* a known `live` square.
           - For a potential push from `l_box_start` to `l_box_dest` (where `l_box_dest` is live),
             it checks if a valid robot position `l_robot` exists that could execute this push
             (i.e., `adjacent(l_robot, l_box_start, push_direction)` must be possible according to map geometry).
           - If the push geometry is valid and `l_box_start` has not yet been marked as `live`,
             it is added to `live_squares` and the queue.
        c. Any location in `self.locations` that is not in the final `live_squares` set is
           considered a dead square.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  Retrieves the current state's facts (`node.state`).
    2.  Parses the state to find the robot's current location (`robot_loc`) and the location
        of each box (`box_locs = {box: location}`). Includes basic error checking for invalid states
        (e.g., missing robot, entities outside the known map).
    3.  Dead-End Check: Iterates through each box `b` at location `loc`. If `loc` is found in
        `self.dead_squares` AND `loc` is not the specific goal location for box `b`
        (i.e., `self.goal_locs.get(b) != loc`), the state is considered unsolvable from this
        point, and the function returns `float('inf')`.
    4.  Calculate Aggregate Box Distances and Minimum Robot Distance:
        - Initializes `box_dist_sum = 0` (to sum estimated pushes).
        - Initializes `min_robot_to_box_dist = float('inf')` (to find distance to nearest misplaced box).
        - Sets `misplaced_boxes_exist = False`.
        - Iterates through each box `b` at `current_loc`:
            - Retrieves its `goal_loc`.
            - If a `goal_loc` is defined for this box and `current_loc != goal_loc`:
                - Sets `misplaced_boxes_exist = True`.
                - Looks up the shortest path distance `dist = self.distances[current_loc].get(goal_loc, float('inf'))`.
                  If `dist` is `inf`, the goal is statically unreachable; returns `inf`.
                - Adds `dist` to `box_dist_sum`.
                - Looks up the shortest path distance from the robot: `robot_dist = self.distances[robot_loc].get(current_loc, float('inf'))`.
                - If `robot_dist` is finite, updates `min_robot_to_box_dist = min(min_robot_to_box_dist, robot_dist)`.
    5.  Goal State Check: If `misplaced_boxes_exist` remains `False` after checking all boxes,
        the state satisfies all box goals; returns 0.
    6.  Robot Reachability Check: If `misplaced_boxes_exist` is `True` but `min_robot_to_box_dist`
        is still `inf`, it means the robot cannot reach any of the boxes that need to be moved;
        returns `inf`.
    7.  Combine Costs and Return Value:
        - Calculates the estimated robot movement cost: `robot_move_estimate = max(0, min_robot_to_box_dist - 1)`.
          (Subtracts 1 because the robot needs to be adjacent, not on the box's square, to push).
        - The final heuristic value is `box_dist_sum + robot_move_estimate`.
        - Returns this value, ensuring it's non-negative.
    """

    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # --- 1. Build adjacency structures and collect locations ---
        self.adj = collections.defaultdict(list) # Undirected graph for APSP
        self.adj_directed = collections.defaultdict(dict) # Directed graph {from: {to: direction}}
        self.locations = set()
        added_edges = set() # Track edges for undirected graph to avoid duplicates

        for fact in static_facts:
            try:
                parts = get_parts(fact)
                if parts[0] == 'adjacent' and len(parts) == 4:
                    l1, l2, direction = parts[1], parts[2], parts[3]
                    self.locations.add(l1)
                    self.locations.add(l2)

                    # Add to undirected graph if not already present
                    edge = tuple(sorted((l1, l2)))
                    if edge not in added_edges:
                        self.adj[l1].append(l2)
                        self.adj[l2].append(l1)
                        added_edges.add(edge)

                    # Add to directed graph
                    self.adj_directed[l1][l2] = direction
            except IndexError:
                # Handle potential errors if a fact doesn't have expected parts
                print(f"Warning: Malformed static fact ignored during parsing: {fact}")
                continue

        # --- 2. Compute All-Pairs Shortest Paths (APSP) ---
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))
        for start_node in self.locations:
            # Only run BFS from nodes that actually have connections
            if start_node not in self.adj: continue

            self.distances[start_node][start_node] = 0
            queue = collections.deque([(start_node, 0)])
            visited = {start_node} # Keep track of visited nodes for this BFS run

            while queue:
                current_node, dist = queue.popleft()
                # Explore neighbors using the undirected graph for path distance
                for neighbor in self.adj[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        # --- 3. Extract goal locations for boxes ---
        self.goal_locs = {}
        for goal_fact in self.goals:
             try:
                 parts = get_parts(goal_fact)
                 # Expect goals of the form (at <box> <location>)
                 if parts[0] == 'at' and len(parts) == 3:
                     box, loc = parts[1], parts[2]
                     # Basic validation: Check if the goal location exists on the map
                     if loc not in self.locations:
                          print(f"Warning: Goal location '{loc}' for box '{box}' is not found in the map's locations derived from 'adjacent' facts. This goal might be unreachable.")
                     self.goal_locs[box] = loc
                 # Extend here if other types of goals need handling
             except IndexError:
                 print(f"Warning: Malformed goal fact ignored: {goal_fact}")
                 continue

        # --- 4. Compute dead squares using backward reachability ---
        # Initialize live squares with all unique goal locations present on the map
        goal_squares = set(loc for loc in self.goal_locs.values() if loc in self.locations)
        live_squares = set(goal_squares)
        queue = collections.deque(list(live_squares))

        # Precompute reverse directed adjacency: {to: {from: direction}}
        # This helps find where a push *into* l_box_dest could have originated
        reverse_adj_directed = collections.defaultdict(dict)
        for l_from, targets in self.adj_directed.items():
            for l_to, direction in targets.items():
                reverse_adj_directed[l_to][l_from] = direction

        while queue:
            l_box_dest = queue.popleft() # A square known to be reachable (live)

            # Find potential squares l_box_start from which a push could land in l_box_dest
            for l_box_start, push_direction in reverse_adj_directed.get(l_box_dest, {}).items():
                # Ensure l_box_start is a valid location on the map
                if l_box_start not in self.locations: continue

                # Check if a robot could geometrically perform this push:
                # Need to find if there exists an l_robot such that
                # adjacent(l_robot, l_box_start, push_direction) holds.
                robot_pos_exists = False
                # Search all locations 'potential_robot_loc' to see if any can push into l_box_start
                # with the required 'push_direction'.
                for potential_robot_loc, neighbors in self.adj_directed.items():
                     # Check if potential_robot_loc has an edge TO l_box_start with the correct direction
                     if neighbors.get(l_box_start) == push_direction:
                          robot_pos_exists = True
                          break # Found a valid geometric position for the robot

                # If the push geometry is valid and l_box_start wasn't previously known to be live:
                if robot_pos_exists and l_box_start not in live_squares:
                    live_squares.add(l_box_start)
                    queue.append(l_box_start)

        # Dead squares are all map locations that are not live
        self.dead_squares = self.locations - live_squares


    def __call__(self, node):
        state = node.state

        # --- 1. Parse state ---
        robot_loc = None
        box_locs = {}
        try:
            for fact in state:
                parts = get_parts(fact)
                predicate = parts[0]
                # Basic validation of parts length helps catch some errors
                if predicate == 'at-robot' and len(parts) == 2:
                    robot_loc = parts[1]
                elif predicate == 'at' and len(parts) == 3:
                    # Assumes the domain ensures the second element is a box
                    box_locs[parts[1]] = parts[2]
                # 'clear' facts are ignored by this heuristic
        except IndexError:
             # Error parsing state facts, likely indicates an issue
             return float('inf')

        # Validate robot location - must exist and be on the map
        if robot_loc is None or robot_loc not in self.locations:
             return float('inf')

        # --- 2. Dead-End Check ---
        for box, loc in box_locs.items():
             # Validate box location
             if loc not in self.locations:
                  return float('inf') # Box is somehow off the map

             if loc in self.dead_squares:
                # If box is in a dead square, it must be its goal location, otherwise unsolvable.
                # Check against the specific goal for *this* box.
                if self.goal_locs.get(box) != loc:
                     return float('inf')

        # --- 3. Calculate Heuristic Value ---
        box_dist_sum = 0
        min_robot_to_box_dist = float('inf')
        misplaced_boxes_exist = False

        # Iterate through all boxes present in the current state
        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locs.get(box) # Get the goal for this specific box

            # Only consider boxes that have a defined goal in the problem
            if goal_loc is not None:
                if current_loc != goal_loc:
                    misplaced_boxes_exist = True

                    # Get box-to-goal distance from precomputed APSP
                    dist = self.distances[current_loc].get(goal_loc, float('inf'))
                    if dist == float('inf'):
                        # Should be caught by dead-end check if goal is in dead area,
                        # but this catches cases where goal is simply unreachable (e.g., disconnected map component)
                        return float('inf')
                    box_dist_sum += dist

                    # Get robot-to-current-box distance from precomputed APSP
                    robot_dist = self.distances[robot_loc].get(current_loc, float('inf'))
                    # Update minimum distance if robot can reach this misplaced box
                    if robot_dist != float('inf'):
                        min_robot_to_box_dist = min(min_robot_to_box_dist, robot_dist)

        # --- 4. Final Heuristic Value Calculation ---
        if not misplaced_boxes_exist:
            # All boxes with defined goals are in their target locations - Goal Reached!
            return 0

        # If there are misplaced boxes, check if the robot can reach at least one
        if min_robot_to_box_dist == float('inf'):
             # Robot cannot reach any of the boxes that need moving from its current position.
             # This state is likely a dead end for the robot.
             return float('inf')

        # Combine costs: sum of box pushes + estimated robot moves to nearest box
        # Subtract 1 from robot distance estimate as robot needs to be adjacent.
        robot_move_estimate = max(0, min_robot_to_box_dist - 1)
        heuristic_value = box_dist_sum + robot_move_estimate

        # Return the final non-negative heuristic value
        return heuristic_value
