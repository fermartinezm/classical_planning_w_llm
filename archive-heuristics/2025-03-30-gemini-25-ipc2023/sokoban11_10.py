import itertools
from collections import deque
from heuristics.heuristic_base import Heuristic
import heapq # Not strictly needed for BFS, but useful if considering Dijkstra later

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact string (e.g., "(pred obj1 obj2)")."""
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban.
    It primarily calculates the sum of the shortest path distances for each box
    from its current location to its designated goal location. To make the
    heuristic more informative, it adds the shortest path distance for the robot
    to reach the location of the nearest misplaced box. The heuristic also includes
    basic deadlock detection, identifying boxes stuck in simple "corner" locations
    that are not goal squares, returning infinity for such states.

    # Assumptions
    - The grid structure (connectivity between locations) is static and defined
      by the 'adjacent' predicates in the initial state/problem file.
    - Each box that appears in the goal definition has a unique target location.
    - The cost of both the 'move' action (robot moves) and the 'push' action
      (robot pushes box) is 1.
    - The 'adjacent' predicates define a grid where movement is possible between
      adjacent cells, and the absence of an 'adjacent' predicate implies a wall
      or boundary for that direction.
    - Box names typically start with 'box', used for parsing state facts.

    # Heuristic Initialization
    - Parses static 'adjacent' facts to build an undirected graph representation
      of the accessible grid locations for pathfinding (BFS).
    - Parses static 'adjacent' facts to build a directed adjacency mapping
      `{(location, direction): neighbor_location}` for deadlock detection.
    - Computes all-pairs shortest paths (APSP) between all reachable locations
      using Breadth-First Search (BFS). Distances are stored.
    - Identifies the specific goal location for each box from the task's goal
      conditions.
    - Identifies simple "corner" deadlock locations: non-goal squares from which
      a box cannot be pushed out in two orthogonal directions (e.g., blocked UP and LEFT).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** In the `__call__` method, iterate through the facts
        in the current state (`node.state`) to find the robot's current location
        (`at-robot`) and the location of each box (`at`).
    2.  **Check for Deadlocks:** Check if any box is currently located in one of the
        pre-calculated `dead_squares`. If so, the goal is considered unreachable
        from this state, and the heuristic returns `float('inf')`.
    3.  **Calculate Box-to-Goal Distances:**
        a. Initialize `box_heuristic = 0`.
        b. Iterate through each box found in the current state.
        c. Retrieve its designated goal location using the pre-computed `goal_map`.
        d. If the box is not currently at its goal location:
            i. Mark that at least one box is misplaced (`misplaced_boxes_exist = True`).
            ii. Look up the shortest path distance from the box's current location (`bloc`)
               to its goal location (`gloc`) using the pre-computed APSP table (`self.dist`).
            iii. If the goal location is unreachable from the box's current location
                 (distance is infinity or not found), return `float('inf')`.
            iv. Add this distance to `box_heuristic`.
    4.  **Calculate Minimum Robot-to-Box Distance:**
        a. Initialize `min_robot_to_box_dist = float('inf')`.
        b. During the loop for misplaced boxes (step 3d), also look up the shortest
           path distance from the robot's current location (`robot_loc`) to the box's
           current location (`bloc`) using the APSP table.
        c. Update `min_robot_to_box_dist` to be the minimum of its current value and
           the newly calculated distance. Handle cases where the robot might not be
           able to reach a box (distance infinity).
    5.  **Combine Heuristic Components:**
        a. If no boxes were misplaced (`misplaced_boxes_exist` is False), the state
           is a goal state, return 0.
        b. If boxes are misplaced, but the robot cannot reach *any* of them
           (`min_robot_to_box_dist` remains infinity), return `float('inf')` as the
           state seems unsolvable or requires complex maneuvers not captured.
        c. Otherwise, the final heuristic value is `box_heuristic + min_robot_to_box_dist`.
    6.  **Ensure Non-Negativity:** Although distances should be non-negative, return
        `max(0, final_heuristic)` as a safeguard.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information, computing distances,
        and identifying deadlocks.
        """
        self.task = task
        # Undirected adjacency list for BFS pathfinding: loc -> [neighbor1, neighbor2,...]
        self.adj = {}
        # Set of all location objects defined in the problem
        self.locations = set()
        # Mapping from box name to its goal location: box -> goal_location
        self.goal_map = {}
        # Set of locations that are goal squares for any box
        self.goal_locs_set = set()
        # All-pairs shortest paths distances: loc1 -> {loc2: distance}
        self.dist = {}
        # Set of non-goal locations identified as simple corner deadlocks
        self.dead_squares = set()
        # Directed adjacency mapping: (loc, direction) -> neighbor_loc
        self.directed_adj = {}

        # --- Parse static facts (like 'adjacent') and goal conditions ---
        self._parse_static_info()
        self._parse_goals()

        # --- Compute all-pairs shortest paths on the grid ---
        self._compute_apsp()

        # --- Identify simple deadlock squares ---
        self._identify_dead_squares()


    def _parse_static_info(self):
        """Builds adjacency lists and identifies all locations from static facts."""
        for fact in self.task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            # Process 'adjacent' facts for connectivity
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]

                # Add locations to the set of all known locations
                self.locations.add(loc1)
                self.locations.add(loc2)

                # Build undirected adjacency list for BFS pathfinding
                # Assumes if loc1 is adjacent to loc2, loc2 is also adjacent to loc1
                # This is typical for Sokoban grids.
                self.adj.setdefault(loc1, []).append(loc2)
                self.adj.setdefault(loc2, []).append(loc1)

                # Store directed adjacency for deadlock checks using the exact PDDL fact
                self.directed_adj[(loc1, direction)] = loc2

            # Potentially parse other static info if needed in the future


    def _parse_goals(self):
        """Extracts box goal locations from the task's goal specification."""
        for goal in self.task.goals:
            parts = get_parts(goal)
            # Check if it's an 'at' predicate for a box
            # Assuming box names contain 'box' or checking type if available
            # A safer check might rely on knowing box object names from the :objects list
            if parts[0] == 'at' and len(parts) == 3 and 'box' in parts[1]: # Simple check
                box, loc = parts[1], parts[2]
                self.goal_map[box] = loc
                self.goal_locs_set.add(loc)


    def _compute_apsp(self):
        """Computes all-pairs shortest paths using BFS starting from each location."""
        for start_node in self.locations:
            # Initialize distances from start_node
            self.dist[start_node] = {start_node: 0}
            queue = deque([start_node])
            visited = {start_node} # Keep track of visited nodes for this BFS run

            while queue:
                current_node = queue.popleft()
                current_dist = self.dist[start_node][current_node]

                # Explore neighbors using the undirected adjacency list
                if current_node in self.adj:
                    for neighbor in self.adj[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.dist[start_node][neighbor] = current_dist + 1
                            queue.append(neighbor)


    def _identify_dead_squares(self):
        """Identifies simple corner deadlock locations (non-goal squares)."""
        # Define the standard directions used in the PDDL file
        directions = ['up', 'down', 'left', 'right']
        # Define pairs of orthogonal directions to check for corners
        orthogonal_pairs = [('up', 'left'), ('up', 'right'),
                            ('down', 'left'), ('down', 'right')]

        for loc in self.locations:
            # Skip if the location is a goal square for any box
            if loc in self.goal_locs_set:
                continue

            # Check if there's an adjacent location in each direction
            neighbor_exists = {d: (loc, d) in self.directed_adj for d in directions}

            is_dead = False
            # A location is a simple corner deadlock if it lacks neighbors
            # in two orthogonal directions.
            for d1, d2 in orthogonal_pairs:
                if not neighbor_exists[d1] and not neighbor_exists[d2]:
                    is_dead = True
                    break # Found a deadlock condition

            if is_dead:
                self.dead_squares.add(loc)


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state
        robot_loc = None
        box_locs = {} # Current locations of boxes: box_name -> location

        # --- Parse current state to find robot and box locations ---
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and 'box' in parts[1]: # Simple check
                box_locs[parts[1]] = parts[2]

        # If robot location is not found, state is invalid or unusual
        if robot_loc is None:
             return float('inf')

        # --- Check for immediate deadlocks ---
        for box, bloc in box_locs.items():
            if bloc in self.dead_squares:
                # If any box is in a deadlock square, the goal is unreachable
                return float('inf')

        # --- Calculate heuristic components ---
        box_heuristic = 0
        min_robot_to_box_dist = float('inf')
        misplaced_boxes_exist = False

        for box, bloc in box_locs.items():
            gloc = self.goal_map.get(box)

            # If this box doesn't have a defined goal, skip it
            if gloc is None:
                 continue

            if bloc != gloc:
                misplaced_boxes_exist = True

                # --- Calculate distance for the box to reach its goal ---
                # Check if distance information exists for this box's location
                if bloc not in self.dist:
                     # Box is in a location not reachable in the graph? Problematic.
                     return float('inf')
                # Check if the goal location is reachable from the box's current location
                box_dist_to_goal = self.dist[bloc].get(gloc, float('inf'))

                if box_dist_to_goal == float('inf'):
                    # Goal is unreachable for this box from its current position
                    return float('inf')

                box_heuristic += box_dist_to_goal

                # --- Calculate distance for the robot to reach this box ---
                # Check if distance information exists for the robot's location
                if robot_loc not in self.dist:
                     # Robot is in an unreachable location? Problematic.
                     # We might still calculate box heuristic but robot part will be inf.
                     robot_dist_to_box = float('inf')
                else:
                     # Check if the box's location is reachable from the robot's location
                     robot_dist_to_box = self.dist[robot_loc].get(bloc, float('inf'))

                # Update the minimum distance from robot to any misplaced box
                min_robot_to_box_dist = min(min_robot_to_box_dist, robot_dist_to_box)


        # --- Determine final heuristic value ---
        # If no boxes are misplaced, we are at the goal state
        if not misplaced_boxes_exist:
            return 0

        # If boxes are misplaced, but the robot cannot reach any of them
        if min_robot_to_box_dist == float('inf'):
             # This suggests a potentially difficult or unsolvable state from the robot's perspective
             # Return infinity for greedy search to avoid exploring this path further.
             return float('inf')

        # Combine the box distances and the minimum robot distance
        final_heuristic = box_heuristic + min_robot_to_box_dist

        # Ensure the heuristic value is non-negative
        return max(0, final_heuristic)

