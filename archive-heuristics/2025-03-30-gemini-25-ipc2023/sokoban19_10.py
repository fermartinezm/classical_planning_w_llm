import sys
from collections import deque
from fnmatch import fnmatch
# Assuming the heuristic base class `Heuristic` is available in this path.
# Adjust the import path based on your project structure if necessary.
# e.g., sys.path.append('/path/to/planner')
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space. Handles potential errors.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    Returns an empty list if the format is invalid.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # print(f"Warning: Invalid fact format encountered: {fact}")
    return [] # Return empty list for invalid format


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    Estimates the cost to reach the goal state in Sokoban. The heuristic value
    is primarily based on the sum of shortest path distances for each box from
    its current location to its target goal location. It also includes an
    estimate of the robot's movement cost to start pushing the nearest
    misplaced box. The distances are precomputed considering the static layout
    (walls) but ignoring dynamic obstacles (other boxes, robot). This heuristic
    is designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - Each box (e.g., 'box1', 'box2') has a unique designated goal location
      specified in the problem's goal description using '(at boxN goal_loc)' facts.
    - The cost of moving the robot ('move' action) and pushing a box
      ('push' action) is uniform (assumed to be 1).
    - The underlying grid connectivity is derived solely from 'adjacent' facts.
      Implicit walls exist where no 'adjacent' fact allows movement between locations.
    - Simple deadlocks (like boxes in non-goal corners) are not explicitly
      detected by this version, but the pathfinding respects static walls.

    # Heuristic Initialization (`__init__`)
    - Parses the task's goal conditions (`task.goals`) to identify the target
      location for each box, storing them in `self.goal_locations`. Also collects
      all box names in `self.boxes`.
    - Parses the static facts (`task.static`), specifically the 'adjacent' predicates,
      to build a graph representation of the grid layout. This graph is stored as
      an adjacency list (`self.adj`) where keys are locations and values are lists
      of directly reachable neighboring locations. All unique locations encountered
      are stored in `self.locations`.
    - Precomputes all-pairs shortest path distances between all reachable locations
      using Breadth-First Search (BFS). The distances are calculated based on the
      static grid layout (`self.adj`), ignoring dynamic elements like the current
      positions of the robot or other boxes. These distances are stored in a
      nested dictionary `self.distances[loc1][loc2]`. Unreachable pairs have a
      distance of `float('inf')`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse Current State:** Extract the robot's current location (`robot_loc`)
        and the current location of each box (`box_locations`) from the input `node.state`.
        Simultaneously, count how many boxes (`num_goals_met`) are already at their
        respective goal locations defined in `self.goal_locations`.
    2.  **Check Goal State:** If `num_goals_met` equals the total number of boxes defined
        in the goal, the goal state is reached, and the heuristic returns 0.
    3.  **Validate State:** Check if the robot's location was found and if all expected
        boxes were located in the state. If not (e.g., missing robot or box fact),
        return `float('inf')` as the state is considered invalid or incomplete for
        heuristic calculation.
    4.  **Calculate Box Component:** Initialize `total_heuristic_value = 0`. Iterate
        through each box `b` and its goal location `g_loc` from `self.goal_locations`.
        a. Get the box's current location `c_loc` from `box_locations`.
        b. If `c_loc` is not equal to `g_loc` (the box is misplaced):
           i. Retrieve the precomputed shortest path distance `dist = self.distances[c_loc][g_loc]`.
              Handle potential lookup errors (e.g., if `c_loc` or `g_loc` were somehow
              not in the precomputed map) by defaulting to infinity or returning infinity.
           ii. If `dist` is `float('inf')`, it means this box cannot reach its goal
              based on the static map layout (e.g., separated by walls). Return
              `float('inf')` immediately, as this path is considered a dead end.
           iii. Add `dist` to `total_heuristic_value`.
           iv. Mark that at least one misplaced box was found (`found_misplaced_box = True`).
    5.  **Calculate Robot Component:** Initialize `min_robot_to_misplaced_box_dist = float('inf')`.
        If `found_misplaced_box` is true (meaning there's work to be done):
        a. Retrieve the distance map for the robot's current location: `robot_dist_map = self.distances[robot_loc]`. Handle potential lookup errors.
        b. Iterate through the misplaced boxes identified in the previous step.
        c. For each misplaced box `b` at `c_loc`:
           i. Retrieve the precomputed distance `robot_dist = robot_dist_map[c_loc]`. Handle potential lookup errors.
           ii. Update `min_robot_to_misplaced_box_dist = min(min_robot_to_misplaced_box_dist, robot_dist)`.
    6.  **Combine Components:** If `found_misplaced_box` is true:
        a. Check if `min_robot_to_misplaced_box_dist` is still `float('inf')`. This
           would mean the robot cannot reach *any* of the misplaced boxes based on the
           static map (e.g., robot is trapped). If so, return `float('inf')`.
        b. Otherwise, add `min_robot_to_misplaced_box_dist` to `total_heuristic_value`.
           This represents the estimated cost for the robot to reach the nearest box
           that needs pushing.
    7.  **Return Value:** Return the final `total_heuristic_value`. If no misplaced boxes
        were found, the function should have returned 0 in step 2. The value represents
        the sum of box-to-goal distances plus the distance for the robot to reach the
        nearest misplaced box.
    """

    def __init__(self, task):
        """
        Initializes the heuristic: parses goals, builds grid graph, precomputes distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify Goal Locations and Boxes
        self.goal_locations = {}
        self.boxes = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Expect goal like: (at boxN loc_X_Y)
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc
                self.boxes.add(box)

        # 2. Build Grid Graph & Identify Locations from static 'adjacent' facts
        self.locations = set()
        self.adj = {} # loc -> list of neighbor locs
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2 = parts[1], parts[2]
                # Add locations to the set of all known locations
                self.locations.add(l1)
                self.locations.add(l2)
                # Add edge l1 -> l2 to the adjacency list
                if l1 not in self.adj: self.adj[l1] = []
                # Avoid adding duplicate neighbors if facts are redundant
                if l2 not in self.adj[l1]:
                    self.adj[l1].append(l2)
                # Note: We rely on the PDDL providing adjacent facts for both directions
                # if movement is bidirectional (e.g., (adj a b right), (adj b a left)).

        # Ensure all locations found exist as keys in the adjacency map, even if they have no outgoing edges listed
        for loc in self.locations:
            if loc not in self.adj:
                self.adj[loc] = []

        # 3. Precompute All-Pairs Shortest Path Distances using BFS
        self.distances = {}
        for start_loc in self.locations:
            # Calculate distances from start_loc to all other locations in the grid
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search on the grid graph (self.adj) starting
        from start_node to find shortest path distances to all other locations.
        Returns a dictionary mapping each location to its distance (int or float('inf')).
        """
        # Initialize distances: infinite for all locations, 0 for the start_node
        distances = {loc: float('inf') for loc in self.locations}

        # Check if the start node is valid before proceeding
        if start_node not in self.locations:
             # This indicates an issue, perhaps start_node isn't a valid location
             # print(f"Warning: BFS start node {start_node} not in known locations.")
             return distances # Return map with all infinities

        distances[start_node] = 0
        queue = deque([start_node]) # Queue for BFS traversal

        while queue:
            current_node = queue.popleft()

            # Explore neighbors of the current node using the pre-built adjacency list
            for neighbor in self.adj.get(current_node, []):
                # If we found the first path to this neighbor (distance is inf)
                if distances[neighbor] == float('inf'):
                    # Update distance and add neighbor to the queue
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # 1. Parse Current State: Find robot and box locations
        robot_loc = None
        box_locations = {}
        num_goals_met = 0

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip potential empty lists from invalid facts

            predicate = parts[0]
            # Find robot location
            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            # Find box locations
            elif predicate == 'at' and len(parts) == 3 and parts[1] in self.boxes:
                box, loc = parts[1], parts[2]
                box_locations[box] = loc
                # Check if this box is at its designated goal location
                if box in self.goal_locations and self.goal_locations[box] == loc:
                    num_goals_met += 1

        # 2. Check Goal State: If all boxes are at their goals
        if num_goals_met == len(self.goal_locations):
            return 0

        # 3. Validate State: Ensure robot and all boxes were found
        if robot_loc is None:
             # print("Heuristic Error: Robot location not found in state.")
             return float('inf') # Cannot calculate heuristic without robot position
        if len(box_locations) != len(self.boxes):
            # This implies a mismatch, possibly an invalid state representation
            # print(f"Heuristic Error: Found {len(box_locations)} boxes, expected {len(self.boxes)}.")
            return float('inf') # Treat as invalid state

        # --- Start Heuristic Calculation ---
        total_heuristic_value = 0
        min_robot_to_misplaced_box_dist = float('inf')
        found_misplaced_box = False

        # Use precomputed distances for robot's location, handle if robot loc is invalid
        robot_dist_map = self.distances.get(robot_loc)
        if robot_dist_map is None:
            # print(f"Heuristic Error: Robot location {robot_loc} not found in distance maps.")
            return float('inf') # Robot is in an unknown/unreachable location

        # 4. Calculate Box Component & 5. Prepare Robot Component Calculation
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations[box]

            # Only consider boxes not already at their goal
            if current_loc != goal_loc:
                found_misplaced_box = True # Mark that robot needs to move

                # --- Box distance component ---
                # Get distance map for the box's current location
                current_dist_map = self.distances.get(current_loc)
                if current_dist_map is None:
                    # print(f"Heuristic Error: Box location {current_loc} not found in distance maps.")
                    return float('inf') # Box is in an unknown/unreachable location

                # Get distance from current location to goal location
                dist = current_dist_map.get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # This box cannot reach its goal based on static map analysis
                    return float('inf') # This state leads to a dead end for this box

                total_heuristic_value += dist

                # --- Robot distance component (update minimum) ---
                # Find distance from robot to this misplaced box's current location
                robot_dist_to_box = robot_dist_map.get(current_loc, float('inf'))
                min_robot_to_misplaced_box_dist = min(min_robot_to_misplaced_box_dist, robot_dist_to_box)

        # If no boxes were misplaced, goal check should have returned 0.
        # If we are here, found_misplaced_box should be True.
        if not found_misplaced_box:
             # This indicates an inconsistency, should not happen if goal check is correct.
             # print("Heuristic Warning: No misplaced boxes found, but not goal state?")
             return 0 # Safest return value if goal state wasn't detected properly

        # 6. Combine Components: Add robot cost if needed
        if min_robot_to_misplaced_box_dist == float('inf'):
            # Robot cannot reach *any* of the boxes that need moving.
            return float('inf') # This state is a dead end for the robot.
        else:
            # Add the cost for the robot to get to the nearest misplaced box
            total_heuristic_value += min_robot_to_misplaced_box_dist

        # 7. Return Final Value
        # Value should be non-negative. Using max is a safeguard.
        return max(0, total_heuristic_value)

