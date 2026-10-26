import sys
from collections import deque
# Import base class for heuristics
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    Handles potential errors with malformed fact strings.
    """
    try:
        # Remove parentheses and split by space
        return fact[1:-1].split()
    except:
        # Return an empty list if the fact format is unexpected
        return []

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It calculates the sum of shortest path distances for each box from its current
    location to its designated goal location. Additionally, it adds the shortest
    path distance for the robot to reach the nearest box that is not currently
    in its goal location. The heuristic aims to guide a greedy best-first search
    by prioritizing states where boxes are closer to their goals and the robot is
    near a box that requires pushing.

    # Assumptions
    - Each box specified in the goal has a unique target location.
      For example, the goal contains facts like `(at box1 goal_loc1)`.
    - Both the 'move' action (robot moves) and the 'push' action (robot pushes box)
      have a uniform cost of 1.
    - The heuristic calculates distances based on the static grid structure defined
      by `adjacent` facts. It precomputes shortest paths assuming the grid is empty
      (ignoring current robot and box positions for pathfinding). This makes the
      distance calculation efficient but potentially underestimates the true cost
      when paths are blocked.

    # Heuristic Initialization
    - The `__init__` method processes the task's static information and goal conditions once.
    - It identifies the target location for each box from the `task.goals`.
    - It builds an adjacency list representation (`self.adj`) of the grid using the
      static `adjacent` facts, storing all unique location names in `self.locations`.
    - It precomputes all-pairs shortest path distances between every pair of locations
      using Breadth-First Search (BFS). These distances are stored in `self.dist_matrix`.
      If a location is unreachable from another, the distance is stored as infinity.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Goals & Build Grid (in `__init__`)**:
        - Extract `(at box loc)` goal facts into `self.goal_locations` (box -> loc).
        - Process `(adjacent loc1 loc2 dir)` static facts to build `self.adj` graph
          and populate `self.locations`.
    2.  **Precompute Distances (in `__init__`)**:
        - For each location `start_loc` in `self.locations`:
            - Run BFS to find the shortest path distance to all other reachable locations `end_loc`.
            - Store the distances in `self.dist_matrix[start_loc][end_loc]`. Initialize
              all distances to `float('inf')` and distance to self as 0.
    3.  **Get Current State Info (in `__call__`)**:
        - Given the current `node.state` (a frozenset of fact strings):
            - Find the robot's location `robot_loc` from `(at-robot ?loc)`.
            - Find the current location for each box relevant to the goal using
              `(at ?box ?loc)` facts, storing them in `box_locations`.
    4.  **Calculate Sum of Box-to-Goal Distances (in `__call__`)**:
        - Initialize `heuristic_value = 0.0`.
        - Keep track of misplaced boxes in `misplaced_boxes_info`.
        - For each `box` in `box_locations`:
            - Get its `current_loc` and `goal_loc`.
            - If `current_loc != goal_loc`:
                - Retrieve the precomputed distance `d = self.dist_matrix[current_loc].get(goal_loc, float('inf'))`.
                - If `d` is infinity, the goal is statically unreachable for this box; return `float('inf')`.
                - Add `d` to `heuristic_value`.
                - Add `(box, current_loc)` to `misplaced_boxes_info`.
    5.  **Handle Goal State (in `__call__`)**:
        - If `misplaced_boxes_info` is empty after checking all relevant boxes found
          in the state, it means all present goal-boxes are in their correct locations. Return 0.
          (This assumes the problem guarantees all goal boxes are always present).
    6.  **Calculate Min Robot-to-Misplaced-Box Distance (in `__call__`)**:
        - If there are misplaced boxes:
            - Initialize `min_robot_dist = float('inf')`.
            - For each `(box, box_loc)` in `misplaced_boxes_info`:
                - Find distance `d_robot = self.dist_matrix[robot_loc].get(box_loc, float('inf'))`.
                - Update `min_robot_dist = min(min_robot_dist, d_robot)`.
            - If `min_robot_dist` is still infinity, the robot cannot reach any box
              that needs moving; return `float('inf')`.
            - Otherwise, add `min_robot_dist` to `heuristic_value`. This estimates the
              cost for the robot to move to the nearest task (pushing a misplaced box).
    7.  **Return Final Value (in `__call__`)**:
        - Return the total `heuristic_value`. If it's infinity, return `float('inf')`.
        - Otherwise, return the value as an integer (since individual move/push costs are 1).
    """

    def __init__(self, task):
        """
        Initializes the heuristic: parses goals, builds the grid graph from static
        'adjacent' facts, and precomputes all-pairs shortest path distances using BFS.
        """
        self.goals = task.goals
        static_facts = task.static
        self.locations = set()
        self.adj = {}  # Adjacency list: loc -> [neighbor_loc, ...]
        self.goal_locations = {}  # box_name -> goal_location_name

        # 1. Parse goals and build adjacency graph from static facts
        for goal in self.goals:
            parts = get_parts(goal)
            # Expect goal like: (at box1 loc_goal)
            if len(parts) == 3 and parts[0] == 'at':
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc

        for fact in static_facts:
            parts = get_parts(fact)
            # Expect adjacent fact: (adjacent loc1 loc2 dir)
            if len(parts) == 4 and parts[0] == 'adjacent':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                # Add edge l1 -> l2 based on the PDDL definition
                if l1 not in self.adj: self.adj[l1] = []
                # Avoid duplicates if PDDL defines adjacency redundantly
                if l2 not in self.adj[l1]:
                    self.adj[l1].append(l2)

        # 2. Precompute all-pairs shortest paths using BFS
        self.dist_matrix = {loc: {} for loc in self.locations}
        for start_node in self.locations:
            # Initialize distances for this start_node
            # Using sys.maxsize as a large number, safer than float('inf') for int sums
            # but float('inf') is standard for unreachable. Let's stick to float('inf').
            for loc in self.locations:
                self.dist_matrix[start_node][loc] = float('inf')

            # Check if start_node is a valid location (might be isolated if not in adj keys/values)
            if start_node not in self.locations:
                 continue # Should not happen if locations set is built correctly

            self.dist_matrix[start_node][start_node] = 0
            queue = deque([start_node])
            # visited dictionary for this specific BFS run
            visited = {start_node: 0} # Store distance in visited for efficiency

            while queue:
                current_node = queue.popleft()
                current_dist = visited[current_node] # Get distance from visited dict

                # Explore neighbors using the adjacency list
                if current_node in self.adj:
                    for neighbor in self.adj[current_node]:
                        # Check if neighbor is a valid location and not visited yet
                        if neighbor in self.locations and neighbor not in visited:
                            visited[neighbor] = current_dist + 1
                            self.dist_matrix[start_node][neighbor] = current_dist + 1
                            queue.append(neighbor)

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns the estimated cost (integer) or float('inf') if goal is unreachable.
        """
        state = node.state
        heuristic_value = 0.0  # Use float to handle potential infinities
        robot_loc = None
        box_locations = {}  # box_name -> current_location_name
        misplaced_boxes_info = []  # List of (box_name, current_loc) tuples

        # 3. Get current locations from the state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue  # Skip potential malformed facts

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Only track locations of boxes that are part of the goal definition
                if box in self.goal_locations:
                    box_locations[box] = loc

        # If robot location is not found or is not a known location, state is invalid/unreachable
        if robot_loc is None or robot_loc not in self.locations:
             return float('inf')

        # 4. Calculate sum of box-to-goal distances for misplaced boxes
        any_box_misplaced = False
        for box, current_loc in box_locations.items():
            goal_loc = self.goal_locations[box]

            if current_loc != goal_loc:
                any_box_misplaced = True
                # Ensure current_loc is a valid, known location from the grid
                if current_loc not in self.locations:
                    return float('inf')  # Box is in an unknown/invalid location

                # Retrieve precomputed shortest path distance
                dist = self.dist_matrix[current_loc].get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # Goal is statically unreachable for this box from its current location.
                    return float('inf')

                heuristic_value += dist
                misplaced_boxes_info.append((box, current_loc))

        # 5. Handle Goal State: If no boxes (among those relevant to the goal) are misplaced
        if not any_box_misplaced:
            # Check if all goal boxes were actually found in the state.
            # If a goal box is missing, it's not the goal state.
            if len(box_locations) == len(self.goal_locations):
                 return 0 # All goal boxes are present and in the correct location
            else:
                 # Not all goal boxes are present in the state, so not a goal state.
                 # Assign a small non-zero cost? Or rely on goal check?
                 # A simple heuristic could return 0, letting the goal test fail later.
                 # Let's return 0 here, assuming the search framework checks full goal satisfaction.
                 # If we wanted to be stricter, we could return 1 or infinity.
                 return 0


        # 6. Calculate minimum robot distance to any misplaced box
        min_robot_to_box_dist = float('inf')
        # Ensure robot_loc has entries in dist_matrix (checked earlier)
        if robot_loc not in self.dist_matrix:
             return float('inf') # Should not happen due to earlier check

        for _, box_loc in misplaced_boxes_info:
             # box_loc validity checked during box distance calculation
             dist = self.dist_matrix[robot_loc].get(box_loc, float('inf'))
             min_robot_to_box_dist = min(min_robot_to_box_dist, dist)

        # If the robot cannot reach any of the misplaced boxes
        if min_robot_to_box_dist == float('inf'):
            # Avoid returning inf + inf if heuristic_value is already inf
            return float('inf')
        else:
            # Add the cost for the robot to reach the nearest misplaced box
            heuristic_value += min_robot_to_box_dist

        # 7. Return the final heuristic value
        if heuristic_value == float('inf'):
            return float('inf')
        else:
            # The sum of integer distances should be an integer.
            # Convert to int for consistency, assuming costs are integers.
            return int(round(heuristic_value))
