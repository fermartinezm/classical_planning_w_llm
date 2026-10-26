import collections
from fnmatch import fnmatch
# Assuming the Heuristic base class is defined in heuristics.heuristic_base
# If the structure is different, adjust the import path accordingly.
from heuristics.heuristic_base import Heuristic

# Helper functions can be defined outside the class or as static methods.
def get_parts(fact: str) -> list[str]:
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Removes the surrounding parentheses and splits the string by spaces.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    """
    return fact[1:-1].split()

def match(fact_parts: list[str], *pattern: str) -> bool:
    """
    Checks if the parts of a fact match a given pattern.
    Supports '*' as a wildcard in the pattern.
    Example: match(["at", "box1", "loc_1_1"], "at", "*", "loc*") -> True
    """
    if len(fact_parts) != len(pattern):
        return False
    # Compare each part with the corresponding pattern element using fnmatch for wildcard support
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban problems.
    The estimation is based on two components:
    1. The sum of shortest path distances for each box from its current location
       to its target goal location. This estimates the total number of 'push' actions.
    2. The shortest path distance for the robot to reach the *closest* position
       from which it can push *any* of the currently misplaced boxes. This estimates
       the number of 'move' actions required to initiate the pushing sequence.

    The distances are computed using Breadth-First Search (BFS) on the static map layout
    defined by the 'adjacent' predicates. The heuristic ignores dynamic obstacles
    (like other boxes blocking paths or the robot's position) for computational
    efficiency. As such, it is generally non-admissible but aims to provide
    good guidance for greedy search algorithms.

    # Assumptions
    - The map structure is static and defined by 'adjacent' predicates.
    - Goal conditions primarily consist of '(at <box> <location>)' facts, specifying
      the target location for each box. Goals related to the robot's final position
      or specific 'clear' locations are not explicitly handled by this heuristic's
      core calculation but are implicitly checked by the planner's goal test.
    - The cost of a 'move' action and a 'push' action is 1.
    - The heuristic does not perform complex deadlock detection (e.g., boxes in
      corners from which they cannot reach a goal) beyond basic reachability checks
      via BFS.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the task's static information and goals.
    - It parses all static 'adjacent' facts to build two adjacency representations:
        - `self.adj`: A dictionary mapping each location to a list of its neighbors
          (locations reachable directly from it). Used for forward BFS.
        - `self.adj_rev`: A dictionary mapping each location `L` to a list of tuples
          `(N, D)`, where `N` is a neighboring location and `D` is the direction such
          that `adjacent(N, L, D)` is true. This helps identify the locations (`N`)
          the robot must occupy to push a box currently at `L`.
    - It parses the task's goal conditions to create `self.goal_map`, a dictionary
      mapping each box name (string) to its target goal location (string).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the robot's current location (`robot_loc`)
        and the current location of each box (`box_locs`) from the input `node.state`.
    2.  **Calculate Total Box Distance:**
        - Initialize `total_box_distance = 0`.
        - Keep track of whether any boxes are misplaced (`misplaced_boxes_exist = False`)
          and store details of misplaced boxes (`misplaced_box_details`).
        - Iterate through each `(box, goal_loc)` pair in `self.goal_map`.
        - Find the `current_loc` of the `box` from `box_locs`.
        - If `current_loc != goal_loc`:
            - Mark `misplaced_boxes_exist = True`.
            - Add `(box, current_loc)` to `misplaced_box_details`.
            - Compute the shortest path distance `dist` from `current_loc` to `goal_loc`
              using BFS (`self.bfs_distance`). This estimates the pushes needed for this box.
            - If `dist` is infinity, the goal is unreachable for this box; return `float('inf')`
              as the heuristic value (unsolvable state).
            - Add `dist` to `total_box_distance`.
    3.  **Check for Goal State:** If `misplaced_boxes_exist` is still `False` after checking
        all boxes, it means all boxes are in their goal locations. Return 0.
    4.  **Calculate Minimum Robot Distance to Push:**
        - Initialize `min_robot_distance_to_push = float('inf')`.
        - Iterate through the `(box, current_loc)` pairs in `misplaced_box_details`.
        - For each misplaced box at `current_loc`, find all possible locations
          `push_from_loc` from which the robot could push it. These are the locations `N`
          found in `self.adj_rev[current_loc]`.
        - For each potential `push_from_loc`:
            - Compute the shortest path distance `dist` from the `robot_loc` to
              `push_from_loc` using BFS.
            - Update `min_robot_distance_to_push = min(min_robot_distance_to_push, dist)`.
    5.  **Handle Robot Reachability:** If `min_robot_distance_to_push` remains `float('inf')`
        after checking all possibilities, it means the robot cannot reach any position
        to push any of the misplaced boxes. Return `float('inf')` (unsolvable state).
    6.  **Combine Components:** The final heuristic value is the sum of the estimated pushes
        and the estimated initial robot moves: `heuristic_value = total_box_distance + min_robot_distance_to_push`.
    7.  **Return Value:** Return the calculated `heuristic_value`. It will be 0 for goal states,
        a positive finite number for reachable non-goal states, and infinity for states
        detected as unsolvable by the reachability checks.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information (adjacencies)
        and goal conditions (box target locations) from the planning task.
        """
        self.goals = task.goals
        static_facts = task.static

        # Adjacency lists for graph representation
        self.adj = collections.defaultdict(list)
        # Reverse adjacency: maps location L to list of (Neighbor N, Direction D)
        # such that adjacent(N, L, D) is true. Helps find where robot needs to be.
        self.adj_rev = collections.defaultdict(list)

        # Build adjacency lists from static 'adjacent' facts
        for fact in static_facts:
            parts = get_parts(fact)
            # Ensure the fact structure is correct before accessing parts
            if len(parts) == 4 and parts[0] == "adjacent":
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.adj[loc1].append(loc2)
                self.adj_rev[loc2].append((loc1, direction))

        # Parse goals to find the target location for each box
        self.goal_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure the goal fact structure is correct (at predicate, 2 args)
            if len(parts) == 3 and parts[0] == "at":
                # Assume the first argument is the box and the second is the location
                box, loc = parts[1], parts[2]
                self.goal_map[box] = loc

        # Cache for BFS results within a single heuristic evaluation (__call__)
        # This avoids redundant BFS computations for the same start/end pair
        # within one state evaluation, e.g., when calculating robot distance
        # to multiple push positions around the same box.
        self._bfs_cache = {}

    def bfs_distance(self, start_loc: str, end_loc: str) -> float:
        """
        Performs Breadth-First Search (BFS) on the static location graph
        represented by self.adj to find the shortest distance between two locations.

        Args:
            start_loc: The starting location name.
            end_loc: The target location name.

        Returns:
            The shortest distance (number of steps) if reachable,
            or float('inf') if the target is not reachable from the start.
            Uses a cache (self._bfs_cache) for efficiency within a single heuristic call.
        """
        if start_loc == end_loc:
            return 0

        # Check cache first to avoid re-computation
        cache_key = (start_loc, end_loc)
        if cache_key in self._bfs_cache:
            return self._bfs_cache[cache_key]

        # Initialize BFS queue with (location, distance)
        queue = collections.deque([(start_loc, 0)])
        # Keep track of visited locations to avoid cycles and redundant exploration
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            # Explore neighbors using the precomputed adjacency list
            for neighbor in self.adj.get(current_loc, []):
                if neighbor == end_loc:
                    # Goal found, cache and return distance
                    self._bfs_cache[cache_key] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # Target not reachable if the queue becomes empty
        self._bfs_cache[cache_key] = float('inf')
        return float('inf')

    def __call__(self, node) -> float:
        """
        Computes the heuristic value for the given state node.

        Args:
            node: A search node containing the state (`node.state`).

        Returns:
            An estimated cost (float) to reach the goal from the node's state.
            Returns 0 for goal states, infinity for detected dead ends/unsolvable states.
        """
        state = node.state
        self._bfs_cache.clear() # Clear cache for the new state evaluation

        # --- 1. Parse Current State ---
        robot_loc = None
        box_locs = {} # Maps box name -> current location
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "at-robot":
                robot_loc = parts[1]
            elif len(parts) == 3 and parts[0] == "at":
                # Assuming the object type 'box' is implicitly handled by goal_map keys
                box_name, loc = parts[1], parts[2]
                box_locs[box_name] = loc

        # Basic check for state validity (robot must exist)
        if robot_loc is None:
             # This indicates an invalid state according to the domain definition
             return float('inf')

        # --- 2. Calculate Total Box Distance & Identify Misplaced Boxes ---
        total_box_distance = 0.0
        misplaced_boxes_exist = False
        # Store details (box_name, current_location) for misplaced boxes
        misplaced_box_details = []

        for box, goal_loc in self.goal_map.items():
            current_loc = box_locs.get(box)
            if current_loc is None:
                 # A box required by the goal is missing from the state.
                 # This implies an invalid or unsolvable state scenario.
                 return float('inf')

            if current_loc != goal_loc:
                misplaced_boxes_exist = True
                misplaced_box_details.append((box, current_loc))

                # Estimate pushes needed for this box
                dist = self.bfs_distance(current_loc, goal_loc)

                if dist == float('inf'):
                    # If any box cannot reach its goal location, the state is unsolvable.
                    return float('inf')
                total_box_distance += dist

        # --- 3. Check for Goal State ---
        if not misplaced_boxes_exist:
            # If no boxes are misplaced, we assume the goal is reached.
            # A more rigorous check `self.goals <= state` could be used if goals
            # involved more than just box positions, but is slower.
            return 0.0

        # --- 4. Calculate Minimum Robot Distance to Push ---
        min_robot_distance_to_push = float('inf')

        # Iterate through only the boxes that are actually misplaced
        for _box, current_loc in misplaced_box_details:
            # Find all locations 'push_from_loc' adjacent to the box's current location
            # using the precomputed reverse adjacency list.
            if current_loc in self.adj_rev:
                for push_from_loc, _direction in self.adj_rev[current_loc]:
                    # Calculate distance from the robot's current location
                    # to this potential pushing spot.
                    dist = self.bfs_distance(robot_loc, push_from_loc)
                    min_robot_distance_to_push = min(min_robot_distance_to_push, dist)

        # --- 5. Handle Robot Reachability ---
        if min_robot_distance_to_push == float('inf'):
            # If the robot cannot reach a position to push *any* of the misplaced boxes,
            # then the state is effectively unsolvable from the robot's perspective.
            return float('inf')

        # --- 6. Combine Components ---
        # The heuristic value is the sum of estimated pushes and estimated robot moves.
        heuristic_value = total_box_distance + min_robot_distance_to_push

        # --- 7. Return Value ---
        # Ensure heuristic is strictly 0 only for goal states.
        # Since we return 0 if misplaced_boxes_exist is False, and handle
        # infinite cases, any finite positive value represents a non-goal state.
        # A safeguard check in case of calculation errors leading to h=0 incorrectly:
        if heuristic_value == 0 and misplaced_boxes_exist:
             # This should not happen with correct BFS returning positive distances or inf.
             # If it does, return a small positive value to indicate non-goal state.
             return 1.0

        return heuristic_value
