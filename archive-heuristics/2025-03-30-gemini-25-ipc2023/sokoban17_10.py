import collections
import itertools # Not strictly required but potentially useful for advanced heuristics
from fnmatch import fnmatch # Not used in final code, but kept for consistency with examples
from heuristics.heuristic_base import Heuristic
# collections.deque is used internally in _compute_all_pairs_shortest_paths

# Helper function to parse PDDL facts stored as strings
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string by removing parentheses
    and splitting by spaces.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    """
    # Basic check for valid fact format (starts with '(', ends with ')')
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Return empty list or raise error for invalid format
        return []
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It primarily calculates the sum of shortest path distances for each box from its
    current location to its designated goal location. This sum approximates the minimum
    number of 'push' actions required. Additionally, it adds an estimate of the robot's
    initial movement cost, calculated as the shortest distance from the robot's current
    position to a location from which it can push *any* of the misplaced boxes.
    The heuristic incorporates basic dead-end detection, identifying states where a
    box is trapped in a non-goal corner location as potentially unsolvable (infinite cost).

    # Assumptions
    - The main component of the cost is the number of pushes required to move boxes
      to their goals. Robot movement cost is approximated only by the initial approach
      to the first box push.
    - Shortest path distances for both boxes and the robot are calculated based on the
      static grid layout (using 'adjacent' facts), ignoring dynamic obstacles like the
      current positions of the robot and other boxes during pathfinding. This provides
      a lower bound on movement/push counts on an empty grid.
    - Dead-end detection is limited to simple cases: a box located in a non-goal
      square that forms a corner between two 'walls' (unpassable directions). More
      complex deadlocks (e.g., two boxes mutually blocking each other in a narrow
      passage) are not detected by this heuristic.
    - The goal specification consists of '(at ?box ?location)' predicates, assigning
      each relevant box to a unique target location.
    - The environment is static, meaning the layout defined by 'adjacent' facts
      does not change.
    - All specified locations are reachable within the grid if the graph defined by
      'adjacent' facts is connected.

    # Heuristic Initialization
    - **Goal Parsing:** Extracts all '(at ?box ?location)' goals from `task.goals`
      and stores them in `self.goals_dict` (mapping box names to goal locations).
      Also collects the set of all goal-relevant boxes in `self.boxes`.
    - **Map Representation:** Parses static `(adjacent ?l1 ?l2 ?dir)` facts from
      `task.static` to build graph representations:
        - `self.adj`: Forward adjacency list (loc -> list of (neighbor, direction)).
        - `self.reverse_adj`: Reverse adjacency list (loc -> list of (neighbor, direction)
          where neighbor can reach loc via direction). Used to find where the robot
          needs to be to push a box *at* loc.
        - `self.locations`: A set of all unique location names encountered.
    - **Distance Precomputation:** Calculates all-pairs shortest path distances between
      every pair of locations using Breadth-First Search (BFS) starting from each
      location. Results are stored in `self.dist_map`, a nested dictionary where
      `self.dist_map[loc1][loc2]` gives the distance. Unreachable pairs have infinite distance.
    - **Dead-End Precomputation:** Identifies simple dead-end locations. A location `l`
      is considered a dead end if it is not a goal location for any box and it represents
      a corner (e.g., has no adjacent location upwards AND no adjacent location leftwards).
      These locations are stored in the `self.dead_ends` set.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Iterate through the facts in the current state (`node.state`,
        a frozenset of strings). Identify the robot's location (`robot_loc`) from the
        `(at-robot ?loc)` fact and the current location of each goal-relevant box
        (`box_locations` dictionary) from `(at ?box ?loc)` facts. Perform basic validation
        (e.g., ensure robot and all goal boxes are found).
    2.  **Check Goal Completion:** Iterate through the `self.goals_dict`. If all boxes
        are found to be at their respective goal locations, the current state is a goal
        state. Return 0.
    3.  **Initialize Costs:** Set `heuristic_value = 0` (to accumulate box push estimates)
        and `min_robot_to_box_dist = float('inf')` (to find the minimum robot approach distance).
        Set `misplaced_boxes_exist = False`.
    4.  **Iterate Through Goal Boxes:** For each box `b` and its goal location `g_b` in `self.goals_dict`:
        a.  Get the box's current location `l_b` from `box_locations`.
        b.  If `l_b != g_b` (the box is misplaced):
            i.   Set `misplaced_boxes_exist = True`.
            ii.  **Dead End Check:** If `l_b` is present in the `self.dead_ends` set, this
                 state is considered a dead end. Return `float('inf')`.
            iii. **Box Distance:** Look up the shortest path distance `d = self.dist_map.get(l_b, {}).get(g_b, float('inf'))`.
                 If `d` is infinity, the goal is unreachable for this box. Return `float('inf')`.
                 Otherwise, add `d` to `heuristic_value`. This estimates the minimum pushes for this box.
            iv.  **Robot Distance to Push Position:** Find all locations `p` from which the
                 robot could push the box at `l_b`. These are the locations adjacent to `l_b`
                 in the reverse direction, found using `self.reverse_adj.get(l_b, [])`. For each
                 such valid push position `p`, find the distance `d_robot = self.dist_map.get(robot_loc, {}).get(p, float('inf'))`.
                 Keep track of the minimum `d_robot` found across all push positions for *this* box
                 (`min_dist_to_this_box_adj`).
            v.   Update the overall minimum robot approach distance:
                 `min_robot_to_box_dist = min(min_robot_to_box_dist, min_dist_to_this_box_adj)`.
    5.  **Handle Goal State (Post-Loop Check):** If `misplaced_boxes_exist` is still `False`
        after the loop, it means no boxes were misplaced. Return 0.
    6.  **Check Robot Reachability:** If `min_robot_to_box_dist` remains `float('inf')`, it implies
        the robot cannot reach any valid position to push any of the misplaced boxes (either the robot
        is trapped or the required push positions are unreachable). Return `float('inf')`.
    7.  **Combine Costs:** The final heuristic estimate is the sum of estimated pushes (`heuristic_value`)
        plus the estimated initial robot moves (`min_robot_to_box_dist`). Return this sum.
        Ensure the returned value is non-negative (it should be by construction if distances are non-negative).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing goals, map layout, precomputing
        distances, and identifying dead ends.
        """
        self.goals_dict = {} # Map box -> goal location
        self.boxes = set()
        for goal in task.goals:
            parts = get_parts(goal)
            # Ensure the goal is an 'at' predicate with 3 parts (predicate, box, location)
            if parts and parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.goals_dict[box] = loc
                self.boxes.add(box)

        self.adj = collections.defaultdict(list) # loc -> list of (neighbor_loc, direction)
        self.reverse_adj = collections.defaultdict(list) # loc -> list of (neighbor_loc, direction) where adjacent(neighbor_loc, loc, direction)
        self.locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            # Ensure the fact is an 'adjacent' predicate with 4 parts
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.adj[l1].append((l2, direction))
                self.reverse_adj[l2].append((l1, direction))
                self.locations.add(l1)
                self.locations.add(l2)

        # Precompute all-pairs shortest paths using BFS
        self.dist_map = self._compute_all_pairs_shortest_paths()

        # Precompute dead-end locations (simple corners)
        self.dead_ends = self._compute_dead_ends()

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        dist_map = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))
        if not self.locations:
             # print("Warning: No locations found. Cannot compute distances.") # Optional debug
             return dist_map # Return empty map if no locations

        for start_node in self.locations:
            # Basic check if start_node is valid (it should be from self.locations)
            if start_node not in self.locations: continue

            dist_map[start_node][start_node] = 0
            queue = collections.deque([start_node])
            visited_bfs = {start_node} # Visited set for this specific BFS run

            while queue:
                u = queue.popleft()
                # Use .get(u, []) for safer access to adjacency list
                for v, _ in self.adj.get(u, []): # Direction ('_') not needed for distance calculation
                    # Ensure neighbor v is a valid location and not visited in this BFS run
                    if v in self.locations and v not in visited_bfs:
                        visited_bfs.add(v)
                        dist_map[start_node][v] = dist_map[start_node][u] + 1
                        queue.append(v)
        return dist_map

    def _compute_dead_ends(self):
        """
        Computes simple dead-end locations. A non-goal location is a dead end
        if it's a corner formed by walls in two orthogonal directions.
        """
        dead_ends = set()
        all_goal_locs = set(self.goals_dict.values())

        for loc in self.locations:
            # A location cannot be a dead end if it's a goal location for any box
            if loc in all_goal_locs:
                continue

            # Determine which directions are blocked by walls from this location
            # A direction is blocked if there's no adjacent location in that direction.
            has_wall = {'up': True, 'down': True, 'left': True, 'right': True}
            for _, direction in self.adj.get(loc, []):
                 if direction in has_wall:
                     has_wall[direction] = False

            # Check for corner conditions: blocked in one vertical AND one horizontal direction.
            is_dead = False
            if (has_wall['up'] and has_wall['left']) or \
               (has_wall['up'] and has_wall['right']) or \
               (has_wall['down'] and has_wall['left']) or \
               (has_wall['down'] and has_wall['right']):
                is_dead = True

            if is_dead:
                dead_ends.add(loc)
        return dead_ends


    def __call__(self, node):
        """Calculate the heuristic value for the given state node."""
        state = node.state
        robot_loc = None
        box_locations = {} # Map box -> current location

        # Parse state to find robot and box locations
        for fact in state:
            parts = get_parts(fact)
            # Skip invalid/empty facts
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif predicate == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Track only boxes that are part of the goal definition
                if box in self.boxes:
                    box_locations[box] = loc

        # --- Basic State Validation ---
        if not robot_loc or robot_loc not in self.locations:
             # print(f"Warning: Robot location '{robot_loc}' not found or invalid.") # Optional debug
             return float('inf') # Invalid state if robot location is unknown/invalid

        # Check if all goal boxes are present in the current state's facts
        if len(box_locations) != len(self.boxes):
            # print(f"Warning: Not all goal boxes found in state. Goals: {self.boxes}, Found: {box_locations.keys()}") # Optional debug
             # This implies an invalid or unexpected state, return infinity
             return float('inf')
        # --- End Validation ---

        heuristic_value = 0
        min_robot_to_box_dist = float('inf')
        misplaced_boxes_exist = False

        # --- Calculate Heuristic Components ---
        for box, goal_loc in self.goals_dict.items():
            # box_locations should contain the box due to validation above
            current_loc = box_locations[box]

            if current_loc != goal_loc:
                misplaced_boxes_exist = True

                # 1. Dead End Check
                if current_loc in self.dead_ends:
                    # print(f"Dead End: Box {box} at {current_loc}") # Optional debug
                    return float('inf')

                # 2. Box Distance to Goal (Estimated Pushes)
                # Use .get() for safer dictionary access, providing default {} or inf
                box_dist = self.dist_map.get(current_loc, {}).get(goal_loc, float('inf'))
                if box_dist == float('inf'):
                    # print(f"Unreachable: Box {box} cannot reach {goal_loc} from {current_loc}") # Optional debug
                    return float('inf') # Goal is unreachable for this box
                heuristic_value += box_dist

                # 3. Min Robot Distance to a Push Position for this Box
                min_dist_to_this_box_adj = float('inf')
                # Iterate through locations 'push_pos' from which robot can push box at 'current_loc'
                # These are found using the reverse adjacency list for 'current_loc'
                for push_pos, _ in self.reverse_adj.get(current_loc, []): # Direction ('_') not needed here
                    # Ensure push_pos is a valid location before calculating distance
                    if push_pos in self.locations:
                        # Find distance from robot's current location to this push position
                        dist = self.dist_map.get(robot_loc, {}).get(push_pos, float('inf'))
                        min_dist_to_this_box_adj = min(min_dist_to_this_box_adj, dist)

                # Update the overall minimum distance for the robot to reach *any* push position
                # This finds the cost for the robot to get into position for the 'easiest' first push
                min_robot_to_box_dist = min(min_robot_to_box_dist, min_dist_to_this_box_adj)
        # --- End Calculation Loop ---

        # --- Final Heuristic Value ---
        if not misplaced_boxes_exist:
            # If the loop finished and no boxes were misplaced, it's a goal state
            return 0

        # If robot cannot reach any position to push any misplaced box
        if min_robot_to_box_dist == float('inf'):
            # print(f"Unreachable: Robot at {robot_loc} cannot reach any push position.") # Optional debug
            return float('inf')

        # Combine estimated pushes and estimated initial robot moves
        total_heuristic_value = heuristic_value + min_robot_to_box_dist

        # Return the non-negative heuristic value
        # (Should be non-negative if distances are non-negative)
        return total_heuristic_value # No need for max(0, ...) if distances >= 0
