import heapq
import math
from collections import deque, defaultdict

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


# --- Hungarian Algorithm Implementation ---
# A simple pure Python implementation of the Hungarian algorithm
# Assumes a square cost matrix.
# Returns the minimum cost of a perfect assignment.
# Handles infinite costs by replacing them with a large number and checking original costs.
class HungarianAlgorithm:
    def __init__(self, cost_matrix):
        self.matrix = [list(row) for row in cost_matrix] # Create a mutable copy
        self.original_matrix = cost_matrix # Keep original for final cost calculation
        self.rows = len(self.matrix)
        self.cols = len(self.matrix[0])
        assert self.rows == self.cols, "Cost matrix must be square"
        self.n = self.rows # Use n for size

        # Replace inf with a very large number for internal calculation
        # Calculate a large number based on finite costs
        finite_costs = [c for row in self.matrix for c in row if math.isfinite(c)]
        # Choose a number larger than any possible finite sum + 1
        very_large_number = sum(finite_costs) + 1 if finite_costs else 1e9
        # Ensure it's large enough even if sum is 0 or negative (though costs are distances >= 0)
        very_large_number = max(very_large_number, 1e9) # Use a large default if sum is small

        for r in range(self.n):
            for c in range(self.n):
                if not math.isfinite(self.matrix[r][c]):
                    self.matrix[r][c] = very_large_number

    def solve(self):
        matrix = self.matrix # Use the potentially modified matrix

        # Step 1: Subtract row minimums
        for i in range(self.n):
            min_val = min(matrix[i])
            # min_val can be the large number if the row was all inf originally
            if math.isfinite(min_val):
                for j in range(self.n):
                    if math.isfinite(matrix[i][j]): # Only subtract if finite
                         matrix[i][j] -= min_val

        # Step 2: Subtract column minimums
        for j in range(self.n):
            min_val = min(matrix[i][j] for i in range(self.n))
            # min_val can be the large number
            if math.isfinite(min_val):
                for i in range(self.n):
                    if math.isfinite(matrix[i][j]): # Only subtract if finite
                         matrix[i][j] -= min_val

        # Steps 3 & 4: Cover zeros and adjust matrix until perfect matching exists
        while True:
            # Find a maximum matching in the zero-valued entries
            row_match = [-1] * self.n # row_match[r] = c
            col_match = [-1] * self.n # col_match[c] = r
            row_visited = [False] * self.n

            def find_augmenting_path(r):
                row_visited[r] = True
                for c in range(self.n):
                    # Check for zero, handle potential floating point inaccuracies
                    if abs(matrix[r][c]) < 1e-9: # Treat values close to zero as zero
                        if col_match[c] == -1 or (not row_visited[col_match[c]] and find_augmenting_path(col_match[c])):
                            row_match[r] = c
                            col_match[c] = r
                            return True
                return False

            # Find initial matching
            num_matched = 0
            for r in range(self.n):
                row_visited = [False] * self.n
                if find_augmenting_path(r):
                    num_matched += 1

            if num_matched == self.n:
                break # Perfect matching found

            # Find minimum set of lines to cover zeros (using Konig's theorem logic)
            # Build alternating forest from unmatched rows
            unmatched_rows = [r for r in range(self.n) if row_match[r] == -1]
            S = set(unmatched_rows) # Rows in forest
            T = set() # Columns in forest
            q = deque(unmatched_rows)

            while q:
                r = q.popleft()
                for c in range(self.n):
                    # Check for zero
                    if abs(matrix[r][c]) < 1e-9:
                        if c not in T:
                            T.add(c)
                            if col_match[c] != -1: # If column is matched
                                r_prime = col_match[c]
                                if r_prime not in S:
                                    S.add(r_prime)
                                    q.append(r_prime)

            # Rows covered by lines: R_covered = set(range(self.n)) - S
            # Columns covered by lines: C_covered = T

            # Find minimum uncovered value
            min_uncovered = float('inf')
            for i in range(self.n):
                for j in range(self.n):
                    # Check if element is uncovered
                    is_covered = (i in (set(range(self.n)) - S)) or (j in T)
                    if not is_covered:
                         min_uncovered = min(min_uncovered, matrix[i][j])

            # If min_uncovered is inf, it means no finite cost path exists for some required assignment
            # This should ideally be caught earlier if dist_grid is inf
            # But as a safeguard, if the matrix reduction leads to this,
            # it implies no finite cost perfect matching exists.
            if not math.isfinite(min_uncovered):
                 # This case should be rare if dist_grid was computed correctly
                 # and the problem is solvable on the full grid.
                 # It might happen if the large replacement number wasn't large enough,
                 # or due to floating point issues.
                 # Let's return inf as it indicates an issue or unsolvability.
                 return float('inf')


            # Add min_uncovered to elements covered twice (intersection of R_covered and C_covered)
            # Subtract min_uncovered from all uncovered elements
            for i in range(self.n):
                for j in range(self.n):
                    is_covered_by_row_line = (i in (set(range(self.n)) - S))
                    is_covered_by_col_line = (j in T)

                    if is_covered_by_row_line and is_covered_by_col_line:
                        # Covered twice
                        if math.isfinite(matrix[i][j]):
                             matrix[i][j] += min_uncovered
                    elif not is_covered_by_row_line and not is_covered_by_col_line:
                        # Uncovered
                        if math.isfinite(matrix[i][j]):
                             matrix[i][j] -= min_uncovered


        # Step 5: Sum the original costs for the optimal assignment
        total_cost = 0
        # The optimal assignment is given by the final zero positions
        # We need to find a perfect matching in the final zero matrix.
        # The 'row_match' from the last iteration of the while loop holds the optimal matching.
        for r in range(self.n):
            c = row_match[r]
            # c should not be -1 for a perfect matching of size n
            cost = self.original_matrix[r][c]
            if not math.isfinite(cost):
                 # This means the optimal matching involves an edge that was originally infinite.
                 # This state is likely unsolvable.
                 return float('inf')
            total_cost += cost

        return total_cost


# --- Sokoban Heuristic Class ---
class sokobanHeuristic(Heuristic):
    """
    Sokoban Domain-Dependent Heuristic.

    Summary:
    This heuristic estimates the cost to reach the goal state by combining
    two main components:
    1. The cost to move all misplaced boxes to their respective goal locations.
       This is calculated as the sum of shortest path distances for a minimum-cost
       matching between current box locations and goal locations. The shortest
       paths are computed on the full grid graph (ignoring dynamic obstacles).
    2. The cost for the robot to reach a position where it can start pushing
       one of the misplaced boxes. This is calculated as the shortest path
       distance for the robot from its current location to the location of
       the closest misplaced box, considering dynamic obstacles (other boxes).

    The total heuristic value is the sum of these two components.

    Assumptions:
    - The number of boxes equals the number of goal locations for boxes.
    - Location names follow the format 'loc_row_col'.
    - The grid connectivity is defined solely by 'adjacent' predicates in the static facts.
    - A pure Python implementation of the Hungarian algorithm is included.
    - The state representation is a frozenset of strings as shown in the example.
    - Static facts are available in task.static as a frozenset of strings.
    - Goal facts are available in task.goals as a frozenset of strings.

    Heuristic Initialization:
    1. Parse all 'adjacent' facts from task.static to build a graph representing
       the grid connectivity. Map location names (strings) to unique integer IDs.
    2. Compute All-Pairs Shortest Paths (APSP) on this grid graph using BFS
       starting from each location. Store these distances in a matrix (dist_grid).
       This represents the minimum number of steps (moves or pushes) between any
       two locations on the unobstructed grid.
    3. Parse goal facts from task.goals to create a mapping from each box name
       to its target goal location name (box_goals).
    4. Identify all box names present in the problem.

    Step-By-Step Thinking for Computing Heuristic (__call__):
    1. Parse the current state (node.state) to find:
       - The robot's current location (l_r).
       - The current location for each box (box_locations).
       - The set of occupied locations (robot + boxes).
    2. Identify which boxes are not currently at their designated goal locations
       (misplaced_boxes).
    3. If there are no misplaced boxes, the state is a goal state, return 0.
    4. If there are misplaced boxes:
       a. Calculate the 'box movement cost' (h_boxes) using matching:
          - Create a list of current locations for *all* boxes (sorted by box name).
          - Create a list of goal locations for *all* boxes (sorted by box name).
          - Build a cost matrix where C[i][j] is the precomputed shortest path
            distance (dist_grid) from the i-th current box location to the j-th
            goal box location.
          - If the number of boxes does not match the number of box goals, return infinity.
          - Use the Hungarian algorithm on this cost matrix to find the minimum
            cost perfect matching. The sum of distances in the matching is h_boxes.
          - If the Hungarian algorithm returns infinity, propagate it.
       b. Calculate the 'robot accessibility cost' (h_robot):
          - Build a temporary graph representing the robot's traversable locations
            in the *current* state. Edges exist between adjacent locations l1, l2
            if both l1 and l2 are not occupied by a box. The robot's current
            location is always traversable.
          - Run BFS from the robot's current location (l_r) on this temporary
            graph to find the shortest path distance to all other locations.
          - Find the minimum distance from l_r to any location currently occupied
            by a misplaced box. This minimum distance is h_robot.
          - If no misplaced box location is reachable by the robot, return infinity
            (as the state is likely a dead end).
    5. The total heuristic value is h_boxes + h_robot.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

        # --- Initialization Steps ---

        # 1. Build location graph and map names to IDs
        self.location_names = set()
        self.adjacency_list = defaultdict(list)
        for fact_str in task.static:
            if fact_str.startswith('(adjacent '):
                parts = fact_str.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                # dir = parts[3] # Direction is not needed for distance
                self.location_names.add(loc1)
                self.location_names.add(loc2)
                self.adjacency_list[loc1].append(loc2)
                self.adjacency_list[loc2].append(loc1) # Assuming adjacency is symmetric

        self.location_list = sorted(list(self.location_names)) # Consistent order
        self.name_to_id = {name: i for i, name in enumerate(self.location_list)}
        self.id_to_name = {i: name for name, i in self.name_to_id.items()}
        self.num_locations = len(self.location_list)

        # 2. Compute All-Pairs Shortest Paths (APSP) on the full grid
        self.dist_grid = self._compute_apsp()

        # 3. Parse goal facts (box -> goal location)
        self.box_goals = {}
        self.all_boxes = set()
        self.all_goal_locs = set()
        for goal_fact_str in task.goals:
            if goal_fact_str.startswith('(at '):
                parts = goal_fact_str.strip('()').split()
                box_name = parts[1]
                goal_loc_name = parts[2]
                self.box_goals[box_name] = goal_loc_name
                self.all_boxes.add(box_name)
                self.all_goal_locs.add(goal_loc_name)

        # Ensure we have all box names from initial state too, just in case
        # (though problem definition usually lists all objects)
        for fact_str in task.initial_state:
             if fact_str.startswith('(at '):
                parts = fact_str.strip('()').split()
                box_name = parts[1]
                self.all_boxes.add(box_name)

        # Sort box names for consistent ordering in matching
        self.sorted_box_names = sorted(list(self.all_boxes))

        # Check problem structure assumption
        if len(self.all_boxes) != len(self.box_goals):
             # This heuristic assumes a one-to-one mapping between boxes and goals
             # If not, it might indicate an unsolvable problem or a different domain variant.
             # We cannot return inf from __init__, so we store a flag or rely on __call__
             # to detect this mismatch and return inf.
             print("Warning: Number of boxes does not match number of box goals. Heuristic may be inaccurate or return inf.")
             # The check in __call__ before calling Hungarian will handle this.


    def _compute_apsp(self):
        """Computes All-Pairs Shortest Paths on the location grid using BFS."""
        dist_matrix = [[float('inf')] * self.num_locations for _ in range(self.num_locations)]

        for start_id in range(self.num_locations):
            start_loc = self.id_to_name[start_id]
            dist_matrix[start_id][start_id] = 0
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, current_dist = queue.popleft()
                current_id = self.name_to_id[current_loc]

                for neighbor_loc in self.adjacency_list.get(current_loc, []):
                    if neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        neighbor_id = self.name_to_id[neighbor_loc]
                        dist_matrix[start_id][neighbor_id] = current_dist + 1
                        queue.append((neighbor_loc, current_dist + 1))

        return dist_matrix

    def _compute_robot_bfs(self, start_loc, occupied_locations):
        """
        Computes shortest path distances for the robot from start_loc,
        avoiding locations occupied by boxes.
        """
        distances = {loc: float('inf') for loc in self.location_list}
        distances[start_loc] = 0
        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        # Filter occupied_locations to only include boxes (or other non-robot entities)
        # The robot's current location is traversable by the robot.
        box_occupied_locations = {loc for loc in occupied_locations if loc != start_loc}


        while queue:
            current_loc, current_dist = queue.popleft()
            # current_id = self.name_to_id[current_loc] # Not strictly needed here

            for neighbor_loc in self.adjacency_list.get(current_loc, []):
                # Robot can move to neighbor if it's not occupied by a box
                if neighbor_loc not in box_occupied_locations and neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    distances[neighbor_loc] = current_dist + 1
                    queue.append((neighbor_loc, current_dist + 1))

        return distances


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 1. Parse current state
        robot_location = None
        box_locations = {} # box_name -> location_name
        occupied_locations = set() # Locations occupied by robot or boxes

        for fact_str in state:
            if fact_str.startswith('(at-robot '):
                robot_location = fact_str.strip('()').split()[1]
                occupied_locations.add(robot_location)
            elif fact_str.startswith('(at '):
                parts = fact_str.strip('()').split()
                box_name = parts[1]
                loc_name = parts[2]
                box_locations[box_name] = loc_name
                occupied_locations.add(loc_name)

        # Ensure we found robot and all boxes
        if robot_location is None or len(box_locations) != len(self.all_boxes):
             # State is malformed or incomplete, treat as unsolvable
             return float('inf')

        # 2. Identify misplaced boxes
        misplaced_boxes = [b for b in self.sorted_box_names if box_locations.get(b) != self.box_goals.get(b)]

        # 3. If no misplaced boxes, return 0
        if not misplaced_boxes:
            return 0

        # 4. Calculate costs
        h_boxes = 0
        h_robot = float('inf') # Initialize robot cost to infinity

        # a. Calculate box movement cost (h_boxes) using matching
        # We match current locations of ALL boxes to goal locations of ALL boxes
        # This finds the minimum total distance to get the *set* of boxes
        # to the *set* of goal locations.
        if len(self.sorted_box_names) != len(self.box_goals):
             # Problem structure mismatch detected, return inf
             return float('inf')

        all_current_locs = [box_locations[b] for b in self.sorted_box_names]
        all_target_locs = [self.box_goals[b] for b in self.sorted_box_names]

        num_boxes = len(self.sorted_box_names)
        cost_matrix = [[0] * num_boxes for _ in range(num_boxes)]

        for i in range(num_boxes):
            for j in range(num_boxes):
                loc1_name = all_current_locs[i]
                loc2_name = all_target_locs[j]
                loc1_id = self.name_to_id.get(loc1_name)
                loc2_id = self.name_to_id.get(loc2_name)

                if loc1_id is None or loc2_id is None:
                     # Location not found in graph, problem with static facts or state
                     return float('inf')

                cost = self.dist_grid[loc1_id][loc2_id]
                cost_matrix[i][j] = cost


        # Solve matching using Hungarian algorithm
        try:
            hungarian = HungarianAlgorithm(cost_matrix)
            h_boxes = hungarian.solve()
        except AssertionError as e:
             # Handle non-square matrix error if necessary, though problem structure implies square
             print(f"Hungarian algorithm error: {e}")
             return float('inf') # Should not happen if box count == goal count

        # If Hungarian returns inf, propagate it
        if not math.isfinite(h_boxes):
             return float('inf')


        # b. Calculate robot accessibility cost (h_robot)
        # Need robot distance to *any* location occupied by a misplaced box
        misplaced_box_locations = {box_locations[b] for b in misplaced_boxes}

        # Compute robot distances considering current obstacles (other boxes)
        robot_distances = self._compute_robot_bfs(robot_location, occupied_locations)

        # Find minimum distance from robot to any misplaced box location
        min_robot_dist_to_box = float('inf')
        for box_loc in misplaced_box_locations:
             min_robot_dist_to_box = min(min_robot_dist_to_box, robot_distances.get(box_loc, float('inf')))

        h_robot = min_robot_dist_to_box

        # If robot cannot reach any misplaced box, it's a dead end
        if not math.isfinite(h_robot):
             return float('inf')

        # 5. Total Heuristic
        # The sum of box distances and robot distance to the closest box.
        return h_boxes + h_robot
