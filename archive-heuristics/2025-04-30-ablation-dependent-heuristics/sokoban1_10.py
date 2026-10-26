import re
from collections import defaultdict, deque
import math

# Helper function to parse location string like 'loc_row_col'
def parse_location_string(loc_str):
    """Parses a location string 'loc_row_col' into a tuple (row, col)."""
    match = re.match(r'loc_(\d+)_(\d+)', loc_str)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return f'loc_{row}_{col}', (row, col) # Return both string and tuple
    return None, None

# Helper function to parse location from a fact string like '(at-robot loc_6_4)'
def parse_location_from_fact(fact_str):
    """Extracts and parses a location string from a PDDL fact string."""
    match = re.search(r'loc_(\d+)_(\d+)', fact_str)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return f'loc_{row}_{col}', (row, col) # Return both string and tuple
    return None, None

# Helper function to get coordinates from a location string
def get_coords(loc_str):
    """Returns the (row, col) tuple for a location string."""
    _, coords = parse_location_string(loc_str)
    return coords

# Helper function to get location string from coordinates
def get_loc_string(coords):
    """Returns the location string 'loc_row_col' for a coordinate tuple."""
    if coords:
        return f'loc_{coords[0]}_{coords[1]}'
    return None

# Map directions to coordinate changes (relative)
direction_map = {'down': (1, 0), 'up': (-1, 0), 'right': (0, 1), 'left': (0, -1)}
# Map directions to their opposites
opposite_direction_map = {'down': 'up', 'up': 'down', 'right': 'left', 'left': 'right'}

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    costs for each box to reach its target goal location. The estimated cost
    for a single box is the sum of:
    1. The minimum number of pushes required to move the box from its current
       location to its goal location, considering currently clear cells and
       other boxes/robot as obstacles for the box's path, and only allowing
       moves for which a valid pushing position exists.
    2. The minimum number of robot moves required to reach the pushing position
       needed for the *first* push along the box's shortest path to the goal,
       considering other boxes as obstacles for the robot.

    Assumptions:
    - The grid structure is defined by `adjacent` facts.
    - Locations are named `loc_row_col` allowing coordinate extraction.
    - Each box has a unique target goal location specified in the task goals.
    - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
    - For any two adjacent locations L1, L2, if a box can be pushed from L1 to L2,
      there is a unique location L0 such that the robot at L0 can push the box
      at L1 to L2. This is typically the location adjacent to L1 on the opposite
      side of L2, and this relationship is captured by the `adjacent` facts.

    Heuristic Initialization:
    The constructor pre-computes the grid graph from `adjacent` facts and
    builds a map from (box_current_loc, box_next_loc) pairs to the required
    robot pushing location, based on the grid structure defined by the
    `adjacent` facts. This static information is extracted from the `task.static` facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the state is a goal state using `self.task.goal_reached(state)`.
       If yes, the heuristic is 0.
    2. Extract the robot's current location coordinates (`robot_loc_coords`)
       from the state.
    3. Extract the current location coordinates of each box (`box_locs_coords`)
       from the state.
    4. Extract the set of currently clear location coordinates (`clear_locs_coords`)
       from the state.
    5. Retrieve the target goal location coordinates for each box
       (`self.goal_box_locations`) which were pre-computed from the task goals.
    6. Initialize the total heuristic value to 0.
    7. Iterate through each box and its current location:
        a. Get the box's current location coordinates (`box_loc_coords`) and its
           goal location coordinates (`goal_loc_coords`).
        b. Identify the location coordinates of all *other* boxes (`other_box_coords`).
        c. Calculate the minimum number of pushes (`pushes`) and the coordinates
           of the first step (`first_step_coords`) required for this box to reach
           its GoalLoc using a BFS (`_min_pushes_bfs`) on box locations. This BFS considers:
           - The grid connectivity (`self.graph`).
           - That the target cell for a push must be clear or the goal (`clear_locs_coords`, `goal_loc_coords`).
           - That the target cell cannot be occupied by another box or the robot (`other_box_coords`, `robot_loc_coords`).
           - That a valid pushing position must exist for the move, checked via `self.pushing_pos_map`.
        d. If `pushes` is infinity, it means the box cannot reach its goal from this state.
           The state is likely unsolvable, so return infinity for the total heuristic.
        e. If `pushes` is 0, the box is already at the goal. The cost for this box is 0.
        f. If `pushes > 0`, determine the required robot pushing location coordinates
           (`pushing_pos_coords`) for the first step (`first_step_coords`) using the
           pre-computed `self.pushing_pos_map`.
        g. Calculate the minimum number of robot moves (`robot_dist`) required for the
           robot to reach `pushing_pos_coords` using a BFS (`_bfs_distance_coords`)
           on grid locations. This BFS considers locations occupied by *any* box
           (including the one being pushed) as obstacles for the robot.
        h. If `robot_dist` is infinity, the robot cannot reach the required pushing
           position. The state is likely unsolvable, so return infinity.
        i. The estimated cost for this box is `pushes + robot_dist`.
        j. Add this cost to the total heuristic.
    8. Return the total heuristic value.
    """
    def __init__(self, task):
        self.task = task
        self.graph = defaultdict(list)
        # {(box_loc_coords, next_box_loc_coords): robot_pushing_coords}
        self.pushing_pos_map = {}
        self.goal_box_locations = self._get_goal_box_locations(task.goals)

        # Build grid graph and adjacency by direction
        # {loc_coords: {direction: [neighbor_coords]}}
        adj_by_dir = defaultdict(lambda: defaultdict(list))
        # {loc_coords: {direction: [neighbor_coords]}} # neighbor is adjacent to loc in direction
        rev_adj_by_dir = defaultdict(lambda: defaultdict(list))

        for fact in task.static:
            if fact.startswith('(adjacent'):
                parts = fact.strip('()').split()
                loc1_str = parts[1]
                loc2_str = parts[2]
                dir_str = parts[3]
                _, loc1_coords = parse_location_string(loc1_str)
                _, loc2_coords = parse_location_string(loc2_str)
                if loc1_coords and loc2_coords:
                    self.graph[loc1_coords].append(loc2_coords)
                    # Assuming adjacency is symmetric, add reverse edge for graph
                    self.graph[loc2_coords].append(loc1_coords)

                    # Store directed adjacency for pushing map calculation
                    adj_by_dir[loc1_coords][dir_str].append(loc2_coords)
                    # Store reverse directed adjacency (l1 is adjacent to l2 in direction dir)
                    rev_adj_by_dir[loc2_coords][dir_str].append(loc1_coords)

        # Build the pushing_pos_map
        # To push box from L1 to L2 (where L2 is adjacent to L1 in direction D),
        # robot must be at L0 where L0 is adjacent to L1 in direction Opposite(D).
        for l1_coords in self.graph:
            for l2_coords in self.graph.get(l1_coords, []):
                # Find the direction D from L1 to L2
                direction_from_l1_to_l2 = None
                # Iterate through directions from L1 to find L2
                for d, neighbors in adj_by_dir[l1_coords].items():
                    if l2_coords in neighbors:
                        direction_from_l1_to_l2 = d
                        break

                if direction_from_l1_to_l2:
                    opposite_dir = opposite_direction_map.get(direction_from_l1_to_l2)
                    if opposite_dir:
                        # Find L0 such that L0 is adjacent to L1 in opposite_dir
                        # L0 is a neighbor of L1, and L1 is adjacent to L0 in opposite_dir
                        pushing_locations = rev_adj_by_dir[l1_coords].get(opposite_dir, [])

                        # For a grid, there should be at most one such location.
                        if pushing_locations:
                             # Assuming unique pushing position for a given push direction
                             self.pushing_pos_map[(l1_coords, l2_coords)] = pushing_locations[0]
                        # else: This push L1->L2 is impossible because there's no space behind L1 for the robot.
                        # We don't add it to the map, and the BFS will implicitly not explore this move.


    def _get_goal_box_locations(self, goals):
        """Extracts goal locations for each box from the task goals."""
        goal_locs = {} # {box_name: goal_loc_coords}
        for fact in goals:
            if fact.startswith('(at box'):
                parts = fact.strip('()').split()
                box_name = parts[1]
                loc_str = parts[2]
                _, loc_coords = parse_location_string(loc_str)
                if loc_coords:
                    goal_locs[box_name] = loc_coords
        return goal_locs

    def _get_state_info(self, state):
        """Extracts robot location, box locations, and clear locations from the state."""
        robot_loc_coords = None
        box_locs_coords = {} # {box_name: loc_coords}
        clear_locs_coords = set() # {loc_coords}

        for fact in state:
            if fact.startswith('(at-robot'):
                _, robot_loc_coords = parse_location_from_fact(fact)
            elif fact.startswith('(at box'):
                parts = fact.strip('()').split()
                box_name = parts[1]
                loc_str = parts[2]
                _, loc_coords = parse_location_string(loc_str)
                if loc_coords:
                    box_locs_coords[box_name] = loc_coords
            elif fact.startswith('(clear'):
                 _, loc_coords = parse_location_from_fact(fact)
                 if loc_coords:
                    clear_locs_coords.add(loc_coords)

        return robot_loc_coords, box_locs_coords, clear_locs_coords

    def _bfs_distance_coords(self, start_coords, end_coords, obstacles=None):
        """
        Calculates shortest path distance using BFS on the grid graph.
        Obstacles are locations (coords) that cannot be entered.
        Returns distance or float('inf').
        """
        if obstacles is None:
            obstacles = set()

        if start_coords == end_coords:
            return 0
        # If start is an obstacle, it's unreachable from anywhere else, but we start *at* it.
        # if start_coords in obstacles:
        #     return float('inf')

        queue = deque([(start_coords, 0)])
        visited = {start_coords}

        while queue:
            current_coords, dist = queue.popleft()

            for neighbor_coords in self.graph.get(current_coords, []):
                if neighbor_coords not in visited and neighbor_coords not in obstacles:
                    if neighbor_coords == end_coords:
                        return dist + 1
                    visited.add(neighbor_coords)
                    queue.append((neighbor_coords, dist + 1))

        return float('inf') # Not reachable

    def _min_pushes_bfs(self, start_box_coords, goal_box_coords, current_clear_coords, other_box_coords, robot_coords):
        """
        Calculates minimum pushes for a box to reach a goal using BFS on box locations.
        Considers current clear cells, other boxes/robot as obstacles for the box's path,
        AND whether a valid pushing position exists for each step.
        Returns (min_pushes, first_step_coords) or (float('inf'), None).
        """
        if start_box_coords == goal_box_coords:
            return 0, None # Already at goal

        queue = deque([(start_box_coords, 0, None)]) # (current_box_coords, pushes, first_step_coords)
        visited = {start_box_coords}

        # Obstacles for the box's path: other boxes and the robot
        box_path_obstacles = set(other_box_coords)
        if robot_coords:
             box_path_obstacles.add(robot_coords)

        while queue:
            current_coords, pushes, first_step_coords = queue.popleft()

            for neighbor_coords in self.graph.get(current_coords, []):
                # A box can move from current_coords (L1) to neighbor_coords (L2)
                # if:
                # 1. A valid pushing position exists for this move (L1, L2) -> pushing_pos_map[(L1, L2)] exists.
                # 2. L2 is clear OR L2 is the goal location.
                # 3. L2 is not occupied by another box or the robot (box_path_obstacles).

                if (current_coords, neighbor_coords) in self.pushing_pos_map:
                    is_clear_or_goal = neighbor_coords in current_clear_coords or neighbor_coords == goal_box_coords
                    is_obstacle = neighbor_coords in box_path_obstacles

                    if is_clear_or_goal and not is_obstacle and neighbor_coords not in visited:
                        visited.add(neighbor_coords)
                        step = neighbor_coords if pushes == 0 else first_step_coords
                        if neighbor_coords == goal_box_coords:
                             return pushes + 1, step # Found path to goal
                        queue.append((neighbor_coords, pushes + 1, step))

        return float('inf'), None # Goal not reachable for the box


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        robot_loc_coords, box_locs_coords, clear_locs_coords = self._get_state_info(state)

        total_heuristic = 0

        # If there are no boxes or no goals defined for boxes, and goal is not reached,
        # something is wrong or it's an unsolvable state in practice.
        if not box_locs_coords or not self.goal_box_locations:
             return float('inf')

        for box_name, box_loc_coords in box_locs_coords.items():
            goal_loc_coords = self.goal_box_locations.get(box_name)

            # If a box doesn't have a goal location specified, it's unsolvable or problem is malformed
            if goal_loc_coords is None:
                 return float('inf')

            # Locations of other boxes for obstacle calculation
            other_box_coords = {loc for name, loc in box_locs_coords.items() if name != box_name}

            # Calculate minimum pushes for this box to reach its goal
            pushes, first_step_coords = self._min_pushes_bfs(
                box_loc_coords,
                goal_loc_coords,
                clear_locs_coords,
                other_box_coords,
                robot_loc_coords # Robot is an obstacle for the box's path
            )

            if pushes == float('inf'):
                # Box cannot reach its goal
                return float('inf') # State is likely unsolvable

            if pushes == 0:
                # Box is already at the goal, cost is 0 for this box
                cost_for_box = 0
            else:
                # Calculate robot distance to the pushing position for the first step
                # The pushing position is relative to the box's current location and the first step location
                pushing_pos_coords = self.pushing_pos_map.get((box_loc_coords, first_step_coords))

                if pushing_pos_coords is None:
                     # This should not happen if min_pushes_bfs only explored valid pushes,
                     # but as a safeguard, if the first step is somehow invalid for pushing.
                     # This state is likely unsolvable.
                     return float('inf')

                # Robot cannot move into any box location (including the box it's about to push)
                robot_obstacles = set(box_locs_coords.values())

                robot_dist = self._bfs_distance_coords(
                    robot_loc_coords,
                    pushing_pos_coords,
                    obstacles=robot_obstacles
                )

                if robot_dist == float('inf'):
                    # Robot cannot reach the required pushing position
                    return float('inf') # State is likely unsolvable

                cost_for_box = pushes + robot_dist

            total_heuristic += cost_for_box

        return total_heuristic

