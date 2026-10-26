import collections
import math
from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing two components:
    1. The minimum number of pushes required for each box to reach any of its
       goal locations, calculated independently for each box ignoring other
       dynamic obstacles.
    2. The minimum number of robot moves required to reach any clear location
       adjacent to any box, considering other boxes as obstacles.

    Assumptions:
    - The PDDL domain follows the standard Sokoban structure with predicates
      like `at-robot`, `at`, `clear`, and `adjacent`.
    - Locations are represented as strings in the format `loc_R_C` where R and C
      are integers.
    - `adjacent` facts define the traversable grid structure, and adjacency is
      assumed to be symmetric (handled by adding edges in both directions).
    - Box goals are specified using `(at boxN loc_R_C)` facts in the goal state.
    - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
    - Parses static `adjacent` facts to build an adjacency map (`adj_map`)
      representing the traversable grid locations and their connections.
    - Creates mappings between location strings (`loc_R_C`) and tuple
      representations (`(R, C)`) for easier processing.
    - Parses goal facts (`task.goals`) to store the set of goal locations for
      each box (`box_goals`). Only boxes mentioned in the goal state are
      considered.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state using `self.task.goal_reached(state)`.
       If true, the heuristic value is 0.
    2. Extract the robot's current location, the current location of each box,
       and the set of clear locations from the input `state`. Convert these
       locations from strings to tuple representations using the pre-computed
       mappings. If the robot's location is not found, return infinity as the
       state is likely invalid.
    3. Calculate the 'box-to-goal cost':
       - Initialize a total cost for boxes to 0.
       - For each box currently present in the state:
         - If the box has goals defined in `self.box_goals`:
           - Find the minimum shortest path distance (number of pushes) from the
             box's current location to any of its goal locations. This distance
             is computed using a Breadth-First Search (BFS) on the grid defined
             by `adj_map`, treating only locations outside the grid as obstacles
             (i.e., ignoring other boxes and the robot).
           - If a box cannot reach any of its goals (BFS returns infinity), the
             state is considered a dead end or requires complex maneuvers not
             captured by this relaxation; return infinity for the heuristic.
           - Add the minimum distance for the current box to the total box cost.
         - If the box does not have goals defined, it does not contribute to the
           goal cost.
    4. Calculate the 'robot-to-box cost':
       - Identify all potential target locations for the robot: these are locations
         that are adjacent to any box's current location AND are currently clear
         in the state.
       - If there are no such potential target locations, the robot cannot get
         into a position to push any box. If the total box cost is greater than 0
         (meaning boxes are not yet at their goals), return infinity as the robot
         is stuck. Otherwise (if total box cost is 0), the goal is reached, which
         was handled in step 1.
       - Calculate the minimum shortest path distance (number of robot moves)
         from the robot's current location to any of these potential target
         locations. This distance is computed using a BFS on the grid, treating
         locations occupied by boxes (excluding the target location itself)
         and locations outside the grid as obstacles.
       - If the robot cannot reach any potential target location (BFS returns
         infinity), return infinity for the heuristic.
    5. The total heuristic value is the sum of the 'box-to-goal cost' and the
       'robot-to-box cost'.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.box_goals = collections.defaultdict(set)
        self.adj_map = collections.defaultdict(set)
        self.loc_str_to_tuple = {}
        self.loc_tuple_to_str = {}

        # 5) Extract information from static facts and goals
        self._parse_static(task.static)
        self._parse_goals(task.goals)

    @staticmethod
    def _parse_location(loc_str):
        # Helper to parse 'loc_R_C' into (R, C)
        # Assumes loc_str is clean, e.g., 'loc_4_7'
        parts = loc_str.split('_')
        # Basic validation
        if len(parts) == 3 and parts[0] == 'loc':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                # Should not happen with valid PDDL, but good practice
                # print(f"Warning: Could not parse location string {loc_str}")
                return None
        # Should not happen with valid PDDL
        # print(f"Warning: Unexpected location string format {loc_str}")
        return None


    def _parse_static(self, static_facts):
        # Build adjacency map and location string/tuple mappings
        for fact_str in static_facts:
            # Example: '(adjacent loc_4_2 loc_4_3 right)'
            parts = fact_str.strip('()').split()
            if parts[0] == 'adjacent' and len(parts) >= 3: # Ensure enough parts
                loc1_str = parts[1]
                loc2_str = parts[2]
                loc1_tuple = self._parse_location(loc1_str)
                loc2_tuple = self._parse_location(loc2_str)

                if loc1_tuple is not None and loc2_tuple is not None:
                    self.adj_map[loc1_tuple].add(loc2_tuple)
                    self.adj_map[loc2_tuple].add(loc1_tuple) # Assuming adjacency is symmetric

                    self.loc_str_to_tuple[loc1_str] = loc1_tuple
                    self.loc_tuple_to_str[loc1_tuple] = loc1_str
                    self.loc_str_to_tuple[loc2_str] = loc2_tuple
                    self.loc_tuple_to_str[loc2_tuple] = loc2_str
                # else:
                     # print(f"Warning: Skipping invalid adjacent fact: {fact_str}")


    def _parse_goals(self, goal_facts):
        # Extract box goal locations
        # Goal facts are typically a frozenset of strings like state facts
        # Example: '(at box1 loc_2_4)'
        # Note: task.goals is already a frozenset of strings
        for fact_str in goal_facts:
            parts = fact_str.strip('()').split()
            # Check for '(at boxN loc_R_C)' pattern
            if len(parts) == 3 and parts[0] == 'at' and parts[1].startswith('box'):
                box_name = parts[1]
                loc_str = parts[2]
                loc_tuple = self._parse_location(loc_str)
                if loc_tuple is not None:
                    self.box_goals[box_name].add(loc_tuple)
                # else:
                     # print(f"Warning: Skipping invalid goal fact location: {fact_str}")


    def _shortest_path_bfs(self, start_node, end_node, obstacles):
        # BFS on the grid graph
        if start_node == end_node:
            return 0
        # Check if start or end node are even part of the defined grid
        if start_node not in self.adj_map or end_node not in self.adj_map:
             return math.inf
        # Check if start node is an obstacle (should not happen for robot start
        # in a valid state, but included for robustness).
        if start_node in obstacles:
             return math.inf

        q = collections.deque([(start_node, 0)])
        visited = {start_node}

        while q:
            current_node, dist = q.popleft()

            if current_node == end_node:
                return dist

            # Only explore neighbors that are part of the grid and not obstacles
            if current_node in self.adj_map:
                for neighbor in self.adj_map[current_node]:
                    if neighbor not in visited and neighbor not in obstacles:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

        return math.inf # End node not reachable


    def __call__(self, node):
        state = node.state

        # 1) Check for goal state
        if self.task.goal_reached(state):
            return 0

        # 2) & 3) Parse State Information
        robot_loc_tuple = None
        box_locations_tuple = {} # box_name -> loc_tuple
        clear_locations_tuple = set() # loc_tuple

        for fact_str in state:
            parts = fact_str.strip('()').split()
            if len(parts) >= 2: # Ensure minimum parts
                predicate = parts[0]
                if predicate == 'at-robot':
                    loc_str = parts[1]
                    robot_loc_tuple = self._parse_location(loc_str)
                elif predicate == 'at' and len(parts) >= 3 and parts[1].startswith('box'):
                    box_name = parts[1]
                    loc_str = parts[2]
                    loc_tuple = self._parse_location(loc_str)
                    if loc_tuple is not None:
                        box_locations_tuple[box_name] = loc_tuple
                elif predicate == 'clear':
                     loc_str = parts[1]
                     loc_tuple = self._parse_location(loc_str)
                     if loc_tuple is not None:
                        clear_locations_tuple.add(loc_tuple)

        # Basic check if robot location was found
        if robot_loc_tuple is None:
             # This state is likely invalid or represents a problem
             # where robot location is not tracked, but Sokoban requires it.
             return math.inf

        # 3) Calculate box-to-goal-cost
        total_box_cost = 0
        # Iterate over boxes currently in the state
        for box_name, loc_b_tuple in box_locations_tuple.items():
            # Only consider boxes that actually have goals defined
            if box_name not in self.box_goals or not self.box_goals[box_name]:
                 # This box doesn't need to go anywhere or has no valid goals.
                 # Its cost component is 0.
                 continue

            min_dist = math.inf
            for loc_g_tuple in self.box_goals[box_name]:
                # BFS for box ignores other boxes and robot as obstacles
                # Obstacles for box movement are only walls (locations not in adj_map)
                dist = self._shortest_path_bfs(loc_b_tuple, loc_g_tuple, obstacles=set())
                min_dist = min(min_dist, dist)

            if min_dist == math.inf:
                # Box cannot reach any of its goals through traversable grid locations
                return math.inf
            total_box_cost += min_dist

        # 4) Calculate robot-to-box-cost
        potential_robot_targets_tuple = set()
        # Find clear locations adjacent to any box
        for loc_b_tuple in box_locations_tuple.values():
            if loc_b_tuple in self.adj_map: # Ensure box is on the grid
                for neighbor_tuple in self.adj_map[loc_b_tuple]:
                    # Robot can push from neighbor_tuple if it's clear
                    if neighbor_tuple in clear_locations_tuple:
                        potential_robot_targets_tuple.add(neighbor_tuple)

        min_dist_robot = math.inf

        # If there are no clear locations adjacent to any box, the robot cannot push.
        # If total_box_cost > 0, it's a dead end.
        if not potential_robot_targets_tuple:
             if total_box_cost > 0:
                 return math.inf
             else:
                 # This case should ideally be caught by the goal_reached check.
                 # If total_box_cost is 0, all boxes are at goals, h should be 0.
                 # If we reach here, something is inconsistent, but returning 0
                 # seems safest if boxes are at goals. However, the goal_reached
                 # check should prevent this path. Let's return inf as a fallback
                 # for unexpected states where pushes are needed but impossible.
                 return math.inf # Robot cannot enable any push

        # Obstacles for robot BFS: locations occupied by boxes
        robot_obstacles_tuple = set(box_locations_tuple.values())

        for loc_r_target_tuple in potential_robot_targets_tuple:
            # Robot BFS considers box locations as obstacles
            # The target location itself is NOT an obstacle for the robot trying to reach it
            current_robot_obstacles = robot_obstacles_tuple - {loc_r_target_tuple}
            dist = self._shortest_path_bfs(robot_loc_tuple, loc_r_target_tuple, obstacles=current_robot_obstacles)
            min_dist_robot = min(min_dist_robot, dist)

        if min_dist_robot == math.inf:
            # Robot cannot reach any clear location adjacent to a box
            return math.inf

        # 5) Total heuristic
        return total_box_cost + min_dist_robot
