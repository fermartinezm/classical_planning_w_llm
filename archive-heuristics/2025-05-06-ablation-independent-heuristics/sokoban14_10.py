from collections import deque
import math

from heuristics.heuristic_base import Heuristic
# The Task class is implicitly available via the Heuristic base class constructor


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    individual costs for each box that is not yet at its goal location.
    The cost for a single box is estimated as the sum of:
    1. The shortest path distance for the box to reach its goal location,
       calculated on the grid graph considering static obstacles (walls)
       and other boxes that are already at their goal locations as obstacles.
       This represents the minimum number of pushes required for the box
       in a relaxed setting where other non-goal boxes can be moved freely.
    2. The minimum shortest path distance for the robot to reach a valid
       pushing position adjacent to the box, from which the box can be pushed
       one step towards its goal. This distance is calculated considering
       static obstacles and all boxes as obstacles for the robot's movement.
       This represents the robot's effort to set up the first push for this box.

    The heuristic is non-admissible but aims to guide a greedy best-first
    search efficiently by prioritizing states where boxes are closer to their
    goals and the robot is well-positioned to make progress.

    Assumptions:
    - Location names are in the format 'loc_ROW_COL' where ROW and COL are integers.
    - The grid structure is defined by 'adjacent' facts.
    - Each box has a unique goal location specified in the task goals.
    - The grid is a standard 2D grid (although the adjacency facts define the actual graph).

    Heuristic Initialization:
    The constructor pre-processes the static information from the task:
    - Parses 'adjacent' facts to build a graph representation of the grid,
      mapping location strings to (row, col) tuples and vice versa.
      It stores adjacency lists (`grid_adj`) and directed adjacency maps (`grid_adj_dir`).
    - Identifies all locations present in the problem (from initial state, goals, and adjacent facts).
    - Determines static obstacles (walls) as locations present in the problem
      but not part of the traversable grid defined by 'adjacent' facts.
    - Parses goal facts to map each box to its specific goal location (`box_goals`).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Identify the current location of the robot and all boxes from the state facts.
    3. Initialize the total heuristic value `h` to 0.
    4. Determine the coordinates of all boxes and the robot.
    5. Identify locations of boxes that are currently at their goal locations. These are obstacles for other boxes.
    6. Identify locations of all boxes. These are obstacles for the robot.
    7. For each box that is not yet at its goal location:
        a. Calculate the shortest path distance for the box from its current location
           to its goal location using BFS on the grid graph. Obstacles for this BFS
           include static walls and boxes already at their goals. If the box cannot
           reach its goal even under this relaxation, the state is likely a dead end,
           return infinity for the total heuristic.
        b. Find the minimum cost for the robot to reach a valid pushing position
           for this box, considering pushes that move the box towards its goal
           (reduce Manhattan distance to the goal).
           - Iterate through the four possible push directions (up, down, left, right).
           - For each direction, determine the required robot location (adjacent to the box,
             opposite the push direction) and the box's target location (adjacent to the box,
             in the push direction).
           - Check if the push is valid in the current state:
             - The box's target location must be a valid grid location and must be clear
               (not occupied by the robot or any box).
             - The required robot pushing location must be a valid grid location and
               must be clear of any boxes (the robot might already be there or needs to move there).
             - The push must move the box closer to its goal (reduce Manhattan distance).
           - If the push is valid, calculate the shortest path distance for the robot
             from its current location to the required pushing location using BFS.
             Obstacles for this robot BFS include static walls and all boxes.
           - Keep track of the minimum robot distance found among all valid push directions.
        c. If no valid push towards the goal is possible for this box, it might be stuck,
           return infinity for the total heuristic.
        d. Add the minimum robot distance found (for the first push) plus the box's
           shortest path distance (minimum pushes) to the total heuristic `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static

        # Map location strings to (row, col) tuples and vice versa
        self.loc_to_coords = {}
        self.coords_to_loc = {}
        # Build adjacency list representation of the grid
        self.grid_adj = {}  # {(r, c): [(nr1, nc1), (nr2, nc2), ...], ...}
        self.grid_adj_dir = {}  # {(r, c): {direction_str: (nr, nc), ...}, ...}
        self.grid_locations_str = set() # Set of location strings present in adjacent facts

        # Parse adjacent facts to build grid graph and location mappings
        for fact in self.static_facts:
            if fact.startswith('(adjacent'):
                parts = fact.strip('()').split()
                loc1_str = parts[1]
                loc2_str = parts[2]
                direction_str = parts[3]

                # Add locations to mapping if not already present
                if loc1_str not in self.loc_to_coords:
                    # Assuming loc_X_Y format
                    r1, c1 = map(int, loc1_str.split('_')[1:])
                    self.loc_to_coords[loc1_str] = (r1, c1)
                    self.coords_to_loc[(r1, c1)] = loc1_str
                    self.grid_adj[(r1, c1)] = []
                    self.grid_adj_dir[(r1, c1)] = {}
                if loc2_str not in self.loc_to_coords:
                    r2, c2 = map(int, loc2_str.split('_')[1:])
                    self.loc_to_coords[loc2_str] = (r2, c2)
                    self.coords_to_loc[(r2, c2)] = loc2_str
                    self.grid_adj[(r2, c2)] = []
                    self.grid_adj_dir[(r2, c2)] = {}

                coords1 = self.loc_to_coords[loc1_str]
                coords2 = self.loc_to_coords[loc2_str]

                # Add adjacency
                self.grid_adj[coords1].append(coords2)
                self.grid_adj_dir[coords1][direction_str] = coords2

                self.grid_locations_str.add(loc1_str)
                self.grid_locations_str.add(loc2_str)

        # Map boxes to goal locations
        self.box_goals = {}  # {box_name: goal_loc_str, ...}
        all_problem_locations_str = set(self.grid_locations_str) # Start with grid locations

        for goal in self.goals:
            if goal.startswith('(at '):
                parts = goal.strip('()').split()
                box_name = parts[1]
                goal_loc_str = parts[2]
                self.box_goals[box_name] = goal_loc_str
                all_problem_locations_str.add(goal_loc_str)

        # Static obstacles are locations in the problem but not in the grid
        # We need to find all locations mentioned in the initial state as well
        for fact in task.initial_state:
             if fact.startswith('(at-robot ') or fact.startswith('(at '):
                 parts = fact.strip('()').split()
                 all_problem_locations_str.add(parts[-1]) # Last part is location

        self.static_obstacle_coords = {
            self.loc_to_coords[loc_str]
            for loc_str in all_problem_locations_str
            if loc_str not in self.grid_locations_str
        }

    def get_coords_in_direction(self, start_coords, direction_str):
        """Returns coordinates of the adjacent location in the given direction, or None."""
        return self.grid_adj_dir.get(start_coords, {}).get(direction_str)

    def get_opposite_direction(self, direction_str):
        """Returns the opposite direction string."""
        if direction_str == 'up': return 'down'
        if direction_str == 'down': return 'up'
        if direction_str == 'left': return 'right'
        if direction_str == 'right': return 'left'
        return None # Should not happen with valid directions

    def bfs(self, start_coords, goal_coords_set, grid_adj, obstacles):
        """
        Performs BFS on the grid graph to find the shortest path distance.

        Args:
            start_coords: The starting coordinates (row, col).
            goal_coords_set: A set of goal coordinates.
            grid_adj: The adjacency list representation of the grid graph.
            obstacles: A set of coordinates that are obstacles.

        Returns:
            The shortest distance, or float('inf') if no path exists.
        """
        if start_coords in goal_coords_set:
            return 0
        queue = deque([(start_coords, 0)])
        visited = {start_coords}
        while queue:
            curr_coords, dist = queue.popleft()

            # Check neighbors using grid_adj
            for next_coords in grid_adj.get(curr_coords, []):
                if next_coords not in visited and next_coords not in obstacles:
                    if next_coords in goal_coords_set:
                        return dist + 1
                    visited.add(next_coords)
                    queue.append((next_coords, dist + 1))
        return math.inf

    def bfs_robot(self, start_coords, goal_coords_set, grid_adj, static_obstacles, box_coords):
        """
        Performs BFS for the robot on the grid graph, avoiding static obstacles and boxes.

        Args:
            start_coords: The starting coordinates (row, col).
            goal_coords_set: A set of goal coordinates for the robot.
            grid_adj: The adjacency list representation of the grid graph.
            static_obstacles: A set of static obstacle coordinates.
            box_coords: A set of coordinates occupied by boxes.

        Returns:
            The shortest distance, or float('inf') if no path exists.
        """
        if start_coords in goal_coords_set:
            return 0
        queue = deque([(start_coords, 0)])
        visited = {start_coords}
        # Robot cannot enter a cell with a box or a static obstacle
        obstacles = static_obstacles | box_coords
        while queue:
            curr_coords, dist = queue.popleft()

            # Check neighbors using grid_adj
            for next_coords in grid_adj.get(curr_coords, []):
                if next_coords not in visited and next_coords not in obstacles:
                    if next_coords in goal_coords_set:
                        return dist + 1
                    visited.add(next_coords)
                    queue.append((next_coords, dist + 1))
        return math.inf


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The heuristic value (estimated cost to goal), or float('inf') if
            the state is estimated to be a dead end.
        """
        state = node.state

        # 1. Check if the state is a goal state
        if self.task.goal_reached(state):
            return 0

        # 2. Identify robot and box locations
        r_loc_str = None
        box_locs_str = {} # {box_name: loc_str}
        for fact in state:
            if fact.startswith('(at-robot '):
                r_loc_str = fact.strip('()').split()[-1]
            elif fact.startswith('(at '):
                parts = fact.strip('()').split()
                box_name = parts[1]
                loc_str = parts[2]
                box_locs_str[box_name] = loc_str

        if r_loc_str is None:
             # Should not happen in valid Sokoban states, but handle defensively
             return math.inf

        r_coords = self.loc_to_coords.get(r_loc_str)
        if r_coords is None:
             # Robot is in a location not part of the grid? Should not happen.
             return math.inf

        # 3. Initialize total heuristic
        h = 0

        # 4. Determine coordinates of all boxes
        all_box_coords = {
            self.loc_to_coords.get(loc)
            for loc in box_locs_str.values()
            if loc in self.loc_to_coords # Ensure location is in our grid map
        }
        # Remove None values in case a box location wasn't in the grid map (shouldn't happen)
        all_box_coords.discard(None)


        # 5. Identify locations of boxes at their goals (obstacles for other boxes)
        boxes_at_goal_coords = {
            self.loc_to_coords.get(loc)
            for box, loc in box_locs_str.items()
            if self.box_goals.get(box) == loc and loc in self.loc_to_coords # Ensure box has a goal and loc is in grid map
        }
        boxes_at_goal_coords.discard(None)

        # Obstacles for box BFS: static walls + boxes at goal
        box_bfs_obstacles = self.static_obstacle_coords | boxes_at_goal_coords

        # Obstacles for robot BFS: static walls + all boxes
        robot_bfs_obstacles = self.static_obstacle_coords | all_box_coords

        # 7. For each box not at its goal
        for box, b_loc_str in box_locs_str.items():
            g_loc_str = self.box_goals.get(box)

            # If box has no goal or is already at goal, skip
            if g_loc_str is None or b_loc_str == g_loc_str:
                continue

            b_coords = self.loc_to_coords.get(b_loc_str)
            g_coords = self.loc_to_coords.get(g_loc_str)

            # If box or goal location not in grid map, treat as unsolvable
            if b_coords is None or g_coords is None:
                 return math.inf

            # a. Calculate box path length (min pushes)
            box_path_len = self.bfs(b_coords, {g_coords}, self.grid_adj, box_bfs_obstacles)

            # If box cannot reach goal even relaxed, it's a dead end
            if box_path_len == math.inf:
                return math.inf

            # b. Find minimum robot cost for the first push
            min_robot_first_push_cost = math.inf

            # Calculate initial Manhattan distance to goal for push direction check
            initial_md = abs(b_coords[0] - g_coords[0]) + abs(b_coords[1] - g_coords[1])

            for dir_str in ['up', 'down', 'left', 'right']:
                floc_coords = self.get_coords_in_direction(b_coords, dir_str)

                # Check if target location is valid
                if floc_coords is None:
                    continue

                push_from_coords = self.get_coords_in_direction(b_coords, self.get_opposite_direction(dir_str))

                # Check if pushing location is valid
                if push_from_coords is None:
                    continue

                # Check if this push moves the box towards the goal (reduces Manhattan distance)
                floc_is_closer = abs(floc_coords[0] - g_coords[0]) + abs(floc_coords[1] - g_coords[1]) < initial_md

                # Check if floc_coords is clear in the current state
                # A location is clear if it's not the robot's location and not occupied by any box
                floc_is_clear = floc_coords != r_coords and floc_coords not in all_box_coords

                # Check if push_from_coords is clear of boxes (robot can move here)
                # The robot's current location is not an obstacle for the BFS path calculation itself,
                # but the target location must be free of boxes for the robot to occupy it.
                push_from_is_clear_of_boxes = push_from_coords not in all_box_coords

                # If the push is valid and moves towards the goal
                if floc_is_closer and floc_is_clear and push_from_is_clear_of_boxes:
                    # Calculate robot distance to the pushing position
                    robot_dist = self.bfs_robot(r_coords, {push_from_coords}, self.grid_adj, self.static_obstacle_coords, all_box_coords)

                    if robot_dist != math.inf:
                        min_robot_first_push_cost = min(min_robot_first_push_cost, robot_dist)

            # If robot cannot get into position for any valid push towards goal
            if min_robot_first_push_cost == math.inf:
                return math.inf # Dead end

            # h. Add contribution for this box
            # The cost is robot moves to position + box pushes.
            # The first push is enabled by min_robot_first_push_cost moves.
            # Subsequent box_path_len - 1 pushes are assumed to cost 1 each (ignoring robot repositioning cost between pushes for simplicity).
            # A simpler estimate is just sum of robot moves to first push + total box pushes.
            h += min_robot_first_push_cost + box_path_len

        # 8. Return total heuristic
        return h

