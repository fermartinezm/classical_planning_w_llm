import collections
import math

# Assume Heuristic base class is available in the environment
from heuristics.heuristic_base import Heuristic

class sokobanHeuristic(Heuristic):
    """
    Sokoban Domain-Dependent Heuristic.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing two components:
    1. The sum of shortest path distances (on an empty grid) for each box from its
       current location to its assigned goal location. This estimates the minimum
       number of push actions required.
    2. The minimum cost for the robot to reach a position from which it can perform
       a "useful" push action. A useful push action is one that moves a box
       towards its goal location along a shortest path on the empty grid. The cost
       is the number of robot moves required to reach the push position plus one
       (for the push action itself). Robot movement considers other boxes as obstacles.
    The heuristic returns 0 if the goal is reached and infinity if no useful push
    position is reachable for any box that needs pushing.

    Assumptions:
    - The PDDL instance defines a grid-like structure using `adjacent` facts.
    - `adjacent` facts are provided for both directions (e.g., if A is adjacent to B 'right', B is adjacent to A 'left').
    - Each box has a unique assigned goal location specified in the goal state.
    - Location names follow a consistent format (e.g., 'loc_row_col') although the heuristic does not rely on parsing row/col.
    - The set of locations is finite and defined by the `adjacent` facts in the static information.

    Heuristic Initialization:
    1. Parses `adjacent` facts from `task.static` to build a bidirectional graph
       representation of the grid connectivity (`self.graph`).
    2. Parses `at` facts from `task.goals` to determine the assigned goal location
       for each box (`self.box_goals`).
    3. Precomputes all-pairs shortest path distances on the empty grid graph using BFS
       (`self.distances`). This allows quick lookup of the minimum number of steps
       (pushes) required for a box to reach its goal without considering obstacles.
    4. Stores a mapping of directions to their reverse (`self.reverse_dir`).

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state. If yes, return 0.
    2. Parse the current state to find the robot's location (`loc_robot`),
       the location of each box (`loc_boxes`), and the set of clear locations (`clear_locs`).
       Also identify all occupied locations (`occupied_locs`).
    3. Calculate the first component (`h1`):
       - Initialize `h1 = 0`.
       - For each box and its current location (`loc_b`) in `loc_boxes`:
         - Get the box's assigned goal location (`goal_loc`) from `self.box_goals`.
         - If the box is not at its goal (`loc_b != goal_loc`):
           - Look up the precomputed shortest distance from `loc_b` to `goal_loc`
             in `self.distances`.
           - If the distance is infinity (box is disconnected from goal), return infinity.
           - Add this distance to `h1`.
       - `h1` represents the sum of minimum pushes needed for all boxes.
    4. Calculate the second component (`h2`):
       - Initialize `min_robot_dist_to_useful_push_pos = math.inf`.
       - Identify locations occupied by other boxes (obstacles for robot movement):
         `robot_obstacles = occupied_locs - {loc_robot}`.
       - Iterate through each box that is not yet at its goal.
       - For each such box at `loc_b` with goal `goal_loc`:
         - Get the precomputed distance from `loc_b` to `goal_loc` (`current_box_goal_dist`).
         - If `current_box_goal_dist` is infinity, return infinity (already checked in h1, but defensive).
         - Iterate through all locations `loc_p` adjacent to `loc_b`. `loc_p` is a potential position for the robot to push from.
         - Determine the direction (`dir_rev`) from `loc_b` to `loc_p`.
         - Determine the opposite direction (`dir`) using `self.reverse_dir`.
         - Find the location (`loc_next`) where the box would move if pushed from `loc_p` in direction `dir`. `loc_next` is adjacent to `loc_b` in direction `dir`.
         - Check if this push is "useful": `self.distances.get(loc_next, {}).get(goal_loc, math.inf) < current_box_goal_dist`.
         - If the push is useful:
           - Check if `loc_next` is blocked by another box (required for the push action). If yes, this push is not currently possible, skip.
           - Check if `loc_p` is blocked by another box (robot cannot move there). If yes, robot cannot reach this push position, skip.
           - Calculate the robot's shortest path distance from `loc_robot` to `loc_p` considering `robot_obstacles` using `self._robot_bfs_distance`.
           - If the robot can reach `loc_p` (`robot_dist < math.inf`):
             - The cost to enable this useful push: `robot_dist` (moves) + 1 (the push action).
             - Update `min_robot_dist_to_useful_push_pos = min(min_robot_dist_to_useful_push_pos, robot_dist + 1)`.
       - `h2 = min_robot_dist_to_useful_push_pos`.
    5. Combine components:
       - If `h2` is still infinity, it means the robot cannot reach any position to make a useful push
         for any box that needs pushing, considering other boxes as obstacles.
         This is a strong indicator of a dead end for this heuristic's strategy.
         Return infinity.
       - Otherwise, return `h1 + h2`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        self.reverse_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # 1. Build the grid graph from adjacent facts
        self.graph = collections.defaultdict(list)
        self.locations = set()
        # Only add facts explicitly listed in static
        for fact_string in task.static:
            parts = self._parse_fact(fact_string)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.graph[l1].append((l2, direction))
                self.locations.add(l1)
                self.locations.add(l2)

        # Ensure all locations in graph keys/values are in self.locations
        for loc in list(self.graph.keys()):
             self.locations.add(loc)
             for adj_loc, _ in self.graph[loc]:
                  self.locations.add(adj_loc)

        # Ensure all locations mentioned in goals are in the graph/locations set
        for goal_fact in self.goals:
             parts = self._parse_fact(goal_fact)
             if parts[0] == 'at':
                  self.locations.add(parts[2])

        # 2. Determine box-goal assignments
        self.box_goals = {}
        for goal_fact in self.goals:
            parts = self._parse_fact(goal_fact)
            if parts[0] == 'at':
                box_name, goal_loc = parts[1], parts[2]
                self.box_goals[box_name] = goal_loc # Assuming unique goal per box

        # 3. Precompute all-pairs shortest path distances on the empty grid
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs_distance(start_loc, self.graph)

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
        # Remove outer brackets and split by spaces
        parts = fact_string[1:-1].split()
        # The first part is the predicate name
        predicate = parts[0]
        # The rest are the arguments
        args = parts[1:]
        return (predicate, *args)

    def _bfs_distance(self, start, graph):
        """Computes shortest path distances from start to all reachable locations in the graph."""
        distances = {loc: math.inf for loc in self.locations} # Use self.locations for all possible nodes
        if start not in distances: # Handle start node not in graph/locations
             return distances # Return empty/inf distances

        distances[start] = 0
        queue = collections.deque([start])
        visited = {start}

        while queue:
            curr_loc = queue.popleft()

            # Iterate through neighbors (adjacent_loc, direction)
            # Graph is {loc: [(adj_loc, dir), ...]}
            for next_loc, direction in graph.get(curr_loc, []):
                if next_loc not in visited:
                    visited.add(next_loc)
                    distances[next_loc] = distances[curr_loc] + 1
                    queue.append(next_loc)
        return distances

    def _robot_bfs_distance(self, start, target, obstacles, graph):
        """Computes shortest path distance for the robot from start to target, avoiding obstacles."""
        if start == target:
            return 0
        queue = collections.deque([(start, 0)])
        visited = {start}

        while queue:
            curr_loc, dist = queue.popleft()

            # Iterate through neighbors (adjacent_loc, direction)
            for next_loc, direction in graph.get(curr_loc, []):
                if next_loc == target:
                    return dist + 1 # Found target
                # Robot cannot move into an obstacle or a visited location
                if next_loc not in visited and next_loc not in obstacles:
                    visited.add(next_loc)
                    queue.append((next_loc, dist + 1))

        return math.inf # Target not reachable


    def __call__(self, node):
        state = node.state

        # 1. Check if goal reached
        if self.goals <= state:
            return 0

        # 2. Parse current state
        loc_robot = None
        loc_boxes = {}
        occupied_locs = set()

        for fact_string in state:
            parts = self._parse_fact(fact_string)
            predicate = parts[0]
            if predicate == 'at-robot':
                loc_robot = parts[1]
                occupied_locs.add(loc_robot)
            elif predicate == 'at':
                box_name, loc_b = parts[1], parts[2]
                loc_boxes[box_name] = loc_b
                occupied_locs.add(loc_b)

        # Handle case where robot or a box is not found (shouldn't happen in valid state)
        if loc_robot is None or not loc_boxes:
             # This state is likely invalid or represents a failure state
             return math.inf # Or some large value

        # 3. Calculate h1 (sum of box-goal distances)
        h1 = 0
        all_boxes_at_goal = True
        for box_name, loc_b in loc_boxes.items():
            goal_loc = self.box_goals.get(box_name)
            if goal_loc is None:
                 # Box has no assigned goal? Invalid problem definition for this heuristic.
                 # Treat as unsolvable.
                 return math.inf

            if loc_b != goal_loc:
                all_boxes_at_goal = False
                # Use .get with default math.inf for robustness
                dist = self.distances.get(loc_b, {}).get(goal_loc, math.inf)
                if dist == math.inf:
                     # Box is in a part of the graph disconnected from its goal
                     return math.inf
                h1 += dist

        # If all boxes are at goal, h1 is 0. The initial check should have caught this,
        # but this provides a fallback.
        if all_boxes_at_goal:
             return 0 # Should be caught by self.goals <= state

        # 4. Calculate h2 (robot-box distance to a useful push position)
        min_robot_dist_to_useful_push_pos = math.inf
        robot_obstacles = occupied_locs - {loc_robot} # Other boxes are obstacles for robot movement

        for box_name, loc_b in loc_boxes.items():
            goal_loc = self.box_goals.get(box_name)
            if loc_b == goal_loc:
                continue # Box is already at goal, no need to push it

            current_box_goal_dist = self.distances.get(loc_b, {}).get(goal_loc, math.inf)

            if current_box_goal_dist == math.inf:
                 # Box is disconnected from goal (already handled in h1, but defensive)
                 return math.inf

            # Find potential push actions that move the box towards the goal
            # Iterate through potential robot positions adjacent to the box (loc_p)
            for loc_p, dir_rev in self.graph.get(loc_b, []):
                dir = self.reverse_dir.get(dir_rev)
                if dir is None: continue # Should not happen with standard directions

                # Find loc_next: the square the box would move into
                loc_next = None
                for next_l, d in self.graph.get(loc_b, []):
                    if d == dir:
                        loc_next = next_l
                        break

                if loc_next is None: continue # No square to push into in this direction

                # Check if this push is "useful": moves box strictly closer to goal
                # Check if loc_next is a valid location in the distance map
                dist_loc_next_to_goal = self.distances.get(loc_next, {}).get(goal_loc, math.inf)

                if dist_loc_next_to_goal < current_box_goal_dist:
                    # This is a useful push direction. Robot needs to reach loc_p.

                    # Check if loc_next is blocked by another box (required for the push action)
                    is_loc_next_occupied_by_other_box = False
                    for other_box_name, other_box_loc in loc_boxes.items():
                         if other_box_name != box_name and other_box_loc == loc_next:
                              is_loc_next_occupied_by_other_box = True
                              break

                    if is_loc_next_occupied_by_other_box:
                         continue # This push direction is blocked by another box

                    # Check if loc_p is blocked by another box (robot cannot move there)
                    if loc_p in robot_obstacles:
                         continue # Robot cannot reach this push position because another box is there

                    # Calculate robot distance from loc_robot to loc_p
                    robot_dist = self._robot_bfs_distance(loc_robot, loc_p, robot_obstacles, self.graph)

                    if robot_dist < math.inf:
                        # Cost to enable this useful push: robot_dist (moves) + 1 (the push action)
                        min_robot_dist_to_useful_push_pos = min(min_robot_dist_to_useful_push_pos, robot_dist + 1)

        h2 = min_robot_dist_to_useful_push_pos

        # 5. Combine components
        # If h2 is still infinity, it means the robot cannot reach any position to make a useful push
        # for any box that needs pushing, considering other boxes as obstacles.
        # This is a strong indicator of a dead end for this heuristic's strategy.
        if h2 == math.inf:
             return math.inf

        return h1 + h2
