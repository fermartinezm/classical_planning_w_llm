from heuristics.heuristic_base import Heuristic
from task import Task
import collections
import math

class sokobanHeuristic(Heuristic):
    """
    Summary:
    Domain-dependent heuristic for the Sokoban domain.
    Estimates the cost to reach the goal state by summing two components:
    1. The minimum number of pushes required for each box to reach its goal
       location, ignoring obstacles (ideal box distance).
    2. The minimum number of robot moves required to reach a clear location
       adjacent to any box that is not yet at its goal (robot accessibility).

    Assumptions:
    - The input task is a valid Sokoban problem instance.
    - Locations are represented as strings (e.g., 'loc_R_C').
    - Adjacent relationships define a connected grid graph (potentially with holes).
    - The goal specifies a unique target location for each box using '(at box_name goal_loc)' facts.
    - Adjacency is symmetric for robot movement (if A is adjacent to B, B is adjacent to A).

    Heuristic Initialization:
    - Parses static 'adjacent' facts to build an undirected graph representing
      the grid connectivity. Stores this in `self.graph`.
    - Collects all unique location names from adjacent facts into `self.locations`.
    - Parses goal facts to determine the target location for each box. Stores
      this mapping in `self.box_goals`.
    - Precomputes all-pairs shortest paths on the grid graph using BFS. This
      provides the 'ideal box distance' component, representing the minimum
      pushes needed without considering dynamic obstacles or robot position.
      Stores these distances in `self.dist_map[start_loc][end_loc]`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state using `self.task.goal_reached(state)`.
       If yes, return 0.
    2. Parse the current state (`node.state`) to find:
       - The robot's current location (`robot_loc`).
       - The current location of each box (`box_locations` dictionary).
       - The set of clear locations (`clear_locations`).
       - The set of occupied locations (by boxes) (`occupied_locations`).
    3. Calculate the 'ideal box distance' component (`h_boxes`):
       - Initialize `h_boxes = 0`.
       - Create a list `boxes_not_at_goal` containing names of boxes whose current
         location is different from their goal location.
       - For each box name in `boxes_not_at_goal`:
         - Get its current location (`current_loc`) and its goal location (`goal_loc`)
           from `box_locations` and `self.box_goals`.
         - Look up the precomputed shortest path distance between `current_loc`
           and `goal_loc` in `self.dist_map`.
         - Add this distance to `h_boxes`. If the goal is unreachable from the
           current location in the grid graph, the distance is infinity, and
           `h_boxes` becomes infinity.
       - If `h_boxes` is infinity, the state is likely unsolvable regarding box
         positions; return infinity.
    4. Calculate the 'robot accessibility' component (`h_robot`):
       - Initialize `h_robot = 0`.
       - If `boxes_not_at_goal` is not empty (i.e., there are boxes needing to be moved):
         - Identify target locations for the robot: any location `L_adj` such that
           `L_adj` is adjacent to a location of a box in `boxes_not_at_goal` AND
           `L_adj` is currently in `clear_locations`. Store these in `robot_target_locs`.
         - If `robot_target_locs` is empty, the robot cannot reach a push position
           for any non-goal box; return infinity.
         - Define obstacles for the robot's movement BFS: any location in `occupied_locations`.
           The robot cannot move TO a square with a box.
         - Perform a BFS starting from `robot_loc` on the grid graph (`self.graph`).
           The BFS should avoid `robot_obstacles`.
           The BFS search terminates when the first location in `robot_target_locs`
           is reached. The distance to this location is `h_robot`.
         - If the BFS completes without reaching any target location, `h_robot` is
           infinity. If `h_robot` is infinity, return infinity.
    5. The total heuristic value is `h_boxes + h_robot`.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Build the location graph from adjacent facts
        self.graph = collections.defaultdict(set)
        self.locations = set()
        for fact in self.static:
            if fact.startswith('(adjacent '):
                parts = fact.split()
                loc1 = parts[1]
                loc2 = parts[2].rstrip(')') # Ensure parenthesis is stripped
                # dir = parts[3].rstrip(')') # Direction is not needed for undirected graph
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Assuming adjacency is symmetric for movement
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Parse box-goal mapping
        self.box_goals = {}
        for goal_fact in self.goals:
            if goal_fact.startswith('(at '):
                parts = goal_fact.split()
                box_name = parts[1]
                goal_loc = parts[2].rstrip(')')
                self.box_goals[box_name] = goal_loc

        # Precompute all-pairs shortest paths on the location graph (for ideal box distance)
        self.dist_map = {}
        for start_node in self.locations:
            self.dist_map[start_node] = self._bfs(start_node, self.locations, self.graph) # BFS returns distances from start_node

    def _bfs(self, start_node, all_nodes, graph, obstacles=None, targets=None):
        """
        Performs BFS on the graph.
        Returns a dictionary of distances from start_node if targets is None.
        Returns the distance to the first found target if targets is not None.
        Obstacles are locations the BFS cannot enter.
        """
        if obstacles is None:
            obstacles = set()

        # If start_node is not in the graph, it's unreachable from anywhere
        if start_node not in graph:
             if targets is not None:
                  return math.inf
             else:
                  distances = {node: math.inf for node in all_nodes}
                  return distances

        q = collections.deque([(start_node, 0)])
        visited = {start_node}
        distances = {node: math.inf for node in all_nodes} # Initialize all distances to infinity
        distances[start_node] = 0 # Distance from start_node to itself is 0

        # If the start node is itself a target, distance is 0
        if targets is not None and start_node in targets:
             return 0

        while q:
            curr_node, dist = q.popleft()

            for next_node in graph.get(curr_node, []):
                # Cannot move to a visited node or an obstacle node
                if next_node not in visited and next_node not in obstacles:
                    visited.add(next_node)
                    distances[next_node] = dist + 1
                    q.append((next_node, dist + 1))

                    # If we found a target, return the distance immediately
                    if targets is not None and next_node in targets:
                         return dist + 1

        # If BFS finishes
        if targets is not None:
            return math.inf # No target found
        else:
            return distances # Return all distances from start_node

    def __call__(self, node):
        state = node.state

        # Check for goal state
        if self.task.goal_reached(state):
             return 0

        # Parse current state
        robot_loc = None
        box_locations = {} # {box_name: location}
        clear_locations = set()
        occupied_locations = set() # Locations occupied by boxes

        for fact in state:
            if fact.startswith('(at-robot '):
                robot_loc = fact.split()[1].rstrip(')')
            elif fact.startswith('(at '):
                parts = fact.split()
                box_name = parts[1]
                loc = parts[2].rstrip(')')
                box_locations[box_name] = loc
                occupied_locations.add(loc)
            elif fact.startswith('(clear '):
                loc = fact.split()[1].rstrip(')')
                clear_locations.add(loc)

        # --- Calculate h_boxes (ideal box distance) ---
        h_boxes = 0
        boxes_not_at_goal = []
        for box_name, current_loc in box_locations.items():
            goal_loc = self.box_goals.get(box_name) # Get goal for this specific box
            # Ensure goal_loc exists and box is not already there
            if goal_loc and current_loc != goal_loc:
                boxes_not_at_goal.append(box_name)
                # Add ideal distance from current box location to goal location
                # Use precomputed distances
                if current_loc in self.dist_map and goal_loc in self.dist_map[current_loc]:
                     h_boxes += self.dist_map[current_loc][goal_loc]
                else:
                     # Should not happen in valid instances if graph is connected, but handle defensively
                     h_boxes += math.inf # Goal unreachable for this box

        # If h_boxes is already infinity, the state is likely unsolvable regarding box positions
        if h_boxes == math.inf:
             return math.inf

        # --- Calculate h_robot (robot accessibility) ---
        # Only calculate if there are boxes not at their goals
        h_robot = 0
        if boxes_not_at_goal:
            # Find target locations for the robot: clear locations adjacent to any non-goal box
            robot_target_locs = set()
            for box_name in boxes_not_at_goal:
                 box_loc = box_locations[box_name]
                 # Get adjacent locations from the graph
                 for adj_loc in self.graph.get(box_loc, []):
                      # Check if the adjacent location is clear
                      if adj_loc in clear_locations:
                           robot_target_locs.add(adj_loc)

            # If there are no clear adjacent locations to any non-goal box, the robot is stuck
            if not robot_target_locs:
                 return math.inf # Robot cannot reach a push position

            # Perform BFS for the robot from its current location to any target location
            # Obstacles for the robot are locations occupied by boxes.
            # The robot cannot move TO a square with a box.
            robot_obstacles = occupied_locations # Robot cannot move TO a square with a box

            h_robot = self._bfs(robot_loc, self.locations, self.graph, obstacles=robot_obstacles, targets=robot_target_locs)

            # If robot cannot reach any target, h_robot is infinity
            if h_robot == math.inf:
                 return math.inf

        # Total heuristic
        return h_boxes + h_robot
