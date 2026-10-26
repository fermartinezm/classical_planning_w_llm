# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

from collections import deque
import math # For infinity

def get_parts(fact):
    """Helper function to parse a PDDL fact string into a list of parts."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def bfs(start, end, graph, obstacles):
    """
    Performs Breadth-First Search to find the shortest path distance
    and predecessor map from start to end in the graph, avoiding obstacles.

    Args:
        start (str): The starting location.
        end (str): The target location.
        graph (dict): Adjacency list representation {location: [neighbor_location, ...]}
        obstacles (set): A set of locations that cannot be visited.

    Returns:
        tuple: (distance, predecessor_map) or (float('inf'), None) if no path.
    """
    queue = deque([(start, 0)])
    visited = {start}
    predecessor = {start: None}

    # If start is an obstacle, we can't start the BFS.
    # In this heuristic, start is either robot_location or box_location.
    # robot_location is never an obstacle. box_location is never an obstacle
    # for the box's own path. So this check isn't strictly needed based on
    # how BFS is called in this heuristic, but good practice.
    if start in obstacles:
         return float('inf'), None

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist, predecessor

        # Get neighbors from the graph (graph stores only locations, not directions)
        neighbors = graph.get(current_loc, [])

        for neighbor in neighbors:
            # The target location 'end' should not be treated as an obstacle
            # if the BFS is for the robot trying to reach the location of the box
            # it needs to push. The push action allows the robot to move into
            # the box's previous spot.
            # The `obstacles` set passed to robot BFS should exclude the location
            # of the box being pushed. So, we just check `neighbor not in obstacles`.
            if neighbor not in visited and neighbor not in obstacles:
                visited.add(neighbor)
                predecessor[neighbor] = current_loc
                queue.append((neighbor, dist + 1))

    return float('inf'), None # No path found

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated costs for each box to reach its target location.
        For each misplaced box, the estimated cost is the sum of:
        1. The shortest path distance for the box from its current location
           to its goal location (number of pushes). This is calculated on the
           location graph ignoring obstacles (a relaxation).
        2. The shortest path distance for the robot from its current location
           to the specific location required to push the box one step along
           a shortest path towards its goal. This is calculated on the location
           graph considering other boxes as obstacles.
        The heuristic is non-admissible but aims to guide a greedy search
        efficiently. It returns 0 if and only if the state is a goal state.

    Assumptions:
        - The locations form a graph defined by the 'adjacent' predicates.
        - The goal is specified by '(at box location)' facts.
        - The heuristic simplifies the problem by:
            - Calculating box path distances ignoring obstacles.
            - Calculating robot path distances considering only other boxes
              as obstacles (not walls implied by lack of adjacency, which
              are handled by the BFS graph).
            - Summing costs for each box independently, ignoring potential
              conflicts or complex interactions between boxes.
            - Assuming a shortest path for the box is always pushable by the robot.
        - PDDL facts are represented as strings parseable by `get_parts`.

    Heuristic Initialization:
        The constructor processes the static facts provided by the task.
        - It builds an adjacency map (`adjacent_map`) from 'adjacent' facts,
          mapping each location to a list of (neighbor_location, direction) tuples.
        - It builds a simple adjacency graph (`adj_graph`) suitable for BFS,
          mapping each location to a list of neighbor locations.
        - It creates a map (`opposite_dir_map`) for finding the opposite direction.
        - It extracts the goal location for each box from the task's goal facts
          and stores them in `box_goal_locations`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state and parse it to find the robot's location
           and the current location of each box.
        2. Check if the current state is the goal state. If all boxes are at
           their goal locations (as defined in `self.goals`), the heuristic is 0.
        3. Initialize the total heuristic value to 0.
        4. Iterate through each box that has a goal location
           stored during initialization (`self.box_goal_locations`).
        5. For a given box:
           a. Get its current location from the state (`box_current_locations`).
           b. If the box is not found in the current state or is already at goal, skip.
           c. If the box is not at its goal location:
              i. Calculate the shortest path distance for the box from its
                 current location to its goal location using BFS on the
                 adjacency graph (`self.adj_graph`), ignoring obstacles.
                 This distance represents the minimum number of pushes required.
                 If no path exists, the state is likely unsolvable, return infinity.
              ii. Determine the required location for the robot to perform the
                  first push. This location is adjacent to the box's current
                  location, in the direction opposite to the first step of
                  a shortest path for the box towards its goal. Reconstruct
                  a shortest path for the box using the predecessor map from BFS
                  to find the first step (`path[1]`). Find the direction from
                  `current_location` to `path[1]` using `self.adjacent_map`.
                  Find the location adjacent to `current_location` in the
                  opposite direction using `self.opposite_dir_map` and
                  `self.adjacent_map`.
              iii. Calculate the shortest path distance for the robot from its
                   current location (`robot_location`) to the required push location.
                   This is done using BFS on the adjacency graph (`self.adj_graph`),
                   considering locations occupied by *other* boxes as obstacles.
                   The location of the box being pushed is *not* an obstacle for
                   the robot's path to the required push location. If no path exists,
                   return infinity.
              iv. Add the box distance (pushes) and the robot distance (moves
                  to position) to the total heuristic value.
        6. Return the total calculated heuristic value.
    """
    def __init__(self, task):
        # Assuming Heuristic base class has a constructor that takes task
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.adjacent_map = {}
        self.adj_graph = {}
        self.location_objects = set()

        # Build adjacency map and graph from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.location_objects.add(loc1)
                self.location_objects.add(loc2)

                if loc1 not in self.adjacent_map:
                    self.adjacent_map[loc1] = []
                self.adjacent_map[loc1].append((loc2, direction))

                if loc1 not in self.adj_graph:
                    self.adj_graph[loc1] = []
                self.adj_graph[loc1].append(loc2)

        # Define opposite directions
        self.opposite_dir_map = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }

        # Extract box goal locations
        self.box_goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box location)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.box_goal_locations[box] = location
            # Handle potential 'and' in goals if needed, but example shows direct 'at' facts
            # If goals were like (and (at box1 l1) (at box2 l2)), task.goals would be the set of (at ...) facts.

    def __call__(self, node):
        state = node.state

        # 1. Get current state info
        robot_location = None
        box_current_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                box_current_locations[box] = location

        # 2. Check if goal state (all box goals met)
        # The search algorithm handles the terminal goal check.
        # The heuristic must be 0 iff it is a goal state.
        # A state is a goal state if all (at box goal_loc) facts are true.
        # We can check if the set of goal facts is a subset of the current state facts.
        if self.goals <= state:
             return 0

        # 3. Initialize total heuristic
        total_heuristic = 0

        # 4. Iterate through each box that has a goal location
        # We iterate through self.box_goal_locations because these are the boxes
        # we care about moving to specific spots.
        for box, goal_location in self.box_goal_locations.items():
            current_location = box_current_locations.get(box)

            # If the box is not found in the current state or is already at goal, skip
            if current_location is None or current_location == goal_location:
                continue

            # 5. Calculate cost for this misplaced box
            # i. Box distance (pushes needed)
            # BFS for box path ignores obstacles for simplicity
            box_dist, box_pred = bfs(current_location, goal_location, self.adj_graph, set())

            if box_dist == float('inf'):
                # Box cannot reach its goal location
                return float('inf') # State is likely unsolvable

            # ii. Determine required robot location for the first push
            # Reconstruct a shortest path for the box to find the first step
            path = [goal_location]
            # Traverse back from goal using predecessors until current_location is reached
            current_in_path_reconstruction = goal_location
            while current_in_path_reconstruction != current_location:
                 pred = box_pred.get(current_in_path_reconstruction)
                 if pred is None:
                      # Should not happen if box_dist was not inf, indicates BFS issue
                      return float('inf')
                 path.insert(0, pred)
                 current_in_path_reconstruction = pred

            # The first step is from current_location (path[0]) to path[1]
            next_box_loc = path[1]

            # Find the direction from current_location to next_box_loc
            push_direction = None
            # Check neighbors of current_location in the adjacent_map
            for neighbor, direction in self.adjacent_map.get(current_location, []):
                if neighbor == next_box_loc:
                    push_direction = direction
                    break

            if push_direction is None:
                 # Should not happen if next_box_loc is a valid neighbor from BFS path
                 return float('inf') # Indicates graph inconsistency or BFS issue

            # Find the required robot location adjacent to current_location
            # in the opposite direction of the push
            required_robot_location = None
            opposite_dir = self.opposite_dir_map.get(push_direction)
            if opposite_dir:
                 # Check neighbors of current_location in the adjacent_map
                 for neighbor, direction in self.adjacent_map.get(current_location, []):
                      if direction == opposite_dir:
                           required_robot_location = neighbor
                           break

            if required_robot_location is None:
                 # Cannot find a location to push from in the required direction
                 # This might indicate a wall or edge case in the grid structure
                 return float('inf') # State is likely unsolvable

            # iii. Robot distance (moves to position)
            # BFS for robot path considers other boxes as obstacles
            # The robot can move into the location of the box it is about to push
            # as part of the push action, so the current box's location is NOT an obstacle
            # for the robot's path *to the required_robot_location*.
            # Obstacles are locations occupied by OTHER boxes.
            robot_obstacles = {loc for b, loc in box_current_locations.items() if b != box}

            robot_dist, robot_pred = bfs(robot_location, required_robot_location, self.adj_graph, robot_obstacles)

            if robot_dist == float('inf'):
                # Robot cannot reach the required push location
                return float('inf') # State is likely unsolvable

            # iv. Add costs for this box
            # Each push moves the box one step. The box_dist is the number of pushes.
            # The robot_dist is the cost for the robot to get into position for the *first* push.
            # A push action costs 1. Getting into position costs robot_dist moves.
            # The heuristic is non-admissible, so we can sum these.
            # A simple sum: box_dist (pushes) + robot_dist (moves to get behind box)
            total_heuristic += box_dist + robot_dist

        # 6. Return total heuristic
        return total_heuristic

# Assuming Heuristic base class is defined elsewhere and imported as 'Heuristic'
# Example:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError
