from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

# BFS function
def bfs(start, end, graph, obstacles):
    """
    Performs BFS to find the shortest path distance between start and end locations
    in the given graph, avoiding obstacle locations.

    Args:
        start (str): The starting location name.
        end (str): The target location name.
        graph (dict): Adjacency list representation of the location graph.
        obstacles (set): A set of location names that cannot be traversed.

    Returns:
        int or float('inf'): The shortest distance, or infinity if unreachable.
    """
    if start == end:
        return 0
    queue = collections.deque([(start, 0)])
    visited = {start}
    obstacle_set = set(obstacles)

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Explore neighbors
        for neighbor in graph.get(current_loc, []):
            # A location is an obstacle if it's in the obstacle_set AND it's not the target location itself.
            # The target location (end) can be occupied by the object we are pathfinding *to*.
            is_obstacle = neighbor in obstacle_set and neighbor != end

            if neighbor not in visited and not is_obstacle:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # Unreachable


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each box
    that is not yet at its goal location:
    1. The shortest path distance from the box's current location to its goal location,
       considering other objects (robot and other boxes) as obstacles.
    2. The shortest path distance from the robot's current location to the box's current location,
       considering boxes as obstacles.
    The total heuristic is the sum of these values over all misplaced boxes.

    # Assumptions
    - The grid structure and connectivity are defined by `adjacent` facts.
    - Locations occupied by other objects (boxes or robot) are obstacles for pathfinding,
      except for the target location itself when pathfinding to an object's location.
    - The cost of moving a box is related to its distance to the goal.
    - The cost of enabling a push is related to the robot's distance to the box.
    - Uniform action costs are assumed (each move or push costs 1).
    - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an adjacency list representation of the location graph based on `adjacent` facts
      from the static information. This graph is used for shortest path calculations (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify the goal location for each box (pre-calculated during initialization).
    3. Initialize the total heuristic value `h` to 0.
    4. Create a set of `occupied_locations` containing the current locations of the robot and all boxes.
    5. For each box that has a goal location defined:
       a. Get the box's current location. If the box is not found in the state, this might indicate an invalid state; return infinity.
       b. Check if the box is already at its goal location. If yes, contribute 0 to `h` for this box and continue to the next box.
       c. If the box is not at its goal:
          i. Define obstacles for the box's path to the goal: all locations in `occupied_locations` *except* the box's own current location. A box cannot move onto a square occupied by the robot or another box.
          ii. Calculate `dist_box_to_goal` using BFS from the box's current location to its goal location, avoiding `box_path_obstacles`.
          iii. If `dist_box_to_goal` is infinity, the box cannot reach its goal; return infinity for the heuristic value (deadlock).
          iv. Define obstacles for the robot's path to the box: all locations in `occupied_locations` *except* the robot's own current location. The robot cannot move onto a square occupied by a box.
          v. Calculate `dist_robot_to_box` using BFS from the robot's current location to the box's current location, avoiding `robot_path_obstacles`.
          vi. If `dist_robot_to_box` is infinity, the robot cannot reach the box; return infinity for the heuristic value (deadlock).
          vii. Add the sum of `dist_box_to_goal` and `dist_robot_to_box` to the total heuristic `h`.
    6. After processing all boxes, return the total heuristic value `h`. If the initial goal check passed, this value will be 0. Otherwise, if no deadlocks were detected, it will be a positive integer.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the location graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Goal facts are typically (at boxX locY)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.box_goals[box] = location

        # Build the location graph from adjacent facts.
        self.adj_list = collections.defaultdict(list)
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent" and len(args) == 3:
                loc1, loc2, direction = args
                self.adj_list[loc1].append(loc2)
                # Assuming adjacency is symmetric, add the reverse edge
                self.adj_list[loc2].append(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is already reached
        if self.goals <= state:
             return 0

        # Find current locations of robot and boxes
        robot_loc = None
        box_locations = {}
        occupied_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
                occupied_locations.add(robot_loc)
            elif parts[0] == "at" and len(parts) == 3 and parts[1].startswith("box"):
                box, loc = parts[1], parts[2]
                box_locations[box] = loc
                occupied_locations.add(loc)

        # Defensive check: Robot must be present
        if robot_loc is None:
             return float('inf') # Invalid state

        total_cost = 0

        # Calculate cost for each box that has a goal defined
        for box, goal_loc in self.box_goals.items():
            box_loc = box_locations.get(box)

            # Defensive check: Box must be present in the state
            if box_loc is None:
                 return float('inf') # Invalid state (box not at any location?)

            if box_loc == goal_loc:
                continue # This box is already at its goal

            # Obstacles for box path: all occupied locations *except* the box's current location
            # A box cannot move onto a square occupied by the robot or another box.
            box_path_obstacles = occupied_locations - {box_loc}

            # Calculate box distance to goal
            dist_box_to_goal = bfs(box_loc, goal_loc, self.adj_list, box_path_obstacles)

            if dist_box_to_goal == float('inf'):
                # Box cannot reach goal (e.g., trapped by other boxes or walls)
                return float('inf') # Deadlock detected

            # Obstacles for robot path: all occupied locations *except* the robot's current location
            # The robot cannot move onto a square occupied by a box.
            robot_path_obstacles = occupied_locations - {robot_loc}

            # Calculate robot distance to box
            dist_robot_to_box = bfs(robot_loc, box_loc, self.adj_list, robot_path_obstacles)

            if dist_robot_to_box == float('inf'):
                 # Robot cannot reach the box
                 return float('inf') # Deadlock detected

            # Add costs for this box
            # This sum is a non-admissible estimate.
            total_cost += dist_box_to_goal + dist_robot_to_box

        # If we reached here, either all boxes were at the goal (handled by initial check)
        # or total_cost is the sum of finite distances > 0.
        return total_cost
