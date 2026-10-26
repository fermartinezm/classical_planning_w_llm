from collections import deque
import math

# Assume Heuristic base class is available in heuristics/heuristic_base.py
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally for standalone testing
# In a real environment, this dummy class should be removed.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function for BFS
def bfs(start, goals, adj_list, obstacles, locations):
    """
    Performs BFS from start to find shortest distance to any location in goals,
    avoiding obstacles.
    Returns distance or float('inf') if unreachable.
    """
    q = deque([(start, 0)])
    visited = {start}

    if start in goals:
        return 0

    while q:
        curr_loc, dist = q.popleft()

        # Check neighbors
        if curr_loc in adj_list:
            for neighbor in adj_list[curr_loc]:
                # Check if neighbor is a valid location and not an obstacle
                if neighbor in locations and neighbor not in visited and neighbor not in obstacles:
                    if neighbor in goals:
                        return dist + 1
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))

    return float('inf') # Not reachable

# Helper function for BFS to precompute all-pairs shortest paths
def bfs_all_paths(start, adj_list, locations):
    """
    Performs BFS from start to find shortest distance to all other locations.
    Returns a dictionary {location: distance}.
    """
    distances = {loc: float('inf') for loc in locations}
    distances[start] = 0
    q = deque([start])

    while q:
        curr_loc = q.popleft()
        dist = distances[curr_loc]

        if curr_loc in adj_list:
            for neighbor in adj_list[curr_loc]:
                 # In the empty grid BFS, there are no dynamic obstacles.
                 # We only care about the static grid connectivity defined by adj_list.
                 # Ensure neighbor is a known location
                if neighbor in locations and distances[neighbor] == float('inf'):
                    distances[neighbor] = dist + 1
                    q.append(neighbor)

    return distances

# Helper function to find opposite direction
def opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen with valid directions

# Helper function to find location adjacent in a specific direction
def get_adjacent_in_direction(location, direction, adj_list):
    """
    Finds the location adjacent to 'location' in the given 'direction'.
    Returns the adjacent location string or None if no such location exists.
    """
    if location in adj_list:
        for neighbor, d in adj_list[location].items():
            if d == direction:
                return neighbor
    return None


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each box not yet at its goal:
    1. The minimum number of pushes required to move the box from its current location to its goal location on an empty grid (shortest path distance).
    2. The minimum number of robot moves required to reach a position adjacent to the box from which a push towards the goal is currently possible, considering other boxes and obstacles.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Locations not connected by `adjacent` facts or not `clear` are obstacles.
    - The heuristic sums costs for each box independently, ignoring complex interactions like one box blocking another's path or blocking the robot's path to another box.
    - Assumes a push is possible if the target location for the box is clear.

    # Heuristic Initialization
    - Parses `adjacent` facts to build the grid graph (`self.adj`).
    - Collects all location names (`self.locations`).
    - Parses goal conditions to map boxes to their goal locations (`self.box_goals`).
    - Precomputes all-pairs shortest path distances (`self.dist_grid`) on the empty grid graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location and the current location of each box.
    2. Identify the set of currently clear locations.
    3. Initialize the total heuristic cost to 0.
    4. For each box that is not at its goal location:
        a. Calculate the minimum number of pushes required for this box to reach its goal. This is the precomputed shortest path distance (`self.dist_grid`) between the box's current location and its goal location on the empty grid. Add this distance to the total heuristic.
        b. Determine the set of locations from which the robot *could* push this box towards its goal *in the current state*. A location `loc_r` is a potential push position if it's adjacent to the box's location `loc_b`, and the location `loc_next` (adjacent to `loc_b` in the opposite direction from `loc_r`) is currently clear.
        c. Calculate the shortest path distance for the robot from its current location to *any* of the potential push positions identified in step 4b. This BFS considers locations occupied by other boxes and locations that are not clear (walls, etc.) as obstacles. The robot's own location is not an obstacle for the start of the path. The box's location is an obstacle for the robot's pathfinding.
        d. Add the calculated robot distance to the total heuristic. If the robot cannot reach any push position (distance is infinity), the heuristic for this box becomes infinity.
    5. The total heuristic is the sum of the costs calculated for each box. If any box's cost is infinity, the total heuristic is infinity.
    6. If all boxes are at their goals, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, goal locations,
        and precomputing distances on the empty grid.
        """
        self.goals = task.goals
        static_facts = task.static

        self.adj = {} # location -> {neighbor_location: direction_from_location_to_neighbor}
        self.locations = set()

        # Parse adjacent facts to build the grid graph
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.adj:
                    self.adj[loc1] = {}
                self.adj[loc1][loc2] = direction

        # Parse goal locations for boxes
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming objects starting with 'box' are boxes and goal is always (at box loc)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                 box_name, location = parts[1], parts[2]
                 self.box_goals[box_name] = location

        # Precompute all-pairs shortest paths on the empty grid
        self.dist_grid = {}
        for start_loc in self.locations:
            self.dist_grid[start_loc] = bfs_all_paths(start_loc, self.adj, self.locations)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot and box locations
        loc_robot = None
        box_locations = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                loc_robot = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box_locations[parts[1]] = parts[2]
            elif parts[0] == 'clear' and len(parts) == 2:
                 clear_locations.add(parts[1])

        total_heuristic = 0
        all_boxes_at_goal = True

        # Determine obstacles for robot pathfinding
        # Obstacles are locations that are not clear, excluding the robot's current location
        # The box being pushed is an obstacle for the robot's pathfinding.
        # We build this set once per state.
        robot_obstacles_base = {loc for loc in self.locations if loc not in clear_locations and loc != loc_robot}


        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box)

            # If box is not in the state or is already at its goal, skip it
            if current_loc is None or current_loc == goal_loc:
                continue

            all_boxes_at_goal = False

            # Component 1: Minimum pushes for the box (empty grid distance)
            # Check if goal is reachable on the empty grid from the box's current location
            if current_loc not in self.dist_grid or goal_loc not in self.dist_grid.get(current_loc, {}):
                 # This indicates a problem with the grid or goal definition, or an unsolvable state
                 return float('inf')

            box_dist_to_goal = self.dist_grid[current_loc][goal_loc]

            # If box_dist_to_goal is infinity, the box cannot reach the goal even on an empty grid
            if box_dist_to_goal == float('inf'):
                 return float('inf')

            total_heuristic += box_dist_to_goal

            # Component 2: Robot cost to reach a push position

            # Find potential push locations for this box in the current state
            push_locations = set()
            if current_loc in self.adj:
                for loc_next in self.adj[current_loc]:
                    # Check if loc_next is a known location and is clear - this push direction is possible
                    if loc_next in self.locations and loc_next in clear_locations:
                        # Find the direction from current_loc to loc_next
                        dir_push = self.adj[current_loc][loc_next]
                        # Find the opposite direction
                        dir_robot = opposite_direction(dir_push)
                        # Find the location loc_push adjacent to current_loc in dir_robot
                        loc_push = get_adjacent_in_direction(current_loc, dir_robot, self.adj)
                        # Check if loc_push is a known location
                        if loc_push and loc_push in self.locations:
                            push_locations.add(loc_push)

            # If no push locations are currently possible, the robot cannot move the box
            if not push_locations:
                 # This box is currently stuck in all directions towards any adjacent clear spot.
                 # The robot cannot get into a push position that enables movement.
                 # This is likely a dead end state in terms of making progress on this box.
                 return float('inf')

            # Obstacles for robot BFS: base obstacles + the location of the box being pushed
            # The robot cannot pathfind *through* the box it needs to push.
            # The box's location 'current_loc' is not in clear_locations (unless it's the robot's spot, which is handled).
            # So, current_loc is in robot_obstacles_base unless current_loc == loc_robot.
            # If current_loc == loc_robot, the robot is on the box, which shouldn't happen in a valid state.
            # The robot_obstacles_base set correctly includes all non-clear locations except the robot's start.
            robot_obstacles = robot_obstacles_base

            # Calculate robot distance to any of the potential push locations
            robot_dist = bfs(loc_robot, push_locations, self.adj, robot_obstacles, self.locations)

            if robot_dist == float('inf'):
                 # Robot cannot reach any currently possible push position for this box
                 return float('inf')

            total_heuristic += robot_dist

        # If loop finished and all_boxes_at_goal is still True, it's a goal state
        if all_boxes_at_goal:
            return 0

        return total_heuristic
