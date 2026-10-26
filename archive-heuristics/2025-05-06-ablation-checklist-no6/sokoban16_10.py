import collections
# Assuming heuristics.heuristic_base exists and defines the base class
from heuristics.heuristic_base import Heuristic


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at box1 loc_4_4)" -> ["at", "box1", "loc_4_4"]
    return fact[1:-1].split()


# Helper function for BFS path
def bfs_path(graph, start, end, obstacles):
    """
    Performs BFS to find the shortest path (list of locations).
    Returns None if no path exists or start is an obstacle.
    """
    if start == end:
        return [start]
    if start in obstacles:
        return None # Cannot start inside an obstacle

    queue = collections.deque([(start, [start])]) # Store (location, path)
    visited = {start}

    while queue:
        current_loc, path = queue.popleft()

        if current_loc == end:
            return path

        if current_loc in graph:
            for neighbor, _ in graph[current_loc]:
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return None # End not reachable


# Helper function for BFS distance only
def bfs_distance(graph, start, end, obstacles):
    """
    Performs BFS to find the shortest path distance.
    Returns float('inf') if no path exists or start is an obstacle.
    """
    if start == end:
        return 0
    if start in obstacles:
        return float('inf') # Cannot start inside an obstacle

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        if current_loc in graph:
            for neighbor, _ in graph[current_loc]:
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # End not reachable


# Helper function to get neighbor in a specific direction
def get_neighbor_in_direction(graph, loc, direction):
    """Returns the neighbor location from loc in the given direction, or None."""
    if loc in graph:
        for neighbor, d in graph[loc]:
            if d == direction:
                return neighbor
    return None


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum, over all misplaced boxes, of:
    1. The shortest path distance for the box to its goal (ignoring other boxes).
    2. The shortest path distance for the robot to reach the location
       behind the box required for the first push towards the goal.

    This heuristic is non-admissible. It aims to guide a greedy best-first search
    by prioritizing states where boxes are closer to their goals and the robot
    is closer to a position from which it can push a box towards its goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the graph representing the grid
        and storing goal locations for each box.
        """
        # The task object contains initial_state, goals, operators, static facts.
        self.goals = task.goals
        static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = {}
        all_locations = set()

        # Collect all locations mentioned in initial state (robot, boxes, clear)
        # This ensures we have all relevant locations even if not all are in adjacent facts
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                all_locations.add(parts[1])
            elif parts[0] == 'at':
                all_locations.add(parts[2])
            elif parts[0] == 'clear':
                all_locations.add(parts[1])

        # Add locations from adjacent facts, ensuring all possible locations are included
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Initialize graph with all locations. Locations not in adjacent facts will have empty neighbor lists.
        for loc in all_locations:
            self.graph[loc] = []

        # Populate graph with adjacent relationships
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.graph[loc1].append((loc2, direction))

        # Store goal locations for each box
        self.goal_locations = {} # box -> location
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Mapping for opposite directions
        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        robot_loc = None
        box_locs = {} # box -> location

        # Parse current state to find robot and box locations
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at':
                box, loc = parts[1], parts[2]
                box_locs[box] = loc

        h = 0
        all_box_locs = set(box_locs.values())

        # Calculate heuristic for each misplaced box
        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locations.get(box)

            # If box has no goal or is already at goal, skip
            if goal_loc is None or current_loc == goal_loc:
                continue

            # --- Part 1: Box distance to goal ---
            # Obstacles for box movement: other boxes
            other_box_locs = all_box_locs - {current_loc}
            # Use bfs_path to find the path and the first step direction
            # This BFS considers other boxes as static obstacles.
            path_box = bfs_path(self.graph, current_loc, goal_loc, obstacles=other_box_locs)

            if path_box is None:
                # Box is blocked by other boxes or walls from reaching its goal
                # This state is likely unsolvable or requires moving other boxes first.
                # A high heuristic value indicates this difficulty.
                return float('inf') # Indicate unsolvable or very high cost

            dist_box_to_goal = len(path_box) - 1

            # --- Part 2: Robot distance to the required pushing position ---
            # The required pushing position is adjacent to the box's current location,
            # in the direction opposite to the first step of the box's path towards the goal.

            # Find the direction of the first step of the box's path
            if dist_box_to_goal == 0:
                 # This case should be caught by the initial check, but handle defensively
                 continue

            first_step_loc = path_box[1]
            first_step_dir = None
            # Find the direction from current_loc to first_step_loc
            if current_loc in self.graph:
                for neighbor, d in self.graph[current_loc]:
                    if neighbor == first_step_loc:
                        first_step_dir = d
                        break
            # first_step_dir should be found if path_box exists and has length > 1
            if first_step_dir is None:
                 # Should not happen in a valid grid with a path, but return inf defensively
                 # This might happen if the graph is disconnected or malformed.
                 return float('inf')

            # Find the location adjacent to the box in the opposite direction
            required_robot_loc = get_neighbor_in_direction(self.graph, current_loc, self.opposite_dir[first_step_dir])

            if required_robot_loc is None:
                 # The location behind the box is a wall/boundary.
                 # This box might be in a deadlock depending on the goal.
                 # Returning inf is a safe bet for a non-admissible heuristic.
                 # This can happen if the box is against a wall and needs to move into it.
                 return float('inf')

            # Obstacles for robot movement: all boxes
            robot_obstacles = all_box_locs

            # Calculate distance from robot's current location to the required pushing location
            # The required_robot_loc must NOT be in robot_obstacles for the robot to reach it.
            # The bfs_distance function handles this check.
            dist_robot_to_pred = bfs_distance(self.graph, robot_loc, required_robot_loc, obstacles=robot_obstacles)

            if dist_robot_to_pred == float('inf'):
                # Robot cannot reach the required pushing position
                return float('inf') # Indicate unsolvable or very high cost

            # Add costs: distance for the box (pushes) + distance for the robot (moves)
            # Each box step is 1 push action. Robot moves are 1 move action.
            # This is a simplified cost model that doesn't account for robot movement
            # needed *between* pushes if the path isn't straight or the robot needs to circle.
            h += dist_box_to_goal + dist_robot_to_pred

        # The heuristic is 0 if and only if all boxes are at their goal locations.
        # The goal only specifies box locations, not robot location.
        return h
