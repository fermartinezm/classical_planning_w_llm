# Import necessary modules
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    # Removes leading/trailing parentheses and splits by whitespace
    return fact[1:-1].split()

# BFS for shortest path on the grid graph (ignoring state)
def bfs_grid_distance(start, end, graph):
    """
    Calculates the shortest path distance between two locations on the grid graph
    defined by adjacent facts, ignoring current state (obstacles).
    """
    if start == end:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # No path found

# BFS for shortest path for the robot (considering traversable locations)
def bfs_robot_distance(start, targets, graph, clear_locations):
    """
    Calculates the shortest path distance for the robot from a start location
    to any target location, only traversing through locations in clear_locations.
    """
    targets = set(targets)
    if start in targets:
        return 0 # Robot is already at a target location

    # Robot can move from start to any neighbor if neighbor is in clear_locations
    # The start location itself doesn't need to be in clear_locations to start the BFS
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                # Robot can move to neighbor if neighbor is clear and not visited
                if neighbor in clear_locations and neighbor not in visited:
                    if neighbor in targets:
                        return dist + 1 # Found shortest path to a target
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # No path found


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing:
    1. The grid distance for each box from its current location to its goal location.
    2. The shortest path distance for the robot to reach any location adjacent to any off-goal box, considering obstacles (other boxes and walls/non-clear locations).

    # Assumptions
    - The grid structure is defined by `adjacent` predicates.
    - Box movement cost is approximated by grid distance, ignoring state obstacles.
    - Robot movement cost considers locations occupied by other boxes or not explicitly clear as obstacles.
    - The heuristic does not detect deadlocks caused by box placement against walls/corners.

    # Heuristic Initialization
    - Extracts goal locations for each box from `task.goals`.
    - Builds the grid graph (adjacency list) from `adjacent` facts in `task.static`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to find the robot's location and the location of each box.
    2. Identify the set of locations that are currently 'clear' (traversable by the robot).
    3. Initialize `total_box_distance = 0` and an empty list `off_goal_box_locations`.
    4. For each box specified in the goal (`self.goal_locations`):
       - Get its current location from the state and its goal location from `self.goal_locations`.
       - If the box is not at its goal location:
         - Calculate the shortest path distance between the box's current location and its goal location on the grid graph (ignoring state obstacles). Add this distance to `total_box_distance`.
         - Add the box's current location to `off_goal_box_locations`.
    5. If `off_goal_box_locations` is empty, the heuristic is 0 (goal state).
    6. If `off_goal_box_locations` is not empty:
       - Identify all locations that are adjacent to *any* location in `off_goal_box_locations` using the grid graph. These are the potential target locations for the robot to initiate a push.
       - Calculate the shortest path distance for the robot from its current location to any of these target locations, only traversing through the 'clear_locations' identified in step 2. Let this be `robot_approach_distance`.
       - If no path exists for the robot to reach any target adjacent location, the state is likely unsolvable or requires complex unblocking; return infinity.
    7. The total heuristic value is `total_box_distance + robot_approach_distance`.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph from adjacent facts
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Assuming symmetric adjacency

        # Convert sets to lists for consistent iteration
        for loc in self.graph:
            self.graph[loc] = list(self.graph[loc])

        # Store goal locations for each box specified in the goal
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is typically (at boxX locY)
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                 box, location = parts[1], parts[2]
                 self.goal_locations[box] = location

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Parse current state
        robot_location = None
        box_locations = {}
        current_clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box_locations[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                current_clear_locations.add(parts[1])

        # Ensure robot location was found (should always be the case in a valid state)
        if robot_location is None:
             # This indicates an invalid state representation, return infinity
             return float('inf')

        # 3. Initialize distances and off-goal box list
        total_box_distance = 0
        off_goal_box_locations = []

        # 4. Calculate box distances for off-goal boxes
        for box, goal_loc in self.goal_locations.items():
            if box in box_locations: # Ensure the box exists in the current state
                current_loc = box_locations[box]
                if current_loc != goal_loc:
                    # Calculate grid distance for the box
                    dist = bfs_grid_distance(current_loc, goal_loc, self.graph)
                    if dist == float('inf'):
                         # If a box cannot reach its goal on the grid, it's likely unsolvable
                         return float('inf')
                    total_box_distance += dist
                    off_goal_box_locations.append(current_loc)
            # else: # Box not found in state? Should not happen in valid states.

        # 5. Check if goal is reached
        if not off_goal_box_locations:
            return 0 # All boxes in goal are at their goal locations

        # 6. Calculate robot approach distance
        robot_target_locations = set()
        for box_loc in off_goal_box_locations:
            if box_loc in self.graph:
                # Robot needs to reach a location adjacent to the box
                robot_target_locations.update(self.graph[box_loc])

        # If there are no adjacent locations to any off-goal box (shouldn't happen in valid grids),
        # return inf.
        if not robot_target_locations:
             return float('inf')

        # Calculate robot path distance to any target adjacent location
        robot_approach_distance = bfs_robot_distance(
            robot_location,
            robot_target_locations,
            self.graph,
            current_clear_locations # Robot can only move into clear locations
        )

        if robot_approach_distance == float('inf'):
             # Robot cannot reach any location adjacent to any off-goal box
             return float('inf')

        # 7. Total heuristic value
        return total_box_distance + robot_approach_distance
