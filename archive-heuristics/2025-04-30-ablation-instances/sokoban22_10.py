import collections
# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def build_location_graph(static_facts):
    """Builds an adjacency list graph from adjacent facts."""
    graph = collections.defaultdict(list)
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            l1, l2, direction = parts[1], parts[2], parts[3]
            graph[l1].append(l2)
    return graph

def bfs_distance(graph, start, goals, obstacles=None):
    """
    Calculates the shortest path distance from start to any goal location
    in the graph, avoiding obstacles.
    Goals can be a single location string or a set of location strings.
    Obstacles is a set of location strings to avoid.
    Returns distance or float('inf') if no path.
    """
    if obstacles is None:
        obstacles = set()
    if isinstance(goals, str):
        goals = {goals}

    # If the start is one of the goals, distance is 0.
    if start in goals:
        return 0

    # If the start is an obstacle and not a goal, it's unreachable.
    if start in obstacles:
         return float('inf')

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        # Explore neighbors
        for neighbor in graph.get(current_loc, []):
            # Only visit if not already visited and not an obstacle
            if neighbor not in visited and neighbor not in obstacles:
                if neighbor in goals:
                    return dist + 1 # Found a goal
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the total cost to reach the goal by summing
    estimated costs for each misplaced box. The estimated cost for a single
    misplaced box is the sum of:
    1. The shortest path distance for the box to its goal location, treating
       other boxes and the robot as temporary obstacles. This estimates the
       minimum number of push actions required for this box.
    2. The minimum shortest path distance for the robot to reach any location
       adjacent to the box's current location, treating other boxes as temporary
       obstacles. This estimates the robot movement cost to get into a position
       to potentially push the box.

    # Assumptions
    - The goal state specifies a unique target location for each box.
    - The grid structure and connectivity are defined by the `adjacent` facts
      in the static information.
    - Locations not connected in the graph are considered walls (static obstacles).
    - Other boxes and the robot are considered temporary obstacles for pathfinding.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds the location graph representing the grid connectivity based on
      `adjacent` facts from the static information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Initialize the total heuristic cost to 0.
    3. For each box that has a specified goal location (extracted during initialization):
        a. Check if the box is currently at its goal location. If yes, continue to the next box (cost for this box is 0).
        b. If the box is not at its goal:
            i. Determine the set of obstacles for the box's pathfinding: the locations of all *other* boxes and the robot's current location.
            ii. Calculate the shortest path distance (`d_box`) for the box from its current location to its goal location using Breadth-First Search (BFS) on the location graph, avoiding the box obstacles. This distance represents the minimum number of push actions required for this box if the path were clear of these obstacles. If no path exists (distance is infinity), the state is likely a deadlock or unsolvable; return infinity immediately as the total heuristic.
            iii. Determine the set of obstacles for the robot's pathfinding: the locations of all *other* boxes. The robot cannot move through other boxes.
            iv. Find all locations that are directly adjacent to the box's current location using the pre-built location graph.
            v. Filter these adjacent locations to find only those that are not occupied by other boxes (these are the reachable adjacent locations for the robot to target).
            vi. Calculate the minimum shortest path distance (`min_d_robot_to_adj`) for the robot from its current location to *any* of the reachable adjacent locations found in the previous step, using BFS and avoiding the robot obstacles. This estimates the robot movement cost to get near the box. If the robot cannot reach any reachable adjacent location (minimum distance is infinity), return infinity immediately.
            vii. Add the sum `d_box + min_d_robot_to_adj` to the total heuristic cost.
    4. After processing all boxes, return the accumulated total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting box-goal mappings and building
        the location graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at boxX locY)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # Build the location graph from adjacent facts.
        self.graph = build_location_graph(static_facts)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot and box locations in the current state.
        robot_loc = None
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        # If robot location is not found, something is wrong with the state representation.
        if robot_loc is None:
             # This shouldn't happen in a valid Sokoban state representation
             # but handle defensively.
             return float('inf') # Cannot proceed without robot location

        total_heuristic = 0

        # Iterate through the boxes we know need to reach a goal.
        for box, goal_loc in self.box_goals.items():
            current_box_loc = box_locations.get(box)

            # If the box is not in the state (shouldn't happen) or is already at its goal, no cost from this box.
            if current_box_loc is None or current_box_loc == goal_loc:
                continue

            # --- Calculate cost for the box to reach its goal ---
            # Obstacles for the box: other boxes and the robot.
            other_box_locations = {loc for b, loc in box_locations.items() if b != box}
            box_obstacles = other_box_locations | {robot_loc}

            # Calculate shortest path distance for the box.
            d_box = bfs_distance(self.graph, current_box_loc, goal_loc, obstacles=box_obstacles)

            # If the box cannot reach its goal, the state is likely unsolvable/deadlocked.
            if d_box == float('inf'):
                return float('inf')

            # --- Calculate cost for the robot to get near the box ---
            # Obstacles for the robot: other boxes. Robot cannot move through other boxes.
            robot_obstacles_for_path = other_box_locations

            # Find locations adjacent to the box's current location.
            adjacent_locations_of_box = self.graph.get(current_box_loc, [])

            # Filter adjacent locations to find those the robot can potentially move into.
            # Robot cannot move into a square occupied by another box.
            reachable_adjacent_locations = {
                adj_loc for adj_loc in adjacent_locations_of_box
                if adj_loc not in other_box_locations
            }

            # If there are reachable adjacent locations, find the minimum distance
            min_d_robot_to_adj = float('inf')
            if reachable_adjacent_locations:
                 # BFS from robot_loc to any of the reachable_adjacent_locations
                 min_d_robot_to_adj = bfs_distance(self.graph, robot_loc, reachable_adjacent_locations, obstacles=robot_obstacles_for_path)


            # If the robot cannot reach any reachable adjacent location to the box, it's stuck.
            if min_d_robot_to_adj == float('inf'):
                 return float('inf')

            # Add costs for this box to the total heuristic.
            total_heuristic += d_box + min_d_robot_to_adj

        return total_heuristic
