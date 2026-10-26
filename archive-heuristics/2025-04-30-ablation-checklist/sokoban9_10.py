import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(location_str):
    """Parses a location string like 'loc_row_col' into a (row, col) tuple."""
    try:
        _, row_str, col_str = location_str.split('_')
        return (int(row_str), int(col_str))
    except ValueError:
        # Handle unexpected location formats if necessary, or raise error
        # For this domain, we assume loc_row_col format
        raise ValueError(f"Unexpected location format: {location_str}")

def build_graph(static_facts):
    """Builds an adjacency list graph from adjacent facts."""
    graph = collections.defaultdict(list)
    opposite_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

    for fact in static_facts:
        if match(fact, "adjacent", "*", "*", "*"):
            _, loc1, loc2, direction = get_parts(fact)
            graph[loc1].append((loc2, direction))
            # Add the reverse edge as well
            graph[loc2].append((loc1, opposite_direction[direction]))
    return graph

def bfs(start_loc, target_loc, graph, obstacles):
    """
    Performs a Breadth-First Search to find the shortest path distance.

    Args:
        start_loc: The starting location.
        target_loc: The target location.
        graph: The adjacency list graph.
        obstacles: A set of locations that cannot be traversed.

    Returns:
        The shortest path distance (number of moves) or float('inf') if unreachable.
    """
    if start_loc == target_loc:
        return 0

    queue = collections.deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == target_loc:
            return dist

        if current_loc not in graph:
             continue # Location might be isolated or invalid

        for neighbor_loc, _ in graph[current_loc]:
            if neighbor_loc not in visited and neighbor_loc not in obstacles:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return float('inf') # Target is unreachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing:
    1. The minimum number of pushes required for each box to reach its goal
       (shortest path distance for the box, ignoring other boxes as obstacles).
    2. The minimum number of moves for the robot to reach *any* box that needs moving
       (shortest path distance for the robot, considering all boxes as obstacles).
    3. An estimated cost for the robot to reposition itself after each push
       (approximated as 2 moves per push after the first push on each box).

    # Assumptions:
    - Locations are represented as 'loc_row_col'.
    - The grid structure is defined by 'adjacent' facts.
    - Boxes can only be moved by pushing.
    - Walls are implicitly defined by the lack of 'adjacent' facts.
    - Other boxes are obstacles for the robot.
    - Other boxes are *not* obstacles for calculating a box's minimum push distance to its goal (a simplification).
    - A fixed cost (2 moves) is sufficient to estimate robot repositioning after a push.

    # Heuristic Initialization
    - Builds the grid graph from 'adjacent' facts.
    - Extracts goal locations for each box from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes.
    2. Identify the goal location for each box.
    3. For each box not at its goal:
       a. Calculate the minimum number of pushes needed (`box_pushes`) using BFS from the box's current location to its goal, considering only walls as obstacles (implicitly handled by the graph). If unreachable, the state is likely a dead end, return infinity.
       b. Calculate the minimum number of moves for the robot to reach the box's current location (`robot_to_box_dist`) using BFS, considering walls and *all other boxes* as obstacles. If unreachable, return infinity.
       c. Keep track of the total pushes needed across all boxes (`total_pushes`).
       d. Keep track of the minimum `robot_to_box_dist` among all boxes that need moving (`min_robot_dist`).
       e. Count the number of boxes that need moving (`boxes_to_move_count`).
    4. If no boxes need moving, the heuristic is 0.
    5. If any box is unreachable from its goal or the robot cannot reach any box that needs moving, the heuristic is infinity.
    6. Otherwise, the heuristic is calculated as:
       `min_robot_dist` (cost to reach the first box)
       + `total_pushes` (cost of the push actions themselves)
       + `(total_pushes - boxes_to_move_count) * 2` (estimated repositioning cost: 2 moves per push after the first push on each box).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the graph."""
        self.goals = task.goals
        self.static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = build_graph(self.static_facts)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is (at boxX loc_Y_Z)
                box, location = args
                self.goal_locations[box] = location

        # Store all possible locations for quick lookup
        self.all_locations = set(self.graph.keys())
        for neighbors in self.graph.values():
            for neighbor_loc, _ in neighbors:
                self.all_locations.add(neighbor_loc)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot and box locations
        robot_location = None
        current_box_locations = {} # {box_name: location_str}

        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                box, location = get_parts(fact)
                current_box_locations[box] = location

        if robot_location is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf')

        total_pushes_needed = 0
        min_robot_dist_to_any_box = float('inf')
        boxes_to_move_count = 0

        # Obstacles for the robot: all locations occupied by boxes
        robot_obstacles = set(current_box_locations.values())

        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box) # Use .get() in case a box isn't in the state (shouldn't happen)

            if current_loc is None:
                 # Box not found in state, something is wrong
                 return float('inf')

            if current_loc != goal_loc:
                boxes_to_move_count += 1

                # Obstacles for this specific box: locations occupied by *other* boxes
                other_box_obstacles = {loc for b, loc in current_box_locations.items() if b != box}

                # Calculate minimum pushes for this box to reach its goal
                # BFS for box movement ignores other boxes as obstacles (simplification)
                # It only considers walls (implicit in graph) and potentially locations occupied by *other* boxes
                # Let's use the simpler version ignoring other boxes for box path for speed
                # pushes = bfs(current_loc, goal_loc, self.graph, other_box_obstacles)
                pushes = bfs(current_loc, goal_loc, self.graph, set()) # Simpler: box path ignores other boxes

                if pushes == float('inf'):
                    # This box cannot reach its goal location
                    return float('inf')

                total_pushes_needed += pushes

                # Calculate distance for robot to reach this box
                dist_robot_to_box = bfs(robot_location, current_loc, self.graph, robot_obstacles)

                if dist_robot_to_box == float('inf'):
                    # Robot cannot reach this box
                    return float('inf')

                min_robot_dist_to_any_box = min(min_robot_dist_to_any_box, dist_robot_to_box)

        # If all boxes are at their goals
        if boxes_to_move_count == 0:
            return 0

        # If robot cannot reach any box that needs moving (already checked inside loop, but defensive)
        if min_robot_dist_to_any_box == float('inf'):
             return float('inf')

        # Calculate heuristic value
        # Cost = (cost to reach first box) + (cost of pushes) + (cost of repositioning between pushes)
        # Repositioning cost: 2 moves per push, except for the very first push on each box
        # Total pushes needing repositioning = total_pushes_needed - boxes_to_move_count
        repositioning_cost = (total_pushes_needed - boxes_to_move_count) * 2

        heuristic_value = min_robot_dist_to_any_box + total_pushes_needed + repositioning_cost

        return heuristic_value

