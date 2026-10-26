# Required imports
from collections import deque

# Cache for static data: {task_key: (robot_graph, box_goals)}
# Using task.name as the key. Assuming task names are unique per instance.
# A more robust key could be hash(frozenset(task.static) | frozenset(task.goals))
# but task.name is simpler and likely sufficient for typical planner usage.
sokobanHeuristic._cache = {}

def bfs_distance(graph, start, end):
    """
    Calculates the shortest path distance between two locations in the grid graph
    using Breadth-First Search (BFS).

    Args:
        graph (dict): Adjacency list representation of the grid graph.
                      {location: set(neighbor_locations)}
        start (str): The starting location.
        end (str): The target location.

    Returns:
        int or float('inf'): The shortest distance, or infinity if not reachable.
    """
    if start == end:
        return 0
    # Ensure start and end are valid locations in the graph
    # If a location from state/goal isn't in the graph built from static facts,
    # it implies a problem definition issue or unreachable location.
    if start not in graph or end not in graph:
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        # Check if current_loc has neighbors in the graph (should be true if in graph keys)
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # Not reachable

def min_robot_dist_to_push_box(robot_loc, box_loc, graph):
    """
    Calculates the minimum shortest path distance for the robot from its current
    location to any location adjacent to the box's location. This estimates
    the cost for the robot to get into a position from which it *could* push the box.

    Args:
        robot_loc (str): The robot's current location.
        box_loc (str): The box's current location.
        graph (dict): Adjacency list representation of the grid graph.

    Returns:
        int or float('inf'): The minimum distance, or infinity if no adjacent
                              locations are reachable by the robot.
    """
    # If box_loc is not in the graph, it's isolated or invalid
    if box_loc not in graph:
        return float('inf')

    min_dist = float('inf')
    # Potential push locations are simply the neighbors of the box location
    potential_push_locations = graph.get(box_loc, set())

    if not potential_push_locations:
        # Box is in a location with no adjacent locations in the graph
        return float('inf')

    # Calculate distance from robot to each potential push location
    for push_loc in potential_push_locations:
        dist = bfs_distance(graph, robot_loc, push_loc)
        min_dist = min(min_dist, dist)

    return min_dist

def sokobanHeuristic(state, task):
    """
    Domain-dependent heuristic for the Sokoban planning domain.

    Summary:
    Estimates the cost to reach the goal state by summing two components for each box
    that is not yet at its goal location:
    1. The shortest path distance for the box to reach its goal location (number of pushes).
    2. The minimum shortest path distance for the robot to reach any location adjacent
       to the box's current location (cost to get into a position to push the box).
    Distances are calculated using BFS on the grid graph defined by adjacent predicates.

    Assumptions:
    - The problem instance defines a grid-like structure using `adjacent` predicates.
    - Locations are connected as described by `adjacent` facts.
    - Each box specified in the goal has a unique target location specified in `task.goals`.
    - The state representation includes `(at-robot ...)` and `(at box ...)` facts for relevant objects.
    - The `Task` object provides `static` facts, `goals`, and `goal_reached` method.
    - The heuristic value is 0 if and only if the state is a goal state.
    - The heuristic value is finite for solvable states and infinity for potentially unsolvable states (e.g., disconnected components).

    Heuristic Initialization:
    The heuristic pre-calculates static information (the grid graph from `adjacent`
    facts and the goal location for each box from `task.goals`) and caches it
    based on the task object's name. This avoids redundant parsing and graph building
    across multiple heuristic calls for the same task instance.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the static data (grid graph, box goals) for the current task is
       already cached using `task.name` as the key.
    2. If the data is not cached:
       a. Initialize an empty dictionary `robot_graph` to represent the grid's
          adjacency list.
       b. Initialize an empty dictionary `box_goals` to store box-goal mappings.
       c. Iterate through `task.static` facts. If a fact starts with `'(adjacent '`,
          parse the two location names and add undirected edges between them in
          `robot_graph`.
       d. Iterate through `task.goals` facts. If a fact starts with `'(at '`,
          parse the box name and location name and store the mapping in `box_goals`.
       e. Store the built `robot_graph` and `box_goals` in the `sokobanHeuristic._cache`
          using `task.name` as the key.
    3. If the data is cached, retrieve `robot_graph` and `box_goals` from the cache.
    4. Check if the current `state` is a goal state using `task.goal_reached(state)`.
       If it is, return 0.
    5. Parse the current `state`:
       a. Find the robot's location by iterating through facts starting with `'(at-robot '`.
       b. Build a dictionary `box_locations` mapping each box name to its current location
          by iterating through facts starting with `'(at '`.
    6. If the robot's location was not found (should not happen in valid states), return infinity.
    7. Initialize the total heuristic value `h` to 0.
    8. Iterate through each box and its goal location stored in `box_goals`.
    9. For a box `box_name` with goal `goal_loc`:
       a. Get the box's current location `current_loc` from `box_locations`.
       b. If `current_loc` is not found (should not happen for goal boxes in valid states),
          return infinity.
       c. If `current_loc` is not equal to `goal_loc`:
          i. Calculate `box_dist = bfs_distance(robot_graph, current_loc, goal_loc)`.
             This is the shortest path distance for the box's position. Add `box_dist` to `h`.
             If `box_dist` is infinity, return infinity (goal unreachable for box).
          ii. Calculate `robot_dist_to_push = min_robot_dist_to_push_box(robot_loc, current_loc, robot_graph)`.
              This is the minimum distance for the robot to reach a position adjacent to the box.
              Add `robot_dist_to_push` to `h`. If `robot_dist_to_push` is infinity, return infinity
              (robot cannot reach the box).
    10. Return the total heuristic value `h`.
    """
    # Check cache for pre-calculated static data
    task_key = task.name
    if task_key not in sokobanHeuristic._cache:
        # Pre-calculate static data
        robot_graph = {}
        box_goals = {}

        # Build robot graph from adjacent facts
        for fact in task.static:
            if fact.startswith('(adjacent '):
                parts = fact.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                # Add undirected edges for robot movement graph
                robot_graph.setdefault(loc1, set()).add(loc2)
                robot_graph.setdefault(loc2, set()).add(loc1)

        # Extract box goals from task.goals
        for goal_fact in task.goals:
            if goal_fact.startswith('(at '):
                parts = goal_fact.strip('()').split()
                box_name = parts[1]
                goal_loc = parts[2]
                box_goals[box_name] = goal_loc

        # Store in cache
        sokobanHeuristic._cache[task_key] = (robot_graph, box_goals)
    else:
        # Retrieve from cache
        robot_graph, box_goals = sokobanHeuristic._cache[task_key]

    # Check for goal state
    if task.goal_reached(state):
        return 0

    # Parse current state
    robot_loc = None
    box_locations = {}

    for fact in state:
        if fact.startswith('(at-robot '):
            robot_loc = fact.strip('()').split()[1]
        elif fact.startswith('(at '):
            parts = fact.strip('()').split()
            box_name = parts[1]
            current_loc = parts[2]
            box_locations[box_name] = current_loc

    # If robot location is not found, the state is malformed
    if robot_loc is None:
        # This indicates an invalid state representation
        return float('inf')

    # Calculate heuristic
    h = 0
    for box_name, goal_loc in box_goals.items():
        current_loc = box_locations.get(box_name)

        # If a box specified in the goal is not found in the current state,
        # the state is malformed or unreachable.
        if current_loc is None:
             return float('inf')

        if current_loc != goal_loc:
            # Component 1: Box distance to goal (minimum pushes)
            # This assumes the box can move along the robot_graph paths,
            # which is a simplification ignoring other boxes and clear locations.
            box_dist = bfs_distance(robot_graph, current_loc, goal_loc)
            if box_dist == float('inf'):
                 # Box goal is unreachable from its current location on the grid
                 # This state is likely unsolvable
                 return float('inf')

            h += box_dist

            # Component 2: Robot distance to box (to get into a push position)
            # Calculate minimum distance for robot to reach any location adjacent to the box
            robot_dist_to_push = min_robot_dist_to_push_box(robot_loc, current_loc, robot_graph)
            if robot_dist_to_push == float('inf'):
                 # Robot cannot reach any location adjacent to the box
                 # This state is likely unsolvable
                 return float('inf')

            h += robot_dist_to_push

    return h
