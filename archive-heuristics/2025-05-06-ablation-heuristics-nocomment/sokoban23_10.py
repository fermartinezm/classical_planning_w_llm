import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse PDDL fact string into parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def build_graph(adjacent_facts):
    """Builds an adjacency list graph from adjacent facts."""
    graph = collections.defaultdict(list)
    for fact in adjacent_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            loc1, loc2, direction = parts[1], parts[2], parts[3]
            graph[loc1].append(loc2)
            # Assuming adjacency is symmetric, add the reverse edge
            # PDDL adjacent facts are often given in both directions, but being safe
            if loc2 not in graph or loc1 not in graph[loc2]: # Avoid duplicates if both directions are listed
                 graph[loc2].append(loc1)
    return graph

def bfs_distance(start, goals, graph, blocked_locations):
    """
    Performs BFS to find the shortest distance from start to any goal location,
    avoiding blocked locations.
    """
    if start in blocked_locations:
        return float('inf') # Cannot start from a blocked location

    # If start is one of the goals and not blocked, distance is 0
    if start in goals:
        return 0

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc not in graph:
             continue # Location has no neighbors (e.g., wall in grid representation)

        for neighbor in graph[current_loc]:
            if neighbor not in visited and neighbor not in blocked_locations:
                if neighbor in goals:
                    return dist + 1 # Found a goal
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float('inf') # No path found

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing two components
    for each box that is not yet at its goal location:
    1. The shortest path distance for the box to reach its goal location on the static
       grid (ignoring dynamic obstacles like other boxes or the robot). This estimates
       the minimum number of pushes required for the box.
    2. The shortest path distance for the robot to reach *any* location adjacent to
       the box's current location, considering other boxes (but not the box itself)
       as obstacles. This estimates the robot's effort to get into a position to
       potentially push the box.

    The total heuristic is the sum of these combined costs for all misplaced boxes.
    It is non-admissible because it doesn't account for the specific push direction
    needed, the need to move other boxes out of the way, or the fact that pushing
    a box also moves the robot. However, it provides a reasonable estimate of the
    work required (moving boxes + moving robot positioning).

    Assumptions:
    - The 'adjacent' facts define a connected graph representing the traversable
      locations in the environment.
    - The goal is specified by '(at box goal_location)' facts for one or more boxes.

    Heuristic Initialization:
    The constructor pre-processes the static facts to build an adjacency graph
    of the locations based on the 'adjacent' predicates. It also extracts the
    goal locations for each box from the task's goal state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state from the input node.
    2. Parse the state to find the robot's current location and the current location
       of each box.
    3. Identify locations currently occupied by boxes. These locations are considered
       blocked for robot movement.
    4. Initialize the total heuristic value to 0.
    5. Iterate through each box for which a goal location is defined.
    6. For the current box, get its current location and its goal location.
    7. If the box is already at its goal location, add 0 cost for this box and continue
       to the next box.
    8. If the box is not at its goal location:
        a. Calculate the shortest path distance for the box from its current location
           to its goal location using BFS on the static location graph. This BFS ignores
           dynamic obstacles (other boxes, robot). This distance represents the minimum
           number of pushes needed for this box. If no path exists, the state is likely
           unsolvable or very bad; return infinity.
        b. Calculate the shortest path distance for the robot from its current location
           to *any* location adjacent to the box's current location. This BFS is performed
           on the static location graph, but treats locations occupied by *other* boxes
           (excluding the current box being considered) as blocked. If no path exists
           to any adjacent location, return infinity.
        c. Add the box distance and the robot distance to the total heuristic value.
    9. Return the total heuristic value. If any distance calculation resulted in
       infinity, the total heuristic will be infinity, indicating a likely dead end
       or unsolvable state.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Build the static graph from adjacent facts
        self.static_adj_graph = build_graph(static_facts)

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                obj_name = parts[1] # This is the box name
                loc = parts[2]
                # Assuming objects starting with 'box' are boxes based on examples.
                # A more robust way would parse the :objects section of the problem file.
                # For this problem, let's assume 'at ?o ?l' in goal means ?o is a box.
                self.box_goals[obj_name] = loc


    def __call__(self, node):
        state = node.state

        # Find robot and box locations in the current state
        robot_location = None
        box_locations = {} # box_name -> location
        all_box_locations_in_state = set() # Locations occupied by ANY box

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                obj_name = parts[1]
                # Check if this object is one of the boxes we care about (i.e., has a goal)
                if obj_name in self.box_goals:
                     loc = parts[2]
                     box_locations[obj_name] = loc
                     all_box_locations_in_state.add(loc)

        # Heuristic calculation
        total_heuristic = 0

        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box) # Get current location of the box

            # If box is not in state (shouldn't happen in valid states) or already at goal
            if current_loc is None or current_loc == goal_loc:
                continue # Box is already at its goal

            # Calculate box distance (BFS on static graph, no dynamic blocks for box path)
            # Box path BFS ignores robot and other boxes as dynamic obstacles
            dist_box = bfs_distance(current_loc, {goal_loc}, self.static_adj_graph, set())

            # Calculate robot distance to any location adjacent to the box
            # Robot path BFS is blocked by OTHER boxes (not the current box)
            robot_blocked_locations = set(all_box_locations_in_state) - {current_loc}

            # Find all locations adjacent to the current box location
            adjacent_to_box = self.static_adj_graph.get(current_loc, [])
            robot_goal_locations = set(adjacent_to_box)

            # If the box location has no adjacent locations (e.g., surrounded by walls),
            # it cannot be pushed. This state is likely unsolvable.
            if not robot_goal_locations:
                 return float('inf')

            dist_robot = bfs_distance(robot_location, robot_goal_locations, self.static_adj_graph, robot_blocked_locations)

            # If either distance is infinite, this path is likely unsolvable or very bad
            if dist_box == float('inf') or dist_robot == float('inf'):
                return float('inf') # Return a large value

            # The cost to move the box one step is roughly robot_dist_to_push_pos + 1 (push action).
            # The box_dist is the number of pushes.
            # A simple sum: dist_box + dist_robot_to_adjacent_location
            total_heuristic += dist_box + dist_robot

        # Ensure heuristic is 0 only at goal state (w.r.t. box locations)
        # The loop adds 0 if all goal boxes are placed.
        # If total_heuristic is 0, it means all boxes in self.box_goals are at their targets.
        # This is sufficient for the h=0 requirement relative to the box part of the goal.
        return total_heuristic
