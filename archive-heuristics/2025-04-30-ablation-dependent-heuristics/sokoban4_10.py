# Required imports
import collections
import math

# Helper functions for graph traversal and direction handling

def get_opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen in this domain

def get_adjacent_location(loc, direction, adjacency_graph):
    """Returns the location adjacent to loc in the given direction."""
    return adjacency_graph.get(loc, {}).get(direction)

def get_direction(loc1, loc2, adjacency_graph):
    """Returns the direction from loc1 to loc2 if they are adjacent."""
    for direction, neighbor in adjacency_graph.get(loc1, {}).items():
        if neighbor == loc2:
            return direction
    return None # Not adjacent

def bfs(start_loc, target_loc, adjacency_graph, obstacles):
    """
    Finds the shortest path distance from start_loc to target_loc
    in the adjacency_graph, avoiding obstacles.
    Obstacles are locations the path cannot pass through.
    Returns distance or math.inf if unreachable.
    """
    if start_loc == target_loc:
        return 0

    # Locations the path cannot pass through
    impassable_locations = set(obstacles)

    # Start location cannot be an obstacle for the path starting there
    impassable_locations.discard(start_loc)
    # Target location cannot be an obstacle for the path ending there
    impassable_locations.discard(target_loc)


    if start_loc in impassable_locations:
        # This check is technically redundant after discard, but good for clarity
        return math.inf

    queue = collections.deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, dist = queue.popleft()

        # Check neighbors in all directions
        for direction, neighbor_loc in adjacency_graph.get(current_loc, {}).items():
            if neighbor_loc == target_loc:
                 return dist + 1 # Found shortest path

            if neighbor_loc not in visited and neighbor_loc not in impassable_locations:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, dist + 1))

    return math.inf # Target not reachable

def bfs_path(start_loc, target_loc, adjacency_graph, obstacles):
    """
    Finds a shortest path from start_loc to target_loc
    in the adjacency_graph, avoiding obstacles.
    Obstacles are locations the path cannot pass through.
    Returns a list of locations representing the path, or None if unreachable.
    """
    if start_loc == target_loc:
        return [start_loc]

    impassable_locations = set(obstacles)
    impassable_locations.discard(start_loc)
    impassable_locations.discard(target_loc)

    if start_loc in impassable_locations:
         return None # Cannot start in an obstacle

    queue = collections.deque([(start_loc, [start_loc])])
    visited = {start_loc}

    while queue:
        current_loc, path = queue.popleft()

        # Check neighbors in all directions
        for direction, neighbor_loc in adjacency_graph.get(current_loc, {}).items():
            if neighbor_loc == target_loc:
                 return path + [neighbor_loc] # Found a shortest path

            if neighbor_loc not in visited and neighbor_loc not in impassable_locations:
                visited.add(neighbor_loc)
                queue.append((neighbor_loc, path + [neighbor_loc]))

    return None # Target not reachable


class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning task.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each box to reach its designated goal location.
    The estimated cost for a single box is the sum of:
    1. The shortest path distance for the box from its current location
       to its goal location, considering other boxes as obstacles.
    2. The shortest path distance for the robot from its current location
       to a position adjacent to the box, from which it can push the box
       along the first step of its shortest path towards the goal,
       considering all boxes as obstacles.
    If any box cannot reach its goal location (e.g., due to deadlocks),
    the heuristic returns a very high value (infinity).

    Assumptions:
    - The PDDL domain follows the standard Sokoban rules as described.
    - Locations are connected via 'adjacent' facts, forming a graph.
    - The goal state specifies a unique target location for each box using '(at box goal_loc)' facts.
    - The state representation includes '(at-robot ?l)', '(at ?b ?l)', and '(clear ?l)' facts.
    - Static facts include all 'adjacent' relations.

    Heuristic Initialization:
    The constructor processes the static information from the task:
    - Stores the task object for later access to goal state checking.
    - Builds an adjacency graph from 'adjacent' facts to represent the grid/map connectivity.
    - Extracts the goal location for each box from the task's goal state.
    - Identifies the set of boxes that need to reach a goal location.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to find the robot's location and the current location of each box.
    2. Check if the current state is the goal state using the stored task object. If yes, return 0.
    3. Initialize the total heuristic value to 0.
    4. Get the set of locations currently occupied by boxes. This set will be used as obstacles for robot pathfinding.
    5. For each box that has a designated goal location:
        a. Get the box's current location from the state and its target goal location (determined during initialization).
        b. If the box is not found in the current state, or its goal is not defined (should not happen in valid problems), return infinity.
        c. If the box is already at its goal location, the cost for this box is 0; continue to the next box.
        d. Determine the obstacles for the box's path: the locations of all *other* boxes.
        e. Calculate the shortest path distance for the box from its current location to its goal location using BFS on the adjacency graph, avoiding the box obstacles.
        f. If the box's goal location is unreachable (distance is infinity), the state is likely a deadlock for this box. Return a very high heuristic value (infinity) immediately.
        g. Find a shortest path for the box from its current location to its goal location using BFS, avoiding box obstacles. This path is needed to determine the first step.
        h. If no path is found (should not happen if distance is finite and not at goal), return infinity.
        i. Identify the location of the first step the box needs to take along the shortest path towards the goal.
        j. Determine the required robot push position: the location adjacent to the box's current location, in the direction opposite to the first step identified in 5i.
        k. If the required push position does not exist (e.g., the box is against a wall and the path requires pushing into it), return a very high heuristic value (infinity) as this move is impossible.
        l. Calculate the shortest path distance for the robot from its current location to the required push position (found in step 5j) using BFS on the adjacency graph, considering *all* box locations as obstacles.
        m. If the robot cannot reach the required push position (distance is infinity), return a very high heuristic value (infinity) as the box cannot be moved towards the goal from its current position.
        n. If reachable, the estimated cost for this box is the sum of the box's path distance (step 5e) and the robot's path distance (step 5m).
        o. Add this estimated cost to the total heuristic value.
    6. Return the total heuristic value.
    """

    def __init__(self, task):
        self.task = task # Store task for goal checking

        # Build adjacency graph from static facts
        self.adjacency_graph = collections.defaultdict(dict)
        for fact in task.static:
            if fact.startswith('(adjacent '):
                parts = fact.strip('()').split()
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                self.adjacency_graph[loc1][direction] = loc2
                # Assuming adjacency is symmetric, add the reverse direction
                self.adjacency_graph[loc2][get_opposite_direction(direction)] = loc1

        # Extract box goal locations from task goals
        self.box_goals = {}
        for goal_fact in task.goals:
             if goal_fact.startswith('(at '):
                 parts = goal_fact.strip('()').split()
                 box_name = parts[1]
                 goal_loc = parts[2]
                 self.box_goals[box_name] = goal_loc # Assuming one goal location per box

        # Identify the set of boxes that need to reach a goal location
        self.all_boxes_with_goals = set(self.box_goals.keys())


    def __call__(self, state):
        # 1. Parse current state
        robot_loc = None
        box_locations = {} # {box_name: current_loc}

        for fact in state:
            if fact.startswith('(at-robot '):
                robot_loc = fact.strip('()').split()[1]
            elif fact.startswith('(at '):
                parts = fact.strip('()').split()
                box_name = parts[1]
                current_loc = parts[2]
                box_locations[box_name] = current_loc

        # 2. Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # Ensure robot location is found
        if robot_loc is None:
             # Robot not found in state? Invalid state.
             return math.inf

        # 3. Initialize total heuristic
        total_heuristic = 0

        # 4. Locations occupied by boxes (used as obstacles for robot path)
        all_box_locations_set = set(box_locations.values())

        # 5. For each box that has a designated goal location:
        for box_name in self.all_boxes_with_goals:
            current_box_loc = box_locations.get(box_name)
            goal_box_loc = self.box_goals.get(box_name)

            # 5b. Check if box is in state and has a goal
            if current_box_loc is None or goal_box_loc is None:
                 # This box should be in the state if it's in the initial state and goal.
                 # If not found, something is wrong with the state or task definition.
                 return math.inf

            # 5c. If box is already at goal, cost for this box is 0
            if current_box_loc == goal_box_loc:
                 total_heuristic += 0
                 continue

            # 5d. Determine the obstacles for the box's path: the locations of all *other* boxes.
            # Box path cannot go through locations occupied by other boxes.
            box_obstacles = all_box_locations_set - {current_box_loc}

            # 5e. Calculate box-to-goal distance
            dist_box_to_goal = bfs(current_box_loc, goal_box_loc, self.adjacency_graph, box_obstacles)

            # 5f. Check for box deadlock
            if dist_box_to_goal == math.inf:
                # Box cannot reach its goal. This state is likely a deadlock.
                return math.inf # Return a very high value

            # 5g. Find a shortest path for the box (needed for the first step)
            box_path = bfs_path(current_box_loc, goal_box_loc, self.adjacency_graph, box_obstacles)

            # 5h. Check if path found (should be if dist_box_to_goal is finite and not at goal)
            if box_path is None or len(box_path) < 2:
                 # This case should be covered by dist_box_to_goal == math.inf or current_box_loc == goal_box_loc
                 # but keep as a safety check.
                 return math.inf

            # 5i. The first step for the box is from box_path[0] to box_path[1]
            next_box_loc = box_path[1]

            # Find the direction of the first step
            direction_of_push = get_direction(current_box_loc, next_box_loc, self.adjacency_graph)

            # 5j. Determine the required robot push position
            # Robot needs to be adjacent to current_box_loc in the opposite direction
            required_robot_dir = get_opposite_direction(direction_of_push)
            required_robot_push_pos = get_adjacent_location(current_box_loc, required_robot_dir, self.adjacency_graph)

            # 5k. Check if the required push position exists
            if required_robot_push_pos is None:
                 # This means the box is against a wall/edge and cannot be pushed
                 # in the required direction (opposite of the first step towards goal).
                 # This is a form of deadlock or impossible move.
                 return math.inf

            # 5l. Calculate robot-to-push-position distance
            # Obstacles for the robot's path: all boxes.
            # Robot path cannot go through locations occupied by any box.
            robot_obstacles = all_box_locations_set

            dist_robot_to_push_pos = bfs(robot_loc, required_robot_push_pos, self.adjacency_graph, robot_obstacles)

            # 5m. Check if robot can reach the push position
            if dist_robot_to_push_pos == math.inf:
                 # Robot cannot get into position to push this box.
                 # This box is currently blocked for movement towards the goal.
                 return math.inf

            # 5n. Estimated cost for this box
            # Sum of robot moves to get ready + box pushes
            # A common non-admissible heuristic is sum of (robot_dist + box_dist).
            # This counts robot moves to get to the *first* push position + number of pushes.
            cost_for_box = dist_robot_to_push_pos + dist_box_to_goal

            # 5o. Add to total heuristic
            total_heuristic += cost_for_box

        # 6. Return total heuristic
        return total_heuristic
