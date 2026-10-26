import math
# Assuming Heuristic base class is available in the planner environment
# from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator classes are available
# from task import Operator, Task

# Note: The code below assumes the existence of a base class `Heuristic`
# and a `Task` class with attributes `static` and `goals`, and a `node`
# object with a `state` attribute, as suggested by the problem description
# and provided code examples.

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple of its components.
    E.g., '(at box1 loc_4_4)' -> ('at', 'box1', 'loc_4_4')
    """
    # Remove outer parentheses and split by spaces
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function to get opposite direction
def opposite_direction(direction):
    """Returns the opposite direction string."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen with valid directions

# BFS to find shortest path and path
def bfs_path(start_loc, end_loc, obstacles, adj_list):
    """
    Finds the shortest path distance and the path itself between two locations
    on the grid defined by adj_list, avoiding locations in obstacles.

    Args:
        start_loc (str): The starting location name.
        end_loc (str): The target location name.
        obstacles (set): A set of location names that cannot be traversed.
        adj_list (dict): Adjacency list {loc: {direction: neighbor_loc}}.

    Returns:
        tuple (int, list) or None: A tuple containing the distance and the path
                                   as a list of location names, or None if
                                   the end_loc is unreachable from start_loc
                                   avoiding obstacles.
    """
    if start_loc == end_loc:
        return (0, [start_loc])
    # Cannot start inside an obstacle
    if start_loc in obstacles:
         return None

    queue = [(start_loc, [start_loc])]
    visited = {start_loc}
    distance = {start_loc: 0}

    while queue:
        (current_loc, path) = queue.pop(0)

        if current_loc == end_loc:
            return (distance[current_loc], path)

        # Check adjacent locations
        if current_loc in adj_list:
            for direction, neighbor_loc in adj_list[current_loc].items():
                # A location is traversable if it's not in the obstacles set
                if neighbor_loc not in visited and neighbor_loc not in obstacles:
                    visited.add(neighbor_loc)
                    distance[neighbor_loc] = distance[current_loc] + 1
                    queue.append((neighbor_loc, path + [neighbor_loc]))

    return None # End location not reachable


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing two components:
    1.  The total minimum number of pushes required for all boxes to reach their
        respective goal locations. This is calculated as the sum of shortest
        path distances for each box from its current location to its goal
        location, considering other boxes as obstacles.
    2.  The minimum number of robot moves required to reach a position from
        which it can make the first necessary push for any box that is not
        yet at its goal. This is calculated as the shortest path distance
        from the robot's current location to a valid pushing position
        adjacent to any box that needs moving, considering all boxes as obstacles.

    Assumptions:
    -   The 'adjacent' predicates correctly define the grid connectivity,
        including defining adjacency in both directions (e.g., (adjacent l1 l2 right)
        and (adjacent l2 l1 left)).
    -   The goal state specifies a unique goal location for each box using
        '(at box_name loc_name)' facts.
    -   Box names start with 'box'.
    -   The heuristic does not explicitly detect complex deadlocks (e.g., boxes
        trapped in corners with no adjacent clear space in the required
        pushing direction), but unreachable goals due to static obstacles
        or other boxes will result in an infinite heuristic value.

    Heuristic Initialization:
    -   Parses the static 'adjacent' facts to build an adjacency list
        representation of the grid graph, mapping {location: {direction: neighbor_location}}.
    -   Parses the goal facts to create a mapping from each box to its
        target goal location.
    -   Pre-computes a mapping from direction strings to their opposite.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify the current location of the robot and all boxes from the state.
    2.  Identify the set of boxes that are not yet at their goal locations.
    3.  If no boxes need moving, the state is a goal state, return 0.
    4.  Calculate the 'box_push_cost':
        -   Initialize total box distance to 0.
        -   Identify the locations of all boxes.
        -   For each box that needs moving:
            -   Determine its current location and its goal location.
            -   Identify obstacles for the box's path: the locations of all *other* boxes.
            -   Compute the shortest path distance from the box's current location to its goal location using BFS on the grid graph, avoiding obstacles.
            -   If the goal is unreachable for this box (BFS returns None), the state is likely unsolvable, return infinity.
            -   Add this distance (number of pushes) to the total box distance.
    5.  Calculate the 'robot_positioning_cost':
        -   Initialize minimum robot distance to infinity.
        -   Identify obstacles for the robot's path: the locations of *all* boxes.
        -   For each box that needs moving:
            -   Determine its current location and its goal location.
            -   Identify obstacles for the box's path (other boxes) again.
            -   Compute the shortest path (including the path itself) from the box's current location to its goal location using BFS, considering *other* boxes as obstacles.
            -   If the box path exists and has length > 1 (meaning the box needs at least one push to move to a different location):
                -   Find the location (L1) immediately after the box's current location (BoxLoc) on the shortest path to the goal.
                -   Determine the direction (D) from BoxLoc to L1 using the adjacency list.
                -   Determine the required pushing direction (opposite of D).
                -   Find the required pushing position (P0) adjacent to BoxLoc in the required pushing direction. This is the location N such that adjacent(N, BoxLoc, required_push_dir), which means BoxLoc is adjacent from N in required_push_dir, or N is adjacent from BoxLoc in opposite(required_push_dir). Using the adjacency list structure {loc: {dir: neighbor}}, P0 is the neighbor of BoxLoc in direction opposite(required_push_dir).
                -   Check if P0 exists in the adjacency list (i.e., BoxLoc has a neighbor in that direction).
                -   If P0 exists and is not occupied by *any* box (P0 must be clear for the robot to move there):
                    -   Compute the shortest path distance from the robot's current location to P0 using BFS, considering *all* boxes as obstacles.
                    -   If P0 is reachable by the robot (BFS returns non-None), update the minimum robot distance found so far.
    6.  If there are boxes to move but the minimum robot distance to any valid pushing position for any box is still infinity, the state is likely unsolvable, return infinity.
    7.  The heuristic value is the sum of the total box push distance and the minimum robot positioning distance.
    """
    def __init__(self, task):
        super().__init__(task)
        self.adj_list = self._build_adjacency_list(task.static)
        self.box_goals = self._parse_box_goals(task.goals)
        self.opposite_dirs = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

    def _build_adjacency_list(self, static_facts):
        # Builds {loc: {direction: neighbor_loc}}
        adj_list = {}
        for fact_string in static_facts:
            fact = parse_fact(fact_string)
            if fact[0] == 'adjacent':
                loc1, loc2, direction = fact[1], fact[2], fact[3]
                if loc1 not in adj_list:
                    adj_list[loc1] = {}
                adj_list[loc1][direction] = loc2
        return adj_list

    def _parse_box_goals(self, goal_facts):
        box_goals = {}
        for fact_string in goal_facts:
            fact = parse_fact(fact_string)
            if fact[0] == 'at' and fact[1].startswith('box'): # Assuming box names start with 'box'
                box_name, goal_loc = fact[1], fact[2]
                box_goals[box_name] = goal_loc
        return box_goals

    def __call__(self, node):
        state = node.state

        # 1. Identify current locations
        robot_loc = None
        box_locations = {} # {box_name: location}

        for fact_string in state: # Iterate directly over frozenset
            fact = parse_fact(fact_string)
            if fact[0] == 'at-robot':
                robot_loc = fact[1]
            elif fact[0] == 'at' and fact[1].startswith('box'):
                box_name, box_loc = fact[1], fact[2]
                box_locations[box_name] = box_loc

        # 2. Identify boxes not at goal
        boxes_to_move = [b for b, loc in box_locations.items() if self.box_goals.get(b) != loc]

        # 3. Goal state check
        if not boxes_to_move:
            return 0

        # 4. Calculate box_push_cost
        total_box_dist = 0
        all_box_locs = set(box_locations.values())

        for box_name in boxes_to_move:
            box_loc = box_locations[box_name]
            goal_loc = self.box_goals[box_name]
            # Obstacles for box path: locations of all *other* boxes
            other_boxes_locs = {loc for b_name, loc in box_locations.items() if b_name != box_name}

            # BFS for box path, avoiding other boxes
            box_path_result = bfs_path(box_loc, goal_loc, other_boxes_locs, self.adj_list)

            if box_path_result is None:
                # Box cannot reach its goal (e.g., trapped by other boxes or walls)
                return math.inf

            box_dist, box_path = box_path_result
            total_box_dist += box_dist

        # 5. Calculate robot_positioning_cost
        min_robot_dist_to_push_pos = math.inf
        # Obstacles for robot path: locations of *all* boxes
        robot_obstacles = all_box_locs

        for box_name in boxes_to_move:
            box_loc = box_locations[box_name]
            goal_loc = self.box_goals[box_name]
            # Obstacles for box path (other boxes) needed again
            other_boxes_locs = {loc for b_name, loc in box_locations.items() if b_name != box_name}


            # Need box path again to find the first step and required push position
            # Use BFS for box path, avoiding other boxes
            box_path_result = bfs_path(box_loc, goal_loc, other_boxes_locs, self.adj_list)

            # If box_path_result is None, we already returned inf in the previous loop.
            # If box_dist is 0, the box is at its goal (shouldn't be in boxes_to_move).
            if box_path_result is not None and box_path_result[0] > 0:
                box_path = box_path_result[1]
                L1 = box_path[1] # The location after the first push

                # Find direction from BoxLoc to L1
                direction_to_L1 = None
                if box_loc in self.adj_list:
                    for direction, neighbor in self.adj_list[box_loc].items():
                        if neighbor == L1:
                            direction_to_L1 = direction
                            break

                if direction_to_L1:
                    required_push_dir = self.opposite_dirs.get(direction_to_L1)
                    if required_push_dir:
                        # P0 is the location adjacent to box_loc in the direction opposite(required_push_dir)
                        # i.e., P0 is the neighbor N of box_loc such that adj_list[BoxLoc][N] == opposite(required_push_dir)
                        dir_from_box_to_P0 = self.opposite_dirs.get(required_push_dir)
                        P0 = None
                        if box_loc in self.adj_list and dir_from_box_to_P0 in self.adj_list[box_loc]:
                             P0 = self.adj_list[box_loc][dir_from_box_to_P0]

                        # Check if P0 is a valid location for the robot to pathfind to (must be clear of any box)
                        if P0 and P0 not in all_box_locs: # P0 must not be occupied by any box
                            # BFS for robot path from robot_loc to P0, avoiding all boxes
                            robot_path_result = bfs_path(robot_loc, P0, robot_obstacles, self.adj_list)
                            if robot_path_result is not None:
                                robot_dist = robot_path_result[0]
                                min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_dist)

        # 6. Check if robot can reach any pushing position if needed
        # If boxes_to_move is not empty, and min_robot_dist_to_push_pos is still infinity,
        # it means the robot cannot reach *any* valid pushing position (P0 must be clear)
        # for the first step of *any* box that needs moving.
        if boxes_to_move and min_robot_dist_to_push_pos == math.inf:
             return math.inf


        # 7. Calculate final heuristic value
        # Add total box pushes and minimum robot moves to get into position for *any* box
        h = total_box_dist + min_robot_dist_to_push_pos

        return h
