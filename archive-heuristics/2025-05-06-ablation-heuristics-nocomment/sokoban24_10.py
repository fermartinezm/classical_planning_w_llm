from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_direction(loc1, loc2, graph):
    """Finds the direction from loc1 to loc2 based on the graph."""
    if loc1 in graph:
        for neighbor, direction in graph[loc1]:
            if neighbor == loc2:
                return direction
    return None

def get_opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None

def get_location_in_direction(loc, direction, graph):
    """Finds the location adjacent to loc in the given direction."""
    if loc in graph:
        for neighbor, dir in graph[loc]:
            if dir == direction:
                return neighbor
    return None

def bfs(start_loc, end_loc, graph, obstacles=None):
    """
    Performs BFS to find the shortest distance and predecessor map.
    Obstacles are locations that cannot be entered.
    """
    queue = deque([(start_loc, 0)])
    visited = {start_loc}
    pred = {start_loc: None}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end_loc:
            return (dist, pred)

        if current_loc in graph:
            for neighbor_loc, _ in graph[current_loc]:
                if neighbor_loc not in visited and (obstacles is None or neighbor_loc not in obstacles):
                    visited.add(neighbor_loc)
                    pred[neighbor_loc] = current_loc
                    queue.append((neighbor_loc, dist + 1))

    return (float('inf'), None) # Not reachable

def reconstruct_path(pred, start_loc, end_loc):
    """Reconstructs path from predecessor map, excluding start_loc."""
    if pred is None or end_loc not in pred or (pred[end_loc] is None and start_loc != end_loc):
        return None # Not reachable

    path = []
    curr = end_loc
    while curr is not None:
        path.append(curr)
        if curr == start_loc: # Stop when we reach the start
            break
        curr = pred[curr]

    if not path or path[-1] != start_loc: # Did not reach start_loc in pred map
         return None

    path.reverse()
    return path[1:] # Exclude start_loc

def get_push_position(box_loc, next_box_loc, graph):
    """
    Finds the required robot location to push box from box_loc to next_box_loc.
    """
    dir_to_push = get_direction(box_loc, next_box_loc, graph)
    if dir_to_push is None:
        return None # next_box_loc is not adjacent to box_loc

    push_dir = get_opposite_direction(dir_to_push)
    if push_dir is None:
        return None # Should not happen with valid directions

    push_loc = get_location_in_direction(box_loc, push_dir, graph)
    return push_loc

def count_turns(path, graph):
    """Counts turns in a path of locations."""
    if path is None or len(path) < 2:
        return 0

    turns = 0
    # Path includes start, so first step is path[0] -> path[1]
    # We need at least 3 locations to have a turn: l0 -> l1 -> l2
    for i in rangelen(path) - 2):
        loc1 = path[i]
        loc2 = path[i+1]
        loc3 = path[i+2]
        dir1 = get_direction(loc1, loc2, graph)
        dir2 = get_direction(loc2, loc3, graph)
        # If direction changes, it's a turn
        if dir1 is not None and dir2 is not None and dir1 != dir2:
            turns += 1
    return turns


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal by summing costs for each box
    that is not yet at its goal location. The cost for a single box is
    estimated as the minimum number of pushes required to move it to its
    goal (shortest path distance on the grid, ignoring other boxes),
    plus the estimated cost for the robot to get into position for the
    first push (considering other boxes as obstacles), plus an estimated
    cost for the robot to reposition itself after turns in the box's path.
    This heuristic is non-admissible as it ignores obstacles (other boxes)
    when calculating the box's path and simplifies robot repositioning costs.

    Assumptions:
    - Location names follow the format 'loc_R_C' where R and C are integers (although parsing is not strictly needed for the current BFS implementation).
    - The grid connectivity is defined solely by 'adjacent' facts in static_facts.
    - The goal is defined by '(at box location)' facts.
    - The robot is never carrying a box (based on the provided domain).
    - The heuristic assumes a simple cost model: 1 for a push, distance for robot movement, 2 moves for robot repositioning after a turn.

    Heuristic Initialization:
    The constructor processes static facts and goal facts to build necessary data structures:
    - self.adj_graph: An adjacency list representing the grid connectivity based on 'adjacent' facts. Keys are location names, values are lists of (adjacent_location_name, direction) tuples.
    - self.goal_locations: A dictionary mapping each box name to its target goal location name.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Get the current state, robot's location, and current locations of all boxes.
    2.  Initialize the total heuristic value to 0.
    3.  Create a set of locations currently occupied by boxes (these are obstacles for the robot).
    4.  Check if the current state is a goal state. If yes, return 0.
    5.  Iterate through each box and its goal location stored during initialization.
    6.  If a box is already at its goal location, its contribution to the heuristic is 0; continue to the next box.
    7.  If the box is not at its goal:
        a.  Calculate the shortest path distance for the *box* from its current location to its goal location using BFS on the grid graph. For simplicity and non-admissibility, this BFS ignores other boxes as obstacles for the box's movement.
        b.  If the goal is unreachable for the box (even without box obstacles), return a large value (e.g., 10000) indicating a likely dead-end or unsolvable state.
        c.  Add this distance (representing the minimum number of pushes) to the total heuristic.
        d.  If the distance is greater than 0 (meaning the box needs to move):
            i.  Determine the required robot location to perform the *first* push action along the shortest path found for the box. This is the location adjacent to the box's current location in the direction opposite the first step of the box's path.
            ii. Calculate the shortest path distance for the *robot* from its current location to this required push position using BFS. This BFS considers *all* locations occupied by boxes as obstacles.
            iii. If the push position is unreachable for the robot, return a large value (e.g., 10000).
            iv. Add this robot distance to the total heuristic.
            v.  Count the number of turns in the shortest path found for the box (including the starting location).
            vi. Add an estimated cost for robot repositioning after each turn (e.g., 2 actions per turn) to the total heuristic.
    8.  Return the calculated total heuristic value.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        self.adj_graph = {}

        # Build adjacency graph
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, l1, l2, dir = get_parts(fact)
                self.adj_graph.setdefault(l1, []).append((l2, dir))

        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.goal_locations[box] = loc

    def __call__(self, node):
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf') # Robot location unknown

        # Find current box locations
        current_box_locations = {}
        box_obstacle_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                current_box_locations[box] = loc
                box_obstacle_locations.add(loc)

        # Check if goal is reached (all boxes at goal locations)
        is_goal_state = True
        for box_name, goal_loc in self.goal_locations.items():
             if box_name not in current_box_locations or current_box_locations[box_name] != goal_loc:
                 is_goal_state = False
                 break

        if is_goal_state:
             return 0

        total_heuristic = 0

        # Calculate heuristic for each box not at its goal
        for box_name, goal_loc in self.goal_locations.items():
            if box_name not in current_box_locations:
                 # Box is missing? Should not happen in valid states.
                 return float('inf') # Indicate unsolvable/invalid state

            current_loc = current_box_locations[box_name]

            if current_loc == goal_loc:
                continue # This box is already at its goal

            # 1. Cost for the box to reach its goal (minimum pushes)
            # BFS for box path, ignoring other boxes as obstacles for the box itself
            dist_box, pred_box = bfs(current_loc, goal_loc, self.adj_graph, obstacles=None)

            if dist_box == float('inf'):
                # Box cannot reach goal even without obstacles - likely unsolvable state
                return 10000 # Use a large finite number instead of inf

            total_heuristic += dist_box # Add minimum pushes

            # 2. Cost for the robot to get into position for the first push
            if dist_box > 0: # Only if the box needs to move
                box_path_locations = reconstruct_path(pred_box, current_loc, goal_loc)
                if box_path_locations is None or len(box_path_locations) == 0:
                     # Should not happen if dist_box > 0 and BFS was correct
                     return 10000 # Error state

                first_step_target = box_path_locations[0]
                push_loc = get_push_position(current_loc, first_step_target, self.adj_graph)

                if push_loc is None:
                     # Cannot find a push position for the first step - indicates a problem
                     # (e.g., box is in a corner/against wall and cannot move in the required direction)
                     # This should ideally be caught by dist_box being inf, but double check.
                     # If dist_box was finite, but push_loc is None, something is wrong.
                     # For safety, return large value.
                     return 10000

                # Obstacles for the robot are all locations occupied by boxes
                robot_obstacles = box_obstacle_locations
                robot_dist, _ = bfs(robot_loc, push_loc, self.adj_graph, obstacles=robot_obstacles)

                if robot_dist == float('inf'):
                    # Robot cannot reach the required push position
                    return 10000

                total_heuristic += robot_dist # Add robot movement cost

                # 3. Estimated cost for robot repositioning after turns
                # Path includes current_loc + box_path_locations
                full_box_path = [current_loc] + box_path_locations
                num_turns = count_turns(full_box_path, self.adj_graph)
                total_heuristic += 2 * num_turns # Add cost for turns (approx 2 moves per turn)

        return total_heuristic
