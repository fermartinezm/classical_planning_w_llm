from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts and locations

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(location_str):
    """Parses a location string like 'loc_r_c' into a tuple (r, c)."""
    try:
        # Remove 'loc_' prefix and split by '_'
        parts = location_str.split('_')
        # Ensure there are enough parts and the row/col parts are integers
        if len(parts) == 3 and parts[0] == 'loc':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
             raise ValueError(f"Unexpected location format: {location_str}")
    except (ValueError, IndexError):
        # Handle unexpected location formats if necessary
        raise ValueError(f"Unexpected location format: {location_str}")

def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two locations."""
    r1, c1 = get_coords(loc1_str)
    r2, c2 = get_coords(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)

def bfs(start_node, graph, occupied_locations):
    """
    Performs BFS from start_node on the graph, avoiding occupied_locations.
    Returns a dictionary of reachable nodes and their distances.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current = queue.popleft()

        # Check if current node is in the graph and has neighbors defined.
        if current not in graph:
             continue

        for neighbor in graph[current]:
            # BFS explores nodes. A node is added to the queue/visited if it's reachable
            # and not yet visited. For robot movement, the neighbor location must be clear.
            if neighbor not in visited and neighbor not in occupied_locations:
                visited.add(neighbor)
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of Manhattan distances for each box to its respective goal location.
    2. The minimum shortest path distance for the robot to reach any location
       adjacent to any box that is not yet at its goal. The shortest path
       is calculated using BFS on the currently clear cells.

    # Assumptions
    - The grid structure and adjacency are defined by the 'adjacent' static facts.
    - Locations are named 'loc_r_c' allowing coordinate extraction for Manhattan distance.
    - The goal is defined by the final positions of the boxes.
    - The heuristic does not explicitly check for deadlocks (e.g., boxes pushed into corners),
      but returns infinity if a box needs pushing but the robot cannot reach any adjacent clear cell.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an adjacency graph representing the grid connectivity from the static 'adjacent' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Determine which locations are currently occupied by boxes. These locations are blocked for robot movement.
    3. Perform a Breadth-First Search (BFS) starting from the robot's current location on the full grid adjacency graph, considering only paths that do not enter locations occupied by boxes. This finds the shortest path distance from the robot to all reachable clear locations.
    4. Initialize the total heuristic cost `h` to 0.
    5. Initialize `min_robot_dist_to_push_pos` to infinity. This will track the minimum distance for the robot to reach a location where it *could* potentially push a box towards its goal.
    6. Iterate through each box and its goal location:
       a. Get the box's current location.
       b. If the box is not at its goal location:
          i. Set a flag `found_box_to_push` to True.
          ii. Add the Manhattan distance between the box's current location and its goal location to `h`. This estimates the minimum number of pushes required for this box.
          iii. Find all locations adjacent to the box's current location using the full adjacency graph.
          iv. For each adjacent location:
              - Check if this adjacent location is currently *clear* (i.e., not occupied by a box).
              - If it is clear and reachable by the robot (i.e., found in the BFS distances):
                  - Update `min_robot_dist_to_push_pos` with the minimum of its current value and the robot's distance to this adjacent location. This finds the closest clear spot next to any box that needs moving.
    7. If `found_box_to_push` is True:
       a. If `min_robot_dist_to_push_pos` is still infinity (meaning no reachable clear adjacent cell was found for any box needing a push), return infinity, as the state is likely unsolvable.
       b. Otherwise, add `min_robot_dist_to_push_pos` to `h`.
    8. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the
        full adjacency graph from static facts.
        """
        # Assuming task object has .goals and .static attributes
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goals are typically (at box loc)
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the full adjacency graph from static facts.
        # Graph maps location string to a list of adjacent location strings.
        self.full_adj_graph = {}
        for fact in self.static:
            # Match adjacent facts like (adjacent loc_1_1 loc_1_2 right)
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.full_adj_graph.setdefault(loc1, []).append(loc2)
                # Assuming adjacency is symmetric, add the reverse edge
                self.full_adj_graph.setdefault(loc2, []).append(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Identify current locations
        robot_location = None
        box_locations = {} # {box_name: location_str}
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, robot_location = get_parts(fact)
            elif match(fact, "at", "*", "*"):
                _, box_name, location = get_parts(fact)
                box_locations[box_name] = location

        # If robot location is not found (shouldn't happen in valid states), return infinity
        if robot_location is None:
             return float('inf')

        # 2. Determine occupied locations by boxes
        # Locations occupied by boxes are not clear for robot movement.
        occupied_locations_for_bfs = set(box_locations.values())

        # 3. Perform BFS from robot's current location on the full graph, avoiding occupied locations
        # This finds shortest paths for the robot through clear cells.
        robot_distances = bfs(robot_location, self.full_adj_graph, occupied_locations_for_bfs)

        # 4. Initialize heuristic cost
        h = 0
        min_robot_dist_to_push_pos = float('inf')
        found_box_to_push = False # Flag to check if there's any box not at goal

        # 6. Iterate through boxes and goals
        for box_name, goal_location in self.goal_locations.items():
            current_box_location = box_locations.get(box_name) # Use .get for safety

            # If box location is not found (shouldn't happen), skip or return inf
            if current_box_location is None:
                 # This indicates a problem with state parsing or task definition
                 return float('inf')

            # b. If the box is not at its goal
            if current_box_location != goal_location:
                found_box_to_push = True
                # ii. Add Manhattan distance for the box
                h += manhattan_distance(current_box_location, goal_location)

                # iii. Find potential push locations adjacent to the box
                # A potential push location is any location adjacent to the box's current location.
                # The robot needs to reach one of these *clear* adjacent locations.
                adjacent_to_box = self.full_adj_graph.get(current_box_location, [])

                # iv. For each adjacent location
                for adj_loc in adjacent_to_box:
                    # Check if this adjacent location is clear (not occupied by a box)
                    # and reachable by the robot (i.e., found in the BFS distances).
                    if adj_loc not in occupied_locations_for_bfs:
                         if adj_loc in robot_distances:
                             # Update min_robot_dist_to_push_pos
                             min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_distances[adj_loc])
                         # Else: adj_loc is clear but unreachable by robot (e.g., isolated clear area)
                         # This case doesn't help find a push position, so we ignore it.


        # 7. Add minimum robot distance if there are boxes to push and a push position is reachable
        # If found_box_to_push is True, we need the robot to get into position.
        # If min_robot_dist_to_push_pos is still infinity, it means no reachable clear adjacent cell was found for any box needing a push.
        # This implies the state is likely unsolvable or requires complex moves not captured by this simple heuristic.
        # Returning infinity is appropriate in this non-admissible heuristic.
        if found_box_to_push:
             if min_robot_dist_to_push_pos == float('inf'):
                 return float('inf')
             else:
                 h += min_robot_dist_to_push_pos

        # 8. Return total heuristic cost
        return h
