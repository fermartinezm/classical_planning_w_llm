from fnmatch import fnmatch
from collections import deque
import math

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have at least as many parts as args, unless args has wildcards
    if len(parts) < len(args) and '*' not in args:
         return False
    # Use zip to handle cases where parts might be longer than args (if args has *)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_name):
    """Convert location name 'loc_R_C' to (R, C) tuple."""
    parts = loc_name.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            return (int(parts[1]), int(parts[2]))
        except ValueError:
            pass # Not a standard loc_R_C format
    return None # Or raise error, depending on expected input

def coords_to_location_name(coords):
    """Convert (R, C) tuple to location name 'loc_R_C'."""
    r, c = coords
    return f'loc_{r}_{c}'

def build_adj_map_and_coords_and_dirs(static_facts):
    """
    Build adjacency map, location-coordinate mapping, and direction map from adjacent facts.
    Returns (adj_map, location_name_to_coords, coords_to_location_name, location_direction_map).
    adj_map: {loc_name: [neighbor_loc_name, ...]}
    location_name_to_coords: {loc_name: (r, c)}
    coords_to_location_name: {(r, c): loc_name}
    location_direction_map: {loc_name: {direction: neighbor_loc_name}}
    """
    adj_map = {}
    location_name_to_coords = {}
    coords_to_location_name = {}
    location_direction_map = {} # {loc_name: {direction: neighbor_loc_name}}

    for fact in static_facts:
        if match(fact, "adjacent", "*", "*", "*"):
            _, loc1, loc2, direction = get_parts(fact)

            if loc1 not in adj_map:
                adj_map[loc1] = []
                location_direction_map[loc1] = {}
            if loc2 not in adj_map:
                adj_map[loc2] = []
                location_direction_map[loc2] = {}

            # Add adjacency
            adj_map[loc1].append(loc2)
            adj_map[loc2].append(loc1)

            # Add direction mapping
            location_direction_map[loc1][direction] = loc2
            # Infer reverse direction
            rev_dir_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
            reverse_direction = rev_dir_map.get(direction)
            if reverse_direction:
                 location_direction_map[loc2][reverse_direction] = loc1


            # Infer coordinates if using loc_R_C format
            coords1 = parse_location(loc1)
            coords2 = parse_location(loc2)

            if coords1 and coords2:
                location_name_to_coords[loc1] = coords1
                coords_to_location_name[coords1] = loc1
                location_name_to_coords[loc2] = coords2
                coords_to_location_name[coords2] = loc2

    # Remove duplicates from adj_map lists
    for loc in adj_map:
        adj_map[loc] = list(set(adj_map[loc]))

    return adj_map, location_name_to_coords, coords_to_location_name, location_direction_map

def shortest_path_bfs(start, end, obstacles, adj_map):
    """
    Find the shortest path distance from start to end avoiding obstacles,
    and return the location of the first step on the path.
    Returns (distance, first_step_location).
    If start == end, returns (0, None).
    If no path, returns (math.inf, None).
    """
    if start == end:
        return (0, None)

    queue = deque([(start, 0)])
    visited = {start}
    predecessor = {start: None} # {location: predecessor_location}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            # Path found, reconstruct path to find the first step
            path = []
            step = end
            while step is not None:
                path.append(step)
                step = predecessor[step]
            path.reverse() # Path is now [start, step1, step2, ..., end]
            return (dist, path[1] if len(path) > 1 else None) # Return distance and step1

        # Check neighbors
        if current_loc in adj_map:
            for neighbor in adj_map[current_loc]:
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    predecessor[neighbor] = current_loc
                    queue.append((neighbor, dist + 1))

    # If queue is empty and end not reached
    return (math.inf, None)


def find_push_pos_from_map(box_loc, next_box_loc, location_direction_map):
    """
    Find the location the robot must be at to push box from box_loc to next_box_loc,
    using the pre-built location_direction_map.
    Returns the push_pos location name, or None if invalid.
    """
    # Find the direction from box_loc to next_box_loc
    push_dir = None
    if box_loc in location_direction_map:
        for direction, neighbor_loc in location_direction_map[box_loc].items():
            if neighbor_loc == next_box_loc:
                push_dir = direction
                break

    if push_dir is None:
        # next_box_loc is not directly adjacent to box_loc? Should not happen.
        return None

    # Find the location adjacent to box_loc in the opposite direction
    opposite_dir_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    required_robot_dir = opposite_dir_map.get(push_dir)

    if required_robot_dir is None:
        return None # Invalid direction

    if box_loc in location_direction_map and required_robot_dir in location_direction_map[box_loc]:
        return location_direction_map[box_loc][required_robot_dir]

    return None # No location found in the opposite direction


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated
    costs for each box that is not yet at its goal location. The estimated cost
    for a single box is the sum of:
    1. The shortest path distance (in robot moves) for the robot to reach a
       position from which it can push the box towards its goal.
    2. The shortest path distance (in pushes) for the box to reach its goal
       location, considering other boxes as obstacles.

    # Assumptions
    - The grid structure and adjacency are defined by the 'adjacent' facts.
    - Locations are named in 'loc_R_C' format, allowing coordinate parsing (though BFS uses names).
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - The heuristic treats each box independently for summation, ignoring
      complex interactions like multiple boxes needing the same path or robot position.
    - Dead ends (boxes pushed into unrecoverable positions) are handled by
      returning a large heuristic value if a box cannot reach its goal or the
      robot cannot reach a pushing position.

    # Heuristic Initialization
    - Parses static facts to build:
        - An adjacency map (`adj_map`) representing the grid connectivity.
        - A mapping from location names (`loc_R_C`) to `(row, col)` coordinates (`location_name_to_coords`).
        - A mapping from `(row, col)` coordinates to location names (`coords_to_location_name`).
        - A map from location and direction to neighbor location (`location_direction_map`).
    - Parses goal conditions to store the target location for each box (`goal_locations`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes.
    2. For each box that is not currently at its goal location:
       a. Determine the set of locations occupied by *other* boxes. These are obstacles for the current box's path.
       b. Calculate the shortest path distance (minimum pushes) for the box from its current location to its goal location, considering other boxes as obstacles. This is done using BFS on the adjacency graph. The BFS also returns the location of the first step on this shortest path. If no path exists, the box is in a dead end, and its cost is set to a large value.
       c. If the box is not in a dead end, find the location (`push_pos`) from which the robot must make the *first* push along the box's shortest path towards the goal. This is determined using the pre-calculated `location_direction_map`.
       d. Determine the set of locations occupied by *all* boxes. These are obstacles for the robot's path.
       e. Calculate the shortest path distance (minimum robot moves) for the robot from its current location to the calculated `push_pos`, considering all boxes as obstacles. This is also done using BFS. If no path exists, the robot cannot reach the pushing position, and its cost is set to a large value.
       f. The estimated cost for this individual box is the sum of the robot's approach distance and the box's path distance (pushes).
    3. The total heuristic value for the state is the sum of the estimated costs for all boxes that are not at their goals. If any individual box cost was set to the large value (indicating a dead end or unreachable box), the total heuristic is also the large value.
    4. If all boxes are at their goal locations, the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build adjacency map, coordinate mappings, and direction map
        self.adj_map, self.location_name_to_coords, self.coords_to_location_name, self.location_direction_map = build_adj_map_and_coords_and_dirs(self.static)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is typically (at box loc)
            if match(goal, "at", "*", "*"):
                _, box, location = get_parts(goal)
                self.goal_locations[box] = location

        # Define a large value for unreachable states
        self.UNREACHABLE_COST = 1000000

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        is_goal = True
        for box, goal_loc in self.goal_locations.items():
            if f'(at {box} {goal_loc})' not in state:
                is_goal = False
                break
        if is_goal:
            return 0

        # Extract current robot and box locations
        robot_loc = None
        box_locations = {} # {box_name: location_name}

        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_locations[box] = loc

        total_heuristic = 0

        # Calculate cost for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # If box is not in the state (shouldn't happen in Sokoban) or already at goal, skip
            if current_box_loc is None or current_box_loc == goal_loc:
                continue

            # --- Calculate box path cost ---
            # Obstacles for box path are other boxes
            other_box_locations = {
                loc for b, loc in box_locations.items() if b != box
            }
            box_path_info = shortest_path_bfs(current_box_loc, goal_loc, other_box_locations, self.adj_map)
            box_dist = box_path_info[0]
            box_next_loc = box_path_info[1] # First step location on the box's path

            if box_dist == math.inf:
                # Box is in a dead end
                return self.UNREACHABLE_COST # Return large value immediately

            # --- Calculate robot approach cost for the first push ---
            # Find the required push position for the first step of the box path
            # This is the location adjacent to current_box_loc in the opposite direction of box_next_loc
            push_pos = find_push_pos_from_map(current_box_loc, box_next_loc, self.location_direction_map)

            if push_pos is None:
                 # Should not happen if box_next_loc is a valid adjacent location, but handle defensively
                 return self.UNREACHABLE_COST

            # Obstacles for robot path are all boxes
            all_box_locations = set(box_locations.values())
            # Robot cannot move into a cell occupied by a box.
            robot_approach_info = shortest_path_bfs(robot_loc, push_pos, all_box_locations, self.adj_map)
            robot_approach_dist = robot_approach_info[0]

            if robot_approach_dist == math.inf:
                # Robot cannot reach the pushing position
                return self.UNREACHABLE_COST # Return large value immediately

            # --- Estimate cost for this box ---
            # Heuristic: robot moves to push_pos + box pushes
            # This counts moves to get behind the box + number of pushes.
            # It's a simple, non-admissible estimate.
            box_cost = robot_approach_dist + box_dist

            total_heuristic += box_cost

        return total_heuristic
