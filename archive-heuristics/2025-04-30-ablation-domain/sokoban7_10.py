from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this is available in the environment

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Simple check: if args has *, it might match multiple parts, but our match utility
    # assumes 1-to-one mapping. Let's stick to the logistics example's simple match.
    # A more robust check would be needed for complex patterns, but for simple facts like
    # (at obj loc) or (adjacent l1 l2 dir), this is fine.
    # Let's just return False if lengths differ, as per the zip behavior in the original match.
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location_string(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    try:
        # Expecting format like 'loc_1_1'
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
             return (int(parts[1]), int(parts[2]))
        else:
             # Handle unexpected format
             # print(f"Warning: Unexpected location string format '{loc_str}'") # Optional warning
             return None
    except ValueError:
        # Handle cases where R or C are not integers
        # print(f"Warning: Could not parse location string '{loc_str}' (integer conversion error)") # Optional warning
        return None

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the remaining cost by summing the shortest path
    distances for each box to its goal location and adding the shortest path
    distance from the robot's current location to the nearest box that is
    not yet at its goal. This aims to capture the work needed for both moving
    the boxes and positioning the robot.

    # Assumptions
    - The environment is a grid-like structure defined by 'adjacent' predicates.
    - Locations are named in the format 'loc_R_C'.
    - Each box has a unique goal location specified in the task goals.
    - The shortest path distance between locations on the grid graph is a
      reasonable estimate of movement cost for both boxes (pushes) and the robot.
    - Ignoring the specific side the robot needs to be on for pushing,
      and the cost of moving between pushes, provides a fast and useful estimate
      for greedy best-first search.
    - Unreachable goals or boxes imply infinite cost.

    # Heuristic Initialization
    1. Collect all unique location strings from the initial state and static facts.
    2. Map each location string ('loc_R_C') to its coordinate tuple (R, C).
    3. Build an adjacency graph where nodes are coordinates and edges are
       derived from 'adjacent' predicates in static facts. The graph is
       treated as bidirectional (since adjacent facts are typically provided
       in both directions).
    4. Precompute the shortest path distance between all pairs of location
       coordinates using Breadth-First Search (BFS) starting from each node.
       Store these distances in a dictionary keyed by (start_coord, end_coord).
    5. Extract the goal location string for each box from the task goals. Store
       these in a dictionary keyed by box name.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location string of the robot by finding the fact '(at-robot ?l)'.
    2. Identify the current location string for each box that has a goal by finding facts '(at ?b ?l)'.
    3. Initialize `total_box_distance = 0`. This will sum the minimum pushes needed for all boxes.
    4. Initialize `min_robot_box_distance = float('inf')`. This will store the robot's distance to the closest box needing a push.
    5. Create a list `boxes_to_move` containing names of boxes not yet at their goal location.
    6. For each box `b` that has a goal:
       a. Get its current location string from the state. If not found, the state is invalid/unreachable, return infinity.
       b. Get its goal location string from the precomputed `self.goal_locations`.
       c. If the current location is not the goal location, add the box name to `boxes_to_move`.
       d. Convert the current and goal location strings to coordinates using `self.loc_to_coord`. If conversion fails, return infinity.
       e. Find the precomputed shortest path distance `dist(current_coord, goal_coord)` from `self.coord_distances`.
       f. If the distance is infinity (meaning the box cannot reach its goal), the state is likely unsolvable; return infinity.
       g. Add this distance to `total_box_distance`.
    7. If `total_box_distance` is 0 (meaning all boxes are at their goals), the state is a goal state; return 0.
    8. Convert the robot's current location string to coordinates `robot_coord`. If conversion fails, return infinity.
    9. For each box `b` in the `boxes_to_move` list:
       a. Get its current location string and convert it to coordinates `box_coord`. If conversion fails, return infinity.
       b. Find the precomputed shortest path distance `dist(robot_coord, box_coord)` from `self.coord_distances`.
       c. Update `min_robot_box_distance` with the minimum distance found so far.
    10. If `min_robot_box_distance` is still infinity after checking all boxes (meaning the robot cannot reach any box that needs moving), return infinity.
    11. The heuristic value is the sum of `total_box_distance` and `min_robot_box_distance`.

    """

    def __init__(self, task):
        """Initialize the heuristic by building the graph and precomputing distances."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Collect all unique location strings
        location_strings = set()
        # Locations appear as arguments in initial state facts (at-robot, at, clear)
        # and static facts (adjacent).
        for fact in initial_state | static_facts:
            parts = get_parts(fact)
            for part in parts:
                if part.startswith('loc_'):
                    location_strings.add(part)

        # 2. Map location strings to coordinates and vice versa
        self.loc_to_coord = {}
        self.coord_to_loc = {}
        all_coords = []
        # Sort location strings to ensure consistent mapping order across runs/systems
        sorted_location_strings = sorted(list(location_strings))
        for loc_str in sorted_location_strings:
            coord = parse_location_string(loc_str)
            if coord: # Only add if parsing was successful
                self.loc_to_coord[loc_str] = coord
                self.coord_to_loc[coord] = loc_str
                all_coords.append(coord)

        # 3. Build the adjacency graph using coordinates
        self.coord_graph = {coord: [] for coord in all_coords}
        for fact in static_facts:
            # Look for adjacent predicates like (adjacent loc_1_1 loc_1_2 right)
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1_str, loc2_str, _ = get_parts(fact)
                coord1 = self.loc_to_coord.get(loc1_str)
                coord2 = self.loc_to_coord.get(loc2_str)
                # Add edge only if both locations were successfully parsed and mapped
                if coord1 is not None and coord2 is not None:
                    self.coord_graph[coord1].append(coord2)

        # 4. Precompute all-pairs shortest paths using BFS
        # Store distances in a nested dictionary: distances[start_coord][end_coord] = dist
        self.coord_distances = {c1: {c2: float('inf') for c2 in all_coords} for c1 in all_coords}
        for start_coord in all_coords:
            self._bfs(start_coord)

        # 5. Extract goal locations for each box
        self.goal_locations = {} # {box_name: goal_loc_string}
        for goal in self.goals:
            # Assuming goals are of the form (at box_name loc_name)
            if match(goal, "at", "*", "*"):
                 _, obj_name, goal_loc_str = get_parts(goal)
                 # Assuming objects with goals are boxes
                 self.goal_locations[obj_name] = goal_loc_str

    def _bfs(self, start_coord):
        """Performs BFS from a start coordinate to compute distances to all reachable coordinates."""
        queue = deque([(start_coord, 0)])
        visited = {start_coord}
        self.coord_distances[start_coord][start_coord] = 0

        while queue:
            (curr_coord, dist) = queue.popleft()

            # Check if curr_coord exists in the graph (should always be true if from all_coords)
            if curr_coord not in self.coord_graph:
                 continue # Should not happen with correct initialization

            for neighbor_coord in self.coord_graph[curr_coord]:
                if neighbor_coord not in visited:
                    visited.add(neighbor_coord)
                    self.coord_distances[start_coord][neighbor_coord] = dist + 1
                    queue.append((neighbor_coord, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify robot location string
        robot_loc_str = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc_str = get_parts(fact)[1]
                break

        # If robot location isn't found, state is invalid/unreachable
        if robot_loc_str is None:
             return float('inf')

        # Convert robot location string to coordinate
        robot_coord = self.loc_to_coord.get(robot_loc_str)
        # If robot location string couldn't be parsed or mapped, state is invalid/unreachable
        if robot_coord is None:
             return float('inf')


        # 2. Identify box locations for boxes that have goals
        box_loc_strs = {} # {box_name: loc_string}
        for fact in state:
            # Look for facts like (at box1 loc_4_4)
            if match(fact, "at", "*", "*"):
                _, obj_name, loc_str = get_parts(fact)
                # Only track locations for objects that are defined as boxes with goals
                if obj_name in self.goal_locations:
                    box_loc_strs[obj_name] = loc_str

        # 3. & 6. Calculate total box distance and identify boxes to move
        total_box_distance = 0
        boxes_to_move = [] # List of box names not at goal

        # Iterate through all boxes that have a goal defined in the task
        for box_name, goal_loc_str in self.goal_locations.items():
            current_loc_str = box_loc_strs.get(box_name)

            # If a box with a goal is not found in the state's 'at' predicates,
            # it might be in an unexpected state (e.g., not 'at' anywhere).
            # Treat this as an unreachable goal for this box.
            if current_loc_str is None:
                 return float('inf')

            # Check if the box is already at its goal
            if current_loc_str != goal_loc_str:
                boxes_to_move.append(box_name)

                # Convert current and goal location strings to coordinates
                current_coord = self.loc_to_coord.get(current_loc_str)
                goal_coord = self.loc_to_coord.get(goal_loc_str)

                # If current or goal location couldn't be parsed/mapped, treat as unreachable
                if current_coord is None or goal_coord is None:
                    return float('inf')

                # Get the precomputed distance between the box's current location and its goal
                # Use .get() with default float('inf') to handle cases where locations might not
                # have been reachable during BFS (e.g., isolated islands in the graph).
                box_dist = self.coord_distances.get(current_coord, {}).get(goal_coord, float('inf'))

                # 6d. If any box cannot reach its goal, the state is likely unsolvable
                if box_dist == float('inf'):
                    return float('inf')

                total_box_distance += box_dist

        # 7. If total_box_distance is 0, it means all boxes are at their goals.
        # Since the goal only requires boxes to be at specific locations, this is a goal state.
        if total_box_distance == 0:
            return 0

        # 4. & 9. Calculate minimum robot-box distance for boxes to move
        min_robot_box_distance = float('inf')

        # Only calculate robot distance if there are boxes that need moving
        if boxes_to_move:
            for box_name in boxes_to_move:
                box_loc_str = box_loc_strs[box_name]
                box_coord = self.loc_to_coord.get(box_loc_str)

                # This check should ideally not be needed if total_box_distance was finite,
                # but included for robustness.
                if box_coord is None:
                     return float('inf')

                # Get the precomputed distance from the robot's location to the box's location
                # Use .get() with default float('inf') for robustness
                robot_dist_to_box = self.coord_distances.get(robot_coord, {}).get(box_coord, float('inf'))

                # Update the minimum distance found so far
                min_robot_box_distance = min(min_robot_box_distance, robot_dist_to_box)

            # 10. If min_robot_box_distance is still infinity, it means the robot cannot reach
            # any of the boxes that need moving. This state is likely unsolvable.
            if min_robot_box_distance == float('inf'):
                 return float('inf')
        # else: # This case is covered by the total_box_distance == 0 check

        # 11. Calculate and return the heuristic value
        # The heuristic is the sum of the minimum pushes needed for boxes
        # plus the cost for the robot to reach the nearest box it needs to push.
        heuristic_value = total_box_distance + min_robot_box_distance

        return heuristic_value
