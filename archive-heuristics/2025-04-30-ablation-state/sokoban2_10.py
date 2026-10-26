from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated costs for each box
    to reach its goal location independently. The estimated cost for a single box considers the
    shortest path distance the box needs to travel and the robot's distance to a position from
    which it can initiate the first push towards the goal. It ignores the robot's movement
    between subsequent pushes of the same box and interactions between different boxes.

    # Assumptions
    - The grid structure and connectivity are defined solely by the `adjacent` facts.
    - Locations follow the 'loc_R_C' naming convention, allowing parsing into (row, column) tuples.
    - The heuristic ignores potential deadlocks caused by walls, other boxes, or corners.
    - The heuristic ignores the `clear` predicate during distance calculations, assuming locations
      are clear when needed for movement or pushing in an ideal scenario.
    - The cost of repositioning the robot between consecutive pushes of the same box is ignored
      in the sum for simplicity and efficiency.

    # Heuristic Initialization
    - Extracts all unique location strings from the task facts and static facts.
    - Creates mappings between location strings (e.g., "loc_1_1") and (row, column) integer tuples.
    - Builds an adjacency list and an adjacency direction map for the location graph based on `adjacent` static facts, using (row, column) tuples as nodes.
    - Computes all-pairs shortest paths and the first step on a shortest path for all pairs of valid (row, column) locations using BFS.
    - Stores the goal locations for each box, mapped to their (row, column) tuple representation.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state to find the robot's location string and the location string for each box.
    2. Convert these location strings into their corresponding (row, column) tuple representations using the precomputed mapping. If a location string cannot be mapped, the state is considered unsolvable (infinity heuristic).
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each box that has a specified goal location (precomputed during initialization).
    5. For the current box, get its current (row, column) location and its goal (row, column) location. If the box's current location cannot be mapped, treat this box's goal as unreachable (infinity contribution).
    6. If the box is already at its goal location, its contribution to the heuristic is 0. Continue to the next box.
    7. If the box is not at its goal:
       a. Compute the shortest path distance (`box_dist`) between the box's current location and its goal location using the precomputed all-pairs shortest paths on the location graph. If the goal is unreachable from the box's current location, the state is likely unsolvable; return infinity.
       b. Identify the location (row, column) that represents the first step on a shortest path from the box's current location towards its goal. This is retrieved from the precomputed first-step map.
       c. Determine the direction of this first step (e.g., 'up', 'down') using the precomputed adjacency direction map.
       d. Determine the required direction for the robot to be adjacent to the box to push it in that direction. This is the opposite of the first-step direction.
       e. Find all locations (row, column) that are adjacent to the box's current location in the required robot push direction, using the built adjacency list. These are the potential "push spots" for the robot to initiate the first push.
       f. Compute the minimum shortest path distance (`min_robot_dist`) from the robot's current location to any of these potential push spots using the precomputed shortest paths. If the robot cannot reach any valid push spot, the state is likely unsolvable; return infinity.
       g. Calculate the estimated cost for this box: `min_robot_dist` (cost for the robot to reach the first push spot) + `box_dist` (cost for the total number of push actions required).
       h. Add this calculated cost for the current box to the total heuristic sum.
8. Return the final total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and computing shortest paths."""
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # Use all facts to find all locations

        # 1. Extract all unique location strings
        all_locations_str = set()
        for fact in all_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['at', 'at-robot', 'clear']:
                 # Location is always the last argument in these predicates
                if len(parts) > 1: # Ensure there's an argument after the predicate
                    all_locations_str.add(parts[-1])
            # Also get locations from adjacent facts
            elif parts and parts[0] == 'adjacent' and len(parts) == 4:
                 all_locations_str.add(parts[1])
                 all_locations_str.add(parts[2])


        # 2. Create mappings between location strings and (row, column) tuples
        self.loc_str_to_rc = {}
        self.rc_to_loc_str = {}
        for loc_str in all_locations_str:
            parts = loc_str.split('_')
            # Assuming loc_R_C format. Handle potential errors gracefully.
            if len(parts) == 3 and parts[0] == 'loc':
                try:
                    r, c = int(parts[1]), int(parts[2])
                    self.loc_str_to_rc[loc_str] = (r, c)
                    self.rc_to_loc_str[(r, c)] = loc_str
                except ValueError:
                    # If a location string doesn't parse as loc_R_C, we skip it for RC mapping
                    # but it remains in all_locations_str if needed elsewhere.
                    # For this heuristic, we rely on RC mapping for graph/distances.
                    pass
            # If location format is different, we might need a different mapping strategy
            # For now, we only map loc_R_C formatted strings

        # Filter out locations that couldn't be parsed into RC tuples
        valid_locations_rc = list(self.rc_to_loc_str.keys())

        # 3. Build adjacency list and direction map for the location graph (using RC tuples)
        self.adj_list_rc = defaultdict(list)
        self.adj_dir_rc = defaultdict(dict)
        opp_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        self.opp_dir = opp_dir # Store for later use

        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, l1_str, l2_str, d_str = get_parts(fact)
                # Only add adjacency if both locations could be mapped to RC
                if l1_str in self.loc_str_to_rc and l2_str in self.loc_str_to_rc:
                    l1_rc = self.loc_str_to_rc[l1_str]
                    l2_rc = self.loc_str_to_rc[l2_str]
                    if l2_rc not in self.adj_list_rc[l1_rc]: # Avoid duplicates
                         self.adj_list_rc[l1_rc].append(l2_rc)
                    self.adj_dir_rc[l1_rc][l2_rc] = d_str
                    # Add reverse direction if it exists in opp_dir
                    if d_str in opp_dir:
                         if l1_rc not in self.adj_list_rc[l2_rc]: # Avoid duplicates
                              self.adj_list_rc[l2_rc].append(l1_rc)
                         self.adj_dir_rc[l2_rc][l1_rc] = opp_dir[d_str]


        # 4. Compute all-pairs shortest paths and first steps using BFS
        self.shortest_paths = {}
        self.first_step = {}

        # Perform BFS from every valid location
        for start_loc_rc in valid_locations_rc:
            q = deque([start_loc_rc])
            distance = {loc_rc: math.inf for loc_rc in valid_locations_rc}
            distance[start_loc_rc] = 0
            first_step_map = {start_loc_rc: start_loc_rc} # Map location to the next step on a shortest path from start_loc_rc

            self.shortest_paths[start_loc_rc] = distance
            self.first_step[start_loc_rc] = first_step_map

            # Use a set for visited nodes for faster lookup
            visited = {start_loc_rc}

            while q:
                curr_rc = q.popleft()

                # Only process if curr_rc has defined adjacencies in the graph we built
                if curr_rc in self.adj_list_rc:
                    for neighbor_rc in self.adj_list_rc[curr_rc]:
                        # Ensure neighbor is a valid RC location and not visited
                        if neighbor_rc in valid_locations_rc and neighbor_rc not in visited:
                            visited.add(neighbor_rc)
                            distance[neighbor_rc] = distance[curr_rc] + 1
                            # The first step to neighbor is the neighbor itself if curr is the start node,
                            # otherwise it's the same as the first step to curr.
                            first_step_map[neighbor_rc] = neighbor_rc if curr_rc == start_loc_rc else first_step_map[curr_rc]
                            q.append(neighbor_rc)

        # 5. Store goal locations for each box (as RC tuples)
        self.goal_locations_rc = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                _, box_str, loc_str = parts
                # Only store goal if the location string could be mapped to RC
                if loc_str in self.loc_str_to_rc:
                    self.goal_locations_rc[box_str] = self.loc_str_to_rc[loc_str]
                # else: Goal location is not a valid parsed location, this goal is ignored by the heuristic.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Parse state to find robot and box locations (strings)
        robot_loc_str = None
        box_locations_str = {} # box_name -> location_string

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc_str = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'): # Assuming box names start with 'box'
                 box_locations_str[parts[1]] = parts[2]

        # Ensure robot location is found and is a known location (mappable to RC)
        if robot_loc_str is None or robot_loc_str not in self.loc_str_to_rc:
             # This state is likely invalid or unreachable in a standard Sokoban problem
             # Return infinity as a high cost
             return math.inf

        # 2. Convert locations to RC tuples
        robot_loc_rc = self.loc_str_to_rc[robot_loc_str]
        box_locations_rc = {}
        for box_str, loc_str in box_locations_str.items():
             # Only include boxes whose locations can be mapped to RC
             if loc_str in self.loc_str_to_rc:
                  box_locations_rc[box_str] = self.loc_str_to_rc[loc_str]
             # else: Box is at an unknown location format, heuristic ignores this box.

        # 3. Initialize total heuristic
        total_heuristic = 0

        # 4. Iterate through boxes that have defined goals (and whose goals are mappable to RC)
        for box_str, goal_loc_rc in self.goal_locations_rc.items():
            # Ensure the box exists in the current state and its location is mappable to RC
            if box_str not in box_locations_rc:
                 # Box is missing from state or its location is unmappable.
                 # Treat as unreachable goal for this box.
                 total_heuristic += math.inf
                 continue

            current_loc_rc = box_locations_rc[box_str]

            # 6. If box is not at goal
            if current_loc_rc != goal_loc_rc:
                # 7a. Compute box distance to goal
                # Check if current_loc_rc is a valid start node for shortest paths
                if current_loc_rc not in self.shortest_paths:
                     # This location was not part of the graph built from adjacencies/facts
                     return math.inf # Cannot calculate distance

                box_dist = self.shortest_paths[current_loc_rc].get(goal_loc_rc, math.inf)

                if box_dist == math.inf:
                    return math.inf # Goal is unreachable from box location

                # 7b. Identify first step location on a shortest path
                # This should exist and be different from current_loc_rc if box_dist > 0
                first_step_rc = self.first_step[current_loc_rc][goal_loc_rc]

                # 7c. Determine push direction for the first step
                # Ensure the first step is a direct neighbor in the built graph
                if first_step_rc not in self.adj_dir_rc.get(current_loc_rc, {}):
                     # This implies an issue with pathfinding or graph structure
                     # Treat as unsolvable for safety
                     return math.inf

                push_direction = self.adj_dir_rc[current_loc_rc][first_step_rc]

                if push_direction not in self.opp_dir:
                     # Unknown direction? Should not happen with standard Sokoban directions
                     return math.inf

                robot_push_spot_dir = self.opp_dir[push_direction]

                # 7e. Find potential push spots for the robot
                valid_push_spots_rc = []
                # Iterate through neighbors of the box's current location in the built graph
                if current_loc_rc in self.adj_list_rc:
                    for l_adj_rc in self.adj_list_rc[current_loc_rc]:
                        # Check if the direction from box to adjacent location matches the required robot push spot direction
                        if l_adj_rc in self.adj_dir_rc.get(current_loc_rc, {}) and \
                           self.adj_dir_rc[current_loc_rc][l_adj_rc] == robot_push_spot_dir:
                             valid_push_spots_rc.append(l_adj_rc)

                # 7f. Compute minimum robot distance to a push spot
                if not valid_push_spots_rc:
                     # No location adjacent to the box allows pushing in the required direction
                     # This box is likely stuck relative to its goal path
                     return math.inf

                min_robot_dist = math.inf
                # Check if robot_loc_rc is a valid start node for shortest paths
                if robot_loc_rc in self.shortest_paths:
                    for push_spot_rc in valid_push_spots_rc:
                        dist_robot_to_spot = self.shortest_paths[robot_loc_rc].get(push_spot_rc, math.inf)
                        min_robot_dist = min(min_robot_dist, dist_robot_to_spot)

                if min_robot_dist == math.inf:
                     # Robot cannot reach any valid push spot for this box
                     return math.inf

                # 7g. Calculate cost for this box
                # Cost = Robot moves to first push spot + Total pushes required
                cost_for_box = min_robot_dist + box_dist

                # 7h. Add to total heuristic
                total_heuristic += cost_for_box

        # 8. Return total heuristic
        return total_heuristic
