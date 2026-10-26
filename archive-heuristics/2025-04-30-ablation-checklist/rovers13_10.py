import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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

def bfs(graph, start_node, goal_nodes=None):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    to all reachable nodes, or to the nearest goal node if goal_nodes is provided.

    Args:
        graph: Adjacency list representation of the graph (dict: node -> set of neighbors).
        start_node: The node to start the search from.
        goal_nodes: An optional set of goal nodes. If provided, returns the minimum
                    distance to any node in this set.

    Returns:
        If goal_nodes is None: A dictionary mapping reachable nodes to their distances
                                from the start_node.
        If goal_nodes is not None: The minimum distance to any node in goal_nodes,
                                    or float('inf') if no goal node is reachable.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    
    min_dist_to_goal = float('inf')

    while queue:
        current_node = queue.popleft()
        
        if goal_nodes is not None and current_node in goal_nodes:
             min_dist_to_goal = min(min_dist_to_goal, distances[current_node])
             # If we only need the minimum distance to *any* goal, we can stop early
             # if we find a goal node, as BFS finds shortest paths first.
             # However, the graph might have multiple paths of the same length,
             # and we need to explore all nodes at the current depth.
             # But for just the *minimum* distance, the first time we hit *any*
             # goal node is sufficient if we are guaranteed positive edge weights (which we have).
             # Let's keep it simple and just update min_dist_to_goal whenever a goal is found.

        for neighbor in graph.get(current_node, set()):
            if neighbor not in distances:
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    if goal_nodes is None:
        return distances
    else:
        return min_dist_to_goal

def get_min_distance_from_set_to_set(distances_map, start_nodes, goal_nodes):
    """
    Finds the minimum distance from any node in start_nodes to any node in goal_nodes
    using precomputed distances.

    Args:
        distances_map: Dictionary where distances_map[u][v] is the shortest distance from u to v.
        start_nodes: A set of starting nodes.
        goal_nodes: A set of goal nodes.

    Returns:
        The minimum distance, or float('inf') if no path exists between any pair
        of nodes from start_nodes to goal_nodes.
    """
    min_dist = float('inf')
    
    if not start_nodes or not goal_nodes:
        return float('inf')

    for start_node in start_nodes:
        # Check if start_node exists in the precomputed distances
        if start_node in distances_map:
            for goal_node in goal_nodes:
                 # Check if goal_node is reachable from start_node
                 if goal_node in distances_map[start_node]:
                    min_dist = min(min_dist, distances_map[start_node][goal_node])
    return min_dist


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve each
    unmet goal fact related to communication. It sums up the estimated costs
    for sampling/imaging, calibration (if needed), communication, and
    navigation for each required piece of data.

    # Heuristic Initialization
    - Parses static facts to build the waypoint graph, identify lander locations,
      rover capabilities, camera information, and objective visibility.
    - Precomputes all-pairs shortest paths between waypoints using BFS.
    - Stores the goal communication facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact `(communicated_X_data ...)` that is not yet true in the state:
    1. Add 1 action for the final `communicate_X_data` action.
    2. Check if the required data (`have_soil_analysis`, `have_rock_analysis`, or `have_image`)
       already exists for any capable rover.
    3. If the data does *not* exist:
       - Add 1 action for the `sample_X` or `take_image` action.
       - If taking an image, add 1 action for the `calibrate` action.
       - Estimate navigation cost to get a capable rover from its current location
         to the required sample/image location, and then from there to a waypoint
         visible from the lander. This involves finding the minimum distance
         from any capable rover's current location to the sample/image waypoint(s),
         and the minimum distance from the sample/image waypoint(s) to any
         lander-visible waypoint, using precomputed distances.
    4. If the data *does* exist:
       - Estimate navigation cost to get a rover that possesses the data from its
         current location to a waypoint visible from the lander. This involves
         finding the minimum distance from any rover holding the data to any
         lander-visible waypoint, using precomputed distances.
    5. Sum the costs for all unachieved goal communication facts.
    6. If any required navigation step is impossible (no path exists), the heuristic
       returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Precomputation from Static Facts ---
        self.waypoint_graph = collections.defaultdict(set)
        self.all_waypoints = set()
        self.lander_locs = set()
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_stores = {} # rover -> store
        self.objective_wps = collections.defaultdict(set) # objective -> set of visible waypoints
        self.calibration_targets = {} # camera -> objective (calibration target)
        self.rover_cameras = collections.defaultdict(set) # rover -> set of cameras
        self.camera_modes = collections.defaultdict(set) # camera -> set of supported modes

        # Collect all waypoints and build the navigation graph
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'visible' and len(parts) == 3:
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_graph[wp1].add(wp2)
                self.waypoint_graph[wp2].add(wp1) # Assuming visibility is symmetric for navigation
                self.all_waypoints.add(wp1)
                self.all_waypoints.add(wp2)
            elif parts[0] == 'at_lander' and len(parts) == 3:
                 self.lander_locs.add(parts[2])
                 self.all_waypoints.add(parts[2])
            elif parts[0] == 'can_traverse' and len(parts) == 4:
                 # can_traverse confirms reachability, but visible is sufficient for graph
                 # We rely on visible for graph structure
                 self.all_waypoints.add(parts[2])
                 self.all_waypoints.add(parts[3])
            elif parts[0] == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.equipped_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.equipped_rock.add(parts[1])
            elif parts[0] == 'equipped_for_imaging' and len(parts) == 2:
                self.equipped_imaging.add(parts[1])
            elif parts[0] == 'store_of' and len(parts) == 3:
                self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif parts[0] == 'visible_from' and len(parts) == 3:
                self.objective_wps[parts[1]].add(parts[2]) # objective -> waypoint
                self.all_waypoints.add(parts[2])
            elif parts[0] == 'calibration_target' and len(parts) == 3:
                self.calibration_targets[parts[1]] = parts[2] # camera -> objective
            elif parts[0] == 'on_board' and len(parts) == 3:
                self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
            elif parts[0] == 'supports' and len(parts) == 3:
                self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            # Add waypoints from initial sample locations if they exist in init state (though they are dynamic)
            # The heuristic init only sees static facts, so we get sample locations from goal/problem init later.
            # For graph building, visible/can_traverse/at_lander is enough.

        # Precompute all-pairs shortest paths
        self.distances = {}
        for wp in self.all_waypoints:
             self.distances[wp] = bfs(self.waypoint_graph, wp)

        # Identify waypoints visible from the lander(s)
        self.lander_comm_wps = set()
        for lander_loc in self.lander_locs:
             # A rover at wp can communicate if (visible wp lander_loc)
             # We need waypoints 'wp' such that (visible wp lander_loc) is true
             # The visible_map stores this directly.
             self.lander_comm_wps.update(self.waypoint_graph.get(lander_loc, set()))
             # Also, the lander location itself might be a communication point if a rover can reach it
             # The domain says (visible ?x ?y) where ?x is rover loc, ?y is lander loc.
             # So, any waypoint visible *from* the lander location is a communication point for the rover.
             # We already added neighbors of lander_loc. What about lander_loc itself?
             # If (visible lander_loc lander_loc) is not in PDDL, rover must be *visible from* lander loc.
             # The graph is built using visible, so neighbors of lander_loc are the comm points.

        # Store goal communication facts
        self.goal_communications = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] in ['communicated_soil_data', 'communicated_rock_data', 'communicated_image_data']:
                self.goal_communications.add(goal)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Parse Dynamic Facts from State ---
        rover_locations = {} # rover -> waypoint
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        store_status = {} # store -> 'empty' or 'full'
        camera_calibrated = {} # (camera, rover) -> True

        # Need sample locations from the current state if they haven't been sampled yet
        current_soil_samples = set() # waypoint
        current_rock_samples = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at' and len(parts) == 3:
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis' and len(parts) == 3:
                have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis' and len(parts) == 3:
                have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image' and len(parts) == 4:
                have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'full' and len(parts) == 2:
                store_status[parts[1]] = 'full'
            elif parts[0] == 'empty' and len(parts) == 2:
                 store_status[parts[1]] = 'empty'
            elif parts[0] == 'calibrated' and len(parts) == 3:
                camera_calibrated[(parts[1], parts[2])] = True
            elif parts[0] == 'at_soil_sample' and len(parts) == 2:
                 current_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample' and len(parts) == 2:
                 current_rock_samples.add(parts[1])


        total_cost = 0  # Initialize action cost counter.

        # --- Estimate Cost for Each Unachieved Goal Communication ---
        for goal in self.goal_communications:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            goal_type = parts[0]

            if goal_type == 'communicated_soil_data':
                waypoint = parts[1]
                
                # Cost for communication action
                cost = 1

                # Check if data exists
                data_exists = any((r, waypoint) in have_soil for r in self.equipped_soil)

                if not data_exists:
                    # Need to sample
                    cost += 1 # sample_soil action
                    
                    # Navigation cost: rover_loc -> sample_loc -> comm_loc
                    # Find capable rovers
                    capable_rovers = self.equipped_soil
                    if not capable_rovers: return float('inf') # Cannot sample if no rover is equipped

                    # Nav cost to sample location
                    start_wps = {rover_locations.get(r) for r in capable_rovers if rover_locations.get(r) is not None}
                    if not start_wps: return float('inf') # Capable rovers not located? (Shouldn't happen in valid states)

                    dist_to_sample = get_min_distance_from_set_to_set(self.distances, start_wps, {waypoint})
                    if dist_to_sample == float('inf'): return float('inf') # Sample location unreachable

                    # Nav cost from sample location to communication location
                    dist_sample_to_comm = get_min_distance_from_set_to_set(self.distances, {waypoint}, self.lander_comm_wps)
                    if dist_sample_to_comm == float('inf'): return float('inf') # Communication location unreachable from sample

                    cost += dist_to_sample + dist_sample_to_comm

                    # Consider store constraint? If sampling is needed and the relevant rover's store is full, add drop cost.
                    # This adds complexity: which rover samples? Assume the one with min nav cost.
                    # Let's skip the store constraint for this version for simplicity and speed.

                else:
                    # Data exists, only need to communicate
                    # Navigation cost: rover_loc (of rover with data) -> comm_loc
                    rovers_with_data = {r for r, w in have_soil if w == waypoint}
                    if not rovers_with_data: return float('inf') # Data exists, but no equipped rover has it? (Shouldn't happen)

                    start_wps = {rover_locations.get(r) for r in rovers_with_data if rover_locations.get(r) is not None}
                    if not start_wps: return float('inf') # Rovers with data not located?

                    dist_to_comm = get_min_distance_from_set_to_set(self.distances, start_wps, self.lander_comm_wps)
                    if dist_to_comm == float('inf'): return float('inf') # Communication location unreachable

                    cost += dist_to_comm

                total_cost += cost

            elif goal_type == 'communicated_rock_data':
                waypoint = parts[1]

                # Cost for communication action
                cost = 1

                # Check if data exists
                data_exists = any((r, waypoint) in have_rock for r in self.equipped_rock)

                if not data_exists:
                    # Need to sample
                    cost += 1 # sample_rock action

                    # Navigation cost: rover_loc -> sample_loc -> comm_loc
                    # Find capable rovers
                    capable_rovers = self.equipped_rock
                    if not capable_rovers: return float('inf') # Cannot sample if no rover is equipped

                    # Nav cost to sample location
                    start_wps = {rover_locations.get(r) for r in capable_rovers if rover_locations.get(r) is not None}
                    if not start_wps: return float('inf') # Capable rovers not located?

                    dist_to_sample = get_min_distance_from_set_to_set(self.distances, start_wps, {waypoint})
                    if dist_to_sample == float('inf'): return float('inf') # Sample location unreachable

                    # Nav cost from sample location to communication location
                    dist_sample_to_comm = get_min_distance_from_set_to_set(self.distances, {waypoint}, self.lander_comm_wps)
                    if dist_sample_to_comm == float('inf'): return float('inf') # Communication location unreachable from sample

                    cost += dist_to_sample + dist_sample_to_comm

                    # Store constraint ignored

                else:
                    # Data exists, only need to communicate
                    # Navigation cost: rover_loc (of rover with data) -> comm_loc
                    rovers_with_data = {r for r, w in have_rock if w == waypoint}
                    if not rovers_with_data: return float('inf') # Data exists, but no equipped rover has it?

                    start_wps = {rover_locations.get(r) for r in rovers_with_data if rover_locations.get(r) is not None}
                    if not start_wps: return float('inf') # Rovers with data not located?

                    dist_to_comm = get_min_distance_from_set_to_set(self.distances, start_wps, self.lander_comm_wps)
                    if dist_to_comm == float('inf'): return float('inf') # Communication location unreachable

                    cost += dist_to_comm

                total_cost += cost

            elif goal_type == 'communicated_image_data':
                objective, mode = parts[1], parts[2]

                # Cost for communication action
                cost = 1

                # Check if data exists
                data_exists = any((r, objective, mode) in have_image for r in self.equipped_imaging)

                if not data_exists:
                    # Need to take image and calibrate
                    cost += 1 # take_image action
                    cost += 1 # calibrate action

                    # Navigation cost: rover_loc -> img/cal_loc -> comm_loc
                    # Find capable rovers/cameras
                    capable_configs = [] # list of (rover, camera) tuples
                    for r in self.equipped_imaging:
                        for c in self.rover_cameras.get(r, set()):
                            if mode in self.camera_modes.get(c, set()):
                                capable_configs.append((r, c))

                    if not capable_configs: return float('inf') # Cannot take image if no suitable rover/camera

                    # Find suitable waypoints for imaging AND calibration
                    image_wps = self.objective_wps.get(objective, set())
                    calib_wps = set()
                    for r, c in capable_configs:
                        calib_target = self.calibration_targets.get(c)
                        if calib_target:
                            calib_wps.update(self.objective_wps.get(calib_target, set()))

                    target_wps = image_wps & calib_wps # Need a waypoint visible from both objective and cal target
                    if not target_wps: return float('inf') # No suitable waypoint for imaging/calibration

                    # Nav cost to image/calibration location
                    start_wps = {rover_locations.get(r) for r, c in capable_configs if rover_locations.get(r) is not None}
                    if not start_wps: return float('inf') # Capable rovers not located?

                    dist_to_img_cal = get_min_distance_from_set_to_set(self.distances, start_wps, target_wps)
                    if dist_to_img_cal == float('inf'): return float('inf') # Image/calibration location unreachable

                    # Nav cost from image/calibration location to communication location
                    dist_img_cal_to_comm = get_min_distance_from_set_to_set(self.distances, target_wps, self.lander_comm_wps)
                    if dist_img_cal_to_comm == float('inf'): return float('inf') # Communication location unreachable from image/cal spot

                    cost += dist_to_img_cal + dist_img_cal_to_comm

                else:
                    # Data exists, only need to communicate
                    # Navigation cost: rover_loc (of rover with data) -> comm_loc
                    rovers_with_data = {r for r, o, m in have_image if o == objective and m == mode}
                    if not rovers_with_data: return float('inf') # Data exists, but no equipped rover has it?

                    start_wps = {rover_locations.get(r) for r in rovers_with_data if rover_locations.get(r) is not None}
                    if not start_wps: return float('inf') # Rovers with data not located?

                    dist_to_comm = get_min_distance_from_set_to_set(self.distances, start_wps, self.lander_comm_wps)
                    if dist_to_comm == float('inf'): return float('inf') # Communication location unreachable

                    cost += dist_to_comm

                total_cost += cost

        return total_cost

