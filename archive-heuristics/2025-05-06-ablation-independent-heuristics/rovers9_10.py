import collections
import heapq # Although not strictly needed for this BFS, it's in the example heuristics

# Assume Heuristic base class is available from heuristic_base
# from heuristics.heuristic_base import Heuristic # Need to import this

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding parentheses and split by space
    parts = fact_string[1:-1].split()
    # The first part is the predicate, the rest are arguments
    return tuple(parts)

# Helper function for BFS
def bfs(graph, start_node):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph: # Handle cases where start_node is not in the graph nodes
         return distances # All distances remain infinity

    distances[start_node] = 0
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in graph: # Ensure current_node is a key in the graph dict
            for neighbor in graph.get(current_node, set()): # Use .get for safety
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

# Helper function to compute all-pairs shortest paths
def compute_all_pairs_shortest_paths(graph, nodes):
    """Computes shortest path distances between all pairs of nodes in a graph."""
    all_distances = {}
    # Ensure all potential nodes are in the graph keys, even if they have no edges
    # This makes the BFS function simpler
    full_graph = {node: graph.get(node, set()) for node in nodes}

    for start_node in nodes:
        all_distances[start_node] = bfs(full_graph, start_node)
    return all_distances

# Helper function to extract waypoints
def get_waypoints_from_facts(facts):
    """Extracts all unique waypoint names from a set of PDDL fact strings."""
    waypoints = set()
    # Map predicate names to the 1-based index of their waypoint arguments
    waypoint_arg_indices = {
        'at': [2],
        'at_lander': [2],
        'can_traverse': [2, 3],
        'visible': [1, 2],
        'at_soil_sample': [1],
        'at_rock_sample': [1],
        'visible_from': [2], # visible_from objective waypoint
    }
    for fact_string in facts:
        parsed = parse_fact(fact_string)
        predicate = parsed[0]
        if predicate in waypoint_arg_indices:
            for arg_index_pddl in waypoint_arg_indices[predicate]:
                arg_index_tuple = arg_index_pddl # In the tuple (pred, arg1, arg2...), arg1 is at index 1
                if arg_index_tuple < len(parsed): # Ensure index is valid
                     waypoints.add(parsed[arg_index_tuple])
    return list(waypoints) # Return as list for consistent iteration order

# Need to import the base Heuristic class
from heuristics.heuristic_base import Heuristic


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    costs for each unsatisfied goal fact. For each unsatisfied goal fact, it
    calculates the minimum number of actions required to achieve it, considering
    navigation costs (precomputed shortest paths) and the actions needed to
    collect and communicate data (sampling, imaging, calibrating, dropping).
    It is a relaxation as it ignores negative effects (except for `at_soil_sample`,
    `at_rock_sample`, and `calibrated` which are explicitly handled as being
    consumed or toggled) and assumes different goal facts can be pursued somewhat
    independently by different rovers or sequentially by the same rover without
    complex interactions beyond navigation and basic resource (store, calibration)
    management.

    Assumptions:
    - The heuristic assumes that if a soil/rock sample is required at a waypoint,
      it must have been present in the initial state (`at_soil_sample` or
      `at_rock_sample` in initial_state). These facts are consumed by sampling.
    - The heuristic assumes that if a rover needs to sample but has no empty store,
      it can make one empty by dropping a sample (cost 1), ignoring which sample
      is dropped or if it was needed for another goal. This is only possible if
      the rover has at least one full store.
    - The heuristic calculates the cost to calibrate a camera if it's not currently
      calibrated when an image is needed. It does not explicitly model the fact
      that `take_image` uncalibrates the camera, potentially requiring recalibration
      for subsequent images with the same camera. It calculates the calibration
      cost based on the current state's calibration status.
    - The heuristic assumes that communication is always possible from any waypoint
      visible *from* the lander's location, provided the rover is at that waypoint.
      It considers all landers and communication waypoints to find the minimum cost.
    - The heuristic assumes action costs are 1. Navigation cost is the number of
      `navigate` actions.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes static information from the task
    description and initial state that does not change during planning.
    1. Parses all relevant static facts (`at_lander`, `equipped_for_soil_analysis`,
       `equipped_for_rock_analysis`, `equipped_for_imaging`, `store_of`, `visible`,
       `can_traverse`, `calibration_target`, `on_board`, `supports`, `visible_from`).
    2. Extracts all unique waypoints from the initial state and static facts.
    3. Builds traversal graphs for each rover based on `can_traverse` facts.
    4. Computes all-pairs shortest path distances for each rover on its traversal
       graph using BFS.
    5. Stores initial locations of soil and rock samples.
    6. Stores lander locations and identifies communication waypoints (visible from lander).
    7. Stores rover capabilities, camera information, and objective visibility.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Initialize total heuristic value `h = 0`.
    2. Identify the set of goal facts that are not true in the current state.
    3. For each unsatisfied goal fact `g`:
        a. Calculate the estimated cost `cost(g)` to achieve `g` from the current state.
           - If `g` is `(communicated_soil_data ?w)`:
             - If `(at_soil_sample ?w)` was not in the initial state, `cost(g) = infinity`.
             - Otherwise, find all rovers `r` equipped for soil analysis. If none, `cost(g) = infinity`.
             - For each such rover `r`:
               - Calculate cost to get `(have_soil_analysis r ?w)`:
                 - If `(have_soil_analysis r ?w)` is in state: 0.
                 - Else: Cost is navigation from `r`'s current location to `?w` + (1 if `r` needs to drop a sample) + 1 (sample_soil). If navigation is impossible or no empty/full store exists, this part is infinity.
               - Calculate cost to communicate `(have_soil_analysis r ?w)`:
                 - Cost is minimum over all landers `l` and communication waypoints `x` (visible from `l`'s location): Navigation from `r`'s current location to `x` + 1 (communicate_soil_data). If no communication is possible, this part is infinity.
               - Total cost for rover `r` = Cost to get `(have_soil_analysis r ?w)` + Cost to communicate.
             - `cost(g)` is the minimum total cost over all suitable rovers `r`.
           - If `g` is `(communicated_rock_data ?w)`: Similar logic as soil data.
           - If `g` is `(communicated_image_data ?o ?m)`:
             - Find rovers `r` with cameras `i` that are on board `r` and support mode `m`. If none, `cost(g) = infinity`.
             - Find waypoints `p` from which `o` is visible. If none, `cost(g) = infinity`.
             - Find communication waypoints `x` visible from lander. If none, `cost(g) = infinity`.
             - For each suitable pair `(r, i)`:
               - Calculate cost to get `(have_image r ?o ?m)`:
                 - If `(have_image r ?o ?m)` is in state: 0.
                 - Else: Minimum over waypoints `p` visible from `o`: Navigation from `r`'s current location to `p` + (Cost to calibrate `i` if `(calibrated i r)` is false) + 1 (take_image). If navigation is impossible or calibration is impossible, this part is infinity.
                 - Cost to calibrate `i` if `(calibrated i r)` is false: Minimum over calibration targets `t` for `i` and waypoints `w_cal` visible from `t`: Navigation from `r`'s current location to `w_cal` + 1 (calibrate). If no calibration possible, this cost is infinity.
               - Calculate cost to communicate `(have_image r ?o ?m)`:
                 - Cost is minimum over all landers `l` and communication waypoints `x`: Navigation from `r`'s current location to `x` + 1 (communicate_image_data). If no communication is possible, this part is infinity.
               - Total cost for `(r, i)` = Cost to get `(have_image r ?o ?m)` + Cost to communicate.
             - `cost(g)` is the minimum total cost over all suitable pairs `(r, i)`.
        b. If `cost(g)` is infinity, the total heuristic `h` becomes infinity. Break the loop.
        c. Otherwise, add `cost(g)` to `h`.
    4. Return `h`. If `h` is infinity, return `float('inf')`. If all goals are satisfied, `h` will be 0.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # --- Precomputation ---

        # 1. Parse static facts and initial state facts for relevant info
        self.lander_location = {} # {lander: waypoint}
        self.rover_capabilities = collections.defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False}) # {rover: {'soil': bool, 'rock': bool, 'imaging': bool}}
        self.store_rover_map = {} # {store: rover}
        self.rover_traversal_graphs = collections.defaultdict(dict) # {rover: {wp1: set(wp2, ...)}}
        self.visibility_graph = collections.defaultdict(set) # {wp1: set(wp2, ...)}
        self.camera_info = collections.defaultdict(dict) # {camera: {'on_board': rover, 'supports': set(mode, ...), 'calibration_target': objective}}
        self.objective_visibility = collections.defaultdict(set) # {objective: set(waypoint, ...)}
        self.calibration_target_visibility = collections.defaultdict(set) # {objective: set(waypoint, ...)} # calibration targets are objectives
        self.initial_soil_samples = set() # {waypoint, ...}
        self.initial_rock_samples = set() # {waypoint, ...}

        all_facts = set(self.initial_state) | set(self.static_facts)
        all_waypoints = get_waypoints_from_facts(all_facts)
        self.waypoints = set(all_waypoints) # Store as set for fast lookup
        self.waypoints_list = list(self.waypoints) # Use list for consistent BFS input

        self.rovers = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        for fact_string in all_facts:
            parsed = parse_fact(fact_string)
            predicate = parsed[0]

            if predicate == 'at_lander':
                lander, wp = parsed[1], parsed[2]
                self.lander_location[lander] = wp
                self.landers.add(lander)
            elif predicate == 'equipped_for_soil_analysis':
                rover = parsed[1]
                self.rover_capabilities[rover]['soil'] = True
                self.rovers.add(rover)
            elif predicate == 'equipped_for_rock_analysis':
                rover = parsed[1]
                self.rover_capabilities[rover]['rock'] = True
                self.rovers.add(rover)
            elif predicate == 'equipped_for_imaging':
                rover = parsed[1]
                self.rover_capabilities[rover]['imaging'] = True
                self.rovers.add(rover)
            elif predicate == 'store_of':
                store, rover = parsed[1], parsed[2]
                self.store_rover_map[store] = rover
                self.stores.add(store)
                self.rovers.add(rover)
            elif predicate == 'visible':
                wp1, wp2 = parsed[1], parsed[2]
                if wp1 not in self.visibility_graph: self.visibility_graph[wp1] = set()
                self.visibility_graph[wp1].add(wp2)
            elif predicate == 'can_traverse':
                rover, wp1, wp2 = parsed[1], parsed[2], parsed[3]
                if rover not in self.rover_traversal_graphs: self.rover_traversal_graphs[rover] = {}
                if wp1 not in self.rover_traversal_graphs[rover]: self.rover_traversal_graphs[rover][wp1] = set()
                self.rover_traversal_graphs[rover][wp1].add(wp2)
                self.rovers.add(rover)
            elif predicate == 'calibration_target':
                camera, objective = parsed[1], parsed[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['calibration_target'] = objective
                self.cameras.add(camera)
                self.objectives.add(objective)
            elif predicate == 'on_board':
                camera, rover = parsed[1], parsed[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['on_board'] = rover
                self.cameras.add(camera)
                self.rovers.add(rover)
            elif predicate == 'supports':
                camera, mode = parsed[1], parsed[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'supports' not in self.camera_info[camera]: self.camera_info[camera]['supports'] = set()
                self.camera_info[camera]['supports'].add(mode)
                self.cameras.add(camera)
                self.modes.add(mode)
            elif predicate == 'visible_from':
                objective, wp = parsed[1], parsed[2]
                if objective not in self.objective_visibility: self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(wp)
                # Also store for calibration targets (which are objectives)
                self.calibration_target_visibility[objective].add(wp)
                self.objectives.add(objective)
            elif predicate == 'at_soil_sample':
                wp = parsed[1]
                self.initial_soil_samples.add(wp)
            elif predicate == 'at_rock_sample':
                wp = parsed[1]
                self.initial_rock_samples.add(wp)

        self.rovers = list(self.rovers) # Convert sets to lists for consistent iteration
        self.landers = list(self.landers)
        self.cameras = list(self.cameras)
        self.objectives = list(self.objectives)
        self.modes = list(self.modes)
        self.stores = list(self.stores)


        # 2. Precompute rover navigation distances
        self.rover_distances = {} # {rover: {wp1: {wp2: distance}}}
        for rover in self.rovers:
             # Need to compute distances between all pairs of waypoints the rover *can* traverse
             # The graph should include all waypoints, even if not connected to the rover's graph component
             self.rover_distances[rover] = compute_all_pairs_shortest_paths(
                 self.rover_traversal_graphs.get(rover, {}), self.waypoints_list
             )

        # 3. Identify communication waypoints for each lander
        self.comm_wps_by_lander = collections.defaultdict(set) # {lander: set(waypoint, ...)}
        # Need reverse visibility graph to find wps visible *from* lander wp
        reverse_visibility_graph = collections.defaultdict(set)
        for fact_string in self.static_facts:
            parsed = parse_fact(fact_string)
            if parsed[0] == 'visible':
                wp1, wp2 = parsed[1], parsed[2]
                reverse_visibility_graph[wp2].add(wp1)

        for lander, lander_wp in self.lander_location.items():
            self.comm_wps_by_lander[lander] = reverse_visibility_graph.get(lander_wp, set())

        # Store mapping from rover to its stores
        self.rover_stores = collections.defaultdict(list)
        for store, rover in self.store_rover_map.items():
            self.rover_stores[rover].append(store)


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.
        """
        state = node.state
        h = 0
        unmet_goals = [g for g in self.goals if g not in state]

        if not unmet_goals:
            return 0 # Goal reached

        # Get current rover locations
        rover_locations = {} # {rover: waypoint}
        for fact_string in state:
            parsed = parse_fact(fact_string)
            if parsed[0] == 'at' and parsed[1] in self.rovers:
                rover_locations[parsed[1]] = parsed[2]

        # Check if any rover is not located - should not happen in valid states, but safety check
        # If a rover is not in rover_locations, its current_wp will be None, handled by .get(rover_curr_wp, {})
        # and distances will be inf.

        # Pre-calculate calibration status for cameras in the current state
        camera_calibrated_status = {} # {(camera, rover): bool}
        for camera in self.cameras:
             # Find which rover it's on (static info)
             rover_on_board = self.camera_info.get(camera, {}).get('on_board')
             if rover_on_board:
                 calibrated_fact = f'(calibrated {camera} {rover_on_board})'
                 camera_calibrated_status[(camera, rover_on_board)] = calibrated_fact in state
             # else: Camera not on board any rover? Ignore.


        # Pre-calculate store status for rovers in the current state
        rover_empty_store_available = {} # {rover: bool}
        rover_full_store_available = {} # {rover: bool}
        for rover in self.rovers:
            has_empty_store = False
            has_full_store = False
            for store in self.rover_stores.get(rover, []):
                if f'(empty {store})' in state:
                    has_empty_store = True
                if f'(full {store})' in state:
                    has_full_store = True
                if has_empty_store and has_full_store: break # Optimization
            rover_empty_store_available[rover] = has_empty_store
            rover_full_store_available[rover] = has_full_store


        # Estimate cost for each unmet goal
        for goal_string in unmet_goals:
            parsed_goal = parse_fact(goal_string)
            goal_predicate = parsed_goal[0]

            cost_g = float('inf') # Cost for this specific goal

            if goal_predicate == 'communicated_soil_data':
                # Goal: (communicated_soil_data ?w)
                waypoint_w = parsed_goal[1]

                # Check if soil sample was initially available at ?w
                if waypoint_w not in self.initial_soil_samples:
                    cost_g = float('inf') # Cannot achieve if no sample was there initially
                else:
                    min_rover_cost = float('inf')
                    # Iterate over rovers equipped for soil analysis
                    for rover in self.rovers:
                        if self.rover_capabilities[rover]['soil']:
                            rover_curr_wp = rover_locations.get(rover)
                            if not rover_curr_wp: continue # Rover location unknown? Skip.

                            # Cost to get (have_soil_analysis rover ?w)
                            have_soil_fact = f'(have_soil_analysis {rover} {waypoint_w})'
                            cost_get_soil = 0
                            if have_soil_fact not in state:
                                # Need to sample soil
                                nav_cost_to_sample = self.rover_distances[rover].get(rover_curr_wp, {}).get(waypoint_w, float('inf'))
                                if nav_cost_to_sample == float('inf'):
                                    cost_get_soil = float('inf') # Cannot reach sample location
                                else:
                                    cost_store = 0
                                    # Check if an empty store is available
                                    if not rover_empty_store_available.get(rover, False):
                                        # Need to drop a sample first?
                                        # Simplification: Assume we can always drop *some* full sample if one exists
                                        if rover_full_store_available.get(rover, False):
                                             cost_store = 1 # Cost to drop
                                        else:
                                             # No empty store, and no full store to drop. Cannot sample.
                                             cost_get_soil = float('inf')

                                    if cost_get_soil != float('inf'): # Check if store logic failed
                                        cost_get_soil = nav_cost_to_sample + cost_store + 1 # navigate + (drop) + sample_soil

                            if cost_get_soil == float('inf'):
                                continue # Cannot get the sample with this rover

                            # Cost to communicate (have_soil_analysis rover ?w)
                            min_comm_cost_rover = float('inf')
                            for lander, lander_wp in self.lander_location.items():
                                lander_comm_wps = self.comm_wps_by_lander.get(lander, set())
                                if not lander_comm_wps:
                                    continue # No communication waypoints for this lander

                                min_comm_cost_lander = float('inf')
                                for comm_wp in lander_comm_wps:
                                    nav_cost_to_comm = self.rover_distances[rover].get(rover_curr_wp, {}).get(comm_wp, float('inf'))
                                    if nav_cost_to_comm != float('inf'):
                                        min_comm_cost_lander = min(min_comm_cost_lander, nav_cost_to_comm + 1) # navigate + communicate

                                min_comm_cost_rover = min(min_comm_cost_rover, min_comm_cost_lander)


                            if min_comm_cost_rover == float('inf'):
                                continue # Cannot communicate with this rover

                            # Total cost for this rover for this goal
                            total_rover_cost = cost_get_soil + min_comm_cost_rover
                            min_rover_cost = min(min_rover_cost, total_rover_cost)

                    cost_g = min_rover_cost # Minimum cost over all suitable rovers

            elif goal_predicate == 'communicated_rock_data':
                # Goal: (communicated_rock_data ?w)
                waypoint_w = parsed_goal[1]

                # Check if rock sample was initially available at ?w
                if waypoint_w not in self.initial_rock_samples:
                    cost_g = float('inf') # Cannot achieve if no sample was there initially
                else:
                    min_rover_cost = float('inf')
                    # Iterate over rovers equipped for rock analysis
                    for rover in self.rovers:
                        if self.rover_capabilities[rover]['rock']:
                            rover_curr_wp = rover_locations.get(rover)
                            if not rover_curr_wp: continue # Rover location unknown? Skip.

                            # Cost to get (have_rock_analysis rover ?w)
                            have_rock_fact = f'(have_rock_analysis {rover} {waypoint_w})'
                            cost_get_rock = 0
                            if have_rock_fact not in state:
                                # Need to sample rock
                                nav_cost_to_sample = self.rover_distances[rover].get(rover_curr_wp, {}).get(waypoint_w, float('inf'))
                                if nav_cost_to_sample == float('inf'):
                                    cost_get_rock = float('inf') # Cannot reach sample location
                                else:
                                    cost_store = 0
                                    # Check if an empty store is available
                                    if not rover_empty_store_available.get(rover, False):
                                        # Need to drop a sample first?
                                        # Simplification: Assume we can always drop *some* full sample if one exists
                                        if rover_full_store_available.get(rover, False):
                                             cost_store = 1 # Cost to drop
                                        else:
                                             # No empty store, and no full store to drop. Cannot sample.
                                             cost_get_rock = float('inf')

                                    if cost_get_rock != float('inf'): # Check if store logic failed
                                        cost_get_rock = nav_cost_to_sample + cost_store + 1 # navigate + (drop) + sample_rock

                            if cost_get_rock == float('inf'):
                                continue # Cannot get the sample with this rover

                            # Cost to communicate (have_rock_analysis rover ?w)
                            min_comm_cost_rover = float('inf')
                            for lander, lander_wp in self.lander_location.items():
                                lander_comm_wps = self.comm_wps_by_lander.get(lander, set())
                                if not lander_comm_wps:
                                    continue # No communication waypoints for this lander

                                min_comm_cost_lander = float('inf')
                                for comm_wp in lander_comm_wps:
                                    nav_cost_to_comm = self.rover_distances[rover].get(rover_curr_wp, {}).get(comm_wp, float('inf'))
                                    if nav_cost_to_comm != float('inf'):
                                        min_comm_cost_lander = min(min_comm_cost_lander, nav_cost_to_comm + 1) # navigate + communicate

                                min_comm_cost_rover = min(min_comm_cost_rover, min_comm_cost_lander)

                            if min_comm_cost_rover == float('inf'):
                                continue # Cannot communicate with this rover

                            # Total cost for this rover for this goal
                            total_rover_cost = cost_get_rock + min_comm_cost_rover
                            min_rover_cost = min(min_rover_cost, total_rover_cost)

                    cost_g = min_rover_cost # Minimum cost over all suitable rovers


            elif goal_predicate == 'communicated_image_data':
                # Goal: (communicated_image_data ?o ?m)
                objective_o, mode_m = parsed_goal[1], parsed_goal[2]

                min_rover_camera_cost = float('inf')

                # Iterate over rovers equipped for imaging with suitable cameras
                for rover in self.rovers:
                    if self.rover_capabilities[rover]['imaging']:
                        rover_curr_wp = rover_locations.get(rover)
                        if not rover_curr_wp: continue # Rover location unknown? Skip.

                        for camera in self.cameras:
                            cam_info = self.camera_info.get(camera, {})
                            if cam_info.get('on_board') == rover and mode_m in cam_info.get('supports', set()):

                                # Cost to get (have_image rover ?o ?m)
                                have_image_fact = f'(have_image {rover} {objective_o} {mode_m})'
                                cost_get_image = 0
                                if have_image_fact not in state:
                                    # Need to take image
                                    image_wps = self.objective_visibility.get(objective_o, set())
                                    if not image_wps:
                                        cost_get_image = float('inf') # Cannot see objective from anywhere
                                    else:
                                        min_take_image_cost = float('inf')
                                        for image_wp in image_wps:
                                            nav_cost_to_image = self.rover_distances[rover].get(rover_curr_wp, {}).get(image_wp, float('inf'))
                                            if nav_cost_to_image == float('inf'):
                                                continue # Cannot reach image location

                                            # Cost to calibrate camera
                                            cost_calibrate = 0
                                            if not camera_calibrated_status.get((camera, rover), False):
                                                # Need to calibrate
                                                cal_target = cam_info.get('calibration_target')
                                                if not cal_target:
                                                    cost_calibrate = float('inf') # No calibration target for this camera
                                                else:
                                                    cal_wps = self.calibration_target_visibility.get(cal_target, set())
                                                    if not cal_wps:
                                                        cost_calibrate = float('inf') # Cannot see calibration target from anywhere
                                                    else:
                                                        min_cal_nav_cost = float('inf')
                                                        for cal_wp in cal_wps:
                                                            nav_cost_to_cal = self.rover_distances[rover].get(rover_curr_wp, {}).get(cal_wp, float('inf'))
                                                            if nav_cost_to_cal != float('inf'):
                                                                min_cal_nav_cost = min(min_cal_nav_cost, nav_cost_to_cal)

                                                        if min_cal_nav_cost == float('inf'):
                                                            cost_calibrate = float('inf') # Cannot reach any calibration waypoint
                                                        else:
                                                            cost_calibrate = min_cal_nav_cost + 1 # navigate + calibrate

                                            if cost_calibrate == float('inf'):
                                                continue # Cannot calibrate camera

                                            # Total cost for this take_image action
                                            current_take_image_cost = nav_cost_to_image + cost_calibrate + 1 # navigate + (calibrate) + take_image
                                            min_take_image_cost = min(min_take_image_cost, current_take_image_cost)

                                        cost_get_image = min_take_image_cost # Minimum cost over all image waypoints

                                if cost_get_image == float('inf'):
                                    continue # Cannot get the image with this rover/camera

                                # Cost to communicate (have_image rover ?o ?m)
                                min_comm_cost_rover = float('inf')
                                for lander, lander_wp in self.lander_location.items():
                                    lander_comm_wps = self.comm_wps_by_lander.get(lander, set())
                                    if not lander_comm_wps:
                                        continue # No communication waypoints for this lander

                                    min_comm_cost_lander = float('inf')
                                    for comm_wp in lander_comm_wps:
                                        nav_cost_to_comm = self.rover_distances[rover].get(rover_curr_wp, {}).get(comm_wp, float('inf'))
                                        if nav_cost_to_comm != float('inf'):
                                            min_comm_cost_lander = min(min_comm_cost_lander, nav_cost_to_comm + 1) # navigate + communicate

                                    min_comm_cost_rover = min(min_comm_cost_rover, min_comm_cost_lander)


                                if min_comm_cost_rover == float('inf'):
                                    continue # Cannot communicate with this rover

                                # Total cost for this rover/camera for this goal
                                total_rc_cost = cost_get_image + min_comm_cost_rover
                                min_rover_camera_cost = min(min_rover_camera_cost, total_rc_cost)

                cost_g = min_rover_camera_cost # Minimum cost over all suitable rovers/cameras


            # Add cost of this goal to total heuristic
            if cost_g == float('inf'):
                return float('inf') # If any goal is impossible, the state is a dead end (or leads to one)
            h += cost_g

        return h
