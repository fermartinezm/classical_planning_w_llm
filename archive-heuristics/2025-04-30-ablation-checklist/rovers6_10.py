from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check: zip stops when the shortest iterable is exhausted.
    # This is sufficient for matching predicate names and arguments.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


def get_shortest_path_distances_cached(start_wp, rover_name, navigation_graph, cache):
    """
    Performs BFS from start_wp for a specific rover.
    Returns a dictionary mapping reachable waypoints to their distance.
    Handles cases where start_wp might be isolated for this rover.
    Uses caching within the current state evaluation.
    """
    cache_key = (start_wp, rover_name)
    if cache_key in cache:
        return cache[cache_key]

    # Ensure the rover exists in the graph structure
    if rover_name not in navigation_graph:
         cache[cache_key] = {}
         return {} # Rover cannot move at all

    graph = navigation_graph[rover_name]

    distances = {}
    queue = deque()

    # Start BFS from the start_wp
    distances[start_wp] = 0
    queue.append(start_wp)

    # Use a set of visited nodes to avoid cycles and redundant processing
    visited = {start_wp}

    while queue:
        current_wp = queue.popleft()
        current_dist = distances[current_wp]

        # Only process neighbors if current_wp has outgoing edges in the graph
        if current_wp in graph:
            for neighbor_wp in graph[current_wp]:
                if neighbor_wp not in visited:
                    visited.add(neighbor_wp)
                    distances[neighbor_wp] = current_dist + 1
                    queue.append(neighbor_wp)

    cache[cache_key] = distances
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing up the estimated costs
    for each unachieved goal fact. The cost for each fact is estimated
    based on the minimum actions and navigation required for the cheapest
    available rover to achieve that specific fact and communicate it.
    Navigation cost is estimated using shortest path (BFS) on the
    rover-specific traverse graph.

    This heuristic is non-admissible as it sums costs for potentially
    interdependent goals and simplifies action sequences.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Pre-process Static Facts ---
        self.lander_location = None
        self.comm_waypoint_candidates = set() # Waypoints visible from lander
        self.rover_capabilities = {} # rover -> set(capabilities)
        self.store_of_rover = {} # store -> rover
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_modes = {} # camera -> set(mode)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_image_wps = {} # objective -> set(waypoint)
        self.rover_navigation_graph = {} # rover -> dict[waypoint -> set(neighbor_waypoint)]

        # Collect all objects by type (useful for iterating rovers, cameras, etc.)
        all_rovers = set()
        all_waypoints = set()
        all_stores = set()
        all_cameras = set()
        all_modes = set()
        all_landers = set()
        all_objectives = set()

        # First pass to identify objects and basic relations from static facts
        for fact in static_facts:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate == 'at_lander':
                 self.lander_location = parts[2]
                 all_landers.add(parts[1])
                 all_waypoints.add(parts[2])
             elif predicate == 'can_traverse':
                 rover, wp1, wp2 = parts[1:]
                 all_rovers.add(rover)
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
                 if rover not in self.rover_navigation_graph:
                     self.rover_navigation_graph[rover] = {}
                 if wp1 not in self.rover_navigation_graph[rover]:
                     self.rover_navigation_graph[rover][wp1] = set()
                 self.rover_navigation_graph[rover][wp1].add(wp2)
             elif predicate.startswith('equipped_for_'):
                 rover = parts[1]
                 all_rovers.add(rover)
                 cap = predicate.split('_')[-2] # 'soil', 'rock', 'imaging'
                 if rover not in self.rover_capabilities: self.rover_capabilities[rover] = set()
                 self.rover_capabilities[rover].add(cap)
             elif predicate == 'store_of':
                 store, rover = parts[1:]
                 all_stores.add(store)
                 all_rovers.add(rover)
                 self.store_of_rover[store] = rover
             elif predicate == 'on_board':
                 camera, rover = parts[1:]
                 all_cameras.add(camera)
                 all_rovers.add(rover)
                 if rover not in self.rover_cameras: self.rover_cameras[rover] = set()
                 self.rover_cameras[rover].add(camera)
             elif predicate == 'supports':
                 camera, mode = parts[1:]
                 all_cameras.add(camera)
                 all_modes.add(mode)
                 if camera not in self.camera_modes: self.camera_modes[camera] = set()
                 self.camera_modes[camera].add(mode)
             elif predicate == 'calibration_target':
                 camera, objective = parts[1:]
                 all_cameras.add(camera)
                 all_objectives.add(objective)
                 self.camera_calibration_target[camera] = objective
             elif predicate == 'visible_from':
                 objective, waypoint = parts[1:]
                 all_objectives.add(objective)
                 all_waypoints.add(waypoint)
                 if objective not in self.objective_image_wps: self.objective_image_wps[objective] = set()
                 self.objective_image_wps[objective].add(waypoint)
             elif predicate == 'visible':
                 wp1, wp2 = parts[1:]
                 all_waypoints.add(wp1)
                 all_waypoints.add(wp2)
                 # This predicate is used for navigation (can_traverse) and communication
                 # We need it to find communication points relative to the lander
                 # Process comm points now if lander location is known
                 if self.lander_location and wp2 == self.lander_location:
                     self.comm_waypoint_candidates.add(wp1)

        # Add all waypoints mentioned in can_traverse to the graph keys, even if no outgoing edges
        # This ensures BFS can be called starting from any waypoint a rover *can* be at.
        for rover, graph in self.rover_navigation_graph.items():
             all_wps_for_rover = set(graph.keys()) | set(wp for neighbors in graph.values() for wp in neighbors)
             for wp in all_wps_for_rover:
                 if wp not in graph:
                     graph[wp] = set() # Add waypoint with no outgoing edges

        # If lander location wasn't found yet (e.g., not in static visible facts), find it in initial state
        # Lander location is typically in the initial state.
        if not self.lander_location:
             for fact in task.initial_state:
                 if match(fact, "at_lander", "*", "*"):
                     self.lander_location = get_parts(fact)[2]
                     all_landers.add(get_parts(fact)[1])
                     all_waypoints.add(self.lander_location)
                     break

        # If lander location is found, identify communication waypoints using 'visible' from static facts
        # This step is repeated in case lander_location was found in initial state on the second pass.
        if self.lander_location:
             for fact in static_facts:
                 if match(fact, "visible", "*", self.lander_location):
                     self.comm_waypoint_candidates.add(get_parts(fact)[1])


        # Map calibration targets back to cameras (needed for cal_wps lookup by camera)
        self.calibration_target_to_camera = {v: k for k, v in self.camera_calibration_target.items()}
        # Map calibration targets to their visible waypoints
        # calibration_target_cal_wps is just objective_image_wps restricted to calibration targets
        self.calibration_target_cal_wps = {
             obj: wps for obj, wps in self.objective_image_wps.items()
             if obj in self.calibration_target_to_camera # Check if this objective is a calibration target
        }

        # Store object lists for easy iteration
        self.all_rovers = list(all_rovers)
        self.all_waypoints = list(all_waypoints)
        self.all_stores = list(all_stores)
        self.all_cameras = list(all_cameras)
        self.all_modes = list(all_modes)
        self.all_landers = list(all_landers)
        self.all_objectives = list(all_objectives)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Cache for BFS results within this state evaluation
        bfs_cache = {}

        # --- Parse Dynamic Facts from State ---
        rover_locations = {} # rover -> waypoint
        soil_samples_at = set() # waypoint
        rock_samples_at = set() # waypoint
        rover_soil_samples = {} # rover -> set(waypoint)
        rover_rock_samples = {} # rover -> set(waypoint)
        store_status = {} # store -> 'empty' or 'full'
        camera_calibrated = set() # (camera, rover)
        rover_images = {} # rover -> set((objective, mode))
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at' and parts[1] in self.rover_navigation_graph: # Check if it's a rover
                rover_locations[parts[1]] = parts[2]
            elif predicate == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif predicate == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif predicate == 'have_soil_analysis':
                rover, waypoint = parts[1:]
                if rover not in rover_soil_samples: rover_soil_samples[rover] = set()
                rover_soil_samples[rover].add(waypoint)
            elif predicate == 'have_rock_analysis':
                rover, waypoint = parts[1:]
                if rover not in rover_rock_samples: rover_rock_samples[rover] = set()
                rover_rock_samples[rover].add(waypoint)
            elif predicate == 'empty':
                store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                store_status[parts[1]] = 'full'
            elif predicate == 'calibrated':
                camera, rover = parts[1:]
                camera_calibrated.add((camera, rover))
            elif predicate == 'have_image':
                rover, objective, mode = parts[1:]
                if rover not in rover_images: rover_images[rover] = set()
                rover_images[rover].add((objective, mode))
            elif predicate == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))

        total_heuristic_cost = 0
        IMPOSSIBLE_COST = 1000 # Penalty for seemingly impossible goals

        # Pre-compute BFS distances from all rover current locations once per state
        rover_current_distances = {}
        for rover, current_pos in rover_locations.items():
             rover_current_distances[rover] = get_shortest_path_distances_cached(current_pos, rover, self.rover_navigation_graph, bfs_cache)


        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            goal_type = parts[0]

            if goal_type == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint in communicated_soil:
                    continue # Goal already achieved

                min_goal_cost = IMPOSSIBLE_COST

                # Find best rover for this soil goal
                for rover in self.all_rovers: # Iterate all rovers
                    if 'soil' not in self.rover_capabilities.get(rover, set()): continue # Must be equipped
                    if rover not in rover_locations: continue # Rover must exist and have a location

                    current_pos_r = rover_locations[rover]
                    dist_r = rover_current_distances.get(rover, {}) # Use pre-computed distances from current pos

                    # Cost if rover already has the sample
                    if waypoint in rover_soil_samples.get(rover, set()):
                        # Need to reach a communication point
                        cost_if_have_sample = min((dist_r.get(cp, IMPOSSIBLE_COST) for cp in self.comm_waypoint_candidates), default=IMPOSSIBLE_COST)
                        if cost_if_have_sample < IMPOSSIBLE_COST:
                             cost_if_have_sample += 1 # communicate action
                             min_goal_cost = min(min_goal_cost, cost_if_have_sample)

                    # Cost if rover needs to sample
                    elif waypoint in soil_samples_at: # Sample is available at the waypoint
                        dist_r_to_w = dist_r.get(waypoint, IMPOSSIBLE_COST)
                        if dist_r_to_w < IMPOSSIBLE_COST:
                            cost_to_sample = dist_r_to_w + 1 # sample action
                            # Check store status
                            rover_store = next((s for s, owner in self.store_of_rover.items() if owner == rover), None)
                            if rover_store and store_status.get(rover_store) == 'full':
                                cost_to_sample += 1 # drop action

                            # Need to reach a communication point from the sample point
                            dist_w = get_shortest_path_distances_cached(waypoint, rover, self.rover_navigation_graph, bfs_cache)
                            min_dist_w_to_comm = min((dist_w.get(cp, IMPOSSIBLE_COST) for cp in self.comm_waypoint_candidates), default=IMPOSSIBLE_COST)

                            if min_dist_w_to_comm < IMPOSSIBLE_COST:
                                # Path: current -> sample_wp -> comm_wp
                                # Nav cost: dist(current, sample_wp) + dist(sample_wp, comm_wp)
                                nav_cost = dist_r_to_w + min_dist_w_to_comm
                                action_cost = (1 if (rover_store and store_status.get(rover_store) == 'full') else 0) + 1 + 1 # drop + sample + communicate
                                total_cost_for_this_rover = nav_cost + action_cost
                                min_goal_cost = min(min_goal_cost, total_cost_for_this_rover)

                total_heuristic_cost += min_goal_cost # Add the minimum cost found for this goal

            elif goal_type == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint in communicated_rock:
                    continue # Goal already achieved

                min_goal_cost = IMPOSSIBLE_COST

                # Find best rover for this rock goal
                for rover in self.all_rovers: # Iterate all rovers
                    if 'rock' not in self.rover_capabilities.get(rover, set()): continue # Must be equipped
                    if rover not in rover_locations: continue

                    current_pos_r = rover_locations[rover]
                    dist_r = rover_current_distances.get(rover, {}) # Use pre-computed distances

                    # Cost if rover already has the sample
                    if waypoint in rover_rock_samples.get(rover, set()):
                        cost_if_have_sample = min((dist_r.get(cp, IMPOSSIBLE_COST) for cp in self.comm_waypoint_candidates), default=IMPOSSIBLE_COST)
                        if cost_if_have_sample < IMPOSSIBLE_COST:
                             cost_if_have_sample += 1 # communicate action
                             min_goal_cost = min(min_goal_cost, cost_if_have_sample)

                    # Cost if rover needs to sample
                    elif waypoint in rock_samples_at: # Sample is available at the waypoint
                        dist_r_to_w = dist_r.get(waypoint, IMPOSSIBLE_COST)
                        if dist_r_to_w < IMPOSSIBLE_COST:
                            cost_to_sample = dist_r_to_w + 1 # sample action
                            # Check store status
                            rover_store = next((s for s, owner in self.store_of_rover.items() if owner == rover), None)
                            if rover_store and store_status.get(rover_store) == 'full':
                                cost_to_sample += 1 # drop action

                            # Need to reach a communication point from the sample point
                            dist_w = get_shortest_path_distances_cached(waypoint, rover, self.rover_navigation_graph, bfs_cache)
                            min_dist_w_to_comm = min((dist_w.get(cp, IMPOSSIBLE_COST) for cp in self.comm_waypoint_candidates), default=IMPOSSIBLE_COST)

                            if min_dist_w_to_comm < IMPOSSIBLE_COST:
                                # Path: current -> sample_wp -> comm_wp
                                # Nav cost: dist(current, sample_wp) + dist(sample_wp, comm_wp)
                                nav_cost = dist_r_to_w + min_dist_w_to_comm
                                action_cost = (1 if (rover_store and store_status.get(rover_store) == 'full') else 0) + 1 + 1 # drop + sample + communicate
                                total_cost_for_this_rover = nav_cost + action_cost
                                min_goal_cost = min(min_goal_cost, total_cost_for_this_rover)

                total_heuristic_cost += min_goal_cost # Add the minimum cost found for this goal

            elif goal_type == 'communicated_image_data':
                objective, mode = parts[1:]
                if (objective, mode) in communicated_image:
                    continue # Goal already achieved

                min_goal_cost = IMPOSSIBLE_COST

                # Find best rover/camera for this image goal
                for rover in self.all_rovers: # Iterate all rovers
                    if 'imaging' not in self.rover_capabilities.get(rover, set()): continue # Must be equipped
                    if rover not in rover_locations: continue

                    current_pos_r = rover_locations[rover]
                    dist_r = rover_current_distances.get(rover, {}) # Use pre-computed distances from current pos

                    # Find cameras on this rover supporting the mode
                    suitable_cameras = [
                        cam for cam in self.rover_cameras.get(rover, set())
                        if mode in self.camera_modes.get(cam, set())
                    ]

                    for camera in suitable_cameras:
                        # Cost if rover already has the image
                        if (objective, mode) in rover_images.get(rover, set()):
                            cost_if_have_image = min((dist_r.get(cp, IMPOSSIBLE_COST) for cp in self.comm_waypoint_candidates), default=IMPOSSIBLE_COST)
                            if cost_if_have_image < IMPOSSIBLE_COST:
                                cost_if_have_image += 1 # communicate action
                                min_goal_cost = min(min_goal_cost, cost_if_have_image)

                        # Cost if rover needs to take the image
                        else:
                            image_wps_o = self.objective_image_wps.get(objective, set())
                            if not image_wps_o: continue # No waypoint to view objective

                            # Find closest image waypoint from current pos
                            min_dist_r_to_image_wp = min((dist_r.get(iw, IMPOSSIBLE_COST) for iw in image_wps_o), default=IMPOSSIBLE_COST)
                            if min_dist_r_to_image_wp == IMPOSSIBLE_COST: continue # Cannot reach any image point
                            best_image_wp = min(image_wps_o, key=lambda wp: dist_r.get(wp, IMPOSSIBLE_COST))

                            # Need calibration?
                            if (camera, rover) not in camera_calibrated:
                                cal_target_i = self.camera_calibration_target.get(camera)
                                if not cal_target_i: continue # Camera has no calibration target
                                cal_wps_i = self.calibration_target_cal_wps.get(cal_target_i, set())
                                if not cal_wps_i: continue # No waypoint to view calibration target

                                # Find closest calibration waypoint from current pos
                                min_dist_r_to_cal_wp = min((dist_r.get(cw, IMPOSSIBLE_COST) for cw in cal_wps_i), default=IMPOSSIBLE_COST)
                                if min_dist_r_to_cal_wp == IMPOSSIBLE_COST: continue # Cannot reach any cal point
                                best_cal_wp = min(cal_wps_i, key=lambda wp: dist_r.get(wp, IMPOSSIBLE_COST))

                                # Estimate navigation cost for calibrate -> image -> communicate path
                                # current -> best_cal_wp -> best_image_wp -> best_comm_wp
                                dist_cal = get_shortest_path_distances_cached(best_cal_wp, rover, self.rover_navigation_graph, bfs_cache)
                                dist_image = get_shortest_path_distances_cached(best_image_wp, rover, self.rover_navigation_graph, bfs_cache)

                                # Find closest comm wp from best_image_wp
                                min_dist_image_to_comm = min((dist_image.get(cp, IMPOSSIBLE_COST) for cp in self.comm_waypoint_candidates), default=IMPOSSIBLE_COST)
                                if min_dist_image_to_comm == IMPOSSIBLE_COST: continue # Cannot reach comm point from image point

                                nav_cost = dist_r.get(best_cal_wp, IMPOSSIBLE_COST) + \
                                           dist_cal.get(best_image_wp, IMPOSSIBLE_COST) + \
                                           min_dist_image_to_comm

                                if nav_cost >= IMPOSSIBLE_COST: continue # Path impossible

                                total_cost_for_this_rover_camera = nav_cost + 1 # calibrate + 1 # take_image + 1 # communicate
                                min_goal_cost = min(min_goal_cost, total_cost_for_this_rover_camera)

                            else: # Camera is calibrated
                                # Estimate navigation cost for image -> communicate path
                                # current -> best_image_wp -> best_comm_wp
                                dist_image = get_shortest_path_distances_cached(best_image_wp, rover, self.rover_navigation_graph, bfs_cache)

                                # Find closest comm wp from best_image_wp
                                min_dist_image_to_comm = min((dist_image.get(cp, IMPOSSIBLE_COST) for cp in self.comm_waypoint_candidates), default=IMPOSSIBLE_COST)
                                if min_dist_image_to_comm == IMPOSSIBLE_COST: continue # Cannot reach comm point from image point

                                nav_cost = dist_r.get(best_image_wp, IMPOSSIBLE_COST) + min_dist_image_to_comm

                                if nav_cost >= IMPOSSIBLE_COST: continue # Path impossible

                                total_cost_for_this_rover_camera = nav_cost + 1 # take_image + 1 # communicate
                                min_goal_cost = min(min_goal_cost, total_cost_for_this_rover_camera)

                total_heuristic_cost += min_goal_cost # Add the minimum cost found for this goal


        return total_heuristic_cost

