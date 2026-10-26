import math
from collections import deque

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each unachieved goal fact. For each unachieved
    goal fact (communicating soil, rock, or image data), it calculates the
    minimum cost required to achieve that specific fact, considering the
    capabilities of different rovers and the necessary steps (navigation,
    sampling/imaging, communication). Navigation costs are estimated using
    precomputed shortest path distances between waypoints for each rover.

    Assumptions:
    - The planning problem is solvable. This implies that all necessary
      waypoints (sample locations, visible_from points for objectives/targets,
      communication points) exist and are reachable by at least one
      capable rover.
    - The static facts provided in the task object are complete and accurate.
    - The internal state representation uses strings for facts, e.g.,
      '(predicate arg1 arg2)'.

    Heuristic Initialization:
    The __init__ method precomputes static information from the task object
    to enable efficient heuristic calculation for each state.
    1. Parses static facts to identify:
       - Lander location(s).
       - Rover capabilities (soil, rock, imaging).
       - Rover-store mapping.
       - Locations of soil and rock samples.
       - Waypoint visibility graph.
       - Rover traversal graphs.
       - Camera information (on-board rover, supported modes, calibration target).
       - Objective visibility information (waypoints visible from objective).
    2. Identifies communication points (waypoints visible from any lander location).
    3. For each rover, computes all-pairs shortest path distances between waypoints
       based on its traversal graph using BFS.
    4. For each rover, objective, and waypoint, precomputes the minimum distance
       from that waypoint to any waypoint visible from the objective, and stores
       the best waypoint to navigate to.
    5. For each rover, camera, objective, and waypoint, precomputes the minimum
       navigation cost sequence: from the current waypoint to a calibration
       waypoint for the camera, and then from the calibration waypoint to an
       image waypoint for the objective.

    Step-By-Step Thinking for Computing Heuristic:
    The __call__ method computes the heuristic value for a given state.
    1. Initialize the total heuristic value `h` to 0.
    2. Get the set of goal facts from the task.
    3. For each goal fact `g` in the set of goals:
        a. If `g` is already present in the current `state`, continue to the next goal.
        b. If `g` is not in the state, calculate the minimum estimated cost `cost_g`
           to achieve `g`. This involves considering all rovers capable of
           contributing to this goal.
        c. Add `cost_g` to the total heuristic `h`.
    4. Return `h`.

    Calculating `cost_g` for an unachieved goal `g`:

    - If `g` is `(communicated_soil_data ?w)`:
        - Find all rovers `?r` equipped for soil analysis and having a store.
        - The soil sample is at waypoint `?w`.
        - Find communication points `X_comm`.
        - `cost_g` is the minimum over all suitable rovers `?r`:
            - Get `?r`'s current location `?r_loc`.
            - Cost to get sample: `dist(?r_loc, ?w) + (1 if ?r`'s store is full) + 1 (sample_soil action).
            - Location after sampling is `?w`.
            - Cost to communicate: `min_{x \in X_comm} dist(?w, x) + 1 (communicate_soil_data action).
            - Total cost for rover `?r`: Cost to get sample + Cost to communicate.

    - If `g` is `(communicated_rock_data ?w)`:
        - Similar logic as for soil data, using rock-equipped rovers, rock sample locations, and communicate_rock_data action.

    - If `g` is `(communicated_image_data ?o ?m)`:
        - Find all rovers `?r` equipped for imaging, having a camera `?i` on board that supports mode `?m`.
        - Find waypoints `P_image` visible from objective `?o`.
        - Find calibration target `?t_cal` for camera `?i`.
        - Find waypoints `W_cal` visible from `?t_cal`.
        - Find communication points `X_comm`.
        - `cost_g` is the minimum over all suitable rover-camera pairs `(?r, ?i)`:
            - Get `?r`'s current location `?r_loc`.
            - Cost to get image `(have_image ?r ?o ?m)`:
                - If `(calibrated ?i ?r)` is true in state:
                    - Navigation cost: `min_{p \in P_image} dist(?r_loc, p)`.
                    - Cost = Navigation cost + 1 (take_image action).
                    - Location after image is the waypoint `p` that achieved the minimum navigation cost.
                - If `(calibrated ?i ?r)` is false in state:
                    - Navigation cost: `min_{w \in W_cal, p \in P_image} (dist(?r_loc, w) + dist(w, p))`.
                    - Cost = Navigation cost + 2 (calibrate + take_image actions).
                    - Location after image is the waypoint `p` that achieved the minimum `dist(w, p)` for the chosen `w`.
            - If Cost to get image is infinity (e.g., no visible_from waypoints), this rover/camera cannot achieve the goal.
            - Location after image is `?p_after_image`.
            - Cost to communicate: `min_{x \in X_comm} dist(?p_after_image, x) + 1 (communicate_image_data action).
            - Total cost for rover-camera pair `(?r, ?i)`: Cost to get image + Cost to communicate.

    - The minimum total cost over all suitable rovers (or rover-camera pairs) is `cost_g`.

    - If no suitable rover/camera combination can achieve the goal (e.g., required samples/visibility points don't exist or are unreachable), `cost_g` remains infinity. However, assuming solvable problems, this case should not result in infinity for unachieved goals. A large finite value could be used if infinity is problematic for the search algorithm, but standard A*/GBFS handles infinity. We will use `float('inf')`.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # --- Parse Static Information ---
        self.lander_waypoint = None
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {}       # rover -> store
        self.soil_sample_locations = set() # waypoint
        self.rock_sample_locations = set() # waypoint
        self.waypoint_graph = {}     # waypoint -> set of visible waypoints
        self.rover_traversal_graphs = {} # rover -> {waypoint -> set of traversable waypoints}
        self.camera_info = {}        # camera -> { 'on_board': rover, 'supports': set(modes), 'calibration_target': objective }
        self.objective_visibility = {} # objective -> set of waypoints visible from objective

        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()

        for fact_str in self.static:
            pred, *args = self._parse_fact(fact_str)
            if pred == 'at_lander':
                self.lander_waypoint = args[1] if len(args) > 1 else None # Assuming only one lander
            elif pred == 'equipped_for_soil_analysis':
                if args:
                    rover = args[0]
                    self.rover_capabilities.setdefault(rover, set()).add('soil')
                    self.all_rovers.add(rover)
            elif pred == 'equipped_for_rock_analysis':
                if args:
                    rover = args[0]
                    self.rover_capabilities.setdefault(rover, set()).add('rock')
                    self.all_rovers.add(rover)
            elif pred == 'equipped_for_imaging':
                if args:
                    rover = args[0]
                    self.rover_capabilities.setdefault(rover, set()).add('imaging')
                    self.all_rovers.add(rover)
            elif pred == 'store_of':
                if len(args) > 1:
                    store, rover = args
                    self.rover_stores[rover] = store
                    self.all_rovers.add(rover)
            elif pred == 'at_soil_sample':
                if args:
                    self.soil_sample_locations.add(args[0])
                    self.all_waypoints.add(args[0])
            elif pred == 'at_rock_sample':
                if args:
                    self.rock_sample_locations.add(args[0])
                    self.all_waypoints.add(args[0])
            elif pred == 'visible':
                if len(args) > 1:
                    w1, w2 = args
                    self.waypoint_graph.setdefault(w1, set()).add(w2)
                    self.waypoint_graph.setdefault(w2, set()).add(w1) # Visibility is symmetric
                    self.all_waypoints.add(w1)
                    self.all_waypoints.add(w2)
            elif pred == 'can_traverse':
                if len(args) > 2:
                    rover, w1, w2 = args
                    self.rover_traversal_graphs.setdefault(rover, {}).setdefault(w1, set()).add(w2)
                    self.all_rovers.add(rover)
                    self.all_waypoints.add(w1)
                    self.all_waypoints.add(w2)
            elif pred == 'calibration_target':
                if len(args) > 1:
                    camera, objective = args
                    self.camera_info.setdefault(camera, {})['calibration_target'] = objective
                    self.all_cameras.add(camera)
                    self.all_objectives.add(objective)
            elif pred == 'on_board':
                if len(args) > 1:
                    camera, rover = args
                    self.camera_info.setdefault(camera, {})['on_board'] = rover
                    self.all_cameras.add(camera)
                    self.all_rovers.add(rover)
            elif pred == 'supports':
                if len(args) > 1:
                    camera, mode = args
                    self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)
                    self.all_cameras.add(camera)
                    self.all_modes.add(mode)
            elif pred == 'visible_from':
                if len(args) > 1:
                    objective, waypoint = args
                    self.objective_visibility.setdefault(objective, set()).add(waypoint)
                    self.all_objectives.add(objective)
                    self.all_waypoints.add(waypoint)

        # Ensure all waypoints mentioned in traversal graphs are in the main set
        for rover_graph in self.rover_traversal_graphs.values():
             for w1, neighbors in rover_graph.items():
                 self.all_waypoints.add(w1)
                 for w2 in neighbors:
                     self.all_waypoints.add(w2)

        # Ensure all waypoints mentioned in visibility graphs are in the main set
        for w1, neighbors in self.waypoint_graph.items():
            self.all_waypoints.add(w1)
            for w2 in neighbors:
                self.all_waypoints.add(w2)


        # --- Precompute Shortest Paths ---
        self.rover_shortest_paths = {} # rover -> {from_wp -> {to_wp -> distance}}
        for rover in self.all_rovers:
            self.rover_shortest_paths[rover] = self._compute_all_pairs_shortest_paths(
                self.rover_traversal_graphs.get(rover, {}), self.all_waypoints)

        # --- Precompute Image Goal Navigation Costs ---
        # min_dist_w_to_p_image[rover][w][objective] = min_{p \in P_image(objective)} dist[rover][w][p]
        # p_best_for_w[rover][w][objective] = waypoint p achieving the minimum
        self.min_dist_w_to_p_image = {}
        self.p_best_for_w = {}

        for rover in self.all_rovers:
            self.min_dist_w_to_p_image.setdefault(rover, {})
            self.p_best_for_w.setdefault(rover, {})
            rover_dist = self.rover_shortest_paths[rover]

            for w in self.all_waypoints:
                self.min_dist_w_to_p_image[rover].setdefault(w, {})
                self.p_best_for_w[rover].setdefault(w, {})

                for objective in self.all_objectives:
                    p_image_waypoints = self.objective_visibility.get(objective, set())
                    min_dist = float('inf')
                    best_p = None

                    if p_image_waypoints:
                        for p in p_image_waypoints:
                            # Check if both w and p are valid keys in the distance map for this rover
                            if w in rover_dist and p in rover_dist[w]:
                                dist = rover_dist[w][p]
                                if dist < min_dist:
                                    min_dist = dist
                                    best_p = p

                    self.min_dist_w_to_p_image[rover][w][objective] = min_dist
                    self.p_best_for_w[rover][w][objective] = best_p # Store the best waypoint

        # d_cal_img[rover][u][camera][objective] = min_{w \in W_cal(camera), p \in P_image(objective)} (dist[rover][u][w] + dist[rover][w][p])
        # w_best_for_u[rover][u][camera][objective] = waypoint w achieving the minimum
        self.d_cal_img = {}
        self.w_best_for_u = {}

        for rover in self.all_rovers:
            self.d_cal_img.setdefault(rover, {})
            self.w_best_for_u.setdefault(rover, {})
            rover_dist = self.rover_shortest_paths[rover]

            for u in self.all_waypoints:
                self.d_cal_img[rover].setdefault(u, {})
                self.w_best_for_u[rover].setdefault(u, {})

                for camera in self.all_cameras:
                    cam_info = self.camera_info.get(camera, {})
                    cal_target = cam_info.get('calibration_target')
                    if not cal_target: continue

                    w_cal_waypoints = self.objective_visibility.get(cal_target, set())
                    if not w_cal_waypoints: continue

                    self.d_cal_img[rover][u].setdefault(camera, {})
                    self.w_best_for_u[rover][u].setdefault(camera, {})

                    for objective in self.all_objectives:
                        p_image_waypoints = self.objective_visibility.get(objective, set())
                        if not p_image_waypoints: continue

                        min_nav_cost = float('inf')
                        best_w = None

                        for w in w_cal_waypoints:
                            # Check if u and w are valid keys in the distance map for this rover
                            if u in rover_dist and w in rover_dist[u]:
                                dist_u_w = rover_dist[u][w]
                                # Check if w is a valid key in the precomputed image distances
                                if w in self.min_dist_w_to_p_image[rover] and objective in self.min_dist_w_to_p_image[rover][w]:
                                    min_dist_w_p = self.min_dist_w_to_p_image[rover][w][objective]

                                    if min_dist_w_p != float('inf'):
                                         total_dist = dist_u_w + min_dist_w_p
                                         if total_dist < min_nav_cost:
                                             min_nav_cost = total_dist
                                             best_w = w

                        self.d_cal_img[rover][u][camera][objective] = min_nav_cost
                        self.w_best_for_u[rover][u][camera][objective] = best_w # Store the best calibration waypoint

        # --- Identify Communication Points ---
        self.communication_points = set()
        if self.lander_waypoint:
            # Communication points are waypoints visible from the lander waypoint
            self.communication_points = self.waypoint_graph.get(self.lander_waypoint, set())


    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into (predicate, args)."""
        # Remove surrounding parentheses and split by space
        parts = fact_str.strip("()").split()
        if not parts:
            return None, []
        return parts[0], parts[1:]

    def _compute_all_pairs_shortest_paths(self, graph, all_nodes):
        """Computes all-pairs shortest paths using BFS."""
        distances = {u: {v: float('inf') for v in all_nodes} for u in all_nodes}

        # Handle nodes that might be in all_nodes but not in the graph keys (isolated or unreachable)
        for node in all_nodes:
             if node in distances:
                 distances[node][node] = 0 # Distance to self is 0

        for start_node in graph: # Only run BFS from nodes that are keys in the graph
            q = deque([(start_node, 0)])
            distances[start_node][start_node] = 0
            visited = {start_node}

            while q:
                current_node, dist = q.popleft()

                # Ensure current_node is still in the graph keys (should be if it came from q)
                if current_node in graph:
                    for neighbor in graph[current_node]:
                        # Ensure neighbor is one of the nodes we care about (in all_nodes)
                        if neighbor in all_nodes and neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_node][neighbor] = dist + 1
                            q.append((neighbor, dist + 1))

        return distances

    def _get_rover_location(self, state, rover_name):
        """Finds the current waypoint of the rover in the state."""
        for fact_str in state:
            pred, *args = self._parse_fact(fact_str)
            if pred == 'at' and args and args[0] == rover_name:
                return args[1] if len(args) > 1 else None
        return None # Should not happen in a valid state

    def _is_store_full(self, state, rover_name):
        """Checks if the rover's store is full."""
        store_name = self.rover_stores.get(rover_name)
        if not store_name: return False # Rover has no store? (Shouldn't happen)
        return f'(full {store_name})' in state

    def _is_camera_calibrated(self, state, camera_name, rover_name):
        """Checks if the camera on the rover is calibrated."""
        return f'(calibrated {camera_name} {rover_name})' in state

    def _has_soil_analysis(self, state, rover_name, waypoint_name):
        """Checks if the rover has soil analysis data for the waypoint."""
        return f'(have_soil_analysis {rover_name} {waypoint_name})' in state

    def _has_rock_analysis(self, state, rover_name, waypoint_name):
        """Checks if the rover has rock analysis data for the waypoint."""
        return f'(have_rock_analysis {rover_name} {waypoint_name})' in state

    def _has_image(self, state, rover_name, objective_name, mode_name):
        """Checks if the rover has image data for the objective and mode."""
        return f'(have_image {rover_name} {objective_name} {mode_name})' in state

    def __call__(self, state):
        h = 0
        state_facts = set(state) # Convert frozenset to set for faster lookups

        unachieved_goals = [g for g in self.goals if g not in state_facts]

        if not unachieved_goals:
            return 0 # Goal reached

        for goal_str in unachieved_goals:
            pred, *args = self._parse_fact(goal_str)
            cost_g = float('inf') # Use infinity initially

            if pred == 'communicated_soil_data':
                if not args: continue # Malformed goal
                waypoint_w = args[0]
                # Find suitable rovers
                suitable_rovers = [r for r in self.all_rovers if 'soil' in self.rover_capabilities.get(r, set()) and r in self.rover_stores]

                for rover in suitable_rovers:
                    rover_loc = self._get_rover_location(state_facts, rover)
                    if rover_loc is None: continue # Rover location unknown (shouldn't happen in valid state)

                    # Cost to get have_soil_analysis
                    cost_to_get_sample = float('inf')
                    loc_after_sample = None

                    if self._has_soil_analysis(state_facts, rover, waypoint_w):
                        cost_to_get_sample = 0
                        loc_after_sample = rover_loc # Rover didn't move for sample
                    elif waypoint_w in self.soil_sample_locations:
                        # Check if rover_loc and waypoint_w are valid in the distance map
                        if rover in self.rover_shortest_paths and rover_loc in self.rover_shortest_paths[rover] and waypoint_w in self.rover_shortest_paths[rover][rover_loc]:
                            dist_to_sample = self.rover_shortest_paths[rover][rover_loc][waypoint_w]
                            if dist_to_sample != float('inf'):
                                cost_sample_action = 1 # sample_soil
                                cost_store = 1 if self._is_store_full(state_facts, rover) else 0
                                cost_to_get_sample = dist_to_sample + cost_store + cost_sample_action
                                loc_after_sample = waypoint_w # Rover is at sample location after sampling
                    # else: Soil sample required by goal does not exist statically (cost_to_get_sample remains inf)

                    if cost_to_get_sample == float('inf') or loc_after_sample is None: continue # Cannot get sample

                    # Cost to communicate
                    loc_before_comm = loc_after_sample
                    cost_to_communicate = float('inf')
                    if self.communication_points:
                        min_dist_to_comm = float('inf')
                        # Check if loc_before_comm is valid in the distance map
                        if rover in self.rover_shortest_paths and loc_before_comm in self.rover_shortest_paths[rover]:
                            for comm_wp in self.communication_points:
                                # Check if comm_wp is valid in the distance map
                                if comm_wp in self.rover_shortest_paths[rover][loc_before_comm]:
                                    min_dist_to_comm = min(min_dist_to_comm, self.rover_shortest_paths[rover][loc_before_comm][comm_wp])

                        if min_dist_to_comm != float('inf'):
                            cost_to_communicate = min_dist_to_comm + 1 # communicate_soil_data

                    if cost_to_communicate == float('inf'): continue # Cannot communicate

                    total_rover_cost = cost_to_get_sample + cost_to_communicate
                    cost_g = min(cost_g, total_rover_cost)

            elif pred == 'communicated_rock_data':
                 if not args: continue # Malformed goal
                 waypoint_w = args[0]
                 # Find suitable rovers
                 suitable_rovers = [r for r in self.all_rovers if 'rock' in self.rover_capabilities.get(r, set()) and r in self.rover_stores]

                 for rover in suitable_rovers:
                     rover_loc = self._get_rover_location(state_facts, rover)
                     if rover_loc is None: continue

                     # Cost to get have_rock_analysis
                     cost_to_get_sample = float('inf')
                     loc_after_sample = None

                     if self._has_rock_analysis(state_facts, rover, waypoint_w):
                         cost_to_get_sample = 0
                         loc_after_sample = rover_loc
                     elif waypoint_w in self.rock_sample_locations:
                         if rover in self.rover_shortest_paths and rover_loc in self.rover_shortest_paths[rover] and waypoint_w in self.rover_shortest_paths[rover][rover_loc]:
                             dist_to_sample = self.rover_shortest_paths[rover][rover_loc][waypoint_w]
                             if dist_to_sample != float('inf'):
                                 cost_sample_action = 1 # sample_rock
                                 cost_store = 1 if self._is_store_full(state_facts, rover) else 0
                                 cost_to_get_sample = dist_to_sample + cost_store + cost_sample_action
                                 loc_after_sample = waypoint_w
                     # else: Rock sample required by goal does not exist statically (cost_to_get_sample remains inf)

                     if cost_to_get_sample == float('inf') or loc_after_sample is None: continue

                     # Cost to communicate
                     loc_before_comm = loc_after_sample
                     cost_to_communicate = float('inf')
                     if self.communication_points:
                         min_dist_to_comm = float('inf')
                         if rover in self.rover_shortest_paths and loc_before_comm in self.rover_shortest_paths[rover]:
                             for comm_wp in self.communication_points:
                                 if comm_wp in self.rover_shortest_paths[rover][loc_before_comm]:
                                     min_dist_to_comm = min(min_dist_to_comm, self.rover_shortest_paths[rover][loc_before_comm][comm_wp])

                         if min_dist_to_comm != float('inf'):
                             cost_to_communicate = min_dist_to_comm + 1 # communicate_rock_data

                     if cost_to_communicate == float('inf'): continue

                     total_rover_cost = cost_to_get_sample + cost_to_communicate
                     cost_g = min(cost_g, total_rover_cost)

            elif pred == 'communicated_image_data':
                if len(args) < 2: continue # Malformed goal
                objective_o, mode_m = args
                # Find suitable rover-camera pairs
                suitable_pairs = [] # (rover, camera)
                for rover in self.all_rovers:
                    if 'imaging' in self.rover_capabilities.get(rover, set()):
                        for camera, cam_info in self.camera_info.items():
                            if cam_info.get('on_board') == rover and mode_m in cam_info.get('supports', set()):
                                suitable_pairs.append((rover, camera))

                for rover, camera in suitable_pairs:
                    rover_loc = self._get_rover_location(state_facts, rover)
                    if rover_loc is None: continue

                    cam_info = self.camera_info.get(camera, {})
                    cal_target = cam_info.get('calibration_target')
                    if not cal_target: continue # Camera has no calibration target

                    w_cal_waypoints = self.objective_visibility.get(cal_target, set())
                    p_image_waypoints = self.objective_visibility.get(objective_o, set())

                    if not w_cal_waypoints or not p_image_waypoints: continue # Cannot calibrate or take image

                    # Cost to get have_image
                    cost_to_get_image = float('inf')
                    loc_after_image = None

                    if self._has_image(state_facts, rover, objective_o, mode_m):
                        cost_to_get_image = 0
                        loc_after_image = rover_loc # Rover didn't move for image (already has it)
                    else:
                        rover_dist = self.rover_shortest_paths[rover]
                        is_calibrated = self._is_camera_calibrated(state_facts, camera, rover)

                        if is_calibrated:
                            # Need to navigate to p_image, take image
                            # Check if rover_loc is valid in the precomputed image distances
                            if rover_loc in self.min_dist_w_to_p_image[rover] and objective_o in self.min_dist_w_to_p_image[rover][rover_loc]:
                                d_to_img = self.min_dist_w_to_p_image[rover][rover_loc][objective_o]
                                if d_to_img != float('inf'):
                                    cost_to_get_image = d_to_img + 1 # take_image
                                    # Check if rover_loc and objective_o are valid in p_best_for_w
                                    if rover_loc in self.p_best_for_w[rover] and objective_o in self.p_best_for_w[rover][rover_loc]:
                                        loc_after_image = self.p_best_for_w[rover][rover_loc][objective_o] # Get the best p_image waypoint
                        else:
                            # Need to navigate to w_cal, calibrate, navigate to p_image, take image
                            # Check if rover_loc, camera, objective_o are valid in d_cal_img
                            if rover_loc in self.d_cal_img[rover] and camera in self.d_cal_img[rover][rover_loc] and objective_o in self.d_cal_img[rover][rover_loc][camera]:
                                d_nav_cal_img = self.d_cal_img[rover][rover_loc][camera][objective_o]
                                if d_nav_cal_img != float('inf'):
                                    cost_to_get_image = d_nav_cal_img + 2 # calibrate + take_image
                                    # Get the waypoint after taking the image
                                    # Check if rover_loc, camera, objective_o are valid in w_best_for_u
                                    if rover_loc in self.w_best_for_u[rover] and camera in self.w_best_for_u[rover][rover_loc] and objective_o in self.w_best_for_u[rover][rover_loc][camera]:
                                        w_best = self.w_best_for_u[rover][rover_loc][camera][objective_o]
                                        # Check if w_best and objective_o are valid in p_best_for_w
                                        if w_best and w_best in self.p_best_for_w[rover] and objective_o in self.p_best_for_w[rover][w_best]:
                                            loc_after_image = self.p_best_for_w[rover][w_best][objective_o]


                    if cost_to_get_image == float('inf') or loc_after_image is None: continue # Cannot get image

                    # Cost to communicate
                    loc_before_comm = loc_after_image
                    cost_to_communicate = float('inf')
                    if self.communication_points:
                        min_dist_to_comm = float('inf')
                        # Check if loc_before_comm is valid in the distance map
                        if rover in self.rover_shortest_paths and loc_before_comm in self.rover_shortest_paths[rover]:
                            for comm_wp in self.communication_points:
                                # Check if comm_wp is valid in the distance map
                                if comm_wp in self.rover_shortest_paths[rover][loc_before_comm]:
                                    min_dist_to_comm = min(min_dist_to_comm, self.rover_shortest_paths[rover][loc_before_comm][comm_wp])

                        if min_dist_to_comm != float('inf'):
                            cost_to_communicate = min_dist_to_comm + 1 # communicate_image_data

                    if cost_to_communicate == float('inf'): continue # Cannot communicate

                    total_pair_cost = cost_to_get_image + cost_to_communicate
                    cost_g = min(cost_g, total_pair_cost)

            # Add the minimum cost for this goal to the total heuristic
            # If cost_g is still inf, it means this goal is unreachable.
            # Assuming solvable, this shouldn't happen for unachieved goals.
            # If it does, adding inf is correct for GBFS.
            h += cost_g

        return h
