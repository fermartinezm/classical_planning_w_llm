# Import necessary modules
import collections
import math

# Define the heuristic class
class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach a goal state by summing the estimated costs
    for each unsatisfied goal predicate. The cost for each goal is estimated
    independently, assuming the best available rover and shortest paths
    for navigation.

    The heuristic is non-admissible but aims to guide a greedy best-first
    search efficiently.

    Summary:
    The heuristic calculates the sum of costs for each goal that is not yet
    satisfied. For each unsatisfied goal (communicating soil, rock, or image data),
    it estimates the minimum number of actions required to satisfy it, considering
    whether the required data/image is already collected. This estimation includes
    navigation costs (shortest path between waypoints for the specific rover),
    sampling/imaging costs (including calibration and store management), and
    communication costs. It finds the minimum cost over all suitable rovers
    and relevant waypoints (sample locations, image points, calibration points,
    communication points).

    Assumptions:
    - Action costs are 1.
    - Navigation cost between waypoints for a rover is the shortest path distance
      in the graph defined by the rover's `can_traverse` predicates.
    - The heuristic assumes that if a goal requires data/image from a specific
      waypoint/objective, that waypoint has a sample or the objective is visible
      from some waypoint (checked via static facts).
    - If a required static condition (like equipment, camera support, visibility)
      or connectivity (via `can_traverse`) is missing such that a goal component
      is unreachable, the heuristic assigns a large finite cost to indicate difficulty.
    - The heuristic estimates costs for goals independently, which might overestimate
      in cases where actions or locations can be shared between goals.

    Heuristic Initialization:
    The constructor `__init__` precomputes static information from the task:
    - Parses objects (rovers, waypoints, etc.) by looking at arguments in static and initial facts.
    - Stores lander locations.
    - Stores rover capabilities (equipped for soil, rock, imaging).
    - Stores store ownership.
    - Stores camera information (on board, supports, calibration target).
    - Builds the general visibility graph (`visible`).
    - Stores image points for objectives (`visible_from`).
    - Stores calibration points for calibration targets (`visible_from`).
    - For each rover, builds its specific navigation graph based on `can_traverse`.
    - Computes all-pairs shortest paths for each rover's navigation graph using BFS.
      Stores distances in `self.nav_dists[rover][from_wp][to_wp]`.
    - Identifies communication points (waypoints visible from any lander location).

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
    1. Check if the current state is a goal state. If yes, return 0.
    2. Parse dynamic facts from the current state to get:
       - Current location of each rover.
       - Which rover holds which soil/rock sample data.
       - Which rover holds which image data.
       - Status of each store (full/empty).
       - Calibration status of each camera on each rover.
       - Locations of remaining soil/rock samples.
    3. Identify the set of communication points (waypoints visible from any lander).
    4. Initialize the total heuristic value `h = 0`.
    5. Iterate through each goal predicate specified in `task.goals`.
    6. For each goal predicate `g`:
       - If `g` is already true in the current state, continue to the next goal.
       - If `g` is `(communicated_soil_data ?w)`:
         - Extract waypoint `w`.
         - Estimate the minimum cost to achieve this goal:
           - Check if any rover currently holds `(have_soil_analysis ?r ?w)`.
           - If data is NOT held:
             - Find the minimum cost over all equipped rovers `r` to:
               - Navigate from `r`'s current location to `w`.
               - (If `r`'s store is full, add cost for `drop` action).
               - Add cost for `sample_soil` action.
               - Navigate from `w` to a communication point.
               - Add cost for `communicate_soil_data` action.
             - Use the precomputed shortest path distances for navigation costs.
             - If no equipped rover can reach `w` or a communication point, the cost is considered infinite (represented by a large number).
           - If data IS held by rover `r_h` at location `p_h`:
             - Find the minimum cost over communication points to:
               - Navigate from `p_h` to a communication point.
               - Add cost for `communicate_soil_data` action.
             - If `r_h` cannot reach any communication point, the cost is considered infinite.
           - Add the minimum estimated cost for this goal to `h`.
       - If `g` is `(communicated_rock_data ?w)`:
         - Similar logic as for soil data, using rock-specific predicates and equipment.
       - If `g` is `(communicated_image_data ?o ?m)`:
         - Extract objective `o` and mode `m`.
         - Estimate the minimum cost to achieve this goal:
           - Check if any rover currently holds `(have_image ?r ?o ?m)`.
           - If image is NOT held:
             - Find the minimum cost over all equipped rovers `r` with a camera `i` supporting `m` to:
               - Find a calibration target `t` for `i` and a waypoint `w_cal` visible from `t`.
               - Find a waypoint `w_img` visible from `o`.
               - (If camera `i` on `r` is NOT calibrated):
                 - Navigate from `r`'s current location to `w_cal`.
                 - Add cost for `calibrate` action.
                 - Navigate from `w_cal` to `w_img`.
               - (If camera `i` on `r` IS calibrated):
                 - Navigate from `r`'s current location to `w_img`.
               - Add cost for `take_image` action.
               - Navigate from `w_img` to a communication point.
               - Add cost for `communicate_image_data` action.
             - Use the precomputed shortest path distances for navigation costs.
             - If no suitable rover can perform the steps or reach the required points, the cost is considered infinite.
           - If image IS held by rover `r_h` at location `p_h`:
             - Find the minimum cost over communication points to:
               - Navigate from `p_h` to a communication point.
               - Add cost for `communicate_image_data` action.
             - If `r_h` cannot reach any communication point, the cost is considered infinite.
           - Add the minimum estimated cost for this goal to `h`.
    7. If at any point a required goal component is deemed unreachable (cost is infinite), return a large constant value (e.g., 1000000) to indicate a likely dead end.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        self.task = task
        self.static_facts = task.static

        # --- Parse Static Information ---
        # Collect all potential objects by type from known predicates
        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()

        # Helper to add objects based on predicate structure
        def add_objects(pred, objs):
            if pred in ('at', 'can_traverse'):
                if len(objs) > 0: self.rovers.add(objs[0])
                if len(objs) > 1: self.waypoints.add(objs[1])
                if len(objs) > 2: self.waypoints.add(objs[2])
            elif pred == 'at_lander':
                if len(objs) > 0: self.landers.add(objs[0])
                if len(objs) > 1: self.waypoints.add(objs[1])
            elif pred == 'store_of':
                if len(objs) > 0: self.stores.add(objs[0])
                if len(objs) > 1: self.rovers.add(objs[1])
            elif pred == 'on_board':
                if len(objs) > 0: self.cameras.add(objs[0])
                if len(objs) > 1: self.rovers.add(objs[1])
            elif pred == 'supports':
                if len(objs) > 0: self.cameras.add(objs[0])
                if len(objs) > 1: self.modes.add(objs[1])
            elif pred == 'calibration_target':
                if len(objs) > 0: self.cameras.add(objs[0])
                if len(objs) > 1: self.objectives.add(objs[1])
            elif pred in ('visible', 'visible_from'):
                 if len(objs) > 0: self.waypoints.add(objs[0])
                 if len(objs) > 1:
                     if pred == 'visible_from': self.objectives.add(objs[0])
                     self.waypoints.add(objs[1])
            elif pred in ('equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'):
                if len(objs) > 0: self.rovers.add(objs[0])
            elif pred in ('have_rock_analysis', 'have_soil_analysis', 'at_soil_sample', 'at_rock_sample'):
                if len(objs) > 0:
                    if pred in ('have_rock_analysis', 'have_soil_analysis'): self.rovers.add(objs[0])
                    if pred in ('at_soil_sample', 'at_rock_sample'): self.waypoints.add(objs[0])
                if len(objs) > 1: self.waypoints.add(objs[1]) # have_... predicates have waypoint second
            elif pred == 'have_image':
                if len(objs) > 0: self.rovers.add(objs[0])
                if len(objs) > 1: self.objectives.add(objs[1])
                if len(objs) > 2: self.modes.add(objs[2])
            elif pred in ('communicated_soil_data', 'communicated_rock_data'):
                if len(objs) > 0: self.waypoints.add(objs[0])
            elif pred == 'communicated_image_data':
                if len(objs) > 0: self.objectives.add(objs[0])
                if len(objs) > 1: self.modes.add(objs[1])
            elif pred in ('empty', 'full'):
                if len(objs) > 0: self.stores.add(objs[0])


        # Iterate through all facts (static + initial) to find objects and populate static info
        all_facts = self.static_facts | self.task.initial_state
        for fact_str in all_facts:
             parts = self._parse_fact(fact_str)
             if not parts: continue
             pred = parts[0]
             objs = parts[1:]
             add_objects(pred, objs) # Populate object sets

        # Now populate specific static info structures
        self.lander_locations = {} # lander -> waypoint
        self.rover_equipped_soil = {r: False for r in self.rovers} # rover -> bool
        self.rover_equipped_rock = {r: False for r in self.rovers} # rover -> bool
        self.rover_equipped_imaging = {r: False for r in self.rovers} # rover -> bool
        self.store_owners = {} # store -> rover
        # camera_info structure: rover -> camera -> {'supports': set(), 'calibration_target': objective}
        self.camera_info = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.visible_graph = collections.defaultdict(set) # waypoint -> set of visible waypoints
        self.image_points = collections.defaultdict(set) # objective -> set of visible waypoints
        self.calibration_points = collections.defaultdict(set) # calibration_target -> set of visible waypoints
        self.rover_nav_graphs = {r: collections.defaultdict(set) for r in self.rovers} # rover -> waypoint -> set of reachable waypoints (in one step)

        # Temporary storage for camera info before linking to rovers
        temp_camera_info = collections.defaultdict(dict) # camera -> {'supports': set(), 'calibration_target': objective}

        for fact_str in self.static_facts:
            parts = self._parse_fact(fact_str)
            if not parts: continue
            pred = parts[0]
            objs = parts[1:]

            if pred == 'at_lander':
                self.lander_locations[objs[0]] = objs[1]
            elif pred == 'equipped_for_soil_analysis':
                self.rover_equipped_soil[objs[0]] = True
            elif pred == 'equipped_for_rock_analysis':
                self.rover_equipped_rock[objs[0]] = True
            elif pred == 'equipped_for_imaging':
                self.rover_equipped_imaging[objs[0]] = True
            elif pred == 'store_of':
                self.store_owners[objs[0]] = objs[1]
            elif pred == 'on_board':
                cam, rover = objs
                self.camera_info[rover][cam]['on_board'] = True
            elif pred == 'supports':
                cam, mode = objs
                if 'supports' not in temp_camera_info[cam]:
                     temp_camera_info[cam]['supports'] = set()
                temp_camera_info[cam]['supports'].add(mode)
            elif pred == 'calibration_target':
                cam, obj = objs
                temp_camera_info[cam]['calibration_target'] = obj
            elif pred == 'visible':
                wp1, wp2 = objs
                self.visible_graph[wp1].add(wp2)
                self.visible_graph[wp2].add(wp1) # Assuming visibility is symmetric
            elif pred == 'visible_from':
                obj, wp = objs
                self.image_points[obj].add(wp)
                # Calibration points are visible_from the calibration target objective
                # We link this after processing all calibration_target facts
            elif pred == 'can_traverse':
                 r, wp1, wp2 = objs
                 self.rover_nav_graphs[r][wp1].add(wp2)
                 self.rover_nav_graphs[r][wp2].add(wp1) # Assuming can_traverse is symmetric

        # Link temp camera info to rovers and populate calibration_points
        for rover in self.rovers:
             for cam in self.camera_info.get(rover, {}): # Iterate cameras on this rover
                 if cam in temp_camera_info:
                     self.camera_info[rover][cam].update(temp_camera_info[cam])
                     cal_target = self.camera_info[rover][cam].get('calibration_target')
                     if cal_target:
                         # Find waypoints visible from this calibration target objective
                         if cal_target in self.image_points: # visible_from objective wp also means visible_from calibration_target wp
                             self.calibration_points[cal_target].update(self.image_points[cal_target])


        # Compute all-pairs shortest paths for each rover
        self.nav_dists = {} # rover -> from_wp -> to_wp -> dist
        for rover in self.rovers:
            # Need all waypoints in the domain, not just those in can_traverse for this rover
            # BFS should explore all reachable nodes from each start node within the rover's graph
            # The graph should contain all waypoints as nodes, even if isolated for a specific rover
            # Let's rebuild the graph with all waypoints first
            full_rover_graph = {wp: set() for wp in self.waypoints}
            if rover in self.rover_nav_graphs:
                 for wp1, neighbors in self.rover_nav_graphs[rover].items():
                     full_rover_graph[wp1].update(neighbors)
                     for neighbor in neighbors:
                         full_rover_graph[neighbor].add(wp1) # Ensure symmetry in graph representation

            self.nav_dists[rover] = self._compute_all_pairs_shortest_paths(full_rover_graph)


        # Identify communication points
        self.comm_points = set()
        # Communication requires being at a waypoint visible from the lander location
        for lander, lander_wp in self.lander_locations.items():
             self.comm_points.update(self.visible_graph.get(lander_wp, set()))


    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and objects."""
        # Remove surrounding parentheses and split by space
        parts = fact_string.strip().strip('()').split()
        if not parts:
            return None
        return parts

    def _compute_all_pairs_shortest_paths(self, graph):
        """Computes shortest paths from all nodes using BFS."""
        dists = {}
        waypoints = list(graph.keys()) # Use all waypoints in the graph as potential start/end nodes

        for start_node in waypoints:
            dists[start_node] = {}
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            dists[start_node][start_node] = 0

            while q:
                curr_node, dist = q.popleft()

                # Ensure curr_node is in the graph's keys before accessing neighbors
                for neighbor in graph.get(curr_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dists[start_node][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))
        return dists

    def get_nav_cost(self, rover, start_wp, end_wp):
        """Returns shortest path distance or None if unreachable."""
        if rover not in self.nav_dists or start_wp not in self.nav_dists[rover] or end_wp not in self.nav_dists[rover][start_wp]:
            return None # Unreachable or waypoint not in rover's graph or start/end node not in precomputed dists
        return self.nav_dists[rover][start_wp][end_wp]

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        if self.task.goal_reached(state):
            return 0

        # --- Parse Dynamic Information from State ---
        rover_locations = {} # rover -> waypoint
        rover_soil_data = collections.defaultdict(set) # rover -> set of waypoints
        rover_rock_data = collections.defaultdict(set) # rover -> set of waypoints
        rover_images = collections.defaultdict(set) # rover -> set of (objective, mode) tuples
        rover_store_full = {} # rover -> bool (keyed by rover, not store)
        rover_camera_calibrated = collections.defaultdict(dict) # rover -> camera -> bool
        soil_samples_at_wp = set() # waypoint with soil sample
        rock_samples_at_wp = set() # waypoint with rock sample

        # Initialize store status to not full for all rovers with stores
        for store, owner in self.store_owners.items():
             rover_store_full[owner] = False # Assume not full unless 'full' fact exists

        for fact_str in state:
            parts = self._parse_fact(fact_str)
            if not parts: continue
            pred = parts[0]
            objs = parts[1:]

            if pred == 'at' and len(objs) == 2:
                rover_locations[objs[0]] = objs[1]
            elif pred == 'have_soil_analysis' and len(objs) == 2:
                rover_soil_data[objs[0]].add(objs[1])
            elif pred == 'have_rock_analysis' and len(objs) == 2:
                rover_rock_data[objs[0]].add(objs[1])
            elif pred == 'have_image' and len(objs) == 3:
                rover_images[objs[0]].add((objs[1], objs[2]))
            elif pred == 'full' and len(objs) == 1:
                 # Find which rover owns this store
                 store_name = objs[0]
                 owner_rover = self.store_owners.get(store_name)
                 if owner_rover:
                     rover_store_full[owner_rover] = True
            elif pred == 'calibrated' and len(objs) == 2:
                cam, rover = objs
                rover_camera_calibrated[rover][cam] = True
            elif pred == 'at_soil_sample' and len(objs) == 1:
                 soil_samples_at_wp.add(objs[0])
            elif pred == 'at_rock_sample' and len(objs) == 1:
                 rock_samples_at_wp.add(objs[0])


        h = 0
        UNREACHABLE_PENALTY = 1000000 # Large finite value for unreachable goals

        unmet_goals = self.task.goals - state

        for goal in unmet_goals:
            parts = self._parse_fact(goal)
            if not parts: continue
            pred = parts[0]

            if pred == 'communicated_soil_data':
                w = parts[1]
                goal_cost = UNREACHABLE_PENALTY

                # Check if data is already collected by any rover
                data_held = False
                rover_with_data = None
                for r, wps in rover_soil_data.items():
                    if w in wps:
                        data_held = True
                        rover_with_data = r
                        break

                if data_held:
                    # Data is collected, need to communicate
                    r_h = rover_with_data
                    r_h_loc = rover_locations.get(r_h)
                    # Fallback location if rover location is not in state (unlikely but defensive)
                    if r_h_loc is None: r_h_loc = w

                    min_nav_to_comm = math.inf
                    for comm_wp in self.comm_points:
                        nav_cost = self.get_nav_cost(r_h, r_h_loc, comm_wp)
                        if nav_cost is not None:
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                    if min_nav_to_comm != math.inf:
                        goal_cost = min_nav_to_comm + 1 # Nav + Communicate
                    # Else: remains UNREACHABLE_PENALTY

                else:
                    # Data is not collected, need to sample and communicate
                    min_acquisition_comm_cost = math.inf
                    for r in self.rovers:
                        if self.rover_equipped_soil.get(r, False):
                            r_loc = rover_locations.get(r)
                            if r_loc is None: continue # Rover location unknown

                            # Cost to acquire data
                            nav_to_sample = self.get_nav_cost(r, r_loc, w)
                            if nav_to_sample is None: continue # Cannot reach sample point

                            acquisition_cost = nav_to_sample + 1 # Nav + Sample
                            store_full = rover_store_full.get(r, False)
                            if store_full: acquisition_cost += 1 # Drop

                            # Cost to communicate from sample point w
                            min_nav_to_comm = math.inf
                            for comm_wp in self.comm_points:
                                nav_cost = self.get_nav_cost(r, w, comm_wp)
                                if nav_cost is not None:
                                    min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                            if min_nav_to_comm != math.inf:
                                total_cost = acquisition_cost + min_nav_to_comm + 1 # Nav + Communicate
                                min_acquisition_comm_cost = min(min_acquisition_comm_cost, total_cost)

                    if min_acquisition_comm_cost != math.inf:
                        goal_cost = min_acquisition_comm_cost
                    # Else: remains UNREACHABLE_PENALTY

                h += goal_cost
                if goal_cost == UNREACHABLE_PENALTY:
                     return UNREACHABLE_PENALTY # Early exit if any goal is unreachable

            elif pred == 'communicated_rock_data':
                w = parts[1]
                goal_cost = UNREACHABLE_PENALTY

                # Check if data is already collected by any rover
                data_held = False
                rover_with_data = None
                for r, wps in rover_rock_data.items():
                    if w in wps:
                        data_held = True
                        rover_with_data = r
                        break

                if data_held:
                    # Data is collected, need to communicate
                    r_h = rover_with_data
                    r_h_loc = rover_locations.get(r_h)
                    # Fallback location
                    if r_h_loc is None: r_h_loc = w

                    min_nav_to_comm = math.inf
                    for comm_wp in self.comm_points:
                        nav_cost = self.get_nav_cost(r_h, r_h_loc, comm_wp)
                        if nav_cost is not None:
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                    if min_nav_to_comm != math.inf:
                        goal_cost = min_nav_to_comm + 1 # Nav + Communicate
                    # Else: remains UNREACHABLE_PENALTY

                else:
                    # Data is not collected, need to sample and communicate
                    min_acquisition_comm_cost = math.inf
                    for r in self.rovers:
                        if self.rover_equipped_rock.get(r, False):
                            r_loc = rover_locations.get(r)
                            if r_loc is None: continue # Rover location unknown

                            # Cost to acquire data
                            nav_to_sample = self.get_nav_cost(r, r_loc, w)
                            if nav_to_sample is None: continue # Cannot reach sample point

                            acquisition_cost = nav_to_sample + 1 # Nav + Sample
                            store_full = rover_store_full.get(r, False)
                            if store_full: acquisition_cost += 1 # Drop

                            # Cost to communicate from sample point w
                            min_nav_to_comm = math.inf
                            for comm_wp in self.comm_points:
                                nav_cost = self.get_nav_cost(r, w, comm_wp)
                                if nav_cost is not None:
                                    min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                            if min_nav_to_comm != math.inf:
                                total_cost = acquisition_cost + min_nav_to_comm + 1 # Nav + Communicate
                                min_acquisition_comm_cost = min(min_acquisition_comm_cost, total_cost)

                    if min_acquisition_comm_cost != math.inf:
                        goal_cost = min_acquisition_comm_cost
                    # Else: remains UNREACHABLE_PENALTY

                h += goal_cost
                if goal_cost == UNREACHABLE_PENALTY:
                     return UNREACHABLE_PENALTY # Early exit

            elif pred == 'communicated_image_data':
                o = parts[1]
                m = parts[2]
                goal_cost = UNREACHABLE_PENALTY

                # Check if image is already taken by any rover
                image_held = False
                rover_with_image = None
                for r, images in rover_images.items():
                    if (o, m) in images:
                        image_held = True
                        rover_with_image = r
                        break

                if image_held:
                    # Image is taken, need to communicate
                    r_h = rover_with_image
                    r_h_loc = rover_locations.get(r_h)
                    # Fallback location
                    if r_h_loc is None:
                         # Assume it's at some image point for this objective if location unknown
                         r_h_loc = next(iter(self.image_points.get(o, [None])), None)
                         if r_h_loc is None:
                             # Cannot even guess location, likely unreachable
                             h += UNREACHABLE_PENALTY
                             return UNREACHABLE_PENALTY


                    min_nav_to_comm = math.inf
                    for comm_wp in self.comm_points:
                        nav_cost = self.get_nav_cost(r_h, r_h_loc, comm_wp)
                        if nav_cost is not None:
                            min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                    if min_nav_to_comm != math.inf:
                        goal_cost = min_nav_to_comm + 1 # Nav + Communicate
                    # Else: remains UNREACHABLE_PENALTY

                else:
                    # Image is not taken, need to acquire and communicate
                    min_acquisition_comm_cost = math.inf
                    for r in self.rovers:
                        if self.rover_equipped_imaging.get(r, False):
                            # Find camera i on r supporting m
                            best_camera = None
                            for cam, cam_info in self.camera_info.get(r, {}).items():
                                if m in cam_info.get('supports', set()):
                                    best_camera = cam
                                    break
                            if not best_camera: continue # Rover r cannot take image in mode m

                            cam_info = self.camera_info[r][best_camera]
                            cal_target = cam_info.get('calibration_target')
                            cal_wps = self.calibration_points.get(cal_target, [])
                            img_wps = self.image_points.get(o, [])
                            comm_wps = self.comm_points

                            if not cal_target or not cal_wps or not img_wps or not comm_wps: continue # Missing static info

                            for cal_wp in cal_wps:
                                for img_wp in img_wps:
                                    for comm_wp in comm_wps:
                                        current_cost = 0
                                        calibrated = rover_camera_calibrated.get(r, {}).get(best_camera, False)
                                        r_loc = rover_locations.get(r)
                                        if r_loc is None: continue # Rover location unknown

                                        if not calibrated:
                                            nav1 = self.get_nav_cost(r, r_loc, cal_wp)
                                            if nav1 is None: continue
                                            current_cost += nav1 + 1 # Nav + Calibrate
                                            nav2 = self.get_nav_cost(r, cal_wp, img_wp)
                                            if nav2 is None: continue
                                            current_cost += nav2
                                        else:
                                            nav1 = self.get_nav_cost(r, r_loc, img_wp)
                                            if nav1 is None: continue
                                            current_cost += nav1

                                        current_cost += 1 # Take image
                                        nav3 = self.get_nav_cost(r, img_wp, comm_wp)
                                        if nav3 is None: continue
                                        current_cost += nav3 + 1 # Nav + Communicate

                                        min_acquisition_comm_cost = min(min_acquisition_comm_cost, current_cost)

                    if min_acquisition_comm_cost != math.inf:
                        goal_cost = min_acquisition_comm_cost
                    # Else: remains UNREACHABLE_PENALTY

                h += goal_cost
                if goal_cost == UNREACHABLE_PENALTY:
                     return UNREACHABLE_PENALTY # Early exit

        return h
