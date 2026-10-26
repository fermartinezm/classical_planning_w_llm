import collections

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the total number of actions required to achieve
    all unachieved goal facts. It sums up the estimated costs for each
    unachieved goal independently. The cost for each goal is broken down
    into the steps needed to collect the required data (soil sample, rock
    sample, or image) and the steps needed to communicate that data.
    Navigation costs are estimated using precomputed shortest paths between
    waypoints for each rover.

    Assumptions:
    - The heuristic assumes that each unachieved goal can be pursued independently
      by the most suitable available rover, and the costs are additive. This
      ignores potential synergies (e.g., one trip to collect multiple samples)
      and conflicts (e.g., multiple rovers needing the same resource or path).
    - The cost of the 'drop' action is included in the sample cost calculation
      if the chosen rover's store is full.
    - The heuristic for image goals estimates the cost of calibration and
      navigation assuming a path from the rover's current location to a
      calibration waypoint (if needed) and then to an image waypoint.
      The communication path is estimated from the location where the data
      was collected (or the rover's current location if data was pre-existing).
    - Reachability is checked using precomputed shortest paths. Unreachable
      goals result in an infinite heuristic value.

    Heuristic Initialization:
    The constructor `__init__` parses the static facts from the planning task.
    It identifies rovers, waypoints, landers, objectives, cameras, modes,
    stores, equipment, camera capabilities, calibration targets, visibility,
    and rover traversability. It then precomputes the all-pairs shortest paths
    for each rover based on its `can_traverse` predicates using BFS. It also
    identifies the communication waypoints (those visible from the lander's location).

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
    1.  Extract dynamic information from the current state: rover locations,
        store status, collected data/images, camera calibration status.
    2.  Initialize the total heuristic value `h` to 0.
    3.  Identify the set of unachieved goal facts. If the set is empty, return 0.
    4.  Group the unachieved goals by type: soil data, rock data, and image data.
    5.  For each unachieved `(communicated_soil_data ?w)` goal:
        a.  Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r` in the current state.
        b.  If no rover has the soil data:
            i.  Estimate the minimum cost (actions + navigation) for a soil-equipped rover to sample at `?w` and then communicate from a communication waypoint. This is the minimum over soil-equipped rovers `?r` of: 1 (sample) + 1 (communicate) + (1 if `?r`'s store is full) + shortest_path(`?r`'s current location, `?w`) + shortest_path(`?w`, any communication waypoint).
            ii. Add this minimum cost to the goal's cost.
        c.  If some rover (`collected_by_rover`) already has the soil data:
            i.  Estimate the minimum cost (action + navigation) for `collected_by_rover` to communicate from a communication waypoint. This is 1 (communicate) + shortest_path(`collected_by_rover`'s current location, any communication waypoint).
            ii. Add this minimum cost to the goal's cost.
        d.  Add the total cost for this soil goal to the overall heuristic `h`. If any required step is impossible, the goal is unreachable, and the heuristic is infinite.
    6.  For each unachieved `(communicated_rock_data ?w)` goal: Follow a similar process as for soil data, using rock-specific predicates and equipped rovers.
    7.  For each unachieved `(communicated_image_data ?o ?m)` goal:
        a.  Check if `(have_image ?r ?o ?m)` is true for any rover `?r` in the current state.
        b.  If no rover has the image data:
            i.  Estimate the minimum cost (actions + navigation + calibration if needed) for a suitable rover/camera pair to take the image and then communicate. This is the minimum over suitable pairs of: 1 (take_image) + 1 (communicate) + (1 if calibration is needed) + navigation cost. The navigation cost is the shortest path from the rover's current location to a calibration waypoint (if needed), then to an image waypoint, then to a communication waypoint.
            ii. Add this minimum cost to the goal's cost.
        c.  If some rover (`taken_by_rover`) already has the image data:
            i.  Estimate the minimum cost (action + navigation) for `taken_by_rover` to communicate from a communication waypoint. This is 1 (communicate) + shortest_path(`taken_by_rover`'s current location, any communication waypoint).
            ii. Add this minimum cost to the goal's cost.
        d.  Add the total cost for this image goal to the overall heuristic `h`. If any required step is impossible, the goal is unreachable, and the heuristic is infinite.
    8.  Return the total heuristic value `h`.
    """
    def __init__(self, task):
        """
        Initializes the heuristic. Precomputes static information and shortest paths.

        Args:
            task: The planning task object (instance of Task class).
        """
        self.task = task
        self.static_facts = task.static
        self.goals = task.goals

        # --- Precompute Static Information ---
        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.objectives = set()
        self.cameras = set()
        self.modes = set()
        self.stores = set()

        self.equipped_soil = set() # {rover}
        self.equipped_rock = set() # {rover}
        self.equipped_imaging = set() # {rover}
        self.store_of_rover = {} # {store: rover}
        self.on_board_cameras = collections.defaultdict(set) # {rover: {camera}}
        self.camera_modes = collections.defaultdict(set) # {camera: {mode}}
        self.calibration_targets = {} # {camera: objective}
        self.visible_wps = set() # {(wp1, wp2)}
        self.visible_from_wps = collections.defaultdict(set) # {objective: {waypoint}}
        self.lander_at = {} # {lander: waypoint}

        # Traversability graph for each rover {rover: {wp1: {wp2}}}
        self.rover_traversal_graph = collections.defaultdict(lambda: collections.defaultdict(set))

        self._parse_static_facts()

        # Precompute shortest paths for each rover
        self.rover_shortest_paths = {} # {rover: {wp1: {wp2: distance}}}
        self._precompute_shortest_paths()

        # Precompute communication waypoints (visible from lander)
        self.comm_wps = set()
        for lander, lander_wp in self.lander_at.items():
             # Find waypoints wp where (visible wp lander_wp) is true
             for w1, w2 in self.visible_wps:
                 if w2 == lander_wp:
                     self.comm_wps.add(w1)

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a tuple."""
        # Remove parentheses and split by space
        parts = fact_string.strip("()").split()
        if not parts:
            return None
        return tuple(parts)

    def _parse_static_facts(self):
        for fact in self.static_facts:
            parsed = self._parse_fact(fact)
            if not parsed: continue # Handle potential parsing errors

            pred = parsed[0]
            args = parsed[1:]

            if pred == 'at_lander':
                if len(args) == 2:
                    self.landers.add(args[0])
                    self.waypoints.add(args[1])
                    self.lander_at[args[0]] = args[1]
            elif pred == 'at': # Initial rover location is static in some domains, dynamic in others. Rovers domain has it dynamic. We extract rovers here.
                 if len(args) == 2:
                    self.rovers.add(args[0])
                    self.waypoints.add(args[1])
            elif pred == 'can_traverse':
                if len(args) == 3:
                    rover, wp1, wp2 = args
                    self.rovers.add(rover)
                    self.waypoints.add(wp1)
                    self.waypoints.add(wp2)
                    self.rover_traversal_graph[rover][wp1].add(wp2)
            elif pred == 'equipped_for_soil_analysis':
                if len(args) == 1:
                    self.rovers.add(args[0])
                    self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                if len(args) == 1:
                    self.rovers.add(args[0])
                    self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging':
                if len(args) == 1:
                    self.rovers.add(args[0])
                    self.equipped_imaging.add(args[0])
            elif pred == 'store_of':
                if len(args) == 2:
                    store, rover = args
                    self.stores.add(store)
                    self.rovers.add(rover)
                    self.store_of_rover[store] = rover
            elif pred == 'on_board':
                if len(args) == 2:
                    camera, rover = args
                    self.cameras.add(camera)
                    self.rovers.add(rover)
                    self.on_board_cameras[rover].add(camera)
            elif pred == 'supports':
                if len(args) == 2:
                    camera, mode = args
                    self.cameras.add(camera)
                    self.modes.add(mode)
                    self.camera_modes[camera].add(mode)
            elif pred == 'calibration_target':
                if len(args) == 2:
                    camera, objective = args
                    self.cameras.add(camera)
                    self.objectives.add(objective)
                    self.calibration_targets[camera] = objective
            elif pred == 'visible':
                if len(args) == 2:
                    wp1, wp2 = args
                    self.waypoints.add(wp1)
                    self.waypoints.add(wp2)
                    self.visible_wps.add((wp1, wp2))
            elif pred == 'visible_from':
                if len(args) == 2:
                    objective, waypoint = args
                    self.objectives.add(objective)
                    self.waypoints.add(waypoint)
                    self.visible_from_wps[objective].add(waypoint)
            # Add other types if necessary, though they seem dynamic

        # Ensure all rovers mentioned in can_traverse are in self.rovers
        for rover in self.rover_traversal_graph:
             self.rovers.add(rover)


    def _precompute_shortest_paths(self):
        """Computes all-pairs shortest paths for each rover using BFS."""
        for rover in self.rovers:
            self.rover_shortest_paths[rover] = {}
            graph = self.rover_traversal_graph.get(rover, {})

            # Collect all waypoints relevant to this rover's graph or initial location
            # We need paths between all waypoints, even if some are isolated for this rover initially.
            # The BFS should explore from all known waypoints.
            all_relevant_wps = set(self.waypoints) # Consider all waypoints in the domain

            for start_node in all_relevant_wps:
                distances = {wp: float('inf') for wp in all_relevant_wps}
                distances[start_node] = 0
                queue = collections.deque([start_node])

                # Use the graph specific to this rover
                current_rover_graph = self.rover_traversal_graph.get(rover, {})

                while queue:
                    current_node = queue.popleft()

                    # Check if current_node is a key in the rover's graph
                    if current_node in current_rover_graph:
                        for neighbor in current_rover_graph[current_node]:
                            if distances[neighbor] == float('inf'):
                                distances[neighbor] = distances[current_node] + 1
                                queue.append(neighbor)
                self.rover_shortest_paths[rover][start_node] = distances


    def get_shortest_path_distance(self, rover, start_wp, end_wp):
        """Looks up the precomputed shortest path distance."""
        if rover not in self.rover_shortest_paths:
             # This rover might not have any traversal edges defined.
             # Path exists only if start == end.
             return 0 if start_wp == end_wp else None

        if start_wp not in self.rover_shortest_paths[rover]:
             # Start waypoint is not in the precomputed graph for this rover.
             # Path exists only if start == end.
             return 0 if start_wp == end_wp else None

        distance = self.rover_shortest_paths[rover][start_wp].get(end_wp, float('inf'))
        return distance if distance != float('inf') else None # Return None if unreachable


    def __call__(self, state, goals):
        """
        Computes the domain-dependent heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).
            goals: The set of goal facts (frozenset of fact strings).

        Returns:
            An estimate of the number of actions to reach a goal state.
            Returns float('inf') if the goal is unreachable.
        """
        # --- Extract Dynamic Information from State ---
        rover_locations = {} # {rover: waypoint}
        store_full = set() # {store}
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        calibrated_cameras_state = set() # {(camera, rover)}
        # communicated_soil, communicated_rock, communicated_image are goal facts,
        # their presence in state is checked via unmet_goals

        for fact in state:
            parsed = self._parse_fact(fact)
            if not parsed: continue

            pred = parsed[0]
            args = parsed[1:]

            if pred == 'at':
                if len(args) == 2: # (at rover waypoint)
                    rover_locations[args[0]] = args[1]
            elif pred == 'full':
                if len(args) == 1: # (full store)
                    store_full.add(args[0])
            elif pred == 'have_soil_analysis':
                 if len(args) == 2: # (have_soil_analysis rover waypoint)
                    have_soil.add((args[0], args[1]))
            elif pred == 'have_rock_analysis':
                 if len(args) == 2: # (have_rock_analysis rover waypoint)
                    have_rock.add((args[0], args[1]))
            elif pred == 'have_image':
                 if len(args) == 3: # (have_image rover objective mode)
                    have_image.add((args[0], args[1], args[2]))
            elif pred == 'calibrated':
                 if len(args) == 2: # (calibrated camera rover)
                    calibrated_cameras_state.add((args[0], args[1]))
            # Ignore other dynamic facts not directly relevant to goal prerequisites

        # --- Heuristic Calculation ---
        h = 0
        unmet_goals = goals - state

        if not unmet_goals:
            return 0 # Goal reached

        # Group unmet goals by type
        unmet_soil_wps = {w for g in unmet_goals for pred, w in [self._parse_fact(g)] if pred == 'communicated_soil_data'}
        unmet_rock_wps = {w for g in unmet_goals for pred, w in [self._parse_fact(g)] if pred == 'communicated_rock_data'}
        unmet_images = {(o, m) for g in unmet_goals for pred, o, m in [self._parse_fact(g)] if pred == 'communicated_image_data'}

        # Cost for Soil Data Goals
        for w in unmet_soil_wps:
            # Check if data is already collected by any rover
            collected_by_rover = None
            for rover in self.rovers:
                if (rover, w) in have_soil:
                    collected_by_rover = rover
                    break

            if collected_by_rover is None:
                # Need to sample and communicate
                cost_w = 1 + 1 # sample + communicate actions

                # Navigation cost to sample + communicate:
                # Min cost for a soil-equipped rover to go from current location to w, then to a comm wp.
                min_total_nav_cost = float('inf')
                for rover in self.equipped_soil:
                    if rover not in rover_locations: continue # Rover location unknown
                    rover_at = rover_locations[rover]
                    dist_to_sample = self.get_shortest_path_distance(rover, rover_at, w)

                    if dist_to_sample is not None:
                        # Cost for potential drop action before sampling
                        # Find the store for this rover
                        rover_store = None
                        for s, r in self.store_of_rover.items():
                            if r == rover:
                                rover_store = s
                                break
                        drop_cost = 1 if rover_store and rover_store in store_full else 0

                        min_dist_sample_to_comm = float('inf')
                        if not self.comm_wps: return float('inf') # No communication points
                        for comm_wp in self.comm_wps:
                            dist = self.get_shortest_path_distance(rover, w, comm_wp)
                            if dist is not None: min_dist_sample_to_comm = min(min_dist_sample_to_comm, dist)

                        if min_dist_sample_to_comm != float('inf'):
                             total_nav = dist_to_sample + min_dist_sample_to_comm
                             min_total_nav_cost = min(min_total_nav_cost, total_nav + drop_cost) # Add drop cost here

                if min_total_nav_cost == float('inf'): return float('inf') # Cannot sample or communicate
                cost_w += min_total_nav_cost

            else: # Data is already collected by collected_by_rover
                # Need to communicate
                cost_w = 1 # communicate action

                # Navigation cost from current location of collected_by_rover to communication point
                min_nav_curr_to_comm = float('inf')
                if not self.comm_wps: return float('inf') # No communication points
                if collected_by_rover not in rover_locations: return float('inf') # Rover location unknown
                rover_at = rover_locations[collected_by_rover]
                for comm_wp in self.comm_wps:
                    dist = self.get_shortest_path_distance(collected_by_rover, rover_at, comm_wp)
                    if dist is not None: min_nav_curr_to_comm = min(min_nav_curr_to_comm, dist)

                if min_nav_curr_to_comm == float('inf'): return float('inf') # Cannot communicate
                cost_w += min_nav_curr_to_comm

            h += cost_w

        # Cost for Rock Data Goals
        for w in unmet_rock_wps:
            # Check if data is already collected by any rover
            collected_by_rover = None
            for rover in self.rovers:
                if (rover, w) in have_rock:
                    collected_by_rover = rover
                    break

            if collected_by_rover is None:
                # Need to sample and communicate
                cost_w = 1 + 1 # sample + communicate actions

                # Navigation cost to sample + communicate:
                # Min cost for a rock-equipped rover to go from current location to w, then to a comm wp.
                min_total_nav_cost = float('inf')
                for rover in self.equipped_rock:
                    if rover not in rover_locations: continue # Rover location unknown
                    rover_at = rover_locations[rover]
                    dist_to_sample = self.get_shortest_path_distance(rover, rover_at, w)

                    if dist_to_sample is not None:
                         # Cost for potential drop action before sampling
                        rover_store = None
                        for s, r in self.store_of_rover.items():
                            if r == rover:
                                rover_store = s
                                break
                        drop_cost = 1 if rover_store and rover_store in store_full else 0

                        min_dist_sample_to_comm = float('inf')
                        if not self.comm_wps: return float('inf') # No communication points
                        for comm_wp in self.comm_wps:
                            dist = self.get_shortest_path_distance(rover, w, comm_wp)
                            if dist is not None: min_dist_sample_to_comm = min(min_dist_sample_to_comm, dist)

                        if min_dist_sample_to_comm != float('inf'):
                             total_nav = dist_to_sample + min_dist_sample_to_comm
                             min_total_nav_cost = min(min_total_nav_cost, total_nav + drop_cost) # Add drop cost here


                if min_total_nav_cost == float('inf'): return float('inf') # Cannot sample or communicate
                cost_w += min_total_nav_cost

            else: # Data is already collected by collected_by_rover
                # Need to communicate
                cost_w = 1 # communicate action

                # Navigation cost from current location of collected_by_rover to communication point
                min_nav_curr_to_comm = float('inf')
                if not self.comm_wps: return float('inf') # No communication points
                if collected_by_rover not in rover_locations: return float('inf') # Rover location unknown
                rover_at = rover_locations[collected_by_rover]
                for comm_wp in self.comm_wps:
                    dist = self.get_shortest_path_distance(collected_by_rover, rover_at, comm_wp)
                    if dist is not None: min_nav_curr_to_comm = min(min_nav_curr_to_comm, dist)

                if min_nav_curr_to_comm == float('inf'): return float('inf') # Cannot communicate
                cost_w += min_nav_curr_to_comm

            h += cost_w

        # Cost for Image Data Goals
        for o, m in unmet_images:
            # Check if image is already taken by any rover
            taken_by_rover = None
            for rover in self.rovers:
                if (rover, o, m) in have_image:
                    taken_by_rover = rover
                    break

            if taken_by_rover is None:
                # Need to take image and communicate
                cost_om = 1 + 1 # take_image + communicate actions

                # Find suitable rovers/cameras
                suitable_rovers_cameras = []
                for rover in self.equipped_imaging:
                    for camera in self.on_board_cameras.get(rover, set()):
                        if m in self.camera_modes.get(camera, set()):
                            suitable_rovers_cameras.append((rover, camera))

                if not suitable_rovers_cameras: return float('inf') # No equipped rover/camera for this image goal

                # Find the minimum cost (navigation + calibration if needed) for a suitable rover/camera
                # to go from current location -> (cal_wp) -> image_wp -> comm_wp
                min_total_task_nav_cost = float('inf')

                image_wps = self.visible_from_wps.get(o, set())
                if not image_wps: return float('inf') # Objective not visible from anywhere

                if not self.comm_wps: return float('inf') # No communication points

                for rover, camera in suitable_rovers_cameras:
                    if rover not in rover_locations: continue # Rover location unknown
                    rover_at = rover_locations[rover]
                    is_calibrated = (camera, rover) in calibrated_cameras_state
                    cal_target = self.calibration_targets.get(camera)
                    cal_wps = self.visible_from_wps.get(cal_target, set()) if cal_target else set()

                    if is_calibrated:
                        # Path: current -> image_wp -> comm_wp
                        min_path_curr_to_image_to_comm = float('inf')
                        for image_wp in image_wps:
                            dist_curr_to_image = self.get_shortest_path_distance(rover, rover_at, image_wp)
                            if dist_curr_to_image is not None:
                                for comm_wp in self.comm_wps:
                                    dist_image_to_comm = self.get_shortest_path_distance(rover, image_wp, comm_wp)
                                    if dist_image_to_comm is not None:
                                        total_dist = dist_curr_to_image + dist_image_to_comm
                                        min_path_curr_to_image_to_comm = min(min_path_curr_to_image_to_comm, total_dist)

                        if min_path_curr_to_image_to_comm != float('inf'):
                            nav_cost = min_path_curr_to_image_to_comm
                            min_total_task_nav_cost = min(min_total_task_nav_cost, nav_cost)

                    else: # Need calibration
                        if not cal_wps: continue # Calibration target not visible from anywhere

                        # Path: current -> cal_wp -> image_wp -> comm_wp
                        min_path_curr_to_cal_to_image_to_comm = float('inf')
                        for cal_wp in cal_wps:
                            dist_curr_to_cal = self.get_shortest_path_distance(rover, rover_at, cal_wp)
                            if dist_curr_to_cal is not None:
                                for image_wp in image_wps:
                                    dist_cal_to_image = self.get_shortest_path_distance(rover, cal_wp, image_wp)
                                    if dist_cal_to_image is not None:
                                        for comm_wp in self.comm_wps:
                                            dist_image_to_comm = self.get_shortest_path_distance(rover, image_wp, comm_wp)
                                            if dist_image_to_comm is not None:
                                                total_dist = dist_curr_to_cal + dist_cal_to_image + dist_image_to_comm
                                                min_path_curr_to_cal_to_image_to_comm = min(min_path_curr_to_cal_to_image_to_comm, total_dist)

                        if min_path_curr_to_cal_to_image_to_comm != float('inf'):
                            nav_cal_cost = min_path_curr_to_cal_to_image_to_comm + 1 # +1 for calibrate action
                            min_total_task_nav_cost = min(min_total_task_nav_cost, nav_cal_cost)

                if min_total_task_nav_cost == float('inf'): return float('inf') # Cannot take image or communicate
                cost_om += min_total_task_nav_cost

            else: # Image is already taken by taken_by_rover
                # Need to communicate
                cost_om = 1 # communicate action

                # Navigation cost from current location of taken_by_rover to communication point
                min_nav_curr_to_comm = float('inf')
                if not self.comm_wps: return float('inf') # No communication points
                if taken_by_rover not in rover_locations: return float('inf') # Rover location unknown
                rover_at = rover_locations[taken_by_rover]
                for comm_wp in self.comm_wps:
                    dist = self.get_shortest_path_distance(taken_by_rover, rover_at, comm_wp)
                    if dist is not None: min_nav_curr_to_comm = min(min_nav_curr_to_comm, dist)

                if min_nav_curr_to_comm == float('inf'): return float('inf') # Cannot communicate
                cost_om += min_nav_curr_to_comm

            h += cost_om

        return h
