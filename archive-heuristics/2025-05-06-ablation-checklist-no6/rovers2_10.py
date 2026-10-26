from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each uncommunicated goal fact. The cost for each goal is estimated
    based on whether the required data (sample/image) is already collected
    and includes simplified action costs (sample, calibrate, take_image,
    communicate, drop) and navigation costs (shortest path distance via BFS).
    It finds the minimum cost over suitable rovers/cameras for each goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Extract Static Information ---
        self.lander_location = None
        self.rover_capabilities = {} # {rover_name: {'soil', 'rock', 'imaging'}}
        self.rover_stores = {} # {rover_name: store_name}
        self.rover_cameras = {} # {rover_name: {camera_name, ...}}
        self.camera_modes = {} # {camera_name: {mode_name, ...}}
        self.camera_targets = {} # {camera_name: objective_name}
        self.objective_visible_from = {} # {objective_name: {waypoint_name, ...}}
        self.waypoint_visibility = set() # {(w1, w2), ...}
        self.rover_traversal = {} # {rover_name: {(w1, w2), ...}}

        for fact in task.static:
            parts = self._get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at_lander':
                self.lander_location = parts[2] # Assuming only one lander
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif pred == 'calibration_target':
                camera, target = parts[1], parts[2]
                self.camera_targets[camera] = target
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)
            elif pred == 'visible':
                w1, w2 = parts[1], parts[2]
                self.waypoint_visibility.add((w1, w2))
            elif pred == 'can_traverse':
                rover, w1, w2 = parts[1], parts[2], parts[3]
                self.rover_traversal.setdefault(rover, set()).add((w1, w2))

        # Find lander visible waypoints
        self.lander_visible_waypoints = set()
        if self.lander_location:
            # A waypoint w1 is lander-visible if (visible w1 lander_location) is true
            self.lander_visible_waypoints = {w1 for w1, w2 in self.waypoint_visibility if w2 == self.lander_location}


        # Precompute shortest path distances for each rover
        self.rover_distances = {} # {rover_name: {start_wp: {end_wp: distance, ...}, ...}}

        # Get all unique waypoints mentioned in can_traverse for each rover
        rover_waypoints_map = {}
        for rover, traversals in self.rover_traversal.items():
             rover_waypoints_map[rover] = set()
             for w1, w2 in traversals:
                 rover_waypoints_map[rover].add(w1)
                 rover_waypoints_map[rover].add(w2)

        for rover, traversals in self.rover_traversal.items():
            graph = {}
            for w1, w2 in traversals:
                graph.setdefault(w1, set()).add(w2)

            self.rover_distances[rover] = {}
            rover_waypoints = rover_waypoints_map.get(rover, set())

            for start_wp in rover_waypoints:
                self.rover_distances[rover][start_wp] = {}
                queue = deque([(start_wp, 0)])
                visited = {start_wp}
                while queue:
                    (current_wp, dist) = queue.popleft()
                    self.rover_distances[rover][start_wp][current_wp] = dist

                    for neighbor in graph.get(current_wp, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

    def _get_parts(self, fact_string):
        """Extract predicate and arguments from a fact string like '(at rover1 waypoint1)'."""
        # Remove parentheses and split by space
        return fact_string[1:-1].split()

    def _match_fact(self, fact_string, predicate, *args):
        """Check if a PDDL fact string matches a given pattern."""
        parts = self._get_parts(fact_string)
        if not parts or parts[0] != predicate:
            return False
        # Check arguments, allowing wildcards '*'
        if len(parts) - 1 != len(args):
            return False
        return all(fnmatch(part, arg) for part, arg in zip(parts[1:], args))

    def _get_distance(self, rover, start_wp, end_wp):
        """Returns shortest distance or infinity if unreachable for a specific rover."""
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover] or end_wp not in self.rover_distances[rover][start_wp]:
            return float('inf')
        return self.rover_distances[rover][start_wp][end_wp]

    def _min_distance_to_set(self, rover, start_wp, target_wps):
        """Returns min shortest distance from start_wp to any waypoint in target_wps for a rover."""
        min_dist = float('inf')
        if rover not in self.rover_distances or start_wp not in self.rover_distances[rover]:
            return min_dist # Rover or start_wp not in graph

        distances_from_start = self.rover_distances[rover][start_wp]
        for target_wp in target_wps:
            if target_wp in distances_from_start:
                min_dist = min(min_dist, distances_from_start[target_wp])
        return min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        current_rover_locations = {} # {rover_name: waypoint_name}
        have_soil = set() # {(rover, waypoint), ...}
        have_rock = set() # {(rover, waypoint), ...}
        have_image = set() # {(rover, objective, mode), ...}
        full_stores = set() # {store_name, ...}
        calibrated_cameras = set() # {(camera, rover), ...}
        current_soil_samples = set() # {waypoint, ...}
        current_rock_samples = set() # {waypoint, ...}


        for fact in state:
            parts = self._get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                # Assuming only rovers have 'at' facts in the state that change
                if obj in self.rover_capabilities: # Check if obj is a rover
                     current_rover_locations[obj] = loc
            elif pred == 'have_soil_analysis':
                have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis':
                have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image':
                have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'full':
                full_stores.add(parts[1])
            elif pred == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'at_soil_sample':
                current_soil_samples.add(parts[1])
            elif pred == 'at_rock_sample':
                current_rock_samples.add(parts[1])

        total_cost = 0

        # --- Estimate Cost for Each Ungoaled Fact ---
        for goal_fact_string in self.goals:
            if goal_fact_string in state:
                continue # Goal already achieved

            parts = self._get_parts(goal_fact_string)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint = parts[1]
                # Find rovers that have the analysis
                rovers_with_analysis = {r for r, w in have_soil if w == waypoint}

                if rovers_with_analysis:
                    # Need to communicate. Find the best rover among those with analysis.
                    min_comm_cost = float('inf')
                    for rover in rovers_with_analysis:
                        current_wp = current_rover_locations.get(rover)
                        if current_wp is None: continue # Should not happen in valid states

                        # Need to reach a lander-visible waypoint
                        nav_cost = self._min_distance_to_set(rover, current_wp, self.lander_visible_waypoints)
                        if nav_cost != float('inf'):
                            min_comm_cost = min(min_comm_cost, nav_cost + 1) # +1 for communicate action

                    total_cost += min_comm_cost
                    if min_comm_cost == float('inf'): return float('inf') # Cannot communicate

                else:
                    # Need to sample and communicate. Find the best equipped rover.
                    soil_rovers = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps}
                    if not soil_rovers: return float('inf') # No rover can sample soil

                    if waypoint not in current_soil_samples:
                         return float('inf') # Sample is gone, and no one has the analysis

                    min_sample_comm_cost = float('inf')
                    for rover in soil_rovers:
                        current_wp = current_rover_locations.get(rover)
                        if current_wp is None: continue

                        store = self.rover_stores.get(rover)
                        drop_cost = 1 if store in full_stores else 0 # Cost to drop if store is full

                        # Cost = drop + move to sample + sample + move to lander + communicate
                        nav_to_sample = self._get_distance(rover, current_wp, waypoint)
                        if nav_to_sample == float('inf'): continue # Cannot reach sample waypoint

                        nav_to_lander = self._min_distance_to_set(rover, waypoint, self.lander_visible_waypoints)
                        if nav_to_lander == float('inf'): continue # Cannot reach lander-visible from sample waypoint

                        cost = drop_cost + nav_to_sample + 1 + nav_to_lander + 1 # +1 for sample, +1 for communicate
                        min_sample_comm_cost = min(min_sample_comm_cost, cost)

                    total_cost += min_sample_comm_cost
                    if min_sample_comm_cost == float('inf'): return float('inf') # Cannot sample/communicate


            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                # Find rovers that have the analysis
                rovers_with_analysis = {r for r, w in have_rock if w == waypoint}

                if rovers_with_analysis:
                    # Need to communicate. Find the best rover among those with analysis.
                    min_comm_cost = float('inf')
                    for rover in rovers_with_analysis:
                        current_wp = current_rover_locations.get(rover)
                        if current_wp is None: continue
                        nav_cost = self._min_distance_to_set(rover, current_wp, self.lander_visible_waypoints)
                        if nav_cost != float('inf'):
                            min_comm_cost = min(min_comm_cost, nav_cost + 1) # +1 for communicate action
                    total_cost += min_comm_cost
                    if min_comm_cost == float('inf'): return float('inf') # Cannot communicate

                else:
                    # Need to sample and communicate. Find the best equipped rover.
                    rock_rovers = {r for r, caps in self.rover_capabilities.items() if 'rock' in caps}
                    if not rock_rovers: return float('inf') # No rover can sample rock

                    if waypoint not in current_rock_samples:
                         return float('inf') # Sample is gone, and no one has the analysis

                    min_sample_comm_cost = float('inf')
                    for rover in rock_rovers:
                        current_wp = current_rover_locations.get(rover)
                        if current_wp is None: continue

                        store = self.rover_stores.get(rover)
                        drop_cost = 1 if store in full_stores else 0 # Cost to drop if store is full

                        # Cost = drop + move to sample + sample + move to lander + communicate
                        nav_to_sample = self._get_distance(rover, current_wp, waypoint)
                        if nav_to_sample == float('inf'): continue

                        nav_to_lander = self._min_distance_to_set(rover, waypoint, self.lander_visible_waypoints)
                        if nav_to_lander == float('inf'): continue

                        cost = drop_cost + nav_to_sample + 1 + nav_to_lander + 1 # +1 for sample, +1 for communicate
                        min_sample_comm_cost = min(min_sample_comm_cost, cost)

                    total_cost += min_sample_comm_cost
                    if min_sample_comm_cost == float('inf'): return float('inf') # Cannot sample/communicate


            elif pred == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                # Find rovers that have the image
                rovers_with_image = {r for r, o, m in have_image if o == objective and m == mode}

                if rovers_with_image:
                    # Need to communicate. Find the best rover among those with the image.
                    min_comm_cost = float('inf')
                    for rover in rovers_with_image:
                        current_wp = current_rover_locations.get(rover)
                        if current_wp is None: continue
                        nav_cost = self._min_distance_to_set(rover, current_wp, self.lander_visible_waypoints)
                        if nav_cost != float('inf'):
                            min_comm_cost = min(min_comm_cost, nav_cost + 1) # +1 for communicate action
                    total_cost += min_comm_cost
                    if min_comm_cost == float('inf'): return float('inf') # Cannot communicate

                else:
                    # Need to take image and communicate. Find the best suitable rover/camera.
                    suitable_options = [] # List of (rover, camera, calib_wps, image_wps)
                    for rover, caps in self.rover_capabilities.items():
                        if 'imaging' in caps:
                            for camera in self.rover_cameras.get(rover, set()):
                                if mode in self.camera_modes.get(camera, set()):
                                    target = self.camera_targets.get(camera)
                                    calib_wps = self.objective_visible_from.get(target, set()) if target else set()
                                    image_wps = self.objective_visible_from.get(objective, set())
                                    # Must have waypoints to calibrate target and view objective
                                    if calib_wps and image_wps:
                                        suitable_options.append((rover, camera, calib_wps, image_wps))

                    if not suitable_options: return float('inf') # No rover/camera can take this image

                    min_image_comm_cost = float('inf')
                    for rover, camera, calib_wps, image_wps in suitable_options:
                        current_wp = current_rover_locations.get(rover)
                        if current_wp is None: continue

                        # Estimate cost: (calibrate action if needed) + move(current->calib/image) + (move(calib->image) if calib needed) + take_image + move(image->lander) + communicate

                        calib_action_cost = 0
                        nav_cost_total = float('inf')

                        if (camera, rover) in calibrated_cameras:
                            # Camera is calibrated. Path: current -> image_wp -> lander_wp
                            nav1 = self._min_distance_to_set(rover, current_wp, image_wps)
                            if nav1 == float('inf'): continue

                            # Find min dist from any image_wp to any lander_wp reachable by rover
                            nav2 = float('inf')
                            for img_wp in image_wps:
                                dist_img_lander = self._min_distance_to_set(rover, img_wp, self.lander_visible_waypoints)
                                if dist_img_lander != float('inf'):
                                    nav2 = min(nav2, dist_img_lander)
                            if nav2 == float('inf'): continue

                            nav_cost_total = nav1 + nav2
                            cost = nav_cost_total + 1 + 1 # +1 take_image, +1 communicate

                        else:
                            # Camera is NOT calibrated. Path: current -> calib_wp -> image_wp -> lander_wp
                            calib_action_cost = 1 # Calibrate action needed

                            min_nav_path = float('inf')
                            for calib_wp in calib_wps:
                                dist_curr_calib = self._get_distance(rover, current_wp, calib_wp)
                                if dist_curr_calib == float('inf'): continue
                                for image_wp in image_wps:
                                    dist_calib_image = self._get_distance(rover, calib_wp, image_wp)
                                    if dist_calib_image == float('inf'): continue
                                    dist_image_lander = self._min_distance_to_set(rover, image_wp, self.lander_visible_waypoints)
                                    if dist_image_lander != float('inf'):
                                        min_nav_path = min(min_nav_path, dist_curr_calib + dist_calib_image + dist_image_lander)

                            if min_nav_path == float('inf'): continue
                            nav_cost_total = min_nav_path
                            cost = nav_cost_total + calib_action_cost + 1 + 1 # +1 calibrate, +1 take_image, +1 communicate

                        min_image_comm_cost = min(min_image_comm_cost, cost)

                    total_cost += min_image_comm_cost
                    if min_image_comm_cost == float('inf'): return float('inf') # Cannot take image/communicate


        # Return the total accumulated cost
        return total_cost
