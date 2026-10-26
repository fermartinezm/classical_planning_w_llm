import heapq
import logging
from collections import deque, defaultdict

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


# Helper function to parse PDDL facts (strings)
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding parentheses and split by spaces
    parts = fact_string.strip('()').split()
    return tuple(parts)


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated
    by finding the cheapest sequence of actions (navigate, sample/image,
    communicate) required to achieve it, considering rover capabilities,
    locations, store status, camera calibration, and available samples/targets.
    Navigation costs are estimated using precomputed shortest paths on the
    rover-specific traversability graphs.

    Assumptions:
    - The planning task is a STRIPS task with the predicates and actions
      defined in the rovers domain PDDL.
    - Goal facts are always of the form (communicated_X_data ...).
    - The lander location is static and given by (at_lander ?l ?y).
    - Rover capabilities (equipped_for_...) and camera properties (on_board,
      supports, calibration_target) are static.
    - Waypoint visibility (visible) and rover traversability (can_traverse)
      are static.
    - Objective visibility from waypoints (visible_from) is static.
    - Samples (at_soil_sample, at_rock_sample) are initially present at
      specific waypoints and are consumed by sampling.
    - Stores are initially empty and become full after sampling, requiring
      a drop action to become empty again.
    - Cameras are initially uncalibrated and become calibrated by the calibrate
      action, but become uncalibrated again after the take_image action.
      (Note: The heuristic simplifies calibration cost estimation by only
       adding it if the camera is *currently* uncalibrated in the state,
       ignoring the uncalibration effect of take_image for simplicity and speed).

    Heuristic Initialization:
    The constructor precomputes static information and navigation costs:
    - Parses static facts to build dictionaries for:
        - Lander location.
        - Rover equipment (soil, rock, imaging).
        - Store ownership (store -> rover).
        - Camera ownership (camera -> rover).
        - Camera supported modes (camera -> set of modes).
        - Camera calibration targets (camera -> objective).
        - Objective visibility from waypoints (objective -> set of waypoints).
        - Waypoint visibility graph (for communication waypoints).
        - Rover-specific traversability graphs (for navigation).
    - Identifies communication waypoints (visible from lander location).
    - For each rover, computes all-pairs shortest path distances between
      waypoints using BFS on its traversability graph (filtered by visible links).

    Step-By-Step Thinking for Computing Heuristic (__call__ method):
    1. Initialize total heuristic value `h = 0`.
    2. Get the current state and the set of goal facts.
    3. Identify the set of goal facts that are not currently true in the state.
    4. For each unachieved goal fact `g`:
        a. Initialize the minimum estimated cost for `g` to infinity (`min_cost_g = float('inf')`).
        b. Parse the goal fact string to identify its type (soil, rock, image) and parameters (waypoint, objective, mode).
        c. **If `g` is a soil communication goal `(communicated_soil_data ?w)`:**
            i. Iterate through all rovers `?r`.
            ii. If `?r` is equipped for soil analysis:
                - Check if `(have_soil_analysis ?r ?w)` is already in the state. If yes, cost to get data is 0.
                - If not, estimate cost to sample:
                    - Find the store `?s` for `?r`.
                    - Cost to navigate `?r` from its current location to `?w`. Use precomputed distances. If unreachable, this path is infinite cost.
                    - Check if `(at_soil_sample ?w)` is in the state. If not, this path is infinite cost (sample already taken by another rover or not present initially).
                    - Cost for store: 1 if `(full ?s)` is in the state (for drop action), else 0.
                    - Cost for sample_soil action: 1.
                    - Total cost to get data = Navigation cost + Store cost + Sample action cost.
                - If cost to get data is finite:
                    # Determine the waypoint from which communication will happen.
                    # If data was just sampled, it's `waypoint_data`.
                    # If data was already held, it's the rover's current location (`rover_wp`).
                    data_location_wp = waypoint_data if cost_get_data > 0 else rover_locations.get(rover)
                    if data_location_wp is None: continue # Should not happen if rover_wp was valid

                    # Cost to communicate
                    cost_communicate_path = float('inf')
                    # Need to navigate to a communication waypoint from the data location (data_location_wp)
                    for comm_wp in self.comm_waypoints:
                        nav_cost_to_comm = self.dist[rover][data_location_wp].get(comm_wp, float('inf'))
                        if nav_cost_to_comm == float('inf'): continue

                        communicate_cost = 1
                        total_cost_rover_path = cost_get_data + nav_cost_to_comm + communicate_cost
                        cost_communicate_path = min(cost_communicate_path, total_cost_rover_path) # This min is over comm_waypoints

                    min_cost_g = min(min_cost_g, cost_communicate_path) # This min is over rovers

            elif pred == 'communicated_rock_data':
                if len(parsed_goal) < 2: continue
                waypoint_data = parsed_goal[1]
                # Need to get rock analysis for waypoint_data and communicate it

                for rover in self.rovers:
                    if 'rock' not in self.rover_equipment.get(rover, set()):
                        continue # Rover not equipped for rock analysis

                    rover_wp = rover_locations.get(rover)
                    if rover_wp is None: continue # Should not happen

                    cost_get_data = float('inf')
                    data_location_wp = waypoint_data # Default: location where data was sampled

                    if self._has_rock_analysis(state, rover, waypoint_data):
                        cost_get_data = 0 # Already have the data
                        data_location_wp = rover_wp # If already have it, communicate from current location
                    else:
                        # Need to sample
                        store = self._get_rover_store(rover)
                        if store is None: continue # Should not happen

                        # Cost to navigate to sample location
                        nav_cost_to_sample = self.dist[rover][rover_wp].get(waypoint_data, float('inf'))
                        if nav_cost_to_sample == float('inf'): continue # Cannot reach sample location

                        # Check if sample exists at the waypoint
                        if not self._has_rock_sample_at(state, waypoint_data):
                             # Sample already taken by someone else or not available
                             # This path is blocked for this goal instance
                             continue

                        # Cost for store
                        store_cost = 1 if self._is_store_full(state, store) else 0

                        # Cost for sample action
                        sample_cost = 1

                        cost_get_data = nav_cost_to_sample + store_cost + sample_cost
                        data_location_wp = waypoint_data # Data is obtained at the sample waypoint

                    if cost_get_data == float('inf') or data_location_wp is None: continue

                    # Cost to communicate
                    cost_communicate_path = float('inf')
                    # Need to navigate to a communication waypoint from the data location (data_location_wp)
                    for comm_wp in self.comm_waypoints:
                        nav_cost_to_comm = self.dist[rover][data_location_wp].get(comm_wp, float('inf'))
                        if nav_cost_to_comm == float('inf'): continue

                        communicate_cost = 1
                        total_cost_rover_path = cost_get_data + nav_cost_to_comm + communicate_cost
                        cost_communicate_path = min(cost_communicate_path, total_cost_rover_path) # This min is over comm_waypoints

                    min_cost_g = min(min_cost_g, cost_communicate_path) # This min is over rovers


            elif pred == 'communicated_image_data':
                if len(parsed_goal) < 3: continue
                objective = parsed_goal[1]
                mode = parsed_goal[2]
                # Need to get image for objective/mode and communicate it

                for rover in self.rovers:
                    if 'imaging' not in self.rover_equipment.get(rover, set()):
                        continue # Rover not equipped for imaging

                    rover_wp = rover_locations.get(rover)
                    if rover_wp is None: continue # Should not happen

                    for camera in self.cameras:
                        if self.on_board.get(camera) != rover:
                            continue # Camera not on this rover
                        if mode not in self.supports.get(camera, set()):
                            continue # Camera does not support this mode

                        cost_get_image = float('inf')
                        image_wp_for_comm = None # Waypoint from which communication will happen

                        if self._has_image(state, rover, objective, mode):
                            cost_get_image = 0 # Already have the image
                            # If already have image, the "image waypoint" for communication is the rover's current location
                            image_wp_for_comm = rover_wp
                        else:
                            # Need to take image
                            # Find a suitable waypoint to take the image from
                            possible_image_wps = self.visible_from.get(objective, set())
                            if not possible_image_wps: continue # Cannot see objective from anywhere

                            min_cost_take_image_path = float('inf')
                            best_image_wp_for_take = None # Store the chosen image waypoint for this rover/camera combo

                            # Iterate through possible waypoints to take the image from
                            for image_wp in possible_image_wps:
                                # Cost to navigate to image location
                                nav_cost_to_image_wp = self.dist[rover][rover_wp].get(image_wp, float('inf'))
                                if nav_cost_to_image_wp == float('inf'): continue # Cannot reach image location

                                # Cost to calibrate
                                calibration_cost = 0
                                if not self._is_camera_calibrated(state, camera, rover):
                                    # Need to calibrate
                                    cal_target = self.calibration_target.get(camera)
                                    if cal_target is None: continue # Camera has no calibration target

                                    possible_cal_wps = self.visible_from.get(cal_target, set())
                                    if not possible_cal_wps: continue # Cannot see calibration target from anywhere

                                    min_nav_cost_to_cal_wp = float('inf')
                                    # Find the closest calibration waypoint from the image waypoint
                                    # (assuming we navigate to image_wp first, then calibrate if needed)
                                    for cal_wp in possible_cal_wps:
                                         nav_cost_img_to_cal = self.dist[rover][image_wp].get(cal_wp, float('inf'))
                                         min_nav_cost_to_cal_wp = min(min_nav_cost_to_cal_wp, nav_cost_img_to_cal)

                                    if min_nav_cost_to_cal_wp == float('inf'): continue # Cannot reach any calibration waypoint

                                    calibration_cost = min_nav_cost_to_cal_wp + 1 # +1 for calibrate action

                                # Cost for take_image action
                                take_image_cost = 1

                                current_cost_take_image = nav_cost_to_image_wp + calibration_cost + take_image_cost

                                if current_cost_take_image < min_cost_take_image_path:
                                     min_cost_take_image_path = current_cost_take_image
                                     best_image_wp_for_take = image_wp # Remember the waypoint used for this path

                            cost_get_image = min_cost_take_image_path
                            image_wp_for_comm = best_image_wp_for_take # Use the waypoint from the best path for communication cost

                        if cost_get_image == float('inf') or image_wp_for_comm is None: continue

                        # Cost to communicate
                        cost_communicate_path = float('inf')
                        # Need to navigate to a communication waypoint from the image location (image_wp_for_comm)
                        for comm_wp in self.comm_waypoints:
                            nav_cost_to_comm = self.dist[rover][image_wp_for_comm].get(comm_wp, float('inf'))
                            if nav_cost_to_comm == float('inf'): continue

                            communicate_cost = 1
                            total_cost_rover_camera_path = cost_get_image + nav_cost_to_comm + communicate_cost
                            cost_communicate_path = min(cost_communicate_path, total_cost_rover_camera_path) # This min is over comm_waypoints

                        min_cost_g = min(min_cost_g, cost_communicate_path) # This min is over rovers/cameras/image_wps/cal_wps

            # Add the minimum cost for this goal to the total heuristic
            if min_cost_g == float('inf'):
                 # If any goal is unreachable, the state is a dead end
                 return float('inf')
            h += min_cost_g

        return h

    def _extract_all_objects(self, facts):
        """Extracts all objects and attempts to infer their types from fact structure."""
        objects = {}
        # Map predicate name and argument index to expected type
        type_map = {
            'at': {0: 'rover', 1: 'waypoint'},
            'at_lander': {0: 'lander', 1: 'waypoint'},
            'can_traverse': {0: 'rover', 1: 'waypoint', 2: 'waypoint'},
            'equipped_for_soil_analysis': {0: 'rover'},
            'equipped_for_rock_analysis': {0: 'rover'},
            'equipped_for_imaging': {0: 'rover'},
            'empty': {0: 'store'},
            'have_rock_analysis': {0: 'rover', 1: 'waypoint'},
            'have_soil_analysis': {0: 'rover', 1: 'waypoint'},
            'full': {0: 'store'},
            'calibrated': {0: 'camera', 1: 'rover'},
            'supports': {0: 'camera', 1: 'mode'},
            'visible': {0: 'waypoint', 1: 'waypoint'},
            'have_image': {0: 'rover', 1: 'objective', 2: 'mode'},
            'communicated_soil_data': {0: 'waypoint'},
            'communicated_rock_data': {0: 'waypoint'},
            'communicated_image_data': {0: 'objective', 1: 'mode'},
            'at_soil_sample': {0: 'waypoint'},
            'at_rock_sample': {0: 'waypoint'},
            'visible_from': {0: 'objective', 1: 'waypoint'},
            'store_of': {0: 'store', 1: 'rover'},
            'calibration_target': {0: 'camera', 1: 'objective'},
            'on_board': {0: 'camera', 1: 'rover'},
        }

        for fact_string in facts:
            parsed = parse_fact(fact_string)
            if not parsed: continue

            pred = parsed[0]
            args = parsed[1:]

            if pred in type_map:
                for i, obj_type in type_map[pred].items():
                    if i < len(args):
                        obj_name = args[i]
                        # Assign the inferred type if not already known
                        if obj_name not in objects:
                             objects[obj_name] = obj_type

        return objects


    def _precompute_static_info(self):
        """Parses static facts and stores relevant information."""
        self.lander_location = None
        self.rover_equipment = defaultdict(set) # rover -> {soil, rock, imaging}
        self.store_of = {} # store -> rover
        self.on_board = {} # camera -> rover
        self.supports = defaultdict(set) # camera -> {mode1, mode2, ...}
        self.calibration_target = {} # camera -> objective
        self.visible_from = defaultdict(set) # objective -> {waypoint1, ...}
        self.visible_graph = defaultdict(set) # waypoint -> {waypoint1, ...}
        self.can_traverse_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> {waypoint1, ...}

        for fact_string in self.static_facts:
            parsed = parse_fact(fact_string)
            if not parsed: continue

            pred = parsed[0]
            args = parsed[1:]

            if pred == 'at_lander':
                # Assuming only one lander
                if len(args) >= 2:
                    self.lander_location = args[1]
            elif pred == 'equipped_for_soil_analysis':
                if len(args) >= 1: self.rover_equipment[args[0]].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                if len(args) >= 1: self.rover_equipment[args[0]].add('rock')
            elif pred == 'equipped_for_imaging':
                if len(args) >= 1: self.rover_equipment[args[0]].add('imaging')
            elif pred == 'store_of':
                if len(args) >= 2: self.store_of[args[0]] = args[1]
            elif pred == 'on_board':
                if len(args) >= 2: self.on_board[args[0]] = args[1]
            elif pred == 'supports':
                if len(args) >= 2: self.supports[args[0]].add(args[1])
            elif pred == 'calibration_target':
                if len(args) >= 2: self.calibration_target[args[0]] = args[1]
            elif pred == 'visible_from':
                if len(args) >= 2: self.visible_from[args[0]].add(args[1])
            elif pred == 'visible':
                if len(args) >= 2:
                    self.visible_graph[args[0]].add(args[1])
                    # visible is often symmetric, but not always required by PDDL
                    # self.visible_graph[args[1]].add(args[0]) # Don't assume symmetry
            elif pred == 'can_traverse':
                 # can_traverse ?r ?x ?y requires visible ?x ?y in navigate action
                 # Build the graph using both conditions
                 if len(args) >= 3:
                     rover, wp_from, wp_to = args
                     # Check if the link is also visible
                     if wp_to in self.visible_graph.get(wp_from, set()):
                         self.can_traverse_graph[rover][wp_from].add(wp_to)
                         # can_traverse is not necessarily symmetric
                         # self.can_traverse_graph[rover][wp_to].add(wp_from) # Don't assume symmetry

        # Identify communication waypoints
        self.comm_waypoints = set()
        if self.lander_location:
            # A waypoint is a communication waypoint if the lander location is visible FROM it.
            # The predicate is (visible ?w ?p), meaning ?w is visible from ?p.
            # The communicate action precondition is (visible ?x ?y) where ?x is rover loc, ?y is lander loc.
            # So, a communication waypoint ?x is one where (visible ?x lander_location) is true.
            lander_loc = self.lander_location
            for wp in self.waypoint_objects:
                 if lander_loc in self.visible_graph.get(wp, set()):
                     self.comm_waypoints.add(wp)


    def _precompute_navigation_distances(self):
        """Computes shortest path distances for each rover between all waypoints."""
        self.dist = defaultdict(lambda: defaultdict(lambda: float('inf'))) # rover -> wp_from -> wp_to -> distance

        for rover in self.rovers:
            graph = self.can_traverse_graph.get(rover, {})
            if not graph: # Rover cannot traverse anywhere
                continue

            for start_wp in self.waypoint_objects:
                # BFS from start_wp
                q = deque([(start_wp, 0)])
                visited = {start_wp}
                self.dist[rover][start_wp][start_wp] = 0

                while q:
                    current_wp, current_dist = q.popleft()

                    for neighbor_wp in graph.get(current_wp, set()):
                        if neighbor_wp not in visited:
                            visited.add(neighbor_wp)
                            self.dist[rover][start_wp][neighbor_wp] = current_dist + 1
                            q.append((neighbor_wp, current_dist + 1))

    def _get_rover_location(self, state, rover):
        """Finds the current location of a rover in the state."""
        # Fact is '(at rover_name waypoint_name)'
        target_fact_prefix = f'(at {rover} '
        for fact_string in state:
            if fact_string.startswith(target_fact_prefix):
                 parsed = parse_fact(fact_string)
                 if len(parsed) == 3:
                     return parsed[2]
        return None # Rover location not found (shouldn't happen in valid states)

    def _get_rover_store(self, rover):
        """Finds the store associated with a rover."""
        # Precomputed in __init__
        # Fact is '(store_of store_name rover_name)'
        for store, owner in self.store_of.items():
            if owner == rover:
                return store
        return None # Should not happen if problem is well-formed

    def _is_store_full(self, state, store):
        """Checks if a store is full in the current state."""
        return f'(full {store})' in state

    def _has_soil_sample_at(self, state, waypoint):
        """Checks if a soil sample is present at a waypoint in the current state."""
        return f'(at_soil_sample {waypoint})' in state

    def _has_rock_sample_at(self, state, waypoint):
        """Checks if a rock sample is present at a waypoint in the current state."""
        return f'(at_rock_sample {waypoint})' in state

    def _has_soil_analysis(self, state, rover, waypoint):
        """Checks if a rover has soil analysis data for a waypoint."""
        return f'(have_soil_analysis {rover} {waypoint})' in state

    def _has_rock_analysis(self, state, rover, waypoint):
        """Checks if a rover has rock analysis data for a waypoint."""
        return f'(have_rock_analysis {rover} {waypoint})' in state

    def _has_image(self, state, rover, objective, mode):
        """Checks if a rover has an image for an objective in a specific mode."""
        return f'(have_image {rover} {objective} {mode})' in state

    def _is_camera_calibrated(self, state, camera, rover):
        """Checks if a camera is calibrated on a rover."""
        return f'(calibrated {camera} {rover})' in state
