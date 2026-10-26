from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a predicate and arguments."""
    # Remove outer parentheses and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function to build navigation graph
def build_nav_graph(static_facts, rover_name):
    """Builds a navigation graph (adjacency list) for a specific rover."""
    graph = {}
    for fact_str in static_facts:
        pred, args = parse_fact(fact_str)
        if pred == 'can_traverse' and args[0] == rover_name:
            wp1, wp2 = args[1], args[2]
            if wp1 not in graph:
                graph[wp1] = set()
            if wp2 not in graph:
                graph[wp2] = set()
            graph[wp1].add(wp2)
            # Assuming can_traverse implies symmetric navigation
            graph[wp2].add(wp1)
    return graph

# Helper function to compute shortest paths using BFS
def compute_shortest_paths(graph):
    """Computes shortest path distances from all nodes to all other nodes using BFS."""
    distances = {}
    for start_node in graph:
        distances[start_node] = {}
        q = deque([(start_node, 0)])
        visited = {start_node}
        while q:
            current_node, dist = q.popleft()
            distances[start_node][current_node] = dist
            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
    return distances

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the estimated
    minimum cost for each unachieved goal fact. For each unachieved goal, it
    considers the actions required (navigation, sampling/imaging, communication)
    and finds the minimum cost sequence across all capable rovers. Navigation
    costs are precomputed shortest paths on the waypoint graph for each rover.

    Assumptions:
    - Navigation between adjacent waypoints costs 1 action.
    - Sampling (soil/rock) costs 1 action.
    - Dropping a sample costs 1 action.
    - Calibrating a camera costs 1 action.
    - Taking an image costs 1 action.
    - Communicating data costs 1 action.
    - Samples (soil/rock) are consumed upon sampling.
    - Taking an image uncalibrates the camera.
    - The heuristic estimates the cost for each unachieved goal independently
      and sums these costs. This ignores potential positive interactions
      (e.g., one navigation step might serve multiple goals).
    - If a soil/rock sample is not at its initial location and no rover has it,
      it is assumed to be already communicated or impossible to get from this state
      for the purpose of sampling cost calculation (cost is infinity for sampling path).
    - For image acquisition, the heuristic models the sequence: Navigate to calibration
      waypoint -> Calibrate -> Navigate to image waypoint -> Take Image. It finds the
      minimum cost path through this sequence across all suitable intermediate waypoints.

    Heuristic Initialization:
    The constructor performs static analysis of the task:
    1. Identifies all objects (rovers, waypoints, cameras, objectives, landers, stores, modes)
       by inspecting all facts in the initial state, goals, and static information.
    2. Stores static relationships: rover capabilities (soil, rock, imaging),
       camera information (on-board rover, supported modes, calibration target),
       store ownership by rover.
    3. Builds a navigation graph for each rover based on 'can_traverse' facts.
    4. Computes all-pairs shortest path distances for each rover's navigation graph
       using BFS (since edge costs are uniform).
    5. Stores the lander's location.
    6. Precomputes sets of waypoints visible from the lander, from each objective,
       and from each calibration target, based on 'visible' and 'visible_from' facts.

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify all goal facts that are not true in the current state.
    3. Extract dynamic information from the current state:
       - Current location of each rover.
       - Which stores are full.
       - Which rovers have which soil/rock samples.
       - Which rovers have which images.
       - Which cameras are calibrated on which rovers (Note: This dynamic fact is not strictly used in the image cost sequence calculation, which assumes calibration is needed if the image isn't held, but it's extracted anyway).
       - Which soil/rock samples are currently at their initial waypoints.
    4. Initialize the total heuristic value `h` to 0.
    5. For each unachieved goal fact:
       a. Initialize the minimum cost for this goal `goal_cost` to infinity.
       b. If the goal is `(communicated_soil_data ?w)`:
          - Calculate `cost_if_already_have`: Minimum cost for any soil-equipped rover that *already* has `(have_soil_analysis ?r ?w)` to navigate from its current location to a lander-visible waypoint and communicate (Nav + 1). Minimize over suitable rovers and lander-visible waypoints. If no such rover exists, this cost is infinity.
          - Calculate `cost_if_need_to_sample`: Only possible if `(at_soil_sample ?w)` is true in the state. Minimum cost for any soil-equipped rover to navigate from its current location to `?w`, sample (including potential drop action if store is full), navigate from `?w` to a lander-visible waypoint, and communicate (Nav1 + Drop(0/1) + Sample(1) + Nav2 + Communicate(1)). Minimize over suitable rovers and lander-visible waypoints. If `(at_soil_sample ?w)` is false or no suitable rover exists, this cost is infinity.
          - The minimum cost for this goal is `min(cost_if_already_have, cost_if_need_to_sample)`.
       c. If the goal is `(communicated_rock_data ?w)`:
          - Similar logic to soil data, using rock-equipped rovers and rock samples.
       d. If the goal is `(communicated_image_data ?o ?m)`:
          - Calculate `cost_if_already_have`: Minimum cost for any imaging-equipped rover that *already* has `(have_image ?r ?o ?m)` to navigate from its current location to a lander-visible waypoint and communicate (Nav + 1). Minimize over suitable rovers and lander-visible waypoints. If no such rover exists, this cost is infinity.
          - Calculate `cost_if_need_to_take`: Minimum cost for any imaging-equipped rover with a suitable camera to navigate from its current location to a calibration waypoint, calibrate, navigate to an image waypoint, take the image, navigate to a lander-visible waypoint, and communicate (Nav1 + Calibrate(1) + Nav2 + TakeImage(1) + Nav3 + Communicate(1)). Minimize over suitable rovers, cameras, calibration waypoints, image waypoints, and lander-visible waypoints. If no suitable path/rover/camera exists, this cost is infinity.
          - The minimum cost for this goal is `min(cost_if_already_have, cost_if_need_to_take)`.
       e. If the minimum cost for the current goal is infinity, the state is considered a dead end, and the heuristic returns infinity immediately.
       f. Add the minimum cost for the current goal to the total heuristic `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.task = task # Store task for access to static info

        # --- Static Analysis ---

        # Identify all objects by type based on predicate usage in static/initial/goals
        rover_set = set()
        waypoint_set = set()
        objective_set = set()
        camera_set = set()
        mode_set = set()
        lander_set = set()
        store_set = set()

        all_relevant_facts = task.initial_state | task.goals | task.static

        for fact_str in all_relevant_facts:
            pred, args = parse_fact(fact_str)
            if pred == 'at': rover_set.add(args[0]); waypoint_set.add(args[1])
            elif pred == 'at_lander': lander_set.add(args[0]); waypoint_set.add(args[1])
            elif pred == 'can_traverse': rover_set.add(args[0]); waypoint_set.add(args[1]); waypoint_set.add(args[2])
            elif pred in ('equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'): rover_set.add(args[0])
            elif pred in ('empty', 'full'): store_set.add(args[0])
            elif pred == 'store_of': store_set.add(args[0]); rover_set.add(args[1])
            elif pred == 'calibrated': camera_set.add(args[0]); rover_set.add(args[1])
            elif pred == 'supports': camera_set.add(args[0]); mode_set.add(args[1])
            elif pred == 'visible': waypoint_set.add(args[0]); waypoint_set.add(args[1])
            elif pred in ('have_rock_analysis', 'have_soil_analysis'): rover_set.add(args[0]); waypoint_set.add(args[1])
            elif pred == 'have_image': rover_set.add(args[0]); objective_set.add(args[1]); mode_set.add(args[2])
            elif pred in ('communicated_soil_data', 'communicated_rock_data'): waypoint_set.add(args[0])
            elif pred == 'communicated_image_data': objective_set.add(args[0]); mode_set.add(args[1])
            elif pred in ('at_soil_sample', 'at_rock_sample'): waypoint_set.add(args[0])
            elif pred == 'visible_from': objective_set.add(args[0]); waypoint_set.add(args[1]) # objective_set includes calibration targets
            elif pred == 'calibration_target': camera_set.add(args[0]); objective_set.add(args[1]) # objective_set includes calibration targets
            elif pred == 'on_board': camera_set.add(args[0]); rover_set.add(args[1])

        self.rovers = list(rover_set)
        self.waypoints = list(waypoint_set)
        self.objectives = list(objective_set)
        self.cameras = list(camera_set)
        self.modes = list(mode_set)
        self.landers = list(lander_set)
        self.stores = list(store_set)

        # Store static information
        self.rover_capabilities = {r: {'soil': False, 'rock': False, 'imaging': False} for r in self.rovers}
        self.camera_info = {c: {'on_board_rover': None, 'supports_modes': set(), 'calibration_target': None} for c in self.cameras}
        self.rover_stores = {r: None for r in self.rovers} # Maps rover to its store

        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'equipped_for_soil_analysis': self.rover_capabilities[args[0]]['soil'] = True
            elif pred == 'equipped_for_rock_analysis': self.rover_capabilities[args[0]]['rock'] = True
            elif pred == 'equipped_for_imaging': self.rover_capabilities[args[0]]['imaging'] = True
            elif pred == 'store_of': self.rover_stores[args[1]] = args[0]
            elif pred == 'on_board': self.camera_info[args[0]]['on_board_rover'] = args[1]
            elif pred == 'supports': self.camera_info[args[0]]['supports_modes'].add(args[1])
            elif pred == 'calibration_target': self.camera_info[args[0]]['calibration_target'] = args[1]

        # Build navigation graphs and compute shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover in self.rovers:
            graph = build_nav_graph(task.static, rover)
            self.rover_shortest_paths[rover] = compute_shortest_paths(graph)

        # Store lander location (found in initial state)
        lander_fact = next(f for f in task.initial_state if f.startswith('(at_lander '))
        _, lander_args = parse_fact(lander_fact)
        self.lander_location = lander_args[1]

        # Precompute lander visible waypoints
        self.lander_visible_waypoints = set()
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'visible':
                wp1, wp2 = args[0], args[1]
                if wp1 == self.lander_location: self.lander_visible_waypoints.add(wp2)
                if wp2 == self.lander_location: self.lander_visible_waypoints.add(wp1)

        # Precompute objective visible waypoints
        self.objective_visible_waypoints = {o: set() for o in self.objectives}
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'visible_from':
                obj = args[0]
                wp = args[1]
                if obj in self.objectives:
                     self.objective_visible_waypoints[obj].add(wp)

        # Precompute calibration target visible waypoints
        self.calibration_target_visible_waypoints = {o: set() for o in self.objectives} # Calibration targets are objectives
        calibration_targets = set(info['calibration_target'] for info in self.camera_info.values() if info['calibration_target'])
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'visible_from':
                obj_or_target = args[0]
                wp = args[1]
                if obj_or_target in calibration_targets:
                     self.calibration_target_visible_waypoints[obj_or_target].add(wp)

    def get_navigation_cost(self, rover, start_wp, end_wp):
        """Returns the shortest path distance between two waypoints for a rover."""
        if rover not in self.rover_shortest_paths:
            return math.inf # Rover doesn't exist or no nav graph built
        if start_wp not in self.rover_shortest_paths[rover]:
             # Start waypoint not in graph (e.g., rover location is unknown or invalid)
             return math.inf
        return self.rover_shortest_paths[rover].get(end_wp, math.inf)

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        unachieved_goals = [g for g in self.goals if g not in state]

        # Extract dynamic information from the current state
        rover_locations = {}
        state_stores_full_set = set()
        rover_has_soil = {} # rover -> {waypoint -> bool}
        rover_has_rock = {} # rover -> {waypoint -> bool}
        rover_has_image = {} # rover -> {objective -> {mode -> bool}}
        # rover_calibrated_cameras = {} # rover -> {camera} # Not strictly needed for current image cost model

        soil_samples_at = set() # waypoint -> bool
        rock_samples_at = set() # waypoint -> bool

        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at':
                rover_locations[args[0]] = args[1]
            elif pred == 'full':
                state_stores_full_set.add(args[0])
            elif pred == 'have_soil_analysis':
                if args[0] not in rover_has_soil: rover_has_soil[args[0]] = {}
                rover_has_soil[args[0]][args[1]] = True
            elif pred == 'have_rock_analysis':
                if args[0] not in rover_has_rock: rover_has_rock[args[0]] = {}
                rover_has_rock[args[0]][args[1]] = True
            elif pred == 'have_image':
                if args[0] not in rover_has_image: rover_has_image[args[0]] = {}
                if args[1] not in rover_has_image[args[0]]: rover_has_image[args[0]][args[1]] = {}
                rover_has_image[args[0]][args[1]][args[2]] = True
            # elif pred == 'calibrated':
            #     if args[1] not in rover_calibrated_cameras: rover_calibrated_cameras[args[1]] = set()
            #     rover_calibrated_cameras[args[1]].add(args[0])
            elif pred == 'at_soil_sample':
                soil_samples_at.add(args[0])
            elif pred == 'at_rock_sample':
                rock_samples_at.add(args[0])

        # Determine which rovers have full stores based on dynamic state and static store info
        rover_stores_full = {} # rover -> bool
        for rover, store in self.rover_stores.items():
            if store is not None and store in state_stores_full_set:
                rover_stores_full[rover] = True


        total_cost = 0

        # Calculate cost for each unachieved goal
        for goal_str in unachieved_goals:
            pred, args = parse_fact(goal_str)
            goal_cost = math.inf

            if pred == 'communicated_soil_data':
                waypoint = args[0]

                # Option 1: A rover already has the sample
                cost_if_already_have = math.inf
                for rover in self.rovers:
                    if rover in rover_has_soil and waypoint in rover_has_soil[rover]:
                        current_rover_loc = rover_locations.get(rover)
                        if current_rover_loc is None: continue

                        min_nav_to_lander_visible = math.inf
                        if self.lander_visible_waypoints:
                            for lander_wp in self.lander_visible_waypoints:
                                nav_cost = self.get_navigation_cost(rover, current_rover_loc, lander_wp)
                                if nav_cost != math.inf:
                                    min_nav_to_lander_visible = min(min_nav_to_lander_visible, nav_cost)

                        if min_nav_to_lander_visible != math.inf:
                             cost_if_already_have = min(cost_if_already_have, min_nav_to_lander_visible + 1) # nav + communicate

                # Option 2: Need to sample the soil
                cost_if_need_to_sample = math.inf
                if waypoint in soil_samples_at: # Can only sample if sample is present
                    for rover in self.rovers:
                        if not self.rover_capabilities[rover]['soil']:
                            continue # Rover cannot do soil analysis

                        current_rover_loc = rover_locations.get(rover)
                        if current_rover_loc is None: continue

                        # Cost to get sample: Nav to waypoint + Drop (if full) + Sample
                        nav_cost_to_sample = self.get_navigation_cost(rover, current_rover_loc, waypoint)
                        if nav_cost_to_sample == math.inf: continue

                        drop_cost = 1 if rover in rover_stores_full and rover_stores_full[rover] else 0
                        cost_to_get_sample = nav_cost_to_sample + drop_cost + 1 # nav + drop + sample
                        rover_loc_after_sample = waypoint # Rover is at the sample waypoint after sampling

                        # Cost to communicate: Nav from sample waypoint to lander-visible waypoint + Communicate
                        cost_to_communicate = math.inf
                        min_nav_to_lander_visible = math.inf
                        if self.lander_visible_waypoints:
                            for lander_wp in self.lander_visible_waypoints:
                                nav_cost = self.get_navigation_cost(rover, rover_loc_after_sample, lander_wp)
                                if nav_cost != math.inf:
                                    min_nav_to_lander_visible = min(min_nav_to_lander_visible, nav_cost)

                        if min_nav_to_lander_visible != math.inf:
                             cost_to_communicate = min_nav_to_lander_visible + 1 # nav + communicate

                        if cost_to_communicate != math.inf:
                             cost_if_need_to_sample = min(cost_if_need_to_sample, cost_to_get_sample + cost_to_communicate)

                # Minimum cost for this goal is the minimum of the two options
                goal_cost = min(cost_if_already_have, cost_if_need_to_sample)

            elif pred == 'communicated_rock_data':
                waypoint = args[0]

                # Option 1: A rover already has the sample
                cost_if_already_have = math.inf
                for rover in self.rovers:
                    if rover in rover_has_rock and waypoint in rover_has_rock[rover]:
                        current_rover_loc = rover_locations.get(rover)
                        if current_rover_loc is None: continue

                        min_nav_to_lander_visible = math.inf
                        if self.lander_visible_waypoints:
                            for lander_wp in self.lander_visible_waypoints:
                                nav_cost = self.get_navigation_cost(rover, current_rover_loc, lander_wp)
                                if nav_cost != math.inf:
                                    min_nav_to_lander_visible = min(min_nav_to_lander_visible, nav_cost)

                        if min_nav_to_lander_visible != math.inf:
                             cost_if_already_have = min(cost_if_already_have, min_nav_to_lander_visible + 1) # nav + communicate

                # Option 2: Need to sample the rock
                cost_if_need_to_sample = math.inf
                if waypoint in rock_samples_at: # Can only sample if sample is present
                    for rover in self.rovers:
                        if not self.rover_capabilities[rover]['rock']:
                            continue # Rover cannot do rock analysis

                        current_rover_loc = rover_locations.get(rover)
                        if current_rover_loc is None: continue

                        # Cost to get sample: Nav to waypoint + Drop (if full) + Sample
                        nav_cost_to_sample = self.get_navigation_cost(rover, current_rover_loc, waypoint)
                        if nav_cost_to_sample == math.inf: continue

                        drop_cost = 1 if rover in rover_stores_full and rover_stores_full[rover] else 0
                        cost_to_get_sample = nav_cost_to_sample + drop_cost + 1 # nav + drop + sample
                        rover_loc_after_sample = waypoint # Rover is at the sample waypoint after sampling

                        # Cost to communicate: Nav from sample waypoint to lander-visible waypoint + Communicate
                        cost_to_communicate = math.inf
                        min_nav_to_lander_visible = math.inf
                        if self.lander_visible_waypoints:
                            for lander_wp in self.lander_visible_waypoints:
                                nav_cost = self.get_navigation_cost(rover, rover_loc_after_sample, lander_wp)
                                if nav_cost != math.inf:
                                    min_nav_to_lander_visible = min(min_nav_to_lander_visible, nav_cost)

                        if min_nav_to_lander_visible != math.inf:
                             cost_to_communicate = min_nav_to_lander_visible + 1 # nav + communicate

                        if cost_to_communicate != math.inf:
                             cost_if_need_to_sample = min(cost_if_need_to_sample, cost_to_get_sample + cost_to_communicate)

                # Minimum cost for this goal is the minimum of the two options
                goal_cost = min(cost_if_already_have, cost_if_need_to_sample)

            elif pred == 'communicated_image_data':
                objective = args[0]
                mode = args[1]

                # Option 1: A rover already has the image
                cost_if_already_have = math.inf
                for rover in self.rovers:
                    if rover in rover_has_image and objective in rover_has_image[rover] and mode in rover_has_image[rover][objective]:
                        current_rover_loc = rover_locations.get(rover)
                        if current_rover_loc is None: continue

                        min_nav_to_lander_visible = math.inf
                        if self.lander_visible_waypoints:
                            for lander_wp in self.lander_visible_waypoints:
                                nav_cost = self.get_navigation_cost(rover, current_rover_loc, lander_wp)
                                if nav_cost != math.inf:
                                    min_nav_to_lander_visible = min(min_nav_to_lander_visible, nav_cost)

                        if min_nav_to_lander_visible != math.inf:
                             cost_if_already_have = min(cost_if_already_have, min_nav_to_lander_visible + 1) # nav + communicate

                # Option 2: Need to take the image
                cost_if_need_to_take = math.inf
                for rover in self.rovers:
                    if not self.rover_capabilities[rover]['imaging']:
                        continue # Rover cannot do imaging

                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None: continue

                    # Find cameras on this rover that support the mode and have a calibration target
                    suitable_cameras = [
                        cam for cam, info in self.camera_info.items()
                        if info['on_board_rover'] == rover and mode in info['supports_modes'] and info['calibration_target'] is not None
                    ]

                    if not suitable_cameras:
                        continue # Rover has no suitable camera with calibration target

                    for camera in suitable_cameras:
                        camera_info = self.camera_info[camera]
                        cal_target = camera_info['calibration_target']

                        cal_wps = self.calibration_target_visible_waypoints.get(cal_target, set())
                        image_wps = self.objective_visible_waypoints.get(objective, set())

                        if not cal_wps or not image_wps:
                            continue # Cannot see calibration target or objective from anywhere

                        min_path_cost_for_rover_camera = math.inf
                        best_final_image_wp = None # To track where the image is taken

                        # Iterate through all possible calibration and image waypoints
                        for cal_wp in cal_wps:
                            nav1 = self.get_navigation_cost(rover, current_rover_loc, cal_wp)
                            if nav1 == math.inf: continue

                            for image_wp in image_wps:
                                nav2 = self.get_navigation_cost(rover, cal_wp, image_wp)
                                if nav2 == math.inf: continue

                                # Cost to get image: Nav(current, cal_wp) + Calibrate + Nav(cal_wp, image_wp) + TakeImage
                                cost_to_get_image_path = nav1 + 1 + nav2 + 1

                                if cost_to_get_image_path < min_path_cost_for_rover_camera:
                                    min_path_cost_for_rover_camera = cost_to_get_image_path
                                    best_final_image_wp = image_wp # Store the image waypoint

                        # If we found a path to take the image
                        if min_path_cost_for_rover_camera != math.inf:
                            rover_loc_after_image = best_final_image_wp

                            # Cost to communicate from rover_loc_after_image
                            cost_to_communicate = math.inf
                            min_nav_to_lander_visible = math.inf
                            if self.lander_visible_waypoints:
                                for lander_wp in self.lander_visible_waypoints:
                                    nav_cost = self.get_navigation_cost(rover, rover_loc_after_image, lander_wp)
                                    if nav_cost != math.inf:
                                        min_nav_to_lander_visible = min(min_nav_to_lander_visible, nav_cost)

                            if min_nav_to_lander_visible != math.inf:
                                 cost_to_communicate = min_nav_to_lander_visible + 1 # nav + communicate

                            # Total cost for this rover/camera path
                            if cost_to_communicate != math.inf:
                                cost_if_need_to_take = min(cost_if_need_to_take, min_path_cost_for_rover_camera + cost_to_communicate)


                # Minimum cost for this goal is the minimum of the two options
                goal_cost = min(cost_if_already_have, cost_if_need_to_take)


            # If any goal is impossible from this state, the heuristic is infinity
            if goal_cost == math.inf:
                return math.inf

            total_cost += goal_cost

        return total_cost
