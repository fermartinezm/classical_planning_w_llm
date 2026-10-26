from collections import defaultdict, deque
import math

class roversHeuristic:
    """
    Summary:
    This heuristic estimates the cost to reach a goal state in the rovers domain
    by summing the estimated costs for each unachieved goal predicate. It is
    domain-dependent and non-admissible, designed for greedy best-first search.
    The cost for each unachieved goal is estimated as the minimum number of
    actions required to achieve that specific goal, considering navigation costs
    and necessary intermediate actions (sampling, calibration, taking image, dropping).
    Navigation costs are precomputed using BFS on the rover traversal graphs.

    Assumptions:
    - The heuristic assumes that samples (soil/rock) are only collected once from a waypoint.
      If `at_soil_sample` or `at_rock_sample` is false for a waypoint, the sample
      is assumed to have been collected and is either held by a rover or has
      already been communicated (in which case the goal would be achieved).
    - When a rover needs to sample and its store is full, the heuristic adds a
      cost of 1 for a `drop` action, assuming any sample can be dropped. It does
      not track which specific sample is in the store or if dropping it is
      detrimental to another goal.
    - Taking an image always uncalibrates the camera, so calibration is typically
      needed before each image action unless the camera is already calibrated
      in the current state.
    - The heuristic sums the costs for individual goals, ignoring potential
      synergies (e.g., navigating to a waypoint that helps achieve multiple goals)
      or conflicts (e.g., store capacity limits, camera calibration state affecting
      multiple image goals for the same rover/camera).
    - Unreachable goals (e.g., no equipped rover, no visible waypoint for required
      actions) are assigned an infinite cost, making the total heuristic infinite
      if any unachieved goal is unreachable.

    Heuristic Initialization:
    The constructor (`__init__`) performs the following steps:
    1. Parses static facts from `task.static` to extract:
       - Lander location.
       - Rover equipment (soil, rock, imaging capabilities).
       - Store ownership for each rover.
       - Camera information (on-board rover, calibration target, supported modes).
       - Visibility between waypoints (`visible` predicate) to build the communication graph.
       - Visibility of objectives from waypoints (`visible_from` predicate) to identify imaging and calibration waypoints.
       - Rover traversal capabilities (`can_traverse` predicate) to build traversal graphs for each rover.
       - Collects sets of all objects by type (rovers, waypoints, cameras, etc.).
    2. For each rover, it computes all-pairs shortest path distances between all waypoints
       using BFS on the rover's traversal graph. These distances are stored in `self.rover_distances`.
       Unreachable waypoints have a distance of `math.inf`.
    3. Identifies the set of communication waypoints (`self.comm_wps`) which are those visible
       from the lander's location.
    4. For each rover and each waypoint, it precomputes the minimum distance from that
       waypoint to any communication waypoint reachable by that rover. This is stored
       in `self.min_dist_to_comm`.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic function (`__call__`) computes the heuristic value for a given state:
    1. It first checks if the state is a goal state. If yes, the heuristic is 0.
    2. It parses the dynamic facts from the current `state` to get:
       - Current location of each rover.
       - Status of each store (empty/full).
       - Samples (soil/rock) held by each rover.
       - Images held by each rover.
       - Calibrated cameras.
       - Locations of remaining soil/rock samples on waypoints.
       - Achieved goals.
    3. It identifies the set of unachieved goals by subtracting achieved goals from `task.goals`.
    4. It initializes the total heuristic value `h` to 0.
    5. It iterates through each unachieved goal:
       - For a `communicated_soil_data ?w` goal:
         - Finds rovers equipped for soil analysis. If none, the goal is unreachable (cost `inf`).
         - For each equipped rover:
           - If the rover already has the sample `(have_soil_analysis ?r ?w)`:
             - The cost is 1 (communicate) + minimum navigation cost from the rover's current location to any communication waypoint.
           - If the rover does not have the sample:
             - Checks if the sample is still at waypoint `w` (`at_soil_sample ?w`). If not, the goal might be unreachable via sampling (cost `inf` for this path).
             - If the sample is at `w`: The cost involves navigating to `w`, sampling, navigating to a communication waypoint, and communicating. This is estimated as: navigation cost from current location to `w` + 1 (sample action) + (1 if store is full for drop action) + minimum navigation cost from `w` to any communication waypoint + 1 (communicate action).
         - The minimum cost over all equipped rovers is taken as the cost for this goal.
       - For a `communicated_rock_data ?w` goal:
         - The logic is analogous to the soil data goal, using rock analysis equipment and rock samples.
       - For a `communicated_image_data ?o ?m` goal:
         - Finds rovers equipped for imaging with cameras on board that support mode `m`. If none, the goal is unreachable (cost `inf`).
         - Identifies waypoints visible from objective `o` (`image_wps`). If none, the goal is unreachable (cost `inf`).
         - For each suitable rover/camera pair:
           - If the rover already has the image `(have_image ?r ?o ?m)`:
             - The cost is 1 (communicate) + minimum navigation cost from the rover's current location to any communication waypoint.
           - If the rover does not have the image:
             - Checks if the camera is calibrated `(calibrated ?i ?r)`.
             - If calibrated: The cost involves navigating to an image waypoint, taking the image, navigating to a communication waypoint, and communicating. This is estimated as: minimum over all image waypoints `p` of (navigation cost from current location to `p` + minimum navigation cost from `p` to any communication waypoint) + 1 (take image action) + 1 (communicate action).
             - If not calibrated: The cost involves navigating to a calibration waypoint, calibrating, navigating to an image waypoint, taking the image, navigating to a communication waypoint, and communicating. This is estimated as: minimum over all calibration waypoints `w` and image waypoints `p` of (navigation cost from current location to `w` + navigation cost from `w` to `p` + minimum navigation cost from `p` to any communication waypoint) + 1 (calibrate action) + 1 (take image action) + 1 (communicate action).
         - The minimum cost over all suitable rover/camera pairs is taken as the cost for this goal.
    6. If the minimum cost for any unachieved goal is `math.inf`, the total heuristic `h` is set to `math.inf`.
    7. Otherwise, the minimum costs for all unachieved goals are summed to get the total heuristic `h`.
    8. The total heuristic value `h` is returned.
    """
    def __init__(self, task):
        # Parse static info
        self.lander_location = None
        self.rover_equipment = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # store -> rover
        self.camera_info = defaultdict(dict) # camera -> {on_board_rover, calibration_target, supported_modes}
        self.objective_visibility = defaultdict(set) # objective -> {waypoint}
        self.calibration_targets = {} # camera -> objective
        self.waypoint_visibility = defaultdict(set) # waypoint -> {waypoint} (visible)
        self.rover_traversal_graphs = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> {waypoint} (can_traverse)
        self.waypoints = set()
        self.rovers = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        # Extract objects and static predicates
        for fact_string in task.static:
            fact = self.parse_fact(fact_string)
            predicate = fact[0]
            if predicate == 'at_lander':
                self.lander_location = fact[2]
                self.landers.add(fact[1])
                self.waypoints.add(fact[2])
            elif predicate == 'equipped_for_soil_analysis':
                self.rover_equipment[fact[1]].add('soil')
                self.rovers.add(fact[1])
            elif predicate == 'equipped_for_rock_analysis':
                self.rover_equipment[fact[1]].add('rock')
                self.rovers.add(fact[1])
            elif predicate == 'equipped_for_imaging':
                self.rover_equipment[fact[1]].add('imaging')
                self.rovers.add(fact[1])
            elif predicate == 'store_of':
                self.rover_stores[fact[1]] = fact[2]
                self.stores.add(fact[1])
                self.rovers.add(fact[2])
            elif predicate == 'supports':
                camera, mode = fact[1], fact[2]
                self.camera_info[camera].setdefault('supported_modes', set()).add(mode)
                self.cameras.add(camera)
                self.modes.add(mode)
            elif predicate == 'visible':
                wp1, wp2 = fact[1], fact[2]
                self.waypoint_visibility[wp1].add(wp2)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
            elif predicate == 'visible_from':
                obj, wp = fact[1], fact[2]
                self.objective_visibility[obj].add(wp)
                self.objectives.add(obj)
                self.waypoints.add(wp)
            elif predicate == 'calibration_target':
                camera, obj = fact[1], fact[2]
                self.calibration_targets[camera] = obj
                self.cameras.add(camera)
                self.objectives.add(obj)
            elif predicate == 'on_board':
                camera, rover = fact[1], fact[2]
                self.camera_info[camera]['on_board_rover'] = rover
                self.cameras.add(camera)
                self.rovers.add(rover)
            elif predicate == 'can_traverse':
                 rover, wp1, wp2 = fact[1], fact[2], fact[3]
                 self.rover_traversal_graphs[rover][wp1].add(wp2)
                 self.rovers.add(rover)
                 self.waypoints.add(wp1)
                 self.waypoints.add(wp2)
            # Add objects from dynamic predicates encountered in static (e.g., initial state facts)
            # This ensures all relevant objects are in the sets for BFS.
            elif predicate == 'at': self.rovers.add(fact[1]); self.waypoints.add(fact[2])
            elif predicate in ['have_soil_analysis', 'have_rock_analysis']: self.rovers.add(fact[1]); self.waypoints.add(fact[2])
            elif predicate == 'have_image': self.rovers.add(fact[1]); self.objectives.add(fact[2]); self.modes.add(fact[3])
            elif predicate == 'calibrated': self.cameras.add(fact[1]); self.rovers.add(fact[2])
            elif predicate in ['empty', 'full']: self.stores.add(fact[1])
            elif predicate in ['at_soil_sample', 'at_rock_sample']: self.waypoints.add(fact[1])
            elif predicate in ['communicated_soil_data', 'communicated_rock_data']: self.waypoints.add(fact[1])
            elif predicate == 'communicated_image_data': self.objectives.add(fact[1]); self.modes.add(fact[2])


        # Ensure all waypoints mentioned in traversal graphs are in the set
        for rover_graph in self.rover_traversal_graphs.values():
             for wp in rover_graph:
                 self.waypoints.add(wp)
                 for neighbor in rover_graph[wp]:
                     self.waypoints.add(neighbor)

        # Precompute rover distances
        self.rover_distances = {}
        all_waypoints_list = list(self.waypoints) # Use a list for consistent order in BFS
        for rover in self.rovers:
            self.rover_distances[rover] = self.compute_all_pairs_shortest_paths(self.rover_traversal_graphs[rover], all_waypoints_list)

        # Identify communication waypoints
        self.comm_wps = self.waypoint_visibility.get(self.lander_location, set())

        # Precompute min distance to communication waypoints for each rover and waypoint
        self.min_dist_to_comm = defaultdict(dict) # rover -> waypoint -> min_dist
        for rover in self.rovers:
            for wp in self.waypoints:
                min_d = math.inf
                # Check if wp is a valid start node for this rover's distances
                if wp in self.rover_distances[rover]:
                    for comm_wp in self.comm_wps:
                         # Check if comm_wp is a valid destination node from wp
                         if comm_wp in self.rover_distances[rover][wp]:
                             min_d = min(min_d, self.rover_distances[rover][wp][comm_wp])
                self.min_dist_to_comm[rover][wp] = min_d


    def parse_fact(self, fact_string):
        """
        Parses a fact string like '(predicate arg1 arg2)' into a tuple ('predicate', 'arg1', 'arg2').
        """
        # Remove parentheses and split by space
        parts = fact_string[1:-1].split()
        return tuple(parts)

    def bfs(self, graph, all_nodes, start_node):
        """
        Performs BFS from start_node on the given graph, returning distances to all_nodes.
        """
        distances = {node: math.inf for node in all_nodes}
        if start_node not in distances:
             return distances # start_node is not in the domain's waypoints

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in the graph (has outgoing edges)
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # Check if neighbor is a valid node in the domain
                    if neighbor in distances and distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def compute_all_pairs_shortest_paths(self, graph, nodes):
        """
        Computes shortest paths from every node in 'nodes' to every other node
        reachable via the given graph.
        """
        all_distances = {}
        for start_node in nodes:
            all_distances[start_node] = self.bfs(graph, nodes, start_node)
        return all_distances

    def __call__(self, task, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        # Check if goal is reached
        if task.goal_reached(state):
            return 0

        # Parse dynamic info from state
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        rover_samples = defaultdict(lambda: defaultdict(set)) # rover -> type (soil/rock) -> {waypoint}
        rover_images = defaultdict(set) # rover -> {(objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        achieved_goals = set() # {(predicate, args)}

        for fact_string in state:
            fact = self.parse_fact(fact_string)
            predicate = fact[0]
            if predicate == 'at':
                rover_locations[fact[1]] = fact[2]
            elif predicate == 'empty':
                store_status[fact[1]] = 'empty'
            elif predicate == 'full':
                store_status[fact[1]] = 'full'
            elif predicate == 'have_soil_analysis':
                rover_samples[fact[1]]['soil'].add(fact[2])
            elif predicate == 'have_rock_analysis':
                rover_samples[fact[1]]['rock'].add(fact[2])
            elif predicate == 'have_image':
                rover_images[fact[1]].add((fact[2], fact[3]))
            elif predicate == 'calibrated':
                calibrated_cameras.add((fact[1], fact[2]))
            elif predicate == 'at_soil_sample':
                soil_samples_at.add(fact[1])
            elif predicate == 'at_rock_sample':
                rock_samples_at.add(fact[1])
            elif predicate in ['communicated_soil_data', 'communicated_rock_data', 'communicated_image_data']:
                 achieved_goals.add(fact_string) # Store as string to match task.goals

        h = 0
        unachieved_goals = task.goals - achieved_goals

        for goal_string in unachieved_goals:
            goal = self.parse_fact(goal_string)
            goal_predicate = goal[0]

            min_cost_for_this_goal = math.inf

            if goal_predicate == 'communicated_soil_data':
                waypoint = goal[1]
                # Find equipped rovers
                equipped_rovers = [r for r in self.rovers if 'soil' in self.rover_equipment[r]]

                if not equipped_rovers:
                    min_cost_for_this_goal = math.inf # Cannot achieve this goal

                for rover in equipped_rovers:
                    current_loc = rover_locations.get(rover)
                    if current_loc is None: continue # Rover location must be known

                    cost = math.inf # Cost for this specific rover

                    # Case 1: Rover already has the sample
                    if waypoint in rover_samples[rover]['soil']:
                        # Need to navigate to comm waypoint and communicate
                        nav_cost = self.min_dist_to_comm[rover].get(current_loc, math.inf)
                        if nav_cost != math.inf:
                             cost = nav_cost + 1 # navigate + communicate
                        min_cost_for_this_goal = min(min_cost_for_this_goal, cost)

                    # Case 2: Rover does not have the sample yet
                    else:
                        # Need to sample and then communicate
                        # Check if sample is still at the waypoint
                        if waypoint in soil_samples_at:
                            # Cost to sample: navigate to waypoint + sample action + (drop action if store full)
                            store = next((s for s, r in self.rover_stores.items() if r == rover), None)
                            if store is None: continue # Rover must have a store

                            store_is_full = (store_status.get(store) == 'full')

                            nav_to_sample_cost = self.rover_distances[rover].get(current_loc, {}).get(waypoint, math.inf)
                            if nav_to_sample_cost == math.inf: continue # Cannot reach sample location

                            sample_actions = 1 # sample
                            if store_is_full:
                                sample_actions += 1 # drop

                            # Cost to communicate: navigate from sample location to comm waypoint + communicate action
                            nav_to_comm_cost = self.min_dist_to_comm[rover].get(waypoint, math.inf)
                            if nav_to_comm_cost == math.inf: continue # Cannot reach comm location from sample location

                            cost = nav_to_sample_cost + sample_actions + nav_to_comm_cost + 1 # nav1 + actions + nav2 + comm
                            min_cost_for_this_goal = min(min_cost_for_this_goal, cost)

            elif goal_predicate == 'communicated_rock_data':
                 waypoint = goal[1]
                 # Find equipped rovers
                 equipped_rovers = [r for r in self.rovers if 'rock' in self.rover_equipment[r]]

                 if not equipped_rovers:
                     min_cost_for_this_goal = math.inf # Cannot achieve this goal

                 for rover in equipped_rovers:
                     current_loc = rover_locations.get(rover)
                     if current_loc is None: continue

                     cost = math.inf # Cost for this specific rover

                     # Case 1: Rover already has the sample
                     if waypoint in rover_samples[rover]['rock']:
                         # Need to navigate to comm waypoint and communicate
                         nav_cost = self.min_dist_to_comm[rover].get(current_loc, math.inf)
                         if nav_cost != math.inf:
                              cost = nav_cost + 1 # navigate + communicate
                         min_cost_for_this_goal = min(min_cost_for_this_goal, cost)

                     # Case 2: Rover does not have the sample yet
                     else:
                         # Need to sample and then communicate
                         # Check if sample is still at the waypoint
                         if waypoint in rock_samples_at:
                             # Cost to sample: navigate to waypoint + sample action + (drop action if store full)
                             store = next((s for s, r in self.rover_stores.items() if r == rover), None)
                             if store is None: continue
                             store_is_full = (store_status.get(store) == 'full')

                             nav_to_sample_cost = self.rover_distances[rover].get(current_loc, {}).get(waypoint, math.inf)
                             if nav_to_sample_cost == math.inf: continue

                             sample_actions = 1 # sample
                             if store_is_full:
                                 sample_actions += 1 # drop

                             # Cost to communicate: navigate from sample location to comm waypoint + communicate action
                             nav_to_comm_cost = self.min_dist_to_comm[rover].get(waypoint, math.inf)
                             if nav_to_comm_cost == math.inf: continue

                             cost = nav_to_sample_cost + sample_actions + nav_to_comm_cost + 1 # nav1 + actions + nav2 + comm
                             min_cost_for_this_goal = min(min_cost_for_this_goal, cost)


            elif goal_predicate == 'communicated_image_data':
                objective, mode = goal[1], goal[2]
                # Find suitable rovers/cameras
                suitable_rover_camera_pairs = []
                for camera, info in self.camera_info.items():
                    rover = info.get('on_board_rover')
                    supported_modes = info.get('supported_modes', set())
                    if rover and 'imaging' in self.rover_equipment[rover] and mode in supported_modes:
                        suitable_rover_camera_pairs.append((rover, camera))

                if not suitable_rover_camera_pairs:
                    min_cost_for_this_goal = math.inf # Cannot achieve this goal

                # Get imaging waypoints for this objective
                image_wps = self.objective_visibility.get(objective, set())
                if not image_wps:
                    # No waypoint to view objective from, cannot take image
                    min_cost_for_this_goal = math.inf

                if min_cost_for_this_goal != math.inf: # Only proceed if imaging is potentially possible
                    for rover, camera in suitable_rover_camera_pairs:
                        current_loc = rover_locations.get(rover)
                        if current_loc is None: continue

                        cost = math.inf # Cost for this specific rover/camera pair

                        # Case 1: Rover already has the image
                        if (objective, mode) in rover_images[rover]:
                            # Need to navigate to comm waypoint and communicate
                            nav_cost = self.min_dist_to_comm[rover].get(current_loc, math.inf)
                            if nav_cost != math.inf:
                                 cost = nav_cost + 1 # navigate + communicate
                            min_cost_for_this_goal = min(min_cost_for_this_goal, cost)

                        # Case 2: Rover does not have the image yet
                        else:
                            # Need to take image and then communicate
                            calibrated = (camera, rover) in calibrated_cameras
                            calib_target = self.calibration_targets.get(camera)

                            if calibrated:
                                # Already calibrated, just need to navigate to image waypoint, take image, navigate to comm, communicate
                                # Cost = min_{p in ImageWps(o)} (dist(current_loc, p) + min_dist_to_comm[r][p]) + take_image + communicate
                                min_nav_img_comm = math.inf
                                for img_wp in image_wps:
                                    nav_img_cost = self.rover_distances[rover].get(current_loc, {}).get(img_wp, math.inf)
                                    comm_cost_from_img = self.min_dist_to_comm[rover].get(img_wp, math.inf)
                                    if nav_img_cost != math.inf and comm_cost_from_img != math.inf:
                                        min_nav_img_comm = min(min_nav_img_comm, nav_img_cost + comm_cost_from_img)

                                if min_nav_img_comm != math.inf:
                                     cost = min_nav_img_comm + 1 + 1 # nav_img_comm + take_image + communicate
                                     min_cost_for_this_goal = min(min_cost_for_this_goal, cost)


                            else: # Not calibrated
                                # Need to calibrate, then navigate to image waypoint, then take image, navigate to comm, communicate
                                # Cost = min_{w in CalibWps(i)} (dist(current_loc, w) + min_{p in ImageWps(o)} (dist(w, p) + min_dist_to_comm[r][p])) + calibrate + take_image + communicate
                                if calib_target is None: continue # Camera has no calibration target

                                calib_wps = self.objective_visibility.get(calib_target, set())
                                if not calib_wps: continue # No waypoint to calibrate from

                                min_nav_calib_img_comm = math.inf
                                for cal_wp in calib_wps:
                                    nav_calib_cost = self.rover_distances[rover].get(current_loc, {}).get(cal_wp, math.inf)
                                    if nav_calib_cost == math.inf: continue

                                    min_nav_img_comm_from_calib = math.inf
                                    for img_wp in image_wps:
                                        nav_calib_img_cost = self.rover_distances[rover].get(cal_wp, {}).get(img_wp, math.inf)
                                        if nav_calib_img_cost == math.inf: continue

                                        comm_cost_from_img = self.min_dist_to_comm[rover].get(img_wp, math.inf)
                                        if comm_cost_from_img != math.inf:
                                             total_nav_from_calib = nav_calib_img_cost + comm_cost_from_img
                                             min_nav_img_comm_from_calib = min(min_nav_img_comm_from_calib, total_nav_from_calib)

                                    if min_nav_img_comm_from_calib != math.inf:
                                         total_nav_cost = nav_calib_cost + min_nav_img_comm_from_calib
                                         min_nav_calib_img_comm = min(min_nav_calib_img_comm, total_nav_cost)


                                if min_nav_calib_img_comm != math.inf:
                                     cost = min_nav_calib_img_comm + 1 + 1 + 1 # nav_calib_img_comm + calibrate + take_image + communicate
                                     min_cost_for_this_goal = min(min_cost_for_this_goal, cost)


            # Add cost for this goal, handling infinity
            if min_cost_for_this_goal == math.inf:
                 # If any unachieved goal is unreachable, the total heuristic should reflect that.
                 # For greedy best-first, infinity is fine.
                 return math.inf
            h += min_cost_for_this_goal

        return h
