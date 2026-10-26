from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
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

# BFS for shortest paths
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         return distances # Start node not in graph nodes

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key before accessing graph[current_node]
        if current_node in graph:
            for neighbor in graph.get(current_node, []): # Use .get for safety
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes in a graph."""
    all_distances = {}
    # Iterate over all nodes that are keys in the graph (all waypoints)
    all_nodes = set(graph.keys())
    for start_node in all_nodes:
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated
    independently by finding the minimum cost sequence of actions (including
    navigation) required to achieve it from the current state, considering
    available rovers, equipment, samples, images, and communication points.
    Navigation costs are precomputed shortest paths on the rover-specific
    traverse/visible graph.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Parse Static Facts ---
        static_facts = task.static

        self.lander_location = None
        self.rover_capabilities = {} # {rover: set(capabilities)}
        self.rover_stores = {} # {rover: store}
        self.camera_info = {} # {camera: {'rover': rover, 'modes': set(), 'calibration_target': objective}}
        self.objective_visibility = {} # {objective: set(waypoints)}
        self.calibration_target_visibility = {} # {objective: set(waypoints)}
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        # Collect objects and initial sample locations (initial samples are static until collected)
        # Collect all waypoints first from all relevant predicates
        all_waypoints_set = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] in ['at', 'at_lander'] and len(parts) == 3: all_waypoints_set.add(parts[2])
            elif parts[0] in ['at_soil_sample', 'at_rock_sample'] and len(parts) == 2: all_waypoints_set.add(parts[1])
        for fact in task.goals:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in ['communicated_soil_data', 'communicated_rock_data'] and len(parts) == 2: all_waypoints_set.add(parts[1])
             # Image goals don't directly mention waypoints, but objectives/modes imply waypoints via visible_from

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at_lander' and len(parts) == 3:
                self.lander_location = parts[2]
                self.landers.add(parts[1])
                all_waypoints_set.add(parts[2])
            elif predicate == 'rover' and len(parts) == 2: self.rovers.add(parts[1])
            elif predicate == 'waypoint' and len(parts) == 2: all_waypoints_set.add(parts[1])
            elif predicate == 'store' and len(parts) == 2: self.stores.add(parts[1])
            elif predicate == 'camera' and len(parts) == 2: self.cameras.add(parts[1])
            elif predicate == 'mode' and len(parts) == 2: self.modes.add(parts[1])
            elif predicate == 'objective' and len(parts) == 2: self.objectives.add(parts[1])
            elif predicate == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.rover_capabilities.setdefault(parts[1], set()).add('soil')
            elif predicate == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.rover_capabilities.setdefault(parts[1], set()).add('rock')
            elif predicate == 'equipped_for_imaging' and len(parts) == 2:
                self.rover_capabilities.setdefault(parts[1], set()).add('imaging')
            elif predicate == 'store_of' and len(parts) == 3:
                self.rover_stores[parts[2]] = parts[1]
            elif predicate == 'on_board' and len(parts) == 3:
                self.camera_info.setdefault(parts[1], {})['rover'] = parts[2]
            elif predicate == 'supports' and len(parts) == 3:
                self.camera_info.setdefault(parts[1], {}).setdefault('modes', set()).add(parts[2])
            elif predicate == 'calibration_target' and len(parts) == 3:
                self.camera_info.setdefault(parts[1], {})['calibration_target'] = parts[2]
            elif predicate == 'visible_from' and len(parts) == 3:
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
                all_waypoints_set.add(waypoint)
            elif predicate == 'at_soil_sample' and len(parts) == 2:
                 all_waypoints_set.add(parts[1])
            elif predicate == 'at_rock_sample' and len(parts) == 2:
                 all_waypoints_set.add(parts[1])
            elif predicate == 'can_traverse' and len(parts) == 4:
                 all_waypoints_set.add(parts[2])
                 all_waypoints_set.add(parts[3])
            elif predicate == 'visible' and len(parts) == 3:
                 all_waypoints_set.add(parts[1])
                 all_waypoints_set.add(parts[2])

        self.waypoints = all_waypoints_set

        # Populate calibration_target_visibility based on objective_visibility and camera_info
        for cam, info in self.camera_info.items():
             cal_target = info.get('calibration_target')
             if cal_target and cal_target in self.objective_visibility:
                 self.calibration_target_visibility[cal_target] = self.objective_visibility[cal_target]


        # Build navigation graphs
        visible_pairs = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'visible' and len(parts) == 3:
                w1, w2 = parts[1], parts[2]
                if w1 in self.waypoints and w2 in self.waypoints:
                    visible_pairs.add((w1, w2))

        self.rover_navigation_graphs = {}
        for rover in self.rovers:
            # Initialize graph with all waypoints as nodes
            self.rover_navigation_graphs[rover] = {wp: set() for wp in self.waypoints}
            for fact in static_facts:
                parts = get_parts(fact)
                if not parts: continue
                if parts[0] == 'can_traverse' and len(parts) == 4 and parts[1] == rover:
                    u, v = parts[2], parts[3]
                    # An edge exists if can_traverse AND visible
                    if u in self.waypoints and v in self.waypoints and (u, v) in visible_pairs:
                         self.rover_navigation_graphs[rover][u].add(v)

        # Precompute distances
        self.rover_distances = {}
        for rover, graph in self.rover_navigation_graphs.items():
            self.rover_distances[rover] = compute_all_pairs_shortest_paths(graph)

        # Identify communication waypoints
        self.communication_waypoints = set()
        if self.lander_location and self.lander_location in self.waypoints:
            # communicate requires (visible ?x ?y) where rover is at ?x and lander at ?y
            # So, we need waypoints ?x such that (visible ?x lander_location)
            for wp1, wp2 in visible_pairs:
                if wp2 == self.lander_location:
                    self.communication_waypoints.add(wp1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # --- Parse State Facts ---
        rover_locations = {} # {rover: waypoint}
        store_status = {store: 'empty' for store in self.stores} # Stores are initially empty
        rover_samples = {rover: set() for rover in self.rovers} # {rover: set(waypoints)} # Contains waypoints for both soil and rock samples the rover has
        rover_images = {rover: set() for rover in self.rovers} # {rover: set((objective, mode))}
        camera_calibration = {camera: False for camera in self.cameras} # Cameras are initially not calibrated
        communicated_data = set()
        remaining_samples_at_waypoint = {} # {waypoint: 'soil' or 'rock'}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3 and parts[1].startswith('rover'):
                rover_locations[parts[1]] = parts[2]
            elif predicate == 'empty' and len(parts) == 2:
                store_status[parts[1]] = 'empty'
            elif predicate == 'full' and len(parts) == 2:
                store_status[parts[1]] = 'full'
            elif predicate == 'have_soil_analysis' and len(parts) == 3:
                rover, waypoint = parts[1], parts[2]
                rover_samples.setdefault(rover, set()).add(waypoint)
            elif predicate == 'have_rock_analysis' and len(parts) == 3:
                rover, waypoint = parts[1], parts[2]
                rover_samples.setdefault(rover, set()).add(waypoint)
            elif predicate == 'have_image' and len(parts) == 4:
                rover, objective, mode = parts[1], parts[2], parts[3]
                rover_images.setdefault(rover, set()).add((objective, mode))
            elif predicate == 'calibrated' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                camera_calibration[camera] = True
            elif predicate in ['communicated_soil_data', 'communicated_rock_data', 'communicated_image_data'] and len(parts) >= 2:
                communicated_data.add(fact)
            elif predicate == 'at_soil_sample' and len(parts) == 2:
                 remaining_samples_at_waypoint[parts[1]] = 'soil'
            elif predicate == 'at_rock_sample' and len(parts) == 2:
                 remaining_samples_at_waypoint[parts[1]] = 'rock'

        # Check if communication is possible at all if there are communication goals
        has_comm_goal = any(get_parts(g)[0].startswith('communicated_') for g in self.goals)
        if has_comm_goal and (not self.lander_location or not self.communication_waypoints):
             # If lander location or comm waypoints are missing, communication goals are impossible
             for goal in self.goals:
                 if goal.startswith('(communicated_') and goal not in communicated_data:
                      return float('inf') # Cannot communicate if no comm path exists


        # --- Compute Heuristic Cost ---
        total_cost = 0

        for goal in self.goals:
            if goal in communicated_data:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]

            min_goal_cost = float('inf')

            if predicate == 'communicated_soil_data' and len(parts) == 2:
                waypoint_to_sample = parts[1]
                required_capability = 'soil'

                for rover in self.rovers:
                    if required_capability not in self.rover_capabilities.get(rover, set()):
                        continue # Rover not equipped

                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None or rover not in self.rover_distances or current_rover_loc not in self.rover_distances[rover]: continue # Rover location unknown or not in graph

                    cost_for_this_rover = float('inf')

                    # Option 1: Rover already has the sample
                    if waypoint_to_sample in rover_samples.get(rover, set()):
                        # Just need to communicate
                        min_comm_nav_cost = float('inf')
                        for comm_wp in self.communication_waypoints:
                            if comm_wp in self.rover_distances[rover].get(current_rover_loc, {}):
                                nav_cost = self.rover_distances[rover][current_rover_loc][comm_wp]
                                min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                        if min_comm_nav_cost != float('inf'):
                             cost_for_this_rover = min(cost_for_this_rover, min_comm_nav_cost + 1) # +1 for communicate

                    # Option 2: Sample is at the waypoint
                    elif waypoint_to_sample in remaining_samples_at_waypoint and remaining_samples_at_waypoint[waypoint_to_sample] == required_capability:
                        # Need to sample and communicate
                        if waypoint_to_sample in self.rover_distances[rover].get(current_rover_loc, {}):
                            nav_cost_to_sample = self.rover_distances[rover][current_rover_loc][waypoint_to_sample]
                            store = self.rover_stores.get(rover)
                            # Check if the rover has a store and if it's full
                            store_cost = 1 if store and store_status.get(store) == 'full' else 0 # Need to drop if store is full
                            sample_action_cost = 1

                            min_comm_nav_cost_from_sample_wp = float('inf')
                            for comm_wp in self.communication_waypoints:
                                if waypoint_to_sample in self.rover_distances.get(rover, {}) and comm_wp in self.rover_distances[rover].get(waypoint_to_sample, {}):
                                    nav_cost_to_comm = self.rover_distances[rover][waypoint_to_sample][comm_wp]
                                    min_comm_nav_cost_from_sample_wp = min(min_comm_nav_cost_from_sample_wp, nav_cost_to_comm)

                            if min_comm_nav_cost_from_sample_wp != float('inf'):
                                comm_action_cost = 1
                                total_path_cost = nav_cost_to_sample + store_cost + sample_action_cost + min_comm_nav_cost_from_sample_wp + comm_action_cost
                                cost_for_this_rover = min(cost_for_this_rover, total_path_cost)

                    min_goal_cost = min(min_goal_cost, cost_for_this_rover)

            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                waypoint_to_sample = parts[1]
                required_capability = 'rock'

                for rover in self.rovers:
                    if required_capability not in self.rover_capabilities.get(rover, set()):
                        continue # Rover not equipped

                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None or rover not in self.rover_distances or current_rover_loc not in self.rover_distances[rover]: continue # Rover location unknown or not in graph

                    cost_for_this_rover = float('inf')

                    # Option 1: Rover already has the sample
                    if waypoint_to_sample in rover_samples.get(rover, set()):
                        # Just need to communicate
                        min_comm_nav_cost = float('inf')
                        for comm_wp in self.communication_waypoints:
                            if comm_wp in self.rover_distances[rover].get(current_rover_loc, {}):
                                nav_cost = self.rover_distances[rover][current_rover_loc][comm_wp]
                                min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                        if min_comm_nav_cost != float('inf'):
                             cost_for_this_rover = min(cost_for_this_rover, min_comm_nav_cost + 1) # +1 for communicate

                    # Option 2: Sample is at the waypoint
                    elif waypoint_to_sample in remaining_samples_at_waypoint and remaining_samples_at_waypoint[waypoint_to_sample] == required_capability:
                        # Need to sample and communicate
                        if waypoint_to_sample in self.rover_distances[rover].get(current_rover_loc, {}):
                            nav_cost_to_sample = self.rover_distances[rover][current_rover_loc][waypoint_to_sample]
                            store = self.rover_stores.get(rover)
                            # Check if the rover has a store and if it's full
                            store_cost = 1 if store and store_status.get(store) == 'full' else 0 # Need to drop if store is full
                            sample_action_cost = 1

                            min_comm_nav_cost_from_sample_wp = float('inf')
                            for comm_wp in self.communication_waypoints:
                                if waypoint_to_sample in self.rover_distances.get(rover, {}) and comm_wp in self.rover_distances[rover].get(waypoint_to_sample, {}):
                                    nav_cost_to_comm = self.rover_distances[rover][waypoint_to_sample][comm_wp]
                                    min_comm_nav_cost_from_sample_wp = min(min_comm_nav_cost_from_sample_wp, nav_cost_to_comm)

                            if min_comm_nav_cost_from_sample_wp != float('inf'):
                                comm_action_cost = 1
                                total_path_cost = nav_cost_to_sample + store_cost + sample_action_cost + min_comm_nav_cost_from_sample_wp + comm_action_cost
                                cost_for_this_rover = min(cost_for_this_rover, total_path_cost)

                    min_goal_cost = min(min_goal_cost, cost_for_this_rover)


            elif predicate == 'communicated_image_data' and len(parts) == 3:
                objective_to_image, mode_required = parts[1], parts[2]
                required_capability = 'imaging'

                for rover in self.rovers:
                    if required_capability not in self.rover_capabilities.get(rover, set()):
                        continue # Rover not equipped

                    current_rover_loc = rover_locations.get(rover)
                    if current_rover_loc is None or rover not in self.rover_distances or current_rover_loc not in self.rover_distances[rover]: continue # Rover location unknown or not in graph

                    cost_for_this_rover = float('inf')

                    # Find suitable cameras on this rover
                    suitable_cameras = [
                        cam for cam, info in self.camera_info.items()
                        if info.get('rover') == rover and mode_required in info.get('modes', set())
                    ]

                    for camera in suitable_cameras:
                        cost_for_this_camera = float('inf')

                        # Option 1: Rover already has the image
                        if (objective_to_image, mode_required) in rover_images.get(rover, set()):
                            # Just need to communicate
                            min_comm_nav_cost = float('inf')
                            for comm_wp in self.communication_waypoints:
                                if comm_wp in self.rover_distances[rover].get(current_rover_loc, {}):
                                    nav_cost = self.rover_distances[rover][current_rover_loc][comm_wp]
                                    min_comm_nav_cost = min(min_comm_nav_cost, nav_cost)
                            if min_comm_nav_cost != float('inf'):
                                cost_for_this_camera = min(cost_for_this_camera, min_comm_nav_cost + 1) # +1 for communicate

                        # Option 2: Need to take the image and communicate
                        else:
                            image_wps = self.objective_visibility.get(objective_to_image, set())
                            if not image_wps: continue # Cannot see objective o

                            for image_wp in image_wps:
                                if image_wp not in self.rover_distances[rover].get(current_rover_loc, {}):
                                     continue # Cannot reach image waypoint

                                nav_cost_to_image_wp = self.rover_distances[rover][current_rover_loc][image_wp]
                                take_image_action_cost = 1

                                # Calculate calibration cost if needed
                                calibration_cost_sequence = 0
                                if not camera_calibration.get(camera, False): # If camera is NOT calibrated
                                    cal_target = self.camera_info.get(camera, {}).get('calibration_target')
                                    cal_wps = self.calibration_target_visibility.get(cal_target, set())
                                    if not cal_wps:
                                        calibration_cost_sequence = float('inf') # Cannot calibrate this camera
                                    else:
                                        min_cal_round_trip = float('inf')
                                        # Cost is from image_wp to cal_wp and back to image_wp + calibrate action
                                        for cal_wp in cal_wps:
                                            if image_wp in self.rover_distances.get(rover, {}) and cal_wp in self.rover_distances[rover].get(image_wp, {}) and image_wp in self.rover_distances[rover].get(cal_wp, {}):
                                                round_trip_cost = self.rover_distances[rover][image_wp][cal_wp] + self.rover_distances[rover][cal_wp][image_wp]
                                                min_cal_round_trip = min(min_cal_round_trip, round_trip_cost)

                                        if min_cal_round_trip == float('inf'):
                                            calibration_cost_sequence = float('inf') # Cannot reach cal_wp from image_wp and back
                                        else:
                                            calibration_cost_sequence = min_cal_round_trip + 1 # +1 for calibrate action

                                if calibration_cost_sequence == float('inf'):
                                     continue # Cannot calibrate, so cannot take image

                                # Now consider communication after taking the image at image_wp
                                min_comm_nav_cost_from_image_wp = float('inf')
                                for comm_wp in self.communication_waypoints:
                                    if image_wp in self.rover_distances.get(rover, {}) and comm_wp in self.rover_distances[rover].get(image_wp, {}):
                                        nav_cost_to_comm = self.rover_distances[rover][image_wp][comm_wp]
                                        min_comm_nav_cost_from_image_wp = min(min_comm_nav_cost_from_image_wp, nav_cost_to_comm)

                                if min_comm_nav_cost_from_image_wp != float('inf'):
                                    comm_action_cost = 1
                                    total_path_cost = nav_cost_to_image_wp + calibration_cost_sequence + take_image_action_cost + min_comm_nav_cost_from_image_wp + comm_action_cost
                                    cost_for_this_image_wp_comm_wp_pair = total_path_cost
                                    cost_for_this_camera = min(cost_for_this_camera, cost_for_this_image_wp_comm_wp_pair)

                        cost_for_this_rover = min(cost_for_this_rover, cost_for_this_camera)

                    min_goal_cost = min(min_goal_cost, cost_for_this_rover)


            # Add the minimum cost found for this goal to the total heuristic
            if min_goal_cost != float('inf'):
                total_cost += min_goal_cost
            else:
                 # If a goal is unreachable, return infinity.
                 # This helps prune branches leading to impossible states.
                 return float('inf')


        return total_cost
