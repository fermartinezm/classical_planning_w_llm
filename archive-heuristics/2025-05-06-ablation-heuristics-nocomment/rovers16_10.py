import collections

# Helper functions
def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    # Simple equality check, no fnmatch needed for this PDDL format
    return all(part == arg or arg == '?' for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Performs BFS to find shortest distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Ensure start_node is in the graph nodes
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph: # Check if current_node has neighbors (should always be true if added to graph)
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

# Heuristic class
class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved communication goal. The cost for each goal is
    estimated based on the minimum actions required to obtain the necessary
    data (sample or image) and communicate it, considering rover locations,
    capabilities, and navigation distances.

    The heuristic is non-admissible and designed to guide a greedy best-first search.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Assumptions:
        - Navigation is possible only via links specified by both (can_traverse r w1 w2) and (visible w1 w2).
        - Rovers have a single store.
        - Samples (soil/rock) can only be collected if they were present in the initial state
          and are still present in the current state. Once sampled, they are consumed.
          If a goal requires a sample that was initially present but is no longer in the state
          and no rover has the corresponding 'have_..._analysis' fact, the goal is considered
          unreachable from this state (cost infinity).
        - Imaging requires calibration before each 'take_image' action using a specific
          calibration target visible from a waypoint.
        - Communication requires the rover to be at a waypoint visible from the lander location.
        - Costs for different unachieved goals are summed independently (ignoring potential
          synergies like one navigation serving multiple goals).
        - Action costs are assumed to be 1. Navigation cost is the shortest path distance.

        Heuristic Initialization:
        1. Collect all waypoints, rovers, cameras, objectives, modes, landers, stores
           from static and initial state facts.
        2. Store rover capabilities (soil, rock, imaging).
        3. Store rover-store mapping.
        4. Store camera information per rover (camera, supported modes, calibration target).
        5. Build the navigation graph for each rover based on (can_traverse) and (visible) facts.
        6. Compute all-pairs shortest path distances for each rover on its navigation graph using BFS.
        7. Identify communication points (waypoints visible from the lander location).
        8. Store initial soil and rock sample locations to check if sampling is possible.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state_facts = task.initial_state # Store initial state to check sample availability

        # 1. Collect objects and basic static info
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.landers = set()
        self.lander_location = None

        self.rover_capabilities = {} # {rover: {cap1, cap2, ...}}
        self.rover_stores = {} # {rover: store}
        self.visible_waypoints = set() # {(wp1, wp2), ...}
        self.objective_visibility = {} # {objective: {wp1, ...}}
        self.calibration_targets = {} # {camera: objective}
        self.rover_cameras_info = {} # {rover: [{'camera': cam, 'modes': {mode, ...}, 'cal_target': obj}, ...]}

        # Parse static facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at_lander':
                self.landers.add(parts[1])
                self.waypoints.add(parts[2])
                self.lander_location = parts[2]
            elif pred == 'can_traverse':
                self.rovers.add(parts[1])
                self.waypoints.add(parts[2])
                self.waypoints.add(parts[3])
            elif pred == 'equipped_for_soil_analysis':
                self.rovers.add(parts[1])
                self.rover_capabilities.setdefault(parts[1], set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rovers.add(parts[1])
                self.rover_capabilities.setdefault(parts[1], set()).add('rock')
            elif pred == 'equipped_for_imaging':
                self.rovers.add(parts[1])
                self.rover_capabilities.setdefault(parts[1], set()).add('imaging')
            elif pred == 'store_of':
                self.stores.add(parts[1])
                self.rovers.add(parts[2])
                self.rover_stores[parts[2]] = parts[1]
            elif pred == 'supports':
                self.cameras.add(parts[1])
                self.modes.add(parts[2])
                # Store temporarily, combine later
            elif pred == 'visible':
                self.waypoints.add(parts[1])
                self.waypoints.add(parts[2])
                self.visible_waypoints.add((parts[1], parts[2]))
            elif pred == 'visible_from':
                self.objectives.add(parts[1])
                self.waypoints.add(parts[2])
                self.objective_visibility.setdefault(parts[1], set()).add(parts[2])
            elif pred == 'calibration_target':
                self.cameras.add(parts[1])
                self.objectives.add(parts[2])
                self.calibration_targets[parts[1]] = parts[2]
            elif pred == 'on_board':
                self.cameras.add(parts[1])
                self.rovers.add(parts[2])
                # Store temporarily, combine later

        # Parse initial state facts for waypoints and initial samples
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact in self.initial_state_facts:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
             elif parts[0] == 'at_soil_sample':
                 self.waypoints.add(parts[1])
                 self.initial_soil_samples.add(parts[1])
             elif parts[0] == 'at_rock_sample':
                 self.waypoints.add(parts[1])
                 self.initial_rock_samples.add(parts[1])
             # Add other objects if they appear in initial state facts not covered above
             # The example initial states only contain basic facts like at, at_lander, at_soil_sample, at_rock_sample, empty, full, calibrated.
             # Derived facts like have_... are not in the initial state definition in the examples.

        # Combine camera info
        cameras_on_board = {} # {camera: rover}
        camera_supports = {} # {camera: {mode, ...}}
        # calibration_targets already populated

        for fact in self.static_facts:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == 'on_board':
                 cameras_on_board[parts[1]] = parts[2]
             elif pred == 'supports':
                 camera_supports.setdefault(parts[1], set()).add(parts[2])

        for cam, rover in cameras_on_board.items():
            if cam in camera_supports and cam in self.calibration_targets:
                self.rover_cameras_info.setdefault(rover, []).append({
                    'camera': cam,
                    'modes': camera_supports[cam],
                    'cal_target': self.calibration_targets[cam]
                })

        # 5. Build navigation graph
        # Graph nodes are all known waypoints
        self.navigation_graph = {r: {wp: set() for wp in self.waypoints} for r in self.rovers}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'can_traverse':
                rover, wp1, wp2 = parts[1:]
                # Navigate action requires both can_traverse and visible
                if (wp1, wp2) in self.visible_waypoints:
                    # Ensure rover and waypoints are known before adding edge
                    if rover in self.navigation_graph and wp1 in self.navigation_graph[rover] and wp2 in self.navigation_graph[rover]:
                         self.navigation_graph[rover][wp1].add(wp2)


        # 6. Compute navigation distances
        self.navigation_distances = {}
        for rover, graph in self.navigation_graph.items():
            self.navigation_distances[rover] = {}
            for start_wp in self.waypoints:
                self.navigation_distances[rover][start_wp] = bfs(graph, start_wp)

        # 7. Identify communication points
        self.communication_points = set()
        if self.lander_location:
            for wp1, wp2 in self.visible_waypoints:
                # A rover needs to be at wp1, and lander at wp2, and (visible wp1 wp2) must be true
                # Lander location is fixed at self.lander_location (which is wp2 in this check)
                if wp2 == self.lander_location:
                    self.communication_points.add(wp1)
                # Also check the reverse direction if visible is bidirectional
                if wp1 == self.lander_location:
                     self.communication_points.add(wp2)


    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify all goal predicates that are not yet true in the current state.
        2. For each unachieved goal predicate, estimate the minimum number of actions
           required to make it true. This estimation considers:
           - Whether the necessary data (sample or image) is already collected.
           - If data needs collecting:
             - Navigation cost to the collection location (sample waypoint or image waypoint).
             - Cost of the collection action (sample_soil/rock, take_image).
             - For imaging, includes calibration cost and navigation to calibration target.
             - For sampling, includes drop cost if the store is full.
             - Check if sampling is even possible (sample must be in the current state and initial state).
           - Navigation cost from the data location (or current location if data is held)
             to a communication point.
           - Cost of the communication action (communicate_soil/rock/image_data).
           - The minimum cost is taken over all suitable rovers and relevant waypoints
             (image location, calibration location, communication location).
           - If a goal is determined to be unreachable from the current state (e.g., sample consumed and lost, no path),
             its estimated cost is infinity.
        3. The total heuristic value is the sum of the estimated minimum costs for all
           unachieved goal predicates.
        4. If all goal predicates are achieved, the heuristic value is 0.
        """
        state = node.state
        infinity = float('inf')

        # Parse current state facts
        current_rover_locations = {} # {rover: waypoint}
        current_at_soil_sample = set() # {waypoint, ...}
        current_at_rock_sample = set() # {waypoint, ...}
        current_store_status = {} # {rover: 'empty' or 'full'}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at':
                current_rover_locations[parts[1]] = parts[2]
            elif pred == 'at_soil_sample':
                current_at_soil_sample.add(parts[1])
            elif pred == 'at_rock_sample':
                current_at_rock_sample.add(parts[1])
            elif pred == 'empty':
                 # Find which rover this store belongs to
                 for r, s in self.rover_stores.items():
                     if s == parts[1]:
                         current_store_status[r] = 'empty'
                         break
            elif pred == 'full':
                 # Find which rover this store belongs to
                 for r, s in self.rover_stores.items():
                     if s == parts[1]:
                         current_store_status[r] = 'full'
                         break


        total_cost = 0
        unachieved_goals = self.goals - state

        if not unachieved_goals:
            return 0 # Goal state

        for goal_fact in unachieved_goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            cost_g = infinity

            if pred == 'communicated_soil_data':
                w = parts[1]
                # Option 1: Communicate existing sample
                # Find any rover that has the sample
                rover_with_sample = None
                for fact in state:
                    if match(fact, "have_soil_analysis", "?r", w):
                        rover_with_sample = get_parts(fact)[2]
                        break # Found one, take the first

                if rover_with_sample and rover_with_sample in current_rover_locations:
                    r = rover_with_sample
                    current_pos = current_rover_locations[r]
                    min_comm_dist = infinity
                    if r in self.navigation_distances and current_pos in self.navigation_distances[r]:
                        for comm_wp in self.communication_points:
                            if comm_wp in self.navigation_distances[r][current_pos]:
                                 min_comm_dist = min(min_comm_dist, self.navigation_distances[r][current_pos][comm_wp])
                    if min_comm_dist != infinity:
                        cost_g = min(cost_g, min_comm_dist + 1) # 1 for communicate

                # Option 2: Sample and communicate
                # Check if sampling is possible: must have been an initial sample AND still be there
                if w in self.initial_soil_samples and w in current_at_soil_sample:
                    min_sample_comm_cost = infinity
                    for r in self.rovers:
                        if 'soil' in self.rover_capabilities.get(r, set()):
                            current_pos = current_rover_locations.get(r)
                            # Rover must have a known location and be able to reach the sample waypoint
                            if current_pos is not None and r in self.navigation_distances and current_pos in self.navigation_distances[r] and w in self.navigation_distances[r][current_pos]:
                                dist_to_sample = self.navigation_distances[r][current_pos][w]
                                sample_cost = dist_to_sample + 1 # navigate + sample
                                drop_cost = 1 if current_store_status.get(r) == 'full' else 0 # Add drop cost if store is full
                                sample_cost += drop_cost

                                min_comm_dist_from_sample_loc = infinity
                                # Rover must be able to reach a communication point from the sample waypoint
                                if r in self.navigation_distances and w in self.navigation_distances[r]:
                                    for comm_wp in self.communication_points:
                                         if comm_wp in self.navigation_distances[r][w]:
                                            min_comm_dist_from_sample_loc = min(min_comm_dist_from_sample_loc, self.navigation_distances[r][w][comm_wp])

                                if min_comm_dist_from_sample_loc != infinity:
                                    comm_cost = min_comm_dist_from_sample_loc + 1 # navigate + communicate
                                    min_sample_comm_cost = min(min_sample_comm_cost, sample_cost + comm_cost)

                    if min_sample_comm_cost != infinity:
                         cost_g = min(cost_g, min_sample_comm_cost)


            elif pred == 'communicated_rock_data':
                w = parts[1]
                # Option 1: Communicate existing sample
                rover_with_sample = None
                for fact in state:
                    if match(fact, "have_rock_analysis", "?r", w):
                        rover_with_sample = get_parts(fact)[2]
                        break # Found one

                if rover_with_sample and rover_with_sample in current_rover_locations:
                    r = rover_with_sample
                    current_pos = current_rover_locations[r]
                    min_comm_dist = infinity
                    if r in self.navigation_distances and current_pos in self.navigation_distances[r]:
                        for comm_wp in self.communication_points:
                            if comm_wp in self.navigation_distances[r][current_pos]:
                                 min_comm_dist = min(min_comm_dist, self.navigation_distances[r][current_pos][comm_wp])
                    if min_comm_dist != infinity:
                        cost_g = min(cost_g, min_comm_dist + 1) # 1 for communicate

                # Option 2: Sample and communicate
                if w in self.initial_rock_samples and w in current_at_rock_sample:
                    min_sample_comm_cost = infinity
                    for r in self.rovers:
                        if 'rock' in self.rover_capabilities.get(r, set()):
                            current_pos = current_rover_locations.get(r)
                            if current_pos is not None and r in self.navigation_distances and current_pos in self.navigation_distances[r] and w in self.navigation_distances[r][current_pos]:
                                dist_to_sample = self.navigation_distances[r][current_pos][w]
                                sample_cost = dist_to_sample + 1 # navigate + sample
                                drop_cost = 1 if current_store_status.get(r) == 'full' else 0 # Add drop cost if store is full
                                sample_cost += drop_cost

                                min_comm_dist_from_sample_loc = infinity
                                if r in self.navigation_distances and w in self.navigation_distances[r]:
                                    for comm_wp in self.communication_points:
                                         if comm_wp in self.navigation_distances[r][w]:
                                            min_comm_dist_from_sample_loc = min(min_comm_dist_from_sample_loc, self.navigation_distances[r][w][comm_wp])

                                if min_comm_dist_from_sample_loc != infinity:
                                    comm_cost = min_comm_dist_from_sample_loc + 1 # navigate + communicate
                                    min_sample_comm_cost = min(min_sample_comm_cost, sample_cost + comm_cost)

                    if min_sample_comm_cost != infinity:
                         cost_g = min(cost_g, min_sample_comm_cost)


            elif pred == 'communicated_image_data':
                o, m = parts[1:]
                # Option 1: Communicate existing image
                rover_with_image = None
                for fact in state:
                    if match(fact, "have_image", "?r", o, m):
                        rover_with_image = get_parts(fact)[2]
                        break # Found one

                if rover_with_image and rover_with_image in current_rover_locations:
                    r = rover_with_image
                    current_pos = current_rover_locations[r]
                    min_comm_dist = infinity
                    if r in self.navigation_distances and current_pos in self.navigation_distances[r]:
                        for comm_wp in self.communication_points:
                            if comm_wp in self.navigation_distances[r][current_pos]:
                                 min_comm_dist = min(min_comm_dist, self.navigation_distances[r][current_pos][comm_wp])
                    if min_comm_dist != infinity:
                        cost_g = min(cost_g, min_comm_dist + 1) # 1 for communicate

                # Option 2: Image and communicate
                min_image_comm_cost = infinity
                for r in self.rovers:
                    if 'imaging' in self.rover_capabilities.get(r, set()):
                        current_pos = current_rover_locations.get(r)
                        if current_pos is None: continue # Rover location unknown? Should not happen.

                        rover_cams_info = self.rover_cameras_info.get(r, [])
                        for cam_info in rover_cams_info:
                            cam = cam_info['camera']
                            cal_target = cam_info['cal_target']
                            supported_modes = cam_info['modes']

                            if m in supported_modes:
                                # Find best calibration waypoint
                                min_cal_nav_cost = infinity
                                best_cal_wp = None
                                cal_target_visible_from = self.objective_visibility.get(cal_target, set())
                                if r in self.navigation_distances and current_pos in self.navigation_distances[r]:
                                    for cal_wp in cal_target_visible_from:
                                        if cal_wp in self.navigation_distances[r][current_pos]:
                                            dist = self.navigation_distances[r][current_pos][cal_wp]
                                            if dist != infinity and dist < min_cal_nav_cost:
                                                min_cal_nav_cost = dist
                                                best_cal_wp = cal_wp

                                if best_cal_wp is not None:
                                    cal_cost = min_cal_nav_cost + 1 # navigate + calibrate

                                    # Find best image waypoint
                                    min_img_nav_cost = infinity
                                    best_img_wp = None
                                    objective_visible_from = self.objective_visibility.get(o, set())
                                    if r in self.navigation_distances and best_cal_wp in self.navigation_distances[r]:
                                        for img_wp in objective_visible_from:
                                            if img_wp in self.navigation_distances[r][best_cal_wp]:
                                                dist = self.navigation_distances[r][best_cal_wp][img_wp]
                                                if dist != infinity and dist < min_img_nav_cost:
                                                    min_img_nav_cost = dist
                                                    best_img_wp = img_wp

                                    if best_img_wp is not None:
                                        image_cost = min_img_nav_cost + 1 # navigate + take_image

                                        # Find best communication waypoint from image location
                                        min_comm_nav_cost = infinity
                                        if r in self.navigation_distances and best_img_wp in self.navigation_distances[r]:
                                            for comm_wp in self.communication_points:
                                                if comm_wp in self.navigation_distances[r][best_img_wp]:
                                                    dist = self.navigation_distances[r][best_img_wp][comm_wp]
                                                    if dist != infinity:
                                                        min_comm_nav_cost = min(min_comm_nav_cost, dist)

                                        if min_comm_nav_cost != infinity:
                                            comm_cost = min_comm_nav_cost + 1 # navigate + communicate
                                            min_image_comm_cost = min(min_image_comm_cost, cal_cost + image_cost + comm_cost)

                if min_image_comm_cost != infinity:
                     cost_g = min(cost_g, min_image_comm_cost)

            # Add cost for this goal to total
            total_cost += cost_g

        # If total_cost is infinity, it means at least one goal is unreachable.
        # Return infinity in that case. Otherwise, return the sum.
        return total_cost
