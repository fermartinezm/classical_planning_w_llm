import collections
import math

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Removes leading/trailing parens and splits by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the estimated costs
        for each unachieved goal fact. The cost for each goal fact is estimated based on
        the actions and navigation required to achieve it, minimizing over available
        rovers and relevant waypoints. Navigation costs are precomputed using BFS
        on rover-specific traversability graphs.

    Assumptions:
        - The PDDL instance is well-formed and follows the rovers domain definition.
        - The navigation graph for each rover (defined by `can_traverse` and `visible`)
          is static and does not change during planning.
        - The locations of samples (`at_soil_sample`, `at_rock_sample`) are dynamic
          and consumed by sampling, but their initial locations are known from the
          initial state. If a sample is not in the initial state, it cannot be sampled.
        - The lander location is static.
        - Camera properties, calibration targets, objective visibility, rover equipment,
          and store ownership are static.
        - The heuristic estimates the cost for each unachieved goal fact independently
          and sums these costs (additive heuristic). This ignores potential negative
          interactions and resource constraints (except for checking if a store is full
          before sampling).
        - The estimated cost for a sample/rock goal includes navigation to the sample,
          sampling, potential store drop, navigation to a communication point, and communication.
        - The estimated cost for an image goal includes navigation to a calibration point,
          calibrating, navigation to an image point, taking the image, navigation to a
          communication point, and communication. The navigation cost for image goals
          when the image is not held is estimated as the minimum sequential path cost
          (current location -> calibration waypoint -> image waypoint -> communication waypoint)
          over all combinations of relevant waypoints and capable rovers/cameras.

    Heuristic Initialization:
        The constructor precomputes static information from the task's static facts:
        - Lander location.
        - Rover capabilities (soil, rock, imaging).
        - Store ownership mapping.
        - Camera information (on-board status, supported modes, calibration targets).
        - Objective visibility mapping (`visible_from`).
        - Initial locations of soil and rock samples.
        - Rover-specific navigation graphs based on `visible` and `can_traverse` facts.
        - All-pairs shortest paths for each rover's navigation graph using BFS.
        - Set of communication waypoints (visible from the lander location).

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state. If yes, return 0.
        2. Initialize the heuristic value `h` to 0.
        3. Parse dynamic information from the current state: rover locations, store fullness,
           held samples, held images, camera calibration status.
        4. Identify the set of goal facts that are not true in the current state.
        5. For each unachieved goal fact:
            a. If the goal is `(communicated_soil_data ?w)`:
                i. Check if any rover `r` currently holds `(have_soil_analysis ?r ?w)`.
                ii. If yes: Calculate the minimum cost to communicate this sample. This is 1 (communicate action) plus the minimum navigation cost for any rover holding the sample to reach any communication waypoint. The navigation cost is the shortest path from the rover's current location to the nearest communication waypoint. Add this minimum cost to `h`.
                iii. If no rover holds the sample: Check if `(at_soil_sample ?w)` was present in the initial state. If not, the goal is unreachable; add infinity to `h`. If yes: Calculate the minimum cost to sample and communicate. This involves finding a soil-equipped rover `r`, navigating it to `w`, sampling, navigating it from `w` to a communication waypoint, and communicating. The cost is estimated as: minimum over soil-equipped rovers `r` of (Shortest path from `r`'s current location to `w` + 1 (sample) + (1 if `r`'s store is full in the current state) + Shortest path from `w` to the nearest communication waypoint for `r` + 1 (communicate)). Add this minimum cost to `h`.
            b. If the goal is `(communicated_rock_data ?w)`: Follow the same logic as for soil data, using rock-specific predicates and capabilities.
            c. If the goal is `(communicated_image_data ?o ?m)`:
                i. Check if any rover `r` currently holds `(have_image ?r ?o ?m)`.
                ii. If yes: Calculate the minimum cost to communicate this image. This is 1 (communicate action) plus the minimum navigation cost for any rover holding the image to reach any communication waypoint. The navigation cost is the shortest path from the rover's current location to the nearest communication waypoint. Add this minimum cost to `h`.
                iii. If no rover holds the image: Calculate the minimum cost to calibrate, image, and communicate. This involves finding an imaging-equipped rover `r` with a camera `i` supporting mode `m`, navigating to a calibration waypoint for `i`, calibrating, navigating to an image point for `o`, taking the image, navigating to a communication waypoint, and communicating. The cost is estimated as: 1 (calibrate) + 1 (take_image) + 1 (communicate) + minimum over capable rovers `r` (with camera `i` supporting `m`) of (minimum sequential navigation cost from `r`'s current location to a calibration waypoint, then to an image waypoint, then to a communication waypoint, minimizing over all combinations of these waypoints). Handle cases where required waypoints are unreachable by adding infinity. Add this minimum cost to `h`.
        6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information and navigation costs.

        Args:
            task: The planning task object.
        """
        self.task = task
        self._parse_static_info()
        self._compute_shortest_paths()
        self._identify_comm_waypoints()

    def _parse_static_info(self):
        """Parses static facts from the task."""
        self.lander_location = {}
        self.rover_capabilities = collections.defaultdict(set) # 'soil', 'rock', 'imaging' -> {rover_names}
        self.store_owner = {} # store_name -> rover_name
        self.rover_cameras = collections.defaultdict(set) # rover_name -> {camera_names}
        self.camera_modes = collections.defaultdict(set) # camera_name -> {mode_names}
        self.camera_cal_target = {} # camera_name -> objective_name
        self.objective_visible_from = collections.defaultdict(set) # objective_name -> {waypoint_names}
        self.rover_nav_graphs = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> wp -> {neighbor_wps}
        self.initial_soil_samples = set() # {waypoint_names}
        self.initial_rock_samples = set() # {waypoint_names}
        self.all_waypoints = set()
        self.all_rovers = set()

        # Collect all waypoints and rovers from static and initial state first
        for fact_string in self.task.static | self.task.initial_state:
             fact = parse_fact(fact_string)
             if fact[0] in ('at', 'at_lander', 'can_traverse', 'visible', 'at_soil_sample', 'at_rock_sample', 'visible_from'):
                 for item in fact[1:]:
                     if item.startswith('waypoint'):
                         self.all_waypoints.add(item)
                     elif item.startswith('rover'):
                         self.all_rovers.add(item)

        # Parse static facts
        visible_graph = collections.defaultdict(set)
        for fact_string in self.task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'at_lander':
                # Assuming one lander for simplicity based on examples
                self.lander_location[fact[1]] = fact[2]
            elif fact[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities['soil'].add(fact[1])
            elif fact[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities['rock'].add(fact[1])
            elif fact[0] == 'equipped_for_imaging':
                self.rover_capabilities['imaging'].add(fact[1])
            elif fact[0] == 'store_of':
                self.store_owner[fact[1]] = fact[2]
            elif fact[0] == 'on_board':
                self.rover_cameras[fact[2]].add(fact[1])
            elif fact[0] == 'supports':
                self.camera_modes[fact[1]].add(fact[2])
            elif fact[0] == 'calibration_target':
                self.camera_cal_target[fact[1]] = fact[2]
            elif fact[0] == 'visible_from':
                self.objective_visible_from[fact[1]].add(fact[2])
            elif fact[0] == 'visible':
                # visible ?w ?p means ?w is visible from ?p
                visible_graph[fact[2]].add(fact[1]) # Map p -> w (from p you can see w)

        # Build rover-specific navigation graphs considering both can_traverse and visible
        # A rover r can move from y to z if (can_traverse r y z) and (visible y z)
        # The navigate action is (navigate ?x - rover ?y - waypoint ?z - waypoint)
        # Precondition: (and (can_traverse ?x ?y ?z) (at ?x ?y) (visible ?y ?z))
        # This means rover ?x can move from ?y to ?z if (can_traverse ?x ?y ?z) is true AND ?z is visible FROM ?y.
        # So the graph edge is y -> z if (can_traverse r y z) and (visible y z).
        # My previous visible_graph mapping was p -> w if (visible w p), meaning w is visible from p.
        # Let's rebuild the visible_graph to map y -> z if (visible y z).
        visible_graph_nav = collections.defaultdict(set)
        for fact_string in self.task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'visible':
                # (visible ?y ?z) means ?z is visible from ?y
                visible_graph_nav[fact[1]].add(fact[2]) # Map y -> z (from y you can see z)

        for rover in self.all_rovers:
             for fact_string in self.task.static:
                 fact = parse_fact(fact_string)
                 if fact[0] == 'can_traverse' and fact[1] == rover:
                     wp_from, wp_to = fact[2], fact[3]
                     # Rover can traverse from wp_from to wp_to if (can_traverse rover wp_from wp_to)
                     # AND (visible wp_from wp_to) is true.
                     if wp_to in visible_graph_nav.get(wp_from, set()):
                         self.rover_nav_graphs[rover][wp_from].add(wp_to)


        # Initial sample locations (needed to check if a sample existed at all)
        for fact_string in self.task.initial_state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at_soil_sample':
                self.initial_soil_samples.add(fact[1])
            elif fact[0] == 'at_rock_sample':
                self.initial_rock_samples.add(fact[1])


    def _compute_shortest_paths(self):
        """Computes all-pairs shortest paths for each rover's navigation graph using BFS."""
        self.rover_shortest_paths = {} # rover -> start_wp -> end_wp -> distance

        for rover in self.all_rovers:
            graph = self.rover_nav_graphs.get(rover, collections.defaultdict(set))
            self.rover_shortest_paths[rover] = {}
            for start_wp in self.all_waypoints:
                self.rover_shortest_paths[rover][start_wp] = {}
                # Perform BFS from start_wp
                queue = collections.deque([(start_wp, 0)])
                visited = {start_wp: 0}

                while queue:
                    current_wp, dist = queue.popleft()
                    self.rover_shortest_paths[rover][start_wp][current_wp] = dist

                    for neighbor_wp in graph.get(current_wp, set()):
                        if neighbor_wp not in visited:
                            visited[neighbor_wp] = dist + 1
                            queue.append((neighbor_wp, dist + 1))

                # Mark unreachable waypoints with infinity
                for end_wp in self.all_waypoints:
                    if end_wp not in self.rover_shortest_paths[rover][start_wp]:
                        self.rover_shortest_paths[rover][start_wp][end_wp] = math.inf

    def _identify_comm_waypoints(self):
        """Identifies waypoints visible from the lander location."""
        self.comm_waypoints = set()
        lander_wp = list(self.lander_location.values())[0] # Assuming one lander
        # Find waypoints x such that (visible x y) is true, where y is lander_wp.
        # (visible ?x ?y) means ?x is visible FROM ?y.
        # So we need waypoints ?x such that (visible ?x lander_wp) is true.
        for fact_string in self.task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'visible' and fact[2] == lander_wp:
                 self.comm_waypoints.add(fact[1])


    def __call__(self, state, task):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).
            task: The planning task object.

        Returns:
            The estimated cost to reach the goal.
        """
        if task.goal_reached(state):
            return 0

        h = 0
        unachieved_goals = task.goals - state

        # Parse dynamic state information
        rover_locations = {} # rover_name -> waypoint_name
        store_full = set() # {store_names}
        held_soil_samples = collections.defaultdict(set) # rover_name -> {waypoint_names}
        held_rock_samples = collections.defaultdict(set) # rover_name -> {waypoint_names}
        held_images = collections.defaultdict(set) # rover_name -> {(objective, mode)}
        # camera_calibrated = set() # {(camera_name, rover_name)} # Not needed for heuristic calculation

        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at':
                rover_locations[fact[1]] = fact[2]
            elif fact[0] == 'full':
                store_full.add(fact[1])
            elif fact[0] == 'have_soil_analysis':
                held_soil_samples[fact[1]].add(fact[2])
            elif fact[0] == 'have_rock_analysis':
                held_rock_samples[fact[1]].add(fact[2])
            elif fact[0] == 'have_image':
                held_images[fact[1]].add((fact[2], fact[3]))
            # elif fact[0] == 'calibrated':
            #     camera_calibrated.add((fact[1], fact[2])) # Not used in this heuristic logic

        # Process unachieved goals
        for goal_fact_string in unachieved_goals:
            goal_fact = parse_fact(goal_fact_string)
            goal_type = goal_fact[0]

            if goal_type == 'communicated_soil_data':
                waypoint = goal_fact[1]
                min_goal_cost = math.inf

                # Check if any rover already has the sample
                rovers_holding_sample = {r for r, wps in held_soil_samples.items() if waypoint in wps}

                if rovers_holding_sample:
                    # Goal: Communicate sample already held
                    min_comm_nav_cost = math.inf
                    for rover in rovers_holding_sample:
                        current_loc = rover_locations.get(rover)
                        if current_loc is None: continue # Should not happen in a valid state

                        min_path_to_comm = math.inf
                        for comm_wp in self.comm_waypoints:
                            min_path_to_comm = min(min_path_to_comm, self.rover_shortest_paths[rover][current_loc][comm_wp])
                        min_comm_nav_cost = min(min_comm_nav_cost, min_path_to_comm)

                    if min_comm_nav_cost != math.inf:
                         min_goal_cost = 1 + min_comm_nav_cost # 1 for communicate action
                    else:
                         min_goal_cost = math.inf # Cannot reach communication point

                else:
                    # Goal: Sample and communicate
                    # Check if sample exists initially
                    if waypoint not in self.initial_soil_samples:
                        min_goal_cost = math.inf # Sample never existed, unreachable
                    else:
                        # Find capable rovers
                        capable_rovers = self.rover_capabilities['soil']
                        if not capable_rovers:
                            min_goal_cost = math.inf # No rover can sample soil
                        else:
                            min_sample_comm_cost = math.inf
                            for rover in capable_rovers:
                                current_loc = rover_locations.get(rover)
                                if current_loc is None: continue # Should not happen

                                # Cost to navigate to sample location
                                nav_cost_to_sample = self.rover_shortest_paths[rover][current_loc][waypoint]
                                if nav_cost_to_sample == math.inf: continue # Cannot reach sample location

                                # Cost for store (if full)
                                rover_store = None
                                for store, owner in self.store_owner.items():
                                    if owner == rover:
                                        rover_store = store
                                        break
                                # Assume rover has a store if equipped for soil analysis
                                store_cost = 1 if rover_store in store_full else 0

                                # Cost to navigate from sample location to communication location
                                min_nav_cost_sample_to_comm = math.inf
                                for comm_wp in self.comm_waypoints:
                                    min_nav_cost_sample_to_comm = min(min_nav_cost_sample_to_comm, self.rover_shortest_paths[rover][waypoint][comm_wp])

                                if min_nav_cost_sample_to_comm == math.inf: continue # Cannot reach communication point from sample location

                                # Total cost for this rover
                                total_rover_cost = nav_cost_to_sample + 1 + store_cost + min_nav_cost_sample_to_comm + 1 # +1 for sample, +1 for communicate

                                min_sample_comm_cost = min(min_sample_comm_cost, total_rover_cost)

                            min_goal_cost = min_sample_comm_cost # Can be math.inf if no capable rover can do it

                h += min_goal_cost

            elif goal_type == 'communicated_rock_data':
                waypoint = goal_fact[1]
                min_goal_cost = math.inf

                # Check if any rover already has the sample
                rovers_holding_sample = {r for r, wps in held_rock_samples.items() if waypoint in wps}

                if rovers_holding_sample:
                    # Goal: Communicate sample already held
                    min_comm_nav_cost = math.inf
                    for rover in rovers_holding_sample:
                        current_loc = rover_locations.get(rover)
                        if current_loc is None: continue

                        min_path_to_comm = math.inf
                        for comm_wp in self.comm_waypoints:
                            min_path_to_comm = min(min_path_to_comm, self.rover_shortest_paths[rover][current_loc][comm_wp])
                        min_comm_nav_cost = min(min_comm_nav_cost, min_path_to_comm)

                    if min_comm_nav_cost != math.inf:
                         min_goal_cost = 1 + min_comm_nav_cost
                    else:
                         min_goal_cost = math.inf

                else:
                    # Goal: Sample and communicate
                    # Check if sample exists initially
                    if waypoint not in self.initial_rock_samples:
                        min_goal_cost = math.inf # Sample never existed, unreachable
                    else:
                        # Find capable rovers
                        capable_rovers = self.rover_capabilities['rock']
                        if not capable_rovers:
                            min_goal_cost = math.inf # No rover can sample rock
                        else:
                            min_sample_comm_cost = math.inf
                            for rover in capable_rovers:
                                current_loc = rover_locations.get(rover)
                                if current_loc is None: continue

                                nav_cost_to_sample = self.rover_shortest_paths[rover][current_loc][waypoint]
                                if nav_cost_to_sample == math.inf: continue

                                rover_store = None
                                for store, owner in self.store_owner.items():
                                    if owner == rover:
                                        rover_store = store
                                        break
                                # Assume rover has a store if equipped for rock analysis
                                store_cost = 1 if rover_store in store_full else 0

                                min_nav_cost_sample_to_comm = math.inf
                                for comm_wp in self.comm_waypoints:
                                    min_nav_cost_sample_to_comm = min(min_nav_cost_sample_to_comm, self.rover_shortest_paths[rover][waypoint][comm_wp])

                                if min_nav_cost_sample_to_comm == math.inf: continue

                                total_rover_cost = nav_cost_to_sample + 1 + store_cost + min_nav_cost_sample_to_comm + 1

                                min_sample_comm_cost = min(min_sample_comm_cost, total_rover_cost)

                            min_goal_cost = min_sample_comm_cost

                h += min_goal_cost

            elif goal_type == 'communicated_image_data':
                objective, mode = goal_fact[1], goal_fact[2]
                min_goal_cost = math.inf

                # Check if any rover already has the image
                rovers_holding_image = {r for r, images in held_images.items() if (objective, mode) in images}

                if rovers_holding_image:
                    # Goal: Communicate image already held
                    min_comm_nav_cost = math.inf
                    for rover in rovers_holding_image:
                        current_loc = rover_locations.get(rover)
                        if current_loc is None: continue

                        min_path_to_comm = math.inf
                        for comm_wp in self.comm_waypoints:
                            min_path_to_comm = min(min_path_to_comm, self.rover_shortest_paths[rover][current_loc][comm_wp])
                        min_comm_nav_cost = min(min_comm_nav_cost, min_path_to_comm)

                    if min_comm_nav_cost != math.inf:
                         min_goal_cost = 1 + min_comm_nav_cost
                    else:
                         min_goal_cost = math.inf

                else:
                    # Goal: Calibrate, image, and communicate
                    # Find capable rovers and cameras
                    capable_rover_cameras = [] # List of (rover, camera) tuples
                    for rover in self.rover_capabilities['imaging']:
                        for camera in self.rover_cameras.get(rover, set()):
                            if mode in self.camera_modes.get(camera, set()):
                                capable_rover_cameras.append((rover, camera))

                    if not capable_rover_cameras:
                        min_goal_cost = math.inf # No rover/camera can take this image
                    else:
                        min_image_comm_cost = math.inf # Min cost over all capable rover/camera combinations
                        for rover, camera in capable_rover_cameras:
                            current_loc = rover_locations.get(rover)
                            if current_loc is None: continue

                            # Find calibration waypoints for this camera
                            cal_target = self.camera_cal_target.get(camera)
                            cal_waypoints = self.objective_visible_from.get(cal_target, set()) if cal_target else set()
                            if not cal_waypoints: continue # No calibration waypoint

                            # Find image waypoints for this objective
                            image_waypoints = self.objective_visible_from.get(objective, set())
                            if not image_waypoints: continue # No image waypoint

                            # Find communication waypoints
                            comm_waypoints = self.comm_waypoints
                            if not comm_waypoints: continue # No communication waypoint

                            # Calculate minimum sequential navigation cost: current -> cal -> image -> comm
                            min_seq_nav_cost = math.inf
                            for w in cal_waypoints:
                                nav_curr_to_w = self.rover_shortest_paths[rover][current_loc][w]
                                if nav_curr_to_w == math.inf: continue
                                for p in image_waypoints:
                                    nav_w_to_p = self.rover_shortest_paths[rover][w][p]
                                    if nav_w_to_p == math.inf: continue
                                    for x in comm_waypoints:
                                        nav_p_to_x = self.rover_shortest_paths[rover][p][x]
                                        if nav_p_to_x == math.inf: continue
                                        total_nav = nav_curr_to_w + nav_w_to_p + nav_p_to_x
                                        min_seq_nav_cost = min(min_seq_nav_cost, total_nav)

                            if min_seq_nav_cost != math.inf:
                                # Total cost for this rover/camera = nav cost + 3 actions (calibrate, take_image, communicate)
                                total_rover_camera_cost = min_seq_nav_cost + 3
                                min_image_comm_cost = min(min_image_comm_cost, total_rover_camera_cost)

                        min_goal_cost = min_image_comm_cost # Can be math.inf

                h += min_goal_cost

            # If any goal is unreachable, the total heuristic should be infinity
            if h == math.inf:
                 return math.inf

        return h
