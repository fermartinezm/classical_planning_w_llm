from collections import deque
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...).
    e.g., '(at rover1 waypoint1)' -> ('at', 'rover1', 'waypoint1')
    """
    # Remove surrounding brackets and split by spaces
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return (predicate, *args)

# Helper function for Breadth-First Search
def bfs(graph, start_node):
    """
    Performs BFS on a graph to find shortest distances from a start node.

    Args:
        graph: Adjacency dictionary {node: [neighbor1, neighbor2, ...]}
        start_node: The starting node for BFS.

    Returns:
        A dictionary {node: distance} containing shortest distances.
        Distance is float('inf') if a node is unreachable.
    """
    # Ensure all nodes in the graph are initialized with infinite distance
    distances = {node: float('inf') for node in graph}

    # If start_node is not even a valid node in the graph, return empty distances
    if start_node not in graph:
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph (should be true if it's a key)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

# Helper function to precompute all-pairs shortest paths for each rover
def precompute_distances(rovers, waypoints, rover_traversal_graph):
    """
    Precomputes shortest path distances between all pairs of waypoints
    for each rover based on their traversal capabilities.

    Args:
        rovers: Set of all rover names.
        waypoints: List or set of all waypoint names.
        rover_traversal_graph: Dictionary mapping rover name to its
                                traversal graph (adjacency dictionary).

    Returns:
        A dictionary {rover: {start_wp: {end_wp: distance}}}.
    """
    rover_pairwise_distances = {}
    all_waypoints_list = list(waypoints) # Use list for consistent iteration

    for rover in rovers:
        rover_pairwise_distances[rover] = {}
        # Build the graph for the current rover, ensuring all waypoints are keys
        # even if they have no outgoing edges, so BFS initializes distances correctly.
        graph = {wp: rover_traversal_graph.get(rover, {}).get(wp, []) for wp in all_waypoints_list}

        for start_wp in all_waypoints_list:
            rover_pairwise_distances[rover][start_wp] = bfs(graph, start_wp)
    return rover_pairwise_distances

# Helper function to get distance from precomputed table
def get_distance(rover_pairwise_distances, rover, start_wp, end_wp):
    """
    Looks up the precomputed shortest distance for a rover between two waypoints.

    Args:
        rover_pairwise_distances: The precomputed distance table.
        rover: The rover name.
        start_wp: The starting waypoint.
        end_wp: The ending waypoint.

    Returns:
        The shortest distance (integer) or float('inf') if unreachable
        or rover/waypoint is invalid or not in the precomputed data.
    """
    if rover not in rover_pairwise_distances or start_wp not in rover_pairwise_distances[rover]:
        return float('inf') # Rover or start waypoint not in precomputed data
    return rover_pairwise_distances[rover][start_wp].get(end_wp, float('inf'))


class roversHeuristic:
    """
    Domain-dependent heuristic for the rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs
    for each unachieved goal. The cost for each goal is estimated by finding
    the minimum cost sequence of navigation and action steps required,
    considering rover capabilities, camera requirements, sample locations,
    objective visibility, calibration targets, and communication points.
    Navigation costs are estimated using precomputed shortest path distances
    on the rover-specific traversal graphs. The heuristic approximates the
    rover's location after completing a stage (e.g., calibration, imaging)
    to calculate the navigation cost for the next stage.

    Assumptions:
    - The initial state contains all necessary information about sample locations (`at_soil_sample`, `at_rock_sample`) that exist at the start.
    - The static facts contain all necessary information about rover capabilities, store ownership, camera properties, visibility, traversal, calibration targets, and objective visibility.
    - Goals only involve `communicated_soil_data`, `communicated_rock_data`, and `communicated_image_data`.
    - Rovers can only traverse according to `can_traverse` facts.
    - The lander location is static.

    Heuristic Initialization:
    - Parses static facts and initial state facts to identify all objects (rovers, waypoints, etc.) and build static data structures:
        - `all_waypoints`: Set of all waypoints found in facts.
        - `rovers`, `stores`, `cameras`, `modes`, `landers`, `objectives`: Sets of object names by type.
        - `lander_location`: Maps lander to its waypoint.
        - `rover_capabilities`: Maps rover to a set of capabilities ('soil', 'rock', 'imaging').
        - `store_rover_map`: Maps store to its rover.
        - `camera_rover_map`: Maps camera to its rover.
        - `camera_modes`: Maps camera to a set of supported modes.
        - `camera_cal_target`: Maps camera to its calibration target objective.
        - `objective_image_wps`: Maps objective to a set of waypoints visible from.
        - `rover_traversal_graph`: Maps rover to its waypoint adjacency list based on `can_traverse`.
        - `waypoint_visibility_graph`: Waypoint adjacency list based on `visible`.
        - `initial_soil_samples`: Set of waypoints with soil samples in the initial state.
        - `initial_rock_samples`: Set of waypoints with rock samples in the initial state.
    - Identifies communication waypoints (`comm_wps`): Waypoints visible from the lander location.
    - Precomputes shortest path distances between all pairs of waypoints for each rover using BFS on the `rover_traversal_graph`. Stores results in `rover_pairwise_distances`.
    """
    def __init__(self, task):
        self.task = task

        # --- Parse Static and Initial Facts ---
        self.all_waypoints = set()
        self.rovers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        self.lander_location = {} # lander -> waypoint
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.store_rover_map = {} # store -> rover
        self.camera_rover_map = {} # camera -> rover
        self.camera_modes = {} # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective
        self.objective_image_wps = {} # objective -> set of waypoints
        self.rover_traversal_graph = {} # rover -> {wp: [neighbor_wp, ...]}
        self.waypoint_visibility_graph = {} # wp -> [visible_wp, ...]
        self.initial_soil_samples = set() # waypoint
        self.initial_rock_samples = set() # waypoint

        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact_string in all_facts:
            parsed = parse_fact(fact_string)
            pred = parsed[0]
            args = parsed[1:]

            # Collect objects by type (based on predicate usage)
            if pred == 'at':
                self.rovers.add(args[0])
                self.all_waypoints.add(args[1])
            elif pred == 'at_lander':
                self.landers.add(args[0])
                self.all_waypoints.add(args[1])
                self.lander_location[args[0]] = args[1]
            elif pred == 'can_traverse':
                self.rovers.add(args[0])
                self.all_waypoints.add(args[1])
                self.all_waypoints.add(args[2])
                rover, wp1, wp2 = args
                self.rover_traversal_graph.setdefault(rover, {}).setdefault(wp1, []).append(wp2)
            elif pred == 'equipped_for_soil_analysis':
                self.rovers.add(args[0])
                self.rover_capabilities.setdefault(args[0], set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rovers.add(args[0])
                self.rover_capabilities.setdefault(args[0], set()).add('rock')
            elif pred == 'equipped_for_imaging':
                self.rovers.add(args[0])
                self.rover_capabilities.setdefault(args[0], set()).add('imaging')
            elif pred == 'empty' or pred == 'full':
                self.stores.add(args[0])
            elif pred == 'have_rock_analysis' or pred == 'have_soil_analysis':
                self.rovers.add(args[0])
                self.all_waypoints.add(args[1])
            elif pred == 'calibrated':
                self.cameras.add(args[0])
                self.rovers.add(args[1])
            elif pred == 'supports':
                self.cameras.add(args[0])
                self.modes.add(args[1])
                self.camera_modes.setdefault(args[0], set()).add(args[1])
            elif pred == 'visible':
                self.all_waypoints.add(args[0])
                self.all_waypoints.add(args[1])
                self.waypoint_visibility_graph.setdefault(args[0], []).append(args[1])
            elif pred == 'have_image':
                self.rovers.add(args[0])
                self.objectives.add(args[1])
                self.modes.add(args[2])
            elif pred == 'communicated_soil_data' or pred == 'communicated_rock_data':
                self.all_waypoints.add(args[0])
            elif pred == 'communicated_image_data':
                self.objectives.add(args[0])
                self.modes.add(args[1])
            elif pred == 'at_soil_sample':
                self.all_waypoints.add(args[0])
                # This fact is in initial state if sample exists
                if fact_string in task.initial_state:
                    self.initial_soil_samples.add(args[0])
            elif pred == 'at_rock_sample':
                self.all_waypoints.add(args[0])
                 # This fact is in initial state if sample exists
                if fact_string in task.initial_state:
                    self.initial_rock_samples.add(args[0])
            elif pred == 'visible_from':
                self.objectives.add(args[0])
                self.all_waypoints.add(args[1])
                self.objective_image_wps.setdefault(args[0], set()).add(args[1])
            elif pred == 'store_of':
                self.stores.add(args[0])
                self.rovers.add(args[1])
                self.store_rover_map[args[0]] = args[1]
            elif pred == 'calibration_target':
                self.cameras.add(args[0])
                self.objectives.add(args[1])
                self.camera_cal_target[args[0]] = args[1]
            elif pred == 'on_board':
                self.cameras.add(args[0])
                self.rovers.add(args[1])
                self.camera_rover_map[args[0]] = args[1]

        # Identify communication waypoints (visible from any lander location)
        self.comm_wps = set()
        lander_wps = set(self.lander_location.values())
        for lander_wp in lander_wps:
             if lander_wp in self.waypoint_visibility_graph:
                 for visible_wp in self.waypoint_visibility_graph[lander_wp]:
                     # Visibility is symmetric in this domain example, but check both directions
                     if visible_wp in self.waypoint_visibility_graph and lander_wp in self.waypoint_visibility_graph[visible_wp]:
                         self.comm_wps.add(visible_wp)
             # Also check visibility the other way around, as communicate action requires rover_at visible_from lander_at
             for wp in self.all_waypoints:
                 if wp in self.waypoint_visibility_graph and lander_wp in self.waypoint_visibility_graph[wp]:
                     self.comm_wps.add(wp)


        # Precompute shortest path distances for each rover
        self.rover_pairwise_distances = precompute_distances(self.rovers, self.all_waypoints, self.rover_traversal_graph)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            An integer estimating the cost to reach the goal.
            Returns a large integer if the goal is unreachable.
        """
        # 1. Check if the state is a goal state
        if self.task.goal_reached(state):
            return 0

        # 2. Initialize total heuristic value
        h = 0

        # 3. Extract dynamic information from the current state
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil_analysis_facts = set()
        have_rock_analysis_facts = set()
        have_image_facts = set()
        calibrated_facts = set()

        for fact_string in state:
            parsed = parse_fact(fact_string)
            pred = parsed[0]
            args = parsed[1:]
            if pred == 'at' and args[0] in self.rovers: # Ensure it's a rover, not lander
                rover_locations[args[0]] = args[1]
            elif pred == 'empty':
                store_status[args[0]] = 'empty'
            elif pred == 'full':
                store_status[args[0]] = 'full'
            elif pred == 'have_soil_analysis':
                have_soil_analysis_facts.add(fact_string)
            elif pred == 'have_rock_analysis':
                have_rock_analysis_facts.add(fact_string)
            elif pred == 'have_image':
                have_image_facts.add(fact_string)
            elif pred == 'calibrated':
                calibrated_facts.add(fact_string)

        # Ensure all rovers have a known location (should be in initial state)
        for r in self.rovers:
            if r not in rover_locations:
                 # This state is likely invalid or malformed, treat as unreachable goal
                 return 1000000


        # 4. Iterate through each goal fact
        for goal_fact in self.task.goals:
            # 5. If the goal fact is already in the state, continue
            if goal_fact in state:
                continue

            # 6. Parse the goal fact
            (pred, *args) = parse_fact(goal_fact)

            # 7. If the goal is (communicated_soil_data w)
            if pred == 'communicated_soil_data':
                w = args[0]
                min_goal_cost = float('inf')

                # Check if sample existed initially
                if w not in self.initial_soil_samples:
                    min_goal_cost = float('inf') # Impossible goal
                else:
                    # Iterate through each rover
                    for r in self.rovers:
                        # If rover is equipped for soil analysis
                        if 'soil' in self.rover_capabilities.get(r, set()):
                            current_loc = rover_locations[r]

                            # Calculate cost to get soil analysis
                            if (f'(have_soil_analysis {r} {w})') in have_soil_analysis_facts:
                                cost_get_sample = 0
                                loc_after_sample = current_loc
                            else:
                                nav_to_sample = get_distance(self.rover_pairwise_distances, r, current_loc, w)
                                if nav_to_sample == float('inf'): continue # Cannot reach sample wp

                                cost_sample_action = 1
                                store_for_r = None
                                for store_name, rover_name in self.store_rover_map.items():
                                    if rover_name == r:
                                        store_for_r = store_name
                                        break

                                if store_for_r is None: # Rover has no store?
                                     cost_drop = float('inf') # Impossible path
                                else:
                                     cost_drop = 1 if store_status.get(store_for_r) == 'full' else 0

                                if cost_drop == float('inf'): continue # Cannot drop

                                cost_get_sample = nav_to_sample + cost_sample_action + cost_drop
                                loc_after_sample = w # Rover is at sample waypoint after sampling

                            # Calculate cost to communicate
                            nav_to_comm = min((get_distance(self.rover_pairwise_distances, r, loc_after_sample, comm_wp) for comm_wp in self.comm_wps), default=float('inf'))
                            if nav_to_comm == float('inf'): continue # Cannot reach comm wp

                            cost_comm_action = 1
                            cost_communicate = nav_to_comm + cost_comm_action

                            # Total cost for rover r
                            total_cost_for_r = cost_get_sample + cost_communicate
                            min_goal_cost = min(min_goal_cost, total_cost_for_r)

                h += min_goal_cost

            # 9. If the goal is (communicated_rock_data w)
            elif pred == 'communicated_rock_data':
                w = args[0]
                min_goal_cost = float('inf')

                # Check if sample existed initially
                if w not in self.initial_rock_samples:
                    min_goal_cost = float('inf') # Impossible goal
                else:
                    # Iterate through each rover
                    for r in self.rovers:
                        # If rover is equipped for rock analysis
                        if 'rock' in self.rover_capabilities.get(r, set()):
                            current_loc = rover_locations[r]

                            # Calculate cost to get rock analysis
                            if (f'(have_rock_analysis {r} {w})') in have_rock_analysis_facts:
                                cost_get_sample = 0
                                loc_after_sample = current_loc
                            else:
                                nav_to_sample = get_distance(self.rover_pairwise_distances, r, current_loc, w)
                                if nav_to_sample == float('inf'): continue # Cannot reach sample wp

                                cost_sample_action = 1
                                store_for_r = None
                                for store_name, rover_name in self.store_rover_map.items():
                                    if rover_name == r:
                                        store_for_r = store_name
                                        break

                                if store_for_r is None: # Rover has no store?
                                     cost_drop = float('inf') # Impossible path
                                else:
                                     cost_drop = 1 if store_status.get(store_for_r) == 'full' else 0

                                if cost_drop == float('inf'): continue # Cannot drop

                                cost_get_sample = nav_to_sample + cost_sample_action + cost_drop
                                loc_after_sample = w # Rover is at sample waypoint after sampling

                            # Calculate cost to communicate
                            nav_to_comm = min((get_distance(self.rover_pairwise_distances, r, loc_after_sample, comm_wp) for comm_wp in self.comm_wps), default=float('inf'))
                            if nav_to_comm == float('inf'): continue # Cannot reach comm wp

                            cost_comm_action = 1
                            cost_communicate = nav_to_comm + cost_comm_action

                            # Total cost for rover r
                            total_cost_for_r = cost_get_sample + cost_communicate
                            min_goal_cost = min(min_goal_cost, total_cost_for_r)

                h += min_goal_cost

            # 10. If the goal is (communicated_image_data o m)
            elif pred == 'communicated_image_data':
                o = args[0]
                m = args[1]
                min_goal_cost = float('inf')

                # Find image waypoints
                W_img = self.objective_image_wps.get(o, set())
                if not W_img: # No waypoints to view objective from
                    min_goal_cost = float('inf') # Impossible goal
                else:
                    # Iterate through each rover
                    for r in self.rovers:
                        # If rover is equipped for imaging
                        if 'imaging' in self.rover_capabilities.get(r, set()):
                            current_loc = rover_locations[r]

                            # Iterate through each camera
                            for i in self.cameras:
                                # If camera is on board rover and supports mode
                                if self.camera_rover_map.get(i) == r and m in self.camera_modes.get(i, set()):

                                    # Calculate cost to get image
                                    if (f'(have_image {r} {o} {m})') in have_image_facts:
                                        cost_get_image = 0
                                        loc_after_image = current_loc
                                    else:
                                        cost_get_image_val = float('inf')
                                        loc_after_image_val = None

                                        # Find calibration target and waypoints
                                        t = self.camera_cal_target.get(i)
                                        if t is None: continue # Camera has no calibration target
                                        W_cal = self.objective_image_wps.get(t, set())
                                        if not W_cal: continue # No waypoints to view calibration target from

                                        if (f'(calibrated {i} {r})') in calibrated_facts:
                                            # Option: Nav to image wp, Take image
                                            nav_to_image = min((get_distance(self.rover_pairwise_distances, r, current_loc, p) for p in W_img), default=float('inf'))
                                            if nav_to_image != float('inf'):
                                                cost_get_image_val = nav_to_image + 1
                                                # Location after image is the reachable image wp closest to current_loc
                                                loc_after_image_val = min((p for p in W_img if get_distance(self.rover_pairwise_distances, r, current_loc, p) != float('inf')), key=lambda p: get_distance(self.rover_pairwise_distances, r, current_loc, p), default=None)

                                        else: # Not calibrated
                                            # Option A: Calibrate then Image
                                            nav_to_cal = min((get_distance(self.rover_pairwise_distances, r, current_loc, w) for w in W_cal), default=float('inf'))
                                            if nav_to_cal != float('inf'):
                                                # Location after cal is the reachable cal wp closest to current_loc
                                                loc_after_cal = min((w for w in W_cal if get_distance(self.rover_pairwise_distances, r, current_loc, w) != float('inf')), key=lambda w: get_distance(self.rover_pairwise_distances, r, current_loc, w), default=None)
                                                if loc_after_cal is not None:
                                                    nav_from_cal_to_image = min((get_distance(self.rover_pairwise_distances, r, loc_after_cal, p) for p in W_img), default=float('inf'))
                                                    if nav_from_cal_to_image != float('inf'):
                                                        cost_option_A = nav_to_cal + 1 + nav_from_cal_to_image + 1
                                                        # Location after image is the reachable image wp closest to loc_after_cal
                                                        loc_after_image_A = min((p for p in W_img if get_distance(self.rover_pairwise_distances, r, loc_after_cal, p) != float('inf')), key=lambda p: get_distance(self.rover_pairwise_distances, r, loc_after_cal, p), default=None)
                                                        if cost_option_A < cost_get_image_val:
                                                            cost_get_image_val = cost_option_A
                                                            loc_after_image_val = loc_after_image_A

                                            # Option C: Calibrate and Image at same wp
                                            W_cal_img = W_cal.intersection(W_img)
                                            if W_cal_img:
                                                nav_same = min((get_distance(self.rover_pairwise_distances, r, current_loc, p) for p in W_cal_img), default=float('inf'))
                                                if nav_same != float('inf'):
                                                    cost_option_C = nav_same + 1 + 1
                                                    # Location after image is the reachable cal/image wp closest to current_loc
                                                    loc_after_image_C = min((p for p in W_cal_img if get_distance(self.rover_pairwise_distances, r, current_loc, p) != float('inf')), key=lambda p: get_distance(self.rover_pairwise_distances, r, current_loc, p), default=None)
                                                    if cost_option_C < cost_get_image_val:
                                                        cost_get_image_val = cost_option_C
                                                        loc_after_image_val = loc_after_image_C

                                        cost_get_image = cost_get_image_val
                                        loc_after_image = loc_after_image_val

                                    if loc_after_image is None or cost_get_image == float('inf'): continue # Cannot get image

                                    # Calculate cost to communicate
                                    nav_to_comm = min((get_distance(self.rover_pairwise_distances, r, loc_after_image, comm_wp) for comm_wp in self.comm_wps), default=float('inf'))
                                    if nav_to_comm == float('inf'): continue # Cannot reach comm wp

                                    cost_comm_action = 1
                                    cost_communicate = nav_to_comm + cost_comm_action

                                    # Total cost for (r, i) pair
                                    total_cost_for_r_i = cost_get_image + cost_communicate
                                    min_goal_cost = min(min_goal_cost, total_cost_for_r_i)

                h += min_goal_cost

        # 11. If h is infinity, return a large integer, otherwise return int(h)
        if h == float('inf'):
            return 1000000 # Represents unreachable goal
        return int(h)

