from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse facts
def get_parts(fact_string):
    """Removes leading/trailing parens and splits by space."""
    return fact_string[1:-1].split()

# Helper function for BFS
def bfs(graph, start_node):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         return distances

    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes in a graph."""
    all_distances = {}
    for start_node in graph:
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for each individual goal fact that is
    not yet satisfied in the current state. The cost for achieving a single
    goal fact (communicating soil data, rock data, or image data) is estimated
    based on whether the required data/image has already been collected and
    the navigation distance required to get to a location where communication
    with the lander is possible. Navigation distances between waypoints for
    each rover are precomputed using Breadth-First Search (BFS) on the
    rover's specific traversal graph.

    Assumptions:
    - The heuristic assumes that all necessary objects (rovers, cameras, stores,
      waypoints, objectives, modes, landers) exist and are correctly typed
      as inferred from the initial state, goals, and static facts.
    - It assumes that if a soil or rock sample is no longer at its original
      waypoint (`at_soil_sample` or `at_rock_sample` is false), the sample
      must have been successfully collected, and the corresponding
      `have_soil_analysis` or `have_rock_analysis` fact should exist for
      some rover (unless it was already communicated). If the sample is gone
      but the data is neither held nor communicated, the heuristic assigns a
      large penalty, treating this as an unlikely or problematic state.
    - It assumes that if an image goal is not met and the image is not held,
      the process of calibration, imaging, and communication is required.
      It estimates the cost based on navigating to a calibration target,
      calibrating, navigating to the objective's visible waypoint, taking the
      image, navigating to a communication waypoint, and communicating. It
      does not explicitly track camera calibration state across multiple image
      goals for simplicity (taking an image uncalibrates).
    - It assumes the lander location is static and communication is possible
      from any waypoint visible from the lander's location.
    - It uses a large fixed penalty (1000) for unsatisfied goals that appear
      unreachable based on the heuristic's simplified logic (e.g., no equipped
      rover, no path, sample gone but data not held/communicated). This is
      a non-admissible choice suitable for greedy search.

    Heuristic Initialization:
    1.  Collect all unique objects of each type (rover, waypoint, store, camera,
        mode, objective, lander) by parsing the initial state, goals, and
        static facts.
    2.  Parse static facts to populate data structures:
        -   `rover_capabilities`: Which rovers are equipped for soil, rock, or imaging.
        -   `rover_stores`: Which store belongs to which rover.
        -   `rover_waypoint_graph`: Build a graph for each rover based on
            `can_traverse` facts.
        -   `lander_location`: The waypoint where the lander is located.
        -   `comm_waypoints`: The set of waypoints visible from the lander's location.
        -   `camera_info`: For each camera, its assigned rover, supported modes,
            and calibration target objective.
        -   `objective_visibility`: For each objective, the set of waypoints
            from which it is visible.
    3.  For each rover, compute all-pairs shortest path distances between all
        waypoints it can traverse using BFS on its waypoint graph. Store these
        distances in `rover_distances`.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is a goal state. If `self.goals` is a subset
        of the current state, return 0.
    2.  Parse the current state to extract dynamic facts:
        -   Current location of each rover (`at`).
        -   Which rovers have collected soil/rock data (`have_soil_analysis`, `have_rock_analysis`).
        -   Which rovers have collected images (`have_image`).
        -   Status of each store (`empty`, `full`).
        -   Calibration status of cameras (`calibrated`).
        -   Location of remaining soil/rock samples (`at_soil_sample`, `at_rock_sample`).
    3.  Initialize `total_cost` to 0.
    4.  Iterate through each goal fact in `self.goals`. If a goal fact is
        present in the current state, skip it.
    5.  For an unsatisfied goal fact:
        -   **If it's `(communicated_soil_data ?w)`:**
            -   Check if any rover currently holds the soil data for `?w`
                (`have_soil_analysis ?r ?w`).
                -   If yes: The cost is the minimum navigation distance for
                    such a rover from its current location to any communication
                    waypoint, plus 1 (for the `communicate_soil_data` action).
            -   If no rover holds the data:
                -   Check if a soil sample still exists at `?w` (`at_soil_sample ?w`).
                    -   If yes: The cost is estimated as the minimum cost over
                        all equipped rovers. For an equipped rover, this cost is
                        (navigation distance to `?w`) + 1 (`sample_soil`) +
                        (navigation distance from `?w` to a communication waypoint) +
                        1 (`communicate_soil_data`). Add 1 if the rover's store
                        is currently full (for a `drop` action).
                    -   If no: The sample is gone, but data is not held/communicated.
                        Assign a large penalty (1000).
            -   Add the minimum estimated cost (over relevant rovers/cases) for
                this goal to `total_cost`. If no valid path or equipped rover
                found, add the large penalty.
        -   **If it's `(communicated_rock_data ?w)`:** Similar logic as soil data,
            using rock-specific predicates and capabilities.
        -   **If it's `(communicated_image_data ?o ?m)`:**
            -   Check if any rover currently holds the image for `?o` in mode `?m`
                (`have_image ?r ?o ?m`).
                -   If yes: The cost is the minimum navigation distance for
                    such a rover from its current location to any communication
                    waypoint, plus 1 (for the `communicate_image_data` action).
            -   If no rover holds the image:
                -   Find rovers equipped for imaging that have a camera supporting `?m`.
                -   Find waypoints from which `?o` is visible (`visible_from ?o ?p`).
                -   For each suitable rover/camera, find waypoints from which its
                    calibration target is visible (`calibration_target ?i ?t`,
                    `visible_from ?t ?w`).
                -   The cost is estimated as the minimum cost over all suitable
                    rover/camera pairs and valid waypoint combinations (calibration
                    waypoint, image waypoint). For a pair, this cost is
                    (navigation distance to a calibration waypoint) + 1 (`calibrate`) +
                    (navigation distance from calibration waypoint to an image waypoint) +
                    1 (`take_image`) + (navigation distance from image waypoint to a
                    communication waypoint) + 1 (`communicate_image_data`).
            -   Add the minimum estimated cost (over relevant rovers/cameras/waypoints)
                for this goal to `total_cost`. If no suitable rover/camera or
                waypoints found, add the large penalty.
    6.  Return the final `total_cost`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static facts and precomputing
        navigation distances.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Heuristic Initialization ---
        # Parse static facts to build data structures needed for heuristic calculation.
        # This includes rover capabilities, store ownership, waypoint graph,
        # lander location, communication waypoints, camera information, and
        # objective visibility.

        # Collect all unique objects of each type (inferred from facts)
        all_waypoints = set()
        all_rovers = set()
        all_stores = set()
        all_cameras = set()
        all_modes = set()
        all_objectives = set()
        all_landers = set()

        # Iterate through relevant facts to collect objects and types
        for fact in task.initial_state | task.goals | task.static:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred in ['at', 'at_lander', 'can_traverse', 'have_rock_analysis', 'have_soil_analysis', 'visible', 'visible_from']:
                 for part in parts[1:]:
                     if part.startswith('waypoint'): all_waypoints.add(part)
                     elif part.startswith('rover'): all_rovers.add(part)
                     elif part.startswith('lander'): all_landers.add(part)
                     elif part.startswith('objective'): all_objectives.add(part)
            elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                 if len(parts) > 1 and parts[1].startswith('rover'): all_rovers.add(parts[1])
            elif pred in ['empty', 'full', 'store_of']:
                 if len(parts) > 1 and parts[1].startswith('store'): all_stores.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('rover'): all_rovers.add(parts[2])
            elif pred in ['calibrated', 'on_board']:
                 if len(parts) > 1 and parts[1].startswith('camera'): all_cameras.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('rover'): all_rovers.add(parts[2])
            elif pred == 'supports':
                 if len(parts) > 1 and parts[1].startswith('camera'): all_cameras.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('mode'): all_modes.add(parts[2])
            elif pred == 'have_image':
                 if len(parts) > 1 and parts[1].startswith('rover'): all_rovers.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('objective'): all_objectives.add(parts[2])
                 if len(parts) > 3 and parts[3].startswith('mode'): all_modes.add(parts[3])
            elif pred in ['communicated_soil_data', 'communicated_rock_data', 'at_soil_sample', 'at_rock_sample']:
                 if len(parts) > 1 and parts[1].startswith('waypoint'): all_waypoints.add(parts[1])
            elif pred == 'communicated_image_data':
                 if len(parts) > 1 and parts[1].startswith('objective'): all_objectives.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('mode'): all_modes.add(parts[2])
            elif pred == 'calibration_target':
                 if len(parts) > 1 and parts[1].startswith('camera'): all_cameras.add(parts[1])
                 if len(parts) > 2 and parts[2].startswith('objective'): all_objectives.add(parts[2])

        self.all_waypoints = list(all_waypoints)
        self.all_rovers = list(all_rovers)
        self.all_stores = list(all_stores)
        self.all_cameras = list(all_cameras)
        self.all_modes = list(all_modes)
        self.all_objectives = list(all_objectives)
        self.all_landers = list(all_landers)


        # Initialize static data structures
        self.rover_capabilities = {r: {'soil': False, 'rock': False, 'imaging': False} for r in self.all_rovers}
        self.rover_stores = {r: None for r in self.all_rovers} # rover -> store
        self.rover_waypoint_graph = {r: {wp: set() for wp in self.all_waypoints} for r in self.all_rovers}
        self.lander_location = None
        self.comm_waypoints = set()
        self.camera_info = {c: {'rover': None, 'modes': set(), 'cal_target': None} for c in self.all_cameras}
        self.objective_visibility = {o: set() for o in self.all_objectives}

        visible_graph = {wp: set() for wp in self.all_waypoints}

        # Populate static data structures from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'at_lander':
                # Assuming only one lander for simplicity, take the last one found
                if len(parts) > 2:
                    self.lander_location = parts[2]
            elif pred == 'visible':
                if len(parts) > 2:
                    wp1, wp2 = parts[1], parts[2]
                    if wp1 in visible_graph and wp2 in visible_graph:
                        visible_graph[wp1].add(wp2)
                        visible_graph[wp2].add(wp1) # Assuming visible is symmetric
            elif pred == 'can_traverse':
                if len(parts) > 3:
                    rover, wp1, wp2 = parts[1], parts[2], parts[3]
                    if rover in self.rover_waypoint_graph and wp1 in self.rover_waypoint_graph[rover] and wp2 in self.rover_waypoint_graph[rover]:
                         self.rover_waypoint_graph[rover][wp1].add(wp2)
            elif pred == 'equipped_for_soil_analysis':
                if len(parts) > 1 and parts[1] in self.rover_capabilities:
                    self.rover_capabilities[parts[1]]['soil'] = True
            elif pred == 'equipped_for_rock_analysis':
                if len(parts) > 1 and parts[1] in self.rover_capabilities:
                    self.rover_capabilities[parts[1]]['rock'] = True
            elif pred == 'equipped_for_imaging':
                if len(parts) > 1 and parts[1] in self.rover_capabilities:
                    self.rover_capabilities[parts[1]]['imaging'] = True
            elif pred == 'store_of':
                if len(parts) > 2 and parts[2] in self.rover_stores:
                    store, rover = parts[1], parts[2]
                    self.rover_stores[rover] = store # Assuming one store per rover
            elif pred == 'on_board':
                if len(parts) > 2 and parts[1] in self.camera_info and parts[2] in self.all_rovers:
                    camera, rover = parts[1], parts[2]
                    self.camera_info[camera]['rover'] = rover
            elif pred == 'supports':
                if len(parts) > 2 and parts[1] in self.camera_info and parts[2] in self.all_modes:
                    camera, mode = parts[1], parts[2]
                    self.camera_info[camera]['modes'].add(mode)
            elif pred == 'calibration_target':
                if len(parts) > 2 and parts[1] in self.camera_info and parts[2] in self.all_objectives:
                    camera, obj = parts[1], parts[2]
                    self.camera_info[camera]['cal_target'] = obj
            elif pred == 'visible_from':
                if len(parts) > 2 and parts[1] in self.objective_visibility and parts[2] in self.all_waypoints:
                    obj, wp = parts[1], parts[2]
                    self.objective_visibility[obj].add(wp)

        # Compute communication waypoints (visible from lander location)
        if self.lander_location and self.lander_location in visible_graph:
            self.comm_waypoints = set(visible_graph[self.lander_location])

        # Compute rover distances
        self.rover_distances = {}
        for rover in self.all_rovers:
            self.rover_distances[rover] = compute_all_pairs_shortest_paths(self.rover_waypoint_graph[rover])


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            An integer estimate of the remaining actions to reach the goal.
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---
        # 1. Check if the goal is already reached. If so, the heuristic is 0.
        if self.goals <= state:
            return 0

        # 2. Parse the current state to extract relevant dynamic facts.
        rover_locations = {} # rover -> wp
        have_soil = set() # (rover, wp)
        have_rock = set() # (rover, wp)
        have_image = set() # (rover, obj, mode)
        store_status = {} # store -> 'empty' or 'full'
        calibrated_cameras = set() # (camera, rover)
        at_soil_sample = set() # wp
        at_rock_sample = set() # wp

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'at' and len(parts) > 2:
                rover, wp = parts[1], parts[2]
                if rover in self.all_rovers and wp in self.all_waypoints:
                    rover_locations[rover] = wp
            elif pred == 'have_soil_analysis' and len(parts) > 2:
                rover, wp = parts[1], parts[2]
                if rover in self.all_rovers and wp in self.all_waypoints:
                    have_soil.add((rover, wp))
            elif pred == 'have_rock_analysis' and len(parts) > 2:
                rover, wp = parts[1], parts[2]
                if rover in self.all_rovers and wp in self.all_waypoints:
                    have_rock.add((rover, wp))
            elif pred == 'have_image' and len(parts) > 3:
                rover, obj, mode = parts[1], parts[2], parts[3]
                if rover in self.all_rovers and obj in self.all_objectives and mode in self.all_modes:
                    have_image.add((rover, obj, mode))
            elif pred == 'empty' and len(parts) > 1:
                store = parts[1]
                if store in self.all_stores:
                    store_status[store] = 'empty'
            elif pred == 'full' and len(parts) > 1:
                store = parts[1]
                if store in self.all_stores:
                    store_status[store] = 'full'
            elif pred == 'calibrated' and len(parts) > 2:
                camera, rover = parts[1], parts[2]
                if camera in self.all_cameras and rover in self.all_rovers:
                    calibrated_cameras.add((camera, rover))
            elif pred == 'at_soil_sample' and len(parts) > 1:
                wp = parts[1]
                if wp in self.all_waypoints:
                    at_soil_sample.add(wp)
            elif pred == 'at_rock_sample' and len(parts) > 1:
                wp = parts[1]
                if wp in self.all_waypoints:
                    at_rock_sample.add(wp)


        # 3. Initialize total heuristic cost.
        total_cost = 0

        # 4. Iterate through each goal fact and estimate its cost if not satisfied.
        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            # Estimate cost for communicated_soil_data goal
            if pred == 'communicated_soil_data' and len(parts) > 1:
                target_wp = parts[1]
                min_goal_cost = float('inf')

                # Case 1: Data is already collected by some rover
                rovers_with_data = [r for r, wp in have_soil if wp == target_wp]
                if rovers_with_data:
                    for r in rovers_with_data:
                        # Check if rover location is known and is a valid start node in its graph
                        if r in rover_locations and rover_locations[r] in self.rover_distances[r]:
                            r_location = rover_locations[r]
                            # Cost = navigate to comm point + communicate
                            dist_to_comm = float('inf')
                            if self.comm_waypoints:
                                dist_to_comm = min(self.rover_distances[r][r_location].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)

                            if dist_to_comm != float('inf'):
                                cost = dist_to_comm + 1 # communicate
                                min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Data is not collected, sample exists
                elif target_wp in at_soil_sample:
                    # Find equipped rovers
                    equipped_rovers = [r for r in self.all_rovers if self.rover_capabilities[r]['soil']]
                    for r in equipped_rovers:
                        # Check if rover location is known and is a valid start node in its graph
                        if r in rover_locations and rover_locations[r] in self.rover_distances[r]:
                            r_location = rover_locations[r]
                            # Check if sample waypoint is reachable
                            if target_wp in self.rover_distances[r][r_location]:
                                # Cost = navigate to sample + sample + navigate to comm + communicate
                                dist_to_sample = self.rover_distances[r][r_location][target_wp]

                                dist_to_comm = float('inf')
                                if self.comm_waypoints:
                                    # Check if sample waypoint is a valid start node for comm navigation
                                    if target_wp in self.rover_distances[r]:
                                        dist_to_comm = min(self.rover_distances[r][target_wp].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)

                                if dist_to_sample != float('inf') and dist_to_comm != float('inf'):
                                    cost = dist_to_sample + 1 # sample
                                    # Check if drop is needed before sampling
                                    store = self.rover_stores.get(r)
                                    if store and store_status.get(store) == 'full':
                                         cost += 1 # drop
                                    cost += dist_to_comm + 1 # communicate
                                    min_goal_cost = min(min_goal_cost, cost)

                # Add the minimum cost for this goal to the total
                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    # If the goal is unreachable based on this heuristic, assign a large penalty
                    total_cost += 1000 # Penalty for seemingly unreachable goal


            # Estimate cost for communicated_rock_data goal
            elif pred == 'communicated_rock_data' and len(parts) > 1:
                target_wp = parts[1]
                min_goal_cost = float('inf')

                # Case 1: Data is already collected by some rover
                rovers_with_data = [r for r, wp in have_rock if wp == target_wp]
                if rovers_with_data:
                    for r in rovers_with_data:
                        # Check if rover location is known and is a valid start node in its graph
                        if r in rover_locations and rover_locations[r] in self.rover_distances[r]:
                            r_location = rover_locations[r]
                            # Cost = navigate to comm point + communicate
                            dist_to_comm = float('inf')
                            if self.comm_waypoints:
                                dist_to_comm = min(self.rover_distances[r][r_location].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)

                            if dist_to_comm != float('inf'):
                                cost = dist_to_comm + 1 # communicate
                                min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Data is not collected, sample exists
                elif target_wp in at_rock_sample:
                    # Find equipped rovers
                    equipped_rovers = [r for r in self.all_rovers if self.rover_capabilities[r]['rock']]
                    for r in equipped_rovers:
                        # Check if rover location is known and is a valid start node in its graph
                        if r in rover_locations and rover_locations[r] in self.rover_distances[r]:
                            r_location = rover_locations[r]
                            # Check if sample waypoint is reachable
                            if target_wp in self.rover_distances[r][r_location]:
                                # Cost = navigate to sample + sample + navigate to comm + communicate
                                dist_to_sample = self.rover_distances[r][r_location][target_wp]

                                dist_to_comm = float('inf')
                                if self.comm_waypoints:
                                    # Check if sample waypoint is a valid start node for comm navigation
                                    if target_wp in self.rover_distances[r]:
                                        dist_to_comm = min(self.rover_distances[r][target_wp].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)

                                if dist_to_sample != float('inf') and dist_to_comm != float('inf'):
                                    cost = dist_to_sample + 1 # sample
                                    # Check if drop is needed before sampling
                                    store = self.rover_stores.get(r)
                                    if store and store_status.get(store) == 'full':
                                         cost += 1 # drop
                                    cost += dist_to_comm + 1 # communicate
                                    min_goal_cost = min(min_goal_cost, cost)

                # Add the minimum cost for this goal to the total
                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost += 1000 # Penalty


            # Estimate cost for communicated_image_data goal
            elif pred == 'communicated_image_data' and len(parts) > 2:
                target_obj, target_mode = parts[1], parts[2]
                min_goal_cost = float('inf')

                # Case 1: Image is already collected by some rover
                rovers_with_image = [r for r, obj, mode in have_image if obj == target_obj and mode == target_mode]
                if rovers_with_image:
                    for r in rovers_with_image:
                        # Check if rover location is known and is a valid start node in its graph
                        if r in rover_locations and rover_locations[r] in self.rover_distances[r]:
                            r_location = rover_locations[r]
                            # Cost = navigate to comm point + communicate
                            dist_to_comm = float('inf')
                            if self.comm_waypoints:
                                dist_to_comm = min(self.rover_distances[r][r_location].get(comm_wp, float('inf')) for comm_wp in self.comm_waypoints)

                            if dist_to_comm != float('inf'):
                                cost = dist_to_comm + 1 # communicate
                                min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Image is not collected, need to take it
                else:
                    # Find suitable rovers/cameras
                    suitable_rover_cameras = []
                    for r in self.all_rovers:
                        if self.rover_capabilities[r]['imaging']:
                            for c in self.all_cameras:
                                if self.camera_info[c]['rover'] == r and target_mode in self.camera_info[c]['modes']:
                                    suitable_rover_cameras.append((r, c))

                    # Find suitable image waypoints
                    image_wps = self.objective_visibility.get(target_obj, set())

                    if suitable_rover_cameras and image_wps:
                        for r, c in suitable_rover_cameras:
                            # Check if rover location is known and is a valid start node in its graph
                            if r not in rover_locations or rover_locations[r] not in self.rover_distances[r]:
                                continue # Rover not at a known location or cannot navigate from there

                            r_location = rover_locations[r]
                            cal_target = self.camera_info[c]['cal_target']
                            cal_wps = self.objective_visibility.get(cal_target, set())

                            if cal_wps:
                                # Cost = nav to cal + calibrate + nav to image + take image + nav to comm + communicate
                                # Find min dist from current location to any cal waypoint
                                min_dist_to_cal = float('inf')
                                for cal_wp in cal_wps:
                                    if cal_wp in self.rover_distances[r][r_location]: # Check if path exists from r_location
                                        min_dist_to_cal = min(min_dist_to_cal, self.rover_distances[r][r_location][cal_wp])

                                # Find min dist from any cal waypoint to any image waypoint
                                min_dist_cal_to_image = float('inf')
                                for cal_wp in cal_wps:
                                    if cal_wp in self.rover_distances[r]: # Check if cal_wp is a valid start node for this rover
                                        for image_wp in image_wps:
                                            if image_wp in self.rover_distances[r][cal_wp]: # Check if path exists from cal_wp
                                                min_dist_cal_to_image = min(min_dist_cal_to_image, self.rover_distances[r][cal_wp][image_wp])

                                # Find min dist from any image waypoint to any comm waypoint
                                min_dist_image_to_comm = float('inf')
                                if self.comm_waypoints:
                                    for image_wp in image_wps:
                                        if image_wp in self.rover_distances[r]: # Check if image_wp is a valid start node
                                            for comm_wp in self.comm_waypoints:
                                                if comm_wp in self.rover_distances[r][image_wp]: # Check if path exists from image_wp
                                                    min_dist_image_to_comm = min(min_dist_image_to_comm, self.rover_distances[r][image_wp][comm_wp])


                                if min_dist_to_cal != float('inf') and min_dist_cal_to_image != float('inf') and min_dist_image_to_comm != float('inf'):
                                    cost = min_dist_to_cal + 1 # calibrate
                                    cost += min_dist_cal_to_image + 1 # take image
                                    cost += min_dist_image_to_comm + 1 # communicate
                                    min_goal_cost = min(min_goal_cost, cost)

                # Add the minimum cost for this goal to the total
                if min_goal_cost != float('inf'):
                    total_cost += min_goal_cost
                else:
                    total_cost += 1000 # Penalty


        # 5. Return the total estimated cost.
        return total_cost
