from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and ensure it's a string
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # This might happen for non-fact elements if they creep in, or malformed facts
         # Assuming inputs are well-formed facts from the planner state/task
         return [] # Return empty list for non-fact strings
    return fact_str[1:-1].split()

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

def bfs(graph, start_node, all_nodes):
    """
    Performs BFS to find shortest path distances from start_node to all other nodes.
    Returns a dictionary {node: distance}. Unreachable nodes have distance infinity.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        # Start node is not a known waypoint, cannot navigate from here
        return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Optimization: If distance is already large, stop exploring from here
        # This is a non-admissible heuristic, so pruning is acceptable for speed
        # if distances[current_node] > some_threshold: continue # Optional threshold

        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'): # Check if not visited
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve each
    unmet goal condition independently, and sums these estimates. It considers
    the cost of sampling (soil/rock), imaging (including calibration),
    dropping samples (if store is full), and communicating data, plus the
    estimated navigation cost to reach necessary locations (sample site,
    image location, calibration target location, communication location).
    Navigation cost is estimated using precomputed shortest paths on the
    rover-specific traversal graphs.

    # Assumptions
    - All actions have a unit cost of 1.
    - Navigation cost between waypoints is the number of 'navigate' actions
      along the shortest path for a specific rover.
    - Calibration is always needed before taking an image with a camera
      unless the camera is currently calibrated (though take_image uncalibrates).
      For simplicity, this heuristic assumes calibration adds 1 action cost
      if the image goal is not yet achieved, regardless of current calibration status,
      as recalibration is often needed for subsequent images.
    - Dropping a sample costs 1 action if the rover's store is full and a
      new sample needs to be taken.
    - If a required sample is no longer at its initial location and no rover
      currently holds the analysis, the heuristic assumes this goal path is
      unreachable via sampling and contributes 0 cost for that path.
    - If a required image cannot be taken (e.g., no equipped rover, no camera,
      no visible_from waypoint), the heuristic assumes this goal path is
      unreachable and contributes 0 cost for that path.
    - The heuristic does not consider interactions between goals (e.g., using
      the same rover for multiple tasks, sharing stores, path planning
      efficiency for multiple locations). It sums independent goal costs.
    - The heuristic is non-admissible.

    # Heuristic Initialization
    - Parses static facts to identify lander locations, rover capabilities,
      camera information (on-board rover, supported modes, calibration target),
      objective/target visibility from waypoints, waypoint visibility graph,
      and rover-specific traversal graphs.
    - Infers object types (rovers, waypoints, cameras, etc.) from the facts
      present in the initial state and static facts.
    - Precomputes all-pairs shortest path navigation costs for each rover
      based on its traversal graph.
    - Identifies waypoints from which communication with the lander is possible.
    - Stores the set of required goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all goal conditions that are not yet satisfied in the current state.
    2.  For each unsatisfied goal:
        -   Initialize the estimated cost for this goal to infinity.
        -   **If the goal is `(communicated_soil_data ?w)` or `(communicated_rock_data ?w)`:**
            -   Check if any rover `?r` currently has the required analysis `(have_X_analysis ?r ?w)`.
                -   If yes for any `?r`: Calculate the minimum cost to navigate `?r` from its current location to any waypoint visible from the lander, plus 1 for the communicate action. Update the goal cost with the minimum found.
            -   Else (no rover has the analysis): Check if the sample `(at_X_sample ?w)` is still at the waypoint `?w`.
                -   If yes: For each rover `?r` equipped for this analysis type:
                    -   Calculate navigation cost from `?r`'s current location to `?w`.
                    -   Determine if a 'drop' action is needed (cost 1) before sampling (if `?r`'s store is full).
                    -   Calculate the minimum navigation cost from `?w` to any waypoint visible from the lander.
                    -   Sum these navigation costs, drop cost, 1 for sample, and 1 for communicate. Update the goal cost with the minimum found over all suitable rovers.
            -   If the sample is gone and no rover has the analysis, this path to the goal is considered unreachable by sampling from this state.
        -   **If the goal is `(communicated_image_data ?o ?m)`:**
            -   Check if any rover `?r` currently has the required image `(have_image ?r ?o ?m)`.
                -   If yes for any `?r`: Calculate the minimum cost to navigate `?r` from its current location to any waypoint visible from the lander, plus 1 for the communicate action. Update the goal cost with the minimum found.
            -   Else (no rover has the image): For each combination of a suitable rover `?r` (equipped for imaging), camera `?i` (on `?r`, supports `?m`), waypoint `?p` (`visible_from ?o ?p`), calibration target `?t` (for `?i`), and waypoint `?w` (`visible_from ?t ?w`):
                -   Calculate navigation cost from `?r`'s current location to `?w` (calibration waypoint).
                -   Add 1 for the calibrate action.
                -   Calculate navigation cost from `?w` to `?p` (image waypoint).
                -   Add 1 for the take_image action.
                -   Calculate the minimum navigation cost from `?p` to any waypoint visible from the lander.
                -   Add 1 for the communicate action.
                -   Sum these costs. Update the goal cost with the minimum found over all suitable combinations.
            -   If no suitable combination of rover/camera/waypoints exists, this path is considered unreachable.
        -   If the goal cost is still infinity after considering all relevant paths (having the data vs needing to collect it), it means the goal is likely unreachable from this state by the actions considered by the heuristic. In this case, add 0 to the total cost (a non-admissible choice to avoid infinite values).
        -   Otherwise, add the calculated finite goal cost to the total cost.
    3.  Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Infer Object Types and Collect All Objects ---
        # This is a robust way to get all objects of relevant types mentioned in the problem
        all_objects_set = set()
        rovers = set()
        waypoints = set()
        landers = set()
        stores = set()
        cameras = set()
        modes = set()
        objectives = set() # Includes calibration targets

        # Helper to add objects based on predicate structure
        def identify_objects_and_types(fact_parts):
            if not fact_parts: return
            pred = fact_parts[0]
            args = fact_parts[1:]
            all_objects_set.update(args)

            # Basic type inference based on predicate argument positions
            if pred in ['at', 'can_traverse', 'equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging', 'store_of', 'have_soil_analysis', 'have_rock_analysis', 'have_image', 'calibrated', 'on_board']:
                 if len(args) > 0: rovers.add(args[0])
            if pred in ['at', 'at_lander', 'can_traverse', 'visible', 'have_soil_analysis', 'have_rock_analysis', 'at_soil_sample', 'at_rock_sample', 'visible_from']:
                 if len(args) > 0: waypoints.add(args[-1]) # Often the last argument is a waypoint
                 if pred in ['can_traverse', 'visible'] and len(args) > 1: waypoints.add(args[0])
                 if pred in ['at', 'at_lander'] and len(args) > 0: waypoints.add(args[1]) # Specific for at/at_lander
                 if pred in ['have_soil_analysis', 'have_rock_analysis'] and len(args) > 1: waypoints.add(args[1]) # Specific for have_analysis
            if pred == 'at_lander' and len(args) > 0: landers.add(args[0])
            if pred in ['empty', 'full', 'store_of'] and len(args) > 0: stores.add(args[0])
            if pred in ['calibrated', 'on_board', 'supports', 'calibration_target'] and len(args) > 0: cameras.add(args[0])
            if pred in ['supports', 'have_image', 'communicated_image_data'] and len(args) > 1: modes.add(args[-1]) # Often the last argument is a mode
            if pred in ['calibration_target', 'have_image', 'visible_from', 'communicated_image_data'] and len(args) > 0: objectives.add(args[0]) # Often the first argument is an objective/target
            if pred in ['have_image', 'communicated_image_data'] and len(args) > 1: objectives.add(args[1]) # Specific for image data

        for fact in task.initial_state:
            try:
                identify_objects_and_types(get_parts(fact))
            except ValueError:
                pass # Ignore malformed facts if any

        for fact in task.static:
             try:
                identify_objects_and_types(get_parts(fact))
             except ValueError:
                pass # Ignore malformed facts if any

        # Filter objects by type - an object might be inferred as multiple types,
        # but we primarily need sets of known rovers, waypoints, etc.
        self.all_waypoints = list(waypoints)
        self.all_rovers = list(rovers)
        self.all_cameras = list(cameras)
        self.all_objectives = list(objectives)
        self.all_modes = list(modes)
        self.all_stores = list(stores)
        self.all_landers = list(landers)


        # --- Parse Static Facts ---
        self.lander_locations = set()
        self.rover_capabilities = defaultdict(set) # rover -> {capabilities}
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> (rover, {modes}, calibration_target)
        self.objective_visibility = defaultdict(set) # objective -> {waypoints visible from}
        self.calibration_target_map = {} # camera -> calibration_target
        self.calibration_target_visibility = defaultdict(set) # calibration_target -> {waypoints visible from}
        self.waypoint_visibility_graph = defaultdict(set) # waypoint -> {visible neighbors}
        self.rover_traversal_graphs = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> {traversable neighbors}
        self.initial_soil_samples = set() # waypoint
        self.initial_rock_samples = set() # waypoint

        camera_modes_temp = defaultdict(set)
        camera_rover_temp = {}
        camera_target_temp = {}

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            pred = parts[0]
            args = parts[1:]

            if pred == 'at_lander' and len(args) == 2:
                self.lander_locations.add(args[1])
            elif pred.startswith('equipped_for_') and len(args) == 1:
                self.rover_capabilities[args[0]].add(pred)
            elif pred == 'store_of' and len(args) == 2:
                self.rover_stores[args[1]] = args[0] # rover -> store
            elif pred == 'on_board' and len(args) == 2:
                camera_rover_temp[args[0]] = args[1] # camera -> rover
            elif pred == 'supports' and len(args) == 2:
                camera_modes_temp[args[0]].add(args[1]) # camera -> {modes}
            elif pred == 'calibration_target' and len(args) == 2:
                camera_target_temp[args[0]] = args[1] # camera -> target
            elif pred == 'visible' and len(args) == 2:
                self.waypoint_visibility_graph[args[0]].add(args[1])
                # Assuming visible is symmetric for graph construction
                self.waypoint_visibility_graph[args[1]].add(args[0])
            elif pred == 'can_traverse' and len(args) == 3:
                self.rover_traversal_graphs[args[0]][args[1]].add(args[2])
                # Assuming can_traverse is symmetric for graph construction
                self.rover_traversal_graphs[args[0]][args[2]].add(args[1])
            elif pred == 'at_soil_sample' and len(args) == 1:
                self.initial_soil_samples.add(args[0])
            elif pred == 'at_rock_sample' and len(args) == 1:
                self.initial_rock_samples.add(args[0])
            elif pred == 'visible_from' and len(args) == 2:
                self.objective_visibility[args[0]].add(args[1]) # objective/target -> {waypoints}

        # Consolidate camera info
        for camera in self.all_cameras:
             rover = camera_rover_temp.get(camera)
             modes = camera_modes_temp.get(camera, set())
             target = camera_target_temp.get(camera)
             if rover: # Only add cameras that are on board a rover
                 self.camera_info[camera] = (rover, modes, target)
                 if target:
                     # Calibration target visibility is the same as objective visibility
                     self.calibration_target_visibility[target] = self.objective_visibility.get(target, set())


        # --- Precompute Navigation Costs ---
        self.rover_nav_costs = {} # rover -> from_wp -> to_wp -> cost
        for rover in self.all_rovers:
            self.rover_nav_costs[rover] = {}
            graph = self.rover_traversal_graphs.get(rover, defaultdict(set))
            for start_wp in self.all_waypoints:
                 self.rover_nav_costs[rover][start_wp] = bfs(graph, start_wp, self.all_waypoints)

        # --- Identify Communication Waypoints ---
        self.comm_waypoints = set()
        # A waypoint X is a communication waypoint if it's visible from any lander location Y
        for lander_loc in self.lander_locations:
             # Need to find X such that (visible X lander_loc) is true
             # Check waypoints that have lander_loc in their visible set
             for wp in self.all_waypoints:
                 if lander_loc in self.waypoint_visibility_graph.get(wp, set()):
                     self.comm_waypoints.add(wp)
             # Also check if lander_loc can see other waypoints (if visible is symmetric)
             if lander_loc in self.waypoint_visibility_graph:
                 self.comm_waypoints.update(self.waypoint_visibility_graph[lander_loc])


        # --- Store Required Goals ---
        self.required_goals = set(task.goals)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State Facts ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        rover_soil_samples = defaultdict(set) # rover -> {waypoints}
        rover_rock_samples = defaultdict(set) # rover -> {waypoints}
        rover_images = defaultdict(set) # rover -> {(objective, mode)}
        camera_calibrated_status = set() # {(camera, rover)}
        # Track current sample locations - start with initial and remove as they are picked up
        current_soil_samples_at_waypoint = set(self.initial_soil_samples)
        current_rock_samples_at_waypoint = set(self.initial_rock_samples)

        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)

        for fact in state:
            try:
                parts = get_parts(fact)
                if not parts: continue

                pred = parts[0]
                args = parts[1:]

                if pred == 'at' and len(args) == 2:
                    rover_locations[args[0]] = args[1]
                elif pred == 'empty' and len(args) == 1:
                    store_status[args[0]] = 'empty'
                elif pred == 'full' and len(args) == 1:
                    store_status[args[0]] = 'full'
                elif pred == 'have_soil_analysis' and len(args) == 2:
                    rover_soil_samples[args[0]].add(args[1])
                    # If a rover has the sample, it's no longer at the waypoint
                    current_soil_samples_at_waypoint.discard(args[1])
                elif pred == 'have_rock_analysis' and len(args) == 2:
                    rover_rock_samples[args[0]].add(args[1])
                     # If a rover has the sample, it's no longer at the waypoint
                    current_rock_samples_at_waypoint.discard(args[1])
                elif pred == 'have_image' and len(args) == 3:
                    rover_images[args[0]].add((args[1], args[2]))
                elif pred == 'calibrated' and len(args) == 2:
                    camera_calibrated_status.add((args[0], args[1]))
                elif pred == 'communicated_soil_data' and len(args) == 1:
                    communicated_soil.add(args[0])
                elif pred == 'communicated_rock_data' and len(args) == 1:
                    communicated_rock.add(args[0])
                elif pred == 'communicated_image_data' and len(args) == 2:
                    communicated_image.add((args[0], args[1]))
                # at_soil_sample and at_rock_sample facts in state are handled by
                # initializing from initial_soil/rock_samples and removing when have_analysis is seen.

            except ValueError:
                 # Ignore malformed facts in state
                 pass


        total_cost = 0

        # Iterate through each required goal
        for goal in self.required_goals:
            try:
                parts = get_parts(goal)
                if not parts: continue

                pred = parts[0]
                args = parts[1:]

                goal_achieved = False
                goal_cost = float('inf') # Use infinity initially

                # --- Soil Data Goal ---
                if pred == 'communicated_soil_data' and len(args) == 1:
                    waypoint = args[0]
                    if waypoint in communicated_soil:
                        goal_achieved = True
                    else:
                        # Option 1: Communicate existing sample
                        min_comm_cost_existing = float('inf')
                        for rover in self.all_rovers:
                            if waypoint in rover_soil_samples.get(rover, set()):
                                rover_loc = rover_locations.get(rover)
                                if rover_loc:
                                    min_nav_to_comm = float('inf')
                                    for comm_wp in self.comm_waypoints:
                                        nav_cost = self.rover_nav_costs.get(rover, {}).get(rover_loc, {}).get(comm_wp, float('inf'))
                                        min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                                    if min_nav_to_comm != float('inf'):
                                         min_comm_cost_existing = min(min_comm_cost_existing, 1 + min_nav_to_comm) # 1 for communicate

                        goal_cost = min(goal_cost, min_comm_cost_existing)

                        # Option 2: Sample and communicate
                        min_sample_comm_cost = float('inf')
                        if waypoint in current_soil_samples_at_waypoint:
                             for rover in self.all_rovers:
                                 if 'equipped_for_soil_analysis' in self.rover_capabilities.get(rover, set()):
                                     rover_loc = rover_locations.get(rover)
                                     if rover_loc:
                                         nav_to_sample = self.rover_nav_costs.get(rover, {}).get(rover_loc, {}).get(waypoint, float('inf'))
                                         if nav_to_sample != float('inf'):
                                             store = self.rover_stores.get(rover)
                                             drop_cost = 1 if store and store_status.get(store) == 'full' else 0
                                             sample_cost = 1

                                             min_nav_to_comm = float('inf')
                                             for comm_wp in self.comm_waypoints:
                                                 nav_cost = self.rover_nav_costs.get(rover, {}).get(waypoint, {}).get(comm_wp, float('inf'))
                                                 min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                                             if min_nav_to_comm != float('inf'):
                                                 min_sample_comm_cost = min(min_sample_comm_cost, nav_to_sample + drop_cost + sample_cost + min_nav_to_comm + 1) # 1 for communicate

                        goal_cost = min(goal_cost, min_sample_comm_cost)


                # --- Rock Data Goal ---
                elif pred == 'communicated_rock_data' and len(args) == 1:
                    waypoint = args[0]
                    if waypoint in communicated_rock:
                        goal_achieved = True
                    else:
                        # Option 1: Communicate existing sample
                        min_comm_cost_existing = float('inf')
                        for rover in self.all_rovers:
                            if waypoint in rover_rock_samples.get(rover, set()):
                                rover_loc = rover_locations.get(rover)
                                if rover_loc:
                                    min_nav_to_comm = float('inf')
                                    for comm_wp in self.comm_waypoints:
                                        nav_cost = self.rover_nav_costs.get(rover, {}).get(rover_loc, {}).get(comm_wp, float('inf'))
                                        min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                                    if min_nav_to_comm != float('inf'):
                                         min_comm_cost_existing = min(min_comm_cost_existing, 1 + min_nav_to_comm) # 1 for communicate

                        goal_cost = min(goal_cost, min_comm_cost_existing)

                        # Option 2: Sample and communicate
                        min_sample_comm_cost = float('inf')
                        if waypoint in current_rock_samples_at_waypoint:
                             for rover in self.all_rovers:
                                 if 'equipped_for_rock_analysis' in self.rover_capabilities.get(rover, set()):
                                     rover_loc = rover_locations.get(rover)
                                     if rover_loc:
                                         nav_to_sample = self.rover_nav_costs.get(rover, {}).get(rover_loc, {}).get(waypoint, float('inf'))
                                         if nav_to_sample != float('inf'):
                                             store = self.rover_stores.get(rover)
                                             drop_cost = 1 if store and store_status.get(store) == 'full' else 0
                                             sample_cost = 1

                                             min_nav_to_comm = float('inf')
                                             for comm_wp in self.comm_waypoints:
                                                 nav_cost = self.rover_nav_costs.get(rover, {}).get(waypoint, {}).get(comm_wp, float('inf'))
                                                 min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                                             if min_nav_to_comm != float('inf'):
                                                 min_sample_comm_cost = min(min_sample_comm_cost, nav_to_sample + drop_cost + sample_cost + min_nav_to_comm + 1) # 1 for communicate

                        goal_cost = min(goal_cost, min_sample_comm_cost)


                # --- Image Data Goal ---
                elif pred == 'communicated_image_data' and len(args) == 2:
                    objective, mode = args
                    if (objective, mode) in communicated_image:
                        goal_achieved = True
                    else:
                        # Option 1: Communicate existing image
                        min_comm_cost_existing = float('inf')
                        for rover in self.all_rovers:
                            if (objective, mode) in rover_images.get(rover, set()):
                                rover_loc = rover_locations.get(rover)
                                if rover_loc:
                                    min_nav_to_comm = float('inf')
                                    for comm_wp in self.comm_waypoints:
                                        nav_cost = self.rover_nav_costs.get(rover, {}).get(rover_loc, {}).get(comm_wp, float('inf'))
                                        min_nav_to_comm = min(min_nav_to_comm, nav_cost)
                                    if min_nav_to_comm != float('inf'):
                                         min_comm_cost_existing = min(min_comm_cost_existing, 1 + min_nav_to_comm) # 1 for communicate

                        goal_cost = min(goal_cost, min_comm_cost_existing)

                        # Option 2: Take image and communicate
                        min_image_comm_cost = float('inf')
                        for camera, (rover, modes_supported, cal_target) in self.camera_info.items():
                            if mode in modes_supported and 'equipped_for_imaging' in self.rover_capabilities.get(rover, set()):
                                # Found a suitable rover/camera/mode combination
                                rover_loc = rover_locations.get(rover)
                                if rover_loc:
                                    # Find calibration waypoints
                                    cal_wps = self.calibration_target_visibility.get(cal_target, set()) if cal_target else set()
                                    # Find image waypoints
                                    img_wps = self.objective_visibility.get(objective, set())

                                    if cal_wps and img_wps:
                                        # Find best path: current -> cal_wp -> img_wp -> comm_wp
                                        min_nav_curr_to_cal = float('inf')
                                        best_cal_wp = None
                                        for cal_wp in cal_wps:
                                            nav_cost = self.rover_nav_costs.get(rover, {}).get(rover_loc, {}).get(cal_wp, float('inf'))
                                            if nav_cost < min_nav_curr_to_cal:
                                                min_nav_curr_to_cal = nav_cost
                                                best_cal_wp = cal_wp

                                        # Need to handle case where no cal_wp is reachable
                                        if best_cal_wp is not None and min_nav_curr_to_cal != float('inf'):
                                            min_nav_cal_to_img = float('inf')
                                            best_img_wp = None
                                            for img_wp in img_wps:
                                                nav_cost = self.rover_nav_costs.get(rover, {}).get(best_cal_wp, {}).get(img_wp, float('inf'))
                                                if nav_cost < min_nav_cal_to_img:
                                                    min_nav_cal_to_img = nav_cost
                                                    best_img_wp = img_wp

                                            # Need to handle case where no img_wp is reachable from best_cal_wp
                                            if best_img_wp is not None and min_nav_cal_to_img != float('inf'):
                                                min_nav_img_to_comm = float('inf')
                                                for comm_wp in self.comm_waypoints:
                                                    nav_cost = self.rover_nav_costs.get(rover, {}).get(best_img_wp, {}).get(comm_wp, float('inf'))
                                                    min_nav_img_to_comm = min(min_nav_img_to_comm, nav_cost)

                                                # Need to handle case where no comm_wp is reachable from best_img_wp
                                                if min_nav_img_to_comm != float('inf'):
                                                    # Cost: nav_to_cal + calibrate + nav_to_img + take_image + nav_to_comm + communicate
                                                    # Assume calibrate is always needed if image not taken yet
                                                    image_path_cost = min_nav_curr_to_cal + 1 + min_nav_cal_to_img + 1 + min_nav_img_to_comm + 1
                                                    min_image_comm_cost = min(min_image_comm_cost, image_path_cost)


                        goal_cost = min(goal_cost, min_image_comm_cost)


                # Add cost for this goal if not achieved and reachable by heuristic logic
                if not goal_achieved:
                    if goal_cost == float('inf'):
                        # Goal seems unreachable by the simple paths considered. Add 0.
                        pass # Add 0 cost
                    else:
                        total_cost += goal_cost

            except ValueError:
                 # Handle potential errors during fact parsing from goal
                 pass # Ignore malformed goals


        return total_cost
