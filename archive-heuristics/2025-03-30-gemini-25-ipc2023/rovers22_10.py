import collections
import heapq
from fnmatch import fnmatch
# Make sure Heuristic base class is available. Assuming it's in the execution environment.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError
        def goal_reached(self, state):
             # Access goals from the task object stored during init
             if not hasattr(self, 'task') or not hasattr(self.task, 'goals'):
                 # Handle case where task might not be properly initialized (defensive)
                 return False
             return self.task.goals <= state


# Helper functions outside the class
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Handles facts like "(pred arg1 arg2)" -> ["pred", "arg1", "arg2"]
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    unmet goals in the current state. It calculates the minimum estimated cost
    for each individual unmet goal (communicating soil data, rock data, or image data)
    and sums these costs. The cost estimation for each goal considers the necessary
    steps like navigation, sampling/calibrating/imaging, and communication,
    using precomputed shortest path distances between waypoints based on visibility.

    # Assumptions
    - Visibility between waypoints is symmetric for pathfinding purposes.
    - The `visible` predicate determines connectivity for navigation pathfinding and communication range check.
    - `can_traverse` predicate is implicitly satisfied if a path exists based on `visible` (simplification).
    - Rovers have the necessary equipment as defined statically.
    - The heuristic estimates the cost for the *cheapest* sequence of actions for each goal *independently*, assuming an optimal rover is chosen.
    - If a goal seems impossible from the current state (e.g., no path, no required equipment, missing static info), a large penalty (1000) is added for that goal.

    # Heuristic Initialization
    - Parses static facts to build knowledge about the map (waypoints, visibility), rover capabilities (equipment, cameras, stores), camera properties (supported modes, calibration targets), objective locations/visibility, and lander position.
    - Computes all-pairs shortest path distances between waypoints using Breadth-First Search (BFS) based on the `visible` predicate graph. Stores infinity if no path exists.
    - Stores goal conditions provided by the task.
    - Identifies the set of waypoints visible from the lander's location for communication checks.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current state is a goal state. If yes, return 0.
    2.  Parse the current state (`node.state`) to determine dynamic information: rover locations, which rovers have which soil/rock analyses, which rovers have which images, the status ('empty' or 'full') of each rover's store, and which cameras are calibrated.
    3.  Initialize `total_cost = 0`.
    4.  Identify the set of goals that are *not* yet achieved in the current state.
    5.  Iterate through each *unmet* goal:
        a.  **`communicated_soil_data(w)` / `communicated_rock_data(w)`:**
            i.   Check if any rover `r` currently holds the required analysis (`have_soil_analysis(r, w)` or `have_rock_analysis(r, w)`).
            ii.  If yes (analysis already obtained): Find the minimum cost among all rovers `r` holding the analysis. The cost for rover `r` is `distance(location(r), nearest_comm_waypoint) + 1 (communicate)`.
            iii. If no (analysis needs to be obtained): Find the minimum cost among all rovers `r` *equipped* for the analysis. The cost for rover `r` is:
                 - `+ 1` (drop action) if `store_of(r)` is full.
                 - `+ distance(location(r), w)` (navigate to sample waypoint `w`).
                 - `+ 1` (sample action).
                 - `+ distance(w, nearest_comm_waypoint)` (navigate from `w` to communication range).
                 - `+ 1` (communicate action).
        b.  **`communicated_image_data(o, m)`:**
            i.   Check if any rover `r` currently holds the required image (`have_image(r, o, m)`).
            ii.  If yes (image already obtained): Find the minimum cost among all rovers `r` holding the image. The cost for rover `r` is `distance(location(r), nearest_comm_waypoint) + 1 (communicate)`.
            iii. If no (image needs to be obtained): Find the minimum cost among all rovers `r` equipped for imaging with a camera `cam` that supports mode `m`. The cost for the pair `(r, cam)` is calculated sequentially:
                 - Start at `current_loc = location(r)`. Initialize `cost_r = 0`.
                 - **Calibration:** If `cam` is not currently calibrated on `r`:
                     - Find the calibration target `t` for `cam`.
                     - Find the set of waypoints `W_cal` where `t` is visible.
                     - Find the waypoint `w_cal` in `W_cal` that is nearest to `current_loc`.
                     - `cost_r += distance(current_loc, w_cal)`.
                     - `cost_r += 1` (calibrate action).
                     - Update `current_loc = w_cal`.
                 - **Navigation to Image Site:** Find the set of waypoints `P_img` where objective `o` is visible.
                     - Find the waypoint `p` in `P_img` that is nearest to `current_loc`.
                     - `cost_r += distance(current_loc, p)`.
                 - **Take Image:** `cost_r += 1` (take_image action).
                 - Update `current_loc = p`.
                 - **Navigation to Comms:** Find the nearest waypoint `x` to `current_loc` that is visible from the lander.
                     - `cost_r += distance(current_loc, x)`.
                 - **Communicate:** `cost_r += 1` (communicate action).
        c.  Determine the minimum cost (`goal_cost`) calculated across all possible rovers/methods for the current goal.
        d.  If `goal_cost` is infinity (meaning no path found, required equipment missing, etc.), set a flag `impossible_goal = True`.
        e.  Add the cost to the total: `total_cost += 1000` if `impossible_goal`, else `total_cost += goal_cost`.
    6.  After iterating through all unmet goals, if `total_cost` is 0 (and the state is not a goal state), return 1 to ensure non-zero heuristic for non-goal states.
    7.  Return `int(round(total_cost))`, capping at a large value if necessary to avoid overflow.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static

        # Initialize data structures
        self.lander_location = None
        self.rover_equipment = collections.defaultdict(set)
        self.rover_cameras = collections.defaultdict(set)
        self.camera_supports = collections.defaultdict(set)
        self.camera_calibration_targets = {}
        self.rover_stores = {}
        self.store_rovers = {}
        self.waypoints = set()
        self.objectives = set()
        self.visibility = collections.defaultdict(set)
        self.objective_visibility = collections.defaultdict(set)
        self.calibration_visibility = collections.defaultdict(set) # Targets are objectives
        self.lander_visibility_waypoints = set()
        self.distances = {}
        self.known_rovers = set()
        self.known_cameras = set()
        self.known_stores = set()
        self.known_modes = set() # Store known modes

        # Parse static facts
        self._parse_static_facts()

        # Compute shortest paths
        if not self.waypoints:
             # This might happen in trivial problems or if parsing failed
             print("Warning: No waypoints found during initialization.")
        self.distances = self._compute_shortest_paths(list(self.waypoints), self.visibility)

        # Find waypoints visible from lander
        if self.lander_location and self.lander_location in self.visibility:
            # Ensure lander location is a valid waypoint before proceeding
            if self.lander_location in self.waypoints:
                 self.lander_visibility_waypoints = {wp for wp in self.visibility[self.lander_location] if wp in self.waypoints}


    def _parse_static_facts(self):
        """Parses static facts to populate internal data structures."""
        # Use PDDL typing info if available, otherwise infer from predicates
        objects_by_type = collections.defaultdict(set)
        has_type_predicates = False # Flag to check if we found unary type predicates

        # Check for unary type predicates first (common in STRIPS)
        for fact in self.static_facts:
            parts = get_parts(fact)
            # Check for structure like "(type object)"
            if len(parts) == 2 and parts[0] in ['rover', 'waypoint', 'store', 'camera', 'mode', 'lander', 'objective']:
                type_name = parts[0]
                obj_name = parts[1]
                objects_by_type[type_name].add(obj_name)
                has_type_predicates = True

        # Populate known sets if type predicates were found
        if has_type_predicates:
            self.known_rovers = objects_by_type.get('rover', set())
            self.waypoints = objects_by_type.get('waypoint', set())
            self.known_stores = objects_by_type.get('store', set())
            self.known_cameras = objects_by_type.get('camera', set())
            self.known_modes = objects_by_type.get('mode', set())
            self.objectives = objects_by_type.get('objective', set())

        # Process relational predicates
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            try:
                # Skip unary type predicates if already processed
                if has_type_predicates and len(parts) == 2 and pred in objects_by_type:
                     continue

                # Infer types and populate structures if type predicates were absent
                if pred == 'at_lander':
                    self.lander_location = args[1]
                    if not has_type_predicates: self.waypoints.add(args[1])
                elif pred == 'equipped_for_soil_analysis':
                    rover = args[0]
                    if not has_type_predicates: self.known_rovers.add(rover)
                    self.rover_equipment[rover].add('soil')
                elif pred == 'equipped_for_rock_analysis':
                    rover = args[0]
                    if not has_type_predicates: self.known_rovers.add(rover)
                    self.rover_equipment[rover].add('rock')
                elif pred == 'equipped_for_imaging':
                    rover = args[0]
                    if not has_type_predicates: self.known_rovers.add(rover)
                    self.rover_equipment[rover].add('imaging')
                elif pred == 'on_board':
                    cam, rov = args
                    if not has_type_predicates: self.known_cameras.add(cam); self.known_rovers.add(rov)
                    self.rover_cameras[rov].add(cam)
                elif pred == 'supports':
                    cam, mode = args
                    if not has_type_predicates: self.known_cameras.add(cam); self.known_modes.add(mode)
                    self.camera_supports[cam].add(mode)
                elif pred == 'calibration_target':
                    cam, obj = args
                    if not has_type_predicates: self.known_cameras.add(cam); self.objectives.add(obj)
                    self.camera_calibration_targets[cam] = obj
                elif pred == 'store_of':
                    store, rov = args
                    if not has_type_predicates: self.known_stores.add(store); self.known_rovers.add(rov)
                    self.rover_stores[rov] = store
                    self.store_rovers[store] = rov
                elif pred == 'visible':
                    wp1, wp2 = args
                    if not has_type_predicates: self.waypoints.add(wp1); self.waypoints.add(wp2)
                    self.visibility[wp1].add(wp2)
                    self.visibility[wp2].add(wp1) # Assume symmetry
                elif pred == 'visible_from':
                    obj, wp = args
                    if not has_type_predicates: self.objectives.add(obj); self.waypoints.add(wp)
                    self.objective_visibility[obj].add(wp)
                    # Calibration targets are objectives, so visibility applies
                    self.calibration_visibility[obj].add(wp)

            except IndexError:
                # Silently skip malformed facts
                continue


    def _compute_shortest_paths(self, nodes_list, graph):
        """Computes all-pairs shortest paths using BFS."""
        nodes = set(nodes_list)
        distances = {n: {m: float('inf') for m in nodes} for n in nodes}
        if not nodes: return distances

        for start_node in nodes:
            # Ensure start_node is in the distance matrix keys
            if start_node not in distances: continue

            distances[start_node][start_node] = 0
            queue = collections.deque([start_node])
            visited_in_bfs = {start_node}

            while queue:
                current_node = queue.popleft()
                # Check if current_node is actually in the graph structure (defensive)
                if current_node not in distances[start_node]: continue

                current_dist = distances[start_node][current_node]

                # Iterate through neighbors from the visibility graph
                for neighbor in graph.get(current_node, set()):
                    # Check if neighbor is a valid node we care about (in 'nodes')
                    if neighbor in nodes and neighbor not in visited_in_bfs:
                        visited_in_bfs.add(neighbor)
                        # Update distance only if a shorter path is found (standard BFS finds shortest in unweighted)
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def _get_min_dist(self, start_wp, end_wps_set):
        """Finds the minimum distance from start_wp to any waypoint in end_wps_set."""
        min_d = float('inf')
        # Check if start_wp is valid and has distance entries
        if start_wp not in self.distances:
            return float('inf')

        dist_from_start = self.distances[start_wp]
        valid_end_wps = {wp for wp in end_wps_set if wp in dist_from_start} # Filter for valid targets

        if not valid_end_wps:
             return float('inf')

        for end_wp in valid_end_wps:
            dist = dist_from_start.get(end_wp, float('inf')) # Use get for safety, though filtered
            min_d = min(min_d, dist)
        return min_d

    def _get_min_dist_and_wp(self, start_wp, end_wps_set):
        """Finds the minimum distance and the corresponding waypoint."""
        min_d = float('inf')
        best_wp = None
        # Check if start_wp is valid and has distance entries
        if start_wp not in self.distances:
            return float('inf'), None

        dist_from_start = self.distances[start_wp]
        valid_end_wps = {wp for wp in end_wps_set if wp in dist_from_start} # Filter for valid targets

        if not valid_end_wps:
             return float('inf'), None

        for end_wp in valid_end_wps:
            dist = dist_from_start.get(end_wp, float('inf'))
            if dist < min_d:
                min_d = dist
                best_wp = end_wp
        # If no path found to any end_wp, best_wp remains None, min_d remains inf
        return min_d, best_wp


    def __call__(self, node):
        state = node.state

        if self.goal_reached(state):
            return 0

        # Parse current state efficiently
        rover_locations = {}
        rover_has_soil = collections.defaultdict(set)
        rover_has_rock = collections.defaultdict(set)
        rover_has_image = collections.defaultdict(set)
        store_status = {}
        camera_calibrated = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            # Quick checks based on predicate name length and first letter
            len_pred = len(pred)
            if len_pred == 2 and pred == 'at':
                 # Assuming 'at' is only for rovers based on domain structure
                 # Check if args[0] is a known rover for robustness
                 if len(parts) == 3 and parts[1] in self.known_rovers and parts[2] in self.waypoints:
                     rover_locations[parts[1]] = parts[2]
            elif len_pred == 18 and pred == 'have_soil_analysis':
                 if len(parts) == 3 and parts[1] in self.known_rovers and parts[2] in self.waypoints:
                     rover_has_soil[parts[1]].add(parts[2])
            elif len_pred == 18 and pred == 'have_rock_analysis':
                 if len(parts) == 3 and parts[1] in self.known_rovers and parts[2] in self.waypoints:
                     rover_has_rock[parts[1]].add(parts[2])
            elif len_pred == 10 and pred == 'have_image':
                 if len(parts) == 4 and parts[1] in self.known_rovers and parts[2] in self.objectives:
                     rover_has_image[parts[1]].add((parts[2], parts[3])) # (obj, mode)
            elif len_pred == 5 and pred == 'empty':
                 if len(parts) == 2 and parts[1] in self.store_rovers:
                     store_status[parts[1]] = 'empty'
            elif len_pred == 4 and pred == 'full':
                 if len(parts) == 2 and parts[1] in self.store_rovers:
                     store_status[parts[1]] = 'full'
            elif len_pred == 10 and pred == 'calibrated':
                 if len(parts) == 3 and parts[2] in self.known_rovers: # Check rover is known
                     # Assume parts[1] is camera name
                     camera_calibrated.add((parts[1], parts[2])) # (camera, rover)

        # Assume stores associated with known rovers are empty if not mentioned
        for rover in self.known_rovers:
            store = self.rover_stores.get(rover)
            if store and store not in store_status:
                store_status[store] = 'empty'


        total_cost = 0
        achieved_goals = state & self.goals

        for goal in self.goals:
            if goal in achieved_goals:
                continue

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            goal_cost = float('inf')
            impossible_goal = False

            try:
                if pred == 'communicated_soil_data':
                    wp = args[0]
                    if wp not in self.waypoints: impossible_goal = True
                    else:
                        rovers_with_analysis = {r for r, wps in rover_has_soil.items() if wp in wps}
                        if rovers_with_analysis:
                            min_comm_cost = float('inf')
                            for r in rovers_with_analysis:
                                if r in rover_locations:
                                    loc_r = rover_locations[r]
                                    # Cost = Nav_Cost + Comm_Action
                                    cost_r = self._get_min_dist(loc_r, self.lander_visibility_waypoints) + 1
                                    min_comm_cost = min(min_comm_cost, cost_r)
                            goal_cost = min_comm_cost
                        else:
                            min_sample_comm_cost = float('inf')
                            possible_rovers = {r for r in self.known_rovers if 'soil' in self.rover_equipment.get(r, set())}
                            if not possible_rovers: impossible_goal = True
                            else:
                                for r in possible_rovers:
                                    if r not in rover_locations: continue
                                    loc_r = rover_locations[r]
                                    store = self.rover_stores.get(r)
                                    if not store: continue # Rover needs a store

                                    cost_r = 0
                                    # Cost = [Drop] + Nav_to_Sample + Sample + Nav_to_Comm + Communicate
                                    if store_status.get(store) == 'full': cost_r += 1
                                    nav1_cost = self.distances.get(loc_r, {}).get(wp, float('inf'))
                                    if nav1_cost == float('inf'): continue
                                    cost_r += nav1_cost + 1 # nav1 + sample
                                    nav2_cost = self._get_min_dist(wp, self.lander_visibility_waypoints)
                                    if nav2_cost == float('inf'): continue
                                    cost_r += nav2_cost + 1 # nav2 + communicate
                                    min_sample_comm_cost = min(min_sample_comm_cost, cost_r)
                                goal_cost = min_sample_comm_cost

                elif pred == 'communicated_rock_data':
                    wp = args[0]
                    if wp not in self.waypoints: impossible_goal = True
                    else:
                        rovers_with_analysis = {r for r, wps in rover_has_rock.items() if wp in wps}
                        if rovers_with_analysis:
                            min_comm_cost = float('inf')
                            for r in rovers_with_analysis:
                                if r in rover_locations:
                                    loc_r = rover_locations[r]
                                    cost_r = self._get_min_dist(loc_r, self.lander_visibility_waypoints) + 1
                                    min_comm_cost = min(min_comm_cost, cost_r)
                            goal_cost = min_comm_cost
                        else:
                            min_sample_comm_cost = float('inf')
                            possible_rovers = {r for r in self.known_rovers if 'rock' in self.rover_equipment.get(r, set())}
                            if not possible_rovers: impossible_goal = True
                            else:
                                for r in possible_rovers:
                                    if r not in rover_locations: continue
                                    loc_r = rover_locations[r]
                                    store = self.rover_stores.get(r)
                                    if not store: continue

                                    cost_r = 0
                                    if store_status.get(store) == 'full': cost_r += 1
                                    nav1_cost = self.distances.get(loc_r, {}).get(wp, float('inf'))
                                    if nav1_cost == float('inf'): continue
                                    cost_r += nav1_cost + 1 # nav1 + sample
                                    nav2_cost = self._get_min_dist(wp, self.lander_visibility_waypoints)
                                    if nav2_cost == float('inf'): continue
                                    cost_r += nav2_cost + 1 # nav2 + communicate
                                    min_sample_comm_cost = min(min_sample_comm_cost, cost_r)
                                goal_cost = min_sample_comm_cost

                elif pred == 'communicated_image_data':
                    o, m = args
                    if o not in self.objectives: impossible_goal = True
                    else:
                        image_tuple = (o, m)
                        rovers_with_image = {r for r, images in rover_has_image.items() if image_tuple in images}
                        if rovers_with_image:
                            min_comm_cost = float('inf')
                            for r in rovers_with_image:
                                if r in rover_locations:
                                    loc_r = rover_locations[r]
                                    cost_r = self._get_min_dist(loc_r, self.lander_visibility_waypoints) + 1
                                    min_comm_cost = min(min_comm_cost, cost_r)
                            goal_cost = min_comm_cost
                        else:
                            min_image_comm_cost = float('inf')
                            possible_rovers_cameras = []
                            for r in self.known_rovers:
                                 if 'imaging' in self.rover_equipment.get(r, set()) and r in self.rover_cameras:
                                    for cam in self.rover_cameras[r]:
                                        # Check if camera supports the mode
                                        if m in self.camera_supports.get(cam, set()):
                                            possible_rovers_cameras.append((r, cam))

                            if not possible_rovers_cameras: impossible_goal = True
                            else:
                                # Waypoints where objective 'o' is visible
                                visible_target_wps = {wp for wp in self.objective_visibility.get(o, set()) if wp in self.waypoints}
                                if not visible_target_wps: impossible_goal = True
                                else:
                                    for r, cam in possible_rovers_cameras:
                                        if r not in rover_locations: continue
                                        loc_r = rover_locations[r]
                                        cost_r = 0
                                        current_loc = loc_r # Track location for sequential actions

                                        # 1. Calibration cost (if needed)
                                        # Cost = [Nav_Cal + Calibrate] + Nav_Img + TakeImg + Nav_Comm + Communicate
                                        if (cam, r) not in camera_calibrated:
                                            calib_target = self.camera_calibration_targets.get(cam)
                                            if not calib_target: continue # Cannot calibrate this camera
                                            # Waypoints where calibration target is visible
                                            visible_calib_wps = {wp for wp in self.calibration_visibility.get(calib_target, set()) if wp in self.waypoints}
                                            if not visible_calib_wps: continue # Cannot see calibration target

                                            nav_to_calib_cost, w_cal = self._get_min_dist_and_wp(current_loc, visible_calib_wps)
                                            if w_cal is None or nav_to_calib_cost == float('inf'): continue # Cannot reach calibration site
                                            cost_r += nav_to_calib_cost + 1 # nav + calibrate
                                            current_loc = w_cal # Update location after calibration

                                        # 2. Navigate to image waypoint p
                                        nav_to_image_cost, p = self._get_min_dist_and_wp(current_loc, visible_target_wps)
                                        if p is None or nav_to_image_cost == float('inf'): continue # Cannot reach imaging site
                                        cost_r += nav_to_image_cost + 1 # nav + take_image
                                        current_loc = p # Update location after taking image

                                        # 3. Navigate to comms waypoint x
                                        nav_to_comm_cost = self._get_min_dist(current_loc, self.lander_visibility_waypoints)
                                        if nav_to_comm_cost == float('inf'): continue # Cannot reach comms site
                                        cost_r += nav_to_comm_cost + 1 # nav + communicate

                                        min_image_comm_cost = min(min_image_comm_cost, cost_r)
                                    goal_cost = min_image_comm_cost

            except Exception: # Catch any unexpected error during goal cost calculation
                # This might indicate missing data or logic error, treat as impossible
                impossible_goal = True

            # Add cost for the goal
            if impossible_goal or goal_cost == float('inf'):
                total_cost += 1000 # Add large penalty
            else:
                total_cost += goal_cost

        # Final adjustments
        if total_cost == 0 and not self.goal_reached(state):
            # If state is not goal but cost is 0, return 1 (minimum cost)
            return 1

        MAX_HEURISTIC_VALUE = 1000000 # Define a reasonable upper bound
        final_cost = min(total_cost, MAX_HEURISTIC_VALUE)

        # Ensure non-negative result
        if final_cost < 0: final_cost = 0

        return int(round(final_cost))
