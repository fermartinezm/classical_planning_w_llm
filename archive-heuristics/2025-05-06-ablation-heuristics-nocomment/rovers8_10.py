from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs(graph, start_node):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., waypoint not in traversal graph)
         return distances

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Check if current_node exists in graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing the
    estimated costs for each individual goal predicate that is not yet satisfied.
    The cost for each unachieved goal is an estimate of the minimum number of
    actions and navigation steps required to achieve it from the current state.
    Navigation costs are based on precomputed shortest paths between waypoints
    for each rover.

    Assumptions:
    - The problem instance is solvable. This implies that necessary resources
      (equipped rovers, samples at locations, calibration targets, visible
      points, landers, communication points) exist statically to achieve the goals.
    - The heuristic is non-admissible; it does not guarantee finding the optimal
      plan but aims to guide a greedy best-first search efficiently.
    - Navigation is possible between required waypoints for the relevant rovers.
      If a required navigation segment is impossible (no path exists), the
      heuristic component for that goal becomes infinite, indicating unreachability
      via that path.

    Heuristic Initialization:
    1. Parse all static facts from the task definition.
    2. Identify all objects (rovers, waypoints, stores, cameras, modes, landers, objectives)
       and relevant static relations (e.g., `can_traverse`, `visible`, `equipped_for_soil_analysis`,
       `store_of`, `at_soil_sample`, `on_board`, `supports`, `calibration_target`, `visible_from`,
       `at_lander`).
    3. For each rover, build a navigation graph where nodes are waypoints and
       edges exist between waypoints `u` and `v` if the rover `r` can traverse
       from `u` to `v` (`(can_traverse r u v)`) AND `v` is visible from `u` (`(visible u v)`).
    4. For each rover, compute all-pairs shortest paths between all waypoints
       using Breadth-First Search (BFS) on its navigation graph. Store these distances.
    5. Identify all waypoints from which a lander is visible (`comm_points`). These are
       potential communication locations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize `total_cost` to 0.
    2. Extract dynamic information from the current state: current location of each rover,
       status of each store (empty/full), samples held by rovers, images held by rovers,
       and calibrated cameras.
    3. Iterate through each goal predicate defined in the task.
    4. If a goal predicate is already present in the current state, its contribution
       to the heuristic is 0. Continue to the next goal.
    5. If a goal predicate is not in the current state, estimate the cost to achieve it:
        - **For `(communicated_soil_data ?w)`:**
            - Add 1 for the `communicate_soil_data` action.
            - Check if any soil-equipped rover currently holds the sample `(have_soil_analysis ?r ?w)`.
            - If no rover holds the sample:
                - Add 1 for the `sample_soil` action.
                - Find the soil-equipped rover closest to waypoint `?w`.
                - Add the shortest path distance for this rover from its current location to `?w`.
                - Add 1 if this rover's store is currently full (for the `drop` action).
                - Assume the rover is now at `?w`.
                - Find the waypoint closest to `?w` among the `comm_points`.
                - Add the shortest path distance for this rover from `?w` to this closest `comm_point`.
            - If a rover holds the sample:
                - Find the rover holding the sample that is closest to any `comm_point`.
                - Add the shortest path distance for this rover from its current location to this closest `comm_point`.
        - **For `(communicated_rock_data ?w)`:** Follow the same logic as soil data, substituting rock-specific predicates and rovers.
        - **For `(communicated_image_data ?o ?m)`:**
            - Add 1 for the `communicate_image_data` action.
            - Check if any imaging-equipped rover currently holds the image `(have_image ?r ?o ?m)`.
            - If no rover holds the image:
                - Add 1 for the `take_image` action.
                - Find a suitable (rover, camera) pair (rover equipped for imaging, camera on board, camera supports mode `?m`).
                - Check if the chosen camera on the chosen rover is calibrated `(calibrated ?i ?r)`.
                - If not calibrated:
                    - Add 1 for the `calibrate` action.
                    - Find the calibration target for the camera.
                    - Find the waypoint closest to the rover's current location among the waypoints from which the calibration target is visible.
                    - Add the shortest path distance for the rover from its current location to this closest calibration waypoint.
                    - Assume the rover is now at this calibration waypoint.
                    - Find the waypoint closest to the current location (the calibration waypoint) among the waypoints from which objective `?o` is visible.
                    - Add the shortest path distance for the rover from the calibration waypoint to this closest image waypoint.
                - If already calibrated:
                    - Find the waypoint closest to the rover's current location among the waypoints from which objective `?o` is visible.
                    - Add the shortest path distance for the rover from its current location to this closest image waypoint.
                - Assume the rover is now at the image waypoint.
                - Find the waypoint closest to the current location (the image waypoint) among the `comm_points`.
                - Add the shortest path distance for the rover from the image waypoint to this closest `comm_point`.
            - If a rover holds the image:
                - Find the rover holding the image that is closest to any `comm_point`.
                - Add the shortest path distance for this rover from its current location to this closest `comm_point`.
        - If at any point a required resource is missing (e.g., no equipped rover, no path exists), the cost for this goal is considered infinite.
    6. Sum the estimated costs for all unachieved goals to get the `total_cost`.
    7. If `total_cost` is infinite (meaning at least one unachieved goal is unreachable based on static facts and navigation), return a large finite number (e.g., 1000000) instead of infinity. Otherwise, return `total_cost`.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # --- Heuristic Initialization ---
        # Parse static facts to extract objects and relations
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        self.rover_can_traverse = {} # {rover: {(wp_from, wp_to), ...}}
        self.visible_waypoints = set() # {(wp1, wp2), ...}
        self.lander_locations = set() # {wp, ...}
        self.soil_rovers = set() # {rover, ...}
        self.rock_rovers = set() # {rover, ...}
        self.imaging_rovers = set() # {rover, ...}
        self.rover_to_store = {} # {rover: store}
        self.soil_sample_locations = set() # {wp, ...}
        self.rock_sample_locations = set() # {wp, ...}
        self.camera_on_rover = {} # {camera: rover}
        self.camera_supports_mode = set() # {(camera, mode), ...}
        self.camera_calibration_target = {} # {camera: objective}
        self.objective_visible_from = {} # {objective: {wp, ...}}

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'can_traverse':
                r, w_from, w_to = parts[1:]
                self.rovers.add(r)
                self.waypoints.add(w_from)
                self.waypoints.add(w_to)
                self.rover_can_traverse.setdefault(r, set()).add((w_from, w_to))
            elif pred == 'visible':
                w1, w2 = parts[1:]
                self.waypoints.add(w1)
                self.waypoints.add(w2)
                self.visible_waypoints.add((w1, w2))
            elif pred == 'at_lander':
                l, w = parts[1:]
                self.landers.add(l)
                self.waypoints.add(w)
                self.lander_locations.add(w)
            elif pred == 'equipped_for_soil_analysis':
                self.rovers.add(parts[1])
                self.soil_rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                self.rovers.add(parts[1])
                self.rock_rovers.add(parts[1])
            elif pred == 'equipped_for_imaging':
                self.rovers.add(parts[1])
                self.imaging_rovers.add(parts[1])
            elif pred == 'store_of':
                s, r = parts[1:]
                self.stores.add(s)
                self.rovers.add(r)
                self.rover_to_store[r] = s
            elif pred == 'at_soil_sample':
                self.waypoints.add(parts[1])
                self.soil_sample_locations.add(parts[1])
            elif pred == 'at_rock_sample':
                self.waypoints.add(parts[1])
                self.rock_sample_locations.add(parts[1])
            elif pred == 'on_board':
                c, r = parts[1:]
                self.cameras.add(c)
                self.rovers.add(r)
                self.camera_on_rover[c] = r
            elif pred == 'supports':
                c, m = parts[1:]
                self.cameras.add(c)
                self.modes.add(m)
                self.camera_supports_mode.add((c, m))
            elif pred == 'calibration_target':
                i, t = parts[1:]
                self.cameras.add(i)
                self.objectives.add(t)
                self.camera_calibration_target[i] = t
            elif pred == 'visible_from':
                o, w = parts[1:]
                self.objectives.add(o)
                self.waypoints.add(w)
                self.objective_visible_from.setdefault(o, set()).add(w)

        # Precompute all-pairs shortest paths for each rover
        self.dist = {}
        for rover in self.rovers:
            self.dist[rover] = {}
            # Build navigation graph for this rover
            graph = {wp: set() for wp in self.waypoints}
            for w_from, w_to in self.rover_can_traverse.get(rover, set()):
                # Navigation requires both can_traverse and visible
                if (w_from, w_to) in self.visible_waypoints:
                     graph[w_from].add(w_to)

            # Compute distances from every waypoint
            for start_wp in self.waypoints:
                self.dist[rover][start_wp] = bfs(graph, start_wp)

        # Identify waypoints visible from any lander location (communication points)
        self.comm_points = set()
        for lander_loc in self.lander_locations:
             for w1, w2 in self.visible_waypoints:
                 if w2 == lander_loc:
                     self.comm_points.add(w1)
                 if w1 == lander_loc:
                     self.comm_points.add(w2)

        # Helper for min nav cost between two points for a rover
        self.min_nav_cost = lambda r, start, end: self.dist.get(r, {}).get(start, {}).get(end, float('inf'))

        # Helper for min nav cost from a point to any in a set for a rover
        def min_nav_cost_to_any_helper(rover, start_wp, target_wps):
            if start_wp is None or not target_wps:
                return float('inf')
            min_dist = float('inf')
            for target_wp in target_wps:
                dist = self.min_nav_cost(rover, start_wp, target_wp)
                min_dist = min(min_dist, dist)
            return min_dist
        self.min_nav_cost_to_any = min_nav_cost_to_any_helper

        # Helper to find the best rover (closest to target)
        def find_best_rover_helper(rover_set, rover_locations, target):
            best_rover = None
            min_dist = float('inf')

            if isinstance(target, str): # Single target waypoint
                target_wps = {target}
            else: # Set of target waypoints
                target_wps = target

            if not rover_set or not target_wps:
                return None, float('inf')

            for rover in rover_set:
                current_loc = rover_locations.get(rover)
                if current_loc is None:
                     continue

                dist = self.min_nav_cost_to_any(rover, current_loc, target_wps)

                if dist < min_dist:
                    min_dist = dist
                    best_rover = rover

            return best_rover, min_dist
        self.find_best_rover = find_best_rover_helper

        # Helper function to find the closest waypoint in a set
        def find_closest_waypoint_helper(rover, start_wp, target_wps):
            """Finds the waypoint in target_wps closest to start_wp for the given rover."""
            if start_wp is None or not target_wps:
                return None, float('inf')
            best_wp = None
            min_dist = float('inf')
            for target_wp in target_wps:
                dist = self.min_nav_cost(rover, start_wp, target_wp)
                if dist < min_dist:
                    min_dist = dist
                    best_wp = target_wp
            return best_wp, min_dist
        self.find_closest_waypoint = find_closest_waypoint_helper


    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An estimated cost (integer or float('inf')) to reach a goal state.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Step-By-Step Thinking for Computing Heuristic ---
        # The heuristic is the sum of estimated costs for each unachieved goal.
        # For each unachieved goal, we estimate the minimum actions + navigation
        # required to satisfy it, assuming we can use any available rover/resource.
        # Navigation costs are estimated using precomputed shortest paths.

        # Extract dynamic state information
        rover_locations = {}
        store_status = {} # {store: 'empty' or 'full'}
        soil_samples_held = set() # {(rover, waypoint), ...}
        rock_samples_held = set() # {(rover, waypoint), ...}
        images_held = set() # {(rover, objective, mode), ...}
        calibrated_cameras = set() # {(camera, rover), ...}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at' and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                 rover_locations[parts[1]] = parts[2]
            elif pred == 'empty' and len(parts) == 2 and parts[1] in self.stores:
                 store_status[parts[1]] = 'empty'
            elif pred == 'full' and len(parts) == 2 and parts[1] in self.stores:
                 store_status[parts[1]] = 'full'
            elif pred == 'have_soil_analysis' and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                 soil_samples_held.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis' and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                 rock_samples_held.add((parts[1], parts[2]))
            elif pred == 'have_image' and len(parts) == 4 and parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                 images_held.add((parts[1], parts[2], parts[3]))
            elif pred == 'calibrated' and len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.rovers:
                 calibrated_cameras.add((parts[1], parts[2]))

        total_cost = 0

        for goal in self.goals:
            # If goal is already achieved, cost for this goal is 0
            if goal in state:
                continue

            parts = get_parts(goal)
            pred = parts[0]

            # Estimate cost for communicated_soil_data goal
            if pred == "communicated_soil_data":
                waypoint_w = parts[1]
                cost_g = 0 # Cost for this specific goal

                # 1. Need to communicate
                cost_g += 1 # communicate_soil_data action

                # 2. Need to have the soil analysis sample
                # Check if any equipped rover already has the sample
                rovers_with_sample = {r for r in self.soil_rovers if (r, waypoint_w) in soil_samples_held}

                rover_for_comm = None # The rover that will eventually communicate
                current_rover_loc_for_comm = None # Its location when ready to communicate

                if not rovers_with_sample:
                    # Need to sample
                    cost_g += 1 # sample_soil action

                    # Find a suitable rover (equipped for soil analysis)
                    if not self.soil_rovers:
                         total_cost = float('inf')
                         break # Goal unreachable

                    # Pick the best rover to sample (closest to the sample location)
                    r_sample, nav_to_sample_cost = self.find_best_rover(self.soil_rovers, rover_locations, waypoint_w)

                    if r_sample is None or nav_to_sample_cost == float('inf'):
                         total_cost = float('inf')
                         break # Goal unreachable

                    cost_g += nav_to_sample_cost # Navigation to sample location

                    # Check if store is full (need to drop first)
                    store = self.rover_to_store.get(r_sample)
                    if store and store_status.get(store, 'empty') == 'full':
                        cost_g += 1 # drop action

                    # After sampling, the rover is at waypoint_w
                    rover_for_comm = r_sample
                    current_rover_loc_for_comm = waypoint_w

                else:
                    # Sample is already held by one or more rovers
                    # Pick the best rover to communicate (closest to a comm point)
                    rover_for_comm, nav_cost_from_current = self.find_best_rover(rovers_with_sample, rover_locations, self.comm_points)

                    if rover_for_comm is None or nav_cost_from_current == float('inf'):
                         total_cost = float('inf')
                         break # Goal unreachable

                    current_rover_loc_for_comm = rover_locations.get(rover_for_comm)


                # 3. Need to navigate to a communication point
                if not self.comm_points:
                     total_cost = float('inf')
                     break # Goal unreachable

                nav_to_lander_cost = self.min_nav_cost_to_any(rover_for_comm, current_rover_loc_for_comm, self.comm_points)

                if nav_to_lander_cost == float('inf'):
                     total_cost = float('inf')
                     break # Goal unreachable

                cost_g += nav_to_lander_cost # Navigation to communication point

                total_cost += cost_g

            # Estimate cost for communicated_rock_data goal
            elif pred == "communicated_rock_data":
                waypoint_w = parts[1]
                cost_g = 0 # Cost for this specific goal

                # 1. Need to communicate
                cost_g += 1 # communicate_rock_data action

                # 2. Need to have the rock analysis sample
                # Check if any equipped rover already has the sample
                rovers_with_sample = {r for r in self.rock_rovers if (r, waypoint_w) in rock_samples_held}

                rover_for_comm = None
                current_rover_loc_for_comm = None

                if not rovers_with_sample:
                    # Need to sample
                    cost_g += 1 # sample_rock action

                    # Find a suitable rover (equipped for rock analysis)
                    if not self.rock_rovers:
                         total_cost = float('inf')
                         break # Goal unreachable

                    # Pick the best rover to sample (closest to the sample location)
                    r_sample, nav_to_sample_cost = self.find_best_rover(self.rock_rovers, rover_locations, waypoint_w)

                    if r_sample is None or nav_to_sample_cost == float('inf'):
                         total_cost = float('inf')
                         break # Goal unreachable

                    cost_g += nav_to_sample_cost # Navigation to sample location

                    # Check if store is full (need to drop first)
                    store = self.rover_to_store.get(r_sample)
                    if store and store_status.get(store, 'empty') == 'full':
                        cost_g += 1 # drop action

                    # After sampling, the rover is at waypoint_w
                    rover_for_comm = r_sample
                    current_rover_loc_for_comm = waypoint_w

                else:
                    # Sample is already held by one or more rovers
                    # Pick the best rover to communicate (closest to a comm point)
                    rover_for_comm, nav_cost_from_current = self.find_best_rover(rovers_with_sample, rover_locations, self.comm_points)

                    if rover_for_comm is None or nav_cost_from_current == float('inf'):
                         total_cost = float('inf')
                         break # Goal unreachable

                    current_rover_loc_for_comm = rover_locations.get(rover_for_comm)

                # 3. Need to navigate to a communication point
                if not self.comm_points:
                     total_cost = float('inf')
                     break # Goal unreachable

                nav_to_lander_cost = self.min_nav_cost_to_any(rover_for_comm, current_rover_loc_for_comm, self.comm_points)

                if nav_to_lander_cost == float('inf'):
                     total_cost = float('inf')
                     break # Goal unreachable

                cost_g += nav_to_lander_cost # Navigation to communication point

                total_cost += cost_g

            # Estimate cost for communicated_image_data goal
            elif pred == "communicated_image_data":
                objective_o = parts[1]
                mode_m = parts[2]
                cost_g = 0 # Cost for this specific goal

                # 1. Need to communicate
                cost_g += 1 # communicate_image_data action

                # 2. Need to have the image
                # Check if any equipped rover already has the image
                rovers_with_image = {r for r in self.imaging_rovers if (r, objective_o, mode_m) in images_held}

                rover_for_comm = None
                current_rover_loc_for_comm = None

                if not rovers_with_image:
                    # Need to take image
                    cost_g += 1 # take_image action

                    # Find a suitable (rover, camera) pair
                    # Rover must be equipped for imaging, camera must be on board, camera must support mode
                    suitable_rc_pairs = {(r, i) for r in self.imaging_rovers for i in self.camera_on_rover if self.camera_on_rover.get(i) == r and (i, mode_m) in self.camera_supports_mode}

                    if not suitable_rc_pairs:
                         total_cost = float('inf')
                         break # Goal unreachable

                    # Find the best rover/camera pair. For simplicity, pick the rover closest to *any* image waypoint.
                    candidate_rovers = {r for r, i in suitable_rc_pairs}
                    image_wps_for_o = self.objective_visible_from.get(objective_o, set())

                    if not image_wps_for_o:
                         total_cost = float('inf')
                         break # Goal unreachable

                    # Find the best rover to take the image (closest to an image waypoint)
                    r_image, nav_to_image_wp_cost_initial = self.find_best_rover(candidate_rovers, rover_locations, image_wps_for_o)

                    if r_image is None or nav_to_image_wp_cost_initial == float('inf'):
                         total_cost = float('inf')
                         break # Goal unreachable

                    # Find a camera on this rover that supports the mode
                    i_image = next(i for i in self.camera_on_rover if self.camera_on_rover.get(i) == r_image and (i, mode_m) in self.camera_supports_mode)

                    current_loc_before_cal_nav = rover_locations.get(r_image)
                    current_loc_after_cal_nav = current_loc_before_cal_nav # Default if no cal needed or nav impossible
                    current_loc_after_image_nav = current_loc_before_cal_nav # Default if no image nav needed or impossible


                    # Check if calibration is needed
                    if (i_image, r_image) not in calibrated_cameras:
                        cost_g += 1 # calibrate action

                        # Find calibration target and waypoints
                        t = self.camera_calibration_target.get(i_image)
                        if t is None:
                             total_cost = float('inf')
                             break # Goal unreachable (camera has no calibration target)

                        w_cal_options = self.objective_visible_from.get(t, set())
                        if not w_cal_options:
                             total_cost = float('inf')
                             break # Goal unreachable (calibration target not visible from anywhere)

                        # Nav to calibration point
                        best_w_cal, nav_to_cal_cost = self.find_closest_waypoint(r_image, current_loc_before_cal_nav, w_cal_options)
                        if best_w_cal is None or nav_to_cal_cost == float('inf'):
                             total_cost = float('inf')
                             break # Goal unreachable

                        cost_g += nav_to_cal_cost
                        current_loc_after_cal_nav = best_w_cal


                    # Nav from calibration point (or current loc if already calibrated) to image point
                    start_loc_for_image_nav = current_loc_after_cal_nav if (i_image, r_image) not in calibrated_cameras else current_loc_before_cal_nav

                    best_p, nav_to_image_cost = self.find_closest_waypoint(r_image, start_loc_for_image_nav, image_wps_for_o)
                    if best_p is None or nav_to_image_cost == float('inf'):
                         total_cost = float('inf')
                         break # Goal unreachable

                    cost_g += nav_to_image_cost
                    current_loc_after_image_nav = best_p

                    # After taking image, the rover is at the image waypoint
                    rover_for_comm = r_image
                    current_rover_loc_for_comm = current_loc_after_image_nav

                else:
                    # Image is already held by one or more rovers
                    # Pick the best rover to communicate (closest to a comm point)
                    rover_for_comm, nav_cost_from_current = self.find_best_rover(rovers_with_image, rover_locations, self.comm_points)

                    if rover_for_comm is None or nav_cost_from_current == float('inf'):
                         total_cost = float('inf')
                         break # Goal unreachable

                    current_rover_loc_for_comm = rover_locations.get(rover_for_comm)


                # 3. Need to navigate to a communication point
                if not self.comm_points:
                     total_cost = float('inf')
                     break # Goal unreachable

                nav_to_lander_cost = self.min_nav_cost_to_any(rover_for_comm, current_rover_loc_for_comm, self.comm_points)

                if nav_to_lander_cost == float('inf'):
                     total_cost = float('inf')
                     break # Goal unreachable

                cost_g += nav_to_lander_cost # Navigation to communication point

                total_cost += cost_g

            # If total_cost became infinity inside the loop, break
            if total_cost == float('inf'):
                 break


        # Return a large finite number if the goal is unreachable (total_cost is infinity)
        return total_cost if total_cost != float('inf') else 1000000
