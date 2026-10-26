from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
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

# BFS for shortest path on waypoint graph
def get_shortest_path_distance(start_waypoint, end_waypoint, rover, waypoint_graph):
    if start_waypoint == end_waypoint:
        return 0

    graph = waypoint_graph.get(rover)
    # Check if rover exists in graph and waypoints are valid nodes
    if not graph or start_waypoint not in graph or end_waypoint not in graph:
        return float('inf')

    queue = deque([(start_waypoint, 0)])
    visited = {start_waypoint}

    while queue:
        current_wp, dist = queue.popleft()

        if current_wp == end_waypoint:
            return dist

        # Ensure current_wp has neighbors in the graph for this rover
        if current_wp in graph:
            for neighbor in graph[current_wp]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # No path found

# Find nearest waypoint from a set reachable by a rover
def find_nearest_waypoint(start_waypoint, target_waypoints, rover, waypoint_graph):
    min_dist = float('inf')
    nearest_wp = None
    if not target_waypoints:
        return None, float('inf') # No target waypoints to reach

    for target_wp in target_waypoints:
        dist = get_shortest_path_distance(start_waypoint, target_wp, rover, waypoint_graph)
        if dist < min_dist:
            min_dist = dist
            nearest_wp = target_wp
    return nearest_wp, min_dist


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve
    the goal conditions by summing the estimated costs for each unachieved
    goal fact. The cost for each goal fact is estimated based on the
    current state of intermediate requirements (e.g., having the sample/image,
    camera calibration) and includes estimated navigation costs using
    shortest paths on the rover's traversal graph.

    # Assumptions
    - Each unachieved goal fact is considered independently.
    - The cost to achieve an intermediate requirement (like sampling or taking
      an image) is a fixed value (1 action) plus necessary navigation.
    - The cost to communicate data is a fixed value (1 action) plus necessary
      navigation to a communication point.
    - Navigation cost is the shortest path distance (number of navigate actions)
      between waypoints traversable by the specific rover.
    - The heuristic assumes suitable rovers, cameras, and stores exist to
      perform the required tasks. It finds the minimum cost over available
      resources in the current state.
    - Dropping a sample/image data before communication is assumed to require
      re-sampling/re-imaging (if the source is still available). The heuristic
      simplifies this by checking for `have_soil/rock/image` facts. If missing,
      it checks if the sample is still available at the original location.
    - If a sample location is exhausted (`at_soil_sample` or `at_rock_sample` is false
      in the current state), and the corresponding `have_soil/rock_analysis` fact
      is also false, the heuristic assumes the sample was lost and cannot be
      re-sampled from that location. This goal component is then considered impossible.
    - Calibration is required before each image taken with that camera.

    # Heuristic Initialization
    - Extracts static information: rover types, waypoints, cameras, objectives, modes, landers, stores,
      rover capabilities, camera details, objective visibility, lander location, waypoint graph for each rover.
    - Identifies goal requirements (which soil/rock data and images are needed).
    - Pre-computes communication points (waypoints visible from the lander).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `h = 0`.
    2. Identify current state facts: rover locations, `have_soil/rock/image` facts,
       `calibrated` facts, `empty`/`full` store facts, `at_soil_sample`/`at_rock_sample` facts.
    3. For each unachieved goal fact `g` in `task.goals`:
        a. If `g` is `(communicated_soil_data ?w)` and not in state:
           - Find if `(have_soil_analysis r w)` exists for any rover `r` in state.
           - If yes (sample collected): Cost is min over suitable rovers `r` of (dist from `r`'s location to nearest comm point + 1 for communicate). Add min cost to `h`. If no suitable rover or path, add infinity.
           - If no (sample not collected):
             - Check if `(at_soil_sample w)` is true in the current state. If not, this goal is impossible from this location (add infinity).
             - If available: Cost is min over suitable rovers `r` (equipped for soil) of (cost to empty store if full + dist from `r`'s location to `w` + 1 for sample + dist from `w` to nearest comm point + 1 for communicate). Add min cost to `h`. If no suitable rover or path, add infinity.
        b. If `g` is `(communicated_rock_data ?w)` and not in state:
           - Similar logic as soil data, using rock-specific predicates and capabilities.
        c. If `g` is `(communicated_image_data ?o ?m)` and not in state:
           - Find if `(have_image r o m)` exists for any rover `r` in state.
           - If yes (image taken): Cost is min over suitable rovers `r` of (dist from `r`'s location to nearest comm point + 1 for communicate). Add min cost to `h`. If no suitable rover or path, add infinity.
           - If no (image not taken):
             - Cost is min over suitable rovers `r` (equipped for imaging) with camera `i` supporting `m`:
               - Find nearest waypoint `p` visible from `o`. If none, this task is impossible (add infinity).
               - Find calibration target `t` for `i`. Find nearest waypoint `w` visible from `t`. If no target or no waypoint, calibration is impossible (add infinity if calibration is needed).
               - If `(calibrated i r)` in state: Cost is (dist from `r`'s location to `p` + 1 for take image + dist from `p` to nearest comm point + 1 for communicate).
               - If not `(calibrated i r)` in state: Cost is (dist from `r`'s location to `w` + 1 for calibrate + dist from `w` to `p` + 1 for take image + dist from `p` to nearest comm point + 1 for communicate).
               - Add min cost over suitable `r`, `i`, `w`, `p` to `h`. If no suitable combination or path, add infinity.
    4. Return `h`. Return 0 if the goal is reached.

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---
        self.rovers = set()
        self.waypoint_names = set() # Use a different name to avoid conflict with graph nodes
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.landers = set()
        self.stores = set()

        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {} # rover -> set of cameras on board
        self.camera_modes = {} # camera -> set of supported modes
        self.camera_calibration_targets = {} # camera -> calibration_target_objective
        self.objective_visible_from = {} # objective -> set of waypoints visible from

        self.lander_location = None
        self.communication_points = set() # waypoints visible from lander location

        self.waypoint_graph = {} # rover -> {wp: [neighbors]}
        visible_map = {} # wp -> set of visible wps

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            pred = parts[0]
            if pred == 'rover': self.rovers.add(parts[1])
            elif pred == 'waypoint': self.waypoint_names.add(parts[1])
            elif pred == 'camera': self.cameras.add(parts[1])
            elif pred == 'objective': self.objectives.add(parts[1])
            elif pred == 'mode': self.modes.add(parts[1])
            elif pred == 'lander': self.landers.add(parts[1])
            elif pred == 'store': self.stores.add(parts[1])
            elif pred == 'at_lander': self.lander_location = parts[2]
            elif pred.startswith('equipped_for_'):
                rover = parts[1]
                capability = pred.split('_')[-2] # soil, rock, imaging
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = set()
                self.rover_capabilities[rover].add(capability)
            elif pred == 'store_of': self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                if rover not in self.rover_cameras: self.rover_cameras[rover] = set()
                self.rover_cameras[rover].add(camera)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_modes: self.camera_modes[camera] = set()
                self.camera_modes[camera].add(mode)
            elif pred == 'calibration_target': self.camera_calibration_targets[parts[1]] = parts[2] # camera -> objective
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                if objective not in self.objective_visible_from: self.objective_visible_from[objective] = set()
                self.objective_visible_from[objective].add(waypoint)
            elif pred == 'visible':
                w1, w2 = parts[1], parts[2]
                if w1 not in visible_map: visible_map[w1] = set()
                visible_map[w1].add(w2)
            # can_traverse is handled when building the graph

        # Build waypoint graph for each rover
        self.waypoint_graph = {rover: {wp: [] for wp in self.waypoint_names} for rover in self.rovers}
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'can_traverse':
                 r, w1, w2 = parts[1], parts[2], parts[3]
                 # Navigate action requires both can_traverse and visible
                 if w1 in visible_map and w2 in visible_map[w1]:
                      self.waypoint_graph[r][w1].append(w2)

        # Pre-compute communication points (waypoints visible from lander)
        self.communication_points = set()
        if self.lander_location and self.lander_location in visible_map:
             self.communication_points = visible_map[self.lander_location]

        # Extract goal requirements
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_images = set() # set of (objective, mode) tuples

        for goal in self.goals:
            if match(goal, 'communicated_soil_data', '*'):
                self.goal_soil_waypoints.add(get_parts(goal)[1])
            elif match(goal, 'communicated_rock_data', '*'):
                self.goal_rock_waypoints.add(get_parts(goal)[1])
            elif match(goal, 'communicated_image_data', '*', '*'):
                self.goal_images.add((get_parts(goal)[1], get_parts(goal)[2]))

        # Define action costs
        self.NAV_COST = 1
        self.SAMPLE_COST = 1
        self.DROP_COST = 1
        self.CALIBRATE_COST = 1
        self.TAKE_IMAGE_COST = 1
        self.COMMUNICATE_COST = 1


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Goal check
        if self.goals <= state:
             return 0

        h = 0

        # Track current state facts for quick lookup
        at_rover_map = {} # rover -> waypoint
        have_soil_map = {} # waypoint -> set of rovers
        have_rock_map = {} # waypoint -> set of rovers
        have_image_map = {} # (objective, mode) -> set of rovers
        calibrated_map = {} # (camera, rover) -> True
        store_status_map = {} # rover -> 'empty' or 'full'
        at_soil_sample_map = {} # waypoint -> True
        at_rock_sample_map = {} # waypoint -> True

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at' and parts[1] in self.rovers and parts[2] in self.waypoint_names: # Ensure it's rover at waypoint
                 at_rover_map[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                 wp, rover = parts[2], parts[1]
                 if wp not in have_soil_map: have_soil_map[wp] = set()
                 have_soil_map[wp].add(rover)
            elif pred == 'have_rock_analysis':
                 wp, rover = parts[2], parts[1]
                 if wp not in have_rock_map: have_rock_map[wp] = set()
                 have_rock_map[wp].add(rover)
            elif pred == 'have_image':
                 obj, mode, rover = parts[2], parts[3], parts[1]
                 key = (obj, mode)
                 if key not in have_image_map: have_image_map[key] = set()
                 have_image_map[key].add(rover)
            elif pred == 'calibrated':
                 calibrated_map[(parts[1], parts[2])] = True
            elif pred == 'empty':
                 # Find rover for this store
                 store = parts[1]
                 for r, s in self.rover_stores.items():
                     if s == store:
                         store_status_map[r] = 'empty'
                         break
            elif pred == 'full':
                 # Find rover for this store
                 store = parts[1]
                 for r, s in self.rover_stores.items():
                     if s == store:
                         store_status_map[r] = 'full'
                         break
            elif pred == 'at_soil_sample':
                 at_soil_sample_map[parts[1]] = True
            elif pred == 'at_rock_sample':
                 at_rock_sample_map[parts[1]] = True


        # --- Estimate cost for each unachieved goal ---

        # Soil Goals
        for w in self.goal_soil_waypoints:
            goal_fact = f'(communicated_soil_data {w})'
            if goal_fact not in state:
                # Check if sample is already collected by any rover
                sample_collected = any(w in have_soil_map and r in have_soil_map[w] for r in self.rovers)

                min_goal_cost = float('inf')

                if sample_collected:
                    # Need to communicate the collected sample
                    # Find rovers that currently hold the sample
                    suitable_rovers = [r for r in self.rovers if w in have_soil_map and r in have_soil_map[w]]
                    for r in suitable_rovers:
                        if r in at_rover_map:
                            current_loc = at_rover_map[r]
                            nearest_comm_wp, nav_cost_comm = find_nearest_waypoint(
                                current_loc, self.communication_points, r, self.waypoint_graph
                            )
                            if nearest_comm_wp is not None:
                                cost = nav_cost_comm * self.NAV_COST + self.COMMUNICATE_COST
                                min_goal_cost = min(min_goal_cost, cost)
                else:
                    # Need to sample and communicate
                    # Check if the sample is available at the waypoint in the current state
                    sample_available_now = w in at_soil_sample_map

                    if not sample_available_now:
                         # Sample was already taken and is not held by any rover (must have been dropped).
                         # Cannot re-sample from this location according to PDDL. Goal is impossible from here.
                         min_goal_cost = float('inf')
                    else:
                         # Find rovers equipped for soil analysis
                         suitable_rovers = [
                             r for r in self.rovers
                             if r in self.rover_capabilities and 'soil' in self.rover_capabilities[r]
                         ]
                         for r in suitable_rovers:
                             if r in at_rover_map:
                                 current_loc = at_rover_map[r]
                                 store_is_full = store_status_map.get(r) == 'full'

                                 # Cost to sample
                                 nav_cost_to_sample = get_shortest_path_distance(current_loc, w, r, self.waypoint_graph)
                                 if nav_cost_to_sample == float('inf'): continue # Cannot reach sample location

                                 cost_to_sample = (self.DROP_COST if store_is_full else 0) + \
                                                  nav_cost_to_sample * self.NAV_COST + \
                                                  self.SAMPLE_COST

                                 # Cost to communicate after sampling (assumed to be at waypoint w after sampling)
                                 nearest_comm_wp, nav_cost_comm = find_nearest_waypoint(
                                     w, self.communication_points, r, self.waypoint_graph
                                 )
                                 if nearest_comm_wp is None: continue # Cannot reach comm point from sample location

                                 cost_to_communicate = nav_cost_comm * self.NAV_COST + self.COMMUNICATE_COST

                                 total_task_cost = cost_to_sample + cost_to_communicate
                                 min_goal_cost = min(min_goal_cost, total_task_cost)

                # Add cost to heuristic
                h += min_goal_cost


        # Rock Goals (Similar to Soil Goals)
        for w in self.goal_rock_waypoints:
            goal_fact = f'(communicated_rock_data {w})'
            if goal_fact not in state:
                # Check if sample is already collected
                sample_collected = any(w in have_rock_map and r in have_rock_map[w] for r in self.rovers)

                min_goal_cost = float('inf')

                if sample_collected:
                    # Need to communicate the collected sample
                    suitable_rovers = [r for r in self.rovers if w in have_rock_map and r in have_rock_map[w]]
                    for r in suitable_rovers:
                        if r in at_rover_map:
                            current_loc = at_rover_map[r]
                            nearest_comm_wp, nav_cost_comm = find_nearest_waypoint(
                                current_loc, self.communication_points, r, self.waypoint_graph
                            )
                            if nearest_comm_wp is not None:
                                cost = nav_cost_comm * self.NAV_COST + self.COMMUNICATE_COST
                                min_goal_cost = min(min_goal_cost, cost)
                else:
                    # Need to sample and communicate
                    # Check if the sample is available at the waypoint in the current state
                    sample_available_now = w in at_rock_sample_map

                    if not sample_available_now:
                         # Sample was already taken and is not held by any rover (must have been dropped).
                         # Cannot re-sample from this location according to PDDL. Goal is impossible from here.
                         min_goal_cost = float('inf')
                    else:
                         suitable_rovers = [
                             r for r in self.rovers
                             if r in self.rover_capabilities and 'rock' in self.rover_capabilities[r]
                         ]
                         for r in suitable_rovers:
                             if r in at_rover_map:
                                 current_loc = at_rover_map[r]
                                 store_is_full = store_status_map.get(r) == 'full'

                                 # Cost to sample
                                 nav_cost_to_sample = get_shortest_path_distance(current_loc, w, r, self.waypoint_graph)
                                 if nav_cost_to_sample == float('inf'): continue # Cannot reach sample location

                                 cost_to_sample = (self.DROP_COST if store_is_full else 0) + \
                                                  nav_cost_to_sample * self.NAV_COST + \
                                                  self.SAMPLE_COST

                                 # Cost to communicate after sampling (assumed to be at waypoint w after sampling)
                                 nearest_comm_wp, nav_cost_comm = find_nearest_waypoint(
                                     w, self.communication_points, r, self.waypoint_graph
                                 )
                                 if nearest_comm_wp is None: continue # Cannot reach comm point from sample location

                                 cost_to_communicate = nav_cost_comm * self.NAV_COST + self.COMMUNICATE_COST

                                 total_task_cost = cost_to_sample + cost_to_communicate
                                 min_goal_cost = min(min_goal_cost, total_task_cost)

                # Add cost to heuristic
                h += min_goal_cost


        # Image Goals
        for o, m in self.goal_images:
            goal_fact = f'(communicated_image_data {o} {m})'
            if goal_fact not in state:
                # Check if image is already taken
                image_taken = any((o, m) in have_image_map and r in have_image_map[(o, m)] for r in self.rovers)

                min_goal_cost = float('inf')

                if image_taken:
                    # Need to communicate the taken image
                    suitable_rovers = [r for r in self.rovers if (o, m) in have_image_map and r in have_image_map[(o, m)]]
                    for r in suitable_rovers:
                         if r in at_rover_map:
                            current_loc = at_rover_map[r]
                            nearest_comm_wp, nav_cost_comm = find_nearest_waypoint(
                                current_loc, self.communication_points, r, self.waypoint_graph
                            )
                            if nearest_comm_wp is not None:
                                cost = nav_cost_comm * self.NAV_COST + self.COMMUNICATE_COST
                                min_goal_cost = min(min_goal_cost, cost)
                else:
                    # Need to take image and communicate
                    suitable_rovers_cameras = [] # list of (rover, camera) tuples
                    for r in self.rovers:
                        if r in self.rover_capabilities and 'imaging' in self.rover_capabilities[r]:
                            if r in self.rover_cameras:
                                for cam in self.rover_cameras[r]:
                                    if cam in self.camera_modes and m in self.camera_modes[cam]:
                                        suitable_rovers_cameras.append((r, cam))

                    for r, i in suitable_rovers_cameras:
                        if r in at_rover_map:
                            current_loc = at_rover_map[r]

                            # Find nearest image waypoint
                            image_wps = self.objective_visible_from.get(o, set())
                            if not image_wps:
                                # Cannot take image of this objective from anywhere
                                continue

                            nearest_image_wp, nav_cost_to_image = find_nearest_waypoint(
                                current_loc, image_wps, r, self.waypoint_graph
                            )
                            if nearest_image_wp is None: continue # Cannot reach any image waypoint

                            # Cost to calibrate if needed
                            calibration_needed = (i, r) not in calibrated_map
                            cost_to_calibrate = 0
                            loc_after_calib = current_loc # Rover is still at current_loc before moving to calib wp
                            if calibration_needed:
                                calib_target = self.camera_calibration_targets.get(i)
                                if not calib_target:
                                    # Camera has no calibration target, cannot calibrate
                                    continue

                                calib_wps = self.objective_visible_from.get(calib_target, set())
                                if not calib_wps:
                                    # Calibration target not visible from anywhere
                                    continue

                                nearest_calib_wp, nav_cost_to_calib = find_nearest_waypoint(
                                    current_loc, calib_wps, r, self.waypoint_graph
                                )
                                if nearest_calib_wp is None: continue # Cannot reach calibration waypoint

                                cost_to_calibrate = nav_cost_to_calib * self.NAV_COST + self.CALIBRATE_COST
                                loc_after_calib = nearest_calib_wp # Rover is at calib wp after calibrating

                                # Update cost to take image, starting from calibration waypoint
                                nav_cost_from_calib_to_image = get_shortest_path_distance(
                                    loc_after_calib, nearest_image_wp, r, self.waypoint_graph
                                )
                                if nav_cost_from_calib_to_image == float('inf'): continue # Cannot reach image wp from calib wp

                                cost_to_take_image_part = nav_cost_from_calib_to_image * self.NAV_COST + self.TAKE_IMAGE_COST
                            else:
                                # Calibration not needed, cost to take image starts from current location
                                cost_to_take_image_part = nav_cost_to_image * self.NAV_COST + self.TAKE_IMAGE_COST


                            # Cost to communicate after taking image (assumed to be at nearest_image_wp after taking image)
                            nearest_comm_wp, nav_cost_comm = find_nearest_waypoint(
                                nearest_image_wp, self.communication_points, r, self.waypoint_graph
                            )
                            if nearest_comm_wp is None: continue # Cannot reach comm point from image location

                            cost_to_communicate = nav_cost_comm * self.NAV_COST + self.COMMUNICATE_COST

                            total_task_cost = cost_to_calibrate + cost_to_take_image_part + cost_to_communicate
                            min_goal_cost = min(min_goal_cost, total_task_cost)

                # Add cost to heuristic
                h += min_goal_cost

        # If any goal component had infinite cost, the total heuristic is infinite
        if h == float('inf'):
             return float('inf')

        return h
