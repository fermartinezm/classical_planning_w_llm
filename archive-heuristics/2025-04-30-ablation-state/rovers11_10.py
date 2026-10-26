import collections
from fnmatch import fnmatch
# from heuristics.heuristic_base import Heuristic # Uncomment this line in the target environment

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for BFS
def _bfs(start_w, graph):
    """Performs BFS from start_w on the given graph, returning distances."""
    distances = {start_w: 0}
    queue = collections.deque([start_w])
    while queue:
        current_w = queue.popleft()
        current_dist = distances[current_w]
        # Check if current_w is a valid node in the graph representation
        if current_w in graph:
            for neighbor_w in graph[current_w]:
                if neighbor_w not in distances:
                    distances[neighbor_w] = current_dist + 1
                    queue.append(neighbor_w)
    return distances

# Helper function to find shortest path to any node in a target set using a precomputed map
def _shortest_path_to_set(dist_map, target_ws):
    """
    Finds the minimum distance from the source of dist_map to any waypoint in target_ws
    using the provided distance map.
    Returns the distance and the target waypoint reached.
    Returns (float('inf'), None) if no target is reachable.
    """
    if not target_ws or not dist_map:
        return (float('inf'), None)

    min_dist = float('inf')
    reached_waypoint = None

    for target_w in target_ws:
        if target_w in dist_map:
            if dist_map[target_w] < min_dist:
                min_dist = dist_map[target_w]
                reached_waypoint = target_w

    return (min_dist, reached_waypoint)


class roversHeuristic: # Inherit from Heuristic if available
# class roversHeuristic(Heuristic): # Use this line in the actual planner environment
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    unsatisfied goal conditions. It sums the estimated costs for each goal
    independently, taking the minimum cost over available rovers and resources
    for each goal. The cost includes navigation actions (estimated by shortest
    path BFS) and specific task actions (sample, calibrate, take_image,
    communicate, drop).

    # Assumptions
    - Each goal can be achieved by a single rover (potentially using its onboard
      resources like stores and cameras). Coordination between rovers is ignored
      in the cost estimation.
    - Navigation cost between waypoints is the number of `navigate` actions
      (shortest path).
    - Sampling requires an empty store; if the store is full, a `drop` action
      is needed first.
    - Taking an image requires calibration; if the camera is not calibrated,
      a `calibrate` action is needed first. Calibration is assumed to be
      consumed by `take_image`.
    - Communication requires the rover to be at a waypoint visible from the lander.
    - The heuristic assumes the necessary static conditions (e.g., rover capabilities,
      camera support, sample locations, visible_from relationships) are met for
      a goal to be potentially achievable. If a goal is unreachable by any single
      suitable rover sequence, its estimated cost is infinity.

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - Lander location and waypoints visible from the lander.
    - Navigation graph for each rover based on `can_traverse` facts.
    - Rover capabilities (`equipped_for_soil_analysis`, etc.).
    - Store ownership (`store_of`).
    - Initial locations of soil and rock samples.
    - Camera information (calibration target, supported modes, onboard rover).
    - Waypoints from which objectives and calibration targets are visible.
    - Precomputes shortest path distances between all pairs of waypoints for each rover
      using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine:
       - Current location of each rover.
       - Status (empty/full) of each store.
       - Which soil/rock samples are held by which rovers.
       - Which images are held by which rovers for which objectives/modes.
       - Which cameras are calibrated on which rovers.
       - Which soil/rock samples are still at their initial waypoints.
       - Which rock samples are still at their initial waypoints.
    2. Initialize total heuristic cost to 0.
    3. For each goal condition in the task's goals:
       - If the goal is already satisfied in the current state, add 0 to the total cost.
       - If the goal is `(communicated_soil_data ?w)`:
         - Find the minimum cost among all rovers equipped for soil analysis.
         - For a rover `r`:
           - If `r` already holds the soil sample from `?w`: Cost is navigation from `r`'s current location to a communication waypoint + 1 (communicate).
           - If `r` does not hold the sample but `?w` still has a sample: Cost is (1 if store is full) + navigation from `r`'s current location to `?w` + 1 (sample) + navigation from `?w` to a communication waypoint + 1 (communicate).
           - Use precomputed shortest paths for navigation costs.
         - If no suitable rover can achieve this goal, the minimum cost is infinity.
         - Add the minimum cost for this goal to the total heuristic.
       - If the goal is `(communicated_rock_data ?w)`:
         - Similar logic as soil data, using rock-specific facts and capabilities.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Find the minimum cost among all rovers equipped for imaging that have a camera supporting mode `?m`.
         - For a rover `r` and camera `i`:
           - If `r` already holds the image for `?o` in mode `?m`: Cost is navigation from `r`'s current location to a communication waypoint + 1 (communicate).
           - If `r` does not hold the image:
             - Cost includes:
               - (If camera `i` is not calibrated on `r`): Navigation from `r`'s current location to a calibration waypoint for `i` + 1 (calibrate). After calibration, assume rover is at the reached calibration waypoint for the next navigation step.
               - Navigation from the (potentially new) current location to an image waypoint for `?o` + 1 (take_image). After taking the image, assume rover is at the reached image waypoint for the next navigation step.
               - Navigation from the (potentially new) current location to a communication waypoint + 1 (communicate).
             - Use precomputed shortest paths for navigation costs between sets of waypoints.
         - If no suitable rover/camera can achieve this goal, the minimum cost is infinity.
         - Add the minimum cost for this goal to the total heuristic.
    4. If the minimum cost for any unsatisfied goal was infinity, return infinity.
    5. Otherwise, return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Extract Static Information ---
        self.lander_pos = None
        self.waypoints_visible_from_lander = set()
        self.rover_graphs = {} # rover -> {waypoint -> set(neighbors)}
        self.rover_capabilities = collections.defaultdict(set) # rover -> {capability_string}
        self.rover_stores = {} # rover -> store
        self.store_owner = {} # store -> rover
        self.camera_info = collections.defaultdict(dict) # camera -> {info}
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoint}
        self.calibration_target_visible_from = collections.defaultdict(set) # cal_target -> {waypoint}
        self.calibration_targets = set() # Keep track of which objectives are calibration targets
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()


        # Infer types and build basic static structures
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'at_lander' and len(parts) == 3:
                self.landers.add(parts[1])
                self.waypoints.add(parts[2])
                self.lander_pos = parts[2] # Assuming one lander
            elif predicate == 'can_traverse' and len(parts) == 4:
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.waypoints.add(parts[3])
                 rover, w1, w2 = parts[1], parts[2], parts[3]
                 if rover not in self.rover_graphs:
                     self.rover_graphs[rover] = collections.defaultdict(set)
                 self.rover_graphs[rover][w1].add(w2)
            elif predicate == 'equipped_for_soil_analysis' and len(parts) == 2:
                 self.rovers.add(parts[1])
                 self.rover_capabilities[parts[1]].add('soil')
            elif predicate == 'equipped_for_rock_analysis' and len(parts) == 2:
                 self.rovers.add(parts[1])
                 self.rover_capabilities[parts[1]].add('rock')
            elif predicate == 'equipped_for_imaging' and len(parts) == 2:
                 self.rovers.add(parts[1])
                 self.rover_capabilities[parts[1]].add('imaging')
            elif predicate == 'store_of' and len(parts) == 3:
                 self.stores.add(parts[1])
                 self.rovers.add(parts[2])
                 self.rover_stores[parts[2]] = parts[1]
                 self.store_owner[parts[1]] = parts[2]
            elif predicate == 'supports' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.modes.add(parts[2])
                 if 'supported_modes' not in self.camera_info[parts[1]]:
                     self.camera_info[parts[1]]['supported_modes'] = set()
                 self.camera_info[parts[1]]['supported_modes'].add(parts[2])
            elif predicate == 'on_board' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
                 self.camera_info[parts[1]]['on_board_rover'] = parts[2]
            elif predicate == 'calibration_target' and len(parts) == 3:
                 self.cameras.add(parts[1])
                 self.objectives.add(parts[2]) # Calibration target is an objective
                 self.calibration_targets.add(parts[2])
                 self.camera_info[parts[1]]['calibration_target'] = parts[2]
            elif predicate == 'visible_from' and len(parts) == 3:
                 # This can be objective or calibration target
                 self.objectives.add(parts[1]) # Assume anything visible_from is an objective
                 self.waypoints.add(parts[2])
                 self.objective_visible_from[parts[1]].add(parts[2])
            elif predicate == 'visible' and len(parts) == 3:
                 self.waypoints.add(parts[1])
                 self.waypoints.add(parts[2])
                 # Handled later based on lander_pos

        # Collect waypoints visible from lander
        for fact in task.static:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == 'visible' and len(parts) == 3:
                 w1, w2 = parts[1], parts[2]
                 if self.lander_pos and w2 == self.lander_pos:
                     self.waypoints_visible_from_lander.add(w1)

        # Populate initial sample locations and other initial waypoints from initial state
        for fact in task.initial_state:
            if match(fact, "at_soil_sample", "*"):
                self.waypoints.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.waypoints.add(get_parts(fact)[1])
            elif match(fact, "at", "*", "*"):
                 # Add initial rover locations to waypoints
                 obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                 # Need to know if obj is a rover. Rely on self.rovers collected from static.
                 if obj in self.rovers:
                    self.waypoints.add(loc)


        # Ensure all rovers found have graph entries (even if empty)
        for rover in self.rovers:
            if rover not in self.rover_graphs:
                self.rover_graphs[rover] = collections.defaultdict(set)

        # Ensure all cameras found have info entries (even if empty)
        for camera in self.cameras:
             if camera not in self.camera_info:
                 self.camera_info[camera] = {}
             if 'supported_modes' not in self.camera_info[camera]:
                 self.camera_info[camera]['supported_modes'] = set()

        # Ensure all objectives found have visible_from entries (even if empty)
        for obj in self.objectives:
             if obj not in self.objective_visible_from:
                 self.objective_visible_from[obj] = set()
             if obj in self.calibration_targets and obj not in self.calibration_target_visible_from:
                 self.calibration_target_visible_from[obj] = set()


        # Precompute shortest paths for each rover from all relevant waypoints
        self.rover_dist_maps = {}
        # Relevant waypoints for BFS start nodes are all waypoints in the problem
        all_problem_waypoints = list(self.waypoints) # Use list for consistent iteration order if needed, set is fine too
        for rover, graph in self.rover_graphs.items():
            self.rover_dist_maps[rover] = {}
            for start_w in all_problem_waypoints:
                 self.rover_dist_maps[rover][start_w] = _bfs(start_w, graph)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        current_rover_pos = {} # rover -> waypoint
        current_store_status = {} # store -> 'empty' or 'full'
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated = set() # (camera, rover)
        current_soil_samples_at = set() # waypoint
        current_rock_samples_at = set() # waypoint


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3:
                # Could be rover or lander. Assume it's a rover if it's in our rovers set.
                obj, loc = parts[1], parts[2]
                if obj in self.rovers:
                    current_rover_pos[obj] = loc
            elif predicate == 'empty' and len(parts) == 2:
                current_store_status[parts[1]] = 'empty'
            elif predicate == 'full' and len(parts) == 2:
                current_store_status[parts[1]] = 'full'
            elif predicate == 'have_soil_analysis' and len(parts) == 3:
                current_have_soil.add((parts[1], parts[2]))
            elif predicate == 'have_rock_analysis' and len(parts) == 3:
                current_have_rock.add((parts[1], parts[2]))
            elif predicate == 'have_image' and len(parts) == 4:
                current_have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == 'calibrated' and len(parts) == 3:
                current_calibrated.add((parts[1], parts[2]))
            elif match(fact, "at_soil_sample", "*"):
                 current_soil_samples_at.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                 current_rock_samples_at.add(get_parts(fact)[1])


        total_heuristic = 0

        # Check if lander position is known (should be from static facts)
        if self.lander_pos is None:
             # This shouldn't happen based on domain, but as a safeguard
             # If communication goals exist and lander pos is unknown, they are unreachable.
             for goal in self.goals:
                 if goal not in state and (match(goal, "communicated_soil_data", "*") or match(goal, "communicated_rock_data", "*") or match(goal, "communicated_image_data", "*")):
                     return float('inf')


        # --- Estimate Cost for Each Unsatisfied Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]

            min_goal_cost = float('inf')

            # Goal: Communicate Soil Data
            if predicate == 'communicated_soil_data' and len(parts) == 2:
                waypoint_w = parts[1]
                # Find rovers equipped for soil analysis
                suitable_rovers = [r for r in self.rovers if 'soil' in self.rover_capabilities.get(r, set())]

                for r in suitable_rovers:
                    if r not in current_rover_pos: continue # Rover location unknown

                    rover_pos = current_rover_pos[r]
                    cost = 0
                    sample_held = (r, waypoint_w) in current_have_soil
                    source_w_for_comm = rover_pos # Default communication start point

                    if not sample_held:
                        # Need to sample
                        if waypoint_w not in current_soil_samples_at: continue # Cannot sample if no sample there

                        store = self.rover_stores.get(r)
                        if store is None: continue # Rover has no store

                        drop_cost = 1 if current_store_status.get(store) == 'full' else 0

                        # Nav to sample location
                        dist_map_from_rover_pos = self.rover_dist_maps.get(r, {}).get(rover_pos, {})
                        nav_result_sample = _shortest_path_to_set(dist_map_from_rover_pos, {waypoint_w})
                        nav_to_sample = nav_result_sample[0]
                        if nav_to_sample is float('inf'): continue # Cannot reach sample

                        cost += drop_cost + nav_to_sample + 1 # +1 for sample_soil
                        source_w_for_comm = waypoint_w # After sampling, rover is at waypoint_w

                    # Need to communicate
                    dist_map_from_source_w = self.rover_dist_maps.get(r, {}).get(source_w_for_comm, {})
                    nav_result_comm = _shortest_path_to_set(dist_map_from_source_w, self.waypoints_visible_from_lander)
                    nav_to_comm = nav_result_comm[0]
                    if nav_to_comm is float('inf'): continue # Cannot reach comm waypoint

                    cost += nav_to_comm + 1 # +1 for communicate_soil_data
                    min_goal_cost = min(min_goal_cost, cost)

            # Goal: Communicate Rock Data
            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                waypoint_w = parts[1]
                # Find rovers equipped for rock analysis
                suitable_rovers = [r for r in self.rovers if 'rock' in self.rover_capabilities.get(r, set())]

                for r in suitable_rovers:
                    if r not in current_rover_pos: continue # Rover location unknown

                    rover_pos = current_rover_pos[r]
                    cost = 0
                    sample_held = (r, waypoint_w) in current_have_rock
                    source_w_for_comm = rover_pos # Default communication start point

                    if not sample_held:
                        # Need to sample
                        if waypoint_w not in current_rock_samples_at: continue # Cannot sample if no sample there

                        store = self.rover_stores.get(r)
                        if store is None: continue # Rover has no store

                        drop_cost = 1 if current_store_status.get(store) == 'full' else 0

                        # Nav to sample location
                        dist_map_from_rover_pos = self.rover_dist_maps.get(r, {}).get(rover_pos, {})
                        nav_result_sample = _shortest_path_to_set(dist_map_from_rover_pos, {waypoint_w})
                        nav_to_sample = nav_result_sample[0]
                        if nav_to_sample is float('inf'): continue # Cannot reach sample

                        cost += drop_cost + nav_to_sample + 1 # +1 for sample_rock
                        source_w_for_comm = waypoint_w # After sampling, rover is at waypoint_w

                    # Need to communicate
                    dist_map_from_source_w = self.rover_dist_maps.get(r, {}).get(source_w_for_comm, {})
                    nav_result_comm = _shortest_path_to_set(dist_map_from_source_w, self.waypoints_visible_from_lander)
                    nav_to_comm = nav_result_comm[0]
                    if nav_to_comm is float('inf'): continue # Cannot reach comm waypoint

                    cost += nav_to_comm + 1 # +1 for communicate_rock_data
                    min_goal_cost = min(min_goal_cost, cost)

            # Goal: Communicate Image Data
            elif predicate == 'communicated_image_data' and len(parts) == 3:
                objective_o = parts[1]
                mode_m = parts[2]

                # Find rovers equipped for imaging
                suitable_rovers = [r for r in self.rovers if 'imaging' in self.rover_capabilities.get(r, set())]

                for r in suitable_rovers:
                    if r not in current_rover_pos: continue # Rover location unknown

                    # Find cameras on this rover supporting this mode
                    suitable_cameras = [
                        cam for cam in self.cameras
                        if self.camera_info.get(cam, {}).get('on_board_rover') == r
                        and mode_m in self.camera_info.get(cam, {}).get('supported_modes', set())
                    ]

                    for i in suitable_cameras:
                        rover_pos = current_rover_pos[r]
                        image_held = (r, objective_o, mode_m) in current_have_image
                        source_w_for_comm = rover_pos # Default communication start point
                        cost = 0

                        if not image_held:
                            # Need to take image
                            cal_target = self.camera_info.get(i, {}).get('calibration_target')
                            if cal_target is None: continue # Camera has no calibration target

                            cal_waypoints = self.calibration_target_visible_from.get(cal_target, set())
                            img_waypoints = self.objective_visible_from.get(objective_o, set())

                            current_nav_pos = rover_pos

                            # Calibration step (if needed)
                            if (i, r) not in current_calibrated:
                                dist_map_from_current_nav_pos = self.rover_dist_maps.get(r, {}).get(current_nav_pos, {})
                                nav_result_cal = _shortest_path_to_set(dist_map_from_current_nav_pos, cal_waypoints)
                                nav_to_cal = nav_result_cal[0]
                                cal_waypoint_reached = nav_result_cal[1]
                                if nav_to_cal is float('inf'): continue # Cannot calibrate

                                cost += nav_to_cal + 1 # +1 for calibrate
                                current_nav_pos = cal_waypoint_reached # Rover is now at this waypoint

                            # Take Image step
                            dist_map_from_current_nav_pos = self.rover_dist_maps.get(r, {}).get(current_nav_pos, {})
                            nav_result_img = _shortest_path_to_set(dist_map_from_current_nav_pos, img_waypoints)
                            nav_to_img = nav_result_img[0]
                            img_waypoint_reached = nav_result_img[1]
                            if nav_to_img is float('inf'): continue # Cannot reach image waypoint

                            cost += nav_to_img + 1 # +1 for take_image
                            source_w_for_comm = img_waypoint_reached # After taking image, rover is at this waypoint

                        # Need to communicate
                        dist_map_from_source_w = self.rover_dist_maps.get(r, {}).get(source_w_for_comm, {})
                        nav_result_comm = _shortest_path_to_set(dist_map_from_source_w, self.waypoints_visible_from_lander)
                        nav_to_comm = nav_result_comm[0]
                        if nav_to_comm is float('inf'): continue # Cannot reach comm waypoint

                        cost += nav_to_comm + 1 # +1 for communicate_image_data
                        min_goal_cost = min(min_goal_cost, cost)


            # If after checking all options for this goal, it's still infinity,
            # the goal is unreachable from this state by a single rover plan.
            # Return infinity for the whole heuristic.
            if min_goal_cost is float('inf'):
                 return float('inf')

            total_heuristic += min_goal_cost

        return total_heuristic
