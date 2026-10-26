import collections
import sys

# Helper function to parse fact strings
def parse_fact(fact_string):
    """Parses a fact string like '(predicate obj1 obj2)' into ('predicate', 'obj1', 'obj2')."""
    # Remove surrounding parentheses and split by spaces
    parts = fact_string.strip()[1:-1].split()
    return tuple(parts)

# Helper function to extract objects from fact strings
def get_objects(fact_string):
    """Extracts objects from a fact string, ignoring surrounding brackets."""
    # Remove surrounding parentheses and split by spaces, skip the predicate
    parts = fact_string.strip()[1:-1].split()
    return tuple(parts[1:])


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers planning domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing the
    estimated costs for each unachieved goal fact. The cost for each goal
    fact is estimated based on the actions required to achieve it, considering
    rover capabilities, locations, required samples/images, and communication
    requirements. Navigation costs are precomputed shortest path distances
    on the waypoint graph for each rover.

    Assumptions:
    - The heuristic assumes a sequential process for achieving goals:
      (Navigate to sample/image location) -> (Sample/Image) -> (Navigate to communication location) -> (Communicate).
    - Image goals requiring calibration assume calibration happens immediately
      before imaging, requiring navigation to a calibration target waypoint first.
    - The waypoint graph connectivity for navigation is determined by
      both `can_traverse` for the specific rover and `visible` predicates.
    - `visible` predicate is assumed to imply symmetric connectivity for
      communication purposes (i.e., if waypoint A is visible from B, B is visible from A).
    - If a sample (soil/rock) is no longer at its initial location and no rover
      holds the corresponding data, the goal requiring that data is considered
      unreachable by this heuristic.
    - The heuristic does not account for resource constraints beyond the single
      store per rover (e.g., battery, time).
    - The heuristic is not admissible; it is designed for greedy best-first search
      to minimize expanded nodes.

    Heuristic Initialization:
    The heuristic's constructor (`__init__`) performs the following steps:
    1. Parses static facts from the task definition (`task.static`) to identify:
       - All objects (rovers, waypoints, cameras, etc.) by inspecting fact arguments.
       - Lander location.
       - Rover capabilities (equipped for soil, rock, imaging).
       - Store ownership (`store_of`).
       - Camera information (on-board rover, supported modes, calibration target).
       - Objective visibility from waypoints (`visible_from`).
       - Initial soil and rock sample locations.
       - Waypoints visible from the lander location (for communication), assuming `visible` is symmetric for this purpose.
    2. Builds a directed waypoint graph for each rover. An edge exists from waypoint A to waypoint B for rover R if `(can_traverse R A B)` and `(visible A B)` are true in the static facts.
    3. Precomputes all-pairs shortest path distances for each rover on its respective waypoint graph using Breadth-First Search (BFS). These distances represent the minimum number of `navigate` actions required.

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
    For a given state, the heuristic calculates the sum of estimated costs for
    each goal fact that is not yet satisfied in the state.

    1.  **Parse Dynamic State:** Extract current dynamic information from the state:
        - Rover locations (`at`).
        - Store status (`empty`, `full`).
        - Acquired data (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
        - Calibrated cameras (`calibrated`).
        - Remaining sample locations (`at_soil_sample`, `at_rock_sample`).

    2.  **Identify Unachieved Goals:** Determine the set of goal facts from `task.goals` that are not present in the current `state`.

    3.  **Iterate Through Unachieved Goals:** For each unachieved goal fact:
        -   **If the goal is `(communicated_soil_data ?w)`:**
            -   Check if any rover `R` currently holds `(have_soil_analysis R ?w)`.
            -   If yes: The cost is the minimum navigation cost for `R` from its current location to any waypoint visible from the lander, plus 1 action for `communicate_soil_data`.
            -   If no (data needs to be sampled):
                -   Check if `(at_soil_sample ?w)` is still in the state. If not, the sample is gone, and this goal is considered unreachable (cost is infinity).
                -   If the sample is available, find all rovers `R` equipped for soil analysis. If none exist, the goal is unreachable (cost is infinity).
                -   For each suitable rover `R`:
                    -   Calculate the cost to get the data: navigation from `R`'s current location to `?w`, plus 1 action for `sample_soil`, plus 1 action for `drop` if `R`'s store is full.
                    -   Calculate the cost to communicate: navigation from `?w` (where sampling occurred) to any waypoint visible from the lander, plus 1 action for `communicate_soil_data`.
                    -   The total cost for this rover is the sum of the cost to get data and the cost to communicate.
                -   The estimated cost for this goal is the minimum total cost across all suitable rovers.
        -   **If the goal is `(communicated_rock_data ?w)`:** Follow the same logic as for soil data, using rock-specific predicates and equipment.
        -   **If the goal is `(communicated_image_data ?o ?m)`:**
            -   Check if any rover `R` currently holds `(have_image R ?o ?m)`.
            -   If yes: The cost is the minimum navigation cost for `R` from its current location to any waypoint visible from the lander, plus 1 action for `communicate_image_data`.
            -   If no (image needs to be taken):
                -   Find all rover-camera pairs `(R, I)` where `R` is equipped for imaging, `I` is on board `R`, and `I` supports mode `?m`. If none exist, the goal is unreachable (cost is infinity).
                -   For each suitable pair `(R, I)`:
                    -   Find the calibration target `T` for camera `I`.
                    -   Find the best waypoint `W_cal` visible from `T` that minimizes navigation cost for `R` from its current location. Cost = navigation to `W_cal` + 1 action for `calibrate`.
                    -   Find the best waypoint `W_img` visible from `?o` that minimizes navigation cost for `R` from `W_cal`. Cost = navigation from `W_cal` to `W_img` + 1 action for `take_image`.
                    -   Find the best waypoint `W_comm` visible from the lander that minimizes navigation cost for `R` from `W_img`. Cost = navigation from `W_img` to `W_comm` + 1 action for `communicate_image_data`.
                    -   The total cost for this pair is the sum of the calibration cost, imaging cost, and communication cost.
                -   The estimated cost for this goal is the minimum total cost across all suitable rover-camera pairs.

    4.  **Sum Costs:** The total heuristic value is the sum of the estimated costs for all unachieved goals. If any goal was determined to be unreachable (cost infinity), the total heuristic is a large integer (sys.maxsize).

    5.  **Return Value:** Return the calculated total heuristic value as an integer.
    """
    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information and distances.
        """
        self.task = task
        self._parse_static(task.static)
        self._precompute_distances()

    def _parse_static(self, static_facts):
        """Parses static facts into useful data structures."""
        # Collect all objects first by inspecting fact arguments
        all_objects_in_static = set()
        for fact_string in static_facts:
             predicate, *objects = parse_fact(fact_string)
             all_objects_in_static.update(objects)

        # Infer object sets based on common naming conventions and predicates
        self.rovers = {obj for obj in all_objects_in_static if obj.startswith('rover')}
        self.waypoints = {obj for obj in all_objects_in_static if obj.startswith('waypoint')}
        self.objectives = {obj for obj in all_objects_in_static if obj.startswith('objective')}
        self.modes = {obj for obj in all_objects_in_static if obj.startswith('mode')}
        self.cameras = {obj for obj in all_objects_in_static if obj.startswith('camera')}
        self.stores = {obj for obj in all_objects_in_static if obj.startswith('store')}
        # Find lander object(s) by looking for the (at_lander ?l ?y) fact
        self.landers = {obj for fact_string in static_facts for predicate, *objects in [parse_fact(fact_string)] if predicate == 'at_lander' for obj in [objects[0]]}


        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_store = {} # rover -> store
        self.waypoint_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> waypoint -> set of connected waypoints
        self.lander_location = None
        self.camera_info = {} # camera -> {'rover': r, 'modes': {m}, 'calibration_target': t}
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoint}
        self.soil_sample_locations_init = set() # Initial locations
        self.rock_sample_locations_init = set() # Initial locations
        self.lander_visible_waypoints = set() # Waypoints visible from lander

        # Temporary storage for graph edges before building rover graphs
        rover_can_traverse = collections.defaultdict(lambda: collections.defaultdict(set))
        visible_pairs = set()

        for fact_string in static_facts:
            predicate, *objects = parse_fact(fact_string)

            if predicate == 'at_lander':
                self.lander_location = objects[1] # (at_lander ?x - lander ?y - waypoint)
            elif predicate == 'equipped_for_soil_analysis':
                self.equipped_soil.add(objects[0])
            elif predicate == 'equipped_for_rock_analysis':
                self.equipped_rock.add(objects[0])
            elif predicate == 'equipped_for_imaging':
                self.equipped_imaging.add(objects[0])
            elif predicate == 'store_of':
                self.rover_store[objects[1]] = objects[0] # (store_of ?s - store ?r - rover)
            elif predicate == 'can_traverse':
                 # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                 rover, wp1, wp2 = objects
                 rover_can_traverse[rover][wp1].add(wp2)
            elif predicate == 'visible':
                 # (visible ?w - waypoint ?p - waypoint)
                 wp1, wp2 = objects
                 visible_pairs.add((wp1, wp2))
                 # Assuming visible is symmetric for communication purposes
                 # Add reverse edge for lander visibility check later
                 if wp2 == self.lander_location:
                     self.lander_visible_waypoints.add(wp1)
                 if wp1 == self.lander_location:
                     self.lander_visible_waypoints.add(wp2)

            elif predicate == 'on_board':
                 # (on_board ?i - camera ?r - rover)
                 camera, rover = objects
                 if camera not in self.camera_info: self.camera_info[camera] = {}
                 self.camera_info[camera]['rover'] = rover
            elif predicate == 'supports':
                 # (supports ?c - camera ?m - mode)
                 camera, mode = objects
                 if camera not in self.camera_info: self.camera_info[camera] = {}
                 if 'modes' not in self.camera_info[camera]: self.camera_info[camera]['modes'] = set()
                 self.camera_info[camera]['modes'].add(mode)
            elif predicate == 'calibration_target':
                 # (calibration_target ?i - camera ?o - objective)
                 camera, objective = objects
                 if camera not in self.camera_info: self.camera_info[camera] = {}
                 self.camera_info[camera]['calibration_target'] = objective
            elif predicate == 'visible_from':
                 # (visible_from ?o - objective ?w - waypoint)
                 objective, waypoint = objects
                 self.objective_visible_from[objective].add(waypoint)
            elif predicate == 'at_soil_sample':
                 self.soil_sample_locations_init.add(objects[0])
            elif predicate == 'at_rock_sample':
                 self.rock_sample_locations_init.add(objects[0])

        # Build rover-specific traversal graphs based on can_traverse and visible
        # navigate ?x ?y ?z requires (can_traverse ?x ?y ?z) and (visible ?y ?z)
        # This means rover ?x can go from ?y to ?z if both are true.
        # The graph edge is y -> z for rover x.
        for rover in self.rovers:
            for wp1 in self.waypoints:
                for wp2 in self.waypoints:
                    if wp2 in rover_can_traverse[rover].get(wp1, set()) and (wp1, wp2) in visible_pairs:
                         self.waypoint_graph[rover][wp1].add(wp2)


    def _precompute_distances(self):
        """Precomputes shortest path distances for each rover."""
        self.dist = {}
        for rover in self.rovers:
            rover_graph = self.waypoint_graph.get(rover, {})
            self.dist[rover] = self._bfs_from_all(rover_graph, self.waypoints)

    def _bfs_from_all(self, graph, nodes):
        """Runs BFS from every node to find distances to all other nodes."""
        distances = {}
        for start_node in nodes:
            distances[start_node] = {}
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            distances[start_node][start_node] = 0

            while queue:
                current_node, dist = queue.popleft()

                for neighbor in graph.get(current_node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        return distances

    def _get_min_dist_and_wp_to_set(self, rover, start_wp, target_waypoints):
        """
        Finds the minimum distance for a rover from start_wp to any waypoint in target_waypoints
        and returns the distance and the target waypoint.
        Returns (float('inf'), None) if no target is reachable or target_waypoints is empty.
        """
        if not target_waypoints or start_wp not in self.dist.get(rover, {}):
            return float('inf'), None

        min_d = float('inf')
        best_wp = None

        distances_from_start = self.dist[rover][start_wp] # Use precomputed distances
        for target_wp in target_waypoints:
             d = distances_from_start.get(target_wp, float('inf'))
             if d < min_d:
                 min_d = d
                 best_wp = target_wp

        return min_d, best_wp

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated number of actions to reach a goal state,
                 or a large integer if a goal is unreachable.
        """
        # Heuristic is 0 for goal states
        if self.task.goal_reached(state):
            return 0

        # Parse dynamic facts from the current state
        rover_location = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil = collections.defaultdict(set) # rover -> {waypoint}
        have_rock = collections.defaultdict(set) # rover -> {waypoint}
        have_image = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> objective -> {mode}
        calibrated_cameras = set() # {camera}
        soil_samples_in_state = set() # Locations where soil samples are currently present
        rock_samples_in_state = set() # Locations where rock samples are currently present


        for fact_string in state:
            predicate, *objects = parse_fact(fact_string)
            if predicate == 'at':
                rover_location[objects[0]] = objects[1]
            elif predicate == 'empty':
                store_status[objects[0]] = 'empty'
            elif predicate == 'full':
                store_status[objects[0]] = 'full'
            elif predicate == 'have_soil_analysis':
                have_soil[objects[0]].add(objects[1])
            elif predicate == 'have_rock_analysis':
                have_rock[objects[0]].add(objects[1])
            elif predicate == 'have_image':
                 # (have_image ?r - rover ?o - objective ?m - mode)
                 rover, objective, mode = objects
                 have_image[rover][objective].add(mode)
            elif predicate == 'calibrated':
                 calibrated_cameras.add(objects[0])
            elif predicate == 'at_soil_sample':
                 soil_samples_in_state.add(objects[0])
            elif predicate == 'at_rock_sample':
                 rock_samples_in_state.add(objects[0])


        total_heuristic = 0
        unachieved_goals = self.task.goals - state

        # Process unachieved goals
        for goal_fact_string in unachieved_goals:
            predicate, *objects = parse_fact(goal_fact_string)

            if predicate == 'communicated_soil_data':
                waypoint = objects[0]
                goal_cost = float('inf')

                # Check if any rover already has the soil analysis data for this waypoint
                rover_with_data = None
                for r in self.rovers:
                    if waypoint in have_soil.get(r, set()):
                        rover_with_data = r
                        break

                if rover_with_data:
                    # Data is available, just need to communicate
                    r = rover_with_data
                    current_pos = rover_location.get(r)
                    if current_pos:
                         # Cost = navigate to comm point + communicate
                         nav_cost, _ = self._get_min_dist_and_wp_to_set(r, current_pos, self.lander_visible_waypoints)
                         if nav_cost != float('inf'):
                             goal_cost = nav_cost + 1
                else:
                    # Need to sample and communicate
                    # Check if sample is still available at the location
                    if waypoint not in soil_samples_in_state:
                         # Sample is gone, cannot sample here. Goal is unreachable.
                         goal_cost = float('inf')
                    else:
                        suitable_rovers = self.equipped_soil
                        if not suitable_rovers:
                             goal_cost = float('inf') # No rover can sample soil
                        else:
                            min_rover_cost = float('inf')
                            for r in suitable_rovers:
                                current_pos = rover_location.get(r)
                                if not current_pos: continue # Rover location unknown (shouldn't happen in valid state)

                                # Cost to sample: navigate to waypoint + (drop if needed) + sample
                                nav_to_sample_cost, _ = self._get_min_dist_and_wp_to_set(r, current_pos, {waypoint}) # Dist to the specific sample waypoint
                                if nav_to_sample_cost == float('inf'): continue # Cannot reach sample location

                                store = self.rover_store.get(r)
                                drop_cost = 1 if store and store_status.get(store) == 'full' else 0

                                cost_to_get_data = nav_to_sample_cost + drop_cost + 1 # navigate + drop (if needed) + sample

                                # Cost to communicate: navigate from sample waypoint to comm point + communicate
                                nav_to_comm_cost, _ = self._get_min_dist_and_wp_to_set(r, waypoint, self.lander_visible_waypoints)
                                if nav_to_comm_cost == float('inf'): continue # Cannot reach comm location from sample location

                                cost_to_communicate = nav_to_comm_cost + 1 # navigate + communicate

                                min_rover_cost = min(min_rover_cost, cost_to_get_data + cost_to_communicate)

                            goal_cost = min_rover_cost

                total_heuristic += goal_cost

            elif predicate == 'communicated_rock_data':
                waypoint = objects[0]
                goal_cost = float('inf')

                # Check if any rover already has the rock analysis data for this waypoint
                rover_with_data = None
                for r in self.rovers:
                    if waypoint in have_rock.get(r, set()):
                        rover_with_data = r
                        break

                if rover_with_data:
                    # Data is available, just need to communicate
                    r = rover_with_data
                    current_pos = rover_location.get(r)
                    if current_pos:
                         nav_cost, _ = self._get_min_dist_and_wp_to_set(r, current_pos, self.lander_visible_waypoints)
                         if nav_cost != float('inf'):
                             goal_cost = nav_cost + 1
                else:
                    # Need to sample and communicate
                    # Check if sample is still available
                    if waypoint not in rock_samples_in_state:
                         goal_cost = float('inf') # Cannot sample if sample is gone
                    else:
                         suitable_rovers = self.equipped_rock
                         if not suitable_rovers:
                              goal_cost = float('inf') # No rover can sample rock
                         else:
                             min_rover_cost = float('inf')
                             for r in suitable_rovers:
                                 current_pos = rover_location.get(r)
                                 if not current_pos: continue

                                 nav_to_sample_cost, _ = self._get_min_dist_and_wp_to_set(r, current_pos, {waypoint})
                                 if nav_to_sample_cost == float('inf'): continue

                                 store = self.rover_store.get(r)
                                 drop_cost = 1 if store and store_status.get(store) == 'full' else 0

                                 cost_to_get_data = nav_to_sample_cost + drop_cost + 1

                                 nav_to_comm_cost, _ = self._get_min_dist_and_wp_to_set(r, waypoint, self.lander_visible_waypoints)
                                 if nav_to_comm_cost == float('inf'): continue

                                 cost_to_communicate = nav_to_comm_cost + 1

                                 min_rover_cost = min(min_rover_cost, cost_to_get_data + cost_to_communicate)

                             goal_cost = min_rover_cost

                 total_heuristic += goal_cost

            elif predicate == 'communicated_image_data':
                objective, mode = objects
                goal_cost = float('inf')

                # Check if any rover already has the image data for this objective and mode
                rover_with_data = None
                for r in self.rovers:
                    if objective in have_image.get(r, {}) and mode in have_image[r].get(objective, set()):
                        rover_with_data = r
                        break

                if rover_with_data:
                    # Data is available, just need to communicate
                    r = rover_with_data
                    current_pos = rover_location.get(r)
                    if current_pos:
                         nav_cost, _ = self._get_min_dist_and_wp_to_set(r, current_pos, self.lander_visible_waypoints)
                         if nav_cost != float('inf'):
                             goal_cost = nav_cost + 1
                else:
                    # Need to image and communicate
                    suitable_rover_camera_pairs = []
                    for camera, info in self.camera_info.items():
                         r = info.get('rover')
                         supported_modes = info.get('modes', set())
                         if r and r in self.equipped_imaging and mode in supported_modes: # Ensure rover exists and is equipped
                             suitable_rover_camera_pairs.append((r, camera))

                    if not suitable_rover_camera_pairs:
                         goal_cost = float('inf') # No rover/camera can take this image
                    else:
                        min_pair_cost = float('inf')
                        for r, camera in suitable_rover_camera_pairs:
                            current_pos = rover_location.get(r)
                            if not current_pos: continue

                            cal_target = self.camera_info.get(camera, {}).get('calibration_target')
                            if not cal_target: continue # Camera has no calibration target

                            # Find best calibration waypoint
                            cal_target_visible_wps = self.objective_visible_from.get(cal_target, set())
                            if not cal_target_visible_wps: continue # Calibration target not visible from anywhere

                            nav_to_cal_cost, best_cal_wp = self._get_min_dist_and_wp_to_set(r, current_pos, cal_target_visible_wps)
                            if nav_to_cal_cost == float('inf'): continue # Cannot reach any calibration waypoint

                            cost_to_calibrate = nav_to_cal_cost + 1 # navigate + calibrate

                            # Find best imaging waypoint (from best_cal_wp)
                            objective_visible_wps = self.objective_visible_from.get(objective, set())
                            if not objective_visible_wps: continue # Objective not visible from anywhere

                            nav_to_img_cost, best_img_wp = self._get_min_dist_and_wp_to_set(r, best_cal_wp, objective_visible_wps)
                            if nav_to_img_cost == float('inf'): continue # Cannot reach any imaging waypoint

                            cost_to_image = nav_to_img_cost + 1 # navigate + take_image

                            # Find best communication waypoint (from best_img_wp)
                            nav_to_comm_cost, _ = self._get_min_dist_and_wp_to_set(r, best_img_wp, self.lander_visible_waypoints)
                            if nav_to_comm_cost == float('inf'): continue # Cannot reach comm location from image location

                            cost_to_communicate = nav_to_comm_cost + 1 # navigate + communicate

                            total_pair_cost = cost_to_calibrate + cost_to_image + cost_to_communicate
                            min_pair_cost = min(min_pair_cost, total_pair_cost)

                        goal_cost = min_pair_cost

                total_heuristic += goal_cost

        # If total_heuristic is still float('inf'), it means at least one goal is unreachable.
        # Return a large integer.
        return int(total_heuristic) if total_heuristic != float('inf') else sys.maxsize

