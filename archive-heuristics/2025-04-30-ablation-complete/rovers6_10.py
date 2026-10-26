import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available in this path

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

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

def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search to find shortest path distances from a start node
    to all other nodes in a graph.

    Args:
        graph: Adjacency list representation of the graph (dict: node -> set of neighbors).
        start_node: The starting node for the BFS.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary mapping each node to its shortest distance from the start_node.
        Returns float('inf') for unreachable nodes.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node in all_nodes:
        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            current_node = queue.popleft()
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated
    costs for each unachieved goal predicate. For each unachieved goal, it estimates
    the cost of the final communication action, the cost of acquiring the necessary
    data (sampling or imaging), and the navigation costs associated with these steps.
    It tries to select the "best" available rover/camera for data acquisition based
    on current location and required movement.

    # Assumptions
    - The cost of each action (navigate, sample, drop, calibrate, take_image, communicate) is 1.
    - Navigation cost between waypoints is the shortest path distance in the
      rover-specific traversable graph.
    - When data needs to be acquired (sampled or imaged), the heuristic estimates
      the cost for the single "best" available rover/camera combination (e.g., closest
      to the sample/objective waypoint).
    - When data is already acquired, the heuristic estimates the cost for the rover
      that currently holds the data.
    - Drop action cost is added only if the chosen rover for sampling has a full store.
    - Calibration cost is added only if the chosen camera for imaging is not calibrated.
    - Unreachable waypoints result in an infinite heuristic value.

    # Heuristic Initialization
    - Extracts static facts: lander location, waypoint visibility, rover capabilities,
      camera information (on-board, supported modes, calibration target), objective
      visibility, store ownership, and rover traversal capabilities.
    - Builds the navigation graph (considering both `can_traverse` and `visible`)
      for each rover.
    - Precomputes all-pairs shortest path distances for each rover using BFS.
    - Identifies waypoints visible from the lander location.
    - Stores goal predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current state is a goal state. If yes, return 0.
    2.  Parse the current state to get dynamic facts: rover locations, `have_` predicates
        (soil, rock, image), store statuses, camera calibration statuses, and
        `communicated_` predicates.
    3.  Initialize `total_cost = 0`.
    4.  Iterate through each goal predicate in the task's goals.
    5.  If a goal predicate is already true in the current state, skip it.
    6.  For an unachieved goal `(communicated_soil_data W)`:
        - Add 1 for the `communicate_soil_data` action.
        - Check if any equipped soil rover `R` has `(have_soil_analysis R W)`.
        - If yes: Find the rover `R_data` that has the data. Add the minimum
          navigation cost for `R_data` from its current location to any waypoint
          visible from the lander. If unreachable, return infinity.
        - If no: Add 1 for the `sample_soil` action. Find the equipped soil rover
          `R_sample` that is closest to `W`. Add the navigation cost for `R_sample`
          from its current location to `W`. If unreachable, return infinity.
          Check if `R_sample`'s store is full; if yes, add 1 for the `drop` action.
    7.  For an unachieved goal `(communicated_rock_data W)`: Follow similar logic
        as for soil data, using rock-specific predicates and equipped rock rovers.
    8.  For an unachieved goal `(communicated_image_data O M)`:
        - Add 1 for the `communicate_image_data` action.
        - Check if any equipped imaging rover `R` with camera `I` supporting mode `M`
          has `(have_image R O M)`.
        - If yes: Find the rover `R_data` that has the image. Add the minimum
          navigation cost for `R_data` from its current location to any waypoint
          visible from the lander. If unreachable, return infinity.
        - If no: Add 1 for the `take_image` action. Find the equipped imaging rover
          `R_img` with camera `I_img` supporting `M` that minimizes navigation
          cost from its current location to any waypoint visible from objective `O`.
          Add this minimum navigation cost. If unreachable, return infinity.
          Check if `(calibrated I_img R_img)` is true; if no, add 1 for the
          `calibrate` action. If calibration is needed, find the calibration target
          `T` for `I_img` and add the minimum navigation cost for `R_img` from its
          current location to any waypoint visible from `T`. If unreachable, return infinity.
    9.  Return the `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = None
        self.lander_visible_waypoints = set()
        self.rover_capabilities = collections.defaultdict(set) # rover -> set(capabilities)
        self.camera_info = {} # camera -> {'on_board': rover, 'supports': set(modes), 'calibration_target': objective}
        self.objective_visibility = collections.defaultdict(set) # objective -> set(waypoints)
        self.store_owners = {} # store -> rover
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_cameras = set()
        self.all_stores = set()

        # Temporary storage for graph building
        rover_can_traverse = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> wp_from -> set(wp_to)
        visible_graph = collections.defaultdict(set) # wp_from -> set(wp_to)
        symmetric_visible_graph = collections.defaultdict(set) # for lander visibility

        # --- Parse Static Facts ---
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'at_lander':
                self.lander_location = parts[2]
            elif pred == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
            elif pred == 'store_of':
                self.store_owners[parts[1]] = parts[2]
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['on_board'] = rover
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                if 'supports' not in self.camera_info[camera]: self.camera_info[camera]['supports'] = set()
                self.camera_info[camera]['supports'].add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info: self.camera_info[camera] = {}
                self.camera_info[camera]['calibration_target'] = objective
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility[objective].add(waypoint)
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                rover_can_traverse[rover][wp1].add(wp2)
            elif pred == 'visible':
                wp1, wp2 = parts[1], parts[2]
                visible_graph[wp1].add(wp2)
                symmetric_visible_graph[wp1].add(wp2)
                symmetric_visible_graph[wp2].add(wp1) # Assume symmetric visibility for lander check

        # --- Collect all object names from static facts and goals ---
        all_facts = set(static_facts) | set(self.goals)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            # Add objects based on predicate signature (simplified)
            for i, part in enumerate(parts[1:]):
                 # Basic type inference based on argument position in common predicates
                 if pred in ['at', 'can_traverse', 'have_rock_analysis', 'have_soil_analysis', 'calibrated', 'on_board', 'store_of', 'equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                     if i == 0: self.all_rovers.add(part) # First arg is often rover
                     if pred in ['at', 'have_rock_analysis', 'have_soil_analysis'] and i == 1: self.all_waypoints.add(part) # Second arg is waypoint
                     if pred == 'can_traverse': self.all_waypoints.add(part) # Second and third args are waypoints
                     if pred in ['calibrated', 'on_board', 'supports', 'calibration_target'] and i == 0: self.all_cameras.add(part) # First arg is camera
                     if pred in ['calibrated', 'on_board'] and i == 1: self.all_rovers.add(part) # Second arg is rover
                     if pred == 'store_of' and i == 0: self.all_stores.add(part) # First arg is store
                     if pred == 'store_of' and i == 1: self.all_rovers.add(part) # Second arg is rover
                 elif pred in ['at_lander', 'communicated_soil_data', 'communicated_rock_data', 'at_soil_sample', 'at_rock_sample']:
                     if i == len(parts) - 2: self.all_waypoints.add(part) # Last arg is waypoint
                 elif pred in ['empty', 'full']:
                     if i == 0: self.all_stores.add(part) # First arg is store
                 elif pred in ['supports']:
                     if i == 1: self.all_modes.add(part) # Second arg is mode
                 elif pred in ['have_image', 'communicated_image_data']:
                     if pred == 'have_image' and i == 0: self.all_rovers.add(part) # First arg is rover
                     if i == len(parts) - 3: self.all_objectives.add(part) # Second to last is objective
                     if i == len(parts) - 2: self.all_modes.add(part) # Last is mode
                 elif pred in ['visible', 'visible_from']:
                     if i == len(parts) - 3: self.all_waypoints.add(part) # Second to last is waypoint
                     if i == len(parts) - 2: self.all_waypoints.add(part) # Last is waypoint
                     if pred == 'visible_from' and i == 0: self.all_objectives.add(part) # First arg is objective
                 elif pred == 'calibration_target':
                     if i == 1: self.all_objectives.add(part) # Second arg is objective

        # Ensure all objects mentioned in static structures are added
        self.all_rovers.update(self.rover_capabilities.keys())
        self.all_stores.update(self.store_owners.keys())
        self.all_rovers.update(self.store_owners.values())
        self.all_cameras.update(self.camera_info.keys())
        for cam_info in self.camera_info.values():
            if 'on_board' in cam_info: self.all_rovers.add(cam_info['on_board'])
            if 'calibration_target' in cam_info: self.all_objectives.add(cam_info['calibration_target'])
            if 'supports' in cam_info: self.all_modes.update(cam_info['supports'])
        self.all_objectives.update(self.objective_visibility.keys())
        for wps in self.objective_visibility.values(): self.all_waypoints.update(wps)
        for wp_from, wps_to in visible_graph.items():
            self.all_waypoints.add(wp_from)
            self.all_waypoints.update(wps_to)
        for rover, wp_from_dict in rover_can_traverse.items():
            self.all_rovers.add(rover)
            for wp_from, wps_to in wp_from_dict.items():
                self.all_waypoints.add(wp_from)
                self.all_waypoints.update(wps_to)


        # --- Calculate Lander Visible Waypoints ---
        if self.lander_location and self.lander_location in self.all_waypoints:
            self.lander_visible_waypoints.add(self.lander_location)
            if self.lander_location in symmetric_visible_graph:
                 self.lander_visible_waypoints.update(symmetric_visible_graph[self.lander_location])


        # --- Calculate Rover Distances ---
        self.rover_distances = {}
        for rover in self.all_rovers:
            rover_graph = collections.defaultdict(set)
            # Edges exist if (can_traverse R W1 W2) AND (visible W1 W2)
            for wp1 in self.all_waypoints:
                for wp2 in self.all_waypoints:
                    can_t = f'(can_traverse {rover} {wp1} {wp2})' in static_facts
                    vis = f'(visible {wp1} {wp2})' in static_facts
                    if can_t and vis:
                         rover_graph[wp1].add(wp2)

            self.rover_distances[rover] = {}
            for start_wp in self.all_waypoints:
                self.rover_distances[rover][start_wp] = bfs(rover_graph, start_wp, self.all_waypoints)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        current_rover_locations = {} # rover -> waypoint
        current_have_soil = collections.defaultdict(set) # rover -> set(waypoints)
        current_have_rock = collections.defaultdict(set) # rover -> set(waypoints)
        current_have_image = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> objective -> set(modes)
        current_store_status = {} # store -> 'empty' | 'full'
        current_calibrated_cameras = set() # set((camera, rover))
        current_communicated_soil = set() # set(waypoints)
        current_communicated_rock = set() # set(waypoints)
        current_communicated_image = set() # set((objective, mode))

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at':
                current_rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                current_have_soil[parts[1]].add(parts[2])
            elif pred == 'have_rock_analysis':
                current_have_rock[parts[1]].add(parts[2])
            elif pred == 'have_image':
                current_have_image[parts[1]][parts[2]].add(parts[3])
            elif pred == 'empty':
                current_store_status[parts[1]] = 'empty'
            elif pred == 'full':
                current_store_status[parts[1]] = 'full'
            elif pred == 'calibrated':
                current_calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'communicated_soil_data':
                current_communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data':
                current_communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data':
                current_communicated_image.add((parts[1], parts[2]))

        # --- Heuristic Calculation ---
        total_cost = 0

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Process unachieved goals
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint_goal = parts[1]
                total_cost += 1 # Cost for communicate action

                # Check if data is already acquired by any equipped soil rover
                rover_with_data = None
                for rover in self.all_rovers:
                    if 'soil' in self.rover_capabilities.get(rover, set()) and waypoint_goal in current_have_soil.get(rover, set()):
                        rover_with_data = rover # Found a rover with the data
                        break

                if rover_with_data:
                    # Data exists, need to move rover_with_data to lander-visible waypoint
                    rover_loc = current_rover_locations.get(rover_with_data)
                    if rover_loc is None: return float('inf') # Rover location unknown

                    min_nav_cost = float('inf')
                    for lander_wp in self.lander_visible_waypoints:
                        dist = self.rover_distances[rover_with_data][rover_loc].get(lander_wp, float('inf'))
                        min_nav_cost = min(min_nav_cost, dist)

                    if min_nav_cost == float('inf'): return float('inf') # Unreachable
                    total_cost += min_nav_cost

                else:
                    # Data not acquired, need to sample
                    total_cost += 1 # Cost for sample action

                    # Find best equipped soil rover to sample (closest to waypoint_goal)
                    min_nav_cost_to_sample = float('inf')
                    best_sampler = None
                    for rover in self.all_rovers:
                        if 'soil' in self.rover_capabilities.get(rover, set()):
                            rover_loc = current_rover_locations.get(rover)
                            if rover_loc:
                                nav_cost = self.rover_distances[rover][rover_loc].get(waypoint_goal, float('inf'))
                                if nav_cost < min_nav_cost_to_sample:
                                    min_nav_cost_to_sample = nav_cost
                                    best_sampler = rover

                    if min_nav_cost_to_sample == float('inf'): return float('inf') # Unreachable
                    total_cost += min_nav_cost_to_sample

                    # Add cost for drop if the chosen sampler's store is full
                    if best_sampler:
                        sampler_store = None
                        for store, owner in self.store_owners.items():
                            if owner == best_sampler:
                                sampler_store = store
                                break
                        if sampler_store and current_store_status.get(sampler_store) == 'full':
                            total_cost += 1 # Cost for drop action

            elif pred == 'communicated_rock_data':
                waypoint_goal = parts[1]
                total_cost += 1 # Cost for communicate action

                # Check if data is already acquired by any equipped rock rover
                rover_with_data = None
                for rover in self.all_rovers:
                    if 'rock' in self.rover_capabilities.get(rover, set()) and waypoint_goal in current_have_rock.get(rover, set()):
                        rover_with_data = rover # Found a rover with the data
                        break

                if rover_with_data:
                    # Data exists, need to move rover_with_data to lander-visible waypoint
                    rover_loc = current_rover_locations.get(rover_with_data)
                    if rover_loc is None: return float('inf') # Rover location unknown

                    min_nav_cost = float('inf')
                    for lander_wp in self.lander_visible_waypoints:
                        dist = self.rover_distances[rover_with_data][rover_loc].get(lander_wp, float('inf'))
                        min_nav_cost = min(min_nav_cost, dist)

                    if min_nav_cost == float('inf'): return float('inf') # Unreachable
                    total_cost += min_nav_cost

                else:
                    # Data not acquired, need to sample
                    total_cost += 1 # Cost for sample action

                    # Find best equipped rock rover to sample (closest to waypoint_goal)
                    min_nav_cost_to_sample = float('inf')
                    best_sampler = None
                    for rover in self.all_rovers:
                        if 'rock' in self.rover_capabilities.get(rover, set()):
                            rover_loc = current_rover_locations.get(rover)
                            if rover_loc:
                                nav_cost = self.rover_distances[rover][rover_loc].get(waypoint_goal, float('inf'))
                                if nav_cost < min_nav_cost_to_sample:
                                    min_nav_cost_to_sample = nav_cost
                                    best_sampler = rover

                    if min_nav_cost_to_sample == float('inf'): return float('inf') # Unreachable
                    total_cost += min_nav_cost_to_sample

                    # Add cost for drop if the chosen sampler's store is full
                    if best_sampler:
                        sampler_store = None
                        for store, owner in self.store_owners.items():
                            if owner == best_sampler:
                                sampler_store = store
                                break
                        if sampler_store and current_store_status.get(sampler_store) == 'full':
                            total_cost += 1 # Cost for drop action

            elif pred == 'communicated_image_data':
                objective_goal, mode_goal = parts[1], parts[2]
                total_cost += 1 # Cost for communicate action

                # Check if data is already acquired by any suitable rover/camera
                rover_with_data = None
                for rover in self.all_rovers:
                     if objective_goal in current_have_image.get(rover, {}) and mode_goal in current_have_image[rover][objective_goal]:
                         # Assume if the fact exists, it was validly acquired by this rover
                         rover_with_data = rover
                         break

                if rover_with_data:
                    # Data exists, need to move rover_with_data to lander-visible waypoint
                    rover_loc = current_rover_locations.get(rover_with_data)
                    if rover_loc is None: return float('inf') # Rover location unknown

                    min_nav_cost = float('inf')
                    for lander_wp in self.lander_visible_waypoints:
                        dist = self.rover_distances[rover_with_data][rover_loc].get(lander_wp, float('inf'))
                        min_nav_cost = min(min_nav_cost, dist)

                    if min_nav_cost == float('inf'): return float('inf') # Unreachable
                    total_cost += min_nav_cost

                else:
                    # Data not acquired, need to take image
                    total_cost += 1 # Cost for take_image action

                    # Find best suitable rover/camera to take image
                    min_nav_cost_to_objective_wp = float('inf')
                    best_imager_rover = None
                    best_imager_camera = None

                    # Find waypoints visible from the objective
                    objective_wps = self.objective_visibility.get(objective_goal, set())
                    if not objective_wps: return float('inf') # Cannot see objective from anywhere

                    for rover in self.all_rovers:
                        if 'imaging' in self.rover_capabilities.get(rover, set()):
                            rover_loc = current_rover_locations.get(rover)
                            if rover_loc is None: continue # Rover location unknown

                            for camera, cam_info in self.camera_info.items():
                                if cam_info.get('on_board') == rover and mode_goal in cam_info.get('supports', set()):
                                    # This rover/camera is suitable
                                    min_nav_from_rover_to_objective_wp = float('inf')
                                    for obj_wp in objective_wps:
                                        dist = self.rover_distances[rover][rover_loc].get(obj_wp, float('inf'))
                                        min_nav_from_rover_to_objective_wp = min(min_nav_from_rover_to_objective_wp, dist)

                                    if min_nav_from_rover_to_objective_wp < min_nav_cost_to_objective_wp:
                                        min_nav_cost_to_objective_wp = min_nav_from_rover_to_objective_wp
                                        best_imager_rover = rover
                                        best_imager_camera = camera

                    if min_nav_cost_to_objective_wp == float('inf'): return float('inf') # Unreachable
                    total_cost += min_nav_cost_to_objective_wp

                    # Add cost for calibration if the chosen camera is not calibrated
                    if best_imager_rover and best_imager_camera and (best_imager_camera, best_imager_rover) not in current_calibrated_cameras:
                         total_cost += 1 # Cost for calibrate action

                         # Find calibration target and visible waypoints
                         cal_target = self.camera_info.get(best_imager_camera, {}).get('calibration_target')
                         if not cal_target: return float('inf') # Camera has no calibration target?
                         cal_target_wps = self.objective_visibility.get(cal_target, set())
                         if not cal_target_wps: return float('inf') # Cannot see calibration target from anywhere

                         rover_loc = current_rover_locations.get(best_imager_rover)
                         if rover_loc is None: return float('inf') # Rover location unknown

                         min_nav_cost_to_cal_wp = float('inf')
                         for cal_wp in cal_target_wps:
                             dist = self.rover_distances[best_imager_rover][rover_loc].get(cal_wp, float('inf'))
                             min_nav_cost_to_cal_wp = min(min_nav_cost_to_cal_wp, dist)
                         if min_nav_cost_to_cal_wp == float('inf'): return float('inf') # Unreachable
                         total_cost += min_nav_cost_to_cal_wp


        # Ensure heuristic is 0 only at goal (already handled by initial check)
        # If total_cost is 0 here, it means all goals were initially in the state,
        # which is caught by the first check.

        return total_cost
