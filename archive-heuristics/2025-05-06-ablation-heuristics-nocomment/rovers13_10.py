from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we have enough parts to match against args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_shortest_path_distance(graph, start, end):
    """
    Performs BFS to find the shortest path distance between two waypoints.
    Returns the distance (number of moves) or float('inf') if unreachable.
    """
    if start == end:
        return 0
    # Check if start or end are valid nodes in the graph
    if start not in graph or end not in graph:
        return float('inf')

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_wp, dist = queue.popleft()

        if current_wp == end:
            return dist

        # Ensure current_wp is in the graph dictionary before accessing neighbors
        if current_wp in graph:
            for neighbor in graph[current_wp]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Not reachable

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing up the estimated costs
    for each unachieved goal fact independently. The cost for each goal
    is estimated based on the sequence of actions required (sample/image,
    calibrate if needed, communicate) and the necessary navigation between
    relevant waypoints (sample location, image location, calibration location,
    communication location). Navigation costs are estimated using shortest
    path distances on the waypoint graph for the relevant rover.

    Assumptions:
    - The heuristic assumes that any required sample or objective/mode
      combination is initially available and reachable by at least one
      capable rover.
    - It assumes that a rover, once it has a sample or image, is the one
      that will communicate it.
    - It simplifies the calibration process by assuming navigation from
      the rover's current location to the calibration point, the calibration
      action, navigation from the calibration point to the image point,
      and the take_image action are the steps needed if calibration is required.
    - It simplifies the store constraint by adding a fixed cost (1) if the
      store is full when sampling is needed, ignoring navigation to a
      drop point.
    - It ignores potential negative interactions between goals (e.g.,
      taking an image uncalibrates the camera, a full store prevents
      sampling for another goal).
    - It assumes communication is possible from any waypoint visible
      from the lander location.

    Heuristic Initialization:
    The constructor pre-processes static facts from the task to build
    data structures useful for the heuristic calculation:
    - Waypoint connectivity graphs for each rover based on 'can_traverse'.
    - Lander location and name based on 'at_lander'.
    - Communication points (waypoints visible from lander location) based on 'visible'.
    - Mappings for objective visibility ('visible_from').
    - Camera calibration targets ('calibration_target').
    - Rover camera equipment ('on_board').
    - Camera mode support ('supports').
    - Rover capabilities ('equipped_for_soil_analysis', etc.).
    - Store ownership ('store_of').
    - Initial locations of soil and rock samples from the initial state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic cost to 0.
    2. Identify all goal facts that are not currently true in the state.
    3. For each unachieved goal fact:
        a. Determine the type of goal (soil, rock, or image communication).
        b. If it's a soil or rock communication goal `(communicated_X_data ?w)`:
            i. Check if the corresponding analysis `(have_X_analysis ?r ?w)`
               exists for any rover `?r` in the current state.
            ii. If the analysis exists:
                - Find the rover `?r` holding the analysis.
                - Calculate the shortest path distance from the rover's
                  current location to the closest waypoint visible from
                  the lander.
                - Add this distance + 1 (for the communicate action) to the
                  total cost. If no such waypoint is reachable, add a large penalty (1000).
            iii. If the analysis does not exist:
                - Check if the sample `(at_X_sample ?w)` is still in the state. If not, and it was in the initial state, the goal is likely unreachable. Add a large penalty (1000).
                - Find the closest capable rover `?r` (equipped for X analysis).
                - Calculate the shortest path distance from the rover's
                  current location to the sample waypoint `?w`.
                - Add this distance + 1 (for the sample action) to the cost.
                - If the rover's store is full, add 1 (for a drop action).
                - Calculate the shortest path distance from the sample waypoint `?w`
                  (where the rover will be after sampling) to the closest waypoint
                  visible from the lander.
                - Add this distance + 1 (for the communicate action) to the cost.
                - If any required navigation is impossible, add a large penalty (1000).
                - Choose the minimum cost among all capable rovers.
                - Add the minimum cost to the total cost.
        c. If it's an image communication goal `(communicated_image_data ?o ?m)`:
            i. Check if the corresponding image `(have_image ?r ?o ?m)`
               exists for any rover `?r` in the current state.
            ii. If the image exists:
                - Find the rover `?r` holding the image.
                - Calculate the shortest path distance from the rover's
                  current location to the closest waypoint visible from
                  the lander.
                - Add this distance + 1 (for the communicate action) to the
                  total cost. If no such waypoint is reachable, add a large penalty (1000).
            iii. If the image does not exist:
                - Find a capable rover `?r` (equipped for imaging) with a camera `?i`
                  that supports mode `?m` and is on board.
                - Find the closest waypoint `w_image` visible from objective `?o`.
                - Calculate the shortest path distance from the rover's current
                  location to `w_image`.
                - Add this distance + 1 (for the take_image action) to the cost.
                - Check if camera `?i` is calibrated for rover `?r`.
                - If not calibrated:
                    - Find the camera's calibration target `?t`.
                    - Find the closest waypoint `w_calib` visible from `?t`.
                    - Calculate the shortest path distance from the rover's current
                      location to `w_calib`.
                    - Add this distance + 1 (for the calibrate action) to the cost.
                    - Calculate the shortest path distance from `w_calib` back to `w_image`.
                    - Add this distance to the cost.
                    - If any required navigation is impossible, add a large penalty (1000).
                    - Recalculate the cost to image including the calibration steps:
                      Nav(curr->calib) + Calibrate + Nav(calib->image) + TakeImage.
                - Calculate the shortest path distance from `w_image` (where the rover
                  will be after taking the image) to the closest waypoint visible
                  from the lander.
                - Add this distance + 1 (for the communicate action) to the cost.
                - Choose the minimum cost among all suitable rover/camera combinations.
                - Add the minimum cost to the total cost.
    4. Return the total cost.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Pre-process static facts
        self.rover_graphs = {} # rover -> {wp -> [neighbor_wp, ...]}
        self.lander_location = None
        self.lander_name = None
        self.comm_points = {} # lander -> [comm_wp, ...]
        self.obj_visible_from = {} # objective -> [visible_wp, ...]
        self.camera_calib_target = {} # camera -> objective
        self.rover_cameras = {} # rover -> [camera, ...]
        self.camera_modes = {} # camera -> [mode, ...]
        self.rover_capabilities = {} # rover -> [capability_string, ...] ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.initial_soil_samples = set() # {wp, ...}
        self.initial_rock_samples = set() # {wp, ...}

        all_waypoints = set()
        all_rovers = set()
        all_cameras = set()
        all_objectives = set()
        all_modes = set()
        all_landers = set()
        all_stores = set()

        # First pass to identify objects and build basic structures
        # Include initial state for samples that are initially present
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at_soil_sample':
                 self.initial_soil_samples.add(parts[1])
             elif parts[0] == 'at_rock_sample':
                 self.initial_rock_samples.add(parts[1])

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            # Collect all objects of each type (useful for graph initialization)
            if pred == 'waypoint': all_waypoints.add(parts[1])
            elif pred == 'rover': all_rovers.add(parts[1])
            elif pred == 'camera': all_cameras.add(parts[1])
            elif pred == 'objective': all_objectives.add(parts[1])
            elif pred == 'mode': all_modes.add(parts[1])
            elif pred == 'lander': all_landers.add(parts[1])
            elif pred == 'store': all_stores.add(parts[1])

            # Extract specific static relationships
            elif pred == 'at_lander':
                self.lander_name = parts[1]
                self.lander_location = parts[2]
            elif pred == 'can_traverse':
                rover, wp1, wp2 = parts[1:]
                if rover not in self.rover_graphs:
                    self.rover_graphs[rover] = {}
                if wp1 not in self.rover_graphs[rover]:
                    self.rover_graphs[rover][wp1] = []
                self.rover_graphs[rover][wp1].append(wp2)
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = []
                self.rover_capabilities[rover].append('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = []
                self.rover_capabilities[rover].append('rock')
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                if rover not in self.rover_capabilities: self.rover_capabilities[rover] = []
                self.rover_capabilities[rover].append('imaging')
            elif pred == 'store_of':
                store, rover = parts[1:]
                self.rover_stores[rover] = store
            elif pred == 'supports':
                camera, mode = parts[1:]
                if camera not in self.camera_modes: self.camera_modes[camera] = []
                self.camera_modes[camera].append(mode)
            elif pred == 'visible_from':
                objective, waypoint = parts[1:]
                if objective not in self.obj_visible_from: self.obj_visible_from[objective] = []
                self.obj_visible_from[objective].append(waypoint)
            elif pred == 'calibration_target':
                camera, objective = parts[1:]
                self.camera_calib_target[camera] = objective
            elif pred == 'on_board':
                camera, rover = parts[1:]
                if rover not in self.rover_cameras: self.rover_cameras[rover] = []
                self.rover_cameras[rover].append(camera)

        # Ensure all waypoints are keys in each rover's graph, even if they have no outgoing edges
        for rover in all_rovers:
             if rover not in self.rover_graphs:
                 self.rover_graphs[rover] = {}
             for wp in all_waypoints:
                 if wp not in self.rover_graphs[rover]:
                     self.rover_graphs[rover][wp] = []


        # Determine communication points (waypoints visible from lander location)
        if self.lander_location and self.lander_name:
            self.comm_points[self.lander_name] = []
            for fact in static_facts:
                # We need waypoints ?x such that (visible ?x lander_loc) for the communicate action
                if match(fact, 'visible', '*', self.lander_location):
                    parts = get_parts(fact)
                    self.comm_points[self.lander_name].append(parts[1])
                # Note: visible is often symmetric, but the precondition is specific.

    def __call__(self, node):
        state = node.state

        # Extract dynamic facts from the current state
        rover_locations = {} # rover -> waypoint
        rover_soil_samples = {} # rover -> [wp, ...]
        rover_rock_samples = {} # rover -> [wp, ...]
        rover_images = {} # rover -> [(obj, mode), ...]
        rover_calibrated_cameras = {} # rover -> [camera, ...]
        rover_full_stores = set() # {store, ...}
        current_soil_samples = set() # {wp, ...} (at_soil_sample ?w)
        current_rock_samples = set() # {wp, ...} (at_rock_sample ?w)


        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == 'at' and parts[1].startswith('rover'): # Assuming objects starting with 'rover' are rovers
                 rover_locations[parts[1]] = parts[2]
            elif pred == 'have_soil_analysis':
                 rover, wp = parts[1:]
                 if rover not in rover_soil_samples: rover_soil_samples[rover] = []
                 rover_soil_samples[rover].append(wp)
            elif pred == 'have_rock_analysis':
                 rover, wp = parts[1:]
                 if rover not in rover_rock_samples: rover_rock_samples[rover] = []
                 rover_rock_samples[rover].append(wp)
            elif pred == 'have_image':
                 rover, obj, mode = parts[1:]
                 if rover not in rover_images: rover_images[rover] = []
                 rover_images[rover].append((obj, mode))
            elif pred == 'calibrated':
                 camera, rover = parts[1:]
                 if rover not in rover_calibrated_cameras: rover_calibrated_cameras[rover] = []
                 rover_calibrated_cameras[rover].append(camera)
            elif pred == 'full':
                 store = parts[1]
                 rover_full_stores.add(store)
            elif pred == 'at_soil_sample':
                 current_soil_samples.add(parts[1])
            elif pred == 'at_rock_sample':
                 current_rock_samples.add(parts[1])


        total_cost = 0
        unachieved_goals = self.goals - state

        # Assign a large penalty for unreachable goals/subgoals
        UNREACHABLE_PENALTY = 1000

        for goal in unachieved_goals:
            parts = get_parts(goal)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                w_goal = parts[1]
                found_analysis = False
                
                # Check if any rover already has the analysis
                for rover, samples in rover_soil_samples.items():
                    if w_goal in samples:
                        found_analysis = True
                        # Cost to communicate this existing sample
                        w_curr = rover_locations.get(rover)
                        if w_curr is None: continue # Should not happen in a valid state

                        min_nav_comm = float('inf')
                        comm_wps = self.comm_points.get(self.lander_name, [])
                        for w_comm in comm_wps:
                            dist = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_curr, w_comm)
                            if dist != float('inf'):
                                min_nav_comm = min(min_nav_comm, dist)

                        if min_nav_comm != float('inf'):
                            total_cost += min_nav_comm + 1 # Communicate action
                        else:
                            total_cost += UNREACHABLE_PENALTY # Penalty for unreachable comm point
                        break # Assume one rover having the sample is enough for this goal

                if not found_analysis:
                    # Need to sample and communicate

                    # Check if sample is still available at the waypoint
                    if w_goal not in current_soil_samples and w_goal in self.initial_soil_samples:
                         # Sample was at w_goal initially, but is gone now and not collected.
                         # This goal might be unreachable or require complex recovery.
                         total_cost += UNREACHABLE_PENALTY
                         continue # Move to next goal

                    min_cost_for_goal = float('inf')
                    # Find the best capable rover
                    for rover in rover_locations: # Iterate through all rovers
                        if 'soil' in self.rover_capabilities.get(rover, []):
                            w_curr = rover_locations[rover]
                            rover_store = self.rover_stores.get(rover)

                            # Cost to sample
                            nav_to_sample = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_curr, w_goal)
                            if nav_to_sample == float('inf'):
                                continue # Cannot reach sample location

                            sample_cost = nav_to_sample + 1 # Sample action
                            drop_cost = 1 if rover_store in rover_full_stores else 0
                            sample_cost += drop_cost

                            # Cost to communicate after sampling (rover is at w_goal)
                            min_nav_comm = float('inf')
                            comm_wps = self.comm_points.get(self.lander_name, [])
                            for w_comm in comm_wps:
                                dist = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_goal, w_comm)
                                if dist != float('inf'):
                                    min_nav_comm = min(min_nav_comm, dist)

                            if min_nav_comm != float('inf'):
                                comm_cost = min_nav_comm + 1 # Communicate action
                                min_cost_for_goal = min(min_cost_for_goal, sample_cost + comm_cost)

                    if min_cost_for_goal != float('inf'):
                        total_cost += min_cost_for_goal
                    else:
                        total_cost += UNREACHABLE_PENALTY # Penalty if no capable rover can sample/communicate

            elif pred == 'communicated_rock_data':
                w_goal = parts[1]
                found_analysis = False
                
                # Check if any rover already has the analysis
                for rover, samples in rover_rock_samples.items():
                    if w_goal in samples:
                        found_analysis = True
                        # Cost to communicate this existing sample
                        w_curr = rover_locations.get(rover)
                        if w_curr is None: continue

                        min_nav_comm = float('inf')
                        comm_wps = self.comm_points.get(self.lander_name, [])
                        for w_comm in comm_wps:
                            dist = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_curr, w_comm)
                            if dist != float('inf'):
                                min_nav_comm = min(min_nav_comm, dist)

                        if min_nav_comm != float('inf'):
                            total_cost += min_nav_comm + 1 # Communicate action
                        else:
                            total_cost += UNREACHABLE_PENALTY # Penalty
                        break

                if not found_analysis:
                     # Check if sample is still available at the waypoint
                    if w_goal not in current_rock_samples and w_goal in self.initial_rock_samples:
                         total_cost += UNREACHABLE_PENALTY
                         continue

                    min_cost_for_goal = float('inf')
                    # Find the best capable rover
                    for rover in rover_locations:
                        if 'rock' in self.rover_capabilities.get(rover, []):
                            w_curr = rover_locations[rover]
                            rover_store = self.rover_stores.get(rover)

                            # Cost to sample
                            nav_to_sample = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_curr, w_goal)
                            if nav_to_sample == float('inf'):
                                continue

                            sample_cost = nav_to_sample + 1 # Sample action
                            drop_cost = 1 if rover_store in rover_full_stores else 0
                            sample_cost += drop_cost

                            # Cost to communicate after sampling (rover is at w_goal)
                            min_nav_comm = float('inf')
                            comm_wps = self.comm_points.get(self.lander_name, [])
                            for w_comm in comm_wps:
                                dist = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_goal, w_comm)
                                if dist != float('inf'):
                                    min_nav_comm = min(min_nav_comm, dist)

                            if min_nav_comm != float('inf'):
                                comm_cost = min_nav_comm + 1 # Communicate action
                                min_cost_for_goal = min(min_cost_for_goal, sample_cost + comm_cost)

                    if min_cost_for_goal != float('inf'):
                        total_cost += min_cost_for_goal
                    else:
                        total_cost += UNREACHABLE_PENALTY # Penalty

            elif pred == 'communicated_image_data':
                o_goal, m_goal = parts[1:]
                found_image = False
                
                # Check if any rover already has the image
                for rover, images in rover_images.items():
                    if (o_goal, m_goal) in images:
                        found_image = True
                        # Cost to communicate this existing image
                        w_curr = rover_locations.get(rover)
                        if w_curr is None: continue

                        min_nav_comm = float('inf')
                        comm_wps = self.comm_points.get(self.lander_name, [])
                        for w_comm in comm_wps:
                            dist = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_curr, w_comm)
                            if dist != float('inf'):
                                min_nav_comm = min(min_nav_comm, dist)

                        if min_nav_comm != float('inf'):
                            total_cost += min_nav_comm + 1 # Communicate action
                        else:
                            total_cost += UNREACHABLE_PENALTY # Penalty
                        break

                if not found_image:
                    # Need to take image and communicate
                    min_cost_for_goal = float('inf')
                    # Find the best capable rover/camera
                    for rover in rover_locations:
                        if 'imaging' in self.rover_capabilities.get(rover, []):
                            for camera in self.rover_cameras.get(rover, []):
                                if m_goal in self.camera_modes.get(camera, []):
                                    # Found a capable rover/camera for this image goal
                                    w_curr = rover_locations[rover]

                                    # Find best waypoint to take the image from
                                    image_wps = self.obj_visible_from.get(o_goal, [])
                                    min_nav_curr_to_image = float('inf')
                                    best_w_image = None
                                    for w_image in image_wps:
                                        dist = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_curr, w_image)
                                        if dist != float('inf') and dist < min_nav_curr_to_image:
                                            min_nav_curr_to_image = dist
                                            best_w_image = w_image

                                    if best_w_image is None:
                                        continue # Cannot reach any waypoint to image from

                                    # Cost to get the image (includes nav and actions)
                                    cost_to_get_image = float('inf')

                                    # Cost for calibration if needed
                                    calib_sequence_cost = 0 # Cost of Nav(curr->calib) + Calibrate + Nav(calib->image)
                                    if camera not in rover_calibrated_cameras.get(rover, []):
                                        t_calib = self.camera_calib_target.get(camera)
                                        if t_calib is None:
                                            calib_sequence_cost = UNREACHABLE_PENALTY # Camera has no calib target
                                        else:
                                            calib_wps = self.obj_visible_from.get(t_calib, [])
                                            min_nav_curr_to_calib = float('inf')
                                            best_w_calib = None
                                            for w_calib in calib_wps:
                                                # Nav from current location (w_curr) to calib point
                                                dist = get_shortest_path_distance(self.rover_graphs.get(rover, {}), w_curr, w_calib)
                                                if dist != float('inf') and dist < min_nav_curr_to_calib:
                                                    min_nav_curr_to_calib = dist
                                                    best_w_calib = w_calib

                                            if best_w_calib is None:
                                                calib_sequence_cost = UNREACHABLE_PENALTY # Cannot reach any calib point
                                            else:
                                                # Nav from calib point to image point
                                                nav_calib_to_image = get_shortest_path_distance(self.rover_graphs.get(rover, {}), best_w_calib, best_w_image)
                                                if nav_calib_to_image == float('inf'):
                                                    calib_sequence_cost = UNREACHABLE_PENALTY # Cannot navigate from calib to image
                                                else:
                                                    # Total cost including calibration: Nav(curr->calib) + Calibrate + Nav(calib->image)
                                                    calib_sequence_cost = min_nav_curr_to_calib + 1 + nav_calib_to_image # Calibrate action + nav

                                    if calib_sequence_cost == UNREACHABLE_PENALTY:
                                         continue # Cannot calibrate

                                    # Total cost to get the image: (Calib sequence if needed) + TakeImage
                                    if camera not in rover_calibrated_cameras.get(rover, []):
                                         # Cost is Nav(curr->calib) + Calib + Nav(calib->image) + TakeImage
                                         cost_to_get_image = calib_sequence_cost + 1 # Add TakeImage action
                                    else:
                                         # Cost is Nav(curr->image) + TakeImage
                                         cost_to_get_image = min_nav_curr_to_image + 1 # Add TakeImage action

                                    if cost_to_get_image == float('inf'):
                                        continue # Should not happen if calib_sequence_cost was not penalty, but double check

                                    # Cost to communicate after imaging (rover is at best_w_image)
                                    min_nav_comm = float('inf')
                                    comm_wps = self.comm_points.get(self.lander_name, [])
                                    for w_comm in comm_wps:
                                        dist = get_shortest_path_distance(self.rover_graphs.get(rover, {}), best_w_image, w_comm)
                                        if dist != float('inf'):
                                            min_nav_comm = min(min_nav_comm, dist)

                                    if min_nav_comm != float('inf'):
                                        comm_cost = min_nav_comm + 1 # Communicate action
                                        min_cost_for_goal = min(min_cost_for_goal, cost_to_get_image + comm_cost)

                    if min_cost_for_goal != float('inf'):
                        total_cost += min_cost_for_goal
                    else:
                        total_cost += UNREACHABLE_PENALTY # Penalty if no capable rover/camera can image/communicate

        return total_cost
