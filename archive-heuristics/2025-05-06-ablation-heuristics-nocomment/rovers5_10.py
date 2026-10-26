# from heuristics.heuristic_base import Heuristic # Assuming this is available

from collections import deque
import math # Use math.inf for infinity

# Helper function to parse PDDL fact strings
def get_parts(fact_string):
    """Removes parentheses and splits a PDDL fact string into parts."""
    return fact_string[1:-1].split()

# Helper function for Breadth-First Search
def bfs(start_node, graph):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: math.inf for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Ensure current_node is a key in the graph before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

# Main heuristic class
# Inherit from Heuristic base class provided by the planner environment
# class roversHeuristic(Heuristic): # Uncomment this line in the actual environment
class roversHeuristic: # Using this for standalone code generation
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the remaining cost to reach the goal by summing
    the minimum estimated costs for each individual unachieved goal fact.
    For each goal fact (communicating soil data, rock data, or image data),
    it calculates the minimum number of actions required by any available
    and capable rover (and camera, for image goals) to achieve that specific
    goal fact, starting from the rover's current location.
    The cost for each goal fact is estimated greedily, considering the steps
    needed (navigate, sample/image, communicate) and using precomputed
    shortest path distances for navigation.

    Assumptions:
    - All actions (navigate, sample, drop, calibrate, take_image, communicate)
      have a cost of 1.
    - The heuristic sums the costs for each unachieved goal fact independently,
      ignoring potential positive interactions (e.g., one navigation action
      serving multiple goals) or negative interactions (e.g., resource
      contention like a single store or consumed calibration). This means
      the heuristic is not admissible but aims to be informative for greedy
      search.
    - Samples at waypoints are removed permanently once collected.
    - Calibration is consumed by the 'take_image' action.
    - Each rover has at least one store.
    - Waypoints visible from the lander's location are communication points.
      A waypoint is considered visible from itself for communication purposes.
    - The 'can_traverse' predicate, combined with 'visible', defines the
      traversable graph for each specific rover.

    Heuristic Initialization:
    During initialization, the heuristic parses the static facts from the
    planning task to build data structures representing:
    - The set of all objects of each type (rovers, waypoints, cameras, etc.)
      present in the static facts and goals.
    - The lander's location.
    - Which equipment each rover has (soil, rock, imaging).
    - Which store belongs to which rover.
    - Details about each camera (owner rover, supported modes, calibration target).
    - The general waypoint visibility graph.
    - The specific traversal graph for each rover (combining 'visible' and
      'can_traverse').
    - Waypoints visible from each objective and calibration target.
    - Waypoints visible from the lander's location (communication points).
    - Shortest path distances between all pairs of waypoints for each rover,
      computed using Breadth-First Search on the rover's traversal graph.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the current state is a goal state by verifying if all goal
        facts are present in the state. If yes, the heuristic value is 0.
    2.  Parse the current state to extract dynamic information: the current
        location of each rover, the locations of remaining soil/rock samples,
        which samples/images each rover has collected, the status (empty/full)
        of each store, which cameras are calibrated, and which data has
        already been communicated.
    3.  Initialize a total heuristic value `total_h` to 0.
    4.  Iterate through each goal fact specified in the task's goals.
    5.  For each goal fact:
        a.  If the goal fact is already true in the current state, it requires
            0 further actions for this specific goal, so continue to the next
            goal fact.
        b.  If the goal fact is `(communicated_soil_data ?w)`:
            -   Find all rovers that are equipped for soil analysis.
            -   Calculate the minimum estimated cost to achieve this goal fact
                among all equipped rovers. Initialize `min_cost_for_goal` to infinity.
            -   For each equipped rover `r`:
                -   Get the rover's current location.
                -   Estimate the cost to obtain the soil sample from waypoint `?w`:
                    -   If rover `r` already has `(have_soil_analysis ?r ?w)`, the cost to get the sample is 0. The rover's location for the next step is its current location.
                    -   If rover `r` does *not* have the sample:
                        -   Check if `(at_soil_sample ?w)` is true in the state. If not, the sample is gone, and this goal is impossible for this rover; set cost to infinity.
                        -   If the sample is at `?w`:
                            -   Cost includes navigating from the rover's current location to `?w` (using precomputed distances). If unreachable, cost is infinity.
                            -   If all of the rover's stores are full, add 1 for a `drop` action.
                            -   Add 1 for the `sample_soil` action.
                            -   The rover's location after sampling is `?w`.
                -   If the cost to get the sample is not infinity, estimate the cost to communicate the data:
                    -   Find the minimum navigation cost from the rover's location (after sampling, or current if sample was held) to any waypoint visible from the lander (using precomputed distances). If no such waypoint is reachable, cost is infinity.
                    -   Add 1 for the `communicate_soil_data` action.
                -   The total cost for this rover is the sum of the cost to get the sample and the cost to communicate. Update `min_cost_for_goal` with the minimum cost found so far across all rovers.
            -   If `min_cost_for_goal` is still infinity after checking all rovers, the goal is unachievable from this state; return infinity immediately.
            -   Otherwise, add `min_cost_for_goal` to `total_h`.
        c.  If the goal fact is `(communicated_rock_data ?w)`: Follow the same logic as for soil data, substituting rock-specific predicates and actions (`equipped_for_rock_analysis`, `at_rock_sample`, `have_rock_analysis`, `sample_rock`, `communicate_rock_data`).
        d.  If the goal fact is `(communicated_image_data ?o ?m)`:
            -   Find all rovers equipped for imaging that have a camera on board supporting mode `?m`.
            -   Calculate the minimum estimated cost to achieve this goal fact among all suitable rover/camera pairs. Initialize `min_cost_for_goal` to infinity.
            -   For each suitable rover `r` and camera `i`:
                -   Get the rover's current location.
                -   Estimate the cost to obtain the image of objective `?o` in mode `?m`:
                    -   If rover `r` already has `(have_image ?r ?o ?m)`, the cost to get the image is 0. The rover's location for the next step is its current location.
                    -   If rover `r` does *not* have the image:
                        -   Find the calibration target `t` for camera `i`.
                        -   Find the set of waypoints `W_cal` visible from `t` and `W_img` visible from `?o`. If either set is empty, this path is impossible; set cost to infinity.
                        -   Otherwise, find the minimum cost path involving: navigate from the rover's current location to some `w_cal` in `W_cal`, perform `calibrate` (1 action), navigate from `w_cal` to some `w_img` in `W_img`, perform `take_image` (1 action). Use precomputed distances for navigation. The minimum is taken over all pairs `(w_cal, w_img)`. If no such path is reachable, cost is infinity.
                        -   The rover's location after taking the image is the chosen `w_img`.
                -   If the cost to get the image is not infinity, estimate the cost to communicate the data:
                    -   Find the minimum navigation cost from the rover's location (after taking the image, or current if image was held) to any waypoint visible from the lander (using precomputed distances). If no such waypoint is reachable, cost is infinity.
                    -   Add 1 for the `communicate_image_data` action.
                -   The total cost for this rover/camera pair is the sum of the cost to get the image and the cost to communicate. Update `min_cost_for_goal` with the minimum cost found so far across all pairs.
            -   If `min_cost_for_goal` is still infinity after checking all pairs, the goal is unachievable from this state; return infinity immediately.
            -   Otherwise, add `min_cost_for_goal` to `total_h`.
    6.  Return the final calculated `total_h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static facts and precomputing
        necessary information like object types, equipment, camera details,
        waypoint connectivity, and shortest path distances for each rover.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # --- Precompute Static Information ---

        # 1. Collect all potential objects by type from static facts and goals
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        # Mapping predicate arguments to types (based on domain file)
        arg_types = {
            'at': {1: 'rover', 2: 'waypoint'},
            'at_lander': {1: 'lander', 2: 'waypoint'},
            'can_traverse': {1: 'rover', 2: 'waypoint', 3: 'waypoint'},
            'equipped_for_soil_analysis': {1: 'rover'},
            'equipped_for_rock_analysis': {1: 'rover'},
            'equipped_for_imaging': {1: 'rover'},
            'empty': {1: 'store'},
            'have_rock_analysis': {1: 'rover', 2: 'waypoint'},
            'have_soil_analysis': {1: 'rover', 2: 'waypoint'},
            'full': {1: 'store'},
            'calibrated': {1: 'camera', 2: 'rover'},
            'supports': {1: 'camera', 2: 'mode'},
            'visible': {1: 'waypoint', 2: 'waypoint'},
            'have_image': {1: 'rover', 2: 'objective', 3: 'mode'},
            'communicated_soil_data': {1: 'waypoint'},
            'communicated_rock_data': {1: 'waypoint'},
            'communicated_image_data': {1: 'objective', 2: 'mode'},
            'at_soil_sample': {1: 'waypoint'},
            'at_rock_sample': {1: 'waypoint'},
            'visible_from': {1: 'objective', 2: 'waypoint'},
            'store_of': {1: 'store', 2: 'rover'},
            'calibration_target': {1: 'camera', 2: 'objective'},
            'on_board': {1: 'camera', 2: 'rover'},
        }

        all_facts = set(self.static_facts) | set(self.goals)
        for fact in all_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate in arg_types:
                for i, obj_type in arg_types[predicate].items():
                    if i < len(parts):
                        obj_name = parts[i]
                        if obj_type == 'rover': self.rovers.add(obj_name)
                        elif obj_type == 'waypoint': self.waypoints.add(obj_name)
                        elif obj_type == 'store': self.stores.add(obj_name)
                        elif obj_type == 'camera': self.cameras.add(obj_name)
                        elif obj_type == 'mode': self.modes.add(obj_name)
                        elif obj_type == 'lander': self.landers.add(obj_name)
                        elif obj_type == 'objective': self.objectives.add(obj_name)

        # 2. Parse specific static predicates
        self.lander_location = None
        self.rover_equipment = {r: set() for r in self.rovers} # rover -> set(equipment_type_str)
        self.store_owner = {s: None for s in self.stores} # store -> rover
        self.camera_info = {c: {'rover': None, 'modes': set(), 'cal_target': None} for c in self.cameras} # camera -> { 'rover': rover, 'modes': set(mode), 'cal_target': objective }
        self.waypoint_graph = {wp: set() for wp in self.waypoints} # wp -> set(visible_wp)
        self.rover_traversal_graph = {r: {wp: set() for wp in self.waypoints} for r in self.rovers} # rover -> { wp -> set(traversable_wp) }
        self.visible_from_objective = {obj: set() for obj in self.objectives} # objective -> set(wp)
        self.visible_from_caltarget = {obj: set() for obj in self.objectives} # objective -> set(wp) # Note: caltarget is an objective

        for fact in self.static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at_lander': self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis': self.rover_equipment[parts[1]].add('soil')
            elif predicate == 'equipped_for_rock_analysis': self.rover_equipment[parts[1]].add('rock')
            elif predicate == 'equipped_for_imaging': self.rover_equipment[parts[1]].add('imaging')
            elif predicate == 'store_of': self.store_owner[parts[1]] = parts[2]
            elif predicate == 'supports': self.camera_info[parts[1]]['modes'].add(parts[2])
            elif predicate == 'visible': self.waypoint_graph[parts[1]].add(parts[2])
            elif predicate == 'visible_from': self.visible_from_objective[parts[1]].add(parts[2])
            elif predicate == 'calibration_target':
                self.camera_info[parts[1]]['cal_target'] = parts[2]
                # Calibration targets are objectives, so they appear in visible_from_objective
                # We'll populate visible_from_caltarget based on visible_from_objective later
            elif predicate == 'on_board': self.camera_info[parts[1]]['rover'] = parts[2]
            elif predicate == 'can_traverse':
                 if parts[1] in self.rover_traversal_graph and parts[2] in self.rover_traversal_graph[parts[1]]:
                    self.rover_traversal_graph[parts[1]][parts[2]].add(parts[3])


        # Populate visible_from_caltarget based on calibration targets and visible_from_objective
        for cam_info in self.camera_info.values():
            cal_target = cam_info.get('cal_target')
            if cal_target and cal_target in self.visible_from_objective:
                 self.visible_from_caltarget[cal_target].update(self.visible_from_objective[cal_target])


        # 3. Precompute shortest path distances for each rover
        self.waypoint_dist = {} # rover -> { start_wp -> { end_wp -> dist } }
        for rover in self.rovers:
            self.waypoint_dist[rover] = {}
            # Build the actual traversal graph for this rover considering 'can_traverse' and 'visible'
            # A rover can traverse from wp1 to wp2 if visible wp1 wp2 AND can_traverse rover wp1 wp2
            rover_graph = {wp: set() for wp in self.waypoints}
            for wp1 in self.waypoints:
                 if wp1 in self.waypoint_graph: # Check if wp1 is a source in the visible graph
                    for wp2 in self.waypoint_graph[wp1]:
                        # Check if rover can traverse from wp1 to wp2
                        if wp1 in self.rover_traversal_graph.get(rover, {}) and wp2 in self.rover_traversal_graph[rover][wp1]:
                             rover_graph[wp1].add(wp2)

            # Add self-loops with distance 0 for BFS to work correctly from any node
            # This is implicitly handled by BFS starting point having distance 0,
            # but explicitly adding it doesn't hurt and ensures nodes with no
            # outgoing edges are still in the graph keys.
            for wp in self.waypoints:
                 if wp not in rover_graph:
                     rover_graph[wp] = set()
                 rover_graph[wp].add(wp) # Distance to self is 0

            for start_wp in self.waypoints:
                self.waypoint_dist[rover][start_wp] = bfs(start_wp, rover_graph)

        # 4. Identify waypoints visible from the lander
        self.lander_visible_waypoints = set()
        if self.lander_location:
             # Waypoints visible *from* the lander's location
             if self.lander_location in self.waypoint_graph:
                 self.lander_visible_waypoints.update(self.waypoint_graph[self.lander_location])
             # Also, the lander location itself is a communication point
             self.lander_visible_waypoints.add(self.lander_location)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # 1. Check if goal state
        if self.goals <= state:
            return 0

        # 2. Parse current state
        rover_location = {} # rover -> wp
        soil_samples_at = set() # wp
        rock_samples_at = set() # wp
        rover_soil_samples = {r: set() for r in self.rovers} # rover -> set(wp)
        rover_rock_samples = {r: set() for r in self.rovers} # rover -> set(wp)
        rover_images = {r: set() for r in self.rovers} # rover -> set((obj, mode))
        store_status = {s: 'unknown' for s in self.stores} # store -> 'empty' or 'full'
        camera_calibrated = set() # set((camera, rover))
        # communicated_soil = set() # wp # Not needed for heuristic calculation, only check if goal is met
        # communicated_rock = set() # wp # Not needed
        # communicated_image = set() # (obj, mode) # Not needed

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                 if parts[1] in self.rovers: # Ensure rover exists
                    rover_location[parts[1]] = parts[2]
            elif predicate == 'at_soil_sample':
                 if parts[1] in self.waypoints: # Ensure waypoint exists
                    soil_samples_at.add(parts[1])
            elif predicate == 'at_rock_sample':
                 if parts[1] in self.waypoints: # Ensure waypoint exists
                    rock_samples_at.add(parts[1])
            elif predicate == 'have_soil_analysis':
                 if parts[1] in self.rovers and parts[2] in self.waypoints: # Ensure rover and waypoint exist
                    rover_soil_samples[parts[1]].add(parts[2])
            elif predicate == 'have_rock_analysis':
                 if parts[1] in self.rovers and parts[2] in self.waypoints: # Ensure rover and waypoint exist
                    rover_rock_samples[parts[1]].add(parts[2])
            elif predicate == 'have_image':
                 if parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes: # Ensure objects exist
                    rover_images[parts[1]].add((parts[2], parts[3]))
            elif predicate == 'empty':
                 if parts[1] in self.stores: # Ensure store exists
                    store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                 if parts[1] in self.stores: # Ensure store exists
                    store_status[parts[1]] = 'full'
            elif predicate == 'calibrated':
                 if parts[1] in self.cameras and parts[2] in self.rovers: # Ensure camera and rover exist
                    camera_calibrated.add((parts[1], parts[2]))
            # elif predicate == 'communicated_soil_data': communicated_soil.add(parts[1]) # Not needed
            # elif predicate == 'communicated_rock_data': communicated_rock.add(parts[1]) # Not needed
            # elif predicate == 'communicated_image_data': communicated_image.add((parts[1], parts[2])) # Not needed


        # 3. Initialize total heuristic
        total_h = 0

        # 4. Process each goal fact
        for goal in self.goals:
            # 5. If goal fact is already achieved, continue
            if goal in state:
                continue

            parts = get_parts(goal)
            predicate = parts[0]

            # b. Goal is (communicated_soil_data ?w)
            if predicate == 'communicated_soil_data':
                wp = parts[1]
                min_cost_for_goal = math.inf

                for rover in self.rovers:
                    # Check if rover is equipped
                    if 'soil' not in self.rover_equipment.get(rover, set()):
                        continue

                    current_loc_r = rover_location.get(rover)
                    if current_loc_r is None: continue # Rover location must be known

                    cost_to_get_sample = 0
                    location_after_sample = current_loc_r
                    has_sample = wp in rover_soil_samples.get(rover, set())

                    if not has_sample:
                        # Need to sample
                        if wp not in soil_samples_at:
                            cost_to_get_sample = math.inf # Sample is gone
                        else:
                            # Navigate to sample location
                            dist_to_sample = self.waypoint_dist.get(rover, {}).get(current_loc_r, {}).get(wp, math.inf)
                            if dist_to_sample == math.inf:
                                cost_to_get_sample = math.inf # Cannot reach sample waypoint
                            else:
                                cost_to_get_sample += dist_to_sample
                                # Check store status - need to find the stores for this rover
                                rover_stores = [s for s, owner in self.store_owner.items() if owner == rover]
                                # Check if *all* stores for this rover are full
                                all_stores_full = rover_stores and all(store_status.get(s) == 'full' for s in rover_stores)
                                if all_stores_full:
                                     cost_to_get_sample += 1 # Add cost for drop action (from one of the full stores)
                                cost_to_get_sample += 1 # Add cost for sample_soil action
                                location_after_sample = wp # Rover is at wp after sampling

                    if cost_to_get_sample != math.inf:
                        # Need to communicate
                        min_dist_to_comm = math.inf
                        if not self.lander_visible_waypoints:
                             cost_to_get_sample = math.inf # Cannot communicate if no lander visible points
                        else:
                            for w_comm in self.lander_visible_waypoints:
                                dist_to_comm = self.waypoint_dist.get(rover, {}).get(location_after_sample, {}).get(w_comm, math.inf)
                                min_dist_to_comm = min(min_dist_to_comm, dist_to_comm)

                            if min_dist_to_comm == math.inf:
                                cost_to_get_sample = math.inf # Cannot reach communication point
                            else:
                                cost_to_communicate = min_dist_to_comm + 1 # communicate_soil_data
                                min_cost_for_rover = cost_to_get_sample + cost_to_communicate
                                min_cost_for_goal = min(min_cost_for_goal, min_cost_for_rover)

                if min_cost_for_goal == math.inf:
                    return math.inf # Unachievable goal
                else:
                    total_h += min_cost_for_goal

            # c. Goal is (communicated_rock_data ?w)
            elif predicate == 'communicated_rock_data':
                wp = parts[1]
                min_cost_for_goal = math.inf

                for rover in self.rovers:
                    # Check if rover is equipped
                    if 'rock' not in self.rover_equipment.get(rover, set()):
                        continue

                    current_loc_r = rover_location.get(rover)
                    if current_loc_r is None: continue # Should not happen

                    cost_to_get_sample = 0
                    location_after_sample = current_loc_r
                    has_sample = wp in rover_rock_samples.get(rover, set())

                    if not has_sample:
                        # Need to sample
                        if wp not in rock_samples_at:
                            cost_to_get_sample = math.inf # Sample is gone
                        else:
                            # Navigate to sample location
                            dist_to_sample = self.waypoint_dist.get(rover, {}).get(current_loc_r, {}).get(wp, math.inf)
                            if dist_to_sample == math.inf:
                                cost_to_get_sample = math.inf # Cannot reach sample waypoint
                            else:
                                cost_to_get_sample += dist_to_sample
                                # Check store status
                                rover_stores = [s for s, owner in self.store_owner.items() if owner == rover]
                                all_stores_full = rover_stores and all(store_status.get(s) == 'full' for s in rover_stores)
                                if all_stores_full:
                                     cost_to_get_sample += 1 # Add cost for drop action
                                cost_to_get_sample += 1 # Add cost for sample_rock action
                                location_after_sample = wp # Rover is at wp after sampling

                    if cost_to_get_sample != math.inf:
                        # Need to communicate
                        min_dist_to_comm = math.inf
                        if not self.lander_visible_waypoints:
                             cost_to_get_sample = math.inf # Cannot communicate
                        else:
                            for w_comm in self.lander_visible_waypoints:
                                dist_to_comm = self.waypoint_dist.get(rover, {}).get(location_after_sample, {}).get(w_comm, math.inf)
                                min_dist_to_comm = min(min_dist_to_comm, dist_to_comm)

                            if min_dist_to_comm == math.inf:
                                cost_to_get_sample = math.inf # Cannot reach communication point
                            else:
                                cost_to_communicate = min_dist_to_comm + 1 # communicate_rock_data
                                min_cost_for_rover = cost_to_get_sample + cost_to_communicate
                                min_cost_for_goal = min(min_cost_for_goal, min_cost_for_rover)

                if min_cost_for_goal == math.inf:
                    return math.inf # Unachievable goal
                else:
                    total_h += min_cost_for_goal

            # d. Goal is (communicated_image_data ?o ?m)
            elif predicate == 'communicated_image_data':
                obj = parts[1]
                mode = parts[2]
                min_cost_for_goal = math.inf

                for rover in self.rovers:
                    # Check if rover is equipped for imaging
                    if 'imaging' not in self.rover_equipment.get(rover, set()):
                        continue

                    current_loc_r = rover_location.get(rover)
                    if current_loc_r is None: continue # Should not happen

                    for camera in self.cameras:
                        cam_info = self.camera_info.get(camera)
                        # Check if camera is on this rover and supports the mode
                        if not cam_info or cam_info.get('rover') != rover or mode not in cam_info.get('modes', set()):
                            continue

                        cost_to_get_image = 0
                        location_after_image = current_loc_r
                        has_image = (obj, mode) in rover_images.get(rover, set())

                        if not has_image:
                            # Need to take image
                            cal_target = cam_info.get('cal_target')
                            W_cal = self.visible_from_caltarget.get(cal_target, set()) if cal_target else set()
                            W_img = self.visible_from_objective.get(obj, set())

                            if not W_cal or not W_img:
                                cost_to_get_image = math.inf # Cannot calibrate or take image

                            else:
                                min_image_path_cost = math.inf
                                best_w_img = None

                                for w_cal in W_cal:
                                    for w_img in W_img:
                                        # Path: current -> w_cal (calibrate) -> w_img (take_image)
                                        dist1 = self.waypoint_dist.get(rover, {}).get(current_loc_r, {}).get(w_cal, math.inf)
                                        dist2 = self.waypoint_dist.get(rover, {}).get(w_cal, {}).get(w_img, math.inf)

                                        if dist1 != math.inf and dist2 != math.inf:
                                            # Cost: navigate to cal + calibrate + navigate to img + take_image
                                            image_path_cost = dist1 + 1 + dist2 + 1
                                            if image_path_cost < min_image_path_cost:
                                                min_image_path_cost = image_path_cost
                                                best_w_img = w_img

                                if min_image_path_cost == math.inf:
                                    cost_to_get_image = math.inf # Cannot reach cal/image points
                                else:
                                    cost_to_get_image += min_image_path_cost
                                    location_after_image = best_w_img # Rover is at w_img after taking image

                        if cost_to_get_image != math.inf:
                            # Need to communicate
                            min_dist_to_comm = math.inf
                            if not self.lander_visible_waypoints:
                                 cost_to_get_image = math.inf # Cannot communicate
                            else:
                                for w_comm in self.lander_visible_waypoints:
                                    dist_to_comm = self.waypoint_dist.get(rover, {}).get(location_after_image, {}).get(w_comm, math.inf)
                                    min_dist_to_comm = min(min_dist_to_comm, dist_to_comm)

                                if min_dist_to_comm == math.inf:
                                    cost_to_get_image = math.inf # Cannot reach communication point
                                else:
                                    cost_to_communicate = min_dist_to_comm + 1 # communicate_image_data
                                    min_cost_for_rover_cam = cost_to_get_image + cost_to_communicate
                                    min_cost_for_goal = min(min_cost_for_goal, min_cost_for_rover_cam)

                if min_cost_for_goal == math.inf:
                    return math.inf # Unachievable goal
                else:
                    total_h += min_cost_for_goal

        # 6. Return total heuristic
        return total_h

# Note: In the actual planner environment, the class should inherit from Heuristic:
# from heuristics.heuristic_base import Heuristic
# class roversHeuristic(Heuristic):
#    ... (rest of the code)
