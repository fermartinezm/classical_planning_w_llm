# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

# Import necessary modules
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque
import sys # To use sys.maxsize for infinity

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the minimum estimated
    costs for each unachieved goal literal. The cost for each goal literal
    is estimated by finding the minimum actions required by any capable rover,
    including navigation, sampling/imaging, and communication steps.
    Navigation costs are precomputed using BFS.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing navigation distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Static Information ---
        self.lander_location = {} # lander -> waypoint (Assuming one lander)
        self.rover_capabilities = defaultdict(set) # rover -> {capability, ...}
        self.store_of_rover = {} # store -> rover
        self.rover_stores = defaultdict(list) # rover -> [store1, ...] (Assuming one store per rover for simplicity)
        self.soil_samples_at = set() # {waypoint, ...}
        self.rock_samples_at = set() # {waypoint, ...}
        self.obj_visible_from_wp = defaultdict(set) # objective -> {waypoint, ...}
        self.cam_on_board_rover = {} # camera -> rover
        self.cam_supports_mode = defaultdict(set) # camera -> {mode, ...}
        self.cam_calib_target = {} # camera -> objective (target)
        self.calib_target_visible_from_wp = defaultdict(set) # target_objective -> {waypoint, ...}
        self.comm_wps_visible_from_lander_wp = set() # {waypoint, ...}
        self.rover_navigation_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> {neighbor_waypoint, ...}

        # Collect all objects of relevant types
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_objectives = set()
        self.all_cameras = set()
        self.all_modes = set()
        self.all_stores = set()
        self.all_landers = set()

        lander_wp = None # To store the lander's waypoint

        # First pass to collect all objects and basic static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            if parts[0] == 'at_lander':
                if len(parts) > 2:
                    self.lander_location[parts[1]] = parts[2]
                    lander_wp = parts[2]
                    self.all_landers.add(parts[1])
                    self.all_waypoints.add(parts[2])
            elif parts[0] == 'equipped_for_soil_analysis':
                if len(parts) > 1:
                    self.rover_capabilities[parts[1]].add('soil')
                    self.all_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                if len(parts) > 1:
                    self.rover_capabilities[parts[1]].add('rock')
                    self.all_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                if len(parts) > 1:
                    self.rover_capabilities[parts[1]].add('imaging')
                    self.all_rovers.add(parts[1])
            elif parts[0] == 'store_of':
                if len(parts) > 2:
                    self.store_of_rover[parts[1]] = parts[2]
                    self.rover_stores[parts[2]].append(parts[1])
                    self.all_stores.add(parts[1])
                    self.all_rovers.add(parts[2])
            elif parts[0] == 'at_soil_sample':
                if len(parts) > 1:
                    self.soil_samples_at.add(parts[1])
                    self.all_waypoints.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                if len(parts) > 1:
                    self.rock_samples_at.add(parts[1])
                    self.all_waypoints.add(parts[1])
            elif parts[0] == 'visible_from':
                if len(parts) > 2:
                    self.obj_visible_from_wp[parts[1]].add(parts[2])
                    self.all_objectives.add(parts[1])
                    self.all_waypoints.add(parts[2])
            elif parts[0] == 'on_board':
                if len(parts) > 2:
                    self.cam_on_board_rover[parts[1]] = parts[2]
                    self.all_cameras.add(parts[1])
                    self.all_rovers.add(parts[2])
            elif parts[0] == 'supports':
                if len(parts) > 2:
                    self.cam_supports_mode[parts[1]].add(parts[2])
                    self.all_cameras.add(parts[1])
                    self.all_modes.add(parts[2])
            elif parts[0] == 'calibration_target':
                if len(parts) > 2:
                    self.cam_calib_target[parts[1]] = parts[2]
                    self.all_cameras.add(parts[1])
                    self.all_objectives.add(parts[2])
            elif parts[0] == 'can_traverse':
                 if len(parts) > 3:
                     self.all_rovers.add(parts[1])
                     self.all_waypoints.add(parts[2])
                     self.all_waypoints.add(parts[3])
            elif parts[0] == 'visible':
                 if len(parts) > 2:
                     self.all_waypoints.add(parts[1])
                     self.all_waypoints.add(parts[2])
            # Ensure all objects mentioned in init are collected even if not in specific predicates above
            elif parts[0] in ('rover', 'waypoint', 'objective', 'camera', 'mode', 'store', 'lander'):
                 if len(parts) > 1:
                     if parts[0] == 'rover': self.all_rovers.add(parts[1])
                     elif parts[0] == 'waypoint': self.all_waypoints.add(parts[1])
                     elif parts[0] == 'objective': self.all_objectives.add(parts[1])
                     elif parts[0] == 'camera': self.all_cameras.add(parts[1])
                     elif parts[0] == 'mode': self.all_modes.add(parts[1])
                     elif parts[0] == 'store': self.all_stores.add(parts[1])
                     elif parts[0] == 'lander': self.all_landers.add(parts[1])


        # Build navigation graph for each rover based on can_traverse and visible
        can_traverse_facts = {tuple(get_parts(f)) for f in static_facts if get_parts(f) and get_parts(f)[0] == 'can_traverse' and len(get_parts(f)) > 3}
        visible_facts = {tuple(get_parts(f)) for f in static_facts if get_parts(f) and get_parts(f)[0] == 'visible' and len(get_parts(f)) > 2}

        for r in self.all_rovers:
            for from_wp in self.all_waypoints:
                for to_wp in self.all_waypoints:
                    # An edge exists if the rover can traverse AND the waypoints are visible
                    if ('can_traverse', r, from_wp, to_wp) in can_traverse_facts and ('visible', from_wp, to_wp) in visible_facts:
                         self.rover_navigation_graph[r][from_wp].add(to_wp)

        # Populate calib_target_visible_from_wp based on cam_calib_target and obj_visible_from_wp
        for cam, target_obj in self.cam_calib_target.items():
             if target_obj in self.obj_visible_from_wp:
                 self.calib_target_visible_from_wp[target_obj].update(self.obj_visible_from_wp[target_obj])

        # Find communication waypoints visible from lander
        if lander_wp: # Ensure lander location was found
            for wp in self.all_waypoints:
                 if ('visible', wp, lander_wp) in visible_facts:
                     self.comm_wps_visible_from_lander_wp.add(wp)

        # Precompute APSP for each rover
        self.rover_distances = {} # rover -> from_wp -> to_wp -> distance
        for r in self.all_rovers:
            self.rover_distances[r] = {}
            for start_node in self.all_waypoints:
                self.rover_distances[r][start_node] = {}
                q = deque([(start_node, 0)])
                visited = {start_node}
                while q:
                    curr_node, dist = q.popleft()
                    self.rover_distances[r][start_node][curr_node] = dist
                    for neighbor in self.rover_navigation_graph[r][curr_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

        # Precompute minimum distances from any waypoint to specific target sets
        # This helps speed up the __call__ method
        self.min_dist_to_comm = defaultdict(dict) # rover -> from_wp -> min_dist
        self.min_dist_to_obj_image_wp = defaultdict(lambda: defaultdict(dict)) # rover -> from_wp -> objective -> min_dist
        self.min_dist_to_calib_wp = defaultdict(lambda: defaultdict(dict)) # rover -> from_wp -> camera -> min_dist

        for r in self.all_rovers:
            for from_wp in self.all_waypoints:
                # Min dist to communication points
                min_d = sys.maxsize
                if self.comm_wps_visible_from_lander_wp:
                    for comm_wp in self.comm_wps_visible_from_lander_wp:
                        if comm_wp in self.rover_distances[r].get(from_wp, {}):
                             min_d = min(min_d, self.rover_distances[r][from_wp][comm_wp])
                self.min_dist_to_comm[r][from_wp] = min_d if min_d != sys.maxsize else sys.maxsize

                # Min dist to image waypoints for each objective
                for obj in self.all_objectives:
                    min_d = sys.maxsize
                    if obj in self.obj_visible_from_wp and self.obj_visible_from_wp[obj]:
                        for img_wp in self.obj_visible_from_wp[obj]:
                            if img_wp in self.rover_distances[r].get(from_wp, {}):
                                min_d = min(min_d, self.rover_distances[r][from_wp][img_wp])
                    self.min_dist_to_obj_image_wp[r][from_wp][obj] = min_d if min_d != sys.maxsize else sys.maxsize

                # Min dist to calibration waypoints for each camera
                for cam in self.all_cameras:
                    min_d = sys.maxsize
                    if cam in self.cam_calib_target:
                        target_obj = self.cam_calib_target[cam]
                        if target_obj in self.calib_target_visible_from_wp and self.calib_target_visible_from_wp[target_obj]:
                             for calib_wp in self.calib_target_visible_from_wp[target_obj]:
                                 if calib_wp in self.rover_distances[r].get(from_wp, {}):
                                     min_d = min(min_d, self.rover_distances[r][from_wp][calib_wp])
                    self.min_dist_to_calib_wp[r][from_wp][cam] = min_d if min_d != sys.maxsize else sys.maxsize


    def get_distance(self, rover, from_wp, to_wp):
        """Helper to get precomputed distance, returns sys.maxsize if unreachable."""
        return self.rover_distances.get(rover, {}).get(from_wp, {}).get(to_wp, sys.maxsize)

    def get_min_dist_to_set(self, rover, from_wp, target_set_type, target_param=None):
        """Helper to get precomputed min distance from a waypoint to a set of waypoints."""
        if target_set_type == 'comm':
            return self.min_dist_to_comm.get(rover, {}).get(from_wp, sys.maxsize)
        elif target_set_type == 'image':
            # target_param is objective
            return self.min_dist_to_obj_image_wp.get(rover, {}).get(from_wp, {}).get(target_param, sys.maxsize)
        elif target_set_type == 'calib':
            # target_param is camera
            return self.min_dist_to_calib_wp.get(rover, {}).get(from_wp, {}).get(target_param, sys.maxsize)
        return sys.maxsize # Should not happen


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        Estimates the sum of minimum costs for each unachieved goal.
        """
        state = node.state

        # --- Parse Current State Information ---
        rover_locations = {} # rover -> waypoint
        rover_soil_data = defaultdict(set) # rover -> {waypoint, ...}
        rover_rock_data = defaultdict(set) # rover -> {waypoint, ...}
        rover_image_data = defaultdict(set) # rover -> {(objective, mode), ...}
        store_status = {} # store -> 'empty' or 'full'
        cam_calibration = {} # camera -> rover if calibrated, else None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            if parts[0] == 'at' and parts[1] in self.all_rovers:
                if len(parts) > 2:
                    rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis':
                if len(parts) > 2:
                    rover_soil_data[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis':
                if len(parts) > 2:
                    rover_rock_data[parts[1]].add(parts[2])
            elif parts[0] == 'have_image':
                if len(parts) > 3:
                    rover_image_data[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'empty':
                if len(parts) > 1:
                    store_status[parts[1]] = 'empty'
            elif parts[0] == 'full':
                if len(parts) > 1:
                    store_status[parts[1]] = 'full'
            elif parts[0] == 'calibrated':
                if len(parts) > 2:
                    cam_calibration[parts[1]] = parts[2]

        total_heuristic = 0
        unreachable_goals_count = 0 # Count goals that are impossible to reach

        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts:
                 unreachable_goals_count += 1 # Malformed goal?
                 continue

            goal_type = parts[0]
            min_cost_for_this_goal = sys.maxsize # Minimum cost to achieve this specific goal

            if goal_type == 'communicated_soil_data':
                if len(parts) < 2:
                    unreachable_goals_count += 1
                    continue
                w_goal = parts[1]
                for r in self.all_rovers:
                    if 'soil' not in self.rover_capabilities.get(r, set()): continue # Rover cannot sample soil
                    loc_r = rover_locations.get(r)
                    if loc_r is None: continue # Rover location unknown

                    cost_get_data = sys.maxsize
                    loc_after_get = None

                    # Option 1: Rover already has the data
                    if w_goal in rover_soil_data.get(r, set()):
                        cost_get_data = 0
                        loc_after_get = loc_r
                    # Option 2: Need to sample
                    elif w_goal in self.soil_samples_at:
                        store = self.rover_stores.get(r, [None])[0] # Assume one store per rover
                        if store is None: continue # Rover has no store?

                        drop_cost = 1 if store_status.get(store) == 'full' else 0
                        dist_to_sample = self.get_distance(r, loc_r, w_goal)
                        if dist_to_sample != sys.maxsize:
                            cost_get_data = dist_to_sample + 1 + drop_cost # navigate + sample + drop (if needed)
                            loc_after_get = w_goal

                    # If data can be obtained (either already had or sampled)
                    if cost_get_data != sys.maxsize:
                        # Cost to communicate from location after getting data
                        dist_to_comm_set = self.get_min_dist_to_set(r, loc_after_get, 'comm')
                        if dist_to_comm_set != sys.maxsize:
                            cost_comm = dist_to_comm_set + 1 # navigate + communicate
                            min_cost_for_this_goal = min(min_cost_for_this_goal, cost_get_data + cost_comm)

            elif goal_type == 'communicated_rock_data':
                 if len(parts) < 2:
                     unreachable_goals_count += 1
                     continue
                 w_goal = parts[1]
                 for r in self.all_rovers:
                     if 'rock' not in self.rover_capabilities.get(r, set()): continue
                     loc_r = rover_locations.get(r)
                     if loc_r is None: continue

                     cost_get_data = sys.maxsize
                     loc_after_get = None

                     # Option 1: Rover already has the data
                     if w_goal in rover_rock_data.get(r, set()):
                         cost_get_data = 0
                         loc_after_get = loc_r
                     # Option 2: Need to sample
                     elif w_goal in self.rock_samples_at:
                         store = self.rover_stores.get(r, [None])[0]
                         if store is None: continue

                         drop_cost = 1 if store_status.get(store) == 'full' else 0
                         dist_to_sample = self.get_distance(r, loc_r, w_goal)
                         if dist_to_sample != sys.maxsize:
                             cost_get_data = dist_to_sample + 1 + drop_cost # navigate + sample + drop (if needed)
                             loc_after_get = w_goal

                     # If data can be obtained
                     if cost_get_data != sys.maxsize:
                         dist_to_comm_set = self.get_min_dist_to_set(r, loc_after_get, 'comm')
                         if dist_to_comm_set != sys.maxsize:
                             cost_comm = dist_to_comm_set + 1
                             min_cost_for_this_goal = min(min_cost_for_this_goal, cost_get_data + cost_comm)

            elif goal_type == 'communicated_image_data':
                if len(parts) < 3:
                    unreachable_goals_count += 1
                    continue
                o_goal, m_goal = parts[1], parts[2]

                for r in self.all_rovers:
                    if 'imaging' not in self.rover_capabilities.get(r, set()): continue
                    loc_r = rover_locations.get(r)
                    if loc_r is None: continue

                    for cam in self.all_cameras:
                        if self.cam_on_board_rover.get(cam) != r: continue
                        if m_goal not in self.cam_supports_mode.get(cam, set()): continue

                        # Cost if image is already held
                        if (o_goal, m_goal) in rover_image_data.get(r, set()):
                            dist_to_comm_set = self.get_min_dist_to_set(r, loc_r, 'comm')
                            if dist_to_comm_set != sys.maxsize:
                                cost_g_option = dist_to_comm_set + 1 # Nav + Communicate
                                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_g_option)
                                # Found a way via existing image, continue to check imaging path for this rover/cam

                        # Cost if image needs to be taken
                        image_wps = self.obj_visible_from_wp.get(o_goal, set())
                        if not image_wps: continue # Cannot image this objective

                        # Option A: Already calibrated
                        if cam_calibration.get(cam) == r:
                            cost_calib = 0
                            loc_after_calib = loc_r

                            # Find min cost: Nav(loc_after_calib -> image_wp) + Image + Nav(image_wp -> comm_wp) + Comm
                            min_cost_from_calib_loc = sys.maxsize
                            for image_wp in image_wps:
                                dist_to_image = self.get_distance(r, loc_after_calib, image_wp)
                                if dist_to_image != sys.maxsize:
                                    cost_image_nav = dist_to_image + 1 # Nav + Image
                                    dist_to_comm_set = self.get_min_dist_to_set(r, image_wp, 'comm')
                                    if dist_to_comm_set != sys.maxsize:
                                        cost_comm_nav = dist_to_comm_set + 1 # Nav + Communicate
                                        min_cost_from_calib_loc = min(min_cost_from_calib_loc, cost_image_nav + cost_comm_nav)

                            if min_cost_from_calib_loc != sys.maxsize:
                                cost_g_option = cost_calib + min_cost_from_calib_loc
                                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_g_option)


                        # Option B: Need calibration
                        target_obj = self.cam_calib_target.get(cam)
                        if target_obj:
                            calib_wps = self.calib_target_visible_from_wp.get(target_obj, set())
                            if not calib_wps: continue # Cannot calibrate this camera

                            # Find min cost: Nav(loc_r -> calib_wp) + Calib + Nav(calib_wp -> image_wp) + Image + Nav(image_wp -> comm_wp) + Comm
                            min_cost_from_loc_r = sys.maxsize
                            for calib_wp in calib_wps:
                                dist_to_calib = self.get_distance(r, loc_r, calib_wp)
                                if dist_to_calib != sys.maxsize:
                                    cost_calib_nav = dist_to_calib + 1 # Nav + Calib
                                    loc_after_calib = calib_wp

                                    # From this calibration waypoint, find min cost to image and communicate
                                    min_cost_from_calib_wp = sys.maxsize
                                    for image_wp in image_wps:
                                        dist_to_image = self.get_distance(r, loc_after_calib, image_wp)
                                        if dist_to_image != sys.maxsize:
                                            cost_image_nav = dist_to_image + 1 # Nav + Image
                                            dist_to_comm_set = self.get_min_dist_to_set(r, image_wp, 'comm')
                                            if dist_to_comm_set != sys.maxsize:
                                                cost_comm_nav = dist_to_comm_set + 1 # Nav + Communicate
                                                min_cost_from_calib_wp = min(min_cost_from_calib_wp, cost_image_nav + cost_comm_nav)

                                    if min_cost_from_calib_wp != sys.maxsize:
                                        min_cost_from_loc_r = min(min_cost_from_loc_r, cost_calib_nav + min_cost_from_calib_wp)

                            if min_cost_from_loc_r != sys.maxsize:
                                cost_g_option = min_cost_from_loc_r
                                min_cost_for_this_goal = min(min_cost_for_this_goal, cost_g_option)


            # Add cost for this goal
            if min_cost_for_this_goal != sys.maxsize:
                 total_heuristic += min_cost_for_this_goal
            else:
                 unreachable_goals_count += 1

        # Final heuristic value
        if unreachable_goals_count > 0:
             # If any goal is unreachable, the total heuristic is infinity
             return sys.maxsize
        else:
             # Otherwise, it's the sum of minimum costs for reachable goals
             return total_heuristic
