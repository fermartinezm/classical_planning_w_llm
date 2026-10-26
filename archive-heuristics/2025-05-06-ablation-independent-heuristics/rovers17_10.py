import heapq
import logging
from collections import defaultdict, deque

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a list of components."""
    # Remove surrounding parentheses and split by space
    # Example: '(at rover1 waypoint1)' -> ['at', 'rover1', 'waypoint1']
    parts = fact_string.strip('()').split()
    return parts

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal state by summing the estimated costs
    for each unachieved goal fact. The cost for each goal fact is estimated
    based on the actions and navigation required to achieve it from the current
    state, considering rover capabilities, locations, and required waypoints
    (sample locations, image locations, calibration locations, communication
    locations). Navigation costs are precomputed shortest path distances
    using BFS on the traversable graph for each rover.

    Assumptions:
    - The planning task is an instance of the Rovers domain.
    - The input state is a frozenset of fact strings.
    - Static facts are provided in task.static.
    - The domain structure (predicates, actions) is as defined in rovers.pddl.
    - Navigation actions have a cost of 1. Other actions (sample, drop, calibrate, take_image, communicate) have a cost of 1.
    - If a soil/rock sample is not at its initial location and no rover has it, it's considered lost and the goal is unreachable.
    - For imaging goals, the heuristic assumes a sequence of navigation to image waypoint, navigation to calibration waypoint, calibrate, navigation back to image waypoint, take image, navigation to communication waypoint, communicate.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic precomputes static information
    from task.static and task.initial_state:
    - Lander locations.
    - Waypoints visible from lander locations (communication waypoints).
    - Rover capabilities (soil, rock, imaging).
    - Store ownership.
    - Camera information (on-board status, supported modes, calibration targets).
    - Waypoints visible from objectives (image waypoints).
    - Waypoints visible from calibration targets (calibration waypoints).
    - Navigation graph for each rover based on 'can_traverse' facts.
    - All-pairs shortest path distances for each rover's navigation graph using BFS.
    - Initial locations of soil/rock samples.

    Step-By-Step Thinking for Computing Heuristic (__call__):
    1. Get the current state and the set of goal facts.
    2. Initialize the total heuristic value h = 0.
    3. Parse the current state to extract dynamic information:
       - Current location of each rover.
       - Which rover has which soil/rock sample data.
       - Which rover has which image data (objective, mode).
       - Which rover's store is full.
       - Which camera on which rover is calibrated (though this isn't strictly used in the current simple image cost estimate).
       - Which soil/rock samples are still at their initial locations.
    4. Iterate through each goal fact in task.goals:
       - If the goal fact is already true in the current state, its cost is 0.
       - If the goal fact is not true, estimate the cost to achieve it:
         - For a 'communicated_soil_data ?w' goal:
           - If any rover 'r' currently has '(have_soil_analysis r w)':
             - Cost is navigation from r's current location to the nearest communication waypoint + 1 (communicate action). Minimize over such rovers.
           - Else (no rover has the data):
             - If '(at_soil_sample w)' is in the current state:
               - Find equipped rover 'r'. Cost is navigation from r's location to 'w' + 1 (sample) + (1 if r's store is full) + navigation from 'w' to nearest communication waypoint + 1 (communicate). Minimize over equipped rovers.
             - Else ('(at_soil_sample w)' is not in current state):
               - The sample was taken and dropped. Assume unreachable. Cost is infinity.
           - The cost for this goal fact is the minimum of the costs from the above cases.
         - For a 'communicated_rock_data ?w' goal: Similar logic to soil data, using rock capabilities and facts.
         - For a 'communicated_image_data ?o ?m' goal:
           - If any rover 'r' currently has '(have_image r o m)':
             - Cost is navigation from r's current location to the nearest communication waypoint + 1 (communicate action). Minimize over such rovers.
           - Else (no rover has the image):
             - Find equipped rover 'r' with camera 'i' supporting mode 'm'.
             - Find a waypoint 'wp_img' visible from 'o'.
             - Find the calibration target 't' for camera 'i'.
             - Find a waypoint 'wp_cal' visible from 't'.
             - If no such 'wp_img' or 'wp_cal' exists for any suitable rover/camera, the goal is unreachable for this (o, m) pair. Cost is infinity.
             - Otherwise, the cost involves:
               - Navigation from r's current location to 'wp_img'.
               - Navigation from 'wp_img' to 'wp_cal'.
               - 1 (calibrate action).
               - Navigation from 'wp_cal' back to 'wp_img'.
               - 1 (take_image action).
               - Navigation from 'wp_img' to the nearest communication waypoint.
               - 1 (communicate action).
             - Minimize this total cost over all suitable rovers 'r', cameras 'i', image waypoints 'wp_img', and calibration waypoints 'wp_cal'.
         - If the estimated cost for a goal fact is infinity, the total heuristic is infinity.
       - Add the estimated cost for the unachieved goal fact to h.
    5. Return the total heuristic value h.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state for sample locations

        # --- Precompute Static Information ---

        self.lander_locations = set()
        self.rover_capabilities = defaultdict(set) # rover_name -> {soil, rock, imaging}
        self.store_owner = {} # store_name -> rover_name
        self.rover_cameras = defaultdict(set) # rover_name -> {camera_name}
        self.camera_modes = defaultdict(set) # camera_name -> {mode_name}
        self.camera_cal_target = {} # camera_name -> objective_name
        self.visible_wps = defaultdict(set) # wp1 -> {wp2, ...} (visible wp1 from wp2)
        self.visible_from_obj = defaultdict(set) # objective -> {waypoint, ...}
        self.visible_from_cal = defaultdict(set) # objective (cal target) -> {waypoint, ...}
        self.rover_traversal_graphs = defaultdict(dict) # rover -> {wp1: {wp2, ...}}
        self.waypoints = set()
        self.rovers = set()
        self.objectives = set()
        self.modes = set()
        self.cameras = set()
        self.stores = set()
        self.landers = set()

        # Extract all objects and static relationships
        for fact_string in self.static:
            parts = parse_fact(fact_string)
            predicate = parts[0]

            if predicate == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.lander_locations.add(wp)
                self.landers.add(lander)
                self.waypoints.add(wp)
            elif predicate == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_capabilities[rover].add('soil')
                self.rovers.add(rover)
            elif predicate == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_capabilities[rover].add('rock')
                self.rovers.add(rover)
            elif predicate == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_capabilities[rover].add('imaging')
                self.rovers.add(rover)
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_owner[store] = rover
                self.stores.add(store)
                self.rovers.add(rover)
            elif predicate == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.visible_wps[wp1].add(wp2)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
            elif predicate == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                if wp1 not in self.rover_traversal_graphs[rover]:
                    self.rover_traversal_graphs[rover][wp1] = set()
                self.rover_traversal_graphs[rover][wp1].add(wp2)
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.camera_cal_target[camera] = objective
                self.cameras.add(camera)
                self.objectives.add(objective)
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras[rover].add(camera)
                self.cameras.add(camera)
                self.rovers.add(rover)
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_modes[camera].add(mode)
                self.cameras.add(camera)
                self.modes.add(mode)
            # visible_from is processed after camera_cal_target is fully populated

        # Process visible_from specifically for calibration targets and objectives
        for fact_string in self.static:
             parts = parse_fact(fact_string)
             if parts[0] == 'visible_from':
                 objective, waypoint = parts[1], parts[2]
                 self.visible_from_obj[objective].add(waypoint)
                 # Check if this objective is a calibration target for any camera
                 if objective in self.camera_cal_target.values():
                     self.visible_from_cal[objective].add(waypoint)
                 self.objectives.add(objective)
                 self.waypoints.add(waypoint)

        # Determine communication waypoints (visible from any lander location)
        self.comm_wps = set()
        for lander_wp in self.lander_locations:
            # A waypoint x is a comm_wp if (visible x y) is true where y is a lander_wp
            # Need to check visibility in both directions if the domain allows it
            # The domain has (visible ?w ?p) and (visible ?p ?w) for bidirectional visibility
            # So, a waypoint x is a comm_wp if it's in visible_wps[y] for any lander_wp y
            self.comm_wps.update(self.visible_wps.get(lander_wp, set()))
            # Also check reverse visibility: if lander_wp is visible from wp1, then wp1 is a comm_wp
            for wp1, connected_wps in self.visible_wps.items():
                 if lander_wp in connected_wps:
                     self.comm_wps.add(wp1)


        # Precompute shortest paths for each rover
        self.rover_distances = {} # rover -> {start_wp -> {end_wp -> distance}}
        for rover in self.rovers:
            self.rover_distances[rover] = {}
            graph = self.rover_traversal_graphs.get(rover, {})
            # Collect all waypoints mentioned in the rover's graph
            all_wps_in_graph = set(graph.keys())
            for destinations in graph.values():
                 all_wps_in_graph.update(destinations)

            for start_wp in all_wps_in_graph:
                self.rover_distances[rover][start_wp] = self._bfs(graph, start_wp, all_wps_in_graph)

        # Store initial sample locations (they disappear after sampling)
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact_string in self.initial_state:
            parts = parse_fact(fact_string)
            if parts[0] == 'at_soil_sample':
                self.initial_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                self.initial_rock_samples.add(parts[1])


    def _bfs(self, graph, start_node, all_nodes):
        """Performs BFS to find shortest distances from start_node."""
        # Initialize distances for all relevant nodes
        distances = {node: float('inf') for node in all_nodes}

        if start_node not in distances:
             # This can happen if start_node is not in all_nodes (e.g., isolated waypoint)
             # Add it to distances, but it won't be able to reach other nodes in the graph
             distances[start_node] = float('inf') # Will be set to 0 below

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in the graph keys before accessing neighbors
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # Ensure neighbor is one of the nodes we care about (in all_nodes)
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # --- Extract Dynamic Information from State ---
        rover_loc = {} # rover_name -> waypoint
        has_soil = defaultdict(set) # rover_name -> {waypoint, ...}
        has_rock = defaultdict(set) # rover_name -> {waypoint, ...}
        has_image = defaultdict(set) # rover_name -> {(objective, mode), ...}
        store_full = {} # rover_name -> bool
        # camera_calibrated = {} # (camera_name, rover_name) -> bool # Not used in current simple cost estimate
        current_soil_samples = set() # waypoint -> bool (is sample still there)
        current_rock_samples = set() # waypoint -> bool (is sample still there)


        # Initialize store_full to False for all rovers with stores
        for store, rover in self.store_owner.items():
             store_full[rover] = False

        # Initialize camera_calibrated to False for all cameras on rovers
        # for rover, cameras in self.rover_cameras.items():
        #      for camera in cameras:
        #          camera_calibrated[(camera, rover)] = False


        for fact_string in state:
            parts = parse_fact(fact_string)
            predicate = parts[0]

            if predicate == 'at':
                rover, wp = parts[1], parts[2]
                rover_loc[rover] = wp
            elif predicate == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                has_soil[rover].add(wp)
            elif predicate == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                has_rock[rover].add(wp)
            elif predicate == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                has_image[rover].add((obj, mode))
            elif predicate == 'full':
                store = parts[1]
                if store in self.store_owner:
                    rover = self.store_owner[store]
                    store_full[rover] = True
            # elif predicate == 'calibrated':
            #      camera, rover = parts[1], parts[2]
            #      camera_calibrated[(camera, rover)] = True
            elif predicate == 'at_soil_sample':
                 current_soil_samples.add(parts[1])
            elif predicate == 'at_rock_sample':
                 current_rock_samples.add(parts[1])


        total_h = 0

        # --- Estimate Cost for Each Unachieved Goal ---
        for goal_fact_string in self.goals:
            if goal_fact_string in state:
                continue # Goal already achieved

            parts = parse_fact(goal_fact_string)
            predicate = parts[0]
            goal_cost = float('inf') # Cost for this specific goal fact

            if predicate == 'communicated_soil_data':
                w = parts[1]
                min_goal_cost = float('inf')

                # Case 1: Data already collected by some rover
                for rover in self.rovers:
                    if w in has_soil[rover]:
                        current_loc = rover_loc.get(rover)
                        if current_loc is None: continue # Rover location unknown (shouldn't happen in valid state)

                        # Find min nav cost from current_loc to a communication waypoint
                        min_nav_to_comm = float('inf')
                        if self.comm_wps:
                            # Check if rover can reach any comm waypoint from its current location
                            if current_loc in self.rover_distances.get(rover, {}):
                                min_nav_to_comm = min((self.rover_distances[rover][current_loc].get(comm_wp, float('inf'))) for comm_wp in self.comm_wps)

                        if min_nav_to_comm != float('inf'):
                            cost = min_nav_to_comm + 1 # + communicate action
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Data needs to be collected and communicated
                # Check if the sample is still available at the waypoint
                if w in current_soil_samples:
                     for rover in self.rovers:
                         if 'soil' in self.rover_capabilities[rover]:
                             current_loc = rover_loc.get(rover)
                             if current_loc is None: continue

                             # Nav to sample location
                             nav_to_sample = float('inf')
                             if current_loc in self.rover_distances.get(rover, {}):
                                 nav_to_sample = self.rover_distances[rover][current_loc].get(w, float('inf'))

                             if nav_to_sample != float('inf'):
                                 sample_action_cost = 1 # sample_soil
                                 drop_action_cost = 1 if store_full.get(rover, False) else 0 # drop if store is full

                                 # Nav from sample location to nearest communication waypoint
                                 min_nav_from_sample_to_comm = float('inf')
                                 if self.comm_wps:
                                     if w in self.rover_distances.get(rover, {}):
                                         min_nav_from_sample_to_comm = min((self.rover_distances[rover][w].get(comm_wp, float('inf'))) for comm_wp in self.comm_wps)

                                 if min_nav_from_sample_to_comm != float('inf'):
                                     communicate_action_cost = 1 # communicate_soil_data
                                     total_cost = nav_to_sample + sample_action_cost + drop_action_cost + min_nav_from_sample_to_comm + communicate_action_cost
                                     min_goal_cost = min(min_goal_cost, total_cost)

                goal_cost = min_goal_cost


            elif predicate == 'communicated_rock_data':
                w = parts[1]
                min_goal_cost = float('inf')

                # Case 1: Data already collected by some rover
                for rover in self.rovers:
                    if w in has_rock[rover]:
                        current_loc = rover_loc.get(rover)
                        if current_loc is None: continue

                        min_nav_to_comm = float('inf')
                        if self.comm_wps:
                             if current_loc in self.rover_distances.get(rover, {}):
                                min_nav_to_comm = min((self.rover_distances[rover][current_loc].get(comm_wp, float('inf'))) for comm_wp in self.comm_wps)

                        if min_nav_to_comm != float('inf'):
                            cost = min_nav_to_comm + 1 # + communicate action
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Data needs to be collected and communicated
                # Check if the sample is still available at the waypoint
                if w in current_rock_samples:
                     for rover in self.rovers:
                         if 'rock' in self.rover_capabilities[rover]:
                             current_loc = rover_loc.get(rover)
                             if current_loc is None: continue

                             nav_to_sample = float('inf')
                             if current_loc in self.rover_distances.get(rover, {}):
                                 nav_to_sample = self.rover_distances[rover][current_loc].get(w, float('inf'))

                             if nav_to_sample != float('inf'):
                                 sample_action_cost = 1 # sample_rock
                                 drop_action_cost = 1 if store_full.get(rover, False) else 0 # drop if store is full

                                 min_nav_from_sample_to_comm = float('inf')
                                 if self.comm_wps:
                                     if w in self.rover_distances.get(rover, {}):
                                         min_nav_from_sample_to_comm = min((self.rover_distances[rover][w].get(comm_wp, float('inf'))) for comm_wp in self.comm_wps)

                                 if min_nav_from_sample_to_comm != float('inf'):
                                     communicate_action_cost = 1 # communicate_rock_data
                                     total_cost = nav_to_sample + sample_action_cost + drop_action_cost + min_nav_from_sample_to_comm + communicate_action_cost
                                     min_goal_cost = min(min_goal_cost, total_cost)

                goal_cost = min_goal_cost


            elif predicate == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                min_goal_cost = float('inf')

                # Case 1: Image already taken by some rover
                for rover in self.rovers:
                    if (obj, mode) in has_image[rover]:
                        current_loc = rover_loc.get(rover)
                        if current_loc is None: continue

                        min_nav_to_comm = float('inf')
                        if self.comm_wps:
                             if current_loc in self.rover_distances.get(rover, {}):
                                min_nav_to_comm = min((self.rover_distances[rover][current_loc].get(comm_wp, float('inf'))) for comm_wp in self.comm_wps)

                        if min_nav_to_comm != float('inf'):
                            cost = min_nav_to_comm + 1 # + communicate action
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Image needs to be taken and communicated
                # Find suitable rover/camera/waypoints
                suitable_options = [] # List of (rover, camera, wp_img, wp_cal) tuples
                img_wps_o = self.visible_from_obj.get(obj, set())

                if img_wps_o: # Must have at least one waypoint to view the objective
                    for rover in self.rovers:
                        if 'imaging' in self.rover_capabilities[rover]:
                            for camera in self.rover_cameras.get(rover, set()):
                                if mode in self.camera_modes.get(camera, set()):
                                    cal_target = self.camera_cal_target.get(camera)
                                    if cal_target:
                                        cal_wps_t = self.visible_from_cal.get(cal_target, set())
                                        if cal_wps_t: # Must have at least one waypoint to view the calibration target
                                            # Check reachability for all combinations of img_wp and cal_wp
                                            for wp_img in img_wps_o:
                                                for wp_cal in cal_wps_t:
                                                    # Check if rover can reach wp_img and wp_cal from its current location
                                                    # and if wp_img and wp_cal are reachable from each other
                                                    if rover_loc.get(rover) in self.rover_distances.get(rover, {}) and \
                                                       wp_img in self.rover_distances.get(rover, {}) and \
                                                       wp_cal in self.rover_distances.get(rover, {}):
                                                         # Check actual reachability distances
                                                         if self.rover_distances[rover][rover_loc[rover]].get(wp_img, float('inf')) != float('inf') and \
                                                            self.rover_distances[rover][wp_img].get(wp_cal, float('inf')) != float('inf') and \
                                                            self.rover_distances[rover][wp_cal].get(wp_img, float('inf')) != float('inf'):
                                                              suitable_options.append((rover, camera, wp_img, wp_cal))


                if suitable_options:
                    min_imaging_sequence_cost = float('inf')

                    for rover, camera, wp_img, wp_cal in suitable_options:
                        current_loc = rover_loc[rover]

                        # Nav to image waypoint
                        nav_to_img = self.rover_distances[rover][current_loc][wp_img] # Already checked != inf in suitable_options

                        # Calibration sequence cost (Nav to cal + Calibrate + Nav back to img)
                        nav_img_to_cal = self.rover_distances[rover][wp_img][wp_cal] # Already checked != inf
                        nav_cal_to_img = self.rover_distances[rover][wp_cal][wp_img] # Already checked != inf
                        calibrate_action_cost = 1

                        cal_sequence_cost = nav_img_to_cal + calibrate_action_cost + nav_cal_to_img

                        # Take image action cost
                        take_image_action_cost = 1

                        # Total cost to get the image at wp_img
                        cost_to_get_image = nav_to_img + cal_sequence_cost + take_image_action_cost

                        # Nav from image waypoint to nearest communication waypoint
                        min_nav_from_img_to_comm = float('inf')
                        if self.comm_wps:
                             if wp_img in self.rover_distances.get(rover, {}):
                                 min_nav_from_img_to_comm = min((self.rover_distances[rover][wp_img].get(comm_wp, float('inf'))) for comm_wp in self.comm_wps)

                        # Total cost for this specific rover/camera/wp_img/wp_cal combination
                        if min_nav_from_img_to_comm != float('inf'):
                            communicate_action_cost = 1 # communicate_image_data
                            total_cost = cost_to_get_image + min_nav_from_img_to_comm + communicate_action_cost
                            min_imaging_sequence_cost = min(min_imaging_sequence_cost, total_cost)

                    min_goal_cost = min(min_goal_cost, min_imaging_sequence_cost)

                goal_cost = min_goal_cost


            # Add cost for this goal fact to total heuristic
            if goal_cost == float('inf'):
                 # If any single goal is unreachable, the whole task is likely unreachable
                 # Returning infinity guides the search away from dead ends
                 return float('inf')
            total_h += goal_cost

        return total_h
