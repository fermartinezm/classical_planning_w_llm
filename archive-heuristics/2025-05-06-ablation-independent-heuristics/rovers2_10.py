# Add necessary imports
from heuristics.heuristic_base import Heuristic
from task import Task # Used in __init__
from collections import deque
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a predicate and arguments."""
    # Remove surrounding parentheses
    fact_string = fact_string.strip()[1:-1]
    parts = fact_string.split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function for BFS shortest path calculation
def bfs(graph, start_node):
    """Computes shortest path distances from start_node in an unweighted graph."""
    distances = {node: math.inf for node in graph}
    if start_node in graph: # Ensure start_node is in the graph nodes
        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == math.inf: # Check if visited
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
                        visited.add(neighbor) # Add to visited here to prevent re-queueing
    return distances

class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for each individual unachieved goal.
    The cost for each goal includes the necessary action costs (sampling, imaging,
    calibrating, dropping, communicating) and the estimated navigation costs
    for a suitable rover to reach the required locations (sample/image waypoint
    and a communication waypoint).

    Assumptions:
    - This heuristic is non-admissible and designed for greedy best-first search.
    - It simplifies the cost calculation by assuming action costs (sample, image,
      calibrate, drop, communicate) are 1. A drop action cost (1) is added only
      if the specific rover chosen for sampling/imaging has a full store. Calibration
      is assumed to cost 1 whenever an image needs to be taken, regardless of
      current calibration status (as calibration is consumed by take_image).
    - Navigation costs are estimated using precomputed shortest paths (BFS)
      on the rover-specific traversability graph.
    - For goals requiring data collection/imaging, the heuristic estimates the
      minimum total cost (navigation + actions) across all capable rovers and
      all relevant waypoints (sample/image point and communication point).
    - For goals where data already exists, the heuristic estimates the minimum
      cost (navigation + communicate) across all rovers possessing the data
      and all communication points.
    - If any necessary navigation step for a goal is impossible (waypoints are
      unreachable), the heuristic returns infinity for that state, indicating
      a likely dead end.
    - Assumes solvable instances unless navigation makes a goal impossible.

    Heuristic Initialization:
    The constructor precomputes static information from the task description
    to enable efficient heuristic calculation during search.
    1. Parses static facts to identify waypoints, rovers, lander location,
       rover capabilities (soil, rock, imaging), store-of mapping, camera
       information (on-board rover, supported modes, calibration target),
       objective visibility from waypoints, and rover traversability edges.
    2. Parses initial state facts to identify initial soil/rock sample locations.
    3. Builds a navigation graph for each rover based on `can_traverse` facts.
    4. Computes all-pairs shortest path distances for each rover using BFS on
       its navigation graph.
    5. Identifies communication points (waypoints visible from the lander location).
    6. Builds a structure mapping cameras to waypoints from which their
       calibration targets are visible.
    7. Pre-calculates sets of rovers capable of soil, rock, and imaging tasks.

    Step-By-Step Thinking for Computing Heuristic (__call__ method):
    1. Get the current state from the search node.
    2. Extract dynamic information from the current state: current location of
       each rover, status of each rover's store (full/empty), current data
       held by rovers (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
       (Calibration status is not explicitly used for cost calculation simplification).
    3. Initialize the total heuristic value `h` to 0.
    4. Identify the set of goals that are not yet satisfied in the current state.
    5. If all goals are satisfied, return 0.
    6. Iterate through each unachieved goal:
        a. Parse the goal fact (e.g., `(communicated_soil_data waypointX)`).
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r` in the current state.
            ii. If data exists: Calculate the minimum cost to communicate. This is 1 (communicate action) plus the minimum navigation distance for a rover that has the data, from its current location to any communication point. Find the minimum over all rovers with the data and all communication points. Add this minimum cost to `h`.
            iii. If data needs collecting: Calculate the minimum total cost including collection and communication. This involves finding the minimum cost over all capable rovers and all communication points. For each rover/comm_point pair, the cost is: navigation from current location to `?w`, plus 1 (sample_soil), plus 1 (drop if that rover's store is full), plus navigation from `?w` to the communication point, plus 1 (communicate). Find the minimum over all capable rovers and all communication points. Add the overall minimum cost to `h`.
            iv. If the minimum cost for this goal is infinity (no path exists or no capable rover/waypoint), return infinity for the total heuristic.
        c. If the goal is `(communicated_rock_data ?w)`: Follow a similar process as for soil data, using rock-specific predicates and capabilities.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Check if `(have_image ?r ?o ?m)` is true for any rover `?r` in the current state.
            ii. If image exists: Calculate the minimum cost to communicate, similar to soil/rock data. Add this minimum cost to `h`.
            iii. If image needs taking: Calculate the minimum total cost including imaging and communication. This involves finding the minimum cost over all capable rovers/cameras, suitable image waypoints, and communication points. For each rover/camera/img_wp/comm_wp combination, the cost is: navigation from current location to `?img_w`, plus 1 (calibrate), plus 1 (take_image), plus navigation from `?img_w` to the communication point, plus 1 (communicate). Find the minimum over all capable rovers/cameras, suitable image waypoints, and communication points.
            iv. If the minimum cost for this goal is infinity, return infinity.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.initial_state = task.initial_state # Keep initial state to check initial samples

        # --- Precomputation from static and initial facts ---
        self.waypoints = set()
        self.rovers = set()
        self.lander_location = None
        self.communication_points = set()
        self.rover_capabilities = {} # rover -> {soil, rock, imaging}
        self.store_of_map = {} # rover -> store
        self.camera_info = {} # camera -> {on_board: rover, supports: {mode}, calibration_target: objective}
        self.objective_visibility = {} # objective -> {waypoint}
        self.soil_sample_locations_init = set() # waypoints with soil sample initially
        self.rock_sample_locations_init = set() # waypoints with rock sample initially

        # Collect all objects by type from relevant predicates in all facts
        all_facts_strings = set(task.initial_state) | set(task.static) | set(task.goals)
        for fact_string in all_facts_strings:
             pred, args = parse_fact(fact_string)
             if pred in ['at', 'at_lander', 'can_traverse', 'visible', 'visible_from', 'at_soil_sample', 'at_rock_sample']:
                 for arg in args:
                     if arg.startswith('waypoint'): self.waypoints.add(arg)
                     if arg.startswith('rover'): self.rovers.add(arg)
                     if arg.startswith('objective'): self.objective_visibility.setdefault(arg, set()) # Ensure objective exists even if no visible_from fact
             elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging', 'store_of', 'on_board']:
                  if args[0].startswith('rover'): self.rovers.add(args[0])
                  if len(args) > 1 and args[1].startswith('rover'): self.rovers.add(args[1])
                  if args[0].startswith('store'): self.store_of_map.setdefault(args[1], args[0]) # rover -> store
                  if args[0].startswith('camera'): self.camera_info.setdefault(args[0], {}) # Ensure camera exists
             elif pred in ['calibration_target', 'supports']:
                  if args[0].startswith('camera'): self.camera_info.setdefault(args[0], {}) # Ensure camera exists
                  if len(args) > 1 and args[1].startswith('objective'): self.objective_visibility.setdefault(args[1], set()) # Ensure objective exists

        # Parse static facts specifically
        rover_can_traverse_edges = {} # rover -> set of (w1, w2)
        visible_facts_set = set() # set of (w1, w2)

        for fact_string in task.static:
            pred, args = parse_fact(fact_string)
            if pred == 'at_lander':
                self.lander_location = args[1]
            elif pred == 'can_traverse':
                rover, w1, w2 = args
                rover_can_traverse_edges.setdefault(rover, set()).add((w1, w2))
            elif pred == 'equipped_for_soil_analysis':
                self.rover_capabilities.setdefault(args[0], set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities.setdefault(args[0], set()).add('rock')
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities.setdefault(args[0], set()).add('imaging')
            elif pred == 'store_of':
                self.store_of_map[args[1]] = args[0] # rover -> store
            elif pred == 'visible':
                w1, w2 = args
                visible_facts_set.add((w1, w2))
                visible_facts_set.add((w2, w1)) # visible is symmetric
            elif pred == 'calibration_target':
                camera, objective = args
                self.camera_info.setdefault(camera, {}).update({'calibration_target': objective})
            elif pred == 'on_board':
                camera, rover = args
                self.camera_info.setdefault(camera, {}).update({'on_board': rover})
            elif pred == 'supports':
                camera, mode = args
                self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)
            elif pred == 'visible_from':
                objective, waypoint = args
                self.objective_visibility.setdefault(objective, set()).add(waypoint)

        # Parse initial state for initial sample locations
        for fact_string in task.initial_state:
             pred, args = parse_fact(fact_string)
             if pred == 'at_soil_sample':
                 self.soil_sample_locations_init.add(args[0])
             elif pred == 'at_rock_sample':
                 self.rock_sample_locations_init.add(args[0])

        # Build rover graphs and compute distances
        self.rover_graphs = {}
        self.rover_distances = {}
        for rover in self.rovers:
            graph = {wp: set() for wp in self.waypoints}
            if rover in rover_can_traverse_edges:
                for w1, w2 in rover_can_traverse_edges[rover]:
                    if w1 in graph and w2 in graph: # Ensure waypoints are known
                         graph[w1].add(w2)
            self.rover_graphs[rover] = graph
            self.rover_distances[rover] = {}
            for start_wp in self.waypoints:
                self.rover_distances[rover][start_wp] = bfs(graph, start_wp)

        # Determine communication points (visible from lander location)
        lander_wp = self.lander_location
        if lander_wp:
            if lander_wp in self.waypoints: # Lander location itself is a communication point
                 self.communication_points.add(lander_wp)
            for w1, w2 in visible_facts_set:
                if w1 == lander_wp and w2 in self.waypoints:
                    self.communication_points.add(w2)
                elif w2 == lander_wp and w1 in self.waypoints:
                     self.communication_points.add(w1)

        # Build calibration_targets_visible_from structure
        # camera -> {waypoint -> {objective}}
        self.calibration_targets_visible_from = {}
        for cam, info in self.camera_info.items():
            cal_target_obj = info.get('calibration_target')
            if cal_target_obj:
                self.calibration_targets_visible_from.setdefault(cam, {})
                # Find waypoints where the calibration target is visible from
                if cal_target_obj in self.objective_visibility:
                    for wp in self.objective_visibility[cal_target_obj]:
                         if wp in self.waypoints: # Ensure waypoint is known
                            self.calibration_targets_visible_from[cam].setdefault(wp, set()).add(cal_target_obj)

        # Pre-calculate capable rover sets
        self._capable_soil_rovers = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps}
        self._capable_rock_rovers = {r for r, caps in self.rover_capabilities.items() if 'rock' in caps}
        self._capable_imaging_rovers = {r for r, caps in self.rover_capabilities.items() if 'imaging' in caps}


    # Helper methods (defined inside the class)
    def get_rover_location(self, state, rover_name):
        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'at' and args[0] == rover_name:
                return args[1]
        return None # Should not happen in valid states

    def get_rover_store_status(self, state, rover_name):
        store_name = self.store_of_map.get(rover_name)
        if not store_name: return 'unknown' # Should not happen if store_of is in static
        if f'(full {store_name})' in state:
            return 'full'
        if f'(empty {store_name})' in state:
            return 'empty'
        return 'unknown' # Should not happen

    def get_capable_soil_rovers(self):
        return self._capable_soil_rovers

    def get_capable_rock_rovers(self):
        return self._capable_rock_rovers

    def get_capable_imaging_rovers_cameras(self, mode_name):
        result = set() # set of (rover_name, camera_name)
        for cam_name, info in self.camera_info.items():
            rover_name = info.get('on_board')
            supported_modes = info.get('supports', set())
            # Check if camera is on a rover equipped for imaging and supports the mode
            if rover_name in self._capable_imaging_rovers and mode_name in supported_modes:
                result.add((rover_name, cam_name))
        return result

    def get_suitable_image_waypoints(self, objective_name, camera_name):
        # A waypoint is suitable if the objective is visible from it AND
        # a calibration target for the camera is visible from it.
        obj_visible_wps = self.objective_visibility.get(objective_name, set())
        cal_target_visible_wps_map = self.calibration_targets_visible_from.get(camera_name, {})
        cal_target_visible_wps = set(cal_target_visible_wps_map.keys())

        # Intersection of waypoints where objective is visible and waypoints where cal target is visible
        suitable_wps = obj_visible_wps.intersection(cal_target_visible_wps)

        return suitable_wps


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state
        h = 0

        # Get current dynamic information from the state
        current_rover_locations = {r: self.get_rover_location(state, r) for r in self.rovers}
        current_store_statuses = {r: self.get_rover_store_status(state, r) for r in self.rovers if r in self.store_of_map}

        # Pre-compute sets for faster lookup of data/calibration in state
        soil_data_in_state = set() # {(rover, waypoint)}
        rock_data_in_state = set() # {(rover, waypoint)}
        image_data_in_state = set() # {(rover, objective, mode)}

        for fact_string in state:
            pred, args = parse_fact(fact_string)
            if pred == 'have_soil_analysis':
                soil_data_in_state.add((args[0], args[1]))
            elif pred == 'have_rock_analysis':
                rock_data_in_state.add((args[0], args[1]))
            elif pred == 'have_image':
                image_data_in_state.add(tuple(args)) # (rover, objective, mode)

        # Identify unachieved goals
        unachieved_goals = [g for g in self.goals if g not in state]

        if not unachieved_goals:
            return 0 # Goal reached

        for goal_string in unachieved_goals:
            pred, args = parse_fact(goal_string)

            if pred == 'communicated_soil_data':
                w_name = args[0]

                # Check if data exists on any rover
                data_exists = any((r, w_name) in soil_data_in_state for r in self.rovers)

                if data_exists:
                    # Cost is navigation to communication point + communicate action
                    rovers_with_data = {r for r in self.rovers if (r, w_name) in soil_data_in_state}
                    min_comm_cost = math.inf # Cost is navigate + communicate

                    for r_name in rovers_with_data:
                        current_w = current_rover_locations.get(r_name)
                        if current_w is None: continue # Should not happen

                        for comm_w in self.communication_points:
                            dist = self.rover_distances.get(r_name, {}).get(current_w, {}).get(comm_w, math.inf)
                            if dist != math.inf:
                                 current_rover_comm_cost = dist + 1 # +1 for communicate action
                                 min_comm_cost = min(min_comm_cost, current_rover_comm_cost)

                    if min_comm_cost == math.inf: return math.inf # Cannot reach lander
                    h += min_comm_cost

                else: # Data needs collecting
                    # Cost is navigate to sample + sample + (drop) + navigate to comm + communicate
                    min_total_cost = math.inf
                    capable_rovers = self.get_capable_soil_rovers()

                    if not capable_rovers: return math.inf # No rover can collect soil

                    for r_name in capable_rovers:
                        current_w = current_rover_locations.get(r_name)
                        if current_w is None: continue

                        # Cost for sample + potential drop
                        sample_action_cost = 1
                        store_status = current_store_statuses.get(r_name)
                        drop_action_cost = 1 if store_status == 'full' else 0

                        for comm_w in self.communication_points:
                            dist_to_sample = self.rover_distances.get(r_name, {}).get(current_w, {}).get(w_name, math.inf)
                            dist_sample_to_comm = self.rover_distances.get(r_name, {}).get(w_name, {}).get(comm_w, math.inf)

                            if dist_to_sample != math.inf and dist_sample_to_comm != math.inf:
                                 # Total cost for this rover and this comm point
                                 current_rover_total_cost = dist_to_sample + sample_action_cost + drop_action_cost + dist_sample_to_comm + 1 # +1 for communicate action
                                 min_total_cost = min(min_total_cost, current_rover_total_cost)

                    if min_total_cost == math.inf: return math.inf # Cannot reach sample or lander
                    h += min_total_cost

            elif pred == 'communicated_rock_data':
                w_name = args[0]

                # Check if data exists on any rover
                data_exists = any((r, w_name) in rock_data_in_state for r in self.rovers)

                if data_exists:
                    # Cost is navigation to communication point + communicate action
                    rovers_with_data = {r for r in self.rovers if (r, w_name) in rock_data_in_state}
                    min_comm_cost = math.inf # Cost is navigate + communicate

                    for r_name in rovers_with_data:
                        current_w = current_rover_locations.get(r_name)
                        if current_w is None: continue

                        for comm_w in self.communication_points:
                            dist = self.rover_distances.get(r_name, {}).get(current_w, {}).get(comm_w, math.inf)
                            if dist != math.inf:
                                 current_rover_comm_cost = dist + 1 # +1 for communicate action
                                 min_comm_cost = min(min_comm_cost, current_rover_comm_cost)

                    if min_comm_cost == math.inf: return math.inf # Cannot reach lander
                    h += min_comm_cost

                else: # Data needs collecting
                    # Cost is navigate to sample + sample + (drop) + navigate to comm + communicate
                    min_total_cost = math.inf
                    capable_rovers = self.get_capable_rock_rovers()

                    if not capable_rovers: return math.inf # No rover can collect rock

                    for r_name in capable_rovers:
                        current_w = current_rover_locations.get(r_name)
                        if current_w is None: continue

                        # Cost for sample + potential drop
                        sample_action_cost = 1
                        store_status = current_store_statuses.get(r_name)
                        drop_action_cost = 1 if store_status == 'full' else 0

                        for comm_w in self.communication_points:
                            dist_to_sample = self.rover_distances.get(r_name, {}).get(current_w, {}).get(w_name, math.inf)
                            dist_sample_to_comm = self.rover_distances.get(r_name, {}).get(w_name, {}).get(comm_w, math.inf)

                            if dist_to_sample != math.inf and dist_sample_to_comm != math.inf:
                                 # Total cost for this rover and this comm point
                                 current_rover_total_cost = dist_to_sample + sample_action_cost + drop_action_cost + dist_sample_to_comm + 1 # +1 for communicate action
                                 min_total_cost = min(min_total_cost, current_rover_total_cost)

                    if min_total_cost == math.inf: return math.inf # Cannot reach sample or lander
                    h += min_total_cost

            elif pred == 'communicated_image_data':
                o_name = args[0]
                m_name = args[1]

                # Check if image exists on any rover
                image_exists = any((r, o_name, m_name) in image_data_in_state for r in self.rovers)

                if image_exists:
                    # Cost is navigation to communication point + communicate action
                    rovers_with_image = {r for r in self.rovers if (r, o_name, m_name) in image_data_in_state}
                    min_comm_cost = math.inf # Cost is navigate + communicate

                    for r_name in rovers_with_image:
                        current_w = current_rover_locations.get(r_name)
                        if current_w is None: continue

                        for comm_w in self.communication_points:
                            dist = self.rover_distances.get(r_name, {}).get(current_w, {}).get(comm_w, math.inf)
                            if dist != math.inf:
                                 current_rover_comm_cost = dist + 1 # +1 for communicate action
                                 min_comm_cost = min(min_comm_cost, current_rover_comm_cost)

                    if min_comm_cost == math.inf: return math.inf # Cannot reach lander
                    h += min_comm_cost

                else: # Image needs taking
                    # Cost is navigate to image point + calibrate + take_image + navigate to comm + communicate
                    min_total_cost = math.inf
                    capable_rovers_cameras = self.get_capable_imaging_rovers_cameras(m_name)

                    if not capable_rovers_cameras: return math.inf # No rover/camera can take this image

                    for r_name, i_name in capable_rovers_cameras:
                        current_w = current_rover_locations.get(r_name)
                        if current_w is None: continue

                        # Cost for take_image + calibrate
                        image_action_cost = 1
                        calibrate_action_cost = 1 # Simplification: always add 1 for calibration

                        # Find suitable image waypoints for this objective and camera
                        suitable_img_waypoints = self.get_suitable_image_waypoints(o_name, i_name)

                        if not suitable_img_waypoints: continue # Cannot take this image with this camera/rover from any suitable waypoint

                        for img_w in suitable_img_waypoints:
                             for comm_w in self.communication_points:
                                dist_to_img = self.rover_distances.get(r_name, {}).get(current_w, {}).get(img_w, math.inf)
                                dist_img_to_comm = self.rover_distances.get(r_name, {}).get(img_w, {}).get(comm_w, math.inf)

                                if dist_to_img != math.inf and dist_img_to_comm != math.inf:
                                     # Total cost for this rover/camera/img_wp/comm_wp combination
                                     current_combo_total_cost = dist_to_img + calibrate_action_cost + image_action_cost + dist_img_to_comm + 1 # +1 for communicate action
                                     min_total_cost = min(min_total_cost, current_combo_total_cost)

                    if min_total_cost == math.inf: return math.inf # Cannot reach image point or lander
                    h += min_total_cost

        return h
