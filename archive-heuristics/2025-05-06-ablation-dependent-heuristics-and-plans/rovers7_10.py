from fnmatch import fnmatch
from collections import deque, defaultdict
# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

# Utility functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(at rover1 waypoint1)'
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if each part matches the corresponding arg pattern
    # Use min length to handle patterns shorter than fact parts (e.g., matching '(at obj loc)' with 'at', '*')
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest paths
def bfs(start_node, graph, all_nodes):
    """
    Performs BFS from start_node on the given graph to find shortest distances.
    Graph is an adjacency list: {node: {neighbor1, neighbor2, ...}}
    all_nodes is the complete set of nodes in the domain (waypoints).
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node in all_nodes: # Check if start_node is a known node
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances.get(neighbor, float('inf')) == float('inf'): # Use .get for safety
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It calculates the minimum cost for each unachieved
    goal independently and sums these minimum costs. The cost for each goal
    is estimated based on the sequence of actions needed (navigation, sampling/imaging,
    communicating) and the shortest path distances between relevant waypoints
    for each rover.

    # Assumptions
    - The heuristic assumes that each unachieved goal can be pursued independently
      by the most capable and conveniently located rover. It does not account
      for resource contention (e.g., multiple goals needing the same rover or store)
      or synergies (e.g., one navigation action serving multiple goals).
    - Navigation costs are estimated using shortest paths on the traversable graph
      for each rover.
    - Action costs are assumed to be 1.
    - Goals that are unreachable by any available rover are assigned an infinite cost.

    # Heuristic Initialization
    The heuristic precomputes static information from the task definition:
    - Identifies all objects by type (rovers, waypoints, cameras, etc.) present
      in the initial state, goals, or static facts.
    - Maps rovers to their capabilities (soil, rock, imaging), stores, and cameras.
    - Maps cameras to supported modes and calibration targets.
    - Maps objectives and calibration targets to waypoints from which they are visible.
    - Identifies the lander location and communication waypoints (visible from lander).
    - Builds the navigation graph for each rover based on `can_traverse` facts.
    - Computes all-pairs shortest paths for each rover on its navigation graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the total estimated cost as follows:
    1. Identify all goal conditions that are not met in the current state.
    2. For each unachieved goal:
       a. Determine the type of goal (communicate soil, rock, or image data).
       b. If the required data (soil sample, rock sample, or image) is already
          held by any rover in the current state:
          - Estimate the cost as 1 (for the communicate action) plus the minimum
            navigation cost for that rover to reach any communication waypoint.
       c. If the required data is not held by any rover:
          - **For soil/rock data:**
            - Identify rovers equipped for the required analysis.
            - For each equipped rover, calculate the cost to:
              - Navigate from its current location to the sample waypoint.
              - Perform the sample action (cost 1, plus 1 if store is full and needs dropping).
              - Navigate from the sample waypoint to any communication waypoint.
              - Perform the communicate action (cost 1).
            - The estimated cost for this goal is the minimum of these costs over all equipped rovers.
          - **For image data:**
            - Identify rovers equipped for imaging with a camera supporting the required mode.
            - For each such rover and camera, calculate the cost to:
              - Navigate from its current location to a suitable calibration waypoint.
              - Perform the calibrate action (cost 1).
              - Navigate from the calibration waypoint to a suitable imaging waypoint.
              - Perform the take_image action (cost 1).
              - Navigate from the imaging waypoint to any communication waypoint.
              - Perform the communicate action (cost 1).
            - The estimated cost for this goal is the minimum of these costs over all
              suitable rover-camera pairs, calibration waypoints, imaging waypoints,
              and communication waypoints.
       d. If the minimum cost for a goal is infinite (e.g., no equipped rover,
          unreachable waypoint), the goal is considered unreachable from this state.
          The heuristic will sum infinities, resulting in infinity.
    3. The total heuristic value is the sum of the estimated minimum costs for all
       unachieved goals. If the sum is infinite, return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Static Information ---

        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.objectives = set()
        self.cameras = set()
        self.modes = set()
        self.stores = set()

        # Populate object sets from all facts (static, initial, goals)
        all_facts = set(static_facts) | set(task.initial_state) | set(task.goals)
        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander': self.landers.add(parts[1]); self.waypoints.add(parts[2])
            elif parts[0] == 'at' and len(parts) == 3: # (at ?rover ?waypoint)
                 if parts[1].startswith('rover'): self.rovers.add(parts[1]); self.waypoints.add(parts[2])
            elif parts[0] == 'can_traverse': self.rovers.add(parts[1]); self.waypoints.add(parts[2]); self.waypoints.add(parts[3])
            elif parts[0] == 'equipped_for_soil_analysis': self.rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis': self.rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging': self.rovers.add(parts[1])
            elif parts[0] == 'store_of': self.stores.add(parts[1]); self.rovers.add(parts[2])
            elif parts[0] == 'visible': self.waypoints.add(parts[1]); self.waypoints.add(parts[2])
            elif parts[0] == 'supports': self.cameras.add(parts[1]); self.modes.add(parts[2])
            elif parts[0] == 'visible_from': self.objectives.add(parts[1]); self.waypoints.add(parts[2])
            elif parts[0] == 'calibration_target': self.cameras.add(parts[1]); self.objectives.add(parts[2])
            elif parts[0] == 'on_board': self.cameras.add(parts[1]); self.rovers.add(parts[2])
            elif parts[0] == 'have_soil_analysis': self.rovers.add(parts[1]); self.waypoints.add(parts[2])
            elif parts[0] == 'have_rock_analysis': self.rovers.add(parts[1]); self.waypoints.add(parts[2])
            elif parts[0] == 'have_image': self.rovers.add(parts[1]); self.objectives.add(parts[2]); self.modes.add(parts[3])
            elif parts[0] == 'communicated_soil_data': self.waypoints.add(parts[1])
            elif parts[0] == 'communicated_rock_data': self.waypoints.add(parts[1])
            elif parts[0] == 'communicated_image_data': self.objectives.add(parts[1]); self.modes.add(parts[2])
            elif parts[0] == 'at_soil_sample': self.waypoints.add(parts[1])
            elif parts[0] == 'at_rock_sample': self.waypoints.add(parts[1])
            elif parts[0] == 'calibrated': self.cameras.add(parts[1]); self.rovers.add(parts[2])
            elif parts[0] == 'empty' or parts[0] == 'full': self.stores.add(parts[1])


        # Static relationships
        self.rover_capabilities = defaultdict(set)
        self.rover_to_store = {}
        self.rover_cameras = defaultdict(set)
        self.camera_modes = defaultdict(set)
        self.camera_calibration_target = {}
        self.objective_imaging_wps = defaultdict(set)
        self.calibration_target_wps = defaultdict(set) # Maps cal target objective to wps
        self.lander_location = None # Assuming one lander

        self.waypoint_visibility = defaultdict(set)
        self.rover_traversal_graph = defaultdict(lambda: defaultdict(set))

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'equipped_for_soil_analysis': self.rover_capabilities[parts[1]].add('soil')
            elif parts[0] == 'equipped_for_rock_analysis': self.rover_capabilities[parts[1]].add('rock')
            elif parts[0] == 'equipped_for_imaging': self.rover_capabilities[parts[1]].add('imaging')
            elif parts[0] == 'store_of': self.rover_to_store[parts[2]] = parts[1] # rover -> store
            elif parts[0] == 'on_board': self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
            elif parts[0] == 'supports': self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif parts[0] == 'calibration_target': self.camera_calibration_target[parts[1]] = parts[2] # camera -> cal_target
            elif parts[0] == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.objective_imaging_wps[obj].add(wp)
            elif parts[0] == 'at_lander': self.lander_location = parts[2] # Assuming one lander
            elif parts[0] == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility[wp1].add(wp2)
                self.waypoint_visibility[wp2].add(wp1) # Visibility is symmetric
            elif parts[0] == 'can_traverse':
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.rover_traversal_graph[rover][wp_from].add(wp_to)


        # Populate calibration_target_wps after camera_calibration_target is complete
        for camera, cal_target in self.camera_calibration_target.items():
             if cal_target in self.objective_imaging_wps:
                 self.calibration_target_wps[cal_target].update(self.objective_imaging_wps[cal_target])


        # Communication waypoints (visible from lander location)
        self.communication_wps = self.waypoint_visibility.get(self.lander_location, set())

        # Precompute shortest paths for each rover
        self.rover_dist = {}
        for rover in self.rovers:
            self.rover_dist[rover] = {}
            rover_graph = self.rover_traversal_graph.get(rover, {})
            # Ensure all known waypoints are considered in the BFS distance map
            for start_wp in self.waypoints:
                 # Run BFS from each waypoint
                 distances = bfs(start_wp, rover_graph, self.waypoints)
                 for end_wp in self.waypoints:
                     self.rover_dist[rover][(start_wp, end_wp)] = distances.get(end_wp, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        rover_locs = {}
        rover_store_full = {} # True if store is full
        rover_soil_samples = defaultdict(set) # rover -> {waypoint, ...}
        rover_rock_samples = defaultdict(set) # rover -> {waypoint, ...}
        rover_images = defaultdict(lambda: defaultdict(set)) # rover -> objective -> {mode, ...}
        rover_calibrated_cameras = defaultdict(set) # rover -> {camera, ...}
        soil_data_communicated = set() # {waypoint, ...}
        rock_data_communicated = set() # {waypoint, ...}
        image_data_communicated = defaultdict(set) # objective -> {mode, ...}
        soil_samples_at_wp = set() # {waypoint, ...}
        rock_samples_at_wp = set() # {waypoint, ...}


        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.rovers: rover_locs[parts[1]] = parts[2]
            elif parts[0] == 'full' and parts[1] in self.stores:
                 # Find which rover this store belongs to
                 store = parts[1]
                 if store in self.rover_to_store.values():
                     # Find rover by store
                     for r, s in self.rover_to_store.items():
                         if s == store:
                             rover_store_full[r] = True
                             break
            elif parts[0] == 'empty' and parts[1] in self.stores:
                 store = parts[1]
                 if store in self.rover_to_store.values():
                     # Find rover by store
                     for r, s in self.rover_to_store.items():
                         if s == store:
                             rover_store_full[r] = False
                             break
            elif parts[0] == 'have_soil_analysis': rover_soil_samples[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis': rover_rock_samples[parts[1]].add(parts[2])
            elif parts[0] == 'have_image': rover_images[parts[1]][parts[2]].add(parts[3])
            elif parts[0] == 'calibrated': rover_calibrated_cameras[parts[1]].add(parts[2])
            elif parts[0] == 'communicated_soil_data': soil_data_communicated.add(parts[1])
            elif parts[0] == 'communicated_rock_data': rock_data_communicated.add(parts[1])
            elif parts[0] == 'communicated_image_data': image_data_communicated[parts[1]].add(parts[2]) # parts[1] is objective, parts[2] is mode
            elif parts[0] == 'at_soil_sample': soil_samples_at_wp.add(parts[1])
            elif parts[0] == 'at_rock_sample': rock_samples_at_wp.add(parts[1])


        total_cost = 0

        # --- Estimate Cost for Unachieved Goals ---

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            goal_type = parts[0]

            if goal_type == 'communicated_soil_data':
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Case 1: Sample is already collected by some rover
                rover_with_sample = None
                for r, samples in rover_soil_samples.items():
                    if waypoint in samples:
                        rover_with_sample = r
                        break # Found one, take the first

                if rover_with_sample:
                    current_loc = rover_locs.get(rover_with_sample)
                    if current_loc:
                        min_comm_nav = float('inf')
                        if self.communication_wps:
                            min_comm_nav = min(self.rover_dist[rover_with_sample].get((current_loc, comm_wp), float('inf')) for comm_wp in self.communication_wps)

                        if min_comm_nav != float('inf'):
                            cost = min_comm_nav + 1 # +1 for communicate
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Sample needs to be collected and communicated
                # Only consider if not already covered by Case 1 (min_goal_cost is still inf)
                if min_goal_cost == float('inf'):
                    # Check if there is a soil sample at the waypoint
                    if waypoint in soil_samples_at_wp:
                        for rover in self.rovers:
                            if 'soil' in self.rover_capabilities.get(rover, set()):
                                current_loc = rover_locs.get(rover)
                                if current_loc:
                                    # Cost to get sample
                                    sample_nav_cost = self.rover_dist[rover].get((current_loc, waypoint), float('inf'))
                                    if sample_nav_cost != float('inf'):
                                        sample_cost = sample_nav_cost + 1 # +1 for sample
                                        if rover_store_full.get(rover, False): # Check if store is full
                                            sample_cost += 1 # +1 for drop

                                        # Cost to communicate from sample location
                                        min_comm_nav = float('inf')
                                        if self.communication_wps:
                                            min_comm_nav = min(self.rover_dist[rover].get((waypoint, comm_wp), float('inf')) for comm_wp in self.communication_wps)

                                        if min_comm_nav != float('inf'):
                                            cost = sample_cost + min_comm_nav + 1 # +1 for communicate
                                            min_goal_cost = min(min_goal_cost, cost)

            elif goal_type == 'communicated_rock_data':
                waypoint = parts[1]
                min_goal_cost = float('inf')

                # Case 1: Sample is already collected by some rover
                rover_with_sample = None
                for r, samples in rover_rock_samples.items():
                    if waypoint in samples:
                        rover_with_sample = r
                        break # Found one, take the first

                if rover_with_sample:
                    current_loc = rover_locs.get(rover_with_sample)
                    if current_loc:
                        min_comm_nav = float('inf')
                        if self.communication_wps:
                            min_comm_nav = min(self.rover_dist[rover_with_sample].get((current_loc, comm_wp), float('inf')) for comm_wp in self.communication_wps)

                        if min_comm_nav != float('inf'):
                            cost = min_comm_nav + 1 # +1 for communicate
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Sample needs to be collected and communicated
                # Only consider if not already covered by Case 1 (min_goal_cost is still inf)
                if min_goal_cost == float('inf'):
                     # Check if there is a rock sample at the waypoint
                    if waypoint in rock_samples_at_wp:
                        for rover in self.rovers:
                            if 'rock' in self.rover_capabilities.get(rover, set()):
                                current_loc = rover_locs.get(rover)
                                if current_loc:
                                    # Cost to get sample
                                    sample_nav_cost = self.rover_dist[rover].get((current_loc, waypoint), float('inf'))
                                    if sample_nav_cost != float('inf'):
                                        sample_cost = sample_nav_cost + 1 # +1 for sample
                                        if rover_store_full.get(rover, False): # Check if store is full
                                            sample_cost += 1 # +1 for drop

                                        # Cost to communicate from sample location
                                        min_comm_nav = float('inf')
                                        if self.communication_wps:
                                            min_comm_nav = min(self.rover_dist[rover].get((waypoint, comm_wp), float('inf')) for comm_wp in self.communication_wps)

                                        if min_comm_nav != float('inf'):
                                            cost = sample_cost + min_comm_nav + 1 # +1 for communicate
                                            min_goal_cost = min(min_goal_cost, cost)

            elif goal_type == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                min_goal_cost = float('inf')

                # Case 1: Image is already collected by some rover
                rover_with_image = None
                for r, obj_modes in rover_images.items():
                    if objective in obj_modes and mode in obj_modes[objective]:
                        rover_with_image = r
                        break # Found one, take the first

                if rover_with_image:
                    current_loc = rover_locs.get(rover_with_image)
                    if current_loc:
                        min_comm_nav = float('inf')
                        if self.communication_wps:
                            min_comm_nav = min(self.rover_dist[rover_with_image].get((current_loc, comm_wp), float('inf')) for comm_wp in self.communication_wps)

                        if min_comm_nav != float('inf'):
                            cost = min_comm_nav + 1 # +1 for communicate
                            min_goal_cost = min(min_goal_cost, cost)

                # Case 2: Image needs to be taken and communicated
                # Only consider if not already covered by Case 1 (min_goal_cost is still inf)
                if min_goal_cost == float('inf'):
                    for rover in self.rovers:
                        if 'imaging' in self.rover_capabilities.get(rover, set()):
                            current_loc = rover_locs.get(rover)
                            if current_loc:
                                for camera in self.rover_cameras.get(rover, set()):
                                    if mode in self.camera_modes.get(camera, set()):
                                        cal_target = self.camera_calibration_target.get(camera)
                                        if cal_target:
                                            cal_wps = self.calibration_target_wps.get(cal_target, set())
                                            img_wps = self.objective_imaging_wps.get(objective, set())

                                            min_path_nav_cost = float('inf')

                                            if cal_wps and img_wps and self.communication_wps:
                                                for w_cal in cal_wps:
                                                    for w_img in img_wps:
                                                        for comm_wp in self.communication_wps:
                                                            nav1 = self.rover_dist[rover].get((current_loc, w_cal), float('inf'))
                                                            nav2 = self.rover_dist[rover].get((w_cal, w_img), float('inf'))
                                                            nav3 = self.rover_dist[rover].get((w_img, comm_wp), float('inf'))

                                                            if nav1 != float('inf') and nav2 != float('inf') and nav3 != float('inf'):
                                                                path_nav_cost = nav1 + nav2 + nav3
                                                                min_path_nav_cost = min(min_path_nav_cost, path_nav_cost)

                                            if min_path_nav_cost != float('inf'):
                                                # +1 calibrate, +1 take_image, +1 communicate
                                                cost = min_path_nav_cost + 3
                                                min_goal_cost = min(min_goal_cost, cost)

            # Add the minimum cost for this goal to the total
            # If min_goal_cost is infinity, the total will become infinity
            total_cost += min_goal_cost

        # If the total cost is infinity, it means at least one goal is unreachable
        # In this case, return infinity. Otherwise, return the sum.
        return total_cost if total_cost != float('inf') else float('inf')
