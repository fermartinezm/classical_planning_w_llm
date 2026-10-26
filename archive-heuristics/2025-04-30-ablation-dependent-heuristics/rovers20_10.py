import sys
from collections import deque

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple of strings.
    e.g., '(at rover1 waypoint1)' -> ('at', 'rover1', 'waypoint1')
    """
    # Removes leading/trailing brackets and splits by space
    parts = fact_string.strip("()'").split()
    return tuple(parts)

# BFS for shortest paths
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    to all reachable nodes in a graph.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # If start_node is not in the graph, it's unreachable from anywhere in the graph
         # Return distances dictionary initialized with inf
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Should always be true here because of the initial check
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """
    Computes shortest path distances between all pairs of nodes in a graph
    using BFS starting from each node.
    """
    all_paths = {}
    # Collect all unique waypoints from graph keys and values
    all_waypoints = set(graph.keys())
    for neighbors in graph.values():
        all_waypoints.update(neighbors)

    for start_node in all_waypoints:
        paths_from_start = bfs(graph, start_node)
        for end_node, dist in paths_from_start.items():
            if dist != float('inf'):
                 all_paths[(start_node, end_node)] = dist
    return all_paths


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each unsatisfied goal condition. The cost for each goal condition
    is estimated based on the actions and navigation required to achieve it,
    considering the most efficient rover/path for each sub-task independently.
    """

    def __init__(self, task):
        """
        Heuristic Initialization:
        Processes static information from the task to build data structures
        used for heuristic calculation.

        - Parses static facts to identify lander location, rover capabilities,
          store mappings, camera information, initial sample locations,
          objective visibility, and the waypoint graph based on 'visible' links.
        - Computes all-pairs shortest paths between waypoints using BFS on the
          waypoint graph.
        - Identifies waypoints visible from the lander's location.
        """
        self.task = task
        self.static_facts = set(parse_fact(f) for f in task.static)

        # Extract static information
        self.lander_location = None
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.store_map = {} # store -> rover
        self.camera_info = {} # camera -> {'rover': rover, 'modes': set(), 'target': objective}
        self.initial_soil_samples = set() # waypoint
        self.initial_rock_samples = set() # waypoint
        self.objective_visibility = {} # objective -> set of waypoints visible from objective

        # Waypoint graph for navigation (assuming visible implies traversable for all rovers)
        self.waypoint_graph = {} # waypoint -> set of connected waypoints

        # Collect all waypoints mentioned in static facts to build the graph nodes
        all_waypoints_mentioned = set()

        for fact in self.static_facts:
            if fact[0] == 'at_lander':
                self.lander_location = fact[2]
                all_waypoints_mentioned.add(fact[2])
            elif fact[0] == 'equipped_for_soil_analysis':
                rover = fact[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif fact[0] == 'equipped_for_rock_analysis':
                rover = fact[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif fact[0] == 'equipped_for_imaging':
                rover = fact[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif fact[0] == 'store_of':
                store, rover = fact[1], fact[2]
                self.store_map[store] = rover
            elif fact[0] == 'on_board':
                camera, rover = fact[1], fact[2]
                self.camera_info.setdefault(camera, {})['rover'] = rover
            elif fact[0] == 'supports':
                camera, mode = fact[1], fact[2]
                self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
            elif fact[0] == 'calibration_target':
                camera, target = fact[1], fact[2]
                self.camera_info.setdefault(camera, {})['target'] = target
            elif fact[0] == 'at_soil_sample':
                 self.initial_soil_samples.add(fact[1])
                 all_waypoints_mentioned.add(fact[1])
            elif fact[0] == 'at_rock_sample':
                 self.initial_rock_samples.add(fact[1])
                 all_waypoints_mentioned.add(fact[1])
            elif fact[0] == 'visible_from':
                objective, waypoint = fact[1], fact[2]
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
                all_waypoints_mentioned.add(waypoint)
            elif fact[0] == 'visible':
                wp1, wp2 = fact[1], fact[2]
                self.waypoint_graph.setdefault(wp1, set()).add(wp2)
                self.waypoint_graph.setdefault(wp2, set()).add(wp1) # Assuming visible is symmetric
                all_waypoints_mentioned.add(wp1)
                all_waypoints_mentioned.add(wp2)
            elif fact[0] == 'can_traverse': # Although we assume visible implies traversable,
                                            # collect all waypoints mentioned in can_traverse too
                 all_waypoints_mentioned.add(fact[2])
                 all_waypoints_mentioned.add(fact[3])
            elif fact[0] == 'at': # Collect initial rover locations
                 all_waypoints_mentioned.add(fact[2])


        # Ensure all mentioned waypoints are in the graph keys, even if isolated
        for wp in all_waypoints_mentioned:
             self.waypoint_graph.setdefault(wp, set())

        # Compute shortest paths
        self.shortest_paths = compute_all_pairs_shortest_paths(self.waypoint_graph)

        # Identify waypoints visible from the lander
        self.lander_visible_waypoints = set()
        if self.lander_location:
             # Find all waypoints directly visible from lander_location
             if self.lander_location in self.waypoint_graph:
                 self.lander_visible_waypoints.update(self.waypoint_graph[self.lander_location])
             # Also the lander location itself might be a communication point
             self.lander_visible_waypoints.add(self.lander_location)


    def __call__(self, state):
        """
        Step-By-Step Thinking for Computing Heuristic:

        1.  Check if the current state is a goal state. If yes, return 0.
        2.  Initialize total heuristic cost to 0.
        3.  Parse the current state to extract dynamic information:
            - Rover locations.
            - Current soil/rock samples present at waypoints.
            - Collected soil/rock/image data held by rovers.
            - Calibrated cameras on rovers.
            - Full stores on rovers.
        4.  Iterate through each goal fact in task.goals:
            - If the goal fact is already true in the current state, skip it.
            - If the goal is (communicated_soil_data ?w):
                - Estimate cost to get soil data at ?w:
                    - If any rover currently has (have_soil_analysis ?r ?w), cost_get_data = 0.
                    - Otherwise, need to sample. Check if (at_soil_sample ?w) is in the current state.
                        - If yes: Find equipped rover closest to ?w. cost_get_data = min_nav_cost + 1 (sample). Add penalty if no equipped rover can reach ?w.
                        - If no: The sample is gone, and no rover has the data. This goal is likely unachievable via sampling. cost_get_data = LARGE_PENALTY.
                - Estimate cost to communicate data for ?w:
                    - Find rovers that currently have the data OR are capable of getting it (equipped and sample is present).
                    - Find lander-visible waypoint closest to such a rover. cost_communicate = min_nav_cost + 1 (communicate). Add penalty if no suitable rover can reach a lander-visible waypoint.
                - Add the sum of estimated costs (cost_get_data + cost_communicate) for this goal to the total heuristic cost.
            - If the goal is (communicated_rock_data ?w):
                - Similar logic as soil data, using rock predicates.
            - If the goal is (communicated_image_data ?o ?m):
                - Estimate cost to get image data for ?o in mode ?m:
                    - If any rover currently has (have_image ?r ?o ?m), cost_get_data = 0.
                    - Otherwise, need to take image. Find equipped rover with suitable camera.
                        - Cost to take image = min_nav_cost to a waypoint visible from ?o + 1 (take_image). Add penalty if no suitable waypoint is reachable.
                        - Cost to calibrate = 0 if camera is calibrated. Otherwise, min_nav_cost to a waypoint visible from camera's target + 1 (calibrate). Add penalty if no calibration waypoint is reachable.
                        - cost_get_data = min over suitable rover/camera combinations (Cost to take image + Cost to calibrate). Add penalty if no suitable combination exists or can reach required locations.
                - Estimate cost to communicate data for ?o ?m:
                    - Find rovers that currently have the image OR are capable of getting it (equipped, suitable camera, objective visible from reachable waypoint).
                    - Find lander-visible waypoint closest to such a rover. cost_communicate = min_nav_cost + 1 (communicate). Add penalty if no suitable rover can reach a lander-visible waypoint.
                - Add the sum of estimated costs (cost_get_data + cost_communicate) for this goal to the total heuristic cost.
        5.  Return the total heuristic cost. If the total cost is 0 but the state is not a goal state, return 1.

        Assumptions:
        - 'visible' predicate implies 'can_traverse' for all rovers for navigation cost calculation.
        - The 'drop' action cost is ignored when estimating sampling cost.
        - If a sample (soil/rock) is no longer at its initial location and no rover currently holds the corresponding data, the goal requiring that data is considered very expensive (penalty).
        - Calibration targets are objectives, and their visibility is given by 'visible_from'.
        - The cost to get data (sampling/imaging) and the cost to communicate data are estimated independently for each goal, picking the best rover(s) for each step based on their *current* location and capabilities. The total cost for a goal is the sum of these two independent costs.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Parse dynamic state information
        current_facts = set(parse_fact(f) for f in state)
        rover_locations = {} # rover -> waypoint
        current_soil_samples = set() # waypoint
        current_rock_samples = set() # waypoint
        current_soil_data = set() # (rover, waypoint)
        current_rock_data = set() # (rover, waypoint)
        current_image_data = set() # (rover, objective, mode)
        current_calibrated_cameras = set() # (camera, rover)
        current_full_stores = set() # store

        for fact in current_facts:
            if fact[0] == 'at':
                rover_locations[fact[1]] = fact[2]
            elif fact[0] == 'at_soil_sample':
                current_soil_samples.add(fact[1])
            elif fact[0] == 'at_rock_sample':
                current_rock_samples.add(fact[1])
            elif fact[0] == 'have_soil_analysis':
                current_soil_data.add((fact[1], fact[2]))
            elif fact[0] == 'have_rock_analysis':
                current_rock_data.add((fact[1], fact[2]))
            elif fact[0] == 'have_image':
                current_image_data.add((fact[1], fact[2], fact[3]))
            elif fact[0] == 'calibrated':
                current_calibrated_cameras.add((fact[1], fact[2]))
            elif fact[0] == 'full':
                current_full_stores.add(fact[1])

        total_heuristic_cost = 0
        LARGE_PENALTY = 1000 # Use a large integer for unreachable sub-goals

        # Process unsatisfied goals
        for goal_fact_str in self.task.goals:
            if goal_fact_str in state:
                continue # Goal already satisfied

            goal_fact = parse_fact(goal_fact_str)
            fact_type = goal_fact[0]

            if fact_type == 'communicated_soil_data':
                goal_wp = goal_fact[1]
                cost_get_data_step = 0
                cost_comm_step = 0

                # Part 1: Cost to get the soil sample data
                has_data_now = any((r, goal_wp) in current_soil_data for r in self.rover_capabilities)

                if not has_data_now:
                    # Need to sample
                    min_sample_nav_cost = float('inf')
                    candidate_samplers_reachable = False

                    # Find equipped rovers that can reach the sample location (if sample is there)
                    if goal_wp in current_soil_samples: # Check if sample is currently available
                        for rover, caps in self.rover_capabilities.items():
                            if 'soil' in caps:
                                current_wp = rover_locations.get(rover)
                                if current_wp:
                                    nav_cost = self.shortest_paths.get((current_wp, goal_wp), float('inf'))
                                    if nav_cost != float('inf'):
                                        min_sample_nav_cost = min(min_sample_nav_cost, nav_cost)
                                        candidate_samplers_reachable = True

                    if candidate_samplers_reachable:
                        cost_get_data_step = min_sample_nav_cost + 1 # navigate + sample
                        # Note: Ignoring store full / drop action cost
                    else:
                        # Sample needed, but no equipped rover can reach it OR sample is gone
                        cost_get_data_step = LARGE_PENALTY # Penalty

                # Part 2: Cost to communicate the data
                min_comm_nav_cost_step = float('inf')
                candidate_communicators_exist = False

                # Find rovers that currently have the data OR are capable of getting it
                candidate_rovers_for_comm = set()
                for r, w in current_soil_data:
                    if w == goal_wp:
                        candidate_rovers_for_comm.add(r)
                # Add rovers equipped for soil if sample is present (they could get it)
                if goal_wp in current_soil_samples:
                     for rover, caps in self.rover_capabilities.items():
                         if 'soil' in caps:
                             candidate_rovers_for_comm.add(rover)


                if candidate_rovers_for_comm:
                    candidate_communicators_exist = True
                    for rover in candidate_rovers_for_comm:
                        current_wp = rover_locations.get(rover)
                        if current_wp:
                            min_dist_to_lander_visible = float('inf')
                            for lander_wp in self.lander_visible_waypoints:
                                 dist = self.shortest_paths.get((current_wp, lander_wp), float('inf'))
                                 min_dist_to_lander_visible = min(min_dist_to_lander_visible, dist)

                            if min_dist_to_lander_visible != float('inf'):
                                 min_comm_nav_cost_step = min(min_comm_nav_cost_step, min_dist_to_lander_visible)

                if candidate_communicators_exist and min_comm_nav_cost_step != float('inf'):
                     cost_comm_step = min_comm_nav_cost_step + 1
                else:
                     cost_comm_step = LARGE_PENALTY # No rover can reach a communication point

                total_heuristic_cost += cost_get_data_step + cost_comm_step

            elif fact_type == 'communicated_rock_data':
                goal_wp = goal_fact[1]
                cost_get_data_step = 0
                cost_comm_step = 0

                # Part 1: Cost to get the rock sample data
                has_data_now = any((r, goal_wp) in current_rock_data for r in self.rover_capabilities)

                if not has_data_now:
                    # Need to sample
                    min_sample_nav_cost = float('inf')
                    candidate_samplers_reachable = False

                    # Find equipped rovers that can reach the sample location (if sample is there)
                    if goal_wp in current_rock_samples: # Check if sample is currently available
                        for rover, caps in self.rover_capabilities.items():
                            if 'rock' in caps:
                                current_wp = rover_locations.get(rover)
                                if current_wp:
                                    nav_cost = self.shortest_paths.get((current_wp, goal_wp), float('inf'))
                                    if nav_cost != float('inf'):
                                        min_sample_nav_cost = min(min_sample_nav_cost, nav_cost)
                                        candidate_samplers_reachable = True

                    if candidate_samplers_reachable:
                        cost_get_data_step = min_sample_nav_cost + 1 # navigate + sample
                        # Note: Ignoring store full / drop action cost
                    else:
                        # Sample needed, but no equipped rover can reach it OR sample is gone
                        cost_get_data_step = LARGE_PENALTY # Penalty

                # Part 2: Cost to communicate the data
                min_comm_nav_cost_step = float('inf')
                candidate_communicators_exist = False

                # Find rovers that currently have the data OR are capable of getting it
                candidate_rovers_for_comm = set()
                for r, w in current_rock_data:
                    if w == goal_wp:
                        candidate_rovers_for_comm.add(r)
                # Add rovers equipped for rock if sample is present (they could get it)
                if goal_wp in current_rock_samples:
                     for rover, caps in self.rover_capabilities.items():
                         if 'rock' in caps:
                             candidate_rovers_for_comm.add(rover)

                if candidate_rovers_for_comm:
                    candidate_communicators_exist = True
                    for rover in candidate_rovers_for_comm:
                        current_wp = rover_locations.get(rover)
                        if current_wp:
                            min_dist_to_lander_visible = float('inf')
                            for lander_wp in self.lander_visible_waypoints:
                                 dist = self.shortest_paths.get((current_wp, lander_wp), float('inf'))
                                 min_dist_to_lander_visible = min(min_dist_to_lander_visible, dist)

                            if min_dist_to_lander_visible != float('inf'):
                                 min_comm_nav_cost_step = min(min_comm_nav_cost_step, min_dist_to_lander_visible)

                if candidate_communicators_exist and min_comm_nav_cost_step != float('inf'):
                     cost_comm_step = min_comm_nav_cost_step + 1
                else:
                     cost_comm_step = LARGE_PENALTY # No rover can reach a communication point

                total_heuristic_cost += cost_get_data_step + cost_comm_step


            elif fact_type == 'communicated_image_data':
                goal_obj = goal_fact[1]
                goal_mode = goal_fact[2]
                cost_get_data_step = 0
                cost_comm_step = 0

                # Part 1: Cost to get the image data
                has_data_now = any((r, goal_obj, goal_mode) in current_image_data for r in self.rover_capabilities)

                if not has_data_now:
                    # Need to take image
                    min_image_step_cost = float('inf')
                    candidate_imagers_reachable = False

                    # Find suitable rover/camera combinations that can reach required locations
                    for rover, caps in self.rover_capabilities.items():
                        if 'imaging' in caps:
                            current_wp = rover_locations.get(rover)
                            if current_wp:
                                # Find cameras on this rover supporting the mode
                                suitable_cameras = [cam for cam, info in self.camera_info.items()
                                                    if info.get('rover') == rover and goal_mode in info.get('modes', set())]

                                for camera in suitable_cameras:
                                    # Cost to take image action + navigation
                                    min_take_image_nav = float('inf')
                                    visible_wps = self.objective_visibility.get(goal_obj, set())
                                    for visible_wp in visible_wps:
                                        dist = self.shortest_paths.get((current_wp, visible_wp), float('inf'))
                                        min_take_image_nav = min(min_take_image_nav, dist)

                                    if min_take_image_nav == float('inf'):
                                        # Cannot reach any waypoint visible from objective with this rover/camera
                                        continue # Try next camera/rover

                                    current_combo_step_cost = min_take_image_nav + 1 # navigate + take_image

                                    # Cost to calibrate action + navigation (if needed)
                                    if (camera, rover) not in current_calibrated_cameras:
                                        # Need to calibrate
                                        cal_target = self.camera_info.get(camera, {}).get('target')
                                        if cal_target:
                                            min_cal_nav = float('inf')
                                            cal_visible_wps = self.objective_visibility.get(cal_target, set())
                                            for cal_visible_wp in cal_visible_wps:
                                                dist = self.shortest_paths.get((current_wp, cal_visible_wp), float('inf'))
                                                min_cal_nav = min(min_cal_nav, dist)

                                            if min_cal_nav == float('inf'):
                                                # Cannot reach any waypoint visible from calibration target
                                                current_combo_step_cost = float('inf') # This combo is blocked
                                            else:
                                                current_combo_step_cost += min_cal_nav + 1 # navigate + calibrate
                                        else:
                                            # Camera has no calibration target defined? Should not happen in valid PDDL.
                                            current_combo_step_cost = float('inf') # This combo is blocked
                                    else:
                                        # Camera is already calibrated, cost is 0 for calibration step
                                        pass # cost_calibrate_step = 0

                                    # Update minimum cost over all valid combos
                                    min_image_step_cost = min(min_image_step_cost, current_combo_step_cost)
                                    if current_combo_step_cost != float('inf'):
                                         candidate_imagers_reachable = True


                    if candidate_imagers_reachable and min_image_step_cost != float('inf'):
                         cost_get_data_step = min_image_step_cost
                    else:
                         # No suitable rover/camera can get the image or reach locations
                         cost_get_data_step = LARGE_PENALTY # Penalty

                # Part 2: Cost to communicate the data
                min_comm_nav_cost_step = float('inf')
                candidate_communicators_exist = False

                # Find rovers that currently have the data OR are capable of getting it
                candidate_rovers_for_comm = set()
                for r, o, m in current_image_data:
                    if o == goal_obj and m == goal_mode:
                        candidate_rovers_for_comm.add(r)
                # Add rovers equipped for imaging with suitable camera (they could get it)
                for rover, caps in self.rover_capabilities.items():
                    if 'imaging' in caps:
                        suitable_cameras = [cam for cam, info in self.camera_info.items()
                                            if info.get('rover') == rover and goal_mode in info.get('modes', set())]
                        if suitable_cameras: # If rover has at least one suitable camera
                             # Check if objective is visible from any reachable waypoint for this rover
                             current_wp = rover_locations.get(rover)
                             if current_wp:
                                 visible_wps = self.objective_visibility.get(goal_obj, set())
                                 if any(self.shortest_paths.get((current_wp, p), float('inf')) != float('inf') for p in visible_wps):
                                     candidate_rovers_for_comm.add(rover)


                if candidate_rovers_for_comm:
                    candidate_communicators_exist = True
                    for rover in candidate_rovers_for_comm:
                        current_wp = rover_locations.get(rover)
                        if current_wp:
                            min_dist_to_lander_visible = float('inf')
                            for lander_wp in self.lander_visible_waypoints:
                                 dist = self.shortest_paths.get((current_wp, lander_wp), float('inf'))
                                 min_dist_to_lander_visible = min(min_dist_to_lander_visible, dist)

                            if min_dist_to_lander_visible != float('inf'):
                                 min_comm_nav_cost_step = min(min_comm_nav_cost_step, min_dist_to_lander_visible)

                if candidate_communicators_exist and min_comm_nav_cost_step != float('inf'):
                     cost_comm_step = min_comm_nav_cost_step + 1
                else:
                     cost_comm_step = LARGE_PENALTY # No rover can reach a communication point

                total_heuristic_cost += cost_get_data_step + cost_comm_step

            # Add other goal types if any (not present in domain file)
            # ...

        # Ensure heuristic is 0 only at goal state.
        # If total_heuristic_cost is 0 but it's not a goal state, return 1.
        # If total_heuristic_cost is very large (due to penalties), return a large int.
        if total_heuristic_cost == 0 and not self.task.goal_reached(state):
             return 1
        elif total_heuristic_cost >= LARGE_PENALTY:
             # Cap the large penalty to avoid overflow or issues, but keep it large
             # Scale penalty by number of goals to differentiate between 1 impossible goal vs many
             return LARGE_PENALTY * len(self.task.goals)
        else:
             return int(total_heuristic_cost) # Ensure integer return

