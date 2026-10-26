from collections import deque, defaultdict
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple of strings.
    e.g., '(at rover1 waypoint1)' -> ('at', 'rover1', 'waypoint1')
    """
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function for Breadth-First Search (BFS)
def bfs(graph, start_node):
    """
    Performs BFS starting from start_node on the given graph to find shortest
    distances to all reachable nodes.

    Args:
        graph: A dictionary representing the graph (waypoint -> set of neighbors).
        start_node: The starting waypoint for the BFS.

    Returns:
        A dictionary mapping each reachable waypoint to its shortest distance
        from the start_node. Unreachable waypoints have distance float('inf').
    """
    distances = {node: float('inf') for node in graph}
    # Ensure start_node is a valid key in the graph
    if start_node not in graph:
         # This should ideally not happen if the graph is built correctly
         # including all possible waypoints, even isolated ones.
         return distances

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # current_node must be in graph if it was added to the queue from graph keys
        for neighbor in graph[current_node]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """
    Computes shortest paths between all pairs of nodes in the graph using BFS
    from each node.

    Args:
        graph: A dictionary representing the graph (waypoint -> set of neighbors).

    Returns:
        A dictionary mapping start_waypoint -> end_waypoint -> distance.
    """
    all_paths = {}
    # Collect all nodes that are keys in the graph or appear as neighbors
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Create a complete graph representation including all nodes, even if isolated
    # This ensures BFS is run for every potential start node.
    full_graph = {node: graph.get(node, set()) for node in all_nodes}

    for start_node in all_nodes:
        all_paths[start_node] = bfs(full_graph, start_node)
    return all_paths


class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        estimated costs for each unachieved goal predicate. For each unachieved
        goal (communicating soil data, rock data, or image data), it calculates
        the minimum cost required by any suitable rover to achieve that specific
        goal. The cost includes navigation steps (estimated by shortest path
        distance) and action steps (each counted as 1). Intermediate steps like
        sampling or taking an image are included if the required data/image is
        not yet collected. Resource constraints (like store capacity beyond full/empty)
        and camera calibration state persistence are simplified.

    Assumptions:
        - The problem is solvable (unless the heuristic returns infinity).
        - Any rover with the required equipment can perform the necessary actions.
        - Navigation cost between waypoints is the shortest path distance on the
          rover's traverse graph.
        - Each action (sample, drop, calibrate, take_image, communicate) costs 1.
        - The heuristic ignores complex interactions like multiple rovers needing
          the same resource or path conflicts.
        - Initial sample locations relevant to goals are present in the initial state.
        - Each rover has at most one store relevant for sampling.

    Heuristic Initialization:
        The constructor precomputes static information from the task definition
        and initial state:
        - Lander locations.
        - Rover equipment capabilities.
        - Store ownership for rovers.
        - Camera capabilities (supported modes, calibration targets, rover ownership).
        - Objective visibility from waypoints.
        - Initial locations of soil and rock samples (relevant for goal checks).
        - Rover-specific navigation graphs based on 'can_traverse'.
        - All-pairs shortest paths for each rover on its navigation graph using BFS.
        - Communication waypoints (visible from any lander location).

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize total heuristic value `h = 0`.
        2. Parse the current state to get dynamic information:
           - Current location of each rover.
           - Full/empty status of each store.
           - Collected samples (`have_soil_analysis`, `have_rock_analysis`).
           - Collected images (`have_image`).
           - Calibrated cameras (`calibrated`).
        3. Iterate through each goal predicate in `task.goals`.
        4. If a goal predicate `g` is already true in the current state, continue to the next goal.
        5. If `g` is not true:
           a. Determine the type of goal (soil, rock, or image) and the specific target (waypoint for samples, objective/mode for images).
           b. Identify the set of rovers that are *equipped* to potentially achieve this goal type.
           c. Initialize `min_rover_cost_for_goal = float('inf')`.
           d. For each equipped rover `r`:
              i. Initialize `current_rover_goal_cost = 0`.
              ii. Determine the rover's current position `current_pos`. If the rover's current position is not known (e.g., not in the state facts), this rover cannot contribute, continue to the next rover.
              iii. Determine the rover's position *after* completing any necessary intermediate steps (sampling or imaging). Initialize `pos_after_intermediate_steps = current_pos`.
              iv. If the goal requires collecting a sample (soil or rock) that the rover `r` does not yet `have_soil_analysis` or `have_rock_analysis` for:
                  - Calculate the cost to navigate to the sample location, potentially drop a full store, and sample. This involves looking up shortest path distances.
                  - Add this cost to `current_rover_goal_cost`.
                  - Update `pos_after_intermediate_steps` to the sample location.
                  - If the sample location is unreachable, mark this rover as unable to achieve the goal and break from the rover loop for this goal.
              v. If the goal requires taking an image that the rover `r` does not yet `have_image` for:
                  - Find suitable cameras on the rover for the required mode.
                  - For each suitable camera, find calibration waypoints and image waypoints.
                  - Calculate the minimum navigation cost to go from `current_pos` to a calibration waypoint and then to an image waypoint.
                  - Add the minimum navigation cost + 1 (calibrate) + 1 (take_image) to `current_rover_goal_cost`.
                  - Update `pos_after_intermediate_steps` to the best image waypoint found.
                  - If no suitable camera or path exists, mark this rover as unable to achieve the goal and break from the rover loop for this goal.
              vi. If the rover was not marked as unable:
                  - Calculate the cost to communicate the data/image. This involves finding the minimum navigation cost from `pos_after_intermediate_steps` to any communication waypoint.
                  - Add the minimum navigation cost + 1 (communicate) to `current_rover_goal_cost`.
                  - If no communication waypoint is reachable, mark this rover as unable to achieve the goal and break from the rover loop for this goal.
              vii. Update `min_rover_cost_for_goal = min(min_rover_cost_for_goal, rover_goal_cost)`.
           e. If `min_rover_cost_for_goal` is still `float('inf')` after checking all equipped rovers, the goal is unreachable. Return `float('inf')` for the total heuristic.
           f. Otherwise, add `min_rover_cost_for_goal` to `h`.
        6. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.task = task
        self.static_facts = task.static
        self.initial_state = task.initial_state
        self.goals = task.goals

        # Precompute static information
        self.lander_locations = set()
        self.rover_equipment = defaultdict(set)
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': r, 'modes': set(), 'calibration_target': t}
        self.objective_visibility = defaultdict(set) # objective -> set of visible waypoints
        self.waypoint_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> set of neighbors
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        self.visible_waypoints = defaultdict(set) # waypoint -> set of visible waypoints

        # Extract static and initial state information
        all_waypoints = set()
        all_rovers = set()

        # Collect all waypoints and rovers mentioned in static/initial facts
        for fact_string in self.static_facts | self.initial_state:
             fact = parse_fact(fact_string)
             # Collect objects that are waypoints or rovers
             for item in fact[1:]:
                 if item.startswith('waypoint'):
                     all_waypoints.add(item)
                 elif item.startswith('rover'):
                     all_rovers.add(item)

        for fact_string in self.static_facts | self.initial_state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at_lander':
                self.lander_locations.add(fact[2]) # (at_lander lander waypoint)
            elif fact[0] == 'equipped_for_soil_analysis':
                self.rover_equipment[fact[1]].add('soil')
            elif fact[0] == 'equipped_for_rock_analysis':
                self.rover_equipment[fact[1]].add('rock')
            elif fact[0] == 'equipped_for_imaging':
                self.rover_equipment[fact[1]].add('imaging')
            elif fact[0] == 'store_of':
                self.rover_stores[fact[2]] = fact[1] # (store_of store rover)
            elif fact[0] == 'supports':
                camera, mode = fact[1], fact[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'calibration_target': None}
                self.camera_info[camera]['modes'].add(mode)
            elif fact[0] == 'calibration_target':
                camera, objective = fact[1], fact[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'calibration_target': None}
                self.camera_info[camera]['calibration_target'] = objective
            elif fact[0] == 'on_board':
                camera, rover = fact[1], fact[2]
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'rover': None, 'modes': set(), 'calibration_target': None}
                self.camera_info[camera]['rover'] = rover
            elif fact[0] == 'visible_from':
                objective, waypoint = fact[1], fact[2]
                self.objective_visibility[objective].add(waypoint)
            elif fact[0] == 'can_traverse':
                 rover, wp1, wp2 = fact[1], fact[2], fact[3]
                 self.waypoint_graph[rover][wp1].add(wp2)
            elif fact[0] == 'visible':
                 wp1, wp2 = fact[1], fact[2]
                 self.visible_waypoints[wp1].add(wp2)
                 self.visible_waypoints[wp2].add(wp1) # Assuming visible is symmetric for communication purposes
            elif fact[0] == 'at_soil_sample' and fact_string in self.initial_state:
                 self.initial_soil_samples.add(fact[1])
            elif fact[0] == 'at_rock_sample' and fact_string in self.initial_state:
                 self.initial_rock_samples.add(fact[1])

        # Ensure all waypoints are nodes in the graph for each rover, even if isolated
        for rover in all_rovers:
             if rover not in self.waypoint_graph:
                 self.waypoint_graph[rover] = defaultdict(set)
             for wp in all_waypoints:
                 if wp not in self.waypoint_graph[rover]:
                     self.waypoint_graph[rover][wp] = set()


        # Compute all-pairs shortest paths for each rover
        self.shortest_paths = {}
        for rover, graph in self.waypoint_graph.items():
             self.shortest_paths[rover] = compute_all_pairs_shortest_paths(graph)

        # Identify communication waypoints (visible from any lander location)
        self.communication_waypoints = set()
        for lander_loc in self.lander_locations:
            # Find waypoints x such that (visible x lander_loc)
            # We built visible_waypoints symmetrically, so just check neighbors of lander_loc
            self.communication_waypoints.update(self.visible_waypoints.get(lander_loc, set()))


    def __call__(self, state):
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Parse dynamic information from the current state
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil = defaultdict(set) # rover -> set of waypoints
        have_rock = defaultdict(set) # rover -> set of waypoints
        have_image = defaultdict(lambda: defaultdict(set)) # rover -> objective -> set of modes
        # calibrated_cameras = set() # Not strictly needed for this heuristic's logic

        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at' and fact[1].startswith('rover'):
                rover_locations[fact[1]] = fact[2]
            elif fact[0] == 'empty':
                store_status[fact[1]] = 'empty'
            elif fact[0] == 'full':
                store_status[fact[1]] = 'full'
            elif fact[0] == 'have_soil_analysis':
                have_soil[fact[1]].add(fact[2])
            elif fact[0] == 'have_rock_analysis':
                have_rock[fact[1]].add(fact[2])
            elif fact[0] == 'have_image':
                rover, objective, mode = fact[1], fact[2], fact[3]
                have_image[rover][objective].add(mode)
            # elif fact[0] == 'calibrated': # Not used in this heuristic's cost calculation logic
            #     calibrated_cameras.add(fact[1])

        total_heuristic = 0

        # Iterate through goal predicates
        for goal_fact_string in self.goals:
            if goal_fact_string in state:
                continue # Goal already achieved

            goal_fact = parse_fact(goal_fact_string)
            goal_type = goal_fact[0]

            min_rover_cost_for_goal = float('inf')

            if goal_type == 'communicated_soil_data':
                waypoint = goal_fact[1]
                # Check if sample was initially at this waypoint (goal is tied to initial location)
                if waypoint not in self.initial_soil_samples:
                     # This goal is impossible if the sample wasn't there initially
                     return float('inf') # Goal unreachable

                # Find rovers equipped for soil analysis
                equipped_rovers = [r for r, equip in self.rover_equipment.items() if 'soil' in equip]

                if not equipped_rovers:
                    return float('inf') # No rover can achieve this goal

                for rover in equipped_rovers:
                    if rover not in rover_locations:
                        # Rover not present in the current state? Should not happen in valid states.
                        # Treat as unable to contribute to this goal from this state.
                        continue

                    current_pos = rover_locations[rover]
                    rover_goal_cost = 0
                    pos_after_intermediate_steps = current_pos # Location after getting sample

                    # Step 1: Get the sample if not already had by this rover
                    if waypoint not in have_soil[rover]:
                        # Cost to navigate to sample location
                        if waypoint not in self.shortest_paths[rover][current_pos] or \
                           self.shortest_paths[rover][current_pos][waypoint] == float('inf'):
                             # Cannot reach sample location with this rover
                             continue # Try next rover

                        sample_nav_cost = self.shortest_paths[rover][current_pos][waypoint]
                        rover_goal_cost += sample_nav_cost

                        # Cost to drop if store is full
                        store = self.rover_stores.get(rover) # Get store for this rover
                        # Check if the store exists and is full
                        if store and store_status.get(store) == 'full':
                            rover_goal_cost += 1 # drop action

                        # Cost to sample
                        rover_goal_cost += 1 # sample_soil action
                        pos_after_intermediate_steps = waypoint # Rover is now at sample location

                    # Step 2: Communicate the data
                    comm_cost = float('inf')
                    best_comm_nav_cost = float('inf')
                    # Find min distance from pos_after_intermediate_steps to any communication waypoint
                    for comm_wp in self.communication_waypoints:
                        if comm_wp in self.shortest_paths[rover][pos_after_intermediate_steps] and \
                           self.shortest_paths[rover][pos_after_intermediate_steps][comm_wp] != float('inf'):
                             best_comm_nav_cost = min(best_comm_nav_cost, self.shortest_paths[rover][pos_after_intermediate_steps][comm_wp])

                    if best_comm_nav_cost == float('inf'):
                         # Cannot reach any communication waypoint from current/acquired location
                         continue # Try next rover

                    rover_goal_cost += best_comm_nav_cost # Navigation to communication waypoint
                    rover_goal_cost += 1 # communicate_soil_data action

                    min_rover_cost_for_goal = min(min_rover_cost_for_goal, rover_goal_cost)

            elif goal_type == 'communicated_rock_data':
                waypoint = goal_fact[1]
                # Check if sample was initially at this waypoint
                if waypoint not in self.initial_rock_samples:
                     return float('inf') # Goal unreachable

                # Find rovers equipped for rock analysis
                equipped_rovers = [r for r, equip in self.rover_equipment.items() if 'rock' in equip]

                if not equipped_rovers:
                    return float('inf') # No rover can achieve this goal

                for rover in equipped_rovers:
                    if rover not in rover_locations: continue

                    current_pos = rover_locations[rover]
                    rover_goal_cost = 0
                    pos_after_intermediate_steps = current_pos # Location after getting sample

                    # Step 1: Get the sample if not already had by this rover
                    if waypoint not in have_rock[rover]:
                        # Cost to navigate to sample location
                        if waypoint not in self.shortest_paths[rover][current_pos] or \
                           self.shortest_paths[rover][current_pos][waypoint] == float('inf'):
                             continue # Try next rover

                        sample_nav_cost = self.shortest_paths[rover][current_pos][waypoint]
                        rover_goal_cost += sample_nav_cost

                        # Cost to drop if store is full
                        store = self.rover_stores.get(rover)
                        if store and store_status.get(store) == 'full':
                            rover_goal_cost += 1 # drop action

                        # Cost to sample
                        rover_goal_cost += 1 # sample_rock action
                        pos_after_intermediate_steps = waypoint # Rover is now at sample location

                    # Step 2: Communicate the data
                    comm_cost = float('inf')
                    best_comm_nav_cost = float('inf')
                    for comm_wp in self.communication_waypoints:
                        if comm_wp in self.shortest_paths[rover][pos_after_intermediate_steps] and \
                           self.shortest_paths[rover][pos_after_intermediate_steps][comm_wp] != float('inf'):
                             best_comm_nav_cost = min(best_comm_nav_cost, self.shortest_paths[rover][pos_after_intermediate_steps][comm_wp])

                    if best_comm_nav_cost == float('inf'):
                         continue # Try next rover

                    rover_goal_cost += best_comm_nav_cost # Navigation to communication waypoint
                    rover_goal_cost += 1 # communicate_rock_data action

                    min_rover_cost_for_goal = min(min_rover_cost_for_goal, rover_goal_cost)

            elif goal_type == 'communicated_image_data':
                objective, mode = goal_fact[1], goal_fact[2]

                # Find rovers equipped for imaging
                equipped_rovers = [r for r, equip in self.rover_equipment.items() if 'imaging' in equip]

                if not equipped_rovers:
                    return float('inf') # No rover can achieve this goal

                for rover in equipped_rovers:
                    if rover not in rover_locations: continue

                    current_pos = rover_locations[rover]
                    rover_goal_cost = 0
                    pos_after_intermediate_steps = current_pos # Location after taking image

                    # Step 1: Get the image if not already had by this rover
                    # Check if rover already has this specific image (objective, mode)
                    if objective not in have_image[rover] or mode not in have_image[rover][objective]:

                        image_acquisition_cost = float('inf')
                        best_w_img_for_rover = None

                        # Find suitable cameras on this rover for this mode
                        suitable_cameras = [
                            cam for cam, info in self.camera_info.items()
                            if info.get('rover') == rover and mode in info.get('modes', set())
                        ]

                        if not suitable_cameras:
                            continue # This rover has no suitable camera

                        min_path_cost_for_rover_image = float('inf')
                        best_w_img_for_rover_image = None

                        for camera in suitable_cameras:
                            cal_target = self.camera_info[camera].get('calibration_target')
                            if not cal_target: continue # Camera has no calibration target

                            cal_wps = self.objective_visibility.get(cal_target, set())
                            img_wps = self.objective_visibility.get(objective, set())

                            if not cal_wps or not img_wps: continue # Cannot calibrate or image

                            min_path_cost_for_camera = float('inf')
                            best_w_img_for_camera = None

                            # Find best path: current_pos -> w_cal -> w_img
                            for w_cal in cal_wps:
                                # Check if w_cal is reachable from current_pos
                                if w_cal not in self.shortest_paths[rover][current_pos] or \
                                   self.shortest_paths[rover][current_pos][w_cal] == float('inf'):
                                    continue

                                for w_img in img_wps:
                                    # Check if w_img is reachable from w_cal
                                    if w_img not in self.shortest_paths[rover][w_cal] or \
                                       self.shortest_paths[rover][w_cal][w_img] == float('inf'):
                                        continue

                                    path_cost = self.shortest_paths[rover][current_pos][w_cal] + self.shortest_paths[rover][w_cal][w_img]

                                    if path_cost < min_path_cost_for_camera:
                                        min_path_cost_for_camera = path_cost
                                        best_w_img_for_camera = w_img

                            if min_path_cost_for_camera != float('inf'):
                                # Cost for this camera = path cost + 1 (calibrate) + 1 (take_image)
                                camera_total_cost = min_path_cost_for_camera + 2
                                if camera_total_cost < min_path_cost_for_rover_image:
                                    min_path_cost_for_rover_image = camera_total_cost
                                    best_w_img_for_rover_image = best_w_img_for_camera


                        if min_path_cost_for_rover_image == float('inf'):
                            # This rover cannot acquire the image using any of its suitable cameras
                            continue # Try next rover

                        rover_goal_cost += min_path_cost_for_rover_image
                        pos_after_intermediate_steps = best_w_img_for_rover_image # Rover is now at image location

                    # Step 2: Communicate the data
                    comm_cost = float('inf')
                    best_comm_nav_cost = float('inf')
                    # Find min distance from pos_after_intermediate_steps to any communication waypoint
                    for comm_wp in self.communication_waypoints:
                        if comm_wp in self.shortest_paths[rover][pos_after_intermediate_steps] and \
                           self.shortest_paths[rover][pos_after_intermediate_steps][comm_wp] != float('inf'):
                             best_comm_nav_cost = min(best_comm_nav_cost, self.shortest_paths[rover][pos_after_intermediate_steps][comm_wp])

                    if best_comm_nav_cost == float('inf'):
                         # Cannot reach any communication waypoint from current/acquired location
                         continue # Try next rover

                    rover_goal_cost += best_comm_nav_cost # Navigation to communication waypoint
                    rover_goal_cost += 1 # communicate_image_data action

                    min_rover_cost_for_goal = min(min_rover_cost_for_goal, rover_goal_cost)

            # After checking all suitable rovers for this goal
            if min_rover_cost_for_goal == float('inf'):
                 # If no rover could find a path or satisfy conditions for this goal
                 return float('inf') # Problem is unsolvable from this state

            total_heuristic += min_rover_cost_for_goal

        return total_heuristic
