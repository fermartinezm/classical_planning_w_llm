import collections
from fnmatch import fnmatch
# Assuming heuristic_base is available in the execution environment
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_path(graph, start, end):
    """
    Finds the shortest path cost (number of edges) between start and end
    in an unweighted graph using BFS.
    Returns float('inf') if no path exists or start/end not in graph.
    """
    if start == end:
        return 0
    # Check if start or end are valid nodes in the graph
    if start not in graph or end not in graph:
         return float('inf')

    queue = collections.deque([(start, 0)])
    visited = {start}

    while queue:
        current_node, current_cost = queue.popleft()

        # Use .get for safety, although nodes from graph should be keys
        for neighbor in graph.get(current_node, []):
            if neighbor == end:
                return current_cost + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_cost + 1))

    return float('inf') # No path found


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve each
    unmet goal fact independently and sums these estimates. It considers
    navigation costs (using BFS on the waypoint graph), sampling/imaging
    costs, calibration costs, drop costs, and communication costs.

    # Assumptions
    - The waypoint graph defined by `can_traverse` is the same for all rovers.
    - Every rover has exactly one store.
    - Every camera has exactly one calibration target.
    - Soil/rock samples required by goals are initially present (`at_soil_sample`, `at_rock_sample`).
    - Objectives/calibration targets required by goals are visible from at least one waypoint (`visible_from`).
    - The lander location and lander-visible waypoints are static.

    # Heuristic Initialization
    The constructor extracts static information from the task:
    - The lander's waypoint.
    - Which capabilities (soil, rock, imaging) each rover has.
    - Which store belongs to which rover.
    - Information about each camera (which rover it's on, which modes it supports, its calibration target).
    - Which objectives/calibration targets are visible from which waypoints.
    - The waypoint graph based on `can_traverse` facts.
    - The set of waypoints visible from the lander's waypoint.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to determine:
       - Current location of each rover.
       - Locations of remaining soil/rock samples.
       - Which rovers have collected which samples or images.
       - Status of each rover's store (empty/full).
       - Which cameras are calibrated on which rovers.
       - Which goal facts (communication of data) are already achieved.
    3. Iterate through each goal fact specified in the problem's goal condition.
    4. If a goal fact is already true in the current state, its cost is 0. Continue to the next goal.
    5. If a goal fact is not true, estimate the minimum cost to achieve it independently:
       - **For `(communicated_soil_data ?w)`:**
         - If any rover already has `(have_soil_analysis ?r ?w)`: Cost is min navigation from rover's current location to any lander-visible waypoint + 1 (communicate).
         - Otherwise (need to sample): Cost is min over equipped rovers `?r`:
           Navigation from `?r`'s location to `?w` + (1 if `?r`'s store is full) + 1 (sample) + navigation from `?w` to any lander-visible waypoint + 1 (communicate).
       - **For `(communicated_rock_data ?w)`:** Similar logic to soil data, using rock capabilities and samples.
       - **For `(communicated_image_data ?o ?m)`:**
         - If any rover already has `(have_image ?r ?o ?m)`: Cost is min navigation from rover's current location to any lander-visible waypoint + 1 (communicate).
         - Otherwise (need to take image): Cost is min over equipped rovers `?r` with camera `?i` supporting `?m`:
           - If camera `?i` is calibrated on `?r`: Navigation from `?r`'s location to any waypoint `?p` visible from `?o` + 1 (take image) + navigation from `?p` to any lander-visible waypoint + 1 (communicate).
           - If camera `?i` is not calibrated on `?r`: Navigation from `?r`'s location to any waypoint `?w` visible from `?i`'s calibration target + 1 (calibrate) + navigation from `?w` to any waypoint `?p` visible from `?o` + 1 (take image) + navigation from `?p` to any lander-visible waypoint + 1 (communicate).
         - Minimize over all suitable rovers, cameras, and relevant waypoints (`?p`, `?w`).
    6. Add the estimated minimum cost for the unachieved goal to the total heuristic cost. If a goal is determined to be unreachable (e.g., no equipped rover, no visible_from waypoint), a large finite penalty (1000) is added.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Parse Static Facts ---
        self.lander_waypoint = None
        self.rover_capabilities = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'on_board': rover, 'supports': {mode}, 'calibration_target': objective}
        self.visible_from_map = collections.defaultdict(set) # objective/target -> {waypoint}
        self.waypoint_graph = collections.defaultdict(set) # waypoint -> {neighbor_waypoint}
        self.lander_visible_waypoints = set() # {waypoint visible from lander}
        self.initial_soil_samples = set() # {waypoint} - locations of samples at the start
        self.initial_rock_samples = set() # {waypoint} - locations of samples at the start

        # Collect all waypoints mentioned in can_traverse to build the graph nodes
        all_waypoints = set()

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at_lander":
                # (at_lander ?x - lander ?y - waypoint)
                if len(parts) == 3:
                    lander, waypoint = parts[1], parts[2]
                    self.lander_waypoint = waypoint
                    all_waypoints.add(waypoint)
            elif predicate == "equipped_for_soil_analysis":
                # (equipped_for_soil_analysis ?r - rover)
                if len(parts) == 2:
                    rover = parts[1]
                    self.rover_capabilities[rover].add('soil')
            elif predicate == "equipped_for_rock_analysis":
                # (equipped_for_rock_analysis ?r - rover)
                if len(parts) == 2:
                    rover = parts[1]
                    self.rover_capabilities[rover].add('rock')
            elif predicate == "equipped_for_imaging":
                # (equipped_for_imaging ?r - rover)
                if len(parts) == 2:
                    rover = parts[1]
                    self.rover_capabilities[rover].add('imaging')
            elif predicate == "store_of":
                # (store_of ?s - store ?r - rover)
                if len(parts) == 3:
                    store, rover = parts[1], parts[2]
                    self.rover_stores[rover] = store
            elif predicate == "on_board":
                # (on_board ?i - camera ?r - rover)
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    if camera not in self.camera_info: self.camera_info[camera] = {'supports': set()}
                    self.camera_info[camera]['on_board'] = rover
            elif predicate == "supports":
                # (supports ?c - camera ?m - mode)
                if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    if camera not in self.camera_info: self.camera_info[camera] = {'supports': set()}
                    self.camera_info[camera]['supports'].add(mode)
            elif predicate == "calibration_target":
                # (calibration_target ?i - camera ?o - objective)
                if len(parts) == 3:
                    camera, objective = parts[1], parts[2]
                    if camera not in self.camera_info: self.camera_info[camera] = {'supports': set()}
                    self.camera_info[camera]['calibration_target'] = objective
            elif predicate == "visible_from":
                # (visible_from ?o - objective ?w - waypoint)
                if len(parts) == 3:
                    objective_or_target, waypoint = parts[1], parts[2]
                    self.visible_from_map[objective_or_target].add(waypoint)
                    all_waypoints.add(waypoint)
            elif predicate == "can_traverse":
                # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                # Assuming graph is same for all rovers based on examples
                if len(parts) == 4:
                    w1, w2 = parts[2], parts[3]
                    self.waypoint_graph[w1].add(w2)
                    self.waypoint_graph[w2].add(w1) # Assuming symmetric traversability
                    all_waypoints.add(w1)
                    all_waypoints.add(w2)
            # Note: 'visible' facts are processed after lander_waypoint is found

        # Ensure all waypoints from the problem exist as keys in the graph, even if isolated
        for wp in all_waypoints:
             if wp not in self.waypoint_graph:
                 self.waypoint_graph[wp] = set()


        # Now process visible facts to find lander_visible_waypoints
        if self.lander_waypoint:
             for fact in task.static:
                 parts = get_parts(fact)
                 if not parts: continue
                 if parts[0] == "visible":
                     if len(parts) == 3:
                         w1, w2 = parts[1], parts[2]
                         if w2 == self.lander_waypoint:
                             self.lander_visible_waypoints.add(w1)
                         if w1 == self.lander_waypoint: # Assuming symmetric
                             self.lander_visible_waypoints.add(w2)

        # Store initial sample locations (they are removed upon sampling)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "at_soil_sample":
                 if len(parts) == 2:
                     self.initial_soil_samples.add(parts[1])
             elif parts[0] == "at_rock_sample":
                 if len(parts) == 2:
                     self.initial_rock_samples.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0
        INF = float('inf') # Use a constant for infinity
        UNREACHABLE_PENALTY = 1000 # Penalty for seemingly unreachable goals

        # --- Parse Dynamic Facts from State ---
        rover_locations = {} # rover -> waypoint
        # soil_samples_at = set() # {waypoint} # Not needed, check initial_soil_samples and state
        # rock_samples_at = set() # {waypoint} # Not needed, check initial_rock_samples and state
        rover_soil_analysis = collections.defaultdict(set) # rover -> {waypoint}
        rover_rock_analysis = collections.defaultdict(set) # rover -> {waypoint}
        rover_images = collections.defaultdict(set) # rover -> {(objective, mode)}
        rover_store_status = {} # rover -> 'empty' or 'full'
        rover_calibrated_cameras = collections.defaultdict(set) # rover -> {camera}
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_images = set() # {(objective, mode)}

        # Initialize store status based on static info, then update from state
        for rover, store in self.rover_stores.items():
             rover_store_status[rover] = 'empty' # Assume empty initially unless state says full

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "at":
                # (at ?x - rover ?y - waypoint)
                if len(parts) == 3 and parts[1] in self.rover_capabilities: # Check if it's a rover
                    rover, waypoint = parts[1], parts[2]
                    rover_locations[rover] = waypoint
            # elif predicate == "at_soil_sample": # Not needed from state, check initial
            #     if len(parts) == 2: soil_samples_at.add(parts[1])
            # elif predicate == "at_rock_sample": # Not needed from state, check initial
            #     if len(parts) == 2: rock_samples_at.add(parts[1])
            elif predicate == "have_soil_analysis":
                # (have_soil_analysis ?r - rover ?w - waypoint)
                if len(parts) == 3:
                    rover, waypoint = parts[1], parts[2]
                    rover_soil_analysis[rover].add(waypoint)
            elif predicate == "have_rock_analysis":
                # (have_rock_analysis ?r - rover ?w - waypoint)
                if len(parts) == 3:
                    rover, waypoint = parts[1], parts[2]
                    rover_rock_analysis[rover].add(waypoint)
            elif predicate == "have_image":
                # (have_image ?r - rover ?o - objective ?m - mode)
                if len(parts) == 4:
                    rover, objective, mode = parts[1], parts[2], parts[3]
                    rover_images[rover].add((objective, mode))
            elif predicate == "empty":
                 # (empty ?s - store)
                 if len(parts) == 2:
                     store = parts[1]
                     # Find rover for this store
                     for r, s in self.rover_stores.items():
                         if s == store:
                             rover_store_status[r] = 'empty'
                             break
            elif predicate == "full":
                 # (full ?s - store)
                 if len(parts) == 2:
                     store = parts[1]
                     # Find rover for this store
                     for r, s in self.rover_stores.items():
                         if s == store:
                             rover_store_status[r] = 'full'
                             break
            elif predicate == "calibrated":
                # (calibrated ?c - camera ?r - rover)
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    rover_calibrated_cameras[rover].add(camera)
            elif predicate == "communicated_soil_data":
                # (communicated_soil_data ?w - waypoint)
                if len(parts) == 2:
                    communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                # (communicated_rock_data ?w - waypoint)
                if len(parts) == 2:
                    communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                # (communicated_image_data ?o - objective ?m - mode)
                if len(parts) == 3:
                    objective, mode = parts[1], parts[2]
                    communicated_images.add((objective, mode))

        # Pre-calculate min nav cost to any lander visible waypoint from a given waypoint
        # This avoids recomputing the min over lander_visible_waypoints repeatedly
        min_nav_to_lander_cache = {} # waypoint -> min_cost

        def get_min_nav_to_lander(start_wp):
            if start_wp in min_nav_to_lander_cache:
                return min_nav_to_lander_cache[start_wp]

            min_cost = INF
            if self.lander_visible_waypoints:
                 # Calculate min cost to reach any lander visible waypoint
                 costs = [bfs_shortest_path(self.waypoint_graph, start_wp, lander_wp)
                          for lander_wp in self.lander_visible_waypoints]
                 # Filter out INF costs before finding min
                 finite_costs = [c for c in costs if c != INF]
                 min_cost = min(finite_costs) if finite_costs else INF # Handle empty list after filtering

            min_nav_to_lander_cache[start_wp] = min_cost
            return min_cost


        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                if len(parts) != 2: continue
                waypoint_goal = parts[1]
                if waypoint_goal in communicated_soil:
                    continue # Goal already achieved

                min_cost_for_goal = INF

                # Option 1: Data already collected by some rover
                for rover in rover_soil_analysis:
                    if waypoint_goal in rover_soil_analysis[rover]:
                        if rover in rover_locations: # Ensure rover location is known
                            nav_cost = get_min_nav_to_lander(rover_locations[rover])
                            if nav_cost != INF:
                                min_cost_for_goal = min(min_cost_for_goal, nav_cost + 1) # +1 communicate

                # Option 2: Need to sample and communicate
                # Check if sample exists initially (assuming samples aren't created)
                if waypoint_goal in self.initial_soil_samples:
                    for rover in self.rover_capabilities:
                        if 'soil' in self.rover_capabilities[rover] and rover in rover_locations:
                            nav_to_sample = bfs_shortest_path(self.waypoint_graph, rover_locations[rover], waypoint_goal)
                            drop_cost = 1 if rover_store_status.get(rover, 'empty') == 'full' else 0 # Default to empty if status unknown
                            nav_to_lander = get_min_nav_to_lander(waypoint_goal)

                            if nav_to_sample != INF and nav_to_lander != INF:
                                cost = nav_to_sample + drop_cost + 1 + nav_to_lander + 1 # +1 sample, +1 communicate
                                min_cost_for_goal = min(min_cost_for_goal, cost)

                total_cost += min_cost_for_goal if min_cost_for_goal != INF else UNREACHABLE_PENALTY


            elif predicate == "communicated_rock_data":
                if len(parts) != 2: continue
                waypoint_goal = parts[1]
                if waypoint_goal in communicated_rock:
                    continue # Goal already achieved

                min_cost_for_goal = INF

                # Option 1: Data already collected by some rover
                for rover in rover_rock_analysis:
                    if waypoint_goal in rover_rock_analysis[rover]:
                         if rover in rover_locations: # Ensure rover location is known
                            nav_cost = get_min_nav_to_lander(rover_locations[rover])
                            if nav_cost != INF:
                                min_cost_for_goal = min(min_cost_for_goal, nav_cost + 1) # +1 communicate

                # Option 2: Need to sample and communicate
                # Check if sample exists initially
                if waypoint_goal in self.initial_rock_samples:
                    for rover in self.rover_capabilities:
                        if 'rock' in self.rover_capabilities[rover] and rover in rover_locations:
                            nav_to_sample = bfs_shortest_path(self.waypoint_graph, rover_locations[rover], waypoint_goal)
                            drop_cost = 1 if rover_store_status.get(rover, 'empty') == 'full' else 0
                            nav_to_lander = get_min_nav_to_lander(waypoint_goal)

                            if nav_to_sample != INF and nav_to_lander != INF:
                                cost = nav_to_sample + drop_cost + 1 + nav_to_lander + 1 # +1 sample, +1 communicate
                                min_cost_for_goal = min(min_cost_for_goal, cost)

                total_cost += min_cost_for_goal if min_cost_for_goal != INF else UNREACHABLE_PENALTY


            elif predicate == "communicated_image_data":
                if len(parts) != 3: continue
                objective_goal, mode_goal = parts[1], parts[2]
                if (objective_goal, mode_goal) in communicated_images:
                    continue # Goal already achieved

                min_cost_for_goal = INF

                # Option 1: Image already taken by some rover
                for rover in rover_images:
                    if (objective_goal, mode_goal) in rover_images[rover]:
                        if rover in rover_locations: # Ensure rover location is known
                            nav_cost = get_min_nav_to_lander(rover_locations[rover])
                            if nav_cost != INF:
                                min_cost_for_goal = min(min_cost_for_goal, nav_cost + 1) # +1 communicate

                # Option 2: Need to take image and communicate
                # Find waypoints visible from the objective
                image_wps = self.visible_from_map.get(objective_goal, set())

                if image_wps: # Check if objective is visible from anywhere
                    for rover in self.rover_capabilities:
                        if 'imaging' in self.rover_capabilities[rover] and rover in rover_locations:
                            # Find cameras on this rover supporting the mode
                            suitable_cameras = [
                                cam for cam, info in self.camera_info.items()
                                if info.get('on_board') == rover and mode_goal in info.get('supports', set())
                            ]

                            for camera in suitable_cameras:
                                calibration_target = self.camera_info.get(camera, {}).get('calibration_target')
                                if not calibration_target: continue # Camera has no calibration target

                                calibration_wps = self.visible_from_map.get(calibration_target, set())

                                if not calibration_wps: continue # Calibration target not visible from anywhere

                                # Case 2a: Camera is currently calibrated
                                if camera in rover_calibrated_cameras.get(rover, set()):
                                    for image_wp in image_wps:
                                        nav_to_image = bfs_shortest_path(self.waypoint_graph, rover_locations[rover], image_wp)
                                        nav_to_lander = get_min_nav_to_lander(image_wp)

                                        if nav_to_image != INF and nav_to_lander != INF:
                                            cost = nav_to_image + 1 + nav_to_lander + 1 # +1 take_image, +1 communicate
                                            min_cost_for_goal = min(min_cost_for_goal, cost)

                                # Case 2b: Camera needs calibration
                                else:
                                    for calibration_wp in calibration_wps:
                                        for image_wp in image_wps:
                                            nav_to_calibrate = bfs_shortest_path(self.waypoint_graph, rover_locations[rover], calibration_wp)
                                            nav_to_image = bfs_shortest_path(self.waypoint_graph, calibration_wp, image_wp)
                                            nav_to_lander = get_min_nav_to_lander(image_wp)

                                            if nav_to_calibrate != INF and nav_to_image != INF and nav_to_lander != INF:
                                                cost = nav_to_calibrate + 1 + nav_to_image + 1 + nav_to_lander + 1 # +1 calibrate, +1 take_image, +1 communicate
                                                min_cost_for_goal = min(min_cost_for_goal, cost)

                total_cost += min_cost_for_goal if min_cost_for_goal != INF else UNREACHABLE_PENALTY

        return total_cost
