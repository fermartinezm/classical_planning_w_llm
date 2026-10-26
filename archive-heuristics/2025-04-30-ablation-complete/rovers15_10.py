from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

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

def bfs_shortest_paths(graph, all_nodes, start_node):
    """Computes shortest paths from start_node to all reachable nodes in a graph."""
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        # Start node is not a known waypoint
        return distances # Cannot reach anything

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph adjacency list
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
    This heuristic estimates the total number of actions required to achieve all
    uncommunicated goal facts. It sums up the estimated cost for each uncommunicated
    goal independently, considering the actions needed (sampling, imaging,
    calibrating, dropping, communicating) and the minimum navigation cost
    required for each step using precomputed shortest paths.

    # Assumptions
    - Each uncommunicated goal is treated independently.
    - Navigation cost between waypoints for a specific rover is the shortest path
      distance in the graph defined by `can_traverse` facts for that rover.
    - Shortest paths are precomputed using BFS.
    - A large finite cost (1000) is returned if a required waypoint for a goal
      is unreachable from any suitable rover's current location.
    - A rover can perform a sampling/imaging task if it is equipped and can reach
      the required location.
    - A rover can communicate data if it possesses the data and can reach a
      communication waypoint (visible from the lander).
    - The cost of dropping a sample is added only if a sample action is required
      for a goal and the store of the chosen rover is full.

    # Heuristic Initialization
    The heuristic initializes by parsing the static facts from the task definition:
    - Identifies the lander location.
    - Maps rovers to their capabilities (soil, rock, imaging), stores, and cameras.
    - Maps cameras to supported modes and calibration targets.
    - Maps objectives and calibration targets to waypoints they are visible from.
    - Builds traversal graphs for each rover based on `can_traverse` facts.
    - Precomputes all-pairs shortest paths for each rover's traversal graph using BFS.
    - Identifies all waypoints visible from the lander location as communication waypoints.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic computes the estimated cost as follows:
    1. Parse the current state to determine:
       - Current location of each rover.
       - Status (empty/full) of each store.
       - Which rovers have which soil/rock analyses.
       - Which rovers have which images.
       - Which cameras on which rovers are calibrated.
       - Which soil/rock samples are still at their initial locations.
       - Which soil/rock data have already been communicated.
       - Which image data have already been communicated.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal fact defined in the task.
    4. For each goal fact:
       - If the goal fact is already true in the current state, add 0 cost for this goal.
       - If the goal is `(communicated_soil_data ?w)` and it's not communicated:
         - Add 1 for the `communicate_soil_data` action.
         - Check if any soil-equipped rover has `(have_soil_analysis ?r ?w)`.
         - If not:
           - Add 1 for the `sample_soil` action.
           - Find the minimum navigation cost for any soil-equipped rover from its current location to waypoint `?w` using precomputed shortest paths. Add this cost. If unreachable, return 1000.
           - If the rover chosen for minimum navigation to `?w` has a full store, add 1 for the `drop` action.
         - Find the minimum navigation cost for any rover from its current location to any communication waypoint (visible from the lander). Add this cost. If unreachable, return 1000.
       - If the goal is `(communicated_rock_data ?w)` and it's not communicated:
         - Similar logic as for soil data, using rock-equipped rovers and rock samples.
       - If the goal is `(communicated_image_data ?o ?m)` and it's not communicated:
         - Add 1 for the `communicate_image_data` action.
         - Check if any imaging-equipped rover has `(have_image ?r ?o ?m)`.
         - If not:
           - Add 1 for the `take_image` action.
           - Check if any camera `?i` on board an imaging-equipped rover `?r` supporting mode `?m` is calibrated `(calibrated ?i ?r)`.
           - If not:
             - Add 1 for the `calibrate` action.
             - Find the minimum navigation cost for any suitable rover/camera from its current location to any waypoint visible from the camera's calibration target. Add this cost. If unreachable, return 1000.
           - Find the minimum navigation cost for any suitable rover/camera from its current location to any waypoint visible from objective `?o`. Add this cost. If unreachable, return 1000.
         - Find the minimum navigation cost for any rover from its current location to any communication waypoint. Add this cost. If unreachable, return 1000.
    5. The total accumulated cost is the heuristic value. If all goals were initially communicated, the cost is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)
        self.goals = task.goals

        # --- Parse Static Facts ---
        self.lander_location = None
        self.rover_capabilities = defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_stores = {} # rover -> store (mapping rover to its store)
        self.rover_cameras = defaultdict(set) # rover -> {camera}
        self.camera_modes = defaultdict(set) # camera -> {mode}
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = defaultdict(set) # objective -> {waypoint}
        self.waypoint_visibility_graph = defaultdict(set) # waypoint -> {waypoint} (visible w1 w2 means w2 is visible from w1)
        self.rover_traversal_graphs = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> {waypoint}
        self.all_waypoints = set()
        self.all_rovers = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_stores = set()
        self.all_landers = set()

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at_lander':
                self.lander_location = parts[2] # (at_lander ?l ?y)
                self.all_landers.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif pred == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('soil')
                self.all_rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('rock')
                self.all_rovers.add(parts[1])
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('imaging')
                self.all_rovers.add(parts[1])
            elif pred == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # (store_of ?s ?r) maps rover to store
                self.all_stores.add(parts[1])
                self.all_rovers.add(parts[2])
            elif pred == 'on_board':
                self.rover_cameras[parts[2]].add(parts[1]) # (on_board ?i ?r) maps rover to camera
                self.all_cameras.add(parts[1])
                self.all_rovers.add(parts[2])
            elif pred == 'supports':
                self.camera_modes[parts[1]].add(parts[2]) # (supports ?c ?m)
                self.all_cameras.add(parts[1])
                self.all_modes.add(parts[2])
            elif pred == 'calibration_target':
                self.camera_calibration_target[parts[1]] = parts[2] # (calibration_target ?i ?t)
                self.all_cameras.add(parts[1])
                self.all_objectives.add(parts[2])
            elif pred == 'visible_from':
                self.objective_visible_from[parts[1]].add(parts[2]) # (visible_from ?o ?w)
                self.all_objectives.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif pred == 'visible':
                self.waypoint_visibility_graph[parts[1]].add(parts[2]) # (visible ?w1 ?w2) means w2 is visible from w1
                self.all_waypoints.add(parts[1])
                self.all_waypoints.add(parts[2])
            elif pred == 'can_traverse':
                self.rover_traversal_graphs[parts[1]][parts[2]].add(parts[3]) # (can_traverse ?r ?y ?z)
                self.all_rovers.add(parts[1])
                self.all_waypoints.add(parts[2])
                self.all_waypoints.add(parts[3])
            elif pred == 'waypoint':
                 self.all_waypoints.add(parts[1])
            elif pred == 'rover':
                 self.all_rovers.add(parts[1])
            elif pred == 'camera':
                 self.all_cameras.add(parts[1])
            elif pred == 'objective':
                 self.all_objectives.add(parts[1])
            elif pred == 'mode':
                 self.all_modes.add(parts[1])
            elif pred == 'store':
                 self.all_stores.add(parts[1])
            elif pred == 'lander':
                 self.all_landers.add(parts[1])

        # Ensure all waypoints mentioned in can_traverse are in the set
        for rover_graph in self.rover_traversal_graphs.values():
            for wp_from, wp_tos in rover_graph.items():
                self.all_waypoints.add(wp_from)
                self.all_waypoints.update(wp_tos)

        # Precompute shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover in self.all_rovers:
            graph = self.rover_traversal_graphs.get(rover, defaultdict(set))
            self.rover_shortest_paths[rover] = {}
            for start_wp in self.all_waypoints:
                self.rover_shortest_paths[rover][start_wp] = bfs_shortest_paths(graph, self.all_waypoints, start_wp)

        # Identify communication waypoints (waypoints visible from the lander location)
        self.communication_waypoints = set()
        if self.lander_location:
             # Need waypoints ?w such that (visible ?w lander_location) is true.
             # This means lander_location is in the adjacency list of ?w in the visible graph.
             # Rebuild the visible graph to be from -> to for easier lookup (w1 is visible from w2)
             visible_from_graph = defaultdict(set)
             for w1, neighbors in self.waypoint_visibility_graph.items():
                 for w2 in neighbors:
                     visible_from_graph[w2].add(w1) # w1 is visible from w2

             self.communication_waypoints = visible_from_graph.get(self.lander_location, set())


        # Filter rovers by capability for faster lookup
        self.rovers_equipped_for_soil = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps}
        self.rovers_equipped_for_rock = {r for r, caps in self.rover_capabilities.items() if 'rock' in caps}
        self.rovers_equipped_for_imaging = {r for r, caps in self.rover_capabilities.items() if 'imaging' in caps}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State Facts ---
        current_rover_locations = {} # rover -> waypoint
        current_store_status = {} # store -> 'empty' or 'full'
        current_rover_analyses = defaultdict(set) # rover -> {(type, waypoint)}
        current_rover_images = defaultdict(set) # rover -> {(objective, mode)}
        current_camera_calibrated = set() # {(camera, rover)}
        current_soil_samples_at = set() # {waypoint}
        current_rock_samples_at = set() # {waypoint}
        current_communicated_soil = set() # {waypoint}
        current_communicated_rock = set() # {waypoint}
        current_communicated_image = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'at':
                # Could be rover or lander, only care about rover location
                if parts[1] in self.all_rovers:
                     current_rover_locations[parts[1]] = parts[2]
            elif pred == 'empty':
                current_store_status[parts[1]] = 'empty'
            elif pred == 'full':
                current_store_status[parts[1]] = 'full'
            elif pred == 'have_soil_analysis':
                current_rover_analyses[parts[1]].add(('soil', parts[2]))
            elif pred == 'have_rock_analysis':
                current_rover_analyses[parts[1]].add(('rock', parts[2]))
            elif pred == 'have_image':
                current_rover_images[parts[1]].add((parts[2], parts[3]))
            elif pred == 'calibrated':
                current_camera_calibrated.add((parts[1], parts[2]))
            elif pred == 'at_soil_sample':
                current_soil_samples_at.add(parts[1])
            elif pred == 'at_rock_sample':
                current_rock_samples_at.add(parts[1])
            elif pred == 'communicated_soil_data':
                current_communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data':
                current_communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data':
                current_communicated_image.add((parts[1], parts[2]))

        total_cost = 0
        UNREACHABLE_PENALTY = 1000 # Large finite cost for unreachable goals

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint not in current_communicated_soil:
                    goal_cost = 1 # communicate action

                    # Check if sample is held by any soil-equipped rover
                    has_sample = any(('soil', waypoint) in current_rover_analyses[r] for r in self.rovers_equipped_for_soil)

                    if not has_sample:
                        goal_cost += 1 # sample action

                        # Find min nav cost to sample location + potential drop cost
                        min_nav_to_sample = float('inf')
                        needs_drop_for_sample = False
                        best_rover_for_sample_nav = None

                        for r in self.rovers_equipped_for_soil:
                            if r in current_rover_locations and waypoint in self.rover_shortest_paths.get(r, {}).get(current_rover_locations[r], {}):
                                dist = self.rover_shortest_paths[r][current_rover_locations[r]][waypoint]
                                if dist != float('inf'):
                                    if dist < min_nav_to_sample:
                                        min_nav_to_sample = dist
                                        best_rover_for_sample_nav = r
                                        store = self.rover_stores.get(r)
                                        needs_drop_for_sample = (store is not None and current_store_status.get(store) == 'full')

                        if min_nav_to_sample != float('inf'):
                            goal_cost += min_nav_to_sample
                        else:
                            # Unreachable sample location for any suitable rover
                            return UNREACHABLE_PENALTY

                        if needs_drop_for_sample:
                            goal_cost += 1 # drop action

                    # Find min nav cost from current location to communication waypoint
                    min_nav_to_comm = float('inf')
                    for r, loc in current_rover_locations.items():
                        # Any rover can communicate if it has the data (implicitly assumed)
                        for comm_w in self.communication_waypoints:
                             if comm_w in self.rover_shortest_paths.get(r, {}).get(loc, {}):
                                 dist = self.rover_shortest_paths[r][loc][comm_w]
                                 if dist != float('inf'):
                                     min_nav_to_comm = min(min_nav_to_comm, dist)

                    if min_nav_to_comm != float('inf'):
                        goal_cost += min_nav_to_comm
                    else:
                        # Unreachable communication waypoint for any rover
                        return UNREACHABLE_PENALTY

                    total_cost += goal_cost

            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint not in current_communicated_rock:
                    goal_cost = 1 # communicate action

                    # Check if sample is held by any rock-equipped rover
                    has_sample = any(('rock', waypoint) in current_rover_analyses[r] for r in self.rovers_equipped_for_rock)

                    if not has_sample:
                        goal_cost += 1 # sample action

                        # Find min nav cost to sample location + potential drop cost
                        min_nav_to_sample = float('inf')
                        needs_drop_for_sample = False
                        best_rover_for_sample_nav = None

                        for r in self.rovers_equipped_for_rock:
                            if r in current_rover_locations and waypoint in self.rover_shortest_paths.get(r, {}).get(current_rover_locations[r], {}):
                                dist = self.rover_shortest_paths[r][current_rover_locations[r]][waypoint]
                                if dist != float('inf'):
                                    if dist < min_nav_to_sample:
                                        min_nav_to_sample = dist
                                        best_rover_for_sample_nav = r
                                        store = self.rover_stores.get(r)
                                        needs_drop_for_sample = (store is not None and current_store_status.get(store) == 'full')

                        if min_nav_to_sample != float('inf'):
                            goal_cost += min_nav_to_sample
                        else:
                            # Unreachable sample location for any suitable rover
                            return UNREACHABLE_PENALTY

                        if needs_drop_for_sample:
                            goal_cost += 1 # drop action

                    # Find min nav cost from current location to communication waypoint
                    min_nav_to_comm = float('inf')
                    for r, loc in current_rover_locations.items():
                        # Any rover can communicate if it has the data (implicitly assumed)
                        for comm_w in self.communication_waypoints:
                             if comm_w in self.rover_shortest_paths.get(r, {}).get(loc, {}):
                                 dist = self.rover_shortest_paths[r][loc][comm_w]
                                 if dist != float('inf'):
                                     min_nav_to_comm = min(min_nav_to_comm, dist)

                    if min_nav_to_comm != float('inf'):
                        goal_cost += min_nav_to_comm
                    else:
                        # Unreachable communication waypoint for any rover
                        return UNREACHABLE_PENALTY

                    total_cost += goal_cost

            elif pred == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                if (objective, mode) not in current_communicated_image:
                    goal_cost = 1 # communicate action

                    # Check if image is held by any imaging-equipped rover
                    has_image = any(((objective, mode) in current_rover_images[r]) for r in self.rovers_equipped_for_imaging)
                    best_rover_for_image_task = None # Rover that handles calibrate/take_image nav

                    if not has_image:
                        goal_cost += 1 # take_image action

                        # Check if any suitable camera is calibrated
                        is_calibrated = False
                        best_rover_camera_for_cal = None # (rover, camera)

                        for r in self.rovers_equipped_for_imaging:
                            for i in self.rover_cameras.get(r, set()):
                                if mode in self.camera_modes.get(i, set()):
                                    if (i, r) in current_camera_calibrated:
                                        is_calibrated = True
                                        best_rover_camera_for_cal = (r, i) # Found a calibrated one, potentially use this path
                                        break
                            if is_calibrated: break

                        if not is_calibrated:
                            goal_cost += 1 # calibrate action
                            # Find best rover/camera and nav cost to calibration waypoint
                            min_nav_to_cal = float('inf')
                            best_rover_camera_for_cal_nav = None # (rover, camera)

                            for r in self.rovers_equipped_for_imaging:
                                for i in self.rover_cameras.get(r, set()):
                                    if mode in self.camera_modes.get(i, set()):
                                        cal_target = self.camera_calibration_target.get(i)
                                        if cal_target:
                                            # Calibration target is an objective, visible from waypoints
                                            cal_wps = self.objective_visible_from.get(cal_target, set())
                                            for cal_w in cal_wps:
                                                if r in current_rover_locations and cal_w in self.rover_shortest_paths.get(r, {}).get(current_rover_locations[r], {}):
                                                    dist = self.rover_shortest_paths[r][current_rover_locations[r]][cal_w]
                                                    if dist != float('inf'):
                                                        if dist < min_nav_to_cal:
                                                            min_nav_to_cal = dist
                                                            best_rover_camera_for_cal_nav = (r, i)

                            if min_nav_to_cal != float('inf'):
                                goal_cost += min_nav_to_cal
                                # The rover that does calibration nav is likely the one to take the image
                                if best_rover_camera_for_cal_nav:
                                    best_rover_for_image_task = best_rover_camera_for_cal_nav[0]
                            else:
                                 # Unreachable calibration waypoint for any suitable rover/camera
                                 return UNREACHABLE_PENALTY

                        # Find min nav cost to image waypoint
                        min_nav_to_image = float('inf')
                        img_wps = self.objective_visible_from.get(objective, set())

                        # If we identified a specific rover for the task (calibration), use it first
                        if best_rover_for_image_task and best_rover_for_image_task in current_rover_locations:
                             loc = current_rover_locations[best_rover_for_image_task]
                             for img_w in img_wps:
                                 if img_w in self.rover_shortest_paths.get(best_rover_for_image_task, {}).get(loc, {}):
                                     dist = self.rover_shortest_paths[best_rover_for_image_task][loc][img_w]
                                     if dist != float('inf'):
                                         min_nav_to_image = min(min_nav_to_image, dist)

                        # If no specific rover was chosen or it couldn't reach, check all imaging rovers
                        if min_nav_to_image == float('inf'):
                            for r in self.rovers_equipped_for_imaging:
                                if r in current_rover_locations:
                                    loc = current_rover_locations[r]
                                    for img_w in img_wps:
                                        if img_w in self.rover_shortest_paths.get(r, {}).get(loc, {}):
                                            dist = self.rover_shortest_paths[r][loc][img_w]
                                            if dist != float('inf'):
                                                min_nav_to_image = min(min_nav_to_image, dist)

                        if min_nav_to_image != float('inf'):
                            goal_cost += min_nav_to_image
                        else:
                            # Unreachable image waypoint for any suitable rover
                            return UNREACHABLE_PENALTY


                    # Find min nav cost from current location to communication waypoint
                    min_nav_to_comm = float('inf')
                    for r, loc in current_rover_locations.items():
                        # Any rover can communicate if it has the data (implicitly assumed)
                        for comm_w in self.communication_waypoints:
                             if comm_w in self.rover_shortest_paths.get(r, {}).get(loc, {}):
                                 dist = self.rover_shortest_paths[r][loc][comm_w]
                                 if dist != float('inf'):
                                     min_nav_to_comm = min(min_nav_to_comm, dist)

                    if min_nav_to_comm != float('inf'):
                        goal_cost += min_nav_to_comm
                    else:
                        # Unreachable communication waypoint for any rover
                        return UNREACHABLE_PENALTY

                    total_cost += goal_cost

        return total_cost
