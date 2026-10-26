import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs(graph, start_node, all_nodes):
    """
    Performs BFS to find shortest paths from a start_node in a graph.

    Args:
        graph: Adjacency list representation {node: {neighbor1, neighbor2, ...}}.
        start_node: The starting node for BFS.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary {node: distance} containing shortest distances from start_node.
        Distance is float('inf') if a node is unreachable.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
         # Start node is not in the graph, no paths possible
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node exists in the graph keys
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
    This heuristic estimates the number of actions required to satisfy all goal
    conditions. It calculates the minimum cost for each unachieved goal
    independently and sums these minimum costs. The cost for a goal is estimated
    by summing the minimum navigation costs and action costs required to collect
    the necessary data (soil, rock, or image) and then communicate it to the lander.

    # Assumptions
    - Navigation cost between two waypoints for a specific rover is the shortest
      path distance in the `can_traverse` graph for that rover.
    - The cost of any action (navigate, sample, drop, calibrate, take_image,
      communicate) is 1.
    - For sampling goals, a rover needs to navigate to the sample location,
      potentially drop a full store, sample, navigate to a communication waypoint,
      and communicate.
    - For imaging goals, a rover needs to navigate to a calibration waypoint (if
      not already calibrated), calibrate, navigate to an image waypoint, take the
      image, navigate to a communication waypoint, and communicate.
    - The heuristic calculates the minimum cost over all suitable rovers, cameras,
      and waypoints for each goal independently.
    - If a required resource (e.g., soil sample, image waypoint, calibration
      waypoint, communication waypoint) is unavailable or unreachable by any
      suitable rover, the heuristic returns infinity, indicating a likely dead end.

    # Heuristic Initialization
    The heuristic precomputes static information from the task definition:
    - Sets of all waypoints, rovers, landers, stores, cameras, objectives, modes.
    - Locations of landers.
    - Waypoints visible from lander locations (communication waypoints).
    - Rover capabilities (soil, rock, imaging).
    - Store associated with each rover.
    - Cameras on board each rover.
    - Modes supported by each camera.
    - Calibration target for each camera.
    - Waypoints from which each objective is visible.
    - Waypoints from which each calibration target is visible.
    - Navigation graphs for each rover based on `can_traverse` facts.
    - All-pairs shortest path distances for each rover's navigation graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to identify:
       - Current position of each rover.
       - Which rovers have which soil/rock samples or images.
       - Status of each rover's store (empty/full).
       - Which cameras on which rovers are calibrated.
       - Which soil/rock samples are still at their original locations.
       - Which goal conditions (communicated data) are already satisfied.
    2. Initialize the total heuristic cost `h` to 0.
    3. Iterate through each goal condition specified in the task.
    4. If a goal condition `g` is already satisfied in the current state, continue to the next goal.
    5. If `g` is `(communicated_soil_data ?w)` and not satisfied:
       - Calculate the minimum cost for *any* soil-equipped rover to get the sample at `?w` and communicate it.
       - This involves finding a soil-equipped rover `r`, calculating the cost to navigate `r` to `?w`, potentially drop a sample (if store is full), sample, navigate `r` from `?w` to a communication waypoint `x`, and communicate.
       - The cost is `nav(r_pos, w) + (1 if store full) + 1 (sample) + nav(w, x) + 1 (communicate)`.
       - If the rover already has the sample `(have_soil_analysis r w)`, the sample cost is 0, and navigation starts from the rover's current position `r_pos`: `nav(r_pos, x) + 1 (communicate)`.
       - Minimize this total cost over all suitable rovers `r` and communication waypoints `x`.
       - If no suitable rover can reach the sample location or a communication waypoint, or if the sample is no longer available, the minimum cost for this goal is infinity.
       - Add the minimum cost for this goal to `h`. If the minimum cost is infinity, return infinity immediately.
    6. If `g` is `(communicated_rock_data ?w)` and not satisfied:
       - Perform a similar calculation as for soil data, using rock-equipped rovers and rock samples.
    7. If `g` is `(communicated_image_data ?o ?m)` and not satisfied:
       - Calculate the minimum cost for *any* imaging-equipped rover with a camera supporting mode `?m` to take an image of `?o` and communicate it.
       - This involves finding a suitable rover `r`, camera `i`, image waypoint `p` (visible from `?o`), calibration target `t` (for `i`), calibration waypoint `w_cal` (visible from `t`), and communication waypoint `x`.
       - If the rover already has the image `(have_image r o m)`, the image cost is 0, and navigation starts from the rover's current position `r_pos`: `nav(r_pos, x) + 1 (communicate)`.
       - If the image needs to be taken:
         - If camera `i` is not calibrated: Navigate `r` from `r_pos` to `w_cal`, calibrate (1 action), navigate `r` from `w_cal` to `p`. Cost: `nav(r_pos, w_cal) + 1 + nav(w_cal, p)`.
         - If camera `i` is calibrated: Navigate `r` from `r_pos` to `p`. Cost: `nav(r_pos, p)`.
         - Add 1 for the `take_image` action at `p`.
         - Navigate `r` from `p` to `x`. Cost: `nav(p, x)`.
         - Add 1 for the `communicate_image_data` action at `x`.
       - Minimize the total cost (image acquisition path + communication path) over all suitable rovers `r`, cameras `i`, image waypoints `p`, calibration waypoints `w_cal`, and communication waypoints `x`.
       - If no suitable path or resource exists (e.g., no image waypoint, no calibration waypoint, no comm waypoint, unreachable location), the minimum cost for this goal is infinity.
       - Add the minimum cost for this goal to `h`. If the minimum cost is infinity, return infinity immediately.
    8. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Facts true at the start

        # --- Extract Static Information ---
        self.waypoints = set()
        self.rovers = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()

        self.lander_locations = set()
        self.rover_capabilities = collections.defaultdict(set) # {rover: {cap1, cap2, ...}}
        self.rover_stores = {} # {rover: store}
        self.rover_cameras = collections.defaultdict(set) # {rover: {camera1, camera2, ...}}
        self.camera_modes_supported = set() # {(camera, mode)}
        self.camera_calibration_targets = {} # {camera: objective}
        self.objective_visible_from = collections.defaultdict(set) # {objective: {waypoint1, ...}}
        self.rover_navigation_graphs = collections.defaultdict(lambda: collections.defaultdict(set)) # {rover: {wp: {neighbor1, ...}}}

        # Parse static facts and initial state to collect objects and relationships
        all_facts = static_facts | initial_state # Consider initial state for objects and initial samples

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.landers.add(lander)
                self.waypoints.add(wp)
                self.lander_locations.add(wp)
            elif predicate == 'at':
                obj, wp = parts[1], parts[2]
                # Only add rovers found via 'at' or 'can_traverse' or capabilities
                # if obj.startswith('rover'): self.rovers.add(obj)
                self.waypoints.add(wp)
            elif predicate == 'can_traverse':
                rover, wp1, wp2 = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
                self.rover_navigation_graphs[rover][wp1].add(wp2)
            elif predicate == 'equipped_for_soil_analysis':
                self.rovers.add(parts[1])
                self.rover_capabilities[parts[1]].add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                self.rovers.add(parts[1])
                self.rover_capabilities[parts[1]].add('rock')
            elif predicate == 'equipped_for_imaging':
                self.rovers.add(parts[1])
                self.rover_capabilities[parts[1]].add('imaging')
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.stores.add(store)
                self.rovers.add(rover)
                self.rover_stores[rover] = store
            elif predicate == 'calibrated':
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                self.cameras.add(camera)
                self.modes.add(mode)
                self.camera_modes_supported.add((camera, mode))
            elif predicate == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
            elif predicate == 'have_rock_analysis':
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif predicate == 'have_soil_analysis':
                 self.rovers.add(parts[1])
                 self.waypoints.add(parts[2])
            elif predicate == 'full' or predicate == 'empty':
                 self.stores.add(parts[1])
            elif predicate == 'have_image':
                 self.rovers.add(parts[1])
                 self.objectives.add(parts[2])
                 self.modes.add(parts[3])
            elif predicate == 'communicated_soil_data':
                 self.waypoints.add(parts[1])
            elif predicate == 'communicated_rock_data':
                 self.waypoints.add(parts[1])
            elif predicate == 'communicated_image_data':
                 self.objectives.add(parts[1])
                 self.modes.add(parts[2])
            elif predicate == 'at_soil_sample':
                 self.waypoints.add(parts[1])
            elif predicate == 'at_rock_sample':
                 self.waypoints.add(parts[1])
            elif predicate == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.objectives.add(obj)
                self.waypoints.add(wp)
                self.objective_visible_from[obj].add(wp)
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.cameras.add(camera)
                self.objectives.add(objective)
                self.camera_calibration_targets[camera] = objective
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.cameras.add(camera)
                self.rovers.add(rover)
                self.rover_cameras[rover].add(camera)

        # Determine communication waypoints (visible from any lander location)
        self.comm_waypoints = set()
        for fact in static_facts:
             if match(fact, "visible", "?x", "?y"):
                 wp_x, wp_y = get_parts(fact)[1], get_parts(fact)[2]
                 if wp_y in self.lander_locations:
                     self.comm_waypoints.add(wp_x)
             # Also check if lander location itself is a comm waypoint (visible from itself)
             if match(fact, "visible", "?x", "?y") and wp_x == wp_y and wp_x in self.lander_locations:
                  self.comm_waypoints.add(wp_x)


        # Determine calibration target visible waypoints
        self.calibration_target_visible_from = {}
        for camera, target_obj in self.camera_calibration_targets.items():
             if target_obj in self.objective_visible_from:
                 self.calibration_target_visible_from[target_obj] = self.objective_visible_from[target_obj]
             else:
                 # This target objective is not visible from anywhere, cannot calibrate using it
                 self.calibration_target_visible_from[target_obj] = set()


        # Compute shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover in self.rovers:
            self.rover_shortest_paths[rover] = {}
            # Ensure all waypoints are considered, even if not in the rover's graph keys
            # BFS needs a list of all nodes to initialize distances
            all_waypoints_list = list(self.waypoints)
            for start_wp in self.waypoints:
                 self.rover_shortest_paths[rover][start_wp] = bfs(self.rover_navigation_graphs[rover], start_wp, self.waypoints)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Parse Current State ---
        rover_current_pos = {} # {rover: waypoint}
        rover_has_soil = collections.defaultdict(set) # {rover: {waypoint, ...}}
        rover_has_rock = collections.defaultdict(set) # {rover: {waypoint, ...}}
        rover_has_image = collections.defaultdict(set) # {rover: {(objective, mode), ...}}
        rover_store_status = {} # {rover: 'empty' or 'full'}
        rover_calibrated_cameras = collections.defaultdict(set) # {rover: {camera, ...}}
        soil_samples_remaining = set() # {waypoint, ...}
        rock_samples_remaining = set() # {waypoint, ...}
        communicated_soil = set() # {waypoint, ...}
        communicated_rock = set() # {waypoint, ...}
        communicated_image = set() # {(objective, mode), ...}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                obj, wp = parts[1], parts[2]
                if obj in self.rovers: # Only track rover positions
                    rover_current_pos[obj] = wp
            elif predicate == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                rover_has_soil[rover].add(wp)
            elif predicate == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                rover_has_rock[rover].add(wp)
            elif predicate == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_has_image[rover].add((obj, mode))
            elif predicate == 'empty':
                store = parts[1]
                # Find which rover this store belongs to
                for r, s in self.rover_stores.items():
                    if s == store:
                        rover_store_status[r] = 'empty'
                        break
            elif predicate == 'full':
                 store = parts[1]
                 for r, s in self.rover_stores.items():
                    if s == store:
                        rover_store_status[r] = 'full'
                        break
            elif predicate == 'calibrated':
                camera, rover = parts[1], parts[2]
                rover_calibrated_cameras[rover].add(camera)
            elif predicate == 'at_soil_sample':
                soil_samples_remaining.add(parts[1])
            elif predicate == 'at_rock_sample':
                rock_samples_remaining.add(parts[1])
            elif predicate == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))

        total_cost = 0  # Initialize action cost counter.

        # --- Calculate Cost for Each Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint in communicated_soil:
                    continue # Goal already satisfied

                min_goal_cost = float('inf')

                # Find a suitable rover
                for rover in self.rovers:
                    if 'soil' not in self.rover_capabilities.get(rover, set()):
                        continue # Rover not equipped for soil

                    r_pos = rover_current_pos.get(rover)
                    if r_pos is None: continue # Rover location unknown (shouldn't happen in valid states)

                    # Cost to get sample
                    sample_cost = float('inf')
                    rover_pos_after_sample = None

                    if waypoint in rover_has_soil.get(rover, set()):
                        # Rover already has the sample
                        sample_cost = 0
                        rover_pos_after_sample = r_pos # Rover stays at current pos if already has it

                    elif waypoint in soil_samples_remaining:
                        # Need to sample
                        nav_r_to_w = self.rover_shortest_paths.get(rover, {}).get(r_pos, {}).get(waypoint, float('inf'))
                        if nav_r_to_w is not float('inf'):
                            drop_cost = 1 if rover_store_status.get(rover) == 'full' else 0
                            sample_cost = nav_r_to_w + drop_cost + 1 # nav + (drop) + sample
                            rover_pos_after_sample = waypoint # Rover ends up at sample location

                    if sample_cost is not float('inf'):
                        # Cost to communicate from rover_pos_after_sample
                        min_comm_nav = float('inf')
                        if not self.comm_waypoints: # No communication waypoints exist
                            comm_cost = float('inf')
                        else:
                            for comm_wp in self.comm_waypoints:
                                nav_w_to_x = self.rover_shortest_paths.get(rover, {}).get(rover_pos_after_sample, {}).get(comm_wp, float('inf'))
                                if nav_w_to_x is not float('inf'):
                                    min_comm_nav = min(min_comm_nav, nav_w_to_x)

                            if min_comm_nav is not float('inf'):
                                comm_cost = min_comm_nav + 1 # nav + communicate
                            else:
                                comm_cost = float('inf') # Cannot reach any comm waypoint

                        if comm_cost is not float('inf'):
                             min_goal_cost = min(min_goal_cost, sample_cost + comm_cost)

                if min_goal_cost is float('inf'):
                    return float('inf') # This goal is unreachable
                total_cost += min_goal_cost

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint in communicated_rock:
                    continue # Goal already satisfied

                min_goal_cost = float('inf')

                # Find a suitable rover
                for rover in self.rovers:
                    if 'rock' not in self.rover_capabilities.get(rover, set()):
                        continue # Rover not equipped for rock

                    r_pos = rover_current_pos.get(rover)
                    if r_pos is None: continue # Rover location unknown

                    # Cost to get sample
                    sample_cost = float('inf')
                    rover_pos_after_sample = None

                    if waypoint in rover_has_rock.get(rover, set()):
                        # Rover already has the sample
                        sample_cost = 0
                        rover_pos_after_sample = r_pos # Rover stays at current pos

                    elif waypoint in rock_samples_remaining:
                        # Need to sample
                        nav_r_to_w = self.rover_shortest_paths.get(rover, {}).get(r_pos, {}).get(waypoint, float('inf'))
                        if nav_r_to_w is not float('inf'):
                            drop_cost = 1 if rover_store_status.get(rover) == 'full' else 0
                            sample_cost = nav_r_to_w + drop_cost + 1 # nav + (drop) + sample
                            rover_pos_after_sample = waypoint # Rover ends up at sample location

                    if sample_cost is not float('inf'):
                        # Cost to communicate from rover_pos_after_sample
                        min_comm_nav = float('inf')
                        if not self.comm_waypoints: # No communication waypoints exist
                            comm_cost = float('inf')
                        else:
                            for comm_wp in self.comm_waypoints:
                                nav_w_to_x = self.rover_shortest_paths.get(rover, {}).get(rover_pos_after_sample, {}).get(comm_wp, float('inf'))
                                if nav_w_to_x is not float('inf'):
                                    min_comm_nav = min(min_comm_nav, nav_w_to_x)

                            if min_comm_nav is not float('inf'):
                                comm_cost = min_comm_nav + 1 # nav + communicate
                            else:
                                comm_cost = float('inf') # Cannot reach any comm waypoint

                        if comm_cost is not float('inf'):
                             min_goal_cost = min(min_goal_cost, sample_cost + comm_cost)

                if min_goal_cost is float('inf'):
                    return float('inf') # This goal is unreachable
                total_cost += min_goal_cost

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                if (objective, mode) in communicated_image:
                    continue # Goal already satisfied

                min_goal_cost = float('inf')

                # Find a suitable rover, camera, image waypoint, calib waypoint
                for rover in self.rovers:
                    if 'imaging' not in self.rover_capabilities.get(rover, set()):
                        continue # Rover not equipped for imaging

                    r_pos = rover_current_pos.get(rover)
                    if r_pos is None: continue # Rover location unknown

                    for camera in self.rover_cameras.get(rover, set()):
                        if (camera, mode) not in self.camera_modes_supported:
                            continue # Camera does not support this mode

                        calib_target = self.camera_calibration_targets.get(camera)
                        suitable_image_wps = self.objective_visible_from.get(objective, set())
                        suitable_calib_wps = self.calibration_target_visible_from.get(calib_target, set()) if calib_target else set()

                        if not suitable_image_wps: continue # Cannot image this objective/mode

                        for image_wp in suitable_image_wps:
                            # Cost to get image ending at image_wp
                            image_cost = float('inf')
                            rover_pos_after_image = None

                            if (objective, mode) in rover_has_image.get(rover, set()):
                                # Rover already has the image
                                image_cost = 0
                                rover_pos_after_image = r_pos # Rover stays at current pos if already has it
                            else:
                                # Need to take image at image_wp
                                calib_needed = camera not in rover_calibrated_cameras.get(rover, set())

                                if calib_needed:
                                    if not suitable_calib_wps: continue # Cannot calibrate this camera

                                    min_calib_path_cost = float('inf')
                                    for calib_wp in suitable_calib_wps:
                                        # Path: r_pos -> w_cal -> p
                                        nav_r_to_wcal = self.rover_shortest_paths.get(rover, {}).get(r_pos, {}).get(calib_wp, float('inf'))
                                        nav_wcal_to_p = self.rover_shortest_paths.get(rover, {}).get(calib_wp, {}).get(image_wp, float('inf'))
                                        if nav_r_to_wcal is not float('inf') and nav_wcal_to_p is not float('inf'):
                                            # Cost: nav(r_pos, w_cal) + 1 (calibrate) + nav(w_cal, p)
                                            total_path_cost = nav_r_to_wcal + 1 + nav_wcal_to_p
                                            min_calib_path_cost = min(min_calib_path_cost, total_path_cost)

                                    if min_calib_path_cost is not float('inf'):
                                        # Total cost to get image via calibration: min_calib_path_cost + 1 (take_image)
                                        image_cost = min_calib_path_cost + 1
                                        rover_pos_after_image = image_wp # Ends up at image location p
                                else: # Already calibrated
                                    nav_r_to_p = self.rover_shortest_paths.get(rover, {}).get(r_pos, {}).get(image_wp, float('inf'))
                                    if nav_r_to_p is not float('inf'):
                                        image_cost = nav_r_to_p + 1 # nav + take_image
                                        rover_pos_after_image = image_wp # Ends up at image location p

                            if image_cost is not float('inf'):
                                # Cost to communicate from rover_pos_after_image
                                min_comm_nav = float('inf')
                                if not self.comm_waypoints: # No communication waypoints exist
                                    comm_cost = float('inf')
                                else:
                                    for comm_wp in self.comm_waypoints:
                                        nav_p_to_x = self.rover_shortest_paths.get(rover, {}).get(rover_pos_after_image, {}).get(comm_wp, float('inf'))
                                        if nav_p_to_x is not float('inf'):
                                            min_comm_nav = min(min_comm_nav, nav_p_to_x)

                                    if min_comm_nav is not float('inf'):
                                        comm_cost = min_comm_nav + 1 # nav + communicate
                                    else:
                                        comm_cost = float('inf') # Cannot reach any comm waypoint

                                if comm_cost is not float('inf'):
                                     min_goal_cost = min(min_goal_cost, image_cost + comm_cost)

                if min_goal_cost is float('inf'):
                    return float('inf') # This goal is unreachable
                total_cost += min_goal_cost

        return total_cost
