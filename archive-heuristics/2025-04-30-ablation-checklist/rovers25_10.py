import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    to all other reachable nodes in a graph.

    Args:
        graph: An adjacency list representation (dict: node -> set of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from start_node.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Handle nodes that might be in distances but not graph (e.g. non-waypoint objects)
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It breaks down the problem into subgoals for
    each required piece of data (soil, rock, image) and estimates the
    minimum steps for each, considering necessary intermediate steps like
    sampling, imaging, calibrating, and communicating, as well as navigation.
    Navigation costs are estimated using precomputed shortest paths on the
    rover's traversal graph.

    # Heuristic Initialization
    - Parses goal conditions to identify required data.
    - Parses static facts to understand the environment:
        - Lander location and communication waypoints.
        - Rover capabilities (soil, rock, imaging).
        - Camera information (on-board, supports, calibration target).
        - Store ownership.
        - Objective visibility for imaging and calibration.
        - Rover traversal graphs (`can_traverse`).
    - Precomputes shortest path distances between all pairs of waypoints
      for each rover using BFS on their respective traversal graphs.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of each rover.
    3. For each goal condition in the task:
        a. If the goal is already satisfied in the current state, add 0 cost for this goal.
        b. If the goal is `(communicated_soil_data ?w)` or `(communicated_rock_data ?w)`:
            - This goal requires sampling at waypoint `?w` and communicating the data.
            - Find the minimum cost among all rovers capable of this task.
            - For a capable rover `r`:
                - Cost starts at 0.
                - Find `r`'s current location.
                - If `(have_X_analysis ?r ?w)` is not in the state:
                    - Add shortest path cost from `r`'s current location to `?w`.
                    - Add 1 for the `sample_X` action.
                    - Check if `r`'s store is full. If yes, add 1 for the `drop` action (needed before sampling).
                    - Update `r`'s effective location to `?w`.
                - Find the closest communication waypoint `comm_wp` to `r`'s effective location.
                - Add shortest path cost from `r`'s effective location to `comm_wp`.
                - Add 1 for the `communicate_X_data` action.
                - The minimum cost for this goal is the minimum over all capable rovers.
        c. If the goal is `(communicated_image_data ?o ?m)`:
            - This goal requires imaging objective `?o` in mode `?m` and communicating the data.
            - Find the minimum cost among all rovers equipped for imaging that have a camera supporting mode `?m`.
            - For a capable rover `r` with a suitable camera `i`:
                - Cost starts at 0.
                - Find `r`'s current location.
                - If `(have_image ?r ?o ?m)` is not in the state:
                    - If `(calibrated ?i ?r)` is not in the state:
                        - Find the calibration target `t` for camera `i`.
                        - Find the closest calibration waypoint `cal_wp` (where `t` is visible from) to `r`'s current location.
                        - Add shortest path cost from `r`'s current location to `cal_wp`.
                        - Add 1 for the `calibrate` action.
                        - Update `r`'s effective location to `cal_wp`.
                    - Find the closest imaging waypoint `img_wp` (where `?o` is visible from) to `r`'s effective location.
                    - Add shortest path cost from `r`'s effective location to `img_wp`.
                    - Add 1 for the `take_image` action.
                    - Update `r`'s effective location to `img_wp`.
                - Find the closest communication waypoint `comm_wp` to `r`'s effective location.
                - Add shortest path cost from `r`'s effective location to `comm_wp`.
                - Add 1 for the `communicate_image_data` action.
                - The minimum cost for this goal is the minimum over all capable rover/camera pairs.
        d. Add the minimum cost for the current goal to the total heuristic cost.
    4. Return the total heuristic cost.

    # Assumptions
    - Navigation cost between waypoints is the shortest path distance.
    - Each unachieved goal contributes independently to the heuristic (additive).
    - For sampling, a rover needs an empty store *before* sampling. If full, a `drop` is needed.
    - For imaging, a camera needs to be calibrated *before* taking an image. Calibration is consumed.
    - Assumes at least one lander exists and its location is static.
    - Assumes relevant waypoints (sample, imaging, calibration, communication) are reachable by at least one capable rover. If not, the cost for that goal will be infinite.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Static Facts ---
        self.lander_location = None
        self.comm_waypoints = set() # Waypoints visible from the lander
        self.rover_capabilities = collections.defaultdict(set) # rover -> {soil, rock, imaging}
        self.rover_cameras = collections.defaultdict(list) # rover -> [camera1, camera2, ...]
        self.camera_info = {} # camera -> {supports: {mode1, mode2}, cal_target: objective}
        self.store_info = {} # store -> rover
        self.objective_imaging_wps = collections.defaultdict(set) # objective -> {wp1, wp2}
        self.objective_cal_wps = collections.defaultdict(set) # objective -> {wp1, wp2} (for calibration targets)
        self.rover_graph = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> wp -> {neighbor_wp1, neighbor_wp2}
        self.all_waypoints = set()

        lander_wp = None
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                lander, wp = parts[1], parts[2]
                self.lander_location = wp
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                # Visibility is symmetric
                # Note: This is used for communication visibility, not necessarily traversal
                # We'll use can_traverse for rover movement graph
            elif predicate == "can_traverse":
                 rover, wp1, wp2 = parts[1], parts[2], parts[3]
                 self.rover_graph[rover][wp1].add(wp2)
                 self.rover_graph[rover][wp2].add(wp1) # Assuming traversal is symmetric
                 self.all_waypoints.add(wp1)
                 self.all_waypoints.add(wp2)
            elif predicate == "equipped_for_soil_analysis":
                self.rover_capabilities[parts[1]].add('soil')
            elif predicate == "equipped_for_rock_analysis":
                self.rover_capabilities[parts[1]].add('rock')
            elif predicate == "equipped_for_imaging":
                self.rover_capabilities[parts[1]].add('imaging')
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.store_info[store] = rover
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.rover_cameras[rover].append(camera)
                if camera not in self.camera_info:
                    self.camera_info[camera] = {'supports': set(), 'cal_target': None}
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'supports': set(), 'cal_target': None}
                self.camera_info[camera]['supports'].add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                if camera not in self.camera_info:
                     self.camera_info[camera] = {'supports': set(), 'cal_target': None}
                self.camera_info[camera]['cal_target'] = objective
            elif predicate == "visible_from":
                objective, wp = parts[1], parts[2]
                self.objective_imaging_wps[objective].add(wp)
                # Check if this objective is a calibration target for any camera
                for cam, info in self.camera_info.items():
                    if info['cal_target'] == objective:
                        self.objective_cal_wps[objective].add(wp) # This wp is a calibration wp for this objective/target

        # Precompute communication waypoints (visible from lander)
        if self.lander_location:
             for fact in static_facts:
                 if match(fact, "visible", "*", self.lander_location):
                     self.comm_waypoints.add(get_parts(fact)[1])
                 if match(fact, "visible", self.lander_location, "*"):
                      self.comm_waypoints.add(get_parts(fact)[2])


        # Precompute shortest path distances for each rover
        self.rover_distances = {} # rover -> start_wp -> end_wp -> distance
        for rover, graph in self.rover_graph.items():
            self.rover_distances[rover] = {}
            # BFS from every waypoint the rover can potentially reach
            all_reachable_wps = set(graph.keys())
            for start_wp in all_reachable_wps:
                 self.rover_distances[rover][start_wp] = bfs(graph, start_wp)

        # Helper to find the closest waypoint from a set
    def _closest_waypoint(self, rover, current_wp, target_wps):
        """Finds the minimum shortest path distance from current_wp to any wp in target_wps for the given rover."""
        min_dist = float('inf')
        if rover not in self.rover_distances or current_wp not in self.rover_distances[rover]:
             # Rover cannot navigate from current_wp or rover has no navigation graph
             return float('inf')

        distances_from_current = self.rover_distances[rover][current_wp]

        for target_wp in target_wps:
            if target_wp in distances_from_current:
                 min_dist = min(min_dist, distances_from_current[target_wp])
        return min_dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Find current location of each rover
        rover_locations = {}
        rover_stores_full = {}
        rover_calibrated_cameras = collections.defaultdict(set) # rover -> {camera1, camera2}
        rover_soil_samples = collections.defaultdict(set) # rover -> {wp1, wp2}
        rover_rock_samples = collections.defaultdict(set) # rover -> {wp1, wp2}
        rover_images = collections.defaultdict(set) # rover -> {(obj1, mode1), (obj2, mode2)}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at" and parts[1].startswith("rover"):
                rover_locations[parts[1]] = parts[2]
            elif predicate == "full" and parts[1].startswith("rover") and parts[1].endswith("store"):
                 rover_stores_full[self.store_info.get(parts[1])] = True # Map store to rover
            elif predicate == "calibrated":
                 camera, rover = parts[1], parts[2]
                 rover_calibrated_cameras[rover].add(camera)
            elif predicate == "have_soil_analysis":
                 rover, wp = parts[1], parts[2]
                 rover_soil_samples[rover].add(wp)
            elif predicate == "have_rock_analysis":
                 rover, wp = parts[1], parts[2]
                 rover_rock_samples[rover].add(wp)
            elif predicate == "have_image":
                 rover, obj, mode = parts[1], parts[2], parts[3]
                 rover_images[rover].add((obj, mode))

        # Iterate through goals and estimate cost for each unachieved one
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data" or predicate == "communicated_rock_data":
                sample_wp = parts[1]
                sample_type = 'soil' if predicate == "communicated_soil_data" else 'rock'
                have_sample_pred = f"(have_{sample_type}_analysis * {sample_wp})"

                goal_cost = float('inf')

                # Check if any rover already has the sample
                has_sample_rover = None
                for rover, samples in (rover_soil_samples if sample_type == 'soil' else rover_rock_samples).items():
                    if sample_wp in samples:
                        has_sample_rover = rover
                        break

                if has_sample_rover:
                    # Sample is collected, need to communicate
                    rover = has_sample_rover
                    current_wp = rover_locations.get(rover)
                    if current_wp:
                        nav_cost = self._closest_waypoint(rover, current_wp, self.comm_waypoints)
                        if nav_cost != float('inf'):
                            goal_cost = min(goal_cost, nav_cost + 1) # +1 for communicate
                else:
                    # Sample not collected, need to sample and communicate
                    required_capability = sample_type
                    for rover in self.rover_capabilities:
                        if required_capability in self.rover_capabilities[rover]:
                            current_wp = rover_locations.get(rover)
                            if current_wp:
                                # Cost to sample
                                nav_to_sample_cost = self._closest_waypoint(rover, current_wp, {sample_wp})
                                if nav_to_sample_cost != float('inf'):
                                    sample_action_cost = 1 # sample action
                                    drop_cost = 1 if rover_stores_full.get(rover, False) else 0 # need to drop if store is full
                                    cost_to_get_sample = nav_to_sample_cost + drop_cost + sample_action_cost

                                    # Cost to communicate after sampling (rover is now at sample_wp)
                                    nav_to_comm_cost = self._closest_waypoint(rover, sample_wp, self.comm_waypoints)
                                    if nav_to_comm_cost != float('inf'):
                                         communicate_action_cost = 1 # communicate action
                                         total_rover_cost = cost_to_get_sample + nav_to_comm_cost + communicate_action_cost
                                         goal_cost = min(goal_cost, total_rover_cost)


            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                have_image_pred = f"(have_image * {objective} {mode})"

                goal_cost = float('inf')

                # Check if any rover already has the image
                has_image_rover = None
                for rover, images in rover_images.items():
                    if (objective, mode) in images:
                        has_image_rover = rover
                        break

                if has_image_rover:
                    # Image is taken, need to communicate
                    rover = has_image_rover
                    current_wp = rover_locations.get(rover)
                    if current_wp:
                        nav_cost = self._closest_waypoint(rover, current_wp, self.comm_waypoints)
                        if nav_cost != float('inf'):
                            goal_cost = min(goal_cost, nav_cost + 1) # +1 for communicate
                else:
                    # Image not taken, need to calibrate, take image, and communicate
                    required_capability = 'imaging'
                    for rover in self.rover_capabilities:
                        if required_capability in self.rover_capabilities[rover]:
                            current_wp = rover_locations.get(rover)
                            if current_wp:
                                for camera in self.rover_cameras.get(rover, []):
                                    cam_info = self.camera_info.get(camera)
                                    if cam_info and mode in cam_info['supports']:
                                        # This rover/camera can potentially achieve the goal
                                        rover_camera_cost = 0
                                        effective_wp = current_wp # Track rover's location throughout the steps

                                        # Cost to calibrate (if needed)
                                        if camera not in rover_calibrated_cameras.get(rover, set()):
                                            cal_target = cam_info['cal_target']
                                            if cal_target and cal_target in self.objective_cal_wps:
                                                cal_wps = self.objective_cal_wps[cal_target]
                                                nav_to_cal_cost = self._closest_waypoint(rover, effective_wp, cal_wps)
                                                if nav_to_cal_cost != float('inf'):
                                                    rover_camera_cost += nav_to_cal_cost + 1 # +1 for calibrate
                                                    # After calibration, rover is at a calibration waypoint
                                                    # For simplicity, assume it's at the *closest* one found
                                                    # A more accurate heuristic might pick the one closest to the *next* step (imaging wp)
                                                    # But finding that optimal intermediate wp is complex.
                                                    # Let's just update effective_wp to *one* of the cal_wps reachable with min cost.
                                                    # Find the actual waypoint that gives min_dist
                                                    min_dist = float('inf')
                                                    next_effective_wp = None
                                                    if rover in self.rover_distances and effective_wp in self.rover_distances[rover]:
                                                        distances_from_current = self.rover_distances[rover][effective_wp]
                                                        for cal_wp in cal_wps:
                                                            if cal_wp in distances_from_current and distances_from_current[cal_wp] < min_dist:
                                                                min_dist = distances_from_current[cal_wp]
                                                                next_effective_wp = cal_wp
                                                    if next_effective_wp:
                                                         effective_wp = next_effective_wp
                                                    else: # Should not happen if nav_to_cal_cost was finite, but safety check
                                                         rover_camera_cost = float('inf') # Cannot calibrate
                                            else:
                                                rover_camera_cost = float('inf') # No calibration target or no visible cal waypoints
                                        # If cost is already inf, skip
                                        if rover_camera_cost == float('inf'):
                                             continue

                                        # Cost to take image
                                        if objective in self.objective_imaging_wps:
                                            img_wps = self.objective_imaging_wps[objective]
                                            nav_to_img_cost = self._closest_waypoint(rover, effective_wp, img_wps)
                                            if nav_to_img_cost != float('inf'):
                                                rover_camera_cost += nav_to_img_cost + 1 # +1 for take_image
                                                # After taking image, rover is at an imaging waypoint
                                                # Update effective_wp similarly to calibration
                                                min_dist = float('inf')
                                                next_effective_wp = None
                                                if rover in self.rover_distances and effective_wp in self.rover_distances[rover]:
                                                    distances_from_current = self.rover_distances[rover][effective_wp]
                                                    for img_wp in img_wps:
                                                        if img_wp in distances_from_current and distances_from_current[img_wp] < min_dist:
                                                            min_dist = distances_from_current[img_wp]
                                                            next_effective_wp = img_wp
                                                if next_effective_wp:
                                                    effective_wp = next_effective_wp
                                                else: # Should not happen if nav_to_img_cost was finite
                                                    rover_camera_cost = float('inf') # Cannot take image
                                            else:
                                                rover_camera_cost = float('inf') # No visible imaging waypoints
                                        else:
                                            rover_camera_cost = float('inf') # Objective not visible from anywhere

                                        # If cost is already inf, skip
                                        if rover_camera_cost == float('inf'):
                                             continue

                                        # Cost to communicate
                                        nav_to_comm_cost = self._closest_waypoint(rover, effective_wp, self.comm_waypoints)
                                        if nav_to_comm_cost != float('inf'):
                                            rover_camera_cost += nav_to_comm_cost + 1 # +1 for communicate
                                            goal_cost = min(goal_cost, rover_camera_cost)
                                        # else: Cannot reach communication point, cost remains inf for this rover/camera path

            # Add the minimum cost for this goal to the total
            if goal_cost != float('inf'):
                total_cost += goal_cost
            else:
                # If a goal is unreachable, the heuristic should ideally be infinity.
                # For greedy best-first, a very large number works too.
                # Let's return infinity if any goal is unreachable.
                return float('inf')

        return total_cost

