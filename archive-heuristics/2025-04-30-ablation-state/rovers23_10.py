from fnmatch import fnmatch
from collections import deque
# Assume heuristics.heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs(graph, start):
    """
    Perform Breadth-First Search to find shortest distances from a start node
    in a graph represented as an adjacency dictionary.
    """
    distances = {node: float('inf') for node in graph}
    if start in distances: # Ensure start node is in the graph
        distances[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the required number of actions to reach the goal state
    by summing the estimated costs for each unachieved goal predicate.
    The cost for each goal is estimated based on whether data needs to be collected/imaged,
    calibrated (for images), and communicated, including estimated navigation costs
    using precomputed shortest path distances.

    # Assumptions
    - Rover capabilities (equipment) are static.
    - Camera properties (on_board, supports, calibration_target) are static.
    - Lander location is static.
    - Waypoint visibility and rover traversability are static.
    - Soil/rock samples, once collected, are held by the rover and removed from the waypoint.
      They do not reappear at the waypoint.
    - Stores are binary (empty/full) and tied to a specific rover. The heuristic simplifies
      store management, assuming a rover can eventually get an empty store if needed for sampling.
    - Calibration is per camera on a specific rover and is consumed by taking an image.
    - Communication requires the rover with the data to be at a waypoint visible from the lander.
    - The heuristic estimates navigation cost for each goal independently, summing up
      estimated path segments (e.g., current -> sample -> comm). This may overestimate
      the cost as it doesn't account for shared travel paths between goals.
    - When multiple rovers or waypoints are suitable for a task segment (e.g., multiple equipped rovers,
      multiple communication waypoints), the heuristic uses the minimum distance found
      among the options.

    # Heuristic Initialization
    - Extracts static facts including rover capabilities, camera details, lander location,
      waypoint visibility, and rover traversability.
    - Builds a navigation graph for each rover based on `can_traverse` facts.
    - Precomputes all-pairs shortest path distances for each rover using BFS.
    - Identifies communication waypoints (visible from the lander).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each unachieved goal.

    For each unachieved goal `G` (e.g., `(communicated_soil_data ?w)`):
    1.  **Base Cost:** Add 1 for the final communication action (`communicate_..._data`).
    2.  **Check Data Availability:** Determine if the required data (`have_soil_analysis`, `have_rock_analysis`, or `have_image`) is already present on any rover in the current state.
    3.  **Estimate Collection/Imaging Cost (if data not available):**
        *   Add 1 for the collection/imaging action (`sample_soil`, `sample_rock`, or `take_image`).
        *   For image goals, identify if calibration is needed for a suitable camera/rover combo (i.e., the camera is not calibrated and is required for this image mode/objective). This check is done once for all unachieved image goals needing that specific camera/rover combo. If needed, add 1 for the `calibrate` action.
        *   Estimate navigation cost for the collection/imaging/calibration sequence:
            *   If calibration is needed: Estimate navigation from the current location of a suitable rover to a calibration waypoint, then from a calibration waypoint to an imaging waypoint, then from an imaging waypoint to a communication waypoint. Sum these minimum distances.
            *   If calibration is not needed: Estimate navigation from the current location of a suitable rover to an imaging waypoint, then from an imaging waypoint to a communication waypoint. Sum these minimum distances.
            *   For soil/rock samples: Estimate navigation from the current location of an equipped rover to the sample waypoint, then from the sample waypoint to a communication waypoint. Sum these minimum distances.
            *   Use precomputed shortest paths and find the minimum distance over all suitable rovers and relevant waypoints for each segment.
    4.  **Estimate Communication Navigation Cost (if data is available):**
        *   If the data is already available on a rover, estimate navigation from that rover's current location to a communication waypoint. Add this minimum distance.
    5.  **Sum Goal Costs:** Add the estimated costs (actions + navigation) for this specific goal to the total heuristic value.

    If any required waypoint is unreachable for the relevant rover(s) during navigation estimation, the heuristic returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Data structures for static information
        self.lander_location = None
        self.rover_capabilities = {} # rover -> set of capabilities (soil, rock, imaging)
        self.camera_info = {} # camera -> {rover: r, supports: set(modes), cal_target: obj}
        self.waypoint_visibility = {} # wp -> set of visible wps
        self.objective_visibility = {} # obj -> set of visible wps
        self.rover_traversability = {} # rover -> graph {wp -> set(neighbors)}
        self.rover_distances = {} # rover -> {start_wp -> {end_wp -> distance}}

        # First pass to identify objects by type (needed for graph building)
        rovers = set()
        waypoints = set()
        cameras = set()
        objectives = set()
        modes = set()
        landers = set()

        for fact in self.static_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty/malformed facts
             if parts[0] == 'at_lander': landers.add(parts[1]); waypoints.add(parts[2])
             elif parts[0] == 'at': rovers.add(parts[1]); waypoints.add(parts[2]) # Rovers initial location might be static
             elif parts[0] == 'equipped_for_soil_analysis': rovers.add(parts[1])
             elif parts[0] == 'equipped_for_rock_analysis': rovers.add(parts[1])
             elif parts[0] == 'equipped_for_imaging': rovers.add(parts[1])
             elif parts[0] == 'store_of': rovers.add(parts[2]) # Store object type not strictly needed
             elif parts[0] == 'can_traverse': rovers.add(parts[1]); waypoints.add(parts[2]); waypoints.add(parts[3])
             elif parts[0] == 'visible': waypoints.add(parts[1]); waypoints.add(parts[2])
             elif parts[0] == 'supports': cameras.add(parts[1]); modes.add(parts[2])
             elif parts[0] == 'visible_from': objectives.add(parts[1]); waypoints.add(parts[2])
             elif parts[0] == 'calibration_target': cameras.add(parts[1]); objectives.add(parts[2])
             elif parts[0] == 'on_board': cameras.add(parts[1]); rovers.add(parts[2])

        # Initialize data structures based on objects
        for r in rovers: self.rover_capabilities[r] = set(); self.rover_traversability[r] = {wp: set() for wp in waypoints}
        for c in cameras: self.camera_info[c] = {'rover': None, 'supports': set(), 'cal_target': None}
        for wp in waypoints: self.waypoint_visibility[wp] = set()
        for obj in objectives: self.objective_visibility[obj] = set()

        # Second pass to populate data structures
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'at_lander': self.lander_location = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis': self.rover_capabilities[parts[1]].add('soil')
            elif parts[0] == 'equipped_for_rock_analysis': self.rover_capabilities[parts[1]].add('rock')
            elif parts[0] == 'equipped_for_imaging': self.rover_capabilities[parts[1]].add('imaging')
            elif parts[0] == 'can_traverse': self.rover_traversability[parts[1]][parts[2]].add(parts[3])
            elif parts[0] == 'visible': self.waypoint_visibility[parts[1]].add(parts[2])
            elif parts[0] == 'supports': self.camera_info[parts[1]]['supports'].add(parts[2])
            elif parts[0] == 'visible_from': self.objective_visibility[parts[1]].add(parts[2])
            elif parts[0] == 'calibration_target': self.camera_info[parts[1]]['cal_target'] = parts[2]
            elif parts[0] == 'on_board': self.camera_info[parts[1]]['rover'] = parts[2]

        # Precompute distances for each rover
        for rover, graph in self.rover_traversability.items():
            self.rover_distances[rover] = {}
            for start_wp in graph:
                self.rover_distances[rover][start_wp] = bfs(graph, start_wp)

        # Identify communication waypoints (visible from lander)
        self.communication_waypoints = set()
        if self.lander_location:
             for wp, visible_wps in self.waypoint_visibility.items():
                 if self.lander_location in visible_wps:
                     self.communication_waypoints.add(wp)
             # Also check visibility the other way, although 'visible' is often symmetric
             for wp, visible_wps in self.waypoint_visibility.items():
                  if wp == self.lander_location:
                      self.communication_waypoints.update(visible_wps)


    def get_distance(self, rover, start_wp, end_wp):
        """Get precomputed distance for a specific rover between two waypoints."""
        if rover not in self.rover_distances or \
           start_wp not in self.rover_distances[rover] or \
           end_wp not in self.rover_distances[rover][start_wp]:
            return None # Waypoint or rover not in graph/distances
        dist = self.rover_distances[rover][start_wp][end_wp]
        return dist if dist != float('inf') else None # Return None if unreachable

    def get_min_distance_to_set(self, rover, start_wp, target_wps_set):
        """Get minimum precomputed distance for a rover from start_wp to any waypoint in the target set."""
        if not target_wps_set: return None # No target waypoints
        min_dist = float('inf')
        found_path = False
        for target_wp in target_wps_set:
            dist = self.get_distance(rover, start_wp, target_wp)
            if dist is not None:
                min_dist = min(min_dist, dist)
                found_path = True
        return min_dist if found_path else None # Return None if set is unreachable from start_wp

    def get_min_distance_from_set_to_set(self, rover_set, start_wps_set, target_wps_set):
        """
        Get minimum precomputed distance from any rover in rover_set
        starting at any waypoint in start_wps_set to any waypoint in target_wps_set.
        """
        if not rover_set or not start_wps_set or not target_wps_set: return None

        min_total_dist = float('inf')
        found_path = False

        for rover in rover_set:
            if rover not in self.rover_distances: continue

            for start_wp in start_wps_set:
                 if start_wp not in self.rover_distances[rover]: continue

                 min_dist_from_start_wp = self.get_min_distance_to_set(rover, start_wp, target_wps_set)
                 if min_dist_from_start_wp is not None:
                     min_total_dist = min(min_total_dist, min_dist_from_start_wp)
                     found_path = True

        return min_total_dist if found_path else None


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        total_cost = 0

        # Get current rover locations
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "?r", "?w"):
                rover_locations[get_parts(fact)[1]] = get_parts(fact)[2]

        # Identify cameras/rovers needing calibration for *any* image goal
        calibrations_to_do = set() # Store (camera, rover)
        cameras_rovers_needed_for_images = set() # Store (camera, rover)

        for goal in self.goals:
             if goal in state: continue
             if match(goal, "communicated_image_data", "?o", "?m"):
                 objective, mode = get_parts(goal)[1:]
                 has_data = any(match(f, "have_image", "*", objective, mode) for f in state)
                 if not has_data:
                     for camera, cam_info in self.camera_info.items():
                         rover = cam_info['rover']
                         if rover and 'imaging' in self.rover_capabilities.get(rover, set()) and mode in cam_info['supports']:
                             cameras_rovers_needed_for_images.add((camera, rover))

        for camera, rover in cameras_rovers_needed_for_images:
             if f"(calibrated {camera} {rover})" not in state:
                 cal_target = self.camera_info[camera]['cal_target']
                 if cal_target and self.objective_visibility.get(cal_target, set()):
                     calibrations_to_do.add((camera, rover))


        # Calculate cost for each unachieved goal
        for goal in self.goals:
            if goal in state: continue

            predicate, *args = get_parts(goal)
            goal_cost = 0

            # Cost for the final communication action
            goal_cost += 1

            # Navigation cost to communication point (if data already exists)
            # This will be calculated later, based on where the data is.

            if predicate == "communicated_soil_data":
                waypoint = args[0]
                has_data = any(match(f, "have_soil_analysis", "?r", waypoint) for f in state)

                if not has_data:
                    goal_cost += 1 # sample action
                    # Find equipped rovers
                    equipped_rovers = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps}
                    if not equipped_rovers: return float('inf') # Cannot sample

                    # Nav cost: curr -> sample -> comm
                    start_wps = {rover_locations[r] for r in equipped_rovers if r in rover_locations}
                    if not start_wps: return float('inf') # Equipped rovers location unknown

                    dist1 = self.get_min_distance_from_set_to_set(equipped_rovers, start_wps, {waypoint})
                    if dist1 is None: return float('inf')
                    goal_cost += dist1

                    dist2 = self.get_min_distance_from_set_to_set(equipped_rovers, {waypoint}, self.communication_waypoints)
                    if dist2 is None: return float('inf')
                    goal_cost += dist2

                else: # Rover already has data
                    # Find rovers with data
                    rovers_with_data = {get_parts(f)[1] for f in state if match(f, "have_soil_analysis", "?r", waypoint)}
                    if not rovers_with_data: return float('inf') # Should not happen if has_data is true

                    # Nav cost: curr -> comm
                    start_wps = {rover_locations[r] for r in rovers_with_data if r in rover_locations}
                    if not start_wps: return float('inf') # Rovers location unknown

                    dist = self.get_min_distance_from_set_to_set(rovers_with_data, start_wps, self.communication_waypoints)
                    if dist is None: return float('inf')
                    goal_cost += dist

            elif predicate == "communicated_rock_data":
                 # Similar logic as soil data
                 waypoint = args[0]
                 goal_cost += 1 # communicate action
                 has_data = any(match(f, "have_rock_analysis", "?r", waypoint) for f in state)

                 if not has_data:
                     goal_cost += 1 # sample action
                     equipped_rovers = {r for r, caps in self.rover_capabilities.items() if 'rock' in caps}
                     if not equipped_rovers: return float('inf')

                     start_wps = {rover_locations[r] for r in equipped_rovers if r in rover_locations}
                     if not start_wps: return float('inf')

                     dist1 = self.get_min_distance_from_set_to_set(equipped_rovers, start_wps, {waypoint})
                     if dist1 is None: return float('inf')
                     goal_cost += dist1

                     dist2 = self.get_min_distance_from_set_to_set(equipped_rovers, {waypoint}, self.communication_waypoints)
                     if dist2 is None: return float('inf')
                     goal_cost += dist2

                 else:
                     rovers_with_data = {get_parts(f)[1] for f in state if match(f, "have_rock_analysis", "?r", waypoint)}
                     if not rovers_with_data: return float('inf')

                     start_wps = {rover_locations[r] for r in rovers_with_data if r in rover_locations}
                     if not start_wps: return float('inf')

                     dist = self.get_min_distance_from_set_to_set(rovers_with_data, start_wps, self.communication_waypoints)
                     if dist is None: return float('inf')
                     goal_cost += dist

            elif predicate == "communicated_image_data":
                objective, mode = args
                goal_cost += 1 # communicate action
                has_data = any(match(f, "have_image", "?r", objective, mode) for f in state)

                if not has_data:
                    goal_cost += 1 # take_image action
                    img_wps = self.objective_visibility.get(objective, set())
                    if not img_wps: return float('inf')

                    # Find suitable imaging rovers/cameras
                    suitable_rovers_cameras = set()
                    for camera, cam_info in self.camera_info.items():
                        rover = cam_info['rover']
                        if rover and 'imaging' in self.rover_capabilities.get(rover, set()) and mode in cam_info['supports']:
                            suitable_rovers_cameras.add((rover, camera))
                    suitable_rovers = {r for r, c in suitable_rovers_cameras}
                    if not suitable_rovers: return float('inf')

                    # Check if calibration is needed for *any* suitable camera/rover for this image
                    cal_needed_for_this_image = False
                    cal_rover_camera = None # A specific one that needs calibrating
                    for r, c in suitable_rovers_cameras:
                        if (c, r) in calibrations_to_do:
                            cal_needed_for_this_image = True
                            cal_rover_camera = (r, c) # Pick one for distance calculation
                            break

                    if cal_needed_for_this_image:
                        goal_cost += 1 # calibrate action
                        camera, rover = cal_rover_camera # Use the specific one that needs calibrating
                        cal_target = self.camera_info[camera]['cal_target']
                        if not cal_target: return float('inf')
                        cal_wps = self.objective_visibility.get(cal_target, set())
                        if not cal_wps: return float('inf')

                        # Nav cost: curr -> cal -> img -> comm
                        start_wps = {rover_locations[rover]} if rover in rover_locations else set()
                        if not start_wps: return float('inf')

                        dist1 = self.get_min_distance_from_set_to_set({rover}, start_wps, cal_wps)
                        if dist1 is None: return float('inf')
                        goal_cost += dist1

                        dist2 = self.get_min_distance_from_set_to_set({rover}, cal_wps, img_wps)
                        if dist2 is None: return float('inf')
                        goal_cost += dist2

                        dist3 = self.get_min_distance_from_set_to_set({rover}, img_wps, self.communication_waypoints)
                        if dist3 is None: return float('inf')
                        goal_cost += dist3

                    else: # Calibration not needed for any suitable camera/rover
                        # Nav cost: curr -> img -> comm
                        start_wps = {rover_locations[r] for r in suitable_rovers if r in rover_locations}
                        if not start_wps: return float('inf')

                        dist1 = self.get_min_distance_from_set_to_set(suitable_rovers, start_wps, img_wps)
                        if dist1 is None: return float('inf')
                        goal_cost += dist1

                        dist2 = self.get_min_distance_from_set_to_set(suitable_rovers, img_wps, self.communication_waypoints)
                        if dist2 is None: return float('inf')
                        goal_cost += dist2

                else: # Rover already has image
                    # Find rovers with data
                    rovers_with_data = {get_parts(f)[1] for f in state if match(f, "have_image", "?r", objective, mode)}
                    if not rovers_with_data: return float('inf')

                    # Nav cost: curr -> comm
                    start_wps = {rover_locations[r] for r in rovers_with_data if r in rover_locations}
                    if not start_wps: return float('inf')

                    dist = self.get_min_distance_from_set_to_set(rovers_with_data, start_wps, self.communication_waypoints)
                    if dist is None: return float('inf')
                    goal_cost += dist

            total_cost += goal_cost

        # Ensure heuristic is 0 at goal (redundant check, but safe)
        if self.goals <= state:
             return 0

        return total_cost
