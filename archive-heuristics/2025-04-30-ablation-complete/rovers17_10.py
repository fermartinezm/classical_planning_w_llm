import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys

# Use a large number to represent infinity for distance
INF = sys.maxsize

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all other nodes."""
    distances = {node: INF for node in graph}
    if start_node in distances:
        distances[start_node] = 0
        queue = collections.deque([start_node])
        while queue:
            current_node = queue.popleft()
            if distances[current_node] == INF: # Should not happen if start_node is in graph
                continue
            for neighbor in graph.get(current_node, set()):
                if distances[neighbor] == INF:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def min_dist_set_to_set(start_wps, end_wps, waypoint_distances):
    """Min distance from any waypoint in start_wps to any waypoint in end_wps."""
    min_d = INF
    if not start_wps or not end_wps:
        return INF # Cannot navigate if no start or end points

    for start_wp in start_wps:
        for end_wp in end_wps:
            dist = waypoint_distances.get((start_wp, end_wp), INF)
            min_d = min(min_d, dist)
    return min_d

def min_dist_rover_set_to_set(rover_locations, rovers, target_wps, waypoint_distances):
    """Min distance from any specified rover's current location to any target waypoint."""
    rover_wps = {rover_locations[r] for r in rovers if r in rover_locations}
    return min_dist_set_to_set(rover_wps, target_wps, waypoint_distances)


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal
    conditions. It sums the estimated costs for each unachieved goal independently.
    The cost for each goal includes the specific actions needed (sample, calibrate,
    take_image, communicate) plus the estimated minimum navigation steps required
    to get a suitable rover to the necessary locations (sample site, calibration
    target viewpoint, imaging viewpoint, lander communication point).

    # Assumptions
    - Navigation is possible between any two waypoints visible to each other,
      assuming at least one rover can traverse that edge. The heuristic uses
      a single waypoint graph based on 'visible' predicates for distance calculation,
      assuming symmetry and universal traversability for simplicity.
    - The cost of each action (navigate, sample, drop, calibrate, take_image,
      communicate) is 1.
    - For sampling goals, it assumes a rover needs to navigate to the sample,
      sample, navigate to a communication point, and communicate. A drop action
      is added if any equipped rover has a full store.
    - For imaging goals, it assumes a rover needs to navigate to a calibration
      point, calibrate, navigate to an imaging point, take the image, navigate
      to a communication point, and communicate. Recalibration is not explicitly
      modeled for multiple images.
    - If a required waypoint (sample, imaging, calibration, communication) is
      unreachable from the current or intermediate location based on the
      precomputed distances, the heuristic considers the goal unreachable
      and returns infinity.
    - The heuristic does not consider resource contention (e.g., multiple rovers
      needing the same store or camera simultaneously).

    # Heuristic Initialization
    The heuristic precomputes and stores static information from the task:
    - The lander's location.
    - Waypoints visible from the lander's location (communication points).
    - Rover capabilities (soil, rock, imaging).
    - Rover-store mapping.
    - Camera information (on-board rover, supported modes, calibration target).
    - Objective visibility from waypoints.
    - Calibration targets for cameras.
    - Calibration waypoints for cameras (derived from calibration targets and objective visibility).
    - A waypoint graph based on 'visible' predicates.
    - All-pairs shortest path distances between waypoints using BFS on the waypoint graph.
    - The set of goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to extract dynamic facts (rover locations, sample
       presence, collected data, store status, camera calibration, communicated data).
    2. Initialize the total heuristic value `h` to 0.
    3. Initialize an `unreachable` flag to False.
    4. Iterate through each goal condition specified in the task.
    5. For each goal condition:
       - If the goal is already satisfied in the current state, continue to the next goal.
       - If the goal is `(communicated_soil_data ?w)`:
         - Add 1 for the `communicate_soil_data` action.
         - Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r`.
           - If yes: Find rovers that have the sample. Add the minimum navigation
             cost from any of these rovers' current locations to any lander-visible waypoint.
           - If no: The sample needs to be collected.
             - Add 1 for the `sample_soil` action.
             - Find rovers equipped for soil analysis. If none, mark as unreachable.
             - Add the minimum navigation cost from any equipped rover's current
               location to waypoint `?w`.
             - Add the minimum navigation cost from waypoint `?w` to any lander-visible waypoint.
             - Add 1 for a `drop` action if any equipped soil rover's store is full.
       - If the goal is `(communicated_rock_data ?w)`: Similar logic as soil data,
         using rock-specific predicates and capabilities.
       - If the goal is `(communicated_image_data ?o ?m)`:
         - Add 1 for the `communicate_image_data` action.
         - Check if `(have_image ?r ?o ?m)` is true for any rover `?r`.
           - If yes: Find rovers that have the image. Add the minimum navigation
             cost from any of these rovers' current locations to any lander-visible waypoint.
           - If no: The image needs to be taken.
             - Add 1 for the `calibrate` action.
             - Add 1 for the `take_image` action.
             - Find suitable rover/camera pairs (equipped for imaging, camera on board,
               supports mode). If none, mark as unreachable.
             - Find imaging waypoints for objective `?o`. If none, mark as unreachable.
             - Find calibration waypoints for suitable cameras. If any suitable
               camera has no calibration waypoint, mark as unreachable.
             - Add the minimum navigation cost from any suitable rover's current
               location to any calibration waypoint for its camera.
             - Add the minimum navigation cost from any calibration waypoint (for
               a suitable camera) to any imaging waypoint for `?o`.
             - Add the minimum navigation cost from any imaging waypoint for `?o`
               to any lander-visible waypoint.
       - If any navigation cost calculation resulted in infinity, set the `unreachable` flag to True.
    6. If the `unreachable` flag is True after checking all goals, return infinity.
    7. Otherwise, return the total accumulated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Precompute static information
        self.lander_location = None
        self.lander_visible_waypoints = set()
        self.rover_capabilities = collections.defaultdict(set)
        self.rover_stores = {} # rover -> store
        self.camera_info = collections.defaultdict(dict) # {camera: {'on_board': rover, 'supports': {mode}, 'calibration_target': obj}}
        self.objective_visibility = collections.defaultdict(set) # objective -> {waypoint}
        self.calibration_targets = {} # camera -> objective
        self.calibration_waypoints = collections.defaultdict(set) # camera -> {waypoint}
        self.waypoints = set()
        self.waypoint_graph = collections.defaultdict(set) # wp -> {neighbor_wp}
        self.waypoint_distances = {} # (wp1, wp2) -> dist

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_location = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities[parts[1]].add('equipped_for_soil_analysis')
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities[parts[1]].add('equipped_for_rock_analysis')
            elif parts[0] == 'equipped_for_imaging':
                self.rover_capabilities[parts[1]].add('equipped_for_imaging')
            elif parts[0] == 'store_of':
                self.rover_stores[parts[2]] = parts[1] # store -> rover, correct mapping
            elif parts[0] == 'on_board':
                self.camera_info[parts[1]]['on_board'] = parts[2]
            elif parts[0] == 'supports':
                if 'supports' not in self.camera_info[parts[1]]:
                    self.camera_info[parts[1]]['supports'] = set()
                self.camera_info[parts[1]]['supports'].add(parts[2])
            elif parts[0] == 'calibration_target':
                self.camera_info[parts[1]]['calibration_target'] = parts[2]
                self.calibration_targets[parts[1]] = parts[2]
            elif parts[0] == 'visible_from':
                self.objective_visibility[parts[1]].add(parts[2])
            elif parts[0] == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_graph[wp1].add(wp2)
                self.waypoint_graph[wp2].add(wp1) # Assuming visible is symmetric
                self.waypoints.add(wp1)
                self.waypoints.add(wp2)
            elif parts[0] == 'can_traverse':
                 # Assuming can_traverse aligns with visible for distance calculation
                 pass # Ignore can_traverse for graph building, rely on visible

        # Compute lander visible waypoints
        if self.lander_location:
             # Find waypoints X such that (visible X lander_location)
             for wp1, neighbors in self.waypoint_graph.items():
                 if self.lander_location in neighbors:
                     self.lander_visible_waypoints.add(wp1)

        # Compute calibration waypoints for each camera
        for camera, target_obj in self.calibration_targets.items():
            if target_obj in self.objective_visibility:
                self.calibration_waypoints[camera] = self.objective_visibility[target_obj]

        # Compute all-pairs shortest paths
        for start_wp in self.waypoints:
            distances = bfs(self.waypoint_graph, start_wp)
            for end_wp, dist in distances.items():
                self.waypoint_distances[(start_wp, end_wp)] = dist

        # Store goal conditions as tuples for easy lookup
        self.goal_conditions = set()
        for goal in self.goals:
            self.goal_conditions.add(tuple(get_parts(goal)))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state facts
        rover_locations = {} # rover -> waypoint
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        rover_soil_samples = collections.defaultdict(set) # rover -> {waypoint}
        rover_rock_samples = collections.defaultdict(set) # rover -> {waypoint}
        rover_images = collections.defaultdict(set) # rover -> {(obj, mode)}
        store_status = {} # store -> 'empty'/'full'
        camera_calibrated = collections.defaultdict(bool) # camera -> bool
        communicated_soil = set() # {waypoint}
        communicated_rock = set() # {waypoint}
        communicated_image = set() # {(obj, mode)}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.rover_capabilities: # Only track rover locations
                rover_locations[parts[1]] = parts[2]
            elif parts[0] == 'at_soil_sample':
                soil_samples_at.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                rock_samples_at.add(parts[1])
            elif parts[0] == 'have_soil_analysis':
                rover_soil_samples[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis':
                rover_rock_samples[parts[1]].add(parts[2])
            elif parts[0] == 'have_image':
                rover_images[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'empty':
                store_status[parts[1]] = 'empty'
            elif parts[0] == 'full':
                store_status[parts[1]] = 'full'
            elif parts[0] == 'calibrated':
                camera_calibrated[parts[1]] = True
            elif parts[0] == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))

        h = 0
        unreachable = False # Flag to indicate if any goal is unreachable

        # Iterate through goals
        for goal_tuple in self.goal_conditions:
            predicate = goal_tuple[0]

            if predicate == 'communicated_soil_data':
                waypoint = goal_tuple[1]
                if waypoint not in communicated_soil:
                    # Goal: communicate soil data from waypoint
                    h += 1 # Cost for communicate action

                    # Check if sample is already collected by any rover
                    has_sample = any(waypoint in rover_soil_samples.get(r, set()) for r in rover_locations)

                    if has_sample:
                        # Sample exists, need to move rover with sample to comm point
                        rovers_with_sample = {r for r, wps in rover_soil_samples.items() if waypoint in wps}
                        nav_cost_to_comm = min_dist_rover_set_to_set(rover_locations, rovers_with_sample, self.lander_visible_waypoints, self.waypoint_distances)
                        if nav_cost_to_comm == INF: unreachable = True
                        h += nav_cost_to_comm
                    else:
                        # Sample does not exist, need to sample first
                        h += 1 # Cost for sample action

                        # Find equipped soil rovers
                        equipped_soil_rovers = {r for r, caps in self.rover_capabilities.items() if 'equipped_for_soil_analysis' in caps}
                        if not equipped_soil_rovers: # No rover can sample soil
                            unreachable = True
                            break # Cannot achieve this goal

                        # Nav cost to sample location
                        nav_cost_to_sample = min_dist_rover_set_to_set(rover_locations, equipped_soil_rovers, {waypoint}, self.waypoint_distances)
                        if nav_cost_to_sample == INF: unreachable = True
                        h += nav_cost_to_sample

                        # Nav cost from sample location to communication point
                        nav_cost_sample_to_comm = min_dist_set_to_set({waypoint}, self.lander_visible_waypoints, self.waypoint_distances)
                        if nav_cost_sample_to_comm == INF: unreachable = True
                        h += nav_cost_sample_to_comm

                        # Drop cost if any equipped rover has a full store (simplification)
                        # We check if *any* equipped rover has a full store, assuming one of them will be used.
                        if any(store_status.get(self.rover_stores.get(r)) == 'full' for r in equipped_soil_rovers):
                             h += 1 # Cost for drop action

            elif predicate == 'communicated_rock_data':
                waypoint = goal_tuple[1]
                if waypoint not in communicated_rock:
                    # Goal: communicate rock data from waypoint
                    h += 1 # Cost for communicate action

                    # Check if sample is already collected by any rover
                    has_sample = any(waypoint in rover_rock_samples.get(r, set()) for r in rover_locations)

                    if has_sample:
                        # Sample exists, need to move rover with sample to comm point
                        rovers_with_sample = {r for r, wps in rover_rock_samples.items() if waypoint in wps}
                        nav_cost_to_comm = min_dist_rover_set_to_set(rover_locations, rovers_with_sample, self.lander_visible_waypoints, self.waypoint_distances)
                        if nav_cost_to_comm == INF: unreachable = True
                        h += nav_cost_to_comm
                    else:
                        # Sample does not exist, need to sample first
                        h += 1 # Cost for sample action

                        # Find equipped rock rovers
                        equipped_rock_rovers = {r for r, caps in self.rover_capabilities.items() if 'equipped_for_rock_analysis' in caps}
                        if not equipped_rock_rovers: # No rover can sample rock
                            unreachable = True
                            break # Cannot achieve this goal

                        # Nav cost to sample location
                        nav_cost_to_sample = min_dist_rover_set_to_set(rover_locations, equipped_rock_rovers, {waypoint}, self.waypoint_distances)
                        if nav_cost_to_sample == INF: unreachable = True
                        h += nav_cost_to_sample

                        # Nav cost from sample location to communication point
                        nav_cost_sample_to_comm = min_dist_set_to_set({waypoint}, self.lander_visible_waypoints, self.waypoint_distances)
                        if nav_cost_sample_to_comm == INF: unreachable = True
                        h += nav_cost_sample_to_comm

                        # Drop cost if any equipped rover has a full store (simplification)
                        if any(store_status.get(self.rover_stores.get(r)) == 'full' for r in equipped_rock_rovers):
                             h += 1 # Cost for drop action


            elif predicate == 'communicated_image_data':
                objective, mode = goal_tuple[1], goal_tuple[2]
                if (objective, mode) not in communicated_image:
                    # Goal: communicate image data for objective/mode
                    h += 1 # Cost for communicate action

                    # Check if image is already taken by any rover
                    has_image = any((objective, mode) in rover_images.get(r, set()) for r in rover_locations)

                    if has_image:
                        # Image exists, need to move rover with image to comm point
                        rovers_with_image = {r for r, images in rover_images.items() if (objective, mode) in images}
                        nav_cost_to_comm = min_dist_rover_set_to_set(rover_locations, rovers_with_image, self.lander_visible_waypoints, self.waypoint_distances)
                        if nav_cost_to_comm == INF: unreachable = True
                        h += nav_cost_to_comm
                    else:
                        # Image does not exist, need to take image first
                        h += 1 # Cost for calibrate action
                        h += 1 # Cost for take_image action

                        # Find suitable rover/camera pairs for this mode
                        suitable_rc_pairs = [(r, i) for r, caps in self.rover_capabilities.items() if 'equipped_for_imaging' in caps
                                             for i, info in self.camera_info.items() if info.get('on_board') == r and mode in info.get('supports', set())]

                        if not suitable_rc_pairs: # No suitable rover/camera
                            unreachable = True
                            break # Cannot achieve this goal

                        # Find imaging waypoints for the objective
                        img_wps = self.objective_visibility.get(objective, set())
                        if not img_wps: # No waypoint to view objective
                            unreachable = True
                            break # Cannot achieve this goal

                        # Find calibration waypoints for suitable cameras
                        cal_wps_by_camera = {i: self.calibration_waypoints.get(i, set()) for (r, i) in suitable_rc_pairs}
                        # Check if all suitable cameras have calibration waypoints
                        if any(not cal_wps_by_camera.get(i, set()) for (r, i) in suitable_rc_pairs):
                             unreachable = True
                             break # Cannot achieve this goal

                        # Nav cost to calibration waypoint
                        suitable_rover_set = {r for r, i in suitable_rc_pairs}
                        all_cal_wps_for_suitable_rovers = set().union(*(cal_wps_by_camera.get(i, set()) for r, i in suitable_rc_pairs))
                        nav_cost_to_cal = min_dist_rover_set_to_set(rover_locations, suitable_rover_set, all_cal_wps_for_suitable_rovers, self.waypoint_distances)
                        if nav_cost_to_cal == INF: unreachable = True
                        h += nav_cost_to_cal

                        # Nav cost from calibration waypoint to imaging waypoint
                        nav_cost_cal_to_img = min_dist_set_to_set(all_cal_wps_for_suitable_rovers, img_wps, self.waypoint_distances)
                        if nav_cost_cal_to_img == INF: unreachable = True
                        h += nav_cost_cal_to_img

                        # Nav cost from imaging waypoint to communication point
                        nav_cost_img_to_comm = min_dist_set_to_set(img_wps, self.lander_visible_waypoints, self.waypoint_distances)
                        if nav_cost_img_to_comm == INF: unreachable = True
                        h += nav_cost_img_to_comm

            # If any goal was found unreachable, stop and return infinity
            if unreachable:
                return INF

        # If the loop finishes and unreachable is False, h is the sum of costs for all unachieved goals.
        # If h is 0, all goals were achieved (goal state).
        return h

