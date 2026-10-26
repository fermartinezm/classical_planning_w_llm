import sys
from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

# Define a large number to represent infinity or unreachable goals
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve each
    pending goal condition (communicating soil, rock, or image data).
    It sums the minimum estimated cost for each unachieved goal, considering
    rover capabilities, locations, required samples/images, and navigation costs.
    Navigation costs are precomputed using BFS on the rover-specific traversal graphs.

    # Heuristic Initialization
    - Parses static facts to identify rovers, waypoints, landers, cameras,
      objectives, capabilities, store ownership, camera properties (on-board,
      supports, target), and visibility/traversal graphs.
    - Computes all-pairs shortest paths for each rover on its traversal graph
      using BFS, storing distances.
    - Stores goal conditions.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Initialize total heuristic cost `h` to 0.
    2. Parse the current state to get locations of rovers, status of stores,
       collected data, camera calibration status, and remaining samples.
    3. Iterate through each goal condition defined in the task.
    4. If a goal condition is already true in the current state, it contributes 0 to `h`.
    5. If a goal condition is not true, estimate the minimum cost to achieve it:
       - Initialize goal cost `goal_h` to a large number (representing infinity).
       - **For Soil/Rock Communication Goal `(communicated_X_data ?w)`:**
         - Identify the target waypoint `?w`.
         - Check if the corresponding analysis data `(have_X_analysis ?r ?w)` is held by *any* rover `?r` in the current state.
           - If Yes (Data Collected):
             - Find the rover `?r` holding the data.
             - Get its current location `current_wp`.
             - Find a waypoint `comm_wp` visible from any lander location.
             - Calculate the navigation cost from `current_wp` to `comm_wp` for rover `?r` using precomputed distances.
             - The cost for this path is `nav_cost + 1` (communicate action).
             - Update `goal_h` with the minimum cost found over all rovers holding the data and all possible communication waypoints.
           - If No (Data Not Collected):
             - Check if the sample `(at_X_sample ?w)` still exists at `?w` in the current state.
             - If Yes (Sample Exists):
               - Find rovers `?r` equipped for the analysis type (soil/rock).
               - For each equipped rover `?r`:
                 - Get its current location `current_wp`.
                 - Check if its store is full (`(full store_of_r)`). Add 1 to cost if full (for drop action).
                 - Calculate navigation cost from `current_wp` to `?w` (`nav_sample_cost`).
                 - If `nav_sample_cost` is finite:
                   - Add 1 for the sample action (`sample_cost`).
                   - Find a waypoint `comm_wp` visible from any lander location.
                   - Calculate navigation cost from `?w` to `comm_wp` (`nav_comm_cost`).
                   - If `nav_comm_cost` is finite:
                     - Add 1 for the communicate action (`communicate_cost`).
                     - Total cost for this rover = `drop_cost + nav_sample_cost + sample_cost + nav_comm_cost + communicate_cost`.
                     - Update `goal_h` with the minimum cost found over all equipped rovers.
       - **For Image Communication Goal `(communicated_image_data ?o ?m)`:**
         - Identify the target objective `?o` and mode `?m`.
         - Check if the image data `(have_image ?r ?o ?m)` is held by *any* rover `?r` in the current state.
           - If Yes (Image Taken):
             - Find the rover `?r` holding the image.
             - Get its current location `current_wp`.
             - Find a waypoint `comm_wp` visible from any lander location.
             - Calculate navigation cost from `current_wp` to `comm_wp` for rover `?r`.
             - The cost for this path is `nav_cost + 1` (communicate action).
             - Update `goal_h` with the minimum cost found over all rovers holding the image and all possible communication waypoints.
           - If No (Image Not Taken):
             - Find rovers `?r` equipped for imaging with a camera `?i` on board that supports mode `?m`.
             - For each such rover `?r` with camera `?i`:
               - Get its current location `current_wp`.
               - Find calibration waypoints `cal_wps` for camera `?i`'s target.
               - Find imaging waypoints `img_wps` for objective `?o`.
               - Find communication waypoints `comm_wps` visible from any lander location.
               - If any of these waypoint sets are empty, this path is impossible for this rover/camera.
               - Calculate the minimum navigation cost for the sequence: `current_wp` -> `cal_wp` -> `img_wp` -> `comm_wp`. This involves nested minimums over possible intermediate waypoints.
               - If the minimum navigation cost is finite:
                 - Add 1 for calibrate, 1 for take_image, 1 for communicate actions.
                 - Total cost for this rover = `min_nav_cost + 1 + 1 + 1`.
                 - Update `goal_h` with the minimum cost found over all suitable rovers/cameras.
       - If `goal_h` is still the large number (infinity), it means the goal is unreachable based on the heuristic's logic and available resources/paths. Add the large number to `total_h`.
       - Otherwise, add `goal_h` to `total_h`.
    6. Return `total_h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and precomputing distances."""
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Static Facts ---
        self.rovers = set()
        self.waypoints = set()
        self.landers = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()
        self.stores = set()

        self.lander_wps = set()
        self.rover_capabilities = defaultdict(lambda: {'soil': False, 'rock': False, 'imaging': False})
        self.rover_stores = {} # {rover: store}
        self.rover_cameras = defaultdict(set) # {rover: {camera}}
        self.camera_modes = defaultdict(set) # {camera: {mode}}
        self.camera_targets = {} # {camera: objective}
        self.objective_visible_wps = defaultdict(set) # {objective: {waypoint}}
        self.waypoint_visibility = defaultdict(set) # {wp: {visible_wp}}
        self.rover_traversal = defaultdict(lambda: defaultdict(set)) # {rover: {wp_from: {wp_to}}}

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            # Collect objects by type (heuristic needs object names)
            if predicate == 'at_lander': self.landers.add(parts[1]); self.waypoints.add(parts[2]); self.lander_wps.add(parts[2])
            elif predicate == 'at': self.rovers.add(parts[1]); self.waypoints.add(parts[2]) # Initial 'at' facts are static in a sense for object identification
            elif predicate == 'can_traverse': self.rovers.add(parts[1]); self.waypoints.add(parts[2]); self.waypoints.add(parts[3]); self.rover_traversal[parts[1]][parts[2]].add(parts[3])
            elif predicate == 'equipped_for_soil_analysis': self.rovers.add(parts[1]); self.rover_capabilities[parts[1]]['soil'] = True
            elif predicate == 'equipped_for_rock_analysis': self.rovers.add(parts[1]); self.rover_capabilities[parts[1]]['rock'] = True
            elif predicate == 'equipped_for_imaging': self.rovers.add(parts[1]); self.rover_capabilities[parts[1]]['imaging'] = True
            elif predicate == 'store_of': self.stores.add(parts[1]); self.rovers.add(parts[2]); self.rover_stores[parts[2]] = parts[1]
            elif predicate == 'calibrated': self.cameras.add(parts[1]); self.rovers.add(parts[2]) # Initial 'calibrated' facts
            elif predicate == 'supports': self.cameras.add(parts[1]); self.modes.add(parts[2]); self.camera_modes[parts[1]].add(parts[2])
            elif predicate == 'visible': self.waypoints.add(parts[1]); self.waypoints.add(parts[2]); self.waypoint_visibility[parts[1]].add(parts[2])
            elif predicate == 'have_rock_analysis': self.rovers.add(parts[1]); self.waypoints.add(parts[2]) # Initial 'have' facts
            elif predicate == 'have_soil_analysis': self.rovers.add(parts[1]); self.waypoints.add(parts[2]) # Initial 'have' facts
            elif predicate == 'have_image': self.rovers.add(parts[1]); self.objectives.add(parts[2]); self.modes.add(parts[3]) # Initial 'have' facts
            elif predicate == 'communicated_soil_data': self.waypoints.add(parts[1]) # Initial 'communicated' facts
            elif predicate == 'communicated_rock_data': self.waypoints.add(parts[1]) # Initial 'communicated' facts
            elif predicate == 'communicated_image_data': self.objectives.add(parts[1]); self.modes.add(parts[2]) # Initial 'communicated' facts
            elif predicate == 'at_soil_sample': self.waypoints.add(parts[1]) # Initial sample locations
            elif predicate == 'at_rock_sample': self.waypoints.add(parts[1]) # Initial sample locations
            elif predicate == 'visible_from': self.objectives.add(parts[1]); self.waypoints.add(parts[2]); self.objective_visible_wps[parts[1]].add(parts[2])
            elif predicate == 'calibration_target': self.cameras.add(parts[1]); self.objectives.add(parts[2]); self.camera_targets[parts[1]] = parts[2]
            elif predicate == 'on_board': self.cameras.add(parts[1]); self.rovers.add(parts[2]); self.rover_cameras[parts[2]].add(parts[1])
            elif predicate == 'empty': self.stores.add(parts[1]) # Initial store status
            elif predicate == 'full': self.stores.add(parts[1]) # Initial store status


        # --- Precompute Rover Distances using BFS ---
        self.rover_distances = {} # {rover: {start_wp: {end_wp: distance}}}
        for rover in self.rovers:
            self.rover_distances[rover] = {}
            for start_wp in self.waypoints:
                self.rover_distances[rover][start_wp] = {wp: INF for wp in self.waypoints}
                self.rover_distances[rover][start_wp][start_wp] = 0
                queue = deque([start_wp])

                while queue:
                    u = queue.popleft()
                    # Check if rover can traverse from u
                    if u in self.rover_traversal[rover]:
                        for v in self.rover_traversal[rover][u]:
                            if self.rover_distances[rover][start_wp][v] == INF:
                                self.rover_distances[rover][start_wp][v] = self.rover_distances[rover][start_wp][u] + 1
                                queue.append(v)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse Current State ---
        rover_locations = {} # {rover: waypoint}
        store_full = set() # {store}
        have_soil = defaultdict(set) # {rover: {waypoint}}
        have_rock = defaultdict(set) # {rover: {waypoint}}
        have_image = defaultdict(lambda: defaultdict(set)) # {rover: {objective: {mode}}}
        calibrated_cameras = defaultdict(set) # {rover: {camera}}
        soil_samples = set() # {waypoint}
        rock_samples = set() # {waypoint}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at': rover_locations[parts[1]] = parts[2]
            elif predicate == 'full': store_full.add(parts[1])
            elif predicate == 'have_soil_analysis': have_soil[parts[1]].add(parts[2])
            elif predicate == 'have_rock_analysis': have_rock[parts[1]].add(parts[2])
            elif predicate == 'have_image': have_image[parts[1]][parts[2]].add(parts[3])
            elif predicate == 'calibrated': calibrated_cameras[parts[2]].add(parts[1])
            elif predicate == 'at_soil_sample': soil_samples.add(parts[1])
            elif predicate == 'at_rock_sample': rock_samples.add(parts[1])

        # Find communication waypoints (visible from any lander)
        comm_wps = set()
        for lander_wp in self.lander_wps:
             if lander_wp in self.waypoint_visibility:
                 comm_wps.update(self.waypoint_visibility[lander_wp])
             # A waypoint is visible from itself
             comm_wps.add(lander_wp)


        total_h = 0

        # --- Estimate Cost for Each Ungoaled Item ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            goal_h = INF # Minimum cost for this specific goal

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                wp = parts[1]
                min_rover_cost = INF

                # Check if data is already collected by any rover
                rover_has_data = {r for r in self.rovers if wp in have_soil[r]}

                if rover_has_data:
                    # Data collected, just need to navigate to lander and communicate
                    for r in rover_has_data:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue # Rover location unknown (shouldn't happen in valid states)

                        min_nav_comm = INF
                        for comm_wp in comm_wps:
                            dist = self.rover_distances[r][current_wp].get(comm_wp, INF)
                            min_nav_comm = min(min_nav_comm, dist)

                        if min_nav_comm != INF:
                            min_rover_cost = min(min_rover_cost, min_nav_comm + 1) # +1 for communicate

                else:
                    # Data not collected, need to sample, navigate, communicate
                    if wp in soil_samples: # Check if sample still exists
                        equipped_rovers = {r for r in self.rovers if self.rover_capabilities[r]['soil']}
                        for r in equipped_rovers:
                            current_wp = rover_locations.get(r)
                            if current_wp is None: continue

                            cost_drop = 1 if self.rover_stores.get(r) in store_full else 0
                            nav_sample_cost = self.rover_distances[r][current_wp].get(wp, INF)

                            if nav_sample_cost != INF:
                                cost_sample = 1
                                min_nav_comm = INF
                                for comm_wp in comm_wps:
                                    dist = self.rover_distances[r][wp].get(comm_wp, INF)
                                    min_nav_comm = min(min_nav_comm, dist)

                                if min_nav_comm != INF:
                                    cost_communicate = 1
                                    rover_total_cost = cost_drop + nav_sample_cost + cost_sample + min_nav_comm + cost_communicate
                                    min_rover_cost = min(min_rover_cost, rover_total_cost)

                goal_h = min_rover_cost

            elif predicate == 'communicated_rock_data':
                wp = parts[1]
                min_rover_cost = INF

                # Check if data is already collected by any rover
                rover_has_data = {r for r in self.rovers if wp in have_rock[r]}

                if rover_has_data:
                    # Data collected, just need to navigate to lander and communicate
                    for r in rover_has_data:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue

                        min_nav_comm = INF
                        for comm_wp in comm_wps:
                            dist = self.rover_distances[r][current_wp].get(comm_wp, INF)
                            min_nav_comm = min(min_nav_comm, dist)

                        if min_nav_comm != INF:
                            min_rover_cost = min(min_rover_cost, min_nav_comm + 1) # +1 for communicate

                else:
                    # Data not collected, need to sample, navigate, communicate
                    if wp in rock_samples: # Check if sample still exists
                        equipped_rovers = {r for r in self.rovers if self.rover_capabilities[r]['rock']}
                        for r in equipped_rovers:
                            current_wp = rover_locations.get(r)
                            if current_wp is None: continue

                            cost_drop = 1 if self.rover_stores.get(r) in store_full else 0
                            nav_sample_cost = self.rover_distances[r][current_wp].get(wp, INF)

                            if nav_sample_cost != INF:
                                cost_sample = 1
                                min_nav_comm = INF
                                for comm_wp in comm_wps:
                                    dist = self.rover_distances[r][wp].get(comm_wp, INF)
                                    min_nav_comm = min(min_nav_comm, dist)

                                if min_nav_comm != INF:
                                    cost_communicate = 1
                                    rover_total_cost = cost_drop + nav_sample_cost + cost_sample + min_nav_comm + cost_communicate
                                    min_rover_cost = min(min_rover_cost, rover_total_cost)

                goal_h = min_rover_cost

            elif predicate == 'communicated_image_data':
                objective = parts[1]
                mode = parts[2]
                min_rover_cost = INF

                # Check if image is already taken by any rover
                rover_has_image = {r for r in self.rovers if objective in have_image[r] and mode in have_image[r][objective]}

                if rover_has_image:
                    # Image taken, just need to navigate to lander and communicate
                    for r in rover_has_image:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue

                        min_nav_comm = INF
                        for comm_wp in comm_wps:
                            dist = self.rover_distances[r][current_wp].get(comm_wp, INF)
                            min_nav_comm = min(min_nav_comm, dist)

                        if min_nav_comm != INF:
                            min_rover_cost = min(min_rover_cost, min_nav_comm + 1) # +1 for communicate

                else:
                    # Image not taken, need to calibrate, take image, navigate, communicate
                    suitable_rovers = {r for r in self.rovers if self.rover_capabilities[r]['imaging']}
                    for r in suitable_rovers:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue

                        for camera in self.rover_cameras[r]:
                            if mode in self.camera_modes[camera]:
                                cal_target = self.camera_targets.get(camera)
                                if cal_target is None: continue # Camera has no calibration target

                                cal_wps = self.objective_visible_wps.get(cal_target, set())
                                img_wps = self.objective_visible_wps.get(objective, set())

                                if not cal_wps or not img_wps or not comm_wps:
                                    continue # Cannot achieve this goal with this rover/camera if required waypoints don't exist

                                # Check if camera is already calibrated
                                # Note: Calibration is consumed by take_image, so if image is not taken,
                                # we assume calibration is needed unless it's explicitly in the state.
                                # However, a simpler heuristic might just assume calibration is always needed
                                # if the image isn't taken, as calibration state is transient.
                                # Let's assume calibration is always needed if the image isn't taken.
                                # cost_calibrate = 1 if camera not in calibrated_cameras[r] else 0 # This is wrong, calibration is consumed

                                cost_calibrate = 1
                                cost_take_image = 1
                                cost_communicate = 1

                                # Find min navigation cost: current -> cal -> img -> comm
                                min_nav_cost = INF
                                for cal_wp in cal_wps:
                                    nav1 = self.rover_distances[r][current_wp].get(cal_wp, INF)
                                    if nav1 == INF: continue
                                    for img_wp in img_wps:
                                        nav2 = self.rover_distances[r][cal_wp].get(img_wp, INF)
                                        if nav2 == INF: continue
                                        for comm_wp in comm_wps:
                                            nav3 = self.rover_distances[r][img_wp].get(comm_wp, INF)
                                            if nav3 == INF: continue
                                            min_nav_cost = min(min_nav_cost, nav1 + nav2 + nav3)

                                if min_nav_cost != INF:
                                    rover_total_cost = min_nav_cost + cost_calibrate + cost_take_image + cost_communicate
                                    min_rover_cost = min(min_rover_cost, rover_total_cost)

                goal_h = min_rover_cost

            # Add the cost for this goal to the total heuristic cost
            # If goal_h is still INF, add a large penalty
            total_h += goal_h if goal_h != INF else 1000 # Penalty for seemingly unreachable goals

        return total_h

