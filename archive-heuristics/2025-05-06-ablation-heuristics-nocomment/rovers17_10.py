# Import necessary modules
from collections import deque
from fnmatch import fnmatch
import math # For float('inf')

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment if needed

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
         return []
    return fact[1:-1].split()

# Helper function to match PDDL fact strings with patterns
def match(fact, *args):
    """Checks if a fact string matches a pattern using fnmatch."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class inheriting from the assumed base class
# If running this code, ensure a base class named 'Heuristic' is available
# or uncomment the placeholder definition below if no external base class is provided.

# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static_facts = task.static
#
#     def __call__(self, node):
#         raise NotImplementedError("Subclasses must implement this method")


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated minimum costs for each unachieved goal fact. For each
        unachieved goal fact (communicating soil, rock, or image data), it
        calculates the minimum number of actions required to collect the
        necessary data (if not already collected) and navigate to a lander-visible
        waypoint to communicate it. Navigation costs are precomputed using BFS.
        The heuristic is non-admissible and aims to guide a greedy best-first search.

    Assumptions:
        - Lander location is static.
        - Rover capabilities, camera properties, store ownership, waypoint
          visibility, objective visibility, and calibration targets are static.
        - Soil and rock samples are static unless collected by a rover.
        - Each rover has one or more stores defined by `store_of` facts.
        - Dropping a sample is always possible if a store is full.
        - Calibration is required before taking an image, and is lost after taking it.
          For simplicity, the heuristic assumes calibration is needed for each image goal
          unless the image is already collected.
        - The navigation graph for each rover is defined by `can_traverse` facts.
        - Action costs are uniform (1).

    Heuristic Initialization:
        - Parses static facts to extract information about landers, rovers,
          stores, cameras, objectives, waypoints, capabilities, visibility,
          and traversability.
        - Builds a navigation graph for each rover based on `can_traverse` facts.
        - Precomputes all-pairs shortest paths (minimum navigation actions)
          for each rover using BFS. Stores these distances.
        - Identifies the lander location(s) and waypoints visible from the lander.
        - Precomputes valid combinations of (rover, camera, view_wp, cal_wp)
          for each (objective, mode) pair to facilitate image goal calculations.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Check if the current state is a goal state. If yes, return 0.
        2.  Initialize total heuristic cost `h = 0`.
        3.  Parse dynamic facts from the current state: rover locations,
            store statuses, collected data (soil, rock, image), calibrated cameras,
            and remaining soil/rock samples.
        4.  Iterate through each goal fact in `self.goals`.
        5.  If the goal fact is already present in the current state, its cost is 0; continue to the next goal.
        6.  For an unachieved goal fact (e.g., `(communicated_soil_data w)`):
            a.  Initialize the minimum cost for this goal `goal_cost = math.inf`.
            b.  **Option 1: Data is already collected.**
                -   Check if the required data fact (e.g., `(have_soil_analysis r w)`) exists in the current state for any rover `r`.
                -   If yes, find the minimum cost to communicate this existing data:
                    -   For each rover `r` that has the data:
                        -   Get the rover's current location `current_wp`.
                        -   Find waypoints `comm_wp` visible from the lander location.
                        -   For each `comm_wp`: Calculate navigation cost `dist(current_wp, comm_wp)` using precomputed paths.
                        -   If path exists, calculate communication cost: `nav_cost + 1` (navigate + communicate).
                        -   Update `goal_cost` with the minimum found so far.
            c.  **Option 2: Data needs to be collected.**
                -   Determine the type of data needed (soil, rock, image).
                -   **For Soil/Rock data `(communicated_soil_data w)` or `(communicated_rock_data w)`:**
                    -   Check if the corresponding sample (`(at_soil_sample w)` or `(at_rock_sample w)`) exists in the current state. If not, this option is impossible.
                    -   If sample exists, find equipped rovers `r`.
                    -   For each suitable rover `r`:
                        -   Get the rover's current location `current_wp`.
                        -   Find stores `s` for `r`. Check if *any* store is empty. If not, a drop action is needed (cost +1).
                        -   Calculate cost to sample: `dist(current_wp, w) + (1 if all stores full else 0) + 1` (navigate + drop(if needed) + sample).
                        -   Calculate cost to communicate from `w`: `min_{comm_wp} dist(w, comm_wp) + 1` over `comm_wp` visible from lander.
                        -   Total cost for this rover = `cost_sample + cost_comm`.
                        -   Update `goal_cost` with the minimum found so far.
                -   **For Image data `(communicated_image_data o m)`:**
                    -   Find equipped rovers `r` with cameras `i` supporting mode `m`.
                    -   Find waypoints `view_wps` visible from objective `o`.
                    -   For each suitable rover `r` and camera `i`:
                        -   Find calibration target `t` for `i` and waypoints `cal_wps` visible from `t`.
                        -   For each `p` in `view_wps` and `w` in `cal_wps`:
                            -   Calculate cost to calibrate: `dist(rover_loc[r], w) + 1` (navigate + calibrate).
                            -   Calculate cost to take image: `dist(w, p) + 1` (navigate + take_image).
                            -   Calculate cost to communicate from `p`: `min_{comm_wp} dist(p, comm_wp) + 1` over `comm_wp` visible from lander.
                            -   Total cost = `cost_calibrate + cost_take_image + cost_comm`.
                            -   Update `goal_cost` with the minimum found so far.
            d.  Add `goal_cost` to `h`. If `goal_cost` was `math.inf`, `h` will become `math.inf`.
        7.  Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        super().__init__(task)

        # --- Parse Static Facts ---
        self.lander_location = {} # lander -> waypoint
        self.rover_capabilities = { # capability -> set of rovers
            'soil': set(),
            'rock': set(),
            'imaging': set()
        }
        self.store_to_rover = {} # store -> rover
        self.waypoint_visibility = {} # wp -> set of visible wps
        self.objective_visibility = {} # obj -> set of visible wps
        self.camera_info = {} # camera -> {'rover': rover, 'modes': set of modes, 'calibration_target': obj}
        self.rover_navigation_graphs = {} # rover -> {wp -> set of reachable wps in 1 step}
        self.waypoints = set() # set of all waypoints
        self.rovers = set() # set of all rovers
        self.objectives = set() # set of all objectives
        self.modes = set() # set of all modes
        self.cameras = set() # set of all cameras
        self.stores = set() # set of all stores
        self.landers = set() # set of all landers

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]

            if pred == 'at_lander':
                if len(parts) == 3:
                    lander, wp = parts[1], parts[2]
                    self.lander_location[lander] = wp
                    self.landers.add(lander)
                    self.waypoints.add(wp)
            elif pred == 'equipped_for_soil_analysis':
                 if len(parts) == 2:
                    self.rover_capabilities['soil'].add(parts[1])
                    self.rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis':
                 if len(parts) == 2:
                    self.rover_capabilities['rock'].add(parts[1])
                    self.rovers.add(parts[1])
            elif pred == 'equipped_for_imaging':
                 if len(parts) == 2:
                    self.rover_capabilities['imaging'].add(parts[1])
                    self.rovers.add(parts[1])
            elif pred == 'store_of':
                 if len(parts) == 3:
                    store, rover = parts[1], parts[2]
                    self.store_to_rover[store] = rover
                    self.stores.add(store)
                    self.rovers.add(rover)
            elif pred == 'visible':
                 if len(parts) == 3:
                    wp1, wp2 = parts[1], parts[2]
                    self.waypoint_visibility.setdefault(wp1, set()).add(wp2)
                    self.waypoint_visibility.setdefault(wp2, set()).add(wp1) # Assuming visible is symmetric
                    self.waypoints.add(wp1)
                    self.waypoints.add(wp2)
            elif pred == 'visible_from':
                 if len(parts) == 3:
                    obj, wp = parts[1], parts[2]
                    self.objective_visibility.setdefault(obj, set()).add(wp)
                    self.objectives.add(obj)
                    self.waypoints.add(wp)
            elif pred == 'on_board':
                 if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).update({'rover': rover})
                    self.cameras.add(camera)
                    self.rovers.add(rover)
            elif pred == 'supports':
                 if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
                    self.cameras.add(camera)
                    self.modes.add(mode)
            elif pred == 'calibration_target':
                 if len(parts) == 3:
                    camera, obj = parts[1], parts[2]
                    self.camera_info.setdefault(camera, {}).update({'calibration_target': obj})
                    self.cameras.add(camera)
                    self.objectives.add(obj)
            elif pred == 'can_traverse':
                 if len(parts) == 4:
                    rover, wp1, wp2 = parts[1], parts[2], parts[3]
                    self.rover_navigation_graphs.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                    self.rovers.add(rover)
                    self.waypoints.add(wp1)
                    self.waypoints.add(wp2)
            # Add other types if they appear in static facts but not covered above
            elif pred in ['rover', 'waypoint', 'store', 'camera', 'mode', 'lander', 'objective']:
                 if len(parts) == 2:
                     if pred == 'rover': self.rovers.add(parts[1])
                     elif pred == 'waypoint': self.waypoints.add(parts[1])
                     elif pred == 'store': self.stores.add(parts[1])
                     elif pred == 'camera': self.cameras.add(parts[1])
                     elif pred == 'mode': self.modes.add(parts[1])
                     elif pred == 'lander': self.landers.add(parts[1])
                     elif pred == 'objective': self.objectives.add(parts[1])


        # --- Precompute Shortest Paths for each rover ---
        self.rover_shortest_paths = {} # rover -> {start_wp -> {end_wp -> distance}}
        for rover in self.rovers:
            self.rover_shortest_paths[rover] = self._bfs_all_pairs(rover)

        # --- Find Lander Communication Waypoints ---
        self.lander_comm_wps = set()
        # Assuming there's at least one lander and we can pick any lander's location
        # Find all lander locations and collect all visible waypoints from them
        all_lander_wps = set(self.lander_location.values())
        for lander_wp in all_lander_wps:
             if lander_wp in self.waypoint_visibility:
                 self.lander_comm_wps.update(self.waypoint_visibility[lander_wp])


        # --- Precompute Rover-Camera-Mode-Objective-Waypoint combinations ---
        # This helps quickly find options for taking images
        self.image_options = {} # (obj, mode) -> set of (rover, camera, view_wp, cal_wp)
        for obj in self.objectives:
            for mode in self.modes:
                possible_options = set()
                # Find waypoints visible from the objective
                view_wps = self.objective_visibility.get(obj, set())
                if not view_wps: continue # Cannot view this objective

                # Find cameras that support this mode
                supporting_cameras = {c for c, info in self.camera_info.items() if mode in info.get('modes', set())}
                if not supporting_cameras: continue # No camera supports this mode

                for camera in supporting_cameras:
                    cam_info = self.camera_info.get(camera, {})
                    rover = cam_info.get('rover')
                    cal_target = cam_info.get('calibration_target')

                    if not rover or not cal_target: continue # Camera not on board or no cal target

                    # Check if the rover is equipped for imaging
                    if rover not in self.rover_capabilities['imaging']: continue

                    # Find waypoints visible from the calibration target
                    cal_wps = self.objective_visibility.get(cal_target, set())
                    if not cal_wps: continue # Cannot view calibration target

                    # Add all combinations of view_wp and cal_wp
                    for view_wp in view_wps:
                        for cal_wp in cal_wps:
                            possible_options.add((rover, camera, view_wp, cal_wp))

                if possible_options:
                    self.image_options[(obj, mode)] = possible_options


    def _bfs_all_pairs(self, rover):
        """
        Computes shortest path distances from every waypoint to every other
        waypoint for a specific rover using BFS.
        Returns a dictionary {start_wp -> {end_wp -> distance}}.
        """
        graph = self.rover_navigation_graphs.get(rover, {})
        distances = {}
        all_wps = list(self.waypoints) # Use a list for consistent iteration order

        for start_node in all_wps:
            distances[start_node] = {wp: math.inf for wp in all_wps}
            distances[start_node][start_node] = 0
            queue = deque([(start_node, 0)]) # Use deque for efficient queue operations
            visited = {start_node}

            while queue:
                (current_wp, dist) = queue.popleft()

                # Check if current_wp exists as a key in the graph
                if current_wp not in graph:
                    continue

                for neighbor in graph[current_wp]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        return distances

    def get_distance(self, rover, start_wp, end_wp):
        """Helper to get precomputed distance, handling missing entries."""
        if rover not in self.rover_shortest_paths:
            return math.inf
        if start_wp not in self.rover_shortest_paths[rover]:
             return math.inf
        return self.rover_shortest_paths[rover][start_wp].get(end_wp, math.inf)


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Parse Dynamic Facts from State ---
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        collected_data = set() # set of (have_soil_analysis r w), (have_rock_analysis r w), (have_image r o m)
        calibrated_cameras = set() # set of (calibrated c r)
        remaining_samples = set() # set of (at_soil_sample w), (at_rock_sample w)

        # Pre-parse state facts for faster lookup
        state_facts_map = {}
        for fact in state:
             parts = get_parts(fact)
             if parts:
                 pred = parts[0]
                 # Store facts by predicate for quick access
                 state_facts_map.setdefault(pred, set()).add(fact)

        # Populate dynamic data structures
        for fact in state_facts_map.get('at', set()):
             parts = get_parts(fact)
             if len(parts) == 3 and parts[1] in self.rovers:
                  rover_locations[parts[1]] = parts[2]

        for fact in state_facts_map.get('empty', set()):
             parts = get_parts(fact)
             if len(parts) == 2 and parts[1] in self.stores:
                  store_status[parts[1]] = 'empty'

        for fact in state_facts_map.get('full', set()):
             parts = get_parts(fact)
             if len(parts) == 2 and parts[1] in self.stores:
                  store_status[parts[1]] = 'full'

        collected_data.update(state_facts_map.get('have_soil_analysis', set()))
        collected_data.update(state_facts_map.get('have_rock_analysis', set()))
        collected_data.update(state_facts_map.get('have_image', set()))
        calibrated_cameras.update(state_facts_map.get('calibrated', set()))
        remaining_samples.update(state_facts_map.get('at_soil_sample', set()))
        remaining_samples.update(state_facts_map.get('at_rock_sample', set()))


        total_heuristic_cost = 0

        # --- Calculate Cost for Each Unachieved Goal ---
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            goal_parts = get_parts(goal_fact)
            if not goal_parts: continue # Skip malformed goal facts
            goal_pred = goal_parts[0]

            goal_cost = math.inf

            if goal_pred == 'communicated_soil_data':
                # Goal: (communicated_soil_data w)
                if len(goal_parts) != 2: continue
                w = goal_parts[1]
                sample_exists = f'(at_soil_sample {w})' in remaining_samples

                # Option 1: Data already collected (have_soil_analysis r w)
                rovers_with_data = {r for r in self.rovers if f'(have_soil_analysis {r} {w})' in collected_data}
                if rovers_with_data:
                    min_comm_cost_option1 = math.inf
                    for r in rovers_with_data:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue # Rover location unknown? Should not happen in valid state
                        for comm_wp in self.lander_comm_wps:
                            nav_cost = self.get_distance(r, current_wp, comm_wp)
                            if nav_cost != math.inf:
                                min_comm_cost_option1 = min(min_comm_cost_option1, nav_cost + 1) # navigate + communicate
                    goal_cost = min(goal_cost, min_comm_cost_option1)

                # Option 2: Data needs collecting (sample_soil at w)
                if sample_exists:
                    equipped_rovers = self.rover_capabilities['soil']
                    if equipped_rovers:
                        min_collect_comm_cost = math.inf
                        for r in equipped_rovers:
                            current_wp = rover_locations.get(r)
                            if current_wp is None: continue

                            rover_stores = [s for s, owner in self.store_to_rover.items() if owner == r]
                            if not rover_stores: continue # Rover has no store? Should not happen

                            # Check if any store is empty
                            any_store_empty = any(store_status.get(s) == 'empty' for s in rover_stores)
                            drop_cost = 0 if any_store_empty else 1 # Need 1 drop action if all are full

                            nav_to_sample_cost = self.get_distance(r, current_wp, w)
                            if nav_to_sample_cost == math.inf: continue

                            cost_sample = nav_to_sample_cost + drop_cost + 1 # navigate + drop (if needed) + sample

                            min_nav_from_sample_to_comm = math.inf
                            for comm_wp in self.lander_comm_wps:
                                nav_cost = self.get_distance(r, w, comm_wp)
                                if nav_cost != math.inf:
                                    min_nav_from_sample_to_comm = min(min_nav_from_sample_to_comm, nav_cost)

                            if min_nav_from_sample_to_comm != math.inf:
                                cost_comm = min_nav_from_sample_to_comm + 1 # navigate + communicate
                                min_collect_comm_cost = min(min_collect_comm_cost, cost_sample + cost_comm)

                        goal_cost = min(goal_cost, min_collect_comm_cost)


            elif goal_pred == 'communicated_rock_data':
                # Goal: (communicated_rock_data w)
                if len(goal_parts) != 2: continue
                w = goal_parts[1]
                sample_exists = f'(at_rock_sample {w})' in remaining_samples

                # Option 1: Data already collected (have_rock_analysis r w)
                rovers_with_data = {r for r in self.rovers if f'(have_rock_analysis {r} {w})' in collected_data}
                if rovers_with_data:
                    min_comm_cost_option1 = math.inf
                    for r in rovers_with_data:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue
                        for comm_wp in self.lander_comm_wps:
                            nav_cost = self.get_distance(r, current_wp, comm_wp)
                            if nav_cost != math.inf:
                                min_comm_cost_option1 = min(min_comm_cost_option1, nav_cost + 1) # navigate + communicate
                    goal_cost = min(goal_cost, min_comm_cost_option1)

                # Option 2: Data needs collecting (sample_rock at w)
                if sample_exists:
                    equipped_rovers = self.rover_capabilities['rock']
                    if equipped_rovers:
                        min_collect_comm_cost = math.inf
                        for r in equipped_rovers:
                            current_wp = rover_locations.get(r)
                            if current_wp is None: continue

                            rover_stores = [s for s, owner in self.store_to_rover.items() if owner == r]
                            if not rover_stores: continue

                            any_store_empty = any(store_status.get(s) == 'empty' for s in rover_stores)
                            drop_cost = 0 if any_store_empty else 1 # Need 1 drop action if all are full

                            nav_to_sample_cost = self.get_distance(r, current_wp, w)
                            if nav_to_sample_cost == math.inf: continue

                            cost_sample = nav_to_sample_cost + drop_cost + 1 # navigate + drop (if needed) + sample

                            min_nav_from_sample_to_comm = math.inf
                            for comm_wp in self.lander_comm_wps:
                                nav_cost = self.get_distance(r, w, comm_wp)
                                if nav_cost != math.inf:
                                    min_nav_from_sample_to_comm = min(min_nav_from_sample_to_comm, nav_cost)

                            if min_nav_from_sample_to_comm != math.inf:
                                cost_comm = min_nav_from_sample_to_comm + 1 # navigate + communicate
                                min_collect_comm_cost = min(min_collect_comm_cost, cost_sample + cost_comm)

                        goal_cost = min(goal_cost, min_collect_comm_cost)


            elif goal_pred == 'communicated_image_data':
                # Goal: (communicated_image_data o m)
                if len(goal_parts) != 3: continue
                o, m = goal_parts[1], goal_parts[2]

                # Option 1: Image already taken (have_image r o m)
                rovers_with_image = {r for r in self.rovers if f'(have_image {r} {o} {m})' in collected_data}
                if rovers_with_image:
                    min_comm_cost_option1 = math.inf
                    for r in rovers_with_image:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue
                        for comm_wp in self.lander_comm_wps:
                            nav_cost = self.get_distance(r, current_wp, comm_wp)
                            if nav_cost != math.inf:
                                min_comm_cost_option1 = min(min_comm_cost_option1, nav_cost + 1) # navigate + communicate
                    goal_cost = min(goal_cost, min_comm_cost_option1)

                # Option 2: Image needs taking (take_image of o in m)
                image_options = self.image_options.get((o, m), set())
                if image_options:
                    min_take_comm_cost = math.inf
                    for (r, camera, view_wp, cal_wp) in image_options:
                        current_wp = rover_locations.get(r)
                        if current_wp is None: continue

                        # Cost to calibrate
                        nav_to_cal_cost = self.get_distance(r, current_wp, cal_wp)
                        if nav_to_cal_cost == math.inf: continue
                        cost_calibrate = nav_to_cal_cost + 1 # navigate + calibrate

                        # Cost to take image
                        nav_cal_to_view_cost = self.get_distance(r, cal_wp, view_wp)
                        if nav_cal_to_view_cost == math.inf: continue
                        cost_take_image = nav_cal_to_view_cost + 1 # navigate + take_image

                        # Cost to communicate from view_wp
                        min_nav_from_view_to_comm = math.inf
                        for comm_wp in self.lander_comm_wps:
                            nav_cost = self.get_distance(r, view_wp, comm_wp)
                            if nav_cost != math.inf:
                                min_nav_from_view_to_comm = min(min_nav_from_view_to_comm, nav_cost)

                        if min_nav_from_view_to_comm != math.inf:
                            cost_comm = min_nav_from_view_to_comm + 1 # navigate + communicate
                            min_take_comm_cost = min(min_take_comm_cost, cost_calibrate + cost_take_image + cost_comm)

                    goal_cost = min(goal_cost, min_take_comm_cost)

            # Add cost for this goal to total heuristic
            total_heuristic_cost += goal_cost

        return total_heuristic_cost
