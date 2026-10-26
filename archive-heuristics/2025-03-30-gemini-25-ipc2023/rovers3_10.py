import math
from collections import deque

# Helper function to parse PDDL facts/goals represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses and splitting."""
    # Handles facts like "(pred obj1 obj2)" -> ["pred", "obj1", "obj2"]
    # Returns empty list if input is not a valid string or format is incorrect
    if isinstance(fact, str) and fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return []

class RoversHeuristic:
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from the current state in the Rovers domain. It works by decomposing the overall
    goal into subgoals (communicating specific soil data, rock data, or image data)
    and estimating the minimum cost to achieve each unmet subgoal independently.
    The final heuristic value is the sum of these minimum costs. The heuristic is
    designed to be informative for a greedy best-first search but is not necessarily
    admissible.

    # Assumptions
    - Each action has a uniform cost of 1.
    - Rovers operate independently when estimating costs for different subgoals (resource conflicts are ignored).
    - The shortest path (in terms of navigation actions) is always preferred for movement.
    - Dropping samples takes 1 action and can be done anywhere (as per the 'drop' action definition).
    - Calibration, sampling, and taking images each take 1 action.
    - Communication takes 1 action.
    - If multiple rovers/cameras can achieve a subgoal, the one with the minimum estimated action sequence is chosen for the cost calculation of that subgoal.

    # Heuristic Initialization (`__init__`)
    - Parses the task's static facts to build internal data structures:
        - `objects`: Sets of objects categorized by type (rover, waypoint, camera, etc.).
        - `lander_location`: The waypoint where the lander is situated.
        - `waypoints_visible_from_lander`: Set of waypoints from which the lander is visible (potential communication spots).
        - `nav_graphs`: Precomputed navigation graphs for each rover, mapping each waypoint to a list of reachable waypoints considering `can_traverse` and `visible` predicates.
        - `rover_equipment`: Dictionary mapping each rover to its capabilities (soil, rock, imaging).
        - `rover_stores`: Dictionary mapping each rover to its store object.
        - `store_owners`: Dictionary mapping each store to its rover.
        - `cameras`: Nested dictionary storing camera details for each rover (supported modes, calibration target).
        - `objective_visibility`: Dictionary mapping each objective to the set of waypoints it's visible from.
        - `calibration_visibility`: Dictionary mapping each calibration target (objective) to the set of waypoints it's visible from.
        - `all_visibilities`: Dictionary mapping each waypoint to the set of waypoints visible from it.
    - Stores the goal facts (`self.goals`).

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Parse the current `state` to determine:
       - Rover locations (`at`).
       - Store status (`empty`/`full`).
       - Camera calibration status (`calibrated`).
       - Data currently held by rovers (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
    2. Initialize `h_value = 0` and `num_unmet_goals = 0`.
    3. Iterate through each `goal_fact` in `self.goals`.
    4. If `goal_fact` is already in the current `state`, continue to the next goal (cost is 0).
    5. If `goal_fact` is unmet, increment `num_unmet_goals` and calculate the minimum estimated cost (`min_cost_for_goal`) to achieve it:
       a. **For `communicated_soil_data(wp)`:**
          - Calculate cost considering two options:
             i. A rover `r` already has `have_soil_analysis(r, wp)`: Cost = `min(path(r, current_wp, comm_wp) + 1)` over such rovers `r`.
             ii. No rover has the analysis: Cost = `min(cost_drop + path(r, current_wp, wp) + 1 + path(r, wp, comm_wp) + 1)` over capable rovers `r`.
          - `min_cost_for_goal` = min(cost_option_i, cost_option_ii).
       b. **For `communicated_rock_data(wp)`:** Analogous to soil data.
       c. **For `communicated_image_data(obj, mode)`:**
          - Calculate cost considering two options:
             i. A rover `r` already has `have_image(r, obj, mode)`: Cost = `min(path(r, current_wp, comm_wp) + 1)` over such rovers `r`.
             ii. No rover has the image: Cost = `min(cost_cal + cost_nav_img + 1 + cost_nav_comm + 1)` over capable (rover `r`, camera `i`) pairs.
                 - `cost_cal`: Cost to navigate to calibration spot and calibrate (if needed). Includes `path(r, current_wp, cal_wp) + 1`. Updates rover position to `wp_after_cal`.
                 - `cost_nav_img`: Cost to navigate from `wp_after_cal` to image spot. Includes `path(r, wp_after_cal, img_wp)`. Updates rover position to `wp_after_img`.
                 - `cost_nav_comm`: Cost to navigate from `wp_after_img` to communication spot. Includes `path(r, wp_after_img, comm_wp)`.
          - `min_cost_for_goal` = min(cost_option_i, cost_option_ii).
    6. If `min_cost_for_goal` is infinity for any goal, return infinity (state is considered a dead end).
    7. Add `min_cost_for_goal` to `h_value`.
    8. After checking all goals, if `h_value` is 0 and `num_unmet_goals > 0`, return `num_unmet_goals` (ensures heuristic is non-zero for non-goal states). Otherwise, return `h_value`.
    """

    def __init__(self, task):
        """Initializes the heuristic by parsing static information from the task."""
        self.goals = task.goals
        self.static = task.static
        # --- Data Structures ---
        self.objects = {'rover': set(), 'waypoint': set(), 'store': set(), 'camera': set(),
                        'mode': set(), 'lander': set(), 'objective': set()}
        self.lander_location = None
        self.waypoints_visible_from_lander = frozenset()
        self.nav_graphs = {} # rover -> {wp: [next_wp, ...]}
        self.rover_equipment = {} # rover -> { 'soil': bool, 'rock': bool, 'imaging': bool }
        self.rover_stores = {} # rover -> store
        self.store_owners = {} # store -> rover
        self.cameras = {} # rover -> {camera: {'modes': set(), 'cal_target': obj}}
        self.objective_visibility = {} # objective -> frozenset(wp)
        self.calibration_visibility = {} # objective -> frozenset(wp) (using objectives as cal targets)
        self.all_visibilities = {} # wp -> frozenset(wp)
        self.waypoints = set() # Will be populated by _extract_objects

        # --- Initialization Steps ---
        self._extract_objects(task)
        self._parse_static_facts()
        self._build_nav_graphs()

    def _extract_objects(self, task):
        """Populates self.objects based on static facts and goals, and initializes rover dicts."""
        # Infer objects and types from static facts
        for fact in self.static:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             args = parts[1:]
             try:
                 if pred == 'at_lander': self.objects['lander'].add(args[0]); self.objects['waypoint'].add(args[1])
                 elif pred == 'equipped_for_soil_analysis': self.objects['rover'].add(args[0])
                 elif pred == 'equipped_for_rock_analysis': self.objects['rover'].add(args[0])
                 elif pred == 'equipped_for_imaging': self.objects['rover'].add(args[0])
                 elif pred == 'store_of': self.objects['store'].add(args[0]); self.objects['rover'].add(args[1])
                 elif pred == 'visible': self.objects['waypoint'].add(args[0]); self.objects['waypoint'].add(args[1])
                 elif pred == 'can_traverse': self.objects['rover'].add(args[0]); self.objects['waypoint'].add(args[1]); self.objects['waypoint'].add(args[2])
                 elif pred == 'calibration_target': self.objects['camera'].add(args[0]); self.objects['objective'].add(args[1])
                 elif pred == 'on_board': self.objects['camera'].add(args[0]); self.objects['rover'].add(args[1])
                 elif pred == 'supports': self.objects['camera'].add(args[0]); self.objects['mode'].add(args[1])
                 elif pred == 'visible_from': self.objects['objective'].add(args[0]); self.objects['waypoint'].add(args[1])
                 # Include predicates defining locations of samples if needed for waypoint identification
                 elif pred == 'at_soil_sample': self.objects['waypoint'].add(args[0])
                 elif pred == 'at_rock_sample': self.objects['waypoint'].add(args[0])
             except IndexError:
                 # Silently ignore facts with unexpected format during object extraction
                 pass

        # Add objects mentioned in goals
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            try:
                if pred == 'communicated_soil_data': self.objects['waypoint'].add(args[0])
                elif pred == 'communicated_rock_data': self.objects['waypoint'].add(args[0])
                elif pred == 'communicated_image_data': self.objects['objective'].add(args[0]); self.objects['mode'].add(args[1])
            except IndexError:
                 pass

        # Initialize rover equipment/cameras based on detected rovers
        for r in self.objects['rover']:
            if r not in self.rover_equipment:
                self.rover_equipment[r] = {'soil': False, 'rock': False, 'imaging': False}
            if r not in self.cameras:
                self.cameras[r] = {}
        # Initialize store owners
        for s in self.objects['store']:
            self.store_owners[s] = None # Will be set in _parse_static_facts

        # Consolidate all found waypoints
        self.waypoints = self.objects['waypoint']


    def _parse_static_facts(self):
        """Parses static facts to populate internal data structures."""
        temp_vis = {}
        temp_obj_vis = {}
        temp_cal_targets = {} # camera -> objective
        temp_on_board = {} # camera -> rover
        temp_supports = {} # camera -> set(modes)

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            try:
                if pred == 'visible':
                    wp1, wp2 = args
                    if wp1 not in temp_vis: temp_vis[wp1] = set()
                    temp_vis[wp1].add(wp2)
                elif pred == 'at_lander':
                    self.lander_location = args[1]
                elif pred == 'equipped_for_soil_analysis':
                    # Ensure rover exists before assigning equipment
                    if args[0] in self.rover_equipment: self.rover_equipment[args[0]]['soil'] = True
                elif pred == 'equipped_for_rock_analysis':
                     if args[0] in self.rover_equipment: self.rover_equipment[args[0]]['rock'] = True
                elif pred == 'equipped_for_imaging':
                     if args[0] in self.rover_equipment: self.rover_equipment[args[0]]['imaging'] = True
                elif pred == 'store_of':
                    store, rover = args
                    # Ensure objects exist before linking
                    if rover in self.objects['rover']: self.rover_stores[rover] = store
                    if store in self.objects['store']: self.store_owners[store] = rover
                elif pred == 'on_board':
                    # Ensure camera and rover exist
                    if args[0] in self.objects['camera'] and args[1] in self.objects['rover']:
                        temp_on_board[args[0]] = args[1]
                elif pred == 'supports':
                    camera, mode = args
                    # Ensure camera and mode exist
                    if args[0] in self.objects['camera'] and args[1] in self.objects['mode']:
                        if camera not in temp_supports: temp_supports[camera] = set()
                        temp_supports[camera].add(mode)
                elif pred == 'calibration_target':
                    # Ensure camera and objective exist
                    if args[0] in self.objects['camera'] and args[1] in self.objects['objective']:
                        temp_cal_targets[args[0]] = args[1]
                elif pred == 'visible_from':
                    objective, waypoint = args
                    # Ensure objective and waypoint exist
                    if args[0] in self.objects['objective'] and args[1] in self.objects['waypoint']:
                        if objective not in temp_obj_vis: temp_obj_vis[objective] = set()
                        temp_obj_vis[objective].add(waypoint)
            except IndexError:
                 pass

        # Finalize visibility dict
        self.all_visibilities = {wp: frozenset(targets) for wp, targets in temp_vis.items()}
        for wp in self.waypoints: # Ensure all waypoints are keys
            if wp not in self.all_visibilities:
                self.all_visibilities[wp] = frozenset()

        # Finalize objective visibility
        self.objective_visibility = {obj: frozenset(wps) for obj, wps in temp_obj_vis.items()}
        for obj in self.objects['objective']: # Ensure all objectives are keys
             if obj not in self.objective_visibility:
                  self.objective_visibility[obj] = frozenset()

        # Populate self.cameras using intermediate dicts
        for camera, rover in temp_on_board.items():
            # Check if rover exists in our initialized dict
            if rover in self.cameras:
                 modes = temp_supports.get(camera, set())
                 cal_target = temp_cal_targets.get(camera, None)
                 self.cameras[rover][camera] = {'modes': modes, 'cal_target': cal_target}

        # Populate calibration visibility using objective visibility
        self.calibration_visibility = {}
        for rover in self.objects['rover']:
             # Check rover exists and has cameras assigned
             if rover in self.cameras:
                 for cam_data in self.cameras[rover].values():
                     cal_target = cam_data.get('cal_target')
                     if cal_target:
                         # Calibration target uses objective visibility info
                         vis_wps = self.objective_visibility.get(cal_target, frozenset())
                         # Store visibility per target objective
                         self.calibration_visibility[cal_target] = vis_wps

        # Find waypoints visible from lander
        if self.lander_location:
            comm_wps = set()
            # Check visibility TO lander location (visible ?x ?lander_loc)
            for wp1, visible_wps in self.all_visibilities.items():
                if self.lander_location in visible_wps:
                    comm_wps.add(wp1)
            self.waypoints_visible_from_lander = frozenset(comm_wps)


    def _build_nav_graphs(self):
        """Builds rover-specific navigation graphs based on can_traverse and visible."""
        can_traverse_data = {} # rover -> {wp_from: set(wp_to)}
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue
            try:
                if parts[0] == 'can_traverse':
                    rover, wp_from, wp_to = parts[1:]
                    # Ensure objects exist before adding edge
                    if rover in self.objects['rover'] and wp_from in self.waypoints and wp_to in self.waypoints:
                        if rover not in can_traverse_data:
                            can_traverse_data[rover] = {}
                        if wp_from not in can_traverse_data[rover]:
                            can_traverse_data[rover][wp_from] = set()
                        can_traverse_data[rover][wp_from].add(wp_to)
            except IndexError:
                 pass

        for rover in self.objects['rover']:
            self.nav_graphs[rover] = {}
            for wp_from in self.waypoints:
                 # Get potential destinations based on traversal capability
                 possible_next_wps = can_traverse_data.get(rover, {}).get(wp_from, set())
                 # Get waypoints visible from the current one
                 visible_from_wp = self.all_visibilities.get(wp_from, frozenset())
                 # Valid moves require both traversal and visibility
                 valid_next_wps = possible_next_wps.intersection(visible_from_wp)
                 # Store reachable waypoints as a list
                 self.nav_graphs[rover][wp_from] = list(valid_next_wps)


    def _bfs_shortest_path_and_dest(self, rover, start_wp, target_wp_set):
        """
        Finds shortest path distance and ONE specific target waypoint reached first using BFS.
        Returns (distance, destination_wp) or (inf, None) if unreachable.
        """
        # Ensure target_wp_set is a set for efficient 'in' check
        if not isinstance(target_wp_set, set):
            target_wp_set = set(target_wp_set)

        if start_wp in target_wp_set:
            return 0, start_wp
        # Basic validity checks
        if not target_wp_set or rover not in self.nav_graphs or not self.nav_graphs[rover] or start_wp not in self.waypoints:
             return float('inf'), None

        q = deque([(start_wp, 0)])
        visited = {start_wp}
        graph = self.nav_graphs[rover]

        while q:
            current_wp, dist = q.popleft()

            # Access neighbors safely using .get() which returns empty list if key not found
            neighbors = graph.get(current_wp, [])

            for neighbor_wp in neighbors:
                if neighbor_wp not in visited:
                    if neighbor_wp in target_wp_set:
                        # Found the first shortest path to a target
                        return dist + 1, neighbor_wp
                    visited.add(neighbor_wp)
                    q.append((neighbor_wp, dist + 1))

        return float('inf'), None # Target not reachable


    def __call__(self, node):
        """Computes the heuristic value for the given state node."""
        state = node.state
        h_value = 0
        num_unmet_goals = 0

        # --- Parse current state ---
        rover_locations = {}
        store_status = {} # store -> 'empty' or 'full'
        calibrated_cameras = set() # set of (camera, rover) tuples
        have_soil = set() # set of (rover, wp) tuples
        have_rock = set() # set of (rover, wp) tuples
        have_image = set() # set of (rover, obj, mode) tuples

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            try:
                # Check if the object is of the expected type before accessing location/status
                if pred == 'at' and args[0] in self.objects['rover']:
                    rover_locations[args[0]] = args[1]
                elif pred == 'empty' and args[0] in self.objects['store']:
                    store_status[args[0]] = 'empty'
                elif pred == 'full' and args[0] in self.objects['store']:
                     store_status[args[0]] = 'full'
                elif pred == 'calibrated' and args[0] in self.objects['camera'] and args[1] in self.objects['rover']:
                    calibrated_cameras.add(tuple(args)) # (camera, rover)
                elif pred == 'have_soil_analysis' and args[0] in self.objects['rover'] and args[1] in self.objects['waypoint']:
                    have_soil.add(tuple(args)) # (rover, wp)
                elif pred == 'have_rock_analysis' and args[0] in self.objects['rover'] and args[1] in self.objects['waypoint']:
                    have_rock.add(tuple(args)) # (rover, wp)
                elif pred == 'have_image' and args[0] in self.objects['rover'] and args[1] in self.objects['objective'] and args[2] in self.objects['mode']:
                    have_image.add(tuple(args)) # (rover, obj, mode)
            except IndexError:
                 pass # Ignore malformed state facts

        # --- Calculate cost for each unmet goal ---
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            num_unmet_goals += 1
            parts = get_parts(goal_fact)
            if not parts: continue # Skip invalid goal facts
            pred = parts[0]
            args = parts[1:]
            min_cost_for_goal = float('inf')

            try:
                # --- Case 1: communicated_soil_data(wp) ---
                if pred == 'communicated_soil_data' and args[0] in self.waypoints:
                    wp_goal = args[0]
                    # Option A: Rover already has data
                    current_min_a = float('inf')
                    rovers_with_data = {r for r, wp in have_soil if wp == wp_goal}
                    for rover in rovers_with_data:
                        if rover not in rover_locations: continue # Rover location unknown
                        rover_wp = rover_locations[rover]
                        cost_nav, _ = self._bfs_shortest_path_and_dest(rover, rover_wp, self.waypoints_visible_from_lander)
                        if cost_nav != float('inf'):
                            current_min_a = min(current_min_a, cost_nav + 1) # navigate + communicate

                    # Option B: Need to sample and communicate
                    current_min_b = float('inf')
                    # Consider only rovers equipped for soil analysis
                    possible_rovers = [r for r, equip in self.rover_equipment.items() if equip['soil']]
                    for rover in possible_rovers:
                        if rover not in rover_locations: continue # Rover location unknown
                        rover_wp = rover_locations[rover]
                        store = self.rover_stores.get(rover) # Get store associated with rover
                        cost_drop = 0
                        # Check store status, default to 'empty' if not specified? Assume empty if not full.
                        if store and store_status.get(store) == 'full': cost_drop = 1

                        # Path to sample location
                        cost_nav1, _ = self._bfs_shortest_path_and_dest(rover, rover_wp, {wp_goal})
                        if cost_nav1 == float('inf'): continue # Cannot reach sample wp

                        # Path from sample location to communication spot
                        cost_nav2, _ = self._bfs_shortest_path_and_dest(rover, wp_goal, self.waypoints_visible_from_lander)
                        if cost_nav2 == float('inf'): continue # Cannot reach comm spot from sample wp

                        cost = cost_drop + cost_nav1 + 1 + cost_nav2 + 1 # drop? + nav1 + sample + nav2 + comm
                        current_min_b = min(current_min_b, cost)

                    min_cost_for_goal = min(current_min_a, current_min_b)

                # --- Case 2: communicated_rock_data(wp) ---
                elif pred == 'communicated_rock_data' and args[0] in self.waypoints:
                    wp_goal = args[0]
                    # Option A: Rover already has data
                    current_min_a = float('inf')
                    rovers_with_data = {r for r, wp in have_rock if wp == wp_goal}
                    for rover in rovers_with_data:
                        if rover not in rover_locations: continue
                        rover_wp = rover_locations[rover]
                        cost_nav, _ = self._bfs_shortest_path_and_dest(rover, rover_wp, self.waypoints_visible_from_lander)
                        if cost_nav != float('inf'): current_min_a = min(current_min_a, cost_nav + 1)

                    # Option B: Need to sample and communicate
                    current_min_b = float('inf')
                    possible_rovers = [r for r, equip in self.rover_equipment.items() if equip['rock']]
                    for rover in possible_rovers:
                        if rover not in rover_locations: continue
                        rover_wp = rover_locations[rover]
                        store = self.rover_stores.get(rover)
                        cost_drop = 0
                        if store and store_status.get(store) == 'full': cost_drop = 1

                        cost_nav1, _ = self._bfs_shortest_path_and_dest(rover, rover_wp, {wp_goal})
                        if cost_nav1 == float('inf'): continue
                        cost_nav2, _ = self._bfs_shortest_path_and_dest(rover, wp_goal, self.waypoints_visible_from_lander)
                        if cost_nav2 == float('inf'): continue

                        cost = cost_drop + cost_nav1 + 1 + cost_nav2 + 1
                        current_min_b = min(current_min_b, cost)

                    min_cost_for_goal = min(current_min_a, current_min_b)

                # --- Case 3: communicated_image_data(obj, mode) ---
                elif pred == 'communicated_image_data' and args[0] in self.objects['objective'] and args[1] in self.objects['mode']:
                    obj_goal, mode_goal = args
                    # Option A: Rover already has image
                    current_min_a = float('inf')
                    rovers_with_data = {r for r, o, m in have_image if o == obj_goal and m == mode_goal}
                    for rover in rovers_with_data:
                        if rover not in rover_locations: continue
                        rover_wp = rover_locations[rover]
                        cost_nav, _ = self._bfs_shortest_path_and_dest(rover, rover_wp, self.waypoints_visible_from_lander)
                        if cost_nav != float('inf'): current_min_a = min(current_min_a, cost_nav + 1)

                    # Option B: Need to take image and communicate
                    current_min_b = float('inf')
                    possible_rovers = [r for r, equip in self.rover_equipment.items() if equip['imaging']]
                    for rover in possible_rovers:
                        if rover not in rover_locations: continue
                        rover_wp = rover_locations[rover]
                        # Check cameras on this rover
                        if rover in self.cameras:
                            for camera, cam_data in self.cameras[rover].items():
                                # Check if camera supports the required mode
                                if mode_goal in cam_data.get('modes', set()):
                                    cost_cal = 0
                                    wp_after_cal = rover_wp # Position after potential calibration

                                    # --- Calibration Step ---
                                    if (camera, rover) not in calibrated_cameras:
                                        cal_target = cam_data.get('cal_target')
                                        # Check if calibration is possible (target exists and is visible)
                                        if not cal_target or cal_target not in self.calibration_visibility: continue
                                        cal_target_visible_from = self.calibration_visibility[cal_target]
                                        if not cal_target_visible_from: continue # Cannot see calibration target

                                        path_to_cal, closest_cal_wp = self._bfs_shortest_path_and_dest(rover, rover_wp, cal_target_visible_from)
                                        if path_to_cal == float('inf'): continue # Cannot reach calibration spot
                                        cost_cal = path_to_cal + 1 # Add cost for nav + calibrate action
                                        wp_after_cal = closest_cal_wp # Update rover position estimate

                                    # --- Imaging Step ---
                                    # Check if objective is visible from anywhere
                                    if obj_goal not in self.objective_visibility: continue
                                    img_visible_from = self.objective_visibility[obj_goal]
                                    if not img_visible_from: continue # Cannot see objective

                                    path_to_img, closest_img_wp = self._bfs_shortest_path_and_dest(rover, wp_after_cal, img_visible_from)
                                    if path_to_img == float('inf'): continue # Cannot reach imaging spot
                                    cost_nav_img = path_to_img # Cost to navigate to image spot
                                    wp_after_img = closest_img_wp # Update position estimate

                                    # --- Communication Step ---
                                    path_to_comm, _ = self._bfs_shortest_path_and_dest(rover, wp_after_img, self.waypoints_visible_from_lander)
                                    if path_to_comm == float('inf'): continue # Cannot reach comm spot
                                    cost_nav_comm = path_to_comm # Cost to navigate to comm spot

                                    # --- Total Cost ---
                                    # (NavToCal + Calibrate) + NavToImage + TakeImage + NavToComm + Communicate
                                    cost = cost_cal + cost_nav_img + 1 + cost_nav_comm + 1
                                    current_min_b = min(current_min_b, cost)

                    min_cost_for_goal = min(current_min_a, current_min_b)

            except IndexError:
                 # Treat as unreachable if goal format is wrong
                 min_cost_for_goal = float('inf')

            # --- Accumulate cost ---
            if min_cost_for_goal == float('inf'):
                # If any goal is deemed unreachable by this heuristic, the state might be a dead end.
                return float('inf')
            else:
                h_value += min_cost_for_goal

        # --- Final check ---
        # If heuristic calculated is 0 but goals are not met, return number of unmet goals.
        # This ensures the heuristic value is > 0 for non-goal states.
        if h_value == 0 and num_unmet_goals > 0:
             return num_unmet_goals

        # Ensure heuristic is non-negative
        return max(0, h_value)
