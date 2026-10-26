import heapq # Not used in current BFS, but potentially useful for Dijkstra
from collections import deque
from fnmatch import fnmatch # Not used in final heuristic, but good practice if pattern matching was needed.

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string by removing the parentheses
    and splitting the string by spaces.

    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]

    Args:
        fact (str): The PDDL fact string.

    Returns:
        list[str]: A list of strings representing the components of the fact.
                   Returns an empty list if the fact is malformed (e.g., not
                   enclosed in parentheses).
    """
    if fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return [] # Return empty list for malformed facts


class RoversHeuristic:
    """
    A domain-dependent heuristic for the PDDL Rovers domain, designed for use
    with Greedy Best-First Search.

    # Summary
    This heuristic estimates the total number of actions required to achieve all goals
    from the current state. It decomposes the problem by estimating the cost for each
    individual unsatisfied goal and summing these costs. The cost for each goal
    considers the necessary steps like navigation, sampling/imaging, calibration,
    and communication, assuming optimal choices for rovers and waypoints for that
    single goal, ignoring potential conflicts or synergies between goals.

    # Assumptions
    - The cost of each action (navigate, sample_soil, sample_rock, drop, calibrate,
      take_image, communicate_*) is 1.
    - The heuristic estimates the cost for each goal independently and sums them up.
      This ignores resource contention (e.g., a rover's store, camera calibration state)
      and potential positive interactions (e.g., one navigation sequence helping multiple goals).
      Therefore, the heuristic is likely inadmissible but aims to be informative for greedy search.
    - Shortest paths for navigation are calculated using Breadth-First Search (BFS) on the
      connectivity graph defined by `can_traverse` predicates for each rover.
    - If a required resource (like a soil sample at a location) is not present in the
      current state and no rover holds the analysis data, the heuristic assumes the goal
      related to that sample is currently impossible to achieve via sampling (cost infinity for that path).
      It *does* consider the case where another rover already holds the data.
    - Rovers need an empty store to sample. If their store is full, they must perform a `drop`
      action first (cost 1). This drop can happen anywhere.
    - Visibility between waypoints is handled by `can_traverse` for navigation and specific
      `visible` predicates for communication with the lander.
    - Visibility for calibration and imaging is handled by `visible_from` predicates.

    # Heuristic Initialization
    - Parses static facts provided in the task to build internal data structures:
        - Identifies all objects (rovers, waypoints, cameras, objectives, modes, lander, stores).
        - Determines the lander's location (`lander_location`).
        - Maps rovers to their capabilities (`rover_equipment`), onboard cameras (`rover_cameras`),
          and associated store (`rover_store`).
        - Maps cameras to supported modes (`camera_supports`) and calibration targets (`camera_calibration_target`).
        - Determines waypoint visibility for objectives (`objective_visibility`) and calibration targets
          (`calibration_target_visibility`).
        - Identifies waypoints suitable for communication (`comm_locations`) based on visibility from the lander.
        - Constructs rover-specific navigation graphs (`waypoint_graph`) based on `can_traverse` predicates.
    - Stores the set of goal predicates (`goals`).
    - Initializes a cache (`path_cache`) to store results of shortest path calculations (BFS) to avoid
      redundant computations during search.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost `h = 0`.
    2.  Parse the current state (a set of facts) to determine the dynamic information:
        - Current location of each rover (`rover_locations`).
        - Status of each rover's store (full or empty, identified via `rover_stores_full`).
        - Calibration status of each camera on each rover (`calibrated_cameras`).
        - Which soil/rock analyses (`have_soil`, `have_rock`) and images (`have_image`) are currently held by which rovers.
        - Current locations of available soil and rock samples (`at_soil_sample_locations`, `at_rock_sample_locations`).
        - Which communication goals have already been achieved (`achieved_goals`).
    3.  Identify all goal predicates defined in the task (`self.goals`) that are *not* satisfied in the current state
        (i.e., not in `achieved_goals`).
    4.  For each unsatisfied goal `g`:
        a.  **Estimate Cost for `g`:** Calculate the minimum estimated actions `cost(g)` required to achieve this specific goal `g`, considering all possible rovers and paths, assuming `g` is the only goal to be achieved right now.
            i.   **Check if data already acquired:** Determine if the necessary data (`have_soil_analysis`, `have_rock_analysis`, `have_image`) for goal `g` is already held by *any* rover `r`.
                 - If yes: Find the rover `r` holding the data. Calculate the cost as `min_nav_cost(r, current_pos, comm_pos_set) + 1 (communicate)`. The `min_nav_cost` finds the shortest path from the rover's current location `current_pos` to *any* waypoint in the set `comm_pos_set` suitable for communication.
                 - If no: Calculate the cost to acquire the data *and then* communicate it.
            ii.  **Cost to Acquire Data (if not already held):**
                 - **Sampling (Soil/Rock):** Find all rovers `r` equipped for the required analysis. For each such rover, calculate the cost: `min_nav_cost(r, current_pos, {sample_pos})` + `1 (drop, if store full)` + `1 (sample)` + `min_nav_cost(r, sample_pos, comm_pos_set)` + `1 (communicate)`. This requires the sample to be available at `sample_pos`. Take the minimum cost across all capable rovers.
                 - **Imaging:** Find all rovers `r` equipped for imaging with a suitable camera `c` supporting the required mode `m`. For each `r` and `c`:
                     - If `c` is calibrated: Cost = `min_nav_cost(r, current_pos, image_pos_set)` + `1 (take_image)` + `min_nav_cost(r, image_pos, comm_pos_set)` + `1 (communicate)`. `image_pos_set` contains waypoints visible from the objective. `image_pos` is the specific waypoint reached on the shortest path.
                     - If `c` is not calibrated: Cost = `min_nav_cost(r, current_pos, calib_pos_set)` + `1 (calibrate)` + `min_nav_cost(r, calib_pos, image_pos_set)` + `1 (take_image)` + `min_nav_cost(r, image_pos, comm_pos_set)` + `1 (communicate)`. `calib_pos_set` contains waypoints visible from the calibration target. `calib_pos` and `image_pos` are specific waypoints reached.
                 Take the minimum cost across all capable rovers/cameras.
            iii. **`min_nav_cost(rover, start, end_set)`:** This helper function calculates the minimum navigation cost (number of `navigate` actions) for the specified `rover` to get from waypoint `start` to *any* waypoint in the frozenset `end_set`. It uses BFS on the rover's specific navigation graph (`can_traverse` predicates) and caches results. If no path exists, it returns infinity. It also determines the specific waypoint reached (`best_wp`) to enable calculating the cost of subsequent steps accurately.
        b.  Add the calculated minimum cost `cost(g)` to the total heuristic cost `h`. If `cost(g)` is infinity for any goal (meaning the heuristic estimates it's impossible to achieve that goal from the current state), the total heuristic value becomes infinity.
    5.  Return `h`. A value of 0 indicates all goals are satisfied. Infinity indicates the heuristic estimates the goal state is unreachable.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.landers = set()
        self.stores = set()

        self.lander_location = None
        self.rover_equipment = {} # rover -> set(equipment_type) where type is 'soil', 'rock', 'imaging'
        self.rover_cameras = {} # rover -> set(camera)
        self.rover_store = {} # rover -> store
        self.camera_supports = {} # camera -> set(mode)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visibility = {} # objective -> set(waypoint)
        self.calibration_target_visibility = {} # objective -> set(waypoint) # Derived
        self.comm_locations = set() # waypoints visible from lander
        self.waypoint_graph = {} # rover -> dict(waypoint -> list(neighbor_waypoints))

        # --- Parse Static Facts ---
        temp_calib_targets = {} # camera -> objective
        lander_wp = None

        # First pass: identify all objects and basic types from static facts
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts
             pred = parts[0]

             # Add objects to sets based on predicates
             if pred == 'rover': self.rovers.add(parts[1])
             elif pred == 'waypoint': self.waypoints.add(parts[1])
             elif pred == 'camera': self.cameras.add(parts[1])
             elif pred == 'objective': self.objectives.add(parts[1])
             elif pred == 'mode': self.modes.add(parts[1])
             elif pred == 'lander': self.landers.add(parts[1])
             elif pred == 'store': self.stores.add(parts[1])
             # Also add objects mentioned in relations
             elif pred in ('at_lander', 'store_of', 'on_board', 'calibration_target',
                           'visible_from', 'can_traverse', 'visible', 'supports',
                           'equipped_for_soil_analysis', 'equipped_for_rock_analysis',
                           'equipped_for_imaging'):
                 for obj in parts[1:]:
                     # Basic check: is it likely an object name?
                     # A proper parser would use types. This is an approximation.
                     if obj.islower() or '-' in obj or obj.isdigit() or any(c.isdigit() for c in obj):
                         # Infer type based on predicate argument position (crude)
                         # This helps ensure all objects are captured even if not explicitly typed
                         pass # We mainly rely on predicate structure later

        # Second pass: populate relations and ensure all objects are captured
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            # Populate relations
            if pred == 'at_lander':
                self.lander_location = parts[2]
                lander_wp = parts[2]
                self.landers.add(parts[1])
                self.waypoints.add(parts[2])
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('soil')
                self.rovers.add(rover)
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('rock')
                self.rovers.add(rover)
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('imaging')
                self.rovers.add(rover)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
                self.cameras.add(camera)
                self.rovers.add(rover)
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_store[rover] = store
                self.stores.add(store)
                self.rovers.add(rover)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_supports.setdefault(camera, set()).add(mode)
                self.cameras.add(camera)
                self.modes.add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                temp_calib_targets[camera] = objective
                self.cameras.add(camera)
                self.objectives.add(objective)
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
                self.objectives.add(objective)
                self.waypoints.add(waypoint)
            elif pred == 'can_traverse':
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.waypoint_graph.setdefault(rover, {}).setdefault(wp_from, []).append(wp_to)
                self.rovers.add(rover)
                self.waypoints.add(wp_from)
                self.waypoints.add(wp_to)
            elif pred == 'visible':
                 # Track waypoints involved in visibility for graph completion and comms check
                 self.waypoints.add(parts[1])
                 self.waypoints.add(parts[2])


        # Derive calibration target visibility using objective_visibility data
        for cam, target_obj in temp_calib_targets.items():
            self.camera_calibration_target[cam] = target_obj
            if target_obj in self.objective_visibility:
                 self.calibration_target_visibility[target_obj] = self.objective_visibility[target_obj]

        # Find communication locations (waypoints visible from the lander's location)
        if lander_wp:
            for fact in static_facts:
                parts = get_parts(fact)
                if not parts: continue
                pred = parts[0]
                if pred == 'visible':
                    # Check visibility relative to the lander's waypoint
                    if parts[1] == lander_wp and parts[2] in self.waypoints:
                         self.comm_locations.add(parts[2])
                    elif parts[2] == lander_wp and parts[1] in self.waypoints:
                         self.comm_locations.add(parts[1])
        # Ensure comm_locations only contains valid waypoints
        self.comm_locations = self.comm_locations.intersection(self.waypoints)


        # Ensure all rovers and waypoints have entries in waypoint_graph for BFS robustness
        for r in self.rovers:
             if r not in self.waypoint_graph:
                 self.waypoint_graph[r] = {}
        for r in self.rovers:
            for wp in self.waypoints:
                 # Ensure each waypoint exists as a key in the rover's graph dict
                 # This handles waypoints that might only be destinations or isolated
                 self.waypoint_graph[r].setdefault(wp, [])


        # --- Initialize Path Cache ---
        # Cache key: (rover, start_wp, frozenset(end_wp_set))
        # Cache value: (distance, target_waypoint_reached)
        self.path_cache = {}


    def _get_shortest_path(self, rover, start_wp, end_wp_set):
        """
        Calculates the shortest path distance (number of 'navigate' actions)
        and the specific target waypoint reached from start_wp to any waypoint
        in the frozenset end_wp_set for a given rover using BFS.

        Args:
            rover (str): The name of the rover.
            start_wp (str): The starting waypoint.
            end_wp_set (frozenset): A frozenset of target waypoints.

        Returns:
            tuple[int | float, str | None]: A tuple containing:
                - The shortest distance (int) or float('inf') if unreachable.
                - The specific waypoint in end_wp_set that was reached first
                  via a shortest path, or None if unreachable.

        Uses caching to store and retrieve results.
        """
        # Ensure end_wp_set is a frozenset for caching
        if not isinstance(end_wp_set, frozenset):
             end_wp_set = frozenset(end_wp_set)

        if not end_wp_set or start_wp is None or start_wp not in self.waypoints:
            return float('inf'), None

        if start_wp in end_wp_set:
            return 0, start_wp

        cache_key = (rover, start_wp, end_wp_set)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

        # Check if rover and start waypoint exist in the graph structure
        if rover not in self.waypoint_graph or start_wp not in self.waypoint_graph[rover]:
             self.path_cache[cache_key] = (float('inf'), None)
             return float('inf'), None

        queue = deque([(start_wp, 0)]) # (waypoint, distance)
        # Visited stores {waypoint: distance} to find shortest paths
        visited = {start_wp: 0}
        # Store predecessor to reconstruct path if needed (not strictly needed for distance)
        # predecessor = {start_wp: None}

        min_dist = float('inf')
        target_reached = None

        while queue:
            current_wp, dist = queue.popleft()

            # If we found a path already, don't explore paths longer than it
            if dist >= min_dist:
                continue

            # Explore neighbors using the rover-specific graph
            neighbors = self.waypoint_graph[rover].get(current_wp, [])
            for neighbor in neighbors:
                # Process neighbor only if it's a shorter path to it
                if neighbor not in visited or visited[neighbor] > dist + 1:
                    visited[neighbor] = dist + 1
                    # predecessor[neighbor] = current_wp # Store path if needed

                    if neighbor in end_wp_set:
                        # Found a path to a target waypoint
                        if (dist + 1) < min_dist:
                            min_dist = dist + 1
                            target_reached = neighbor
                        # Continue BFS briefly to ensure this is the shortest path overall
                        # but don't explore further from here if dist+1 >= min_dist found so far
                    elif (dist + 1) < min_dist: # Only add to queue if potentially shorter
                         queue.append((neighbor, dist + 1))

        # Cache the result (even if unreachable)
        self.path_cache[cache_key] = (min_dist, target_reached)
        return min_dist, target_reached


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.

        Args:
            node: The state node in the search graph, containing the current state.

        Returns:
            int | float: The estimated cost (number of actions) to reach the goal.
                         Returns 0 if the state is a goal state.
                         Returns float('inf') if the goal is deemed unreachable by the heuristic.
        """
        state = node.state
        h_cost = 0

        # --- Parse Current State ---
        rover_locations = {} # rover -> waypoint
        rover_stores_full = set() # set of rovers with full stores
        calibrated_cameras = set() # set of (camera, rover) tuples
        have_soil = set() # set of (rover, waypoint) tuples
        have_rock = set() # set of (rover, waypoint) tuples
        have_image = set() # set of (rover, objective, mode) tuples
        at_soil_sample_locations = set() # set of waypoints
        at_rock_sample_locations = set() # set of waypoints
        achieved_goals = set() # set of goal facts (strings) present in the state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]

            if pred == 'at':
                # Ensure parts[1] is a known rover and parts[2] is a known waypoint
                if parts[1] in self.rovers and parts[2] in self.waypoints:
                    rover_locations[parts[1]] = parts[2]
            elif pred == 'full':
                store = parts[1]
                # Find the rover associated with this store
                for r, s in self.rover_store.items():
                    if s == store:
                        rover_stores_full.add(r)
                        break
            elif pred == 'calibrated':
                # Ensure parts[1] is a camera and parts[2] is a rover
                if parts[1] in self.cameras and parts[2] in self.rovers:
                    calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'have_soil_analysis':
                 if parts[1] in self.rovers and parts[2] in self.waypoints:
                    have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis':
                 if parts[1] in self.rovers and parts[2] in self.waypoints:
                    have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image':
                 if parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                    have_image.add(tuple(parts[1:])) # (rover, objective, mode)
            elif pred == 'at_soil_sample':
                 if parts[1] in self.waypoints:
                    at_soil_sample_locations.add(parts[1])
            elif pred == 'at_rock_sample':
                 if parts[1] in self.waypoints:
                    at_rock_sample_locations.add(parts[1])
            # Check if goal predicates are already true
            elif pred in ('communicated_soil_data', 'communicated_rock_data', 'communicated_image_data'):
                 # Check if this fact is actually one of the goals
                 if fact in self.goals:
                     achieved_goals.add(fact)


        # --- Calculate Cost for Unsatisfied Goals ---
        num_unsatisfied = 0
        for goal_fact in self.goals:
            if goal_fact in achieved_goals:
                continue

            num_unsatisfied += 1
            goal_parts = get_parts(goal_fact)
            if not goal_parts: continue # Skip malformed goals
            goal_pred = goal_parts[0]
            min_goal_cost = float('inf')

            # --- Case 1: communicated_soil_data ---
            if goal_pred == 'communicated_soil_data':
                wp_goal = goal_parts[1]
                if wp_goal not in self.waypoints: continue # Goal involves unknown waypoint

                data_held_by = None
                for r_have, wp_have in have_soil:
                    if wp_have == wp_goal:
                        data_held_by = r_have
                        break

                if data_held_by:
                    # Data exists, just need to communicate
                    r_loc = rover_locations.get(data_held_by)
                    if r_loc:
                         nav_cost, _ = self._get_shortest_path(data_held_by, r_loc, frozenset(self.comm_locations))
                         if nav_cost != float('inf'):
                              min_goal_cost = min(min_goal_cost, nav_cost + 1) # +1 for communicate
                else:
                    # Need to sample and communicate
                    if wp_goal in at_soil_sample_locations:
                        for r in self.rovers:
                            if 'soil' in self.rover_equipment.get(r, set()) and r in rover_locations:
                                r_loc = rover_locations[r]
                                drop_cost = 1 if r in rover_stores_full else 0

                                cost_nav_to_sample, _ = self._get_shortest_path(r, r_loc, frozenset({wp_goal}))
                                if cost_nav_to_sample == float('inf'): continue

                                cost_nav_to_comm, _ = self._get_shortest_path(r, wp_goal, frozenset(self.comm_locations))
                                if cost_nav_to_comm == float('inf'): continue

                                current_rover_cost = cost_nav_to_sample + drop_cost + 1 + cost_nav_to_comm + 1
                                min_goal_cost = min(min_goal_cost, current_rover_cost)

            # --- Case 2: communicated_rock_data ---
            elif goal_pred == 'communicated_rock_data':
                wp_goal = goal_parts[1]
                if wp_goal not in self.waypoints: continue

                data_held_by = None
                for r_have, wp_have in have_rock:
                    if wp_have == wp_goal:
                        data_held_by = r_have
                        break

                if data_held_by:
                    r_loc = rover_locations.get(data_held_by)
                    if r_loc:
                         nav_cost, _ = self._get_shortest_path(data_held_by, r_loc, frozenset(self.comm_locations))
                         if nav_cost != float('inf'):
                              min_goal_cost = min(min_goal_cost, nav_cost + 1)
                else:
                    if wp_goal in at_rock_sample_locations:
                        for r in self.rovers:
                            if 'rock' in self.rover_equipment.get(r, set()) and r in rover_locations:
                                r_loc = rover_locations[r]
                                drop_cost = 1 if r in rover_stores_full else 0
                                cost_nav_to_sample, _ = self._get_shortest_path(r, r_loc, frozenset({wp_goal}))
                                if cost_nav_to_sample == float('inf'): continue
                                cost_nav_to_comm, _ = self._get_shortest_path(r, wp_goal, frozenset(self.comm_locations))
                                if cost_nav_to_comm == float('inf'): continue
                                current_rover_cost = cost_nav_to_sample + drop_cost + 1 + cost_nav_to_comm + 1
                                min_goal_cost = min(min_goal_cost, current_rover_cost)

            # --- Case 3: communicated_image_data ---
            elif goal_pred == 'communicated_image_data':
                obj_goal, mode_goal = goal_parts[1], goal_parts[2]
                if obj_goal not in self.objectives or mode_goal not in self.modes: continue

                data_held_by = None
                for r_have, o_have, m_have in have_image:
                    if o_have == obj_goal and m_have == mode_goal:
                        data_held_by = r_have
                        break

                if data_held_by:
                    r_loc = rover_locations.get(data_held_by)
                    if r_loc:
                         nav_cost, _ = self._get_shortest_path(data_held_by, r_loc, frozenset(self.comm_locations))
                         if nav_cost != float('inf'):
                              min_goal_cost = min(min_goal_cost, nav_cost + 1)
                else:
                    # Need to take image and communicate
                    possible_image_locations = frozenset(self.objective_visibility.get(obj_goal, set()))
                    if not possible_image_locations: continue # Cannot take image if objective not visible anywhere

                    for r in self.rovers:
                        if 'imaging' in self.rover_equipment.get(r, set()) and r in rover_locations:
                            r_loc = rover_locations[r]
                            for cam in self.rover_cameras.get(r, set()):
                                if mode_goal in self.camera_supports.get(cam, set()):
                                    # This rover+camera can potentially do the job
                                    is_calibrated = (cam, r) in calibrated_cameras
                                    cost_to_calibrate = 0
                                    loc_after_cal = r_loc # Start from current location if already calibrated

                                    if not is_calibrated:
                                        cal_target = self.camera_calibration_target.get(cam)
                                        if not cal_target: continue # Camera needs calibration target defined
                                        possible_cal_locations = frozenset(self.calibration_target_visibility.get(cal_target, set()))
                                        if not possible_cal_locations: continue # Target objective must be visible from somewhere

                                        nav_cost_to_cal, best_cal_wp = self._get_shortest_path(r, r_loc, possible_cal_locations)
                                        if nav_cost_to_cal == float('inf'): continue # Cannot reach calibration spot

                                        cost_to_calibrate = nav_cost_to_cal + 1 # nav + calibrate action
                                        loc_after_cal = best_cal_wp # Location after calibration action

                                    # Calculate cost to navigate from loc_after_cal to image spot
                                    cost_nav_to_img, best_img_wp = self._get_shortest_path(r, loc_after_cal, possible_image_locations)
                                    if cost_nav_to_img == float('inf'): continue # Cannot reach imaging spot

                                    cost_to_take_image = cost_nav_to_img + 1 # nav + take_image action
                                    loc_after_image = best_img_wp

                                    # Calculate cost to navigate from image spot to communication spot
                                    cost_nav_to_comm, _ = self._get_shortest_path(r, loc_after_image, frozenset(self.comm_locations))
                                    if cost_nav_to_comm == float('inf'): continue # Cannot reach comm spot

                                    cost_to_communicate = cost_nav_to_comm + 1 # nav + communicate action

                                    current_total_cost = cost_to_calibrate + cost_to_take_image + cost_to_communicate
                                    min_goal_cost = min(min_goal_cost, current_total_cost)


            # Add cost for this goal (if reachable)
            if min_goal_cost != float('inf'):
                h_cost += min_goal_cost
            else:
                # If any single goal is estimated as unreachable, the state is considered dead-end
                return float('inf')

        # If all goals were already satisfied, num_unsatisfied is 0, h_cost is 0.
        # If there were unsatisfied goals but all could be achieved, h_cost > 0.
        # If any unsatisfied goal was estimated as unreachable, infinity was returned.
        return h_cost

