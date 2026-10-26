import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args and all parts match the pattern
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    conditions. It sums up the estimated costs for each unachieved goal fact.
    The cost for each goal fact includes non-navigation actions (sample, image,
    calibrate, communicate, drop) and an estimate of navigation cost based on
    shortest paths between relevant waypoints. The heuristic is non-admissible
    and designed to guide a greedy best-first search.

    # Assumptions
    - The navigation graph defined by `visible` predicates is static and symmetric.
    - Rovers can traverse any edge in the `visible` graph.
    - Lander location is static.
    - Camera calibration targets are static.
    - Camera capabilities and rover equipment are static.
    - Soil/rock samples, once collected, are held by a specific rover and removed from the waypoint.
    - Image data, once taken, is held by a specific rover.
    - Calibration is consumed by taking an image.
    - Dropping a sample frees up the store but the sample is lost (not needed for goals typically).
    - The heuristic assumes a typical sequence of actions for each goal type:
        - Soil/Rock: Navigate to sample -> Sample -> Navigate to lander comm point -> Communicate.
        - Image: Navigate to calibration point -> Calibrate -> Navigate to imaging point -> Take Image -> Navigate to lander comm point -> Communicate.
    - The heuristic tries to find the 'best' rover/camera/waypoint combination to minimize navigation for image goals.
    - If a required resource (like a sample at a waypoint) is no longer available in the state and no rover holds the data, the goal is considered unreachable from this state, and a large penalty is applied.

    # Heuristic Initialization
    - Parses static facts to build:
        - The navigation graph (waypoints and `visible` connections).
        - Shortest path distances between all pairs of waypoints using BFS.
        - Lander location and communication points (waypoints visible from lander).
        - Rover capabilities (equipped_for_soil/rock/imaging).
        - Store ownership.
        - Camera properties (on_board, supports, calibration_target).
        - Objective visibility from waypoints.
    - Stores goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0. Define a large penalty for unreachable goals.
    2. Extract dynamic information from the current state: rover locations, held samples/images, store status, camera calibration status, sample locations at waypoints, and already communicated data.
    3. For each goal fact in `self.goals`:
        a. If the goal fact is already true in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            i. Add 1 for the `communicate_soil_data` action.
            ii. Check if any rover `R` has `(have_soil_analysis R ?w)`.
            iii. If yes: Find `R`'s location `LocR`. Find the nearest communication point `CommP`. Add `dist[LocR][CommP]` to the cost. If `CommP` is unreachable, return penalty.
            iv. If no: Add 1 for `sample_soil`. Check if `(at_soil_sample ?w)` is true. If not, return penalty. Find equipped rover `R`. If `R`'s store is full, add 1 for `drop`. Find `R`'s location `LocR`. Find nearest `CommP`. Add `dist[LocR][?w]` + `dist[?w][CommP]` to the cost. Minimize this navigation cost over all suitable rovers. If no suitable rover or path exists, return penalty.
        c. If the goal is `(communicated_rock_data ?w)`: Similar logic to soil data, using rock-specific predicates and capabilities.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            i. Add 1 for `communicate_image_data`.
            ii. Check if any rover `R` has `(have_image R ?o ?m)`.
            iii. If yes: Find `R`'s location `LocR`. Find nearest `CommP`. Add `dist[LocR][CommP]` to the cost. If `CommP` is unreachable, return penalty.
            iv. If no: Add 1 for `take_image`. Find suitable rover `R` (equipped for imaging), camera `I` (on board R, supports M). Find imaging waypoint `ImgP` (`visible_from ?o ImgP`). Find calibration waypoint `CalP` (`visible_from T CalP` for target `T` of `I`). Find `R`'s location `LocR`. Check if `(calibrated I R)` is true.
                - If not calibrated: Add 1 for `calibrate`. Calculate navigation cost: `dist[LocR][CalP]` + `dist[CalP][ImgP]` + `dist[ImgP][CommP]`.
                - If calibrated: Calculate navigation cost: `dist[LocR][ImgP]` + `dist[ImgP][CommP]`.
                - Minimize the total navigation cost over all suitable combinations of (rover, camera, ImgP, CalP). If no suitable combination or path exists, return penalty. Add the minimum navigation cost found.
    4. Return the total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing navigation distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Extract Static Information ---
        self.waypoints = set()
        self.rovers = set()
        self.landers = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.objectives = set()

        self.rover_capabilities = collections.defaultdict(dict) # rover -> capability_type -> bool
        self.store_of_rover = {} # store -> rover
        self.camera_on_board = {} # camera -> rover
        self.camera_supports_mode = collections.defaultdict(set) # camera -> set(modes)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = collections.defaultdict(set) # objective -> set(waypoints)
        self.lander_location = None
        self.communication_points = set() # waypoints visible from lander

        # Navigation graph: waypoint -> set(neighbor_waypoints)
        self.nav_graph = collections.defaultdict(set)

        # Process static facts
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts

             predicate = parts[0]
             if predicate == 'at_lander':
                 self.landers.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.lander_location = parts[2]
             elif predicate == 'visible':
                 w1, w2 = parts[1], parts[2]
                 self.waypoints.add(w1)
                 self.waypoints.add(w2)
                 self.nav_graph[w1].add(w2)
                 self.nav_graph[w2].add(w1) # Assuming symmetric visibility implies symmetric traversability
             elif predicate == 'equipped_for_soil_analysis':
                 self.rovers.add(parts[1])
                 self.rover_capabilities[parts[1]]['soil'] = True
             elif predicate == 'equipped_for_rock_analysis':
                 self.rovers.add(parts[1])
                 self.rover_capabilities[parts[1]]['rock'] = True
             elif predicate == 'equipped_for_imaging':
                 self.rovers.add(parts[1])
                 self.rover_capabilities[parts[1]]['imaging'] = True
             elif predicate == 'store_of':
                 self.stores.add(parts[1])
                 self.rovers.add(parts[2])
                 self.store_of_rover[parts[1]] = parts[2]
             elif predicate == 'on_board':
                 self.cameras.add(parts[1])
                 self.rovers.add(parts[2])
                 self.camera_on_board[parts[1]] = parts[2]
             elif predicate == 'supports':
                 self.cameras.add(parts[1])
                 self.modes.add(parts[2])
                 self.camera_supports_mode[parts[1]].add(parts[2])
             elif predicate == 'calibration_target':
                 self.cameras.add(parts[1])
                 self.objectives.add(parts[2])
                 self.camera_calibration_target[parts[1]] = parts[2]
             elif predicate == 'visible_from':
                 self.objectives.add(parts[1])
                 self.waypoints.add(parts[2])
                 self.objective_visible_from[parts[1]].add(parts[2])
             # Add other types if needed, though predicates cover most objects

        # Identify communication points (waypoints visible from lander)
        if self.lander_location:
             for fact in static_facts:
                 if match(fact, "visible", "*", self.lander_location):
                     self.communication_points.add(get_parts(fact)[1])
                 elif match(fact, "visible", self.lander_location, "*"):
                      self.communication_points.add(get_parts(fact)[2]) # Should be symmetric, but double check

        # Precompute shortest paths using BFS
        self.dist = {}
        for start_node in self.waypoints:
            self.dist[start_node] = self._bfs(start_node)

        # Store mapping from objective+mode to suitable cameras and imaging/calibration waypoints
        # (objective, mode) -> list of (camera, rover, imaging_waypoint, calibration_waypoint)
        self.image_info = collections.defaultdict(list)
        for obj in self.objectives:
            for mode in self.modes:
                suitable_imaging_wps = self.objective_visible_from.get(obj, set())
                if not suitable_imaging_wps: continue # Cannot image this objective

                for camera in self.cameras:
                    if mode in self.camera_supports_mode.get(camera, set()):
                        rover = self.camera_on_board.get(camera)
                        if rover and self.rover_capabilities.get(rover, {}).get('imaging', False):
                            cal_target = self.camera_calibration_target.get(camera)
                            suitable_cal_wps = self.objective_visible_from.get(cal_target, set()) if cal_target else set()

                            # Store combinations of (camera, rover, imaging_wp, calibration_wp)
                            # We need at least one imaging_wp. Calibration_wp might not exist if no target.
                            if suitable_imaging_wps:
                                if suitable_cal_wps:
                                    for img_wp in suitable_imaging_wps:
                                        for cal_wp in suitable_cal_wps:
                                            self.image_info[(obj, mode)].append((camera, rover, img_wp, cal_wp))
                                else: # No calibration target/waypoint needed/possible
                                     for img_wp in suitable_imaging_wps:
                                         self.image_info[(obj, mode)].append((camera, rover, img_wp, None))


    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.waypoints}
        if start_node not in self.waypoints:
             return distances # Start node not in graph

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            for neighbor in self.nav_graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def _get_nearest_waypoint_dist(self, from_wp, target_wps):
        """Finds the minimum distance from from_wp to any waypoint in target_wps."""
        if not target_wps:
            return float('inf') # No target waypoints available

        min_dist = float('inf')
        from_dist = self.dist.get(from_wp)
        if not from_dist:
             return float('inf') # from_wp is not in the graph

        for target_wp in target_wps:
            min_dist = min(min_dist, from_dist.get(target_wp, float('inf')))
        return min_dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Extract Dynamic Information from State ---
        rover_locations = {} # rover -> waypoint
        rover_has_soil = collections.defaultdict(set) # rover -> set(waypoints_with_soil_data)
        rover_has_rock = collections.defaultdict(set) # rover -> set(waypoints_with_rock_data)
        rover_has_image = collections.defaultdict(set) # rover -> set((objective, mode))
        rover_store_full = {} # rover -> bool
        rover_calibrated_cameras = collections.defaultdict(set) # rover -> set(calibrated_cameras)
        soil_samples_at_waypoint = set() # waypoints with soil samples
        rock_samples_at_waypoint = set() # waypoints with rock samples
        # communicated goals are checked directly against state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                rover_locations[parts[1]] = parts[2]
            elif predicate == 'have_soil_analysis':
                rover_has_soil[parts[1]].add(parts[2])
            elif predicate == 'have_rock_analysis':
                rover_has_rock[parts[1]].add(parts[2])
            elif predicate == 'have_image':
                # Ensure fact has enough parts before accessing indices
                if len(parts) > 3:
                    rover_has_image[parts[1]].add((parts[2], parts[3]))
            elif predicate == 'full':
                 # Find which rover this store belongs to
                 store_name = parts[1]
                 rover_name = self.store_of_rover.get(store_name)
                 if rover_name:
                     rover_store_full[rover_name] = True
            elif predicate == 'calibrated':
                 # Ensure fact has enough parts before accessing indices
                 if len(parts) > 2:
                    rover_calibrated_cameras[parts[2]].add(parts[1]) # camera, rover
            elif predicate == 'at_soil_sample':
                soil_samples_at_waypoint.add(parts[1])
            elif predicate == 'at_rock_sample':
                rock_samples_at_waypoint.add(parts[1])
            # communicated goals are checked directly against self.goals <= state


        total_cost = 0
        large_unreachable_cost = 10000 # Penalty for potentially unreachable goals

        # --- Estimate Cost for Each Unachieved Goal ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                total_cost += 1 # Cost for communicate action

                # Check if data is already collected by any rover
                data_held = False
                for rover, waypoints_held in rover_has_soil.items():
                    if waypoint in waypoints_held:
                        # Data is held by 'rover'
                        data_held = True
                        rover_loc = rover_locations.get(rover)
                        if rover_loc:
                            # Add navigation cost from rover's current location to nearest comm point
                            nav_cost = self._get_nearest_waypoint_dist(rover_loc, self.communication_points)
                            if nav_cost == float('inf'): return large_unreachable_cost # Cannot reach comm point
                            total_cost += nav_cost
                        else:
                             # Rover location unknown? Should not happen in valid states.
                             return large_unreachable_cost
                        break # Found a rover with the data, move to next goal

                if not data_held:
                    # Data needs to be sampled
                    total_cost += 1 # Cost for sample_soil action

                    # Check if sample is still at the waypoint
                    if waypoint not in soil_samples_at_waypoint:
                         # Sample is gone and no rover has it. Goal likely unreachable.
                         return large_unreachable_cost

                    # Find a suitable rover (equipped for soil)
                    suitable_rovers = [r for r in self.rovers if self.rover_capabilities.get(r, {}).get('soil', False)]
                    if not suitable_rovers:
                         # No rover can sample soil. Goal unreachable.
                         return large_unreachable_cost

                    # Find the best rover to perform the task (e.g., closest one)
                    min_nav_cost = float('inf')

                    for rover in suitable_rovers:
                        rover_loc = rover_locations.get(rover)
                        if rover_loc:
                            nav_to_sample = self.dist.get(rover_loc, {}).get(waypoint, float('inf'))
                            if nav_to_sample != float('inf'):
                                # Navigation: LocR -> SampleWp -> CommWp
                                nav_sample_to_comm = self._get_nearest_waypoint_dist(waypoint, self.communication_points)
                                if nav_sample_to_comm != float('inf'):
                                     current_nav_cost = nav_to_sample + nav_sample_to_comm
                                     # Add drop cost if store is full
                                     if rover_store_full.get(rover, False):
                                         current_nav_cost += 1 # Add 1 for drop action (assuming it happens before sampling)

                                     min_nav_cost = min(min_nav_cost, current_nav_cost)

                    if min_nav_cost == float('inf'):
                         # Cannot reach sample or comm point with any suitable rover
                         return large_unreachable_cost

                    total_cost += min_nav_cost


            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                total_cost += 1 # Cost for communicate action

                # Check if data is already collected by any rover
                data_held = False
                for rover, waypoints_held in rover_has_rock.items():
                    if waypoint in waypoints_held:
                        # Data is held by 'rover'
                        data_held = True
                        rover_loc = rover_locations.get(rover)
                        if rover_loc:
                            # Add navigation cost from rover's current location to nearest comm point
                            nav_cost = self._get_nearest_waypoint_dist(rover_loc, self.communication_points)
                            if nav_cost == float('inf'): return large_unreachable_cost # Cannot reach comm point
                            total_cost += nav_cost
                        else:
                             return large_unreachable_cost
                        break # Found a rover with the data, move to next goal

                if not data_held:
                    # Data needs to be sampled
                    total_cost += 1 # Cost for sample_rock action

                    # Check if sample is still at the waypoint
                    if waypoint not in rock_samples_at_waypoint:
                         # Sample is gone and no rover has it. Goal likely unreachable.
                         return large_unreachable_cost

                    # Find a suitable rover (equipped for rock)
                    suitable_rovers = [r for r in self.rovers if self.rover_capabilities.get(r, {}).get('rock', False)]
                    if not suitable_rovers:
                         # No rover can sample rock. Goal unreachable.
                         return large_unreachable_cost

                    # Find the best rover to perform the task (e.g., closest one)
                    min_nav_cost = float('inf')

                    for rover in suitable_rovers:
                        rover_loc = rover_locations.get(rover)
                        if rover_loc:
                            nav_to_sample = self.dist.get(rover_loc, {}).get(waypoint, float('inf'))
                            if nav_to_sample != float('inf'):
                                # Navigation: LocR -> SampleWp -> CommWp
                                nav_sample_to_comm = self._get_nearest_waypoint_dist(waypoint, self.communication_points)
                                if nav_sample_to_comm != float('inf'):
                                     current_nav_cost = nav_to_sample + nav_sample_to_comm
                                     # Add drop cost if store is full
                                     if rover_store_full.get(rover, False):
                                         current_nav_cost += 1 # Add 1 for drop action

                                     min_nav_cost = min(min_nav_cost, current_nav_cost)

                    if min_nav_cost == float('inf'):
                         # Cannot reach sample or comm point with any suitable rover
                         return large_unreachable_cost

                    total_cost += min_nav_cost


            elif predicate == 'communicated_image_data':
                # Ensure goal fact has enough parts
                if len(parts) <= 2:
                     # Malformed goal fact? Treat as unreachable.
                     return large_unreachable_cost

                objective, mode = parts[1], parts[2]
                total_cost += 1 # Cost for communicate action

                # Check if image is already collected by any rover
                image_held = False
                for rover, images_held in rover_has_image.items():
                    if (objective, mode) in images_held:
                        # Image is held by 'rover'
                        image_held = True
                        rover_loc = rover_locations.get(rover)
                        if rover_loc:
                            # Add navigation cost from rover's current location to nearest comm point
                            nav_cost = self._get_nearest_waypoint_dist(rover_loc, self.communication_points)
                            if nav_cost == float('inf'): return large_unreachable_cost # Cannot reach comm point
                            total_cost += nav_cost
                        else:
                             return large_unreachable_cost
                        break # Found a rover with the image, move to next goal

                if not image_held:
                    # Image needs to be taken
                    total_cost += 1 # Cost for take_image action

                    # Find suitable rover/camera/waypoints for this image goal
                    possible_image_paths = self.image_info.get((objective, mode), [])
                    if not possible_image_paths:
                         # No suitable camera/rover/imaging point exists. Goal unreachable.
                         return large_unreachable_cost

                    min_total_nav_cost = float('inf')
                    best_path_found = False

                    # Iterate through all possible combinations of (camera, rover, imaging_wp, calibration_wp)
                    for camera, rover, img_wp, cal_wp in possible_image_paths:
                        rover_loc = rover_locations.get(rover)
                        if not rover_loc: continue # Rover location unknown

                        current_nav_cost = 0
                        is_calibrated = camera in rover_calibrated_cameras.get(rover, set())

                        # Navigation for calibration (if needed)
                        if cal_wp and not is_calibrated:
                            total_cost += 1 # Cost for calibrate action
                            nav_to_cal = self.dist.get(rover_loc, {}).get(cal_wp, float('inf'))
                            if nav_to_cal == float('inf'): continue # Cannot reach calibration point
                            current_nav_cost += nav_to_cal
                            # Navigation from calibration point to imaging point
                            nav_cal_to_img = self.dist.get(cal_wp, {}).get(img_wp, float('inf'))
                            if nav_cal_to_img == float('inf'): continue # Cannot reach imaging point from cal point
                            current_nav_cost += nav_cal_to_img
                        else: # No calibration needed or already calibrated
                            # Navigation directly to imaging point
                            nav_to_img = self.dist.get(rover_loc, {}).get(img_wp, float('inf'))
                            if nav_to_img == float('inf'): continue # Cannot reach imaging point
                            current_nav_cost += nav_to_img

                        # Navigation from imaging point to communication point
                        nav_img_to_comm = self._get_nearest_waypoint_dist(img_wp, self.communication_points)
                        if nav_img_to_comm == float('inf'): continue # Cannot reach comm point from imaging point
                        current_nav_cost += nav_img_to_comm

                        # Update minimum total navigation cost for this image goal
                        min_total_nav_cost = min(min_total_nav_cost, current_nav_cost)
                        best_path_found = True

                    if not best_path_found or min_total_nav_cost == float('inf'):
                         # Cannot find a reachable path for imaging and communication
                         return large_unreachable_cost

                    total_cost += min_total_nav_cost

            # Add costs for other potential goal types if they existed (e.g., (at ?obj ?loc))
            # The rovers domain goals are only about communication.

        # If we reached here, there are unachieved goals, and total_cost should be > 0.
        # The only way it could be 0 is if all unachieved goals resulted in returning
        # large_unreachable_cost, but we didn't return it. This scenario is covered
        # by the logic returning large_unreachable_cost inside the loop.
        # Thus, for any non-goal state where a solution exists, total_cost will be > 0.
        # For non-goal states where no solution exists, large_unreachable_cost is returned.

        return total_cost
