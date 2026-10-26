import math
from collections import deque
from fnmatch import fnmatch
# Ensure this import path is correct for the planner's environment
# This might need adjustment based on how the planner loads heuristics.
# Common options include:
# from heuristics.heuristic_base import Heuristic
# from .heuristic_base import Heuristic  (if in the same package)
from heuristics.heuristic_base import Heuristic # Assuming standard structure

# Helper functions (can be defined outside or as static/private methods)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handles facts like "(pred obj1 obj2)" -> ["pred", "obj1", "obj2"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using fnmatch for wildcard support.
    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern elements (strings, potentially with wildcards like '*').
    - Returns `True` if the fact's structure and elements match the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the number of pattern elements
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define a large number to represent infinity for practical cost calculations.
# This avoids math.inf which might cause issues in summation or comparisons.
LARGE_DISTANCE = 1_000_000

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    communication goals specified in the problem (`communicated_soil_data`,
    `communicated_rock_data`, `communicated_image_data`). It works by calculating
    an estimated cost for achieving each unsatisfied goal independently and summing
    these costs. The cost for each goal considers the necessary steps: navigation,
    potential sample collection (soil/rock) or image capture (including calibration),
    and final communication with the lander. It assigns each goal to the rover
    that can achieve it with the minimum estimated cost based on precomputed shortest paths.

    # Assumptions
    - The heuristic assumes that each goal can be achieved independently, ignoring
      potential positive or negative interactions between achieving different goals
      (e.g., a single rover trip fulfilling multiple objectives, or resource contention).
    - Distances between waypoints are calculated as the shortest path in terms of
      the number of `navigate` actions, based on the rover-specific `can_traverse` predicates.
    - If a required resource (e.g., sample at waypoint, visibility for communication/imaging/calibration,
      or path non-existence) is unavailable, achieving that goal via that path is considered
      infinitely costly (represented by `LARGE_DISTANCE`).
    - Rovers can drop samples anywhere, incurring one action cost (`drop`), as the action
      has no location precondition.
    - The heuristic is designed for greedy best-first search and is not necessarily admissible
      (it might overestimate the true cost).

    # Heuristic Initialization
    - The constructor (`__init__`) performs precomputation based on static information from the task:
        - Identifies all relevant objects: rovers, waypoints, cameras, objectives, modes, stores, and the lander.
        - Stores rover capabilities (`equipped_for_*`) and associations (`store_of`, `on_board`).
        - Stores camera properties (`supports`, `calibration_target`).
        - Determines visibility relationships (`visible`, `visible_from`).
        - Finds the lander's fixed location (`at_lander`).
        - Identifies waypoints from which the lander is visible (for communication).
        - Builds a directed traversal graph for each rover based on `can_traverse` facts.
        - Computes all-pairs shortest paths (number of `navigate` actions) for each rover
          using Breadth-First Search (BFS) on their respective traversal graphs.
        - Parses the goal conditions (`task.goals`) to identify all required communication tasks.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse Current State:** Extract dynamic information from the input `node.state`:
        - Current location of each rover (`at`).
        - Store status for each rover (`empty` or `full`).
        - Calibration status of each camera (`calibrated`).
        - Data currently held by rovers (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
        - Data already communicated to the lander (`communicated_*`).
        - Current availability of samples at waypoints (`at_soil_sample`, `at_rock_sample`).
    2.  **Identify Unsatisfied Goals:** Compare the set of required communication goals (from `__init__`)
        with the `communicated_*` facts in the current state to find goals yet to be achieved.
    3.  **Estimate Cost per Unsatisfied Goal:** For each unsatisfied goal:
        a.  **Determine Goal Type:** Identify if it's soil data, rock data, or image data.
        b.  **Find Best Rover/Path:** Iterate through all rovers capable of performing the task
            (e.g., soil analysis for soil data, imaging with a suitable camera for image data).
        c.  **Calculate Cost for Each Capable Rover:** Estimate the minimum action sequence cost for that
            rover to achieve this single goal, considering the current state:
            i.   **If Data Already Held:** Calculate cost = navigate_to_comm_point + communicate (1 action).
                 - Find the minimum navigation cost from the rover's current location to any waypoint
                   visible to the lander (`waypoints_visible_from_lander`).
            ii.  **If Data Not Held (Soil/Rock):**
                 - Check if the required sample exists at the target waypoint (`at_soil_sample`/`at_rock_sample`
                   in the current state). If not, this path is impossible (cost = `LARGE_DISTANCE`).
                 - Check rover's store status. Add 1 action cost for `drop` if the store is `full`.
                 - Calculate cost = cost_drop + navigate_to_sample + sample (1) + navigate_to_comm + communicate (1).
                 - Navigation costs are based on precomputed shortest paths.
            iii. **If Data Not Held (Image):**
                 - Select a suitable camera on the rover that supports the required mode.
                 - Check if the camera is currently `calibrated`.
                 - **If Calibration Needed:**
                     - Find the closest reachable waypoint (`w_cal`) from the rover's current location that is
                       visible from any valid `calibration_target` for this camera.
                     - If no such waypoint exists or is reachable, calibration (and thus this path) is impossible.
                     - Cost_calibrate = navigate_to_cal_wp + calibrate (1). Update effective location to `w_cal`.
                 - **Take Image:**
                     - Find the closest reachable waypoint (`w_img`) from the location after calibration
                       (or current location if already calibrated) that is `visible_from` the target objective.
                     - If no such waypoint exists or is reachable, imaging is impossible.
                     - Cost_image = navigate_to_img_wp + take_image (1). Update effective location to `w_img`.
                 - **Communicate:**
                     - Find the closest reachable waypoint (`w_comm`) from `w_img` that is visible to the lander.
                     - If unreachable, communication is impossible.
                     - Cost_comm = navigate_to_comm_wp + communicate (1).
                 - Total cost = Cost_calibrate + Cost_image + Cost_comm (adjusting for shared navigation).
                   A simpler sum is used: cost_calibration_steps + cost_imaging_steps + cost_communication_steps.
        d.  **Select Minimum Cost:** Choose the minimum cost calculated across all capable rovers (and suitable cameras for imaging) for achieving this specific goal. If no rover can achieve it, the cost remains `LARGE_DISTANCE`.
    4.  **Sum Costs:** Add the minimum cost found for the current goal to the total `heuristic_value`.
    5.  **Return Total:** The final heuristic value is the sum of the minimum costs for all unsatisfied goals.
        - If all goals are satisfied, the value is 0.
        - If any goal is deemed impossible to achieve (cost >= `LARGE_DISTANCE`), the heuristic value will be very large.
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
        self.stores = set()
        self.lander = None
        self.lander_location = None

        self.rover_capabilities = {} # rover -> set{'soil', 'rock', 'imaging'}
        self.rover_stores_map = {} # rover -> store
        self.store_rover_map = {} # store -> rover
        self.rover_cameras = {} # rover -> set{camera}
        self.camera_supports = {} # camera -> set{mode}
        self.camera_calibration_targets = {} # camera -> set{objective}
        self.objective_visibility = {} # objective -> set{waypoint}
        self.calibration_target_visibility = {} # objective -> set{waypoint} # Objectives used for calibration
        self.waypoint_visibility = {} # waypoint -> set{waypoint}
        self.can_traverse = {} # rover -> from_wp -> set{to_wp}

        # --- Parse Static Facts ---
        # Collect all potential objects mentioned in static facts to help identify waypoints etc.
        potential_waypoints = set()
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            # Infer types and properties
            if pred == 'rover': self.rovers.add(parts[1])
            elif pred == 'camera': self.cameras.add(parts[1])
            elif pred == 'objective': self.objectives.add(parts[1])
            elif pred == 'mode': self.modes.add(parts[1])
            elif pred == 'store': self.stores.add(parts[1])
            elif pred == 'lander': self.lander = parts[1]
            # Collect waypoints from relevant predicates
            elif pred in ['at_lander', 'visible', 'can_traverse', 'visible_from']:
                 for part in parts[1:]:
                     # Basic check: if it looks like a waypoint name (often contains 'waypoint')
                     # A better approach relies on type info if available from the task object.
                     # For now, assume objects used in waypoint slots are waypoints.
                     if pred == 'at_lander' and len(parts) > 2: potential_waypoints.add(parts[2])
                     if pred == 'visible' and len(parts) > 2: potential_waypoints.add(parts[1]); potential_waypoints.add(parts[2])
                     if pred == 'can_traverse' and len(parts) > 3: potential_waypoints.add(parts[2]); potential_waypoints.add(parts[3])
                     if pred == 'visible_from' and len(parts) > 2: potential_waypoints.add(parts[2])

            # Populate relations
            if pred == 'at_lander':
                self.lander = parts[1]
                self.lander_location = parts[2]
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]; self.rovers.add(rover)
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]; self.rovers.add(rover)
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif pred == 'equipped_for_imaging':
                rover = parts[1]; self.rovers.add(rover)
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.stores.add(store); self.rovers.add(rover)
                self.rover_stores_map[rover] = store
                self.store_rover_map[store] = rover
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.cameras.add(camera); self.rovers.add(rover)
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.cameras.add(camera); self.modes.add(mode)
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.cameras.add(camera); self.objectives.add(objective)
                self.camera_calibration_targets.setdefault(camera, set()).add(objective)
            elif pred == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoint_visibility.setdefault(wp1, set()).add(wp2)
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objectives.add(objective)
                self.objective_visibility.setdefault(objective, set()).add(waypoint)
            elif pred == 'can_traverse':
                rover, wp_from, wp_to = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                self.can_traverse.setdefault(rover, {}).setdefault(wp_from, set()).add(wp_to)

        # Finalize the set of waypoints
        self.waypoints = potential_waypoints
        if self.lander_location:
             self.waypoints.add(self.lander_location)
        # Add waypoints mentioned in can_traverse just in case
        for r, paths in self.can_traverse.items():
            for wp_from, wps_to in paths.items():
                self.waypoints.add(wp_from)
                self.waypoints.update(wps_to)

        # Refine calibration target visibility map
        for cam, targets in self.camera_calibration_targets.items():
            for target_obj in targets:
                 if target_obj in self.objective_visibility:
                     # Ensure the visibility info for calibration targets is stored
                     self.calibration_target_visibility.setdefault(target_obj, set()).update(
                         self.objective_visibility[target_obj]
                     )

        # --- Precompute Shortest Paths ---
        self.shortest_paths = {} # rover -> from_wp -> to_wp -> dist
        for rover in self.rovers:
            # Build adjacency list for this rover based on known waypoints
            adj = {wp: set() for wp in self.waypoints}
            rover_graph_data = self.can_traverse.get(rover, {})
            for wp_from, reachable_wps in rover_graph_data.items():
                 if wp_from in adj: # Check waypoint exists
                     # Filter reachable_wps to only include known waypoints
                     valid_neighbors = {wp for wp in reachable_wps if wp in self.waypoints}
                     adj[wp_from].update(valid_neighbors)

            self.shortest_paths[rover] = {}
            for start_wp in self.waypoints:
                # Run BFS from start_wp
                distances = {wp: math.inf for wp in self.waypoints}
                if start_wp not in distances: continue # Should not happen
                distances[start_wp] = 0
                queue = deque([start_wp])

                while queue:
                    current_node = queue.popleft()
                    # Check if current_node is valid and has neighbors defined in adj
                    if current_node in adj:
                        for neighbor in adj[current_node]:
                            # Ensure neighbor is a valid waypoint before proceeding
                            if neighbor in distances and distances[neighbor] == math.inf:
                                distances[neighbor] = distances[current_node] + 1
                                queue.append(neighbor)

                # Store distances, converting inf to LARGE_DISTANCE
                self.shortest_paths[rover][start_wp] = {
                    wp: (d if d != math.inf else LARGE_DISTANCE)
                    for wp, d in distances.items()
                }

        # --- Precompute Lander Visibility ---
        self.waypoints_visible_from_lander = set()
        if self.lander_location:
            # Find waypoints X such that the lander at Y is visible FROM X.
            # Communicate action needs (visible ?x ?y) where rover is at ?x, lander at ?y.
            lander_wp = self.lander_location
            for wp_from, visible_wps in self.waypoint_visibility.items():
                 if lander_wp in visible_wps:
                     # Ensure wp_from is a known waypoint before adding
                     if wp_from in self.waypoints:
                         self.waypoints_visible_from_lander.add(wp_from)

        # --- Parse Goals ---
        self.goal_soil = set()
        self.goal_rock = set()
        self.goal_image = set()
        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == 'communicated_soil_data' and len(parts) == 2:
                wp = parts[1]
                if wp in self.waypoints: self.goal_soil.add(wp)
            elif pred == 'communicated_rock_data' and len(parts) == 2:
                wp = parts[1]
                if wp in self.waypoints: self.goal_rock.add(wp)
            elif pred == 'communicated_image_data' and len(parts) == 3:
                obj, mode = parts[1], parts[2]
                # Ensure objective and mode are known entities
                if obj in self.objectives and mode in self.modes:
                    self.goal_image.add((obj, mode))


    def _get_dist(self, rover, wp_from, wp_to):
        """Safely get shortest path distance, returning LARGE_DISTANCE if unreachable or keys invalid."""
        if rover not in self.shortest_paths or \
           wp_from not in self.shortest_paths[rover] or \
           wp_to not in self.shortest_paths[rover][wp_from]:
            return LARGE_DISTANCE
        dist = self.shortest_paths[rover][wp_from][wp_to]
        # Handles both math.inf during BFS and explicit LARGE_DISTANCE storage
        return LARGE_DISTANCE if dist >= LARGE_DISTANCE else dist


    def __call__(self, node):
        state = node.state
        heuristic_value = 0

        # --- Parse Current State ---
        rover_locations = {}
        rover_store_state = {} # rover -> 'empty' or 'full'
        calibrated_cameras = set() # (camera, rover)
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()
        at_soil_sample = set() # waypoint
        at_rock_sample = set() # waypoint

        # Initialize store state (assume empty unless 'full' is stated)
        for rover in self.rovers:
            if rover in self.rover_stores_map:
                 rover_store_state[rover] = 'empty'

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            # Use match for flexibility if needed, but direct indexing is faster if format is fixed
            if pred == 'at' and len(parts) == 3 and parts[1] in self.rovers:
                rover_locations[parts[1]] = parts[2]
            elif pred == 'full' and len(parts) == 2:
                store = parts[1]
                if store in self.store_rover_map:
                    rover_store_state[self.store_rover_map[store]] = 'full'
            # Note: 'empty' state is inferred if not 'full'
            elif pred == 'calibrated' and len(parts) == 3:
                if parts[1] in self.cameras and parts[2] in self.rovers:
                    calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'have_soil_analysis' and len(parts) == 3:
                 if parts[1] in self.rovers and parts[2] in self.waypoints:
                    have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis' and len(parts) == 3:
                 if parts[1] in self.rovers and parts[2] in self.waypoints:
                    have_rock.add((parts[1], parts[2]))
            elif pred == 'have_image' and len(parts) == 4:
                 if parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                    have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'communicated_soil_data' and len(parts) == 2:
                if parts[1] in self.waypoints: communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data' and len(parts) == 2:
                 if parts[1] in self.waypoints: communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data' and len(parts) == 3:
                 if parts[1] in self.objectives and parts[2] in self.modes:
                    communicated_image.add((parts[1], parts[2]))
            elif pred == 'at_soil_sample' and len(parts) == 2:
                 if parts[1] in self.waypoints: at_soil_sample.add(parts[1])
            elif pred == 'at_rock_sample' and len(parts) == 2:
                 if parts[1] in self.waypoints: at_rock_sample.add(parts[1])

        # --- Calculate Cost for Unsatisfied Goals ---
        unsatisfied_soil = self.goal_soil - communicated_soil
        unsatisfied_rock = self.goal_rock - communicated_rock
        unsatisfied_image = self.goal_image - communicated_image

        total_unsatisfied_goals = len(unsatisfied_soil) + len(unsatisfied_rock) + len(unsatisfied_image)

        if total_unsatisfied_goals == 0:
            return 0 # Goal state reached

        # --- Soil Goals ---
        for wp_goal in unsatisfied_soil:
            min_cost = LARGE_DISTANCE
            capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]

            for rover in capable_rovers:
                current_rover_loc = rover_locations.get(rover)
                if not current_rover_loc: continue # Rover not placed or location unknown

                cost = LARGE_DISTANCE # Default to impossible for this rover
                if (rover, wp_goal) in have_soil:
                    # Already have analysis, just need to communicate
                    comm_dist = min((self._get_dist(rover, current_rover_loc, comm_wp)
                                     for comm_wp in self.waypoints_visible_from_lander), default=LARGE_DISTANCE)
                    if comm_dist < LARGE_DISTANCE:
                        cost = comm_dist + 1 # navigate + communicate
                else:
                    # Need to sample first
                    if wp_goal in at_soil_sample: # Check if sample is available at the location
                        is_full = rover_store_state.get(rover) == 'full'
                        cost_drop = 1 if is_full else 0

                        dist_to_sample = self._get_dist(rover, current_rover_loc, wp_goal)
                        comm_dist_from_sample = min((self._get_dist(rover, wp_goal, comm_wp)
                                                     for comm_wp in self.waypoints_visible_from_lander), default=LARGE_DISTANCE)

                        # Check if all required navigation is possible
                        if dist_to_sample < LARGE_DISTANCE and comm_dist_from_sample < LARGE_DISTANCE:
                            cost = cost_drop + dist_to_sample + 1 + comm_dist_from_sample + 1
                            # drop? + navigate + sample + navigate + communicate

                min_cost = min(min_cost, cost)

            heuristic_value += min_cost

        # --- Rock Goals ---
        for wp_goal in unsatisfied_rock:
            min_cost = LARGE_DISTANCE
            capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]

            for rover in capable_rovers:
                current_rover_loc = rover_locations.get(rover)
                if not current_rover_loc: continue

                cost = LARGE_DISTANCE
                if (rover, wp_goal) in have_rock:
                    # Already have analysis
                    comm_dist = min((self._get_dist(rover, current_rover_loc, comm_wp)
                                     for comm_wp in self.waypoints_visible_from_lander), default=LARGE_DISTANCE)
                    if comm_dist < LARGE_DISTANCE:
                        cost = comm_dist + 1 # navigate + communicate
                else:
                    # Need to sample
                    if wp_goal in at_rock_sample: # Check sample availability
                        is_full = rover_store_state.get(rover) == 'full'
                        cost_drop = 1 if is_full else 0

                        dist_to_sample = self._get_dist(rover, current_rover_loc, wp_goal)
                        comm_dist_from_sample = min((self._get_dist(rover, wp_goal, comm_wp)
                                                     for comm_wp in self.waypoints_visible_from_lander), default=LARGE_DISTANCE)

                        if dist_to_sample < LARGE_DISTANCE and comm_dist_from_sample < LARGE_DISTANCE:
                            cost = cost_drop + dist_to_sample + 1 + comm_dist_from_sample + 1
                            # drop? + navigate + sample + navigate + communicate

                min_cost = min(min_cost, cost)

            heuristic_value += min_cost

        # --- Image Goals ---
        for obj_goal, mode_goal in unsatisfied_image:
            min_cost = LARGE_DISTANCE
            capable_rovers = [r for r, caps in self.rover_capabilities.items() if 'imaging' in caps]

            for rover in capable_rovers:
                current_rover_loc = rover_locations.get(rover)
                if not current_rover_loc: continue

                suitable_cameras = [
                    cam for cam in self.rover_cameras.get(rover, set())
                    if mode_goal in self.camera_supports.get(cam, set())
                ]

                for camera in suitable_cameras:
                    cost = LARGE_DISTANCE # Default cost for this rover/camera combo
                    if (rover, obj_goal, mode_goal) in have_image:
                        # Already have image
                        comm_dist = min((self._get_dist(rover, current_rover_loc, comm_wp)
                                         for comm_wp in self.waypoints_visible_from_lander), default=LARGE_DISTANCE)
                        if comm_dist < LARGE_DISTANCE:
                           cost = comm_dist + 1 # navigate + communicate
                    else:
                        # Need to potentially calibrate, take image, and communicate
                        cost_calibration_steps = 0
                        loc_after_calibration = current_rover_loc
                        calibration_possible = True

                        if (camera, rover) not in calibrated_cameras:
                            # Find best place to calibrate
                            possible_cal_targets = self.camera_calibration_targets.get(camera, set())
                            min_dist_to_cal = LARGE_DISTANCE
                            best_cal_wp = None

                            if not possible_cal_targets:
                                calibration_possible = False
                            else:
                                for target_obj in possible_cal_targets:
                                    # Check if this target is visible from anywhere suitable
                                    visible_wps = self.calibration_target_visibility.get(target_obj, set())
                                    if not visible_wps: continue

                                    for cal_wp in visible_wps:
                                        dist = self._get_dist(rover, current_rover_loc, cal_wp)
                                        if dist < min_dist_to_cal:
                                            min_dist_to_cal = dist
                                            best_cal_wp = cal_wp

                                if best_cal_wp is None or min_dist_to_cal >= LARGE_DISTANCE:
                                    calibration_possible = False
                                else:
                                    cost_calibration_steps = min_dist_to_cal + 1 # navigate + calibrate
                                    loc_after_calibration = best_cal_wp

                        if not calibration_possible:
                             # Cannot achieve goal if calibration is needed but impossible
                             cost = LARGE_DISTANCE
                        else:
                            # Find best place to take image
                            min_dist_to_img = LARGE_DISTANCE
                            best_img_wp = None
                            visible_wps_for_img = self.objective_visibility.get(obj_goal, set())

                            if not visible_wps_for_img: # Objective not visible from anywhere
                                cost = LARGE_DISTANCE
                            else:
                                for img_wp in visible_wps_for_img:
                                    dist = self._get_dist(rover, loc_after_calibration, img_wp)
                                    if dist < min_dist_to_img:
                                        min_dist_to_img = dist
                                        best_img_wp = img_wp

                                if best_img_wp is None or min_dist_to_img >= LARGE_DISTANCE: # Cannot reach imaging spot
                                    cost = LARGE_DISTANCE
                                else:
                                    # Calculate cost for imaging steps
                                    cost_imaging_steps = min_dist_to_img + 1 # navigate + take_image
                                    loc_after_image = best_img_wp

                                    # Find cost to communicate
                                    comm_dist_from_image = min((self._get_dist(rover, loc_after_image, comm_wp)
                                                                for comm_wp in self.waypoints_visible_from_lander), default=LARGE_DISTANCE)

                                    if comm_dist_from_image >= LARGE_DISTANCE: # Cannot reach comm spot
                                        cost = LARGE_DISTANCE
                                    else:
                                        cost_communication_steps = comm_dist_from_image + 1 # navigate + communicate
                                        # Total cost is sum of steps
                                        cost = cost_calibration_steps + cost_imaging_steps + cost_communication_steps

                    min_cost = min(min_cost, cost) # Update min cost for this image goal across cameras/rovers

            heuristic_value += min_cost


        # Final check: if heuristic is 0, ensure goals are met.
        if heuristic_value == 0 and total_unsatisfied_goals > 0:
            # This case should ideally not happen if LARGE_DISTANCE is handled correctly.
            # It implies all remaining goals have a calculated cost of 0, which is wrong.
            # Return 1 to ensure it's not treated as a goal state.
            return 1
        elif heuristic_value >= LARGE_DISTANCE:
             # If any goal was impossible, the sum will be large. Cap reasonably.
             # Using LARGE_DISTANCE itself signals potential unsolvability.
             return LARGE_DISTANCE
        else:
             # Return the calculated sum of minimum costs for each goal.
             return heuristic_value
