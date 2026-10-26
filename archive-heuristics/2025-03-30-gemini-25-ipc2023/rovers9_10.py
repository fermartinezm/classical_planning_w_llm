import itertools
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
# Make sure the Heuristic base class is available.
# If not, define a placeholder:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError

# Helper function to parse PDDL facts "(pred obj1 obj2 ...)" into parts ['pred', 'obj1', 'obj2', ...]
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    return fact[1:-1].split()

# Helper function to match facts with wildcards
def match(fact, *args):
    """Checks if a fact matches a pattern with optional wildcards."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals
    by summing the estimated costs for each individual unmet goal. The cost for
    each goal is estimated by finding the "cheapest" sequence of actions
    (navigation, sampling, calibration, imaging, communication) for any capable
    rover to achieve that goal, considering the current state and precomputed
    shortest path distances between waypoints. It prioritizes using rovers that
    already possess required items (analysis, images) or state (calibrated).

    # Assumptions
    - Action costs are uniform (1).
    - The heuristic relaxes the problem by assuming goals can be achieved
      independently. This might overestimate costs if actions can be shared
      (e.g., one trip serves multiple goals) or underestimate if resource
      contention (e.g., needing an empty store) across different goal-achieving
      sequences becomes a bottleneck.
    - It assumes that if a sample (soil/rock) is required for a communication goal
      and is not present at the target waypoint in the current state, a rover must
      already possess the corresponding analysis to achieve the communication goal.
      If neither is true for any rover, the goal is considered potentially very
      hard or impossible to achieve from the current state via sampling.
    - Shortest paths between waypoints are precomputed using BFS based on static
      traversability (`can_traverse`) and visibility (`visible`) information.

    # Heuristic Initialization
    - Stores the goal predicates provided by the task.
    - Parses static facts from the task to build internal knowledge structures:
        - Identifies all objects (rovers, waypoints, cameras, objectives, modes, stores, lander).
        - Maps rovers to their capabilities (equipment for soil, rock, imaging).
        - Maps rovers to their stores and vice-versa.
        - Maps rovers to the cameras they have on board.
        - Maps cameras to the modes they support and their calibration targets.
        - Maps objectives to the waypoints they are visible from.
        - Determines the lander's location.
        - Identifies waypoints visible from the lander's location (communication points).
        - Builds a waypoint connectivity graph for each rover based on `can_traverse` and `visible` predicates.
        - Computes all-pairs shortest paths (APSP) distances for each rover using BFS on its connectivity graph. Stores these distances.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total heuristic cost `h = 0`.
    2.  Parse the current `state` to get dynamic information: rover locations, store status (empty/full), existing soil/rock analyses held by rovers, existing images held by rovers, camera calibration status, and current locations of available soil/rock samples.
    3.  For each goal predicate `g` specified in the task's goals:
        a.  Check if `g` is already satisfied in the current `state`. If yes, cost for this goal is 0, continue to the next goal.
        b.  If `g` is not satisfied, calculate the minimum estimated cost `min_cost_g` to achieve it:
            i.  Identify the type of goal (`communicated_soil_data`, `communicated_rock_data`, `communicated_image_data`).
            ii. Determine the necessary prerequisite fact for the final communication action (e.g., `(have_soil_analysis r w)`, `(have_image r o m)`).
            iii.Find all rovers `r` that *could* potentially achieve this goal based on their static capabilities (equipment, cameras).
            iv. For each capable rover `r`:
                - Estimate the cost `cost_r` for *this rover* to achieve the goal `g` starting from the current `state`. This involves simulating the necessary action sequence:
                    - Get rover's current location `r_loc`.
                    - **Check Prerequisite:** See if the rover already has the prerequisite fact (e.g., `have_analysis`, `have_image`).
                    - **If Prerequisite Met:** Calculate cost to move from `r_loc` to the nearest communication waypoint (`comm_wp`) + 1 (for the communicate action).
                    - **If Prerequisite Not Met:**
                        - **For Analysis Goal (Soil/Rock):**
                            - Check if the required sample `(at_..._sample wp)` exists at the target waypoint `wp` in the current state.
                            - If sample exists: Calculate cost = `dist(r_loc, wp)` + cost(drop if store full) + 1 (sample action) + `dist(wp, nearest_comm_wp)` + 1 (communicate action).
                            - If sample does not exist: This specific path (sampling by this rover) is not possible. The rover *must* already have the analysis (handled above).
                        - **For Image Goal:**
                            - Check if the required camera `cam` is calibrated `(calibrated cam r)`.
                            - Find necessary intermediate waypoints: `image_wp` (where objective `o` is visible) and potentially `cal_wp` (where calibration target `t` is visible).
                            - If Calibrated: Cost = `dist(r_loc, nearest_image_wp)` + 1 (take_image) + `dist(image_wp, nearest_comm_wp)` + 1 (communicate).
                            - If Not Calibrated: Cost = `dist(r_loc, nearest_cal_wp)` + 1 (calibrate) + `dist(cal_wp, nearest_image_wp)` + 1 (take_image) + `dist(image_wp, nearest_comm_wp)` + 1 (communicate).
                    - Use precomputed shortest path distances (`self.distances`) for all navigation estimates. Find the minimum cost paths by considering the closest valid intermediate waypoints.
                - Keep track of the minimum `cost_r` found across all capable rovers for this goal `g`.
            v. Set `min_cost_g` to this minimum cost. If no rover can achieve the goal (e.g., required sample missing and no rover has analysis, or necessary locations are unreachable), assign a large cost (effectively infinity) to avoid choosing states leading to dead ends.
        c.  Add `min_cost_g` to the total heuristic cost `h`.
    4.  Return `h`. If `h` is 0, it implies all goals are met. A final check ensures 0 is returned *only* if the state truly satisfies all goals.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self._inf = float('inf')

        # --- Preprocessing Static Information ---
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander = None
        self.lander_wp = None

        self.rover_equipment = {} # rover -> set('soil', 'rock', 'imaging')
        self.rover_store = {} # rover -> store
        self.store_rover = {} # store -> rover
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_supports = {} # camera -> set(mode)
        self.camera_calibration_targets = {} # camera -> set(objective)
        self.objective_visibility = {} # objective -> set(waypoint)
        self.calibration_target_visibility = {} # objective -> set(waypoint)

        # Waypoint graph connectivity: rover -> wp_from -> set(wp_to)
        self.connectivity = {}
        # All-pairs shortest paths: rover -> wp_from -> wp_to -> distance
        self.distances = {}
        # Waypoints visible from lander
        self.comm_waypoints = set()

        # Parse static facts to populate structures
        all_objects = set()
        obj_types = {} # obj -> type

        # First pass: identify objects and types roughly
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            # Basic type inference from predicates
            if pred == 'rover': self.rovers.add(args[0]); obj_types[args[0]] = 'rover'
            elif pred == 'waypoint': self.waypoints.add(args[0]); obj_types[args[0]] = 'waypoint'
            elif pred == 'camera': self.cameras.add(args[0]); obj_types[args[0]] = 'camera'
            elif pred == 'objective': self.objectives.add(args[0]); obj_types[args[0]] = 'objective'
            elif pred == 'mode': self.modes.add(args[0]); obj_types[args[0]] = 'mode'
            elif pred == 'store': self.stores.add(args[0]); obj_types[args[0]] = 'store'
            elif pred == 'lander': self.lander = args[0]; obj_types[args[0]] = 'lander'

        # Second pass: extract relationships
        can_traverse_facts = {} # rover -> set((wp_from, wp_to))
        visible_facts = {} # wp_from -> set(wp_to)

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            if pred == 'equipped_for_soil_analysis':
                self.rover_equipment.setdefault(args[0], set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                self.rover_equipment.setdefault(args[0], set()).add('rock')
            elif pred == 'equipped_for_imaging':
                self.rover_equipment.setdefault(args[0], set()).add('imaging')
            elif pred == 'store_of':
                self.rover_store[args[1]] = args[0]
                self.store_rover[args[0]] = args[1]
            elif pred == 'on_board':
                self.rover_cameras.setdefault(args[1], set()).add(args[0])
            elif pred == 'supports':
                self.camera_supports.setdefault(args[0], set()).add(args[1])
            elif pred == 'calibration_target':
                self.camera_calibration_targets.setdefault(args[0], set()).add(args[1])
            elif pred == 'visible_from':
                self.objective_visibility.setdefault(args[0], set()).add(args[1])
                # Objectives used as calibration targets share visibility
                self.calibration_target_visibility.setdefault(args[0], set()).add(args[1])
            elif pred == 'at_lander':
                self.lander = args[0]
                self.lander_wp = args[1]
            elif pred == 'can_traverse':
                 r, w1, w2 = args
                 can_traverse_facts.setdefault(r, set()).add((w1, w2))
            elif pred == 'visible':
                 w1, w2 = args
                 visible_facts.setdefault(w1, set()).add(w2)

        # Determine communication waypoints (waypoints X visible from lander waypoint Y)
        # Need X such that (visible X Y) holds.
        if self.lander_wp:
             for x, visible_set in visible_facts.items():
                 if self.lander_wp in visible_set:
                     self.comm_waypoints.add(x)

        # --- Build Connectivity Graph and Compute Distances ---
        # Initialize connectivity and distances for all known rovers and waypoints
        for r in self.rovers:
            self.connectivity[r] = {wp: set() for wp in self.waypoints}
            self.distances[r] = {wp: {wp2: self._inf for wp2 in self.waypoints} for wp in self.waypoints}
            for wp in self.waypoints:
                self.distances[r][wp][wp] = 0 # Distance to self is 0

        # Build graph edges based on can_traverse AND visible
        for r in self.rovers:
            if r not in can_traverse_facts: continue
            for w1, w2 in can_traverse_facts[r]:
                # Navigate action requires (can_traverse r w1 w2) AND (visible w1 w2)
                if w1 in visible_facts and w2 in visible_facts[w1]:
                    self.connectivity[r][w1].add(w2)

            # Compute APSP using BFS from each waypoint for this rover
            for start_node in self.waypoints:
                if start_node not in self.distances[r]: continue # Skip if waypoint somehow missed
                queue = deque([(start_node, 0)])
                # Use a dictionary for visited distances to handle graph traversal correctly
                visited_dist = {start_node: 0}

                while queue:
                    current_node, dist = queue.popleft()
                    # Update the main distance matrix
                    self.distances[r][start_node][current_node] = dist

                    if current_node not in self.connectivity[r]: continue # Node might have no outgoing edges

                    for neighbor in self.connectivity[r][current_node]:
                        # If neighbor not visited or found a shorter path (shouldn't happen with BFS)
                        if neighbor not in visited_dist:
                            visited_dist[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))


    def _get_dist(self, rover, wp_from, wp_to):
        """Safely retrieves the precomputed distance."""
        if rover not in self.distances or wp_from not in self.distances[rover] or wp_to not in self.distances[rover][wp_from]:
            return self._inf
        dist = self.distances[rover][wp_from].get(wp_to, self._inf)
        return dist

    def _min_dist_to_set(self, rover, wp_from, target_wp_set):
        """Finds the minimum distance from wp_from to any waypoint in target_wp_set."""
        min_d = self._inf
        if not target_wp_set:
             return self._inf # Cannot reach an empty set of targets
        for target_wp in target_wp_set:
            d = self._get_dist(rover, wp_from, target_wp)
            min_d = min(min_d, d)
        return min_d

    def __call__(self, node):
        state = node.state
        h_cost = 0

        # --- Parse Current State ---
        rover_loc = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil = {} # rover -> set(waypoint)
        have_rock = {} # rover -> set(waypoint)
        have_image = {} # rover -> set((objective, mode))
        calibrated_cameras = {} # rover -> set(camera)
        current_soil_samples = set() # waypoints with soil samples now
        current_rock_samples = set() # waypoints with rock samples now

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            pred = parts[0]
            args = parts[1:]

            if pred == 'at' and args[0] in self.rovers: rover_loc[args[0]] = args[1]
            elif pred == 'empty' and args[0] in self.stores: store_status[args[0]] = 'empty'
            elif pred == 'full' and args[0] in self.stores: store_status[args[0]] = 'full'
            elif pred == 'have_soil_analysis': have_soil.setdefault(args[0], set()).add(args[1])
            elif pred == 'have_rock_analysis': have_rock.setdefault(args[0], set()).add(args[1])
            elif pred == 'have_image': have_image.setdefault(args[0], set()).add((args[1], args[2]))
            elif pred == 'calibrated': calibrated_cameras.setdefault(args[1], set()).add(args[0])
            elif pred == 'at_soil_sample': current_soil_samples.add(args[0])
            elif pred == 'at_rock_sample': current_rock_samples.add(args[0])

        # --- Calculate Cost for Each Unmet Goal ---
        num_unmet_goals = 0
        for goal_fact in self.goals:
            if goal_fact in state:
                continue
            num_unmet_goals += 1

            parts = get_parts(goal_fact)
            pred = parts[0]
            args = parts[1:]
            min_goal_cost = self._inf

            # --- Communicated Soil Data Goal ---
            if pred == 'communicated_soil_data':
                wp_goal = args[0]
                # Find rovers equipped for soil analysis
                possible_rovers = [r for r, equip in self.rover_equipment.items() if 'soil' in equip]

                for r in possible_rovers:
                    if r not in rover_loc: continue # Rover not present in this state/problem
                    r_loc_current = rover_loc[r]
                    cost_r = 0

                    # Option 1: Rover already has the analysis
                    if r in have_soil and wp_goal in have_soil[r]:
                        dist_to_comm = self._min_dist_to_set(r, r_loc_current, self.comm_waypoints)
                        if dist_to_comm != self._inf:
                            cost_r = dist_to_comm + 1 # navigate + communicate
                            min_goal_cost = min(min_goal_cost, cost_r)
                        # Continue to check sampling option if analysis exists, maybe sampling is cheaper? No, analysis exists is best.
                        # If analysis exists, this is the cheapest way for this rover.
                        # min_goal_cost = min(min_goal_cost, cost_r if cost_r != self._inf else self._inf) # Simplified below

                    # Option 2: Rover needs to sample
                    elif wp_goal in current_soil_samples: # Check if sample is available
                        r_store = self.rover_store.get(r)
                        store_is_full = r_store and store_status.get(r_store) == 'full'

                        dist_to_sample = self._get_dist(r, r_loc_current, wp_goal)
                        if dist_to_sample == self._inf: continue # Cannot reach sample location

                        # Calculate cost to sample
                        cost_to_sample = dist_to_sample
                        cost_to_sample += 1 if store_is_full else 0 # drop action cost
                        cost_to_sample += 1 # sample_soil action

                        # Calculate cost from sample location to communicate
                        dist_sample_to_comm = self._min_dist_to_set(r, wp_goal, self.comm_waypoints)
                        if dist_sample_to_comm == self._inf: continue # Cannot reach comm location from sample wp

                        cost_r = cost_to_sample + dist_sample_to_comm + 1 # communicate_soil_data action
                        min_goal_cost = min(min_goal_cost, cost_r)

                    # If neither analysis exists nor sample is available, this rover cannot achieve goal this way.

            # --- Communicated Rock Data Goal ---
            elif pred == 'communicated_rock_data':
                wp_goal = args[0]
                possible_rovers = [r for r, equip in self.rover_equipment.items() if 'rock' in equip]

                for r in possible_rovers:
                    if r not in rover_loc: continue
                    r_loc_current = rover_loc[r]
                    cost_r = 0

                    # Option 1: Rover already has the analysis
                    if r in have_rock and wp_goal in have_rock[r]:
                        dist_to_comm = self._min_dist_to_set(r, r_loc_current, self.comm_waypoints)
                        if dist_to_comm != self._inf:
                            cost_r = dist_to_comm + 1
                            min_goal_cost = min(min_goal_cost, cost_r)

                    # Option 2: Rover needs to sample
                    elif wp_goal in current_rock_samples:
                        r_store = self.rover_store.get(r)
                        store_is_full = r_store and store_status.get(r_store) == 'full'

                        dist_to_sample = self._get_dist(r, r_loc_current, wp_goal)
                        if dist_to_sample == self._inf: continue

                        cost_to_sample = dist_to_sample + (1 if store_is_full else 0) + 1 # navigate + drop? + sample

                        dist_sample_to_comm = self._min_dist_to_set(r, wp_goal, self.comm_waypoints)
                        if dist_sample_to_comm == self._inf: continue

                        cost_r = cost_to_sample + dist_sample_to_comm + 1 # communicate
                        min_goal_cost = min(min_goal_cost, cost_r)

            # --- Communicated Image Data Goal ---
            elif pred == 'communicated_image_data':
                obj_goal, mode_goal = args[0], args[1]
                # Find rovers equipped for imaging, with a camera supporting the mode
                possible_rover_cam_pairs = []
                for r, equip in self.rover_equipment.items():
                    if 'imaging' in equip and r in self.rover_cameras:
                        for cam in self.rover_cameras[r]:
                            if cam in self.camera_supports and mode_goal in self.camera_supports[cam]:
                                possible_rover_cam_pairs.append((r, cam))

                for r, cam in possible_rover_cam_pairs:
                    if r not in rover_loc: continue
                    r_loc_current = rover_loc[r]
                    cost_r_cam = 0

                    # Option 1: Rover already has the image
                    if r in have_image and (obj_goal, mode_goal) in have_image[r]:
                        dist_to_comm = self._min_dist_to_set(r, r_loc_current, self.comm_waypoints)
                        if dist_to_comm != self._inf:
                            cost_r_cam = dist_to_comm + 1 # navigate + communicate
                            min_goal_cost = min(min_goal_cost, cost_r_cam)

                    # Option 2: Rover needs to take the image
                    else:
                        image_waypoints = self.objective_visibility.get(obj_goal, set())
                        if not image_waypoints: continue # Objective not visible from anywhere

                        is_calibrated = r in calibrated_cameras and cam in calibrated_cameras[r]

                        # Sub-option 2a: Camera is already calibrated
                        if is_calibrated:
                            # Find closest imaging waypoint from current location
                            min_dist_to_image_wp = self._inf
                            best_image_wp = None
                            for img_wp in image_waypoints:
                                d = self._get_dist(r, r_loc_current, img_wp)
                                if d < min_dist_to_image_wp:
                                    min_dist_to_image_wp = d
                                    best_image_wp = img_wp

                            if best_image_wp is None or min_dist_to_image_wp == self._inf: continue # Cannot reach any imaging waypoint

                            cost_to_take_image = min_dist_to_image_wp + 1 # navigate + take_image

                            # Cost from image waypoint to comm waypoint
                            dist_image_to_comm = self._min_dist_to_set(r, best_image_wp, self.comm_waypoints)
                            if dist_image_to_comm == self._inf: continue # Cannot reach comm from image wp

                            cost_r_cam = cost_to_take_image + dist_image_to_comm + 1 # communicate
                            min_goal_cost = min(min_goal_cost, cost_r_cam)

                        # Sub-option 2b: Camera needs calibration
                        else:
                            cal_targets = self.camera_calibration_targets.get(cam)
                            if not cal_targets: continue # Camera has no calibration target

                            # Find closest calibration waypoint reachable from current location
                            min_dist_to_cal_wp = self._inf
                            best_cal_wp = None
                            for target in cal_targets:
                                cal_waypoints = self.calibration_target_visibility.get(target, set())
                                for cal_wp in cal_waypoints:
                                    d = self._get_dist(r, r_loc_current, cal_wp)
                                    if d < min_dist_to_cal_wp:
                                        min_dist_to_cal_wp = d
                                        best_cal_wp = cal_wp

                            if best_cal_wp is None or min_dist_to_cal_wp == self._inf: continue # Cannot reach any calibration waypoint

                            cost_to_calibrate = min_dist_to_cal_wp + 1 # navigate + calibrate

                            # Find closest image waypoint reachable from calibration waypoint
                            min_dist_cal_to_image_wp = self._inf
                            best_image_wp_after_cal = None
                            for img_wp in image_waypoints:
                                d = self._get_dist(r, best_cal_wp, img_wp)
                                if d < min_dist_cal_to_image_wp:
                                    min_dist_cal_to_image_wp = d
                                    best_image_wp_after_cal = img_wp

                            if best_image_wp_after_cal is None or min_dist_cal_to_image_wp == self._inf: continue # Cannot reach image wp from cal wp

                            cost_to_take_image = min_dist_cal_to_image_wp + 1 # navigate + take_image

                            # Cost from image waypoint to comm waypoint
                            dist_image_to_comm = self._min_dist_to_set(r, best_image_wp_after_cal, self.comm_waypoints)
                            if dist_image_to_comm == self._inf: continue # Cannot reach comm from image wp

                            cost_r_cam = cost_to_calibrate + cost_to_take_image + dist_image_to_comm + 1 # communicate
                            min_goal_cost = min(min_goal_cost, cost_r_cam)

            # Add cost for this goal
            if min_goal_cost != self._inf:
                h_cost += min_goal_cost
            else:
                # If a goal seems unachievable by any rover from this state, assign a large penalty.
                # This helps guide the search away from states where goals become impossible.
                h_cost += 1000 # Arbitrarily large penalty

        # Final check: Ensure heuristic is 0 iff state is a goal state.
        # If h_cost is 0 but not all goals are met, return 1 (minimum possible cost).
        if h_cost == 0 and num_unmet_goals > 0:
            return 1

        return h_cost

