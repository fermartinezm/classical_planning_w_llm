import math
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts from a PDDL fact string like "(pred obj1 obj2)" -> ["pred", "obj1", "obj2"]
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    return fact[1:-1].split()

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    Estimates the cost to reach the goal state by summing the estimated costs
    for achieving each individual unsatisfied goal predicate independently.
    The cost for each goal involves estimating the necessary sequence of actions
    (navigation, sampling/imaging, communication) and associated navigation distances.
    This heuristic is designed for Greedy Best-First Search and does not need to be admissible.

    # Assumptions
    - Navigation cost between waypoints is estimated using the shortest path
      distance in the visibility graph. The `can_traverse` predicate specific
      to rovers is ignored for distance calculation to keep it simple and efficient,
      using only the `(visible ?x ?y)` facts. BFS is used for shortest paths.
    - The heuristic estimates the cost for a goal by assuming a sequence like:
      navigate -> perform task (sample/calibrate/image) -> navigate -> communicate.
    - For imaging goals requiring calibration:
      navigate -> calibrate -> navigate -> take_image -> navigate -> communicate.
    - When multiple rovers, cameras, or locations (for imaging, calibration, communication)
      are available, the heuristic calculates the cost for each possibility for that
      single goal and chooses the minimum estimated cost. It ignores potential
      resource conflicts (e.g., store capacity, camera calibration state shared across goals)
      or synergies between goals.
    - The cost of dropping a sample (1 action) is included if a rover needs to sample
      but its store is currently full.
    - The "closest" waypoint for communication, imaging, or calibration is chosen based
      on the precomputed shortest path distances to minimize estimated navigation cost for that step.

    # Heuristic Initialization
    - Parses static facts provided in the task to build internal data structures:
        - Sets of rovers, waypoints, cameras, objectives, modes, stores.
        - Mappings for rover capabilities (soil, rock, imaging), store ownership, camera equipment.
        - Camera properties: supported modes, calibration targets.
        - Visibility information: which objectives/targets are visible from which waypoints.
        - Lander location.
        - A visibility graph representing connections between waypoints based on `(visible ?x ?y)`.
    - Precomputes all-pairs shortest path distances between waypoints using BFS on the visibility graph.
      Stores these distances for efficient lookup during heuristic calculation.
    - Identifies the set of waypoints suitable for communication (those visible from the lander's location).
    - Precomputes sets of waypoints suitable for imaging specific objectives and calibrating using specific targets.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Receive the current state (a set of PDDL fact strings).
    2.  Check if the current state satisfies all goal conditions. If yes, return 0.
    3.  Parse the current state to extract dynamic information: rover locations, store status (empty/full),
        camera calibration status, and which data (soil, rock, image) has already been acquired or communicated.
    4.  Initialize the total heuristic value `h = 0`.
    5.  Iterate through each goal predicate `g` defined in the task's goal conditions.
    6.  If goal `g` is already satisfied in the current state (i.e., the corresponding `communicated_...` fact exists), skip to the next goal.
    7.  If goal `g` is not satisfied, estimate the minimum cost to achieve it:
        a.  Initialize `min_goal_cost = infinity`.
        b.  **If `g` is `(communicated_soil_data w)`:**
            -   Iterate through all rovers `r` equipped for soil analysis.
            -   Determine the current location `loc_r` of rover `r`.
            -   Check if `r` already has the soil analysis for `w` (`(have_soil_analysis r w)`).
            -   *If `r` has the analysis:*
                -   Find the closest communication waypoint `comm_wp` reachable from `loc_r`.
                -   Estimate cost = `dist(loc_r, comm_wp) + 1` (navigate + communicate).
            -   *If `r` does not have the analysis:*
                -   Check if the rover's store `s` is full. Add `1` (drop cost) if full.
                -   Find cost to navigate from `loc_r` to `w`: `dist(loc_r, w)`.
                -   Find the closest communication waypoint `comm_wp` reachable from `w`.
                -   Estimate cost = (drop_cost) + `dist(loc_r, w)` + `1` (sample) + `dist(w, comm_wp)` + `1` (communicate).
            -   Update `min_goal_cost = min(min_goal_cost, estimated_cost_for_rover_r)`.
        c.  **If `g` is `(communicated_rock_data w)`:**
            -   Perform similar calculations as for soil data, using rock analysis capabilities and facts.
            -   Update `min_goal_cost`.
        d.  **If `g` is `(communicated_image_data o m)`:**
            -   Iterate through rovers `r` equipped for imaging and cameras `c` on `r` supporting mode `m`.
            -   Determine `loc_r`. Check if `r` already has the image (`(have_image r o m)`).
            -   *If `r` has the image:*
                -   Find closest `comm_wp` from `loc_r`.
                -   Estimate cost = `dist(loc_r, comm_wp) + 1`.
            -   *If `r` does not have the image:*
                -   Check if camera `c` is calibrated (`(calibrated c r)`).
                -   *If `c` is calibrated:*
                    -   Find the closest waypoint `image_wp` (visible from objective `o`) reachable from `loc_r`.
                    -   Find the closest `comm_wp` reachable from `image_wp`.
                    -   Estimate cost = `dist(loc_r, image_wp)` + `1` (take_image) + `dist(image_wp, comm_wp)` + `1` (communicate).
                -   *If `c` is not calibrated:*
                    -   Find the best path: `loc_r -> cal_wp -> image_wp` by checking possible calibration targets `t` for `c`, waypoints `cal_wp` visible from `t`, and waypoints `image_wp` visible from `o`. Choose the combination minimizing `dist(loc_r, cal_wp) + dist(cal_wp, image_wp)`.
                    -   Find the closest `comm_wp` reachable from the chosen `image_wp`.
                    -   Estimate cost = `dist(loc_r, cal_wp)` + `1` (calibrate) + `dist(cal_wp, image_wp)` + `1` (take_image) + `dist(image_wp, comm_wp)` + `1` (communicate).
            -   Update `min_goal_cost`.
        e.  If `min_goal_cost` is not infinity, add it to the total heuristic value `h`.
    8.  Return the total heuristic value `h` (as an integer). If `h` is infinity (meaning some goal seems unreachable by the heuristic's estimate), return a large finite number.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures from Static Facts ---
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander = None
        self.lander_location = None

        self.rover_capabilities = {} # rover -> set{'soil', 'rock', 'imaging'}
        self.rover_stores = {}       # rover -> store
        self.store_rover = {}        # store -> rover
        self.rover_cameras = {}      # rover -> set{camera}
        self.camera_rover = {}       # camera -> rover
        self.camera_supports = {}    # camera -> set{mode}
        self.camera_calibration_targets = {} # camera -> set{objective}
        self.objective_visible_from = {} # objective -> set{waypoint}
        self.calibration_target_cameras = {} # objective -> set{camera} # Inverse mapping

        self.visibility_graph = {} # waypoint -> set{waypoint}

        # --- Parse Static Facts ---
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            # Dynamically identify objects and their types/properties
            if pred == 'at_lander':
                self.lander = parts[1]
                self.lander_location = parts[2]
                self.waypoints.add(parts[2])
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
                self.rover_stores[rover] = store
                self.store_rover[store] = rover
            elif pred == 'visible':
                wp1, wp2 = parts[1], parts[2]
                self.waypoints.add(wp1); self.waypoints.add(wp2)
                self.visibility_graph.setdefault(wp1, set()).add(wp2)
            # 'can_traverse' is ignored for distance calculation simplification
            elif pred == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.cameras.add(camera); self.objectives.add(objective)
                self.camera_calibration_targets.setdefault(camera, set()).add(objective)
                self.calibration_target_cameras.setdefault(objective, set()).add(camera)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                self.cameras.add(camera); self.rovers.add(rover)
                self.rover_cameras.setdefault(rover, set()).add(camera)
                self.camera_rover[camera] = rover
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                self.cameras.add(camera); self.modes.add(mode)
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif pred == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objectives.add(objective); self.waypoints.add(waypoint)
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)

        # Ensure all waypoints are keys in the visibility graph
        for wp in self.waypoints:
            self.visibility_graph.setdefault(wp, set())

        # --- Precompute Shortest Paths using BFS ---
        self.distances = self._compute_all_pairs_shortest_paths()

        # --- Identify Communication Waypoints ---
        self.comm_waypoints = set()
        if self.lander_location:
             # Find all waypoints 'x' such that (visible x lander_loc) is true.
             # Assumes visibility is potentially asymmetric, check based on definition.
             # The communicate actions require rover at x, lander at y, visible(x, y).
             for wp, neighbors in self.visibility_graph.items():
                 if self.lander_location in neighbors:
                     self.comm_waypoints.add(wp)

        # --- Precompute useful sets for finding locations ---
        # Waypoints from which a calibration target objective is visible
        self.calibration_waypoints = {} # objective -> set{waypoint}
        for obj, wps in self.objective_visible_from.items():
             # Check if this objective is a calibration target for any camera
             if obj in self.calibration_target_cameras:
                 self.calibration_waypoints[obj] = wps

        # Waypoints from which an imaging objective is visible
        self.imaging_waypoints = self.objective_visible_from # objective -> set{waypoint}


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of waypoints using BFS."""
        distances = {wp: {other_wp: math.inf for other_wp in self.waypoints} for wp in self.waypoints}
        for start_node in self.waypoints:
            distances[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node: 0} # Store distance here too

            while queue:
                current_wp = queue.popleft()
                current_dist = visited[current_wp]

                # Assumes visibility graph represents possible navigation steps
                for neighbor in self.visibility_graph.get(current_wp, set()):
                    if neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
                    # Optimization: If we find a shorter path (shouldn't happen with BFS)
                    # elif current_dist + 1 < visited[neighbor]:
                    #    visited[neighbor] = current_dist + 1
                    #    distances[start_node][neighbor] = current_dist + 1
                    #    # Re-add to queue? Not needed for standard BFS.
        return distances

    def _get_shortest_dist(self, wp1, wp2):
        """Returns the precomputed shortest distance, or infinity if unreachable."""
        if wp1 is None or wp2 is None:
            return math.inf
        # Access precomputed distances
        return self.distances.get(wp1, {}).get(wp2, math.inf)

    def _find_closest_target_wp(self, source_wp, target_wps):
        """Finds the minimum distance and the corresponding waypoint from source_wp to any waypoint in target_wps."""
        min_dist = math.inf
        closest_wp = None
        if not target_wps or source_wp is None:
            return min_dist, closest_wp # Return infinity if no targets or source invalid

        for target_wp in target_wps:
            dist = self._get_shortest_dist(source_wp, target_wp)
            if dist < min_dist:
                min_dist = dist
                closest_wp = target_wp
        return min_dist, closest_wp


    def __call__(self, node):
        state = node.state
        # Goal check: If all goal predicates are in the current state, heuristic is 0.
        if self.goals <= state:
            return 0

        # --- Parse Current State ---
        rover_locations = {}    # rover -> waypoint
        store_status = {}       # store -> 'empty' / 'full'
        camera_calibrated = set() # set of '(calibrated c r)' facts
        have_soil = set()         # set of '(have_soil_analysis r w)' facts
        have_rock = set()         # set of '(have_rock_analysis r w)' facts
        have_image = set()        # set of '(have_image r o m)' facts
        communicated_goals = set() # set of achieved goal predicates in state

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at':
                rover_locations[parts[1]] = parts[2]
            elif pred == 'empty':
                store_status[parts[1]] = 'empty'
            elif pred == 'full':
                store_status[parts[1]] = 'full'
            elif pred == 'calibrated':
                camera_calibrated.add(fact)
            elif pred == 'have_soil_analysis':
                have_soil.add(fact)
            elif pred == 'have_rock_analysis':
                have_rock.add(fact)
            elif pred == 'have_image':
                have_image.add(fact)
            elif pred.startswith('communicated_'):
                 communicated_goals.add(fact) # Track achieved goals


        heuristic_value = 0
        # --- Calculate Cost for Each Unsatisfied Goal ---
        for goal in self.goals:
            if goal in communicated_goals:
                continue # Skip goals already achieved

            goal_parts = get_parts(goal)
            goal_pred = goal_parts[0]
            min_goal_cost = math.inf

            # --- Case 1: communicated_soil_data(w) ---
            if goal_pred == 'communicated_soil_data':
                waypoint_w = goal_parts[1]
                for rover in self.rovers:
                    # Check capability
                    if 'soil' not in self.rover_capabilities.get(rover, set()): continue
                    current_loc = rover_locations.get(rover)
                    if current_loc is None: continue # Skip if rover location unknown

                    cost = 0
                    have_soil_fact = f'(have_soil_analysis {rover} {waypoint_w})'

                    if have_soil_fact not in have_soil:
                        # Need to sample: Drop? -> Nav -> Sample -> Nav -> Comm
                        store = self.rover_stores.get(rover)
                        if store and store_status.get(store) == 'full':
                            cost += 1 # Drop cost

                        nav1_cost = self._get_shortest_dist(current_loc, waypoint_w)
                        if nav1_cost == math.inf: continue # Cannot reach sample location

                        cost += nav1_cost # Navigate to sample
                        cost += 1 # Sample cost

                        # Navigate from sample location to nearest communication point
                        nav2_cost, _ = self._find_closest_target_wp(waypoint_w, self.comm_waypoints)
                        if nav2_cost == math.inf: continue # Cannot reach comm point from sample loc

                        cost += nav2_cost # Navigate sample -> comm
                        cost += 1 # Communicate cost
                    else:
                        # Have sample: Nav -> Comm
                        nav_cost, _ = self._find_closest_target_wp(current_loc, self.comm_waypoints)
                        if nav_cost == math.inf: continue # Cannot reach comm point

                        cost += nav_cost # Navigate current -> comm
                        cost += 1 # Communicate cost

                    min_goal_cost = min(min_goal_cost, cost)

            # --- Case 2: communicated_rock_data(w) ---
            elif goal_pred == 'communicated_rock_data':
                waypoint_w = goal_parts[1]
                for rover in self.rovers:
                    # Check capability
                    if 'rock' not in self.rover_capabilities.get(rover, set()): continue
                    current_loc = rover_locations.get(rover)
                    if current_loc is None: continue

                    cost = 0
                    have_rock_fact = f'(have_rock_analysis {rover} {waypoint_w})'

                    if have_rock_fact not in have_rock:
                        # Need to sample: Drop? -> Nav -> Sample -> Nav -> Comm
                        store = self.rover_stores.get(rover)
                        if store and store_status.get(store) == 'full':
                            cost += 1 # Drop

                        nav1_cost = self._get_shortest_dist(current_loc, waypoint_w)
                        if nav1_cost == math.inf: continue
                        cost += nav1_cost # Navigate to sample

                        cost += 1 # Sample

                        nav2_cost, _ = self._find_closest_target_wp(waypoint_w, self.comm_waypoints)
                        if nav2_cost == math.inf: continue
                        cost += nav2_cost # Navigate sample -> comm

                        cost += 1 # Communicate
                    else:
                        # Have sample: Nav -> Comm
                        nav_cost, _ = self._find_closest_target_wp(current_loc, self.comm_waypoints)
                        if nav_cost == math.inf: continue
                        cost += nav_cost # Navigate current -> comm
                        cost += 1 # Communicate

                    min_goal_cost = min(min_goal_cost, cost)

            # --- Case 3: communicated_image_data(o, m) ---
            elif goal_pred == 'communicated_image_data':
                objective_o, mode_m = goal_parts[1], goal_parts[2]
                possible_image_wps = self.imaging_waypoints.get(objective_o, set())
                if not possible_image_wps: continue # Cannot image this objective from anywhere

                for rover in self.rovers:
                    # Check capability
                    if 'imaging' not in self.rover_capabilities.get(rover, set()): continue
                    current_loc = rover_locations.get(rover)
                    if current_loc is None: continue

                    for camera in self.rover_cameras.get(rover, set()):
                        # Check camera support for mode
                        if mode_m not in self.camera_supports.get(camera, set()): continue

                        cost = 0
                        have_image_fact = f'(have_image {rover} {objective_o} {mode_m})'

                        if have_image_fact not in have_image:
                            # Need to take image
                            calibrated_fact = f'(calibrated {camera} {rover})'
                            needs_calibration = calibrated_fact not in camera_calibrated

                            best_image_wp = None # The waypoint where image will be taken
                            nav_path_cost = math.inf # Cost of navigation before taking image

                            if needs_calibration:
                                # Need to calibrate: Nav -> Calibrate -> Nav -> Take Image
                                possible_cal_targets = self.camera_calibration_targets.get(camera, set())
                                if not possible_cal_targets: continue # Cannot calibrate this camera

                                current_min_cal_path = math.inf
                                best_cal_wp_for_path = None
                                best_img_wp_for_path = None

                                # Find best sequence: current -> cal_wp -> image_wp
                                for cal_target in possible_cal_targets:
                                     possible_cal_wps = self.calibration_waypoints.get(cal_target, set())
                                     for cal_wp in possible_cal_wps:
                                         dist_to_cal = self._get_shortest_dist(current_loc, cal_wp)
                                         if dist_to_cal == math.inf: continue

                                         # Find best image wp reachable from this cal_wp
                                         dist_cal_to_image, img_wp_from_cal = self._find_closest_target_wp(cal_wp, possible_image_wps)
                                         if dist_cal_to_image == math.inf: continue # Cannot reach any image wp from this cal_wp

                                         total_cal_path = dist_to_cal + dist_cal_to_image
                                         if total_cal_path < current_min_cal_path:
                                             current_min_cal_path = total_cal_path
                                             best_cal_wp_for_path = cal_wp
                                             best_img_wp_for_path = img_wp_from_cal

                                if best_cal_wp_for_path is None: continue # No calibration path found

                                nav_path_cost = current_min_cal_path
                                best_image_wp = best_img_wp_for_path
                                cost += nav_path_cost # Nav cost for current -> cal -> image
                                cost += 1 # Calibrate cost
                                cost += 1 # Take image cost

                            else:
                                # Calibrated: Nav -> Take Image
                                nav_to_image_cost, img_wp = self._find_closest_target_wp(current_loc, possible_image_wps)
                                if nav_to_image_cost == math.inf: continue # Cannot reach any image wp

                                best_image_wp = img_wp
                                cost += nav_to_image_cost # Navigate current -> image
                                cost += 1 # Take image cost

                            # Navigate from image location to communication point
                            nav_image_to_comm_cost, _ = self._find_closest_target_wp(best_image_wp, self.comm_waypoints)
                            if nav_image_to_comm_cost == math.inf: continue # Cannot reach comm point

                            cost += nav_image_to_comm_cost # Navigate image -> comm
                            cost += 1 # Communicate cost

                        else:
                            # Have image: Nav -> Comm
                            nav_cost, _ = self._find_closest_target_wp(current_loc, self.comm_waypoints)
                            if nav_cost == math.inf: continue # Cannot reach comm point

                            cost += nav_cost # Navigate current -> comm
                            cost += 1 # Communicate cost

                        min_goal_cost = min(min_goal_cost, cost)

            # --- Add cost for this goal ---
            # If goal seems impossible (min_goal_cost is inf), we add 0 to avoid infinite heuristic.
            # A large penalty could also be used, but sum=0 is simpler. GBFS should still explore other nodes.
            if min_goal_cost != math.inf:
                 heuristic_value += min_goal_cost

        # Final check: if heuristic is 0 but state is not goal, something is wrong.
        # This shouldn't happen due to the initial goal check.
        # If heuristic_value is inf (e.g., if all goals were inf), return large number.
        if heuristic_value == math.inf:
             return 1_000_000 # Return a large number for potentially unsolvable states

        # Return the final heuristic value as an integer
        return int(round(heuristic_value))
