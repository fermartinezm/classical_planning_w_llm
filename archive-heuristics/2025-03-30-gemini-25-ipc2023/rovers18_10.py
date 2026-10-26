import itertools
import heapq
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact (string) by removing parentheses
    and splitting by space.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    """
    # Handles potential extra whitespace
    return fact.strip()[1:-1].split()

def match(fact_parts, *args):
    """Check if a list of fact parts matches a given pattern of strings.
    Wildcards (*) can be used in the pattern.
    Example: match(["at", "rover1", "waypoint1"], "at", "*", "waypoint1") -> True
    """
    if len(fact_parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, args))

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'rovers'.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from the current state in the Rovers domain. It calculates the cost for each
    unsatisfied goal independently and sums them up. The cost for each goal considers
    the necessary actions (sampling, calibrating, taking images, communicating) and
    estimates the navigation cost based on precomputed shortest paths on the
    waypoint visibility graph. It tries to find the cheapest rover/camera assignment
    for each goal based on the current state.

    # Assumptions
    - Navigation cost between waypoints is approximated by the shortest path length
      in the static visibility graph `(visible ?w1 ?w2)`. This ignores rover-specific
      `can_traverse` predicates and assumes visibility implies reachability for
      cost estimation. The cost of a `navigate` action is 1 per step in the path.
    - The cost calculation for each goal is independent; potential positive or
      negative interactions between achieving different goals are ignored (e.g.,
      sharing a calibration action, or one action undoing a precondition for another).
    - Resource contention (multiple rovers needing the same waypoint or sample)
      is not explicitly modeled beyond finding the minimum cost assignment per goal.
    - If a rover's store is full when sampling is needed, a 'drop' action (cost 1)
      is assumed to occur at the rover's current location before navigating to sample.
    - If a goal seems impossible from the current state (e.g., required sample
      doesn't exist and no rover holds it, or necessary locations are unreachable),
      a minimal cost of 1 is added as an optimistic estimate to ensure the heuristic
      is non-zero for non-goal states where progress might still be possible towards
      other goals.

    # Heuristic Initialization
    - The constructor (`__init__`) precomputes static information from `task.static`:
        - Waypoints and the visibility graph `(visible ?w1 ?w2)`.
        - All-pairs shortest paths distances on the visibility graph using BFS.
        - Lander location `(at_lander ?l ?w)`.
        - Communication waypoints: Waypoints `w_r` such that `(visible w_r w_l)` holds,
          where `w_l` is the lander location.
        - Rover equipment `(equipped_for_...)`, store associations `(store_of ...)`.
        - Camera details: `(on_board ...)`, `(supports ...)`, `(calibration_target ...)`.
        - Visibility of objectives/targets: `(visible_from ?obj ?wp)`.
        - Parsed goal conditions `(communicated_...)`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  Check if the current state is a goal state. If yes, return 0.
    2.  Parse the current `state` to efficiently access dynamic facts like rover
        locations (`at`), store status (`empty`/`full`), camera calibration
        (`calibrated`), held samples (`have_soil_analysis`, `have_rock_analysis`),
        held images (`have_image`), and available samples (`at_soil_sample`,
        `at_rock_sample`).
    3.  Initialize total heuristic cost `h = 0`.
    4.  Keep track of processed goals to avoid double counting if a goal predicate
        appears multiple times in the goal list (unlikely but possible).
    5.  For each unsatisfied goal `g`:
        a.  **If `g` is `(communicated_soil_data w)`:**
            i.   Calculate `min_cost_g` to achieve this goal, initialized to infinity.
            ii.  *Option 1 (Already have sample):* If a rover `r` has
                 `(have_soil_analysis r w)`, estimate cost as `dist(at(r), w_comm) + 1`
                 (navigate to communication point + communicate). Update `min_cost_g`.
                 `w_comm` is the closest waypoint to `at(r)` from which the lander is visible.
            iii. *Option 2 (Need to sample):* If `(at_soil_sample w)` exists in the state,
                 for each rover `r` equipped for soil analysis:
                 - Estimate cost: `[cost_drop(1)] + dist(at(r), w) + sample(1) + dist(w, w_comm) + communicate(1)`.
                 - `cost_drop` is 1 if rover's store `s` has `(full s)` in state, 0 otherwise.
                 - `w_comm` is the closest waypoint to the sample location `w` from which the lander is visible.
                 - Update `min_cost_g`.
            iv.  Add `min_cost_g` to `h` if it's finite, otherwise add 1 (fallback cost).
        b.  **If `g` is `(communicated_rock_data w)`:** Similar logic as soil data, using rock analysis equipment and `at_rock_sample`.
        c.  **If `g` is `(communicated_image_data o m)`:**
            i.   Calculate `min_cost_g`, initialized to infinity.
            ii.  *Option 1 (Already have image):* If rover `r` has `(have_image r o m)`,
                 estimate cost `dist(at(r), w_comm) + 1`. Update `min_cost_g`.
            iii. *Option 2 (Need to take image):* For each rover `r` equipped for imaging and each camera `c` on `r` supporting mode `m`:
                 - Check if calibration is needed (`(calibrated c r)` is not in state).
                 - Find the calibration target `t` for camera `c`.
                 - Find the set of waypoints `W_cal` from which `t` is visible.
                 - Find the set of waypoints `W_img` from which objective `o` is visible.
                 - If calibration is needed: Find the closest `w_cal` in `W_cal` to `at(r)`. Calculate `nav_cal_cost = dist(at(r), w_cal)`. Set `cal_cost = 1`. The location after calibration is `w_cal`.
                 - If calibration is not needed: `nav_cal_cost = 0`, `cal_cost = 0`. Location after calibration step is `at(r)`.
                 - Find the closest `w_img` in `W_img` to the location after calibration. Calculate `nav_img_cost = dist(loc_after_cal, w_img)`. Set `take_cost = 1`.
                 - Find the closest communication waypoint `w_comm` to `w_img`. Calculate `nav_comm_cost = dist(w_img, w_comm)`. Set `comm_cost = 1`.
                 - Total cost for this path: `nav_cal_cost + cal_cost + nav_img_cost + take_cost + nav_comm_cost + comm_cost`.
                 - Update `min_cost_g` if this path is valid (all distances finite).
            iv.  Add `min_cost_g` to `h` if it's finite, otherwise add 1.
    6.  Return `h`. If `h` calculated to 0 but state is not goal, return 1.
    """
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        static_facts = task.static
        self.infinity = float('inf')

        # --- Precompute Static Information ---
        self.waypoints = set()
        self.rovers = set()
        self.cameras_all = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander_location = None
        self.lander = None # Store lander name if needed

        # Mappings
        self.rover_equipment = {} # rover -> set(capabilities: 'soil', 'rock', 'imaging')
        self.rover_store = {} # rover -> store
        self.store_rover = {} # store -> rover
        self.cameras_on_rover = {} # rover -> set(cameras)
        self.camera_supports = {} # camera -> set(modes)
        self.camera_calib_target = {} # camera -> objective
        self.visible_from_objective = {} # objective -> set(waypoints)
        self.calibration_targets = set() # set of objectives used for calibration

        # Visibility Graph (waypoint -> set(visible waypoints))
        self.visibility_graph = {}

        # Parse static facts
        for fact in static_facts:
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip empty or malformed facts
                predicate = parts[0]

                # Basic types (implicit from usage, but good to collect)
                # Using PDDL definition which doesn't explicitly declare types like this
                # We infer types from predicates like 'rover', 'waypoint' if they exist,
                # otherwise from usage in other predicates.
                if predicate == 'rover': self.rovers.add(parts[1]); self.rover_equipment[parts[1]] = set()
                elif predicate == 'waypoint': self.waypoints.add(parts[1])
                elif predicate == 'camera': self.cameras_all.add(parts[1])
                elif predicate == 'objective': self.objectives.add(parts[1])
                elif predicate == 'mode': self.modes.add(parts[1])
                elif predicate == 'store': self.stores.add(parts[1])
                elif predicate == 'lander': self.lander = parts[1]

                # Relations
                elif predicate == 'at_lander': self.lander_location = parts[2]; self.lander = parts[1]
                elif predicate == 'equipped_for_soil_analysis': self.rovers.add(parts[1]); self.rover_equipment.setdefault(parts[1], set()).add('soil')
                elif predicate == 'equipped_for_rock_analysis': self.rovers.add(parts[1]); self.rover_equipment.setdefault(parts[1], set()).add('rock')
                elif predicate == 'equipped_for_imaging': self.rovers.add(parts[1]); self.rover_equipment.setdefault(parts[1], set()).add('imaging')
                elif predicate == 'store_of': self.stores.add(parts[1]); self.rovers.add(parts[2]); self.rover_store[parts[2]] = parts[1]; self.store_rover[parts[1]] = parts[2]
                elif predicate == 'on_board':
                    cam, rov = parts[1], parts[2]
                    self.cameras_all.add(cam); self.rovers.add(rov)
                    if rov not in self.cameras_on_rover: self.cameras_on_rover[rov] = set()
                    self.cameras_on_rover[rov].add(cam)
                elif predicate == 'supports':
                    cam, mode = parts[1], parts[2]
                    self.cameras_all.add(cam); self.modes.add(mode)
                    if cam not in self.camera_supports: self.camera_supports[cam] = set()
                    self.camera_supports[cam].add(mode)
                elif predicate == 'calibration_target':
                    cam, obj = parts[1], parts[2]
                    self.cameras_all.add(cam); self.objectives.add(obj)
                    self.camera_calib_target[cam] = obj
                    self.calibration_targets.add(obj)
                elif predicate == 'visible_from':
                    obj, wp = parts[1], parts[2]
                    self.objectives.add(obj); self.waypoints.add(wp)
                    if obj not in self.visible_from_objective: self.visible_from_objective[obj] = set()
                    self.visible_from_objective[obj].add(wp)
                elif predicate == 'visible':
                    w1, w2 = parts[1], parts[2]
                    self.waypoints.add(w1); self.waypoints.add(w2)
                    if w1 not in self.visibility_graph: self.visibility_graph[w1] = set()
                    self.visibility_graph[w1].add(w2)
            except IndexError:
                print(f"Warning: Skipping malformed static fact: {fact}")
                continue


        # Ensure all waypoints are keys in the visibility graph
        for wp in self.waypoints:
            if wp not in self.visibility_graph:
                self.visibility_graph[wp] = set()

        # Compute all-pairs shortest paths using BFS on visibility graph
        self.distances = self._compute_all_pairs_shortest_paths()

        # Find waypoints visible from the lander (communication points)
        self.comm_waypoints = set()
        if self.lander_location:
             # Ensure lander location is a known waypoint
             self.waypoints.add(self.lander_location)
             if self.lander_location not in self.visibility_graph:
                 self.visibility_graph[self.lander_location] = set()

             for wp_from in self.waypoints:
                 # Check if lander_location is visible *from* wp_from
                 if self.lander_location in self.visibility_graph.get(wp_from, set()):
                     self.comm_waypoints.add(wp_from)

        # Map calibration targets to waypoints from which they are visible
        self.visible_from_calib_target = {}
        for target_obj in self.calibration_targets:
             # Calibration targets are objectives, use the objective visibility info
             self.visible_from_calib_target[target_obj] = self.visible_from_objective.get(target_obj, set())

        # Parse goals into sets for quick lookup
        self.goal_soil = set()
        self.goal_rock = set()
        self.goal_image = set() # set of (objective, mode) tuples
        for goal in self.goals:
            try:
                parts = get_parts(goal)
                if not parts: continue
                predicate = parts[0]
                if predicate == 'communicated_soil_data': self.goal_soil.add(parts[1])
                elif predicate == 'communicated_rock_data': self.goal_rock.add(parts[1])
                elif predicate == 'communicated_image_data': self.goal_image.add(tuple(parts[1:]))
            except IndexError:
                print(f"Warning: Skipping malformed goal: {goal}")
                continue

    def _bfs(self, start_node):
        """Performs BFS from start_node on the visibility graph."""
        distances = {wp: self.infinity for wp in self.waypoints}
        if start_node not in self.waypoints:
             return distances # Start node not known, return all infinite
        distances[start_node] = 0
        queue = deque([start_node])
        processed = {start_node}

        while queue:
            current_wp = queue.popleft()
            current_dist = distances[current_wp]

            for neighbor in self.visibility_graph.get(current_wp, set()):
                if neighbor not in processed:
                     if neighbor in distances: # Check if neighbor is a valid waypoint
                         distances[neighbor] = current_dist + 1
                         processed.add(neighbor)
                         queue.append(neighbor)
        return distances

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest paths between all pairs of waypoints."""
        all_distances = {}
        for wp in self.waypoints:
            all_distances[wp] = self._bfs(wp)
        # Add distances from lander location if it's a waypoint
        if self.lander_location and self.lander_location in self.waypoints and self.lander_location not in all_distances:
             all_distances[self.lander_location] = self._bfs(self.lander_location)
        return all_distances

    def _get_dist(self, wp1, wp2):
        """Gets the precomputed shortest distance between two waypoints."""
        if wp1 is None or wp2 is None: return self.infinity
        if wp1 == wp2: return 0
        # Ensure both waypoints exist in our precomputed structure
        if wp1 not in self.distances or wp2 not in self.distances.get(wp1, {}):
            # Try reverse? Visibility might not be symmetric. Stick to BFS result.
            # print(f"Warning: Distance lookup failed for {wp1} -> {wp2}")
            return self.infinity
        return self.distances[wp1].get(wp2, self.infinity)

    def _find_closest_waypoint(self, source_wp, target_wps):
        """Finds the waypoint in target_wps closest to source_wp."""
        min_dist = self.infinity
        closest_wp = None

        if not target_wps or source_wp is None:
            return None, self.infinity

        for target_wp in target_wps:
            dist = self._get_dist(source_wp, target_wp)
            if dist < min_dist:
                min_dist = dist
                closest_wp = target_wp
            # If distance is 0 and source is in targets, return immediately
            if dist == 0 and source_wp == target_wp:
                 return source_wp, 0

        # If no target waypoint is reachable, min_dist remains infinity
        return closest_wp, min_dist

    def _find_closest_comm_waypoint(self, source_wp):
         """Finds the communication waypoint closest to source_wp."""
         # If source_wp itself is a comm waypoint, distance is 0
         if source_wp in self.comm_waypoints:
             return source_wp, 0
         return self._find_closest_waypoint(source_wp, self.comm_waypoints)

    def __call__(self, node):
        state = node.state
        # Check goal condition first
        if self.task.goal_reached(state):
            return 0

        # --- Parse Current State ---
        current_at = {} # rover -> waypoint
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated = set() # (camera, rover)
        current_store_full = set() # store name
        current_at_soil = set() # waypoint
        current_at_rock = set() # waypoint

        for fact in state:
            try:
                parts = get_parts(fact)
                if not parts: continue
                predicate = parts[0]
                # Use match helper for robustness? No, direct indexing is faster if format is reliable.
                if predicate == 'at' and len(parts) == 3 and parts[1] in self.rovers: current_at[parts[1]] = parts[2]
                elif predicate == 'have_soil_analysis' and len(parts) == 3: current_have_soil.add(tuple(parts[1:]))
                elif predicate == 'have_rock_analysis' and len(parts) == 3: current_have_rock.add(tuple(parts[1:]))
                elif predicate == 'have_image' and len(parts) == 4: current_have_image.add(tuple(parts[1:]))
                elif predicate == 'calibrated' and len(parts) == 3: current_calibrated.add(tuple(parts[1:]))
                elif predicate == 'full' and len(parts) == 2: current_store_full.add(parts[1])
                elif predicate == 'at_soil_sample' and len(parts) == 2: current_at_soil.add(parts[1])
                elif predicate == 'at_rock_sample' and len(parts) == 2: current_at_rock.add(parts[1])
            except IndexError:
                # print(f"Warning: Skipping malformed state fact: {fact}")
                continue


        h = 0
        processed_goals = set() # Store goal facts as strings to handle potential duplicates

        # --- Calculate Cost for Soil Goals ---
        for goal_wp in self.goal_soil:
            goal_fact_str = f"(communicated_soil_data {goal_wp})"
            if goal_fact_str in state or goal_fact_str in processed_goals: continue
            processed_goals.add(goal_fact_str)

            min_cost_g = self.infinity

            # Option 1: Rover already has sample
            for (rover, wp) in current_have_soil:
                if wp == goal_wp:
                    rover_loc = current_at.get(rover)
                    if rover_loc:
                        comm_wp, dist_to_comm = self._find_closest_comm_waypoint(rover_loc)
                        if comm_wp is not None and dist_to_comm != self.infinity:
                            cost = dist_to_comm + 1 # navigate + communicate
                            min_cost_g = min(min_cost_g, cost)

            # Option 2: Sample exists at waypoint
            if goal_wp in current_at_soil:
                for rover in self.rovers:
                    if 'soil' in self.rover_equipment.get(rover, set()):
                        rover_loc = current_at.get(rover)
                        rover_store = self.rover_store.get(rover)
                        if rover_loc and rover_store:
                            cost_drop = 1 if rover_store in current_store_full else 0
                            dist_to_sample = self._get_dist(rover_loc, goal_wp)
                            comm_wp, dist_sample_to_comm = self._find_closest_comm_waypoint(goal_wp)

                            # Ensure all components are reachable
                            if dist_to_sample != self.infinity and comm_wp is not None and dist_sample_to_comm != self.infinity:
                                cost = cost_drop + dist_to_sample + 1 + dist_sample_to_comm + 1
                                # drop? + nav_sample + sample + nav_comm + comm
                                min_cost_g = min(min_cost_g, cost)

            h += min_cost_g if min_cost_g != self.infinity else 1

        # --- Calculate Cost for Rock Goals ---
        for goal_wp in self.goal_rock:
            goal_fact_str = f"(communicated_rock_data {goal_wp})"
            if goal_fact_str in state or goal_fact_str in processed_goals: continue
            processed_goals.add(goal_fact_str)

            min_cost_g = self.infinity

            # Option 1: Rover already has sample
            for (rover, wp) in current_have_rock:
                 if wp == goal_wp:
                    rover_loc = current_at.get(rover)
                    if rover_loc:
                        comm_wp, dist_to_comm = self._find_closest_comm_waypoint(rover_loc)
                        if comm_wp is not None and dist_to_comm != self.infinity:
                            cost = dist_to_comm + 1 # navigate + communicate
                            min_cost_g = min(min_cost_g, cost)

            # Option 2: Sample exists at waypoint
            if goal_wp in current_at_rock:
                for rover in self.rovers:
                    if 'rock' in self.rover_equipment.get(rover, set()):
                        rover_loc = current_at.get(rover)
                        rover_store = self.rover_store.get(rover)
                        if rover_loc and rover_store:
                            cost_drop = 1 if rover_store in current_store_full else 0
                            dist_to_sample = self._get_dist(rover_loc, goal_wp)
                            comm_wp, dist_sample_to_comm = self._find_closest_comm_waypoint(goal_wp)

                            if dist_to_sample != self.infinity and comm_wp is not None and dist_sample_to_comm != self.infinity:
                                cost = cost_drop + dist_to_sample + 1 + dist_sample_to_comm + 1
                                min_cost_g = min(min_cost_g, cost)

            h += min_cost_g if min_cost_g != self.infinity else 1

        # --- Calculate Cost for Image Goals ---
        for (goal_obj, goal_mode) in self.goal_image:
            goal_fact_str = f"(communicated_image_data {goal_obj} {goal_mode})"
            if goal_fact_str in state or goal_fact_str in processed_goals: continue
            processed_goals.add(goal_fact_str)

            min_cost_g = self.infinity

            # Option 1: Rover already has image
            for (rover, obj, mode) in current_have_image:
                if obj == goal_obj and mode == goal_mode:
                    rover_loc = current_at.get(rover)
                    if rover_loc:
                        comm_wp, dist_to_comm = self._find_closest_comm_waypoint(rover_loc)
                        if comm_wp is not None and dist_to_comm != self.infinity:
                            cost = dist_to_comm + 1 # navigate + communicate
                            min_cost_g = min(min_cost_g, cost)

            # Option 2: Need to take image
            for rover in self.rovers:
                if 'imaging' in self.rover_equipment.get(rover, set()):
                    rover_loc = current_at.get(rover)
                    if not rover_loc: continue

                    for camera in self.cameras_on_rover.get(rover, set()):
                        supported_modes = self.camera_supports.get(camera, set())
                        if goal_mode in supported_modes:
                            visible_obj_wps = self.visible_from_objective.get(goal_obj, set())
                            if not visible_obj_wps: continue # Cannot take image if objective never visible

                            calib_target = self.camera_calib_target.get(camera)
                            if not calib_target: continue # Cannot calibrate this camera

                            needs_calibration = (camera, rover) not in current_calibrated
                            cost_calib = 0
                            nav_cal_cost = 0
                            loc_after_cal = rover_loc # Location after potential calibration step

                            if needs_calibration:
                                cost_calib = 1
                                visible_calib_wps = self.visible_from_calib_target.get(calib_target, set())
                                if not visible_calib_wps: continue # Cannot calibrate if target never visible from anywhere

                                best_calib_wp, dist_to_calib = self._find_closest_waypoint(rover_loc, visible_calib_wps)
                                if best_calib_wp is None or dist_to_calib == self.infinity: continue # Cannot reach calibration spot

                                nav_cal_cost = dist_to_calib
                                loc_after_cal = best_calib_wp

                            # Find best place to take image (after potential calibration)
                            best_img_wp, dist_cal_to_img = self._find_closest_waypoint(loc_after_cal, visible_obj_wps)
                            if best_img_wp is None or dist_cal_to_img == self.infinity: continue # Cannot reach imaging spot

                            cost_take = 1

                            # Find best place to communicate (after taking image)
                            best_comm_wp, dist_img_to_comm = self._find_closest_comm_waypoint(best_img_wp)
                            if best_comm_wp is None or dist_img_to_comm == self.infinity: continue # Cannot reach comm spot

                            cost_comm = 1

                            # Total cost for this rover/camera
                            total_cost = nav_cal_cost + cost_calib + dist_cal_to_img + cost_take + dist_img_to_comm + cost_comm
                            min_cost_g = min(min_cost_g, total_cost)

            h += min_cost_g if min_cost_g != self.infinity else 1

        # Final check: if h is 0 but state is not goal, return 1.
        if h == 0 and not self.task.goal_reached(state):
            # This case might happen if all goals seem unachievable (min_cost_g=inf)
            # and the fallback '1' sums up to 0 (no goals?). Or if goals are empty.
            # Return 1 ensures non-zero for non-goal states.
            return 1
        elif h < 0: # Should not happen, but as a safeguard
             return 1
        else:
            # Return integer cost
            return int(round(h))
