import heapq
import math
from fnmatch import fnmatch
# Assuming the heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts/goals
def get_parts(fact_string):
    """Removes parentheses and splits PDDL fact string into predicate and arguments."""
    # Ensure the input is a string and has parentheses
    if not isinstance(fact_string, str) or not fact_string.startswith('(') or not fact_string.endswith(')'):
        # Handle unexpected input format, perhaps return an empty list or raise error
        # For robustness, let's return an empty list if format is wrong.
        print(f"Warning: Unexpected fact format: {fact_string}")
        return []
    return fact_string[1:-1].split()


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from the current state. It uses an additive approach (h_add style), summing the
    estimated costs for achieving each unsatisfied goal predicate independently.
    The cost for each goal (communicating soil, rock, or image data) is estimated
    by finding the cheapest sequence of actions (navigation, sampling/imaging,
    calibration, communication) required, considering the current state of rovers
    (location, equipment, store status, camera calibration) and precomputed
    shortest path distances between waypoints. It is designed for Greedy Best-First
    Search and does not guarantee admissibility.

    # Assumptions
    - Action costs are uniform (1 per action).
    - The heuristic ignores most negative interactions between achieving
      different goals (e.g., one rover being needed for multiple tasks,
      resource contention beyond simple checks like store state). It aims for
      guidance in greedy search, not admissibility.
    - The `drop` action can be performed anywhere by a rover with a full store.
    - `can_traverse(rover, wp_from, wp_to)` defines the traversable graph edges
      for each rover used for pathfinding.
    - The `visible(wp_from, wp_to)` precondition in the `navigate` action is assumed
      to be handled by the planner's applicability checks and is not explicitly
      modeled in the heuristic's pathfinding.
    - `visible(wp_comm, wp_lander)` defines line-of-sight for communication between
      a rover at `wp_comm` and the lander at `wp_lander`.
    - `visible_from(objective, waypoint)` defines visibility for imaging/calibration.
    - Calibration targets (`calibration_target(camera, objective)`) are objectives
      listed in `visible_from` predicates.

    # Heuristic Initialization (`__init__`)
    - Parses static facts (`task.static`) to build internal data structures:
        - Rover capabilities (equipment, cameras, stores).
        - Camera properties (supported modes, calibration targets).
        - Waypoint connectivity (`can_traverse`) for each rover.
        - Visibility information (`visible`, `visible_from`).
        - Lander location.
        - Sets of waypoints suitable for communication, imaging (per objective),
          and calibration (per target).
    - Precomputes all-pairs shortest paths (APSP) for each rover based on its
      `can_traverse` graph using BFS. This provides efficient `dist(rover, wp1, wp2)` lookups.
    - Parses the goal conditions (`task.goals`).

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Parse the current `state` to determine:
       - Rover locations (`at`).
       - Store status (`full`/`empty`).
       - Calibrated cameras (`calibrated`).
       - Data currently held by rovers (`have_soil_analysis`, `have_rock_analysis`, `have_image`).
       - Data already communicated (`communicated_...`).
    2. Initialize `total_heuristic_cost = 0`.
    3. For each goal predicate `g` in the task goals:
       a. Check if `g` is already satisfied in the current `state`. If yes, continue.
       b. If `g` is not satisfied, estimate the minimum cost `min_cost_for_goal` to achieve it:
          i. **If `g` is `communicated_soil_data(w)`:**
             - If any rover `r` `have_soil_analysis(r, w)`:
               Cost = `min_r(dist(r_loc, comm_wp) + 1)` (Move + Communicate).
             - Else (need to sample):
               Cost = `min_r(cost_drop + dist(r_loc, w) + 1 + dist(w, comm_wp) + 1)`
               (Drop + Move + Sample + Move + Communicate).
          ii. **If `g` is `communicated_rock_data(w)`:** Similar logic to soil.
          iii. **If `g` is `communicated_image_data(o, m)`:**
              - If any rover `r` `have_image(r, o, m)`:
                Cost = `min_r(dist(r_loc, comm_wp) + 1)` (Move + Communicate).
              - Else (need to take image):
                Cost = `min_{r,i}(cost_to_get_image + cost_to_communicate)`
                - `cost_to_get_image`: Find closest imaging waypoint `p`.
                  - If camera `i` calibrated: `dist(r_loc, p) + 1`.
                  - If not calibrated: Find closest calibration waypoint `w_cal`.
                    Cost = `dist(r_loc, w_cal) + 1 + dist(w_cal, p) + 1`.
                - `cost_to_communicate` = `dist(p, comm_wp) + 1`.
       c. Add `min_cost_for_goal` (if finite) to `total_heuristic_cost`. If any goal is
          unreachable (`math.inf`), return `math.inf` immediately, signaling a dead end.
    4. Check if the current state satisfies all goals. If yes, return 0.
    5. If `total_heuristic_cost` is 0 but the state is not a goal state (e.g., empty goal list
       or calculation error), return 1 to ensure non-goal states have non-zero cost.
    6. Otherwise, return `total_heuristic_cost`.
    """

    def __init__(self, task):
        """Initializes the heuristic, parsing static info and precomputing paths."""
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static

        # --- Precompute data structures ---
        self._parse_static_facts()
        self._compute_all_pairs_shortest_paths()

    def _parse_static_facts(self):
        """Parses static facts to build internal data structures."""
        # Initialize sets for discovered objects
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander = None
        self.lander_location = None

        # Store relationships and properties
        self.rover_equipment = {} # rover -> set('soil', 'rock', 'image')
        self.rover_stores = {} # rover -> store
        self.store_rover = {} # store -> rover
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_rover = {} # camera -> rover
        self.camera_supports = {} # camera -> set(mode)
        self.camera_calibration_target = {} # camera -> objective (target)
        self.objective_visibility = {} # objective -> set(waypoint)
        self.target_visibility = {} # objective (used as target) -> set(waypoint)
        self.comm_waypoints = set()
        self.connectivity = {} # rover -> waypoint -> set(neighbor)
        self.visibility = {} # waypoint -> set(visible_waypoint)

        # --- Pass 1: Discover all objects and waypoints ---
        # Use temporary set for waypoints discovered from various predicates
        temp_waypoints = set()
        for fact in self.static_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts
             pred = parts[0]
             args = parts[1:]
             try:
                 if pred == 'at_lander':
                     self.lander = args[0]
                     self.lander_location = args[1]
                     temp_waypoints.add(args[1])
                 elif pred == 'equipped_for_soil_analysis': self.rovers.add(args[0])
                 elif pred == 'equipped_for_rock_analysis': self.rovers.add(args[0])
                 elif pred == 'equipped_for_imaging': self.rovers.add(args[0])
                 elif pred == 'store_of':
                     self.stores.add(args[0])
                     self.rovers.add(args[1])
                 elif pred == 'on_board':
                     self.cameras.add(args[0])
                     self.rovers.add(args[1])
                 elif pred == 'supports':
                     self.cameras.add(args[0])
                     self.modes.add(args[1])
                 elif pred == 'calibration_target':
                     self.cameras.add(args[0])
                     self.objectives.add(args[1]) # Assume targets are objectives
                 elif pred == 'visible_from':
                     self.objectives.add(args[0])
                     temp_waypoints.add(args[1])
                 elif pred == 'can_traverse':
                     self.rovers.add(args[0])
                     temp_waypoints.add(args[1])
                     temp_waypoints.add(args[2])
                 elif pred == 'visible':
                     temp_waypoints.add(args[0])
                     temp_waypoints.add(args[1])
             except IndexError:
                 # print(f"Warning: Skipping fact due to IndexError: {fact}")
                 pass # Ignore facts with unexpected argument counts
        self.waypoints = temp_waypoints

        # --- Initialize data structures based on discovered objects ---
        for r in self.rovers:
            self.rover_equipment[r] = set()
            self.rover_cameras[r] = set()
            # Initialize connectivity for all known waypoints for this rover
            self.connectivity[r] = {w: set() for w in self.waypoints}
        for c in self.cameras:
            self.camera_supports[c] = set()
        for o in self.objectives:
            self.objective_visibility[o] = set()
            self.target_visibility[o] = set()
        for w in self.waypoints:
            self.visibility[w] = set()

        # --- Pass 2: Populate relationships ---
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            try:
                if pred == 'equipped_for_soil_analysis': self.rover_equipment[args[0]].add('soil')
                elif pred == 'equipped_for_rock_analysis': self.rover_equipment[args[0]].add('rock')
                elif pred == 'equipped_for_imaging': self.rover_equipment[args[0]].add('image')
                elif pred == 'store_of':
                    self.rover_stores[args[1]] = args[0]
                    self.store_rover[args[0]] = args[1]
                elif pred == 'on_board':
                    # Ensure rover and camera exist before adding
                    if args[1] in self.rovers and args[0] in self.cameras:
                        self.rover_cameras[args[1]].add(args[0])
                        self.camera_rover[args[0]] = args[1]
                elif pred == 'supports':
                     if args[0] in self.cameras and args[1] in self.modes:
                        self.camera_supports[args[0]].add(args[1])
                elif pred == 'calibration_target':
                    if args[0] in self.cameras and args[1] in self.objectives:
                        self.camera_calibration_target[args[0]] = args[1]
                elif pred == 'visible_from':
                    if args[0] in self.objectives and args[1] in self.waypoints:
                        self.objective_visibility[args[0]].add(args[1])
                elif pred == 'can_traverse':
                    rover, w1, w2 = args
                    # Ensure rover and waypoints exist before adding edge
                    if rover in self.rovers and w1 in self.waypoints and w2 in self.waypoints:
                        self.connectivity[rover][w1].add(w2)
                elif pred == 'visible':
                    w1, w2 = args
                    if w1 in self.waypoints and w2 in self.waypoints:
                        self.visibility[w1].add(w2)
            except IndexError:
                 # print(f"Warning: Skipping fact due to IndexError: {fact}")
                 pass # Ignore facts with unexpected argument counts

        # Populate target_visibility using objective_visibility for known targets
        for cam, target_obj in self.camera_calibration_target.items():
            if target_obj in self.objective_visibility:
                 self.target_visibility[target_obj] = self.objective_visibility[target_obj]

        # Populate comm_waypoints (waypoints visible from the lander location)
        if self.lander_location and self.lander_location in self.waypoints:
            for w1 in self.waypoints:
                 # Check if w1 can see the lander location
                 if self.lander_location in self.visibility.get(w1, set()):
                     self.comm_waypoints.add(w1)

        # Parse goals into a set of tuples for easier checking
        self.parsed_goals = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts: # Only add if parsing was successful
                self.parsed_goals.add(tuple(parts))


    def _compute_shortest_paths(self, rover, start_node):
        """Computes shortest paths from start_node for a specific rover using BFS."""
        distances = {wp: math.inf for wp in self.waypoints}
        if start_node not in self.waypoints:
             # This case should ideally not happen if start_node is a valid waypoint
             return distances

        distances[start_node] = 0
        # Use a simple list as queue for standard BFS
        queue = [(start_node)]
        head = 0

        while head < len(queue):
            current_wp = queue[head]
            head += 1

            current_dist = distances[current_wp]

            # Use rover-specific connectivity
            # Check if current_wp exists in connectivity for this rover
            if current_wp not in self.connectivity[rover]: continue

            for neighbor in self.connectivity[rover].get(current_wp, set()):
                if neighbor not in distances: continue # Should not happen

                # If neighbor is unvisited (dist is inf) or we found a shorter path
                # (For unit costs, only need to check if inf)
                if distances[neighbor] == math.inf:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def _compute_all_pairs_shortest_paths(self):
        """Computes APSP for all rovers and stores them."""
        self.shortest_paths = {} # rover -> start_wp -> end_wp -> distance
        for rover in self.rovers:
            self.shortest_paths[rover] = {}
            for start_wp in self.waypoints:
                 # Compute paths from start_wp for this rover
                 self.shortest_paths[rover][start_wp] = self._compute_shortest_paths(rover, start_wp)

    def _get_dist(self, rover, wp1, wp2):
        """Gets precomputed shortest distance, returns math.inf if unreachable."""
        if wp1 == wp2: return 0
        # Robust check for existence of keys
        if rover not in self.shortest_paths: return math.inf
        if wp1 not in self.shortest_paths[rover]: return math.inf
        # .get(wp2, math.inf) handles case where wp2 is unreachable from wp1
        dist = self.shortest_paths[rover][wp1].get(wp2, math.inf)
        return dist

    def _find_closest_waypoint_dist(self, rover, current_wp, target_wp_set):
        """Finds minimum distance from current_wp to any waypoint in target_wp_set."""
        min_dist = math.inf
        if not target_wp_set:
             return math.inf # No target waypoints available

        # Ensure current_wp is valid before proceeding
        if current_wp not in self.waypoints:
            return math.inf

        for target_wp in target_wp_set:
            # Ensure target_wp is valid before getting distance
            if target_wp not in self.waypoints: continue
            dist = self._get_dist(rover, current_wp, target_wp)
            min_dist = min(min_dist, dist)
        return min_dist

    def __call__(self, node):
        """Computes the heuristic value for the given state node."""
        state = node.state

        # --- Parse current state ---
        rover_locations = {}
        store_states = {s: 'empty' for s in self.stores} # Default to empty
        calibrated_cameras = set() # (camera, rover) tuple
        have_soil = set() # (rover, waypoint) tuple
        have_rock = set() # (rover, waypoint) tuple
        have_image = set() # (rover, objective, mode) tuple
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode) tuple

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]
            args = parts[1:]

            try:
                if pred == 'at' and args[0] in self.rovers: rover_locations[args[0]] = args[1]
                elif pred == 'full' and args[0] in self.stores: store_states[args[0]] = 'full'
                elif pred == 'empty' and args[0] in self.stores: store_states[args[0]] = 'empty'
                elif pred == 'calibrated' and args[0] in self.cameras:
                    # Find the rover associated with this camera
                    if args[0] in self.camera_rover:
                         calibrated_cameras.add((args[0], self.camera_rover[args[0]]))
                elif pred == 'have_soil_analysis': have_soil.add((args[0], args[1]))
                elif pred == 'have_rock_analysis': have_rock.add((args[0], args[1]))
                elif pred == 'have_image': have_image.add((args[0], args[1], args[2]))
                elif pred == 'communicated_soil_data': communicated_soil.add(args[0])
                elif pred == 'communicated_rock_data': communicated_rock.add(args[0])
                elif pred == 'communicated_image_data': communicated_image.add((args[0], args[1]))
            except IndexError:
                # print(f"Warning: Skipping fact due to IndexError: {fact}")
                pass # Ignore facts with unexpected argument counts


        # --- Calculate heuristic value (h_add style) ---
        total_heuristic_cost = 0
        goal_achieved = True # Assume goal state until proven otherwise

        if not self.parsed_goals: # Handle case with no goals
             return 0

        for goal_tuple in self.parsed_goals:
            goal_pred = goal_tuple[0]
            goal_args = goal_tuple[1:]
            is_satisfied = False

            # --- Check if goal is already satisfied ---
            try:
                if goal_pred == 'communicated_soil_data' and goal_args[0] in communicated_soil: is_satisfied = True
                elif goal_pred == 'communicated_rock_data' and goal_args[0] in communicated_rock: is_satisfied = True
                elif goal_pred == 'communicated_image_data' and (goal_args[0], goal_args[1]) in communicated_image: is_satisfied = True
                # Add checks for other potential goal types if necessary
            except IndexError:
                # Malformed goal in task definition?
                # print(f"Warning: Malformed goal tuple: {goal_tuple}")
                goal_achieved = False # Cannot satisfy a malformed goal
                continue # Skip this goal

            if is_satisfied:
                continue

            # If we reach here, at least one goal is not satisfied
            goal_achieved = False
            min_cost_for_goal = math.inf

            # --- Estimate cost for this unsatisfied goal ---
            try:
                # 1. Communicated Soil Data
                if goal_pred == 'communicated_soil_data':
                    w = goal_args[0]
                    if w not in self.waypoints: continue # Goal waypoint invalid

                    # Option A: A rover already has the analysis
                    rovers_with_soil = {r for r, wp in have_soil if wp == w}
                    for r in rovers_with_soil:
                        if r not in rover_locations: continue
                        current_loc = rover_locations[r]
                        comm_dist = self._find_closest_waypoint_dist(r, current_loc, self.comm_waypoints)
                        if comm_dist != math.inf:
                            min_cost_for_goal = min(min_cost_for_goal, comm_dist + 1) # Move + Communicate

                    # Option B: Need to sample first
                    for r in self.rovers:
                        if 'soil' not in self.rover_equipment.get(r, set()): continue
                        if r not in rover_locations: continue
                        store = self.rover_stores.get(r)
                        if not store: continue # Rover needs a store

                        current_loc = rover_locations[r]
                        cost_drop = 1 if store_states.get(store, 'empty') == 'full' else 0
                        dist_to_sample_loc = self._get_dist(r, current_loc, w)
                        if dist_to_sample_loc == math.inf: continue

                        # Cost to get analysis: drop + move + sample
                        cost_get_analysis = cost_drop + dist_to_sample_loc + 1

                        # Cost to communicate from sample location w
                        comm_dist_from_sample = self._find_closest_waypoint_dist(r, w, self.comm_waypoints)
                        if comm_dist_from_sample == math.inf: continue

                        total_cost = cost_get_analysis + comm_dist_from_sample + 1 # Get analysis + Move to comm + Communicate
                        min_cost_for_goal = min(min_cost_for_goal, total_cost)

                # 2. Communicated Rock Data
                elif goal_pred == 'communicated_rock_data':
                    w = goal_args[0]
                    if w not in self.waypoints: continue

                    # Option A: Rover has analysis
                    rovers_with_rock = {r for r, wp in have_rock if wp == w}
                    for r in rovers_with_rock:
                        if r not in rover_locations: continue
                        current_loc = rover_locations[r]
                        comm_dist = self._find_closest_waypoint_dist(r, current_loc, self.comm_waypoints)
                        if comm_dist != math.inf:
                            min_cost_for_goal = min(min_cost_for_goal, comm_dist + 1)

                    # Option B: Need to sample
                    for r in self.rovers:
                        if 'rock' not in self.rover_equipment.get(r, set()): continue
                        if r not in rover_locations: continue
                        store = self.rover_stores.get(r)
                        if not store: continue

                        current_loc = rover_locations[r]
                        cost_drop = 1 if store_states.get(store, 'empty') == 'full' else 0
                        dist_to_sample_loc = self._get_dist(r, current_loc, w)
                        if dist_to_sample_loc == math.inf: continue

                        cost_get_analysis = cost_drop + dist_to_sample_loc + 1
                        comm_dist_from_sample = self._find_closest_waypoint_dist(r, w, self.comm_waypoints)
                        if comm_dist_from_sample == math.inf: continue

                        total_cost = cost_get_analysis + comm_dist_from_sample + 1
                        min_cost_for_goal = min(min_cost_for_goal, total_cost)

                # 3. Communicated Image Data
                elif goal_pred == 'communicated_image_data':
                    o, m = goal_args[0], goal_args[1]
                    if o not in self.objectives or m not in self.modes: continue # Invalid goal args

                    # Option A: Rover has image
                    rovers_with_image = {r for r, obj, mode in have_image if obj == o and mode == m}
                    for r in rovers_with_image:
                        if r not in rover_locations: continue
                        current_loc = rover_locations[r]
                        comm_dist = self._find_closest_waypoint_dist(r, current_loc, self.comm_waypoints)
                        if comm_dist != math.inf:
                            min_cost_for_goal = min(min_cost_for_goal, comm_dist + 1)

                    # Option B: Need to take image
                    possible_image_wps = self.objective_visibility.get(o, set())
                    if not possible_image_wps: continue # Objective not visible from anywhere relevant

                    for r in self.rovers:
                        if 'image' not in self.rover_equipment.get(r, set()): continue
                        if r not in rover_locations: continue
                        current_loc = rover_locations[r]

                        # Find best camera on this rover supporting the mode
                        best_cam_cost = math.inf
                        for i in self.rover_cameras.get(r, set()):
                            if m not in self.camera_supports.get(i, set()): continue

                            # Find closest waypoint p to take image from current_loc
                            min_dist_to_image_wp = math.inf
                            best_image_wp = None
                            for p in possible_image_wps:
                                dist = self._get_dist(r, current_loc, p)
                                if dist < min_dist_to_image_wp:
                                    min_dist_to_image_wp = dist
                                    best_image_wp = p

                            if best_image_wp is None or min_dist_to_image_wp == math.inf: continue # Cannot reach any imaging waypoint

                            cost_to_take_image = 0
                            loc_after_take_image = best_image_wp

                            # Check calibration
                            if (i, r) in calibrated_cameras:
                                # Move to image_wp + take_image
                                cost_to_take_image = min_dist_to_image_wp + 1
                            else:
                                # Need to calibrate first
                                calib_target = self.camera_calibration_target.get(i)
                                if not calib_target: continue # Camera cannot be calibrated
                                possible_calib_wps = self.target_visibility.get(calib_target, set())
                                if not possible_calib_wps: continue # Target not visible from anywhere

                                # Find closest calibration waypoint w_cal from current_loc
                                min_dist_to_calib_wp = math.inf
                                best_calib_wp = None
                                for w_cal in possible_calib_wps:
                                     dist = self._get_dist(r, current_loc, w_cal)
                                     if dist < min_dist_to_calib_wp:
                                         min_dist_to_calib_wp = dist
                                         best_calib_wp = w_cal

                                if best_calib_wp is None or min_dist_to_calib_wp == math.inf: continue # Cannot reach calibration spot

                                dist_calib_to_image = self._get_dist(r, best_calib_wp, best_image_wp)
                                if dist_calib_to_image == math.inf: continue # Cannot reach image spot from calib spot

                                # Move to calib_wp + calibrate + move to image_wp + take_image
                                cost_to_take_image = min_dist_to_calib_wp + 1 + dist_calib_to_image + 1

                            # Cost to communicate from image location
                            comm_dist_from_image = self._find_closest_waypoint_dist(r, loc_after_take_image, self.comm_waypoints)
                            if comm_dist_from_image == math.inf: continue # Cannot communicate

                            total_cost = cost_to_take_image + comm_dist_from_image + 1 # Get image + Move to comm + Communicate
                            best_cam_cost = min(best_cam_cost, total_cost)

                        min_cost_for_goal = min(min_cost_for_goal, best_cam_cost)

            except IndexError:
                # Malformed goal args?
                # print(f"Warning: IndexError processing goal: {goal_tuple}")
                min_cost_for_goal = math.inf # Treat as unreachable


            # Add the minimum cost for this goal (if feasible)
            if min_cost_for_goal == math.inf:
                # If any goal is impossible to achieve from this state, return infinity
                return math.inf
            else:
                total_heuristic_cost += min_cost_for_goal


        # Final check: if goal_achieved is true, return 0.
        if goal_achieved:
            return 0
        # If cost is 0 but not goal state (e.g. empty goal list?), return 1.
        elif total_heuristic_cost == 0:
             return 1
        else:
            # Return the calculated cost, ensuring it's an integer if possible,
            # but keep as float if infinity was involved (though we return inf earlier)
            return int(total_heuristic_cost) if total_heuristic_cost != math.inf else math.inf

