import heapq
from collections import deque
from fnmatch import fnmatch
import math # Use math.inf for infinity

# Try to import the base class from the expected location.
# If not found, define a dummy base class for standalone execution.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

# Note: The match function provided in the examples is not needed if parsing is done carefully.
# We will parse based on predicate names directly.

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all goals
    by summing the estimated costs for achieving each unmet goal independently.
    It considers navigation, sampling (soil/rock), imaging (calibration, taking image),
    and communication steps. The estimation prioritizes using the rover that can
    achieve a specific goal sub-task with the minimum estimated action cost.

    # Assumptions
    - The heuristic calculates the cost for each goal independently, ignoring potential
      positive or negative interactions (e.g., one rover performing multiple tasks
      in sequence, or resource conflicts like camera calibration being consumed).
    - Navigation costs are based on precomputed shortest paths for each rover,
      considering `can_traverse` and `visible` static predicates.
    - If a rover's store is full and it needs to sample, a `drop` action (cost 1)
      is added.
    - If an image needs to be taken and the required camera is not calibrated,
      the cost includes navigating to a calibration point, calibrating, navigating
      to the imaging point, and taking the image.
    - Communication requires navigating to a waypoint visible from the lander.
    - If multiple rovers/cameras/waypoints can achieve a sub-task, the one
      with the minimum estimated cost is chosen for the heuristic calculation
      for that specific goal.
    - If a goal appears impossible to achieve from the current state based on
      static capabilities and available samples/targets (e.g., no equipped rover,
      no path, sample already taken), a large cost (1000) is assigned for that goal component.

    # Heuristic Initialization
    - Parses the task's static facts to build internal data structures:
        - Rover capabilities (equipment, stores).
        - Camera properties (on-board rover, supported modes, calibration targets).
        - Waypoint connectivity (`can_traverse`, `visible`).
        - Visibility information (`visible_from` for objectives, `visible` for lander communication).
        - Lander location.
    - Precomputes all-pairs shortest path distances for each rover using BFS on the
      static waypoint graph defined by `can_traverse` and `visible`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current state to determine:
        - Rover locations.
        - Store status (empty/full).
        - Calibrated cameras.
        - Samples/images currently held by rovers.
        - Data already communicated.
        - Available samples at waypoints.
    2.  Initialize `total_heuristic_cost = 0`.
    3.  Iterate through each goal predicate in the task's goals.
    4.  If the goal is already satisfied in the current state, continue to the next goal.
    5.  If the goal is unmet, estimate the minimum cost to achieve *this specific goal*:
        a.  **For `(communicated_soil_data ?w)`:**
            i.   Check if any rover `r` currently has `(have_soil_analysis ?r ?w)`.
                 - If yes, find the minimum cost for such a rover `r` to navigate from its current location `rx` to a communication waypoint `vx` and communicate: `min(dist(rx, vx)) + 1`.
            ii.  If no rover has the analysis, check if `(at_soil_sample ?w)` exists.
                 - If yes, find the minimum cost among all rovers `r` equipped for soil analysis: `cost = (1 if store full else 0) + dist(rx, w) + 1 (sample) + min(dist(w, vx)) + 1 (communicate)`.
            iii. If no rover has analysis and no sample exists at `w`, the cost for this path is infinity (handled by large penalty).
            iv.  Select the minimum cost found across steps i and ii. Add penalty if minimum is infinity.
        b.  **For `(communicated_rock_data ?w)`:** Similar logic as soil data.
        c.  **For `(communicated_image_data ?o ?m)`:**
            i.   Check if any rover `r` currently has `(have_image ?r ?o ?m)`.
                 - If yes, find the minimum cost for such a rover `r` to navigate from `rx` to `vx` and communicate: `min(dist(rx, vx)) + 1`.
            ii.  If no rover has the image, find the minimum cost among all rovers `r` equipped for imaging with a suitable camera `i` supporting mode `m`:
                 - Determine if calibration is needed for camera `i`.
                 - Find the best calibration waypoint `wc` (if needed) and imaging waypoint `wp`.
                 - Calculate cost: `[dist(rx, wc) + 1 (calibrate)] + dist(wc_or_rx, wp) + 1 (take_image) + min(dist(wp, vx)) + 1 (communicate)`. The calibration part is only added if needed.
                 - Minimize this cost over all valid `r`, `i`, `wc`, `wp`.
            iii. Select the minimum cost found across steps i and ii. Add penalty if minimum is infinity.
        d.  Handle cases where paths don't exist or prerequisites aren't met by assigning infinite cost for that path/rover, ensuring the minimum selects a feasible option if one exists.
    6.  Add the minimum estimated cost (or penalty) for the current goal to `total_heuristic_cost`.
    7.  Return `total_heuristic_cost` (capped at a large value). Ensure 0 only if state is goal.
    """
    def __init__(self, task):
        super().__init__(task) # Initialize base class if needed
        self.goals = task.goals
        static_facts = task.static
        self.infinity = math.inf
        self.large_penalty = 1000 # Cost assigned when a goal seems impossible
        self.max_heuristic_value = 1000000 # Cap the heuristic value

        # --- Parse Static Facts ---
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.stores = set()
        self.lander = None
        self.lander_location = None
        self.rover_equipment = {} # rover -> set('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.store_of_rover = {} # store -> rover
        self.camera_info = {} # camera -> {'on_board': rover, 'supports': set(modes), 'calibration_target': objective}
        self.calibration_targets = {} # camera -> objective
        self.on_board_cameras = {} # rover -> set(camera)
        self.camera_supports = {} # camera -> set(mode)
        self.visible_from_objective = {} # objective -> set(waypoint)
        self.waypoint_visibility = {} # waypoint -> set(waypoint) # Stores w2 for (visible w1 w2)
        self.can_traverse_map = {} # rover -> waypoint -> set(waypoint) # Stores w2 for (can_traverse r w1 w2)

        # Extract objects and types implicitly from predicates
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            try: # Robust parsing
                if pred == 'rover': self.rovers.add(parts[1])
                elif pred == 'waypoint': self.waypoints.add(parts[1])
                elif pred == 'camera': self.cameras.add(parts[1])
                elif pred == 'objective': self.objectives.add(parts[1])
                elif pred == 'mode': self.modes.add(parts[1])
                elif pred == 'store': self.stores.add(parts[1])
                elif pred == 'lander': self.lander = parts[1]
                elif pred == 'at_lander':
                    self.lander = parts[1]
                    self.lander_location = parts[2]
                    self.waypoints.add(parts[2]) # Ensure lander location is known waypoint
                elif pred == 'equipped_for_soil_analysis':
                    self.rover_equipment.setdefault(parts[1], set()).add('soil')
                    self.rovers.add(parts[1])
                elif pred == 'equipped_for_rock_analysis':
                    self.rover_equipment.setdefault(parts[1], set()).add('rock')
                    self.rovers.add(parts[1])
                elif pred == 'equipped_for_imaging':
                    self.rover_equipment.setdefault(parts[1], set()).add('imaging')
                    self.rovers.add(parts[1])
                elif pred == 'store_of':
                    s, r = parts[1], parts[2]
                    self.rover_stores[r] = s
                    self.store_of_rover[s] = r
                    self.stores.add(s)
                    self.rovers.add(r)
                elif pred == 'on_board':
                    c, r = parts[1], parts[2]
                    self.on_board_cameras.setdefault(r, set()).add(c)
                    self.camera_info.setdefault(c, {})['on_board'] = r
                    self.cameras.add(c)
                    self.rovers.add(r)
                elif pred == 'supports':
                    c, m = parts[1], parts[2]
                    self.camera_supports.setdefault(c, set()).add(m)
                    self.camera_info.setdefault(c, {}).setdefault('supports', set()).add(m)
                    self.cameras.add(c)
                    self.modes.add(m)
                elif pred == 'calibration_target':
                    c, o = parts[1], parts[2]
                    self.calibration_targets[c] = o
                    self.camera_info.setdefault(c, {})['calibration_target'] = o
                    self.cameras.add(c)
                    self.objectives.add(o)
                elif pred == 'visible_from':
                    o, w = parts[1], parts[2]
                    self.visible_from_objective.setdefault(o, set()).add(w)
                    self.objectives.add(o)
                    self.waypoints.add(w)
                elif pred == 'visible':
                    w1, w2 = parts[1], parts[2]
                    self.waypoint_visibility.setdefault(w1, set()).add(w2)
                    self.waypoints.add(w1)
                    self.waypoints.add(w2)
                elif pred == 'can_traverse':
                    r, w1, w2 = parts[1], parts[2], parts[3]
                    self.can_traverse_map.setdefault(r, {}).setdefault(w1, set()).add(w2)
                    self.rovers.add(r)
                    self.waypoints.add(w1)
                    self.waypoints.add(w2)
            except IndexError:
                 # Ignore malformed static facts
                 # print(f"Warning: Malformed static fact ignored: {fact}")
                 pass

        # --- Precompute Shortest Paths ---
        self.distances = {} # rover -> src_wp -> dest_wp -> distance
        for r in self.rovers:
            self.distances[r] = {}
            adj = {} # Adjacency list for this rover's traversable graph
            for w1 in self.waypoints:
                 adj[w1] = set()
                 # Check can_traverse for rover r
                 can_traverse_from_w1 = self.can_traverse_map.get(r, {}).get(w1, set())
                 # Check visibility constraint for navigation
                 visible_from_w1 = self.waypoint_visibility.get(w1, set())
                 for w2 in can_traverse_from_w1:
                     if w2 in visible_from_w1:
                         adj[w1].add(w2)

            # Run BFS from each waypoint for this rover
            for start_node in self.waypoints:
                self.distances[r][start_node] = {wp: self.infinity for wp in self.waypoints}
                # Check if start_node is a valid waypoint before proceeding
                if start_node not in self.waypoints: continue

                self.distances[r][start_node][start_node] = 0
                queue = deque([start_node])
                visited = {start_node}

                while queue:
                    current_wp = queue.popleft()
                    # Ensure current_wp is valid before accessing distances
                    if current_wp not in self.distances[r][start_node]: continue
                    current_dist = self.distances[r][start_node][current_wp]

                    # Use the pre-built adjacency list 'adj' for neighbors
                    for neighbor_wp in adj.get(current_wp, set()):
                        if neighbor_wp not in visited:
                            # Ensure neighbor_wp is valid before adding to queue/visited
                            if neighbor_wp in self.waypoints:
                                visited.add(neighbor_wp)
                                self.distances[r][start_node][neighbor_wp] = current_dist + 1
                                queue.append(neighbor_wp)

        # --- Precompute Communication Waypoints ---
        self.comm_locations = set()
        if self.lander_location:
            # Find waypoints W such that (visible W lander_location) is true
            for w, visible_wps in self.waypoint_visibility.items():
                if self.lander_location in visible_wps:
                    self.comm_locations.add(w)
            # The condition is (visible ?x ?y) where rover is at ?x, lander at ?y.
            # So we need waypoints 'x' such that 'x' can see 'y' (lander_location).

    def _get_dist(self, rover, wp1, wp2):
        """Safely get precomputed distance."""
        # Check if rover and waypoints exist in the precomputed structure
        if rover not in self.distances or wp1 not in self.distances[rover] or wp2 not in self.distances[rover][wp1]:
             return self.infinity
        dist = self.distances[rover][wp1][wp2]
        return dist # Already returns infinity if unreachable

    def _get_min_dist_to_comm(self, rover, current_wp):
        """Find minimum distance from current_wp to any comm location for the rover."""
        min_d = self.infinity
        if not self.comm_locations: # Cannot communicate if no comm locations exist
            return self.infinity
        # Ensure current_wp is valid before calculating distances from it
        if current_wp not in self.waypoints:
            return self.infinity

        for comm_wp in self.comm_locations:
            d = self._get_dist(rover, current_wp, comm_wp)
            min_d = min(min_d, d)
        return min_d


    def __call__(self, node):
        state = node.state
        heuristic_value = 0

        # --- Parse Current State ---
        rover_locations = {} # rover -> waypoint
        store_states = {s: 'empty' for s in self.stores} # store -> 'empty'/'full'
        calibrated_cameras = set() # (camera, rover)
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)
        available_soil = set() # waypoint
        available_rock = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            try: # Add error handling for robustness
                if pred == 'at' and parts[1] in self.rovers: rover_locations[parts[1]] = parts[2]
                elif pred == 'full' and parts[1] in self.stores: store_states[parts[1]] = 'full'
                elif pred == 'calibrated' and parts[1] in self.cameras and parts[2] in self.rovers: calibrated_cameras.add((parts[1], parts[2]))
                elif pred == 'have_soil_analysis' and parts[1] in self.rovers and parts[2] in self.waypoints: have_soil.add((parts[1], parts[2]))
                elif pred == 'have_rock_analysis' and parts[1] in self.rovers and parts[2] in self.waypoints: have_rock.add((parts[1], parts[2]))
                elif pred == 'have_image' and parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes: have_image.add((parts[1], parts[2], parts[3]))
                elif pred == 'communicated_soil_data' and parts[1] in self.waypoints: communicated_soil.add(parts[1])
                elif pred == 'communicated_rock_data' and parts[1] in self.waypoints: communicated_rock.add(parts[1])
                elif pred == 'communicated_image_data' and parts[1] in self.objectives and parts[2] in self.modes: communicated_image.add((parts[1], parts[2]))
                elif pred == 'at_soil_sample' and parts[1] in self.waypoints: available_soil.add(parts[1])
                elif pred == 'at_rock_sample' and parts[1] in self.waypoints: available_rock.add(parts[1])
            except IndexError:
                 # Handle malformed facts gracefully if necessary
                 # print(f"Warning: Malformed fact encountered in state: {fact}")
                 pass


        # --- Evaluate Unmet Goals ---
        num_unmet_goals = 0
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            goal_achieved = False
            min_cost_for_goal = self.infinity

            # --- Check if goal is already achieved ---
            try: # Add error handling
                if pred == 'communicated_soil_data':
                    w = parts[1]
                    if w in communicated_soil: goal_achieved = True
                elif pred == 'communicated_rock_data':
                    w = parts[1]
                    if w in communicated_rock: goal_achieved = True
                elif pred == 'communicated_image_data':
                    o, m = parts[1], parts[2]
                    if (o, m) in communicated_image: goal_achieved = True
            except IndexError:
                 # Malformed goal fact
                 goal_achieved = False # Assume not achieved

            if goal_achieved:
                continue

            num_unmet_goals += 1
            # --- Estimate cost for unmet goal ---
            if pred == 'communicated_soil_data':
                w = parts[1]
                # Option 1: A rover already has the analysis
                rovers_with_analysis = {r for r, wp in have_soil if wp == w}
                cost1 = self.infinity
                if rovers_with_analysis:
                    for r in rovers_with_analysis:
                        if r not in rover_locations: continue
                        rx = rover_locations[r]
                        cost_nav_comm = self._get_min_dist_to_comm(r, rx)
                        if cost_nav_comm != self.infinity:
                            cost1 = min(cost1, cost_nav_comm + 1) # navigate + communicate

                # Option 2: Need to sample first
                cost2 = self.infinity
                if w in available_soil:
                    for r in self.rovers:
                        if 'soil' not in self.rover_equipment.get(r, set()): continue
                        if r not in rover_locations: continue
                        rx = rover_locations[r]
                        store = self.rover_stores.get(r)
                        if not store: continue

                        cost_drop = 1 if store_states.get(store) == 'full' else 0
                        cost_nav1 = self._get_dist(r, rx, w)
                        if cost_nav1 == self.infinity: continue

                        cost_sample = 1
                        cost_nav2 = self._get_min_dist_to_comm(r, w)
                        if cost_nav2 == self.infinity: continue

                        cost_comm = 1
                        current_rover_cost = cost_drop + cost_nav1 + cost_sample + cost_nav2 + cost_comm
                        cost2 = min(cost2, current_rover_cost)

                min_cost_for_goal = min(cost1, cost2)

            elif pred == 'communicated_rock_data':
                w = parts[1]
                rovers_with_analysis = {r for r, wp in have_rock if wp == w}
                cost1 = self.infinity
                if rovers_with_analysis:
                    for r in rovers_with_analysis:
                        if r not in rover_locations: continue
                        rx = rover_locations[r]
                        cost_nav_comm = self._get_min_dist_to_comm(r, rx)
                        if cost_nav_comm != self.infinity:
                            cost1 = min(cost1, cost_nav_comm + 1)

                cost2 = self.infinity
                if w in available_rock:
                    for r in self.rovers:
                        if 'rock' not in self.rover_equipment.get(r, set()): continue
                        if r not in rover_locations: continue
                        rx = rover_locations[r]
                        store = self.rover_stores.get(r)
                        if not store: continue

                        cost_drop = 1 if store_states.get(store) == 'full' else 0
                        cost_nav1 = self._get_dist(r, rx, w)
                        if cost_nav1 == self.infinity: continue

                        cost_sample = 1
                        cost_nav2 = self._get_min_dist_to_comm(r, w)
                        if cost_nav2 == self.infinity: continue

                        cost_comm = 1
                        current_rover_cost = cost_drop + cost_nav1 + cost_sample + cost_nav2 + cost_comm
                        cost2 = min(cost2, current_rover_cost)

                min_cost_for_goal = min(cost1, cost2)

            elif pred == 'communicated_image_data':
                o, m = parts[1], parts[2]
                rovers_with_image = {r for r, obj, mode in have_image if obj == o and mode == m}
                cost1 = self.infinity
                if rovers_with_image:
                    for r in rovers_with_image:
                        if r not in rover_locations: continue
                        rx = rover_locations[r]
                        cost_nav_comm = self._get_min_dist_to_comm(r, rx)
                        if cost_nav_comm != self.infinity:
                            cost1 = min(cost1, cost_nav_comm + 1)

                cost2 = self.infinity
                possible_imaging_wps = self.visible_from_objective.get(o, set())
                if not possible_imaging_wps:
                     min_cost_for_goal = cost1 # cost2 remains infinity
                else:
                    for r in self.rovers:
                        if 'imaging' not in self.rover_equipment.get(r, set()): continue
                        if r not in rover_locations: continue
                        rx = rover_locations[r]

                        for cam in self.on_board_cameras.get(r, set()):
                            if m not in self.camera_supports.get(cam, set()): continue

                            calib_target = self.calibration_targets.get(cam)
                            if not calib_target: continue

                            possible_calib_wps = self.visible_from_objective.get(calib_target, set())
                            if not possible_calib_wps: continue

                            needs_calibration = (cam, r) not in calibrated_cameras
                            min_sequence_cost_for_rover_cam = self.infinity

                            for wp in possible_imaging_wps: # Iterate imaging waypoints
                                cost_nav_image = 0
                                cost_take_image = 1
                                loc_before_image_nav = rx
                                cost_calibrate = 0
                                cost_nav_calibrate = 0

                                if needs_calibration:
                                    best_wc_cost = self.infinity
                                    best_wc = None
                                    for wc in possible_calib_wps: # Find best calibration waypoint
                                        # Ensure wc is a valid waypoint before getting distance
                                        if wc not in self.waypoints: continue
                                        d = self._get_dist(r, rx, wc)
                                        if d < best_wc_cost: # Use < to ensure finite distance selected
                                            best_wc_cost = d
                                            best_wc = wc

                                    if best_wc is None: continue # Cannot reach any calibration spot for this wp
                                    cost_nav_calibrate = best_wc_cost
                                    cost_calibrate = 1
                                    loc_before_image_nav = best_wc
                                else:
                                    loc_before_image_nav = rx

                                # Ensure wp is valid before getting distance
                                if wp not in self.waypoints: continue
                                cost_nav_image = self._get_dist(r, loc_before_image_nav, wp)
                                if cost_nav_image == self.infinity: continue

                                cost_nav_comm = self._get_min_dist_to_comm(r, wp)
                                if cost_nav_comm == self.infinity: continue

                                cost_communicate = 1
                                total_sequence_cost = (cost_nav_calibrate + cost_calibrate +
                                                       cost_nav_image + cost_take_image +
                                                       cost_nav_comm + cost_communicate)
                                min_sequence_cost_for_rover_cam = min(min_sequence_cost_for_rover_cam, total_sequence_cost)

                            cost2 = min(cost2, min_sequence_cost_for_rover_cam) # Update cost2 with best for this rover/cam

                    min_cost_for_goal = min(cost1, cost2)

            # Add cost for this goal
            if min_cost_for_goal == self.infinity:
                 heuristic_value += self.large_penalty
            else:
                 # Add check for infinity just in case something went wrong
                 if min_cost_for_goal != self.infinity:
                     heuristic_value += min_cost_for_goal
                 else: # Should not happen based on above check, but as fallback
                     heuristic_value += self.large_penalty


        # Final checks
        if num_unmet_goals == 0:
            # Double check if the state actually satisfies all goals
            is_goal_state = all(gf in state for gf in self.goals)
            return 0 if is_goal_state else 1 # Return 0 only if truly a goal state

        elif heuristic_value == 0:
             # If goals are unmet but heuristic is 0, return 1
             return 1
        else:
             # Cap the heuristic value and return integer
             return int(min(round(heuristic_value), self.max_heuristic_value))
