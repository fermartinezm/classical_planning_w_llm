import heapq
import collections # For deque in BFS
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this path is correct
import math # For infinity

# Helper function to parse PDDL facts more robustly
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    fact = fact.strip()
    if not fact.startswith("(") or not fact.endswith(")"):
        # Handle potential malformed facts or non-facts if necessary
        return []
    # Split considering potential multiple spaces
    return " ".join(fact[1:-1].split()).split()

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all
    unsatisfied goals in the current state. It decomposes the problem by
    calculating the estimated cost for each individual unsatisfied goal
    (communicating soil data, rock data, or image data) and summing these costs.
    The cost for each goal involves estimating the actions needed: navigation,
    sampling/imaging, calibration (if needed), dropping samples (if store is full),
    and communication. It aims to find the minimum cost sequence for each goal
    by considering the best available rover and locations.

    # Assumptions
    - The heuristic assumes that the rover chosen for a sub-task (like sampling
      or imaging) will also be the one to communicate the data.
    - Navigation costs are estimated using precomputed shortest path distances
      on the waypoint graph. The graph connectivity considers both `can_traverse`
      (for any rover) and `visible` predicates. Each navigation step costs 1.
    - The cost calculation for each goal is done independently, potentially
      overcounting actions if multiple goals can be achieved sequentially by
      the same rover without extra navigation, or undercounting if resource
      conflicts (like needing an empty store or recalibration) arise between
      achieving different goals.
    - If a required sample (soil/rock) is depleted from its original location,
      the heuristic assumes the corresponding goal cannot be achieved by sampling
      again at that location (it must rely on data already collected).
    - If a goal is fundamentally impossible from the current state (e.g., no
      equipped rover, no path, sample depleted and not held), it contributes a
      very large value (effectively infinity) to the total cost.

    # Heuristic Initialization
    - Extracts static information: rover equipment, camera capabilities,
      visibility between waypoints, objective visibility, calibration targets,
      lander location, store mappings.
    - Identifies all waypoints, rovers, cameras, objectives, modes, landers, stores
      from static facts.
    - Builds a waypoint graph based on `can_traverse` (for any rover) and `visible` predicates.
    - Precomputes all-pairs shortest path distances between waypoints using BFS
      on the waypoint graph. Stores these distances.
    - Identifies waypoints suitable for communication (visible from the lander).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current state to determine rover locations, store status (full/empty),
        camera calibration status, collected data (`have_soil`, `have_rock`,
        `have_image`), and currently available samples (`at_soil_sample`, `at_rock_sample`).
    2.  Initialize `total_heuristic_cost = 0`.
    3.  Iterate through each goal predicate defined in the task's goals (`self.goals`).
    4.  If a goal predicate is already present in the current state (`state`), skip it (cost is 0).
    5.  For an unsatisfied goal:
        a.  **Identify Goal Type:** Determine if it's `communicated_soil_data`,
            `communicated_rock_data`, or `communicated_image_data`. Extract target parameters (waypoint `w`, objective `o`, mode `m`).
        b.  **Find Minimum Cost:** Initialize `min_goal_cost = infinity`. Iterate through *all* rovers (`r`) to find the minimum cost for *this specific goal*.
        c.  **Cost Calculation (Per Rover `r`):**
            i.   Initialize `current_rover_cost = infinity`.
            ii.  **Check if Data Already Held:** If rover `r` already holds the required data (`have_soil_analysis(r, w)`, `have_rock_analysis(r, w)`, `have_image(r, o, m)`):
                 - Find the minimum navigation cost from the rover's current location (`rover_loc`) to any communication waypoint (`comm_wp`). Let this be `dist_to_comm`.
                 - If reachable (`dist_to_comm` is finite), cost = `dist_to_comm + 1` (navigate + communicate). Update `current_rover_cost = min(current_rover_cost, cost)`.
            iii. **If Data Not Held (Sampling - Soil/Rock):** If the goal is soil/rock data, check if rover `r` is equipped, and if the sample `w` is currently available (`at_soil/rock_sample w`):
                 - Check if rover `r`'s store is full (`drop_cost = 1` if full, else `0`).
                 - Calculate navigation cost from `rover_loc` to sample waypoint `w` (`dist_to_sample`).
                 - Calculate minimum navigation cost from `w` to any `comm_wp` (`dist_sample_to_comm`).
                 - If all paths are reachable, cost = `dist_to_sample + drop_cost + 1 (sample) + dist_sample_to_comm + 1 (communicate)`. Update `current_rover_cost`.
            iv.  **If Data Not Held (Imaging):** If the goal is image data, check if rover `r` is equipped for imaging:
                 - Find the best (closest reachable) waypoint `p` from which objective `o` is visible (`visible_from o p`). Let distance be `dist_to_image_loc`. If none reachable, skip rover.
                 - Iterate through cameras `i` on rover `r` that support mode `m`.
                 - For each suitable camera `i`:
                     - Check if `(i, r)` is calibrated.
                     - Calculate minimum navigation cost from `p` to any `comm_wp` (`dist_image_to_comm`). If unreachable, skip camera.
                     - **If Calibrated:** Cost = `dist_to_image_loc + 1 (take_image) + dist_image_to_comm + 1 (communicate)`. Update `current_rover_cost`.
                     - **If Not Calibrated:**
                         - Find calibration target `t` for camera `i`. If none, skip camera.
                         - Find the best (closest reachable) waypoint `w_cal` from which `t` is visible (`visible_from t w_cal`). Let distance be `dist_to_cal_loc`. If none reachable, skip camera.
                         - Calculate navigation cost from `w_cal` to `p` (`dist_cal_to_image`). If unreachable, skip camera.
                         - Cost = `dist_to_cal_loc + 1 (calibrate) + dist_cal_to_image + 1 (take_image) + dist_image_to_comm + 1 (communicate)`. Update `current_rover_cost`.
        d.  **Update Minimum Goal Cost:** After checking all possibilities for rover `r`, update `min_goal_cost = min(min_goal_cost, current_rover_cost)`.
        e.  **Add Goal Cost to Total:** After checking all rovers for the current goal, add `min_goal_cost` to `total_heuristic_cost`. If `min_goal_cost` remained infinity, the goal is considered unreachable; return a very large number immediately.
    6.  Return `total_heuristic_cost`. If the state is a goal state (`self.goals <= state`), return 0. Ensure non-goal states have a cost of at least 1 if the calculation yields 0.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        self.infinity = float('inf')
        # Using a large number instead of infinity for practical purposes in search
        self.large_cost = 1_000_000_000

        # --- Initialize Data Structures ---
        self.waypoints = set()
        self.rovers = set()
        self.cameras = set()
        self.objectives = set()
        self.modes = set()
        self.landers = set()
        self.stores = set()

        self.rover_equipment = {} # rover -> set('soil', 'rock', 'imaging')
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_supports = {} # camera -> set(mode)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visibility = {} # objective -> set(waypoint)
        self.calibration_target_visibility = {} # objective -> set(waypoint) (Populated later)
        self.lander_location = None
        self.comm_links = set() # set((wp1, wp2)) where visible(wp1, wp2)
        self.store_map = {} # store -> rover
        self.rover_store = {} # rover -> store

        # Graph related
        self.waypoint_graph = {} # waypoint -> list(neighbor_wp)
        self.distances = {} # dist[wp1][wp2] -> shortest distance

        # Temporary structures for parsing
        temp_can_traverse = {} # rover -> set((wp_from, wp_to))
        visible_pairs = set()
        all_objects = set() # Track all named objects to identify types

        # --- Parse Static Facts ---
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]
            # Add all mentioned objects to a set for potential type inference later
            for arg in args: all_objects.add(arg)

            # Process specific predicates and identify object types
            # Assume types are declared or inferrable from predicate names/args
            if pred == 'rover': self.rovers.add(args[0])
            elif pred == 'waypoint': self.waypoints.add(args[0])
            elif pred == 'camera': self.cameras.add(args[0])
            elif pred == 'objective': self.objectives.add(args[0])
            elif pred == 'mode': self.modes.add(args[0])
            elif pred == 'lander': self.landers.add(args[0])
            elif pred == 'store': self.stores.add(args[0])

            if pred == 'equipped_for_soil_analysis':
                r = args[0]; self.rovers.add(r)
                self.rover_equipment.setdefault(r, set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                r = args[0]; self.rovers.add(r)
                self.rover_equipment.setdefault(r, set()).add('rock')
            elif pred == 'equipped_for_imaging':
                r = args[0]; self.rovers.add(r)
                self.rover_equipment.setdefault(r, set()).add('imaging')
            elif pred == 'on_board':
                cam, r = args[0], args[1]; self.cameras.add(cam); self.rovers.add(r)
                self.rover_cameras.setdefault(r, set()).add(cam)
            elif pred == 'store_of':
                store, r = args[0], args[1]; self.stores.add(store); self.rovers.add(r)
                self.store_map[store] = r
                self.rover_store[r] = store
            elif pred == 'supports':
                cam, mode = args[0], args[1]; self.cameras.add(cam); self.modes.add(mode)
                self.camera_supports.setdefault(cam, set()).add(mode)
            elif pred == 'calibration_target':
                cam, obj = args[0], args[1]; self.cameras.add(cam); self.objectives.add(obj)
                self.camera_calibration_target[cam] = obj
            elif pred == 'visible':
                wp1, wp2 = args[0], args[1]; self.waypoints.add(wp1); self.waypoints.add(wp2)
                self.comm_links.add((wp1, wp2))
                visible_pairs.add((wp1, wp2))
                self.waypoint_graph.setdefault(wp1, []) # Ensure nodes exist
                self.waypoint_graph.setdefault(wp2, [])
            elif pred == 'visible_from':
                obj, wp = args[0], args[1]; self.objectives.add(obj); self.waypoints.add(wp)
                self.objective_visibility.setdefault(obj, set()).add(wp)
            elif pred == 'at_lander':
                l, wp = args[0], args[1]; self.landers.add(l); self.waypoints.add(wp)
                self.lander_location = wp # Assuming one lander's location matters
            elif pred == 'can_traverse':
                r, wp_from, wp_to = args[0], args[1], args[2]
                self.rovers.add(r); self.waypoints.add(wp_from); self.waypoints.add(wp_to)
                temp_can_traverse.setdefault(r, set()).add((wp_from, wp_to))

        # Populate calibration target visibility map using objective visibility data
        for cam, target_obj in self.camera_calibration_target.items():
            if target_obj in self.objective_visibility:
                 self.calibration_target_visibility[target_obj] = self.objective_visibility[target_obj]

        # --- Build Waypoint Graph ---
        # An edge (wp1 -> wp2) exists if visible(wp1, wp2) AND exists rover r: can_traverse(r, wp1, wp2)
        for wp1 in self.waypoints:
            self.waypoint_graph.setdefault(wp1, []) # Ensure all waypoints are keys
            for wp2 in self.waypoints:
                 if wp1 != wp2 and (wp1, wp2) in visible_pairs:
                     # Check if *any* rover can traverse this specific visible link
                     can_traverse_exists = any(
                         r in temp_can_traverse and (wp1, wp2) in temp_can_traverse[r]
                         for r in self.rovers
                     )
                     if can_traverse_exists:
                         self.waypoint_graph[wp1].append(wp2)

        # --- Compute All-Pairs Shortest Paths (BFS) ---
        for start_node in self.waypoints:
            # Initialize distances for this start_node
            self.distances[start_node] = {wp: self.infinity for wp in self.waypoints}
            # Skip if start_node somehow isn't in the identified waypoints
            if start_node not in self.waypoints: continue
            self.distances[start_node][start_node] = 0

            # Use deque for efficient BFS queue operations
            queue = collections.deque([(0, start_node)])

            while queue:
                dist, current_wp = queue.popleft()

                # If current_wp has no outgoing edges in the graph, skip
                if current_wp not in self.waypoint_graph: continue

                for neighbor in self.waypoint_graph[current_wp]:
                    new_dist = dist + 1
                    # If we found a shorter path to the neighbor, update and enqueue
                    if new_dist < self.distances[start_node][neighbor]:
                         self.distances[start_node][neighbor] = new_dist
                         queue.append((new_dist, neighbor))

        # Find waypoints visible from lander for communication
        self.comm_waypoints = set()
        if self.lander_location:
            # Communication requires rover at x, lander at y, visible(x,y)
            # The action implies rover needs to be at a waypoint visible from lander's waypoint
            for wp_comm in self.waypoints:
                 # Check visibility in both directions based on visible_pairs set
                 if (wp_comm, self.lander_location) in visible_pairs or \
                    (self.lander_location, wp_comm) in visible_pairs:
                     self.comm_waypoints.add(wp_comm)


    def _get_shortest_dist(self, wp_from, wp_to):
        """ Safely get shortest distance, return infinity if no path or invalid wp. """
        if wp_from not in self.waypoints or wp_to not in self.waypoints:
            return self.infinity
        if wp_from == wp_to:
            return 0
        # Access precomputed distances safely
        return self.distances.get(wp_from, {}).get(wp_to, self.infinity)


    def _get_min_dist_to_set(self, current_wp, target_wps):
        """ Find minimum distance from current_wp to any waypoint in target_wps set. """
        if not target_wps or current_wp not in self.waypoints:
            return self.infinity
        min_d = self.infinity
        for target_wp in target_wps:
            # Ensure target_wp is valid before calculating distance
            if target_wp in self.waypoints:
                min_d = min(min_d, self._get_shortest_dist(current_wp, target_wp))
        return min_d


    def __call__(self, node):
        state = node.state
        cost = 0

        # Check for goal state first
        if self.goals <= state:
             return 0

        # --- Parse Current State ---
        rover_locations = {}
        rover_stores_full = set() # Stores that are full (by store name)
        calibrated_cameras = set() # (camera, rover) pairs
        have_soil = set() # (rover, waypoint) pairs
        have_rock = set() # (rover, waypoint) pairs
        have_image = set() # (rover, objective, mode) tuples
        current_soil_samples = set() # waypoint
        current_rock_samples = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            # Check predicate arity before accessing args
            if pred == 'at' and len(args) == 2 and args[0] in self.rovers:
                 rover_locations[args[0]] = args[1]
            elif pred == 'full' and len(args) == 1:
                 rover_stores_full.add(args[0])
            elif pred == 'calibrated' and len(args) == 2:
                 calibrated_cameras.add((args[0], args[1]))
            elif pred == 'have_soil_analysis' and len(args) == 2:
                 have_soil.add((args[0], args[1]))
            elif pred == 'have_rock_analysis' and len(args) == 2:
                 have_rock.add((args[0], args[1]))
            elif pred == 'have_image' and len(args) == 3:
                 have_image.add(tuple(args)) # Ensure tuple for set storage
            elif pred == 'at_soil_sample' and len(args) == 1:
                 current_soil_samples.add(args[0])
            elif pred == 'at_rock_sample' and len(args) == 1:
                 current_rock_samples.add(args[0])

        # --- Calculate Cost for Unsatisfied Goals ---
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already satisfied

            goal_parts = get_parts(goal_fact)
            if not goal_parts: continue # Skip malformed goals
            goal_pred = goal_parts[0]
            min_goal_cost = self.infinity

            # --- Goal: communicated_soil_data ?w ---
            if goal_pred == 'communicated_soil_data' and len(goal_parts) == 2:
                w = goal_parts[1]
                for r in self.rovers:
                    rover_loc = rover_locations.get(r)
                    if not rover_loc: continue # Rover not placed or location unknown

                    current_rover_cost = self.infinity
                    # Path 1: Already have data
                    if (r, w) in have_soil:
                        dist_to_comm = self._get_min_dist_to_set(rover_loc, self.comm_waypoints)
                        if dist_to_comm != self.infinity:
                             current_rover_cost = min(current_rover_cost, dist_to_comm + 1) # navigate + communicate

                    # Path 2: Need to sample
                    elif 'soil' in self.rover_equipment.get(r, set()) and w in current_soil_samples:
                        store = self.rover_store.get(r)
                        # Check if store exists and is full
                        is_full = store is not None and store in rover_stores_full
                        drop_cost = 1 if is_full else 0

                        dist_to_sample = self._get_shortest_dist(rover_loc, w)
                        dist_sample_to_comm = self._get_min_dist_to_set(w, self.comm_waypoints)

                        # Check if all required steps are possible (finite distances)
                        if dist_to_sample != self.infinity and dist_sample_to_comm != self.infinity:
                            cost_path2 = dist_to_sample + drop_cost + 1 + dist_sample_to_comm + 1
                            # navigate + drop? + sample + navigate + communicate
                            current_rover_cost = min(current_rover_cost, cost_path2)

                    min_goal_cost = min(min_goal_cost, current_rover_cost)

            # --- Goal: communicated_rock_data ?w ---
            elif goal_pred == 'communicated_rock_data' and len(goal_parts) == 2:
                w = goal_parts[1]
                for r in self.rovers:
                    rover_loc = rover_locations.get(r)
                    if not rover_loc: continue

                    current_rover_cost = self.infinity
                    if (r, w) in have_rock:
                        dist_to_comm = self._get_min_dist_to_set(rover_loc, self.comm_waypoints)
                        if dist_to_comm != self.infinity:
                            current_rover_cost = min(current_rover_cost, dist_to_comm + 1)
                    elif 'rock' in self.rover_equipment.get(r, set()) and w in current_rock_samples:
                        store = self.rover_store.get(r)
                        is_full = store is not None and store in rover_stores_full
                        drop_cost = 1 if is_full else 0
                        dist_to_sample = self._get_shortest_dist(rover_loc, w)
                        dist_sample_to_comm = self._get_min_dist_to_set(w, self.comm_waypoints)
                        if dist_to_sample != self.infinity and dist_sample_to_comm != self.infinity:
                            cost_path2 = dist_to_sample + drop_cost + 1 + dist_sample_to_comm + 1
                            current_rover_cost = min(current_rover_cost, cost_path2)
                    min_goal_cost = min(min_goal_cost, current_rover_cost)

            # --- Goal: communicated_image_data ?o ?m ---
            elif goal_pred == 'communicated_image_data' and len(goal_parts) == 3:
                o, m = goal_parts[1], goal_parts[2]
                for r in self.rovers:
                    rover_loc = rover_locations.get(r)
                    # Check if rover exists, has location, and is equipped for imaging
                    if not rover_loc or 'imaging' not in self.rover_equipment.get(r, set()):
                        continue

                    current_rover_cost = self.infinity
                    # Path 1: Already have image
                    if (r, o, m) in have_image:
                        dist_to_comm = self._get_min_dist_to_set(rover_loc, self.comm_waypoints)
                        if dist_to_comm != self.infinity:
                            current_rover_cost = min(current_rover_cost, dist_to_comm + 1)
                    else:
                        # Path 2: Need to take image
                        possible_cameras = self.rover_cameras.get(r, set())
                        potential_imaging_wps = self.objective_visibility.get(o, set())
                        # Skip if objective not visible from anywhere or rover has no cameras
                        if not potential_imaging_wps or not possible_cameras: continue

                        # Find the closest reachable imaging waypoint p
                        best_imaging_wp = None
                        min_dist_to_imaging_wp = self.infinity
                        for p in potential_imaging_wps:
                            dist = self._get_shortest_dist(rover_loc, p)
                            if dist < min_dist_to_imaging_wp:
                                min_dist_to_imaging_wp = dist
                                best_imaging_wp = p

                        # Skip if no imaging waypoint is reachable
                        if best_imaging_wp is None or min_dist_to_imaging_wp == self.infinity:
                             continue

                        # Check if communication is possible from the chosen imaging spot
                        dist_image_to_comm = self._get_min_dist_to_set(best_imaging_wp, self.comm_waypoints)
                        if dist_image_to_comm == self.infinity: continue

                        # Check cameras on this rover
                        for cam in possible_cameras:
                            # Check if camera supports the required mode
                            if m in self.camera_supports.get(cam, set()):
                                is_calibrated = (cam, r) in calibrated_cameras

                                if is_calibrated:
                                    # Path 2a: Calibrated -> Navigate, Take, Navigate, Comm
                                    cost_path2a = min_dist_to_imaging_wp + 1 + dist_image_to_comm + 1
                                    current_rover_cost = min(current_rover_cost, cost_path2a)
                                else:
                                    # Path 2b: Not Calibrated -> Need to calibrate first
                                    cal_target = self.camera_calibration_target.get(cam)
                                    # Skip if camera has no calibration target specified
                                    if not cal_target: continue

                                    potential_cal_wps = self.calibration_target_visibility.get(cal_target, set())
                                    # Skip if calibration target is not visible from anywhere
                                    if not potential_cal_wps: continue

                                    # Find closest reachable calibration waypoint w_cal
                                    best_cal_wp = None
                                    min_dist_to_cal_wp = self.infinity
                                    for w_cal in potential_cal_wps:
                                        dist = self._get_shortest_dist(rover_loc, w_cal)
                                        if dist < min_dist_to_cal_wp:
                                            min_dist_to_cal_wp = dist
                                            best_cal_wp = w_cal

                                    # Skip if no calibration waypoint is reachable
                                    if best_cal_wp is None or min_dist_to_cal_wp == self.infinity:
                                        continue

                                    # Check if path exists from calibration spot to imaging spot
                                    dist_cal_to_image = self._get_shortest_dist(best_cal_wp, best_imaging_wp)
                                    if dist_cal_to_image == self.infinity: continue

                                    # Path 2b Cost: Nav, Cal, Nav, Take, Nav, Comm
                                    cost_path2b = min_dist_to_cal_wp + 1 + dist_cal_to_image + 1 + dist_image_to_comm + 1
                                    current_rover_cost = min(current_rover_cost, cost_path2b)

                    # Update the minimum cost found for this rover for this goal
                    min_goal_cost = min(min_goal_cost, current_rover_cost)

            # Add the minimum cost found for this goal (across all rovers)
            if min_goal_cost == self.infinity:
                # If any goal is impossible for all rovers, the state is likely a dead end
                return self.large_cost
            cost += min_goal_cost

        # Final check: if cost is 0 but state is not goal, return 1.
        # This heuristic estimates actions needed, so 0 should only be for goal states.
        if cost == 0 and not (self.goals <= state):
             return 1

        # Return integer cost, handling potential floating point inaccuracies
        # and ensuring the large cost value is returned if infinity was reached.
        return int(round(cost)) if cost != self.infinity else self.large_cost
