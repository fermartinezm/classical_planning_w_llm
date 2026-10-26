import heapq
from collections import defaultdict
from fnmatch import fnmatch
# The Heuristic base class is assumed to be available in the environment.
# If not, uncomment the following lines:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    """
    # Remove parentheses and split by space
    return fact[1:-1].split()

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the total number of actions required to achieve all goals
    from the current state. It operates as a relaxed plan heuristic by decomposing
    the problem into subproblems, one for each unmet goal predicate. It estimates
    the minimum cost to achieve each unmet goal independently using the best available
    rover and sums these costs. This ignores negative interactions and resource
    contention, making it non-admissible but potentially informative for greedy search.

    # Assumptions
    - Navigation cost between waypoints is based on the shortest path in the
      directed graph defined by the `(visible ?w1 ?w2)` predicates. The cost of
      each `navigate` action is 1. The `(can_traverse ?r ?w1 ?w2)` predicate is
      ignored for simplicity, assuming visibility is the primary constraint or
      that `can_traverse` generally aligns with `visible`.
    - Sampling soil or rock requires the rover's store to be empty. If the best
      rover for a sampling task currently has a full store, an additional cost of 1
      (for the `drop` action) is added to the estimate for that goal.
    - Taking an image requires the camera to be calibrated. If the best rover/camera
      combination is not calibrated, the cost includes navigating to a suitable
      location and performing the `calibrate` action (cost +1). Taking an image
      uncalibrates the camera, but this effect is handled naturally as the state
      check for subsequent goals will reflect the uncalibrated status.
    - Communication actions require the rover to be at a waypoint visible from/to
      the lander. The heuristic calculates the cost to navigate to the *closest*
      such communication waypoint from the rover's location after acquiring the data.
    - Each goal is assigned to the best rover independently. This relaxation might
      underestimate the true cost if one rover is optimal for multiple tasks that
      interfere (e.g., needing calibration multiple times).

    # Heuristic Initialization
    - Stores the goal predicates from the task definition.
    - Parses static facts (`task.static`) to build internal data structures:
        - `waypoints`: A set of all waypoint names.
        - `lander_location`: The waypoint where the lander is located.
        - `visibility_adj`: A directed adjacency list representing `(visible ?w1 ?w2)`.
        - `rover_equipment`: Maps each rover to its capabilities ('soil', 'rock', 'imaging').
        - `rover_stores`: Maps each rover to its store object name.
        - `rover_cameras`: Maps each rover to a list of its cameras, including supported
          modes and calibration targets `[(camera, {modes}, cal_target_obj)]`.
        - `objective_visibility`: Maps each objective to the set of waypoints from which
          it is visible. Used for imaging and calibration target visibility.
    - Precomputes all-pairs shortest paths between waypoints using BFS on the
      visibility graph. Results are stored in `self.shortest_paths`.
    - Precomputes the minimum navigation cost from each waypoint to *any* waypoint
      suitable for communication (visible from/to the lander). Results are stored
      in `self.min_dist_to_comm`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Get the current state (`node.state`).
    2.  Check if the current state satisfies all goals (`self.goals <= state`). If yes, the
        heuristic value is 0.
    3.  Parse the current state to extract dynamic information:
        - `rover_locations`: Current waypoint for each rover.
        - `rover_stores_full`: Set of rovers whose store is currently full.
        - `have_soil`, `have_rock`, `have_image`: Sets of tuples representing data
          currently held by rovers.
        - `calibrated_cameras`: Set of `(camera, rover)` tuples for currently
          calibrated cameras.
    4.  Initialize `total_heuristic_cost = 0`.
    5.  Initialize `possible = True` (tracks if all goals seem reachable).
    6.  Iterate through each goal predicate in `self.goals`.
    7.  If the goal predicate is already present in the current `state`, skip to the next goal.
    8.  Initialize `min_goal_cost = float('inf')` for the current unmet goal.
    9.  **Determine Goal Type and Calculate `min_goal_cost`:**
        a.  **If `(communicated_soil_data ?w)`:**
            i.  Check rovers `r` that already `(have_soil_analysis r w)`. For each,
                calculate cost = `dist(at(r), closest_comm_wp) + 1`. Update `min_goal_cost`.
            ii. Check rovers `r` `(equipped_for_soil_analysis)`. For each not already
                holding the data, calculate cost = `(1 if store full) + dist(at(r), w) + 1 (sample) + dist(w, closest_comm_wp) + 1 (communicate)`. Update `min_goal_cost`.
        b.  **If `(communicated_rock_data ?w)`:** Similar to soil, using rock predicates.
        c.  **If `(communicated_image_data ?o ?m)`:**
            i.  Check rovers `r` that already `(have_image r o m)`. For each,
                calculate cost = `dist(at(r), closest_comm_wp) + 1`. Update `min_goal_cost`.
            ii. Check rovers `r` `(equipped_for_imaging)` with a camera `c` that is
                `on_board`, `supports` mode `m`. For each `(r, c)` pair not already
                holding the image:
                - Calculate calibration cost: If `(calibrated c r)` is false, find the
                  closest waypoint `cal_wp` reachable from `at(r)` where the camera's
                  `calibration_target` is visible. Cost = `dist(at(r), cal_wp) + 1`.
                  Update the rover's estimated location to `cal_wp`. If calibration is
                  impossible (no target, not visible, unreachable), skip this `(r, c)`.
                - Calculate imaging cost: Find the closest waypoint `img_wp` reachable
                  from the estimated location (after potential calibration) where objective
                  `o` is visible. Cost = `dist(estimated_loc, img_wp)`. Update estimated
                  location to `img_wp`. If impossible, skip.
                - Calculate communication cost: `dist(img_wp, closest_comm_wp) + 1`.
                  If impossible, skip.
                - Total sequence cost = `cal_cost + nav_to_img_cost + 1 (take_image) + nav_to_comm_cost + 1 (communicate)`.
                  Update `min_goal_cost`.
    10. After evaluating all options for the current goal, if `min_goal_cost` remains
        `float('inf')`, it means the heuristic deems this goal unreachable. Set
        `possible = False` and break the loop over goals.
    11. If `min_goal_cost` is finite, add it to `total_heuristic_cost`.
    12. After iterating through all goals, if `possible` is `True`, return
        `total_heuristic_cost`. Otherwise (if any goal was deemed impossible),
        return `float('inf')`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.waypoints = set()
        self.lander_location = None
        self.visibility_adj = defaultdict(set) # w1 -> {w2} if (visible w1 w2)
        self.rover_equipment = defaultdict(set) # rover -> {'soil', 'rock', 'imaging'}
        self.rover_stores = {} # rover -> store_name
        self.rover_cameras = defaultdict(list) # rover -> [(cam, {modes}, cal_target)]
        self.camera_supports = defaultdict(set) # camera -> {modes}
        self.calibration_targets = {} # camera -> objective
        self.objective_visibility = defaultdict(set) # objective -> {wp}
        self.calibration_visibility = defaultdict(set) # cal_target_obj -> {wp}
        self.all_rovers = set()
        self.all_cameras = set()
        self.all_objectives = set()

        # --- Parse Static Facts ---
        all_objects = set()
        potential_waypoints = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            all_objects.update(args)

            if pred == "at_lander":
                self.lander_location = args[1]
                potential_waypoints.add(args[1])
            elif pred == "visible":
                self.visibility_adj[args[0]].add(args[1])
                potential_waypoints.add(args[0])
                potential_waypoints.add(args[1])
            elif pred == "equipped_for_soil_analysis":
                self.rover_equipment[args[0]].add("soil")
                self.all_rovers.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.rover_equipment[args[0]].add("rock")
                self.all_rovers.add(args[0])
            elif pred == "equipped_for_imaging":
                self.rover_equipment[args[0]].add("imaging")
                self.all_rovers.add(args[0])
            elif pred == "store_of":
                self.rover_stores[args[1]] = args[0] # Map rover -> store
            elif pred == "supports":
                self.camera_supports[args[0]].add(args[1])
                self.all_cameras.add(args[0])
            elif pred == "calibration_target":
                self.calibration_targets[args[0]] = args[1]
                self.all_cameras.add(args[0])
                self.all_objectives.add(args[1])
            elif pred == "visible_from":
                self.objective_visibility[args[0]].add(args[1])
                self.all_objectives.add(args[0])
                potential_waypoints.add(args[1])

        # Define waypoints as objects involved in visibility or lander location
        self.waypoints = {obj for obj in all_objects if obj in potential_waypoints}

        # Populate rover_cameras using on_board info
        on_board_info = defaultdict(list)
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "on_board":
                 cam, rov = parts[1], parts[2]
                 # Check if rover and camera are known entities
                 if rov in self.all_rovers and cam in self.all_cameras:
                     on_board_info[rov].append(cam)

        for rover, cameras in on_board_info.items():
            for cam in cameras:
                 modes = self.camera_supports.get(cam, set())
                 cal_target = self.calibration_targets.get(cam, None)
                 if cal_target:
                     # Store waypoints where calibration target is visible
                     self.calibration_visibility[cal_target] = self.objective_visibility.get(cal_target, set())
                 self.rover_cameras[rover].append((cam, modes, cal_target))

        # --- Precompute Shortest Paths (BFS on visibility graph) ---
        self.shortest_paths = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for start_node in self.waypoints:
            self.shortest_paths[start_node][start_node] = 0
            queue = [(0, start_node)] # (distance, node)
            visited_dist = {start_node: 0}

            while queue:
                dist, current_node = heapq.heappop(queue)
                if dist > visited_dist[current_node]: continue

                for neighbor in self.visibility_adj.get(current_node, set()):
                    if neighbor in self.waypoints:
                        new_dist = dist + 1
                        if new_dist < visited_dist.get(neighbor, float('inf')):
                            visited_dist[neighbor] = new_dist
                            self.shortest_paths[start_node][neighbor] = new_dist
                            heapq.heappush(queue, (new_dist, neighbor))

        # --- Precompute Communication Info ---
        self.comm_waypoints = set()
        if self.lander_location and self.lander_location in self.waypoints:
             # Waypoints visible FROM the lander location
             for neighbor in self.visibility_adj.get(self.lander_location, set()):
                 if neighbor in self.waypoints: self.comm_waypoints.add(neighbor)
             # Waypoints FROM which the lander location is visible
             for w1 in self.waypoints:
                 if self.lander_location in self.visibility_adj.get(w1, set()):
                     self.comm_waypoints.add(w1)

        self.min_dist_to_comm = defaultdict(lambda: float('inf'))
        for wp in self.waypoints:
            min_d = float('inf')
            for comm_wp in self.comm_waypoints:
                dist_wp_to_comm = self.shortest_paths[wp].get(comm_wp, float('inf'))
                min_d = min(min_d, dist_wp_to_comm)
            self.min_dist_to_comm[wp] = min_d


    def _get_dist(self, wp1, wp2):
        """Safely get shortest path distance between two waypoints."""
        if wp1 is None or wp2 is None or wp1 not in self.waypoints or wp2 not in self.waypoints:
            return float('inf')
        return self.shortest_paths[wp1].get(wp2, float('inf'))

    def __call__(self, node):
        state = node.state

        # Goal check
        if self.goals <= state:
             return 0

        # --- Parse Current State ---
        rover_locations = {} # rover -> waypoint
        rover_stores_full = set() # rover names
        have_soil = set() # {(rover, waypoint)}
        have_rock = set() # {(rover, waypoint)}
        have_image = set() # {(rover, objective, mode)}
        calibrated_cameras = set() # {(camera, rover)}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]

            if pred == "at" and args[0] in self.all_rovers:
                rover_locations[args[0]] = args[1]
            elif pred == "full":
                 store_name = args[0]
                 # Find rover associated with this store
                 for r, s in self.rover_stores.items():
                     if s == store_name:
                         rover_stores_full.add(r)
                         break
            elif pred == "have_soil_analysis": have_soil.add(tuple(args))
            elif pred == "have_rock_analysis": have_rock.add(tuple(args))
            elif pred == "have_image": have_image.add(tuple(args))
            elif pred == "calibrated": calibrated_cameras.add(tuple(args))

        total_heuristic_cost = 0
        possible = True # Assume goals are reachable

        # --- Calculate Cost for Each Unmet Goal ---
        for goal_fact in self.goals:
            if goal_fact in state: continue # Skip satisfied goals

            parts = get_parts(goal_fact)
            pred = parts[0]
            args = parts[1:]
            min_goal_cost = float('inf')

            # --- Case 1: Communicated Soil Data ---
            if pred == "communicated_soil_data":
                wp_goal = args[0]
                # Option A: Rover already has data
                rovers_with_data = {r for r, w in have_soil if w == wp_goal}
                for r in rovers_with_data:
                    current_loc = rover_locations.get(r)
                    if current_loc:
                        cost_nav_comm = self.min_dist_to_comm.get(current_loc, float('inf'))
                        if cost_nav_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_nav_comm + 1)

                # Option B: Rover needs to collect and communicate
                capable_rovers = {r for r in self.all_rovers if "soil" in self.rover_equipment.get(r, set())}
                for r in capable_rovers:
                    if r in rovers_with_data: continue
                    current_loc = rover_locations.get(r)
                    if not current_loc: continue

                    cost_drop = 1 if r in rover_stores_full else 0
                    cost_nav_to_sample = self._get_dist(current_loc, wp_goal)
                    cost_nav_to_comm = self.min_dist_to_comm.get(wp_goal, float('inf'))

                    if cost_nav_to_sample != float('inf') and cost_nav_to_comm != float('inf'):
                        cost = cost_drop + cost_nav_to_sample + 1 + cost_nav_to_comm + 1
                        min_goal_cost = min(min_goal_cost, cost)

            # --- Case 2: Communicated Rock Data ---
            elif pred == "communicated_rock_data":
                wp_goal = args[0]
                # Option A: Rover already has data
                rovers_with_data = {r for r, w in have_rock if w == wp_goal}
                for r in rovers_with_data:
                    current_loc = rover_locations.get(r)
                    if current_loc:
                        cost_nav_comm = self.min_dist_to_comm.get(current_loc, float('inf'))
                        if cost_nav_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_nav_comm + 1)

                # Option B: Rover needs to collect and communicate
                capable_rovers = {r for r in self.all_rovers if "rock" in self.rover_equipment.get(r, set())}
                for r in capable_rovers:
                    if r in rovers_with_data: continue
                    current_loc = rover_locations.get(r)
                    if not current_loc: continue

                    cost_drop = 1 if r in rover_stores_full else 0
                    cost_nav_to_sample = self._get_dist(current_loc, wp_goal)
                    cost_nav_to_comm = self.min_dist_to_comm.get(wp_goal, float('inf'))

                    if cost_nav_to_sample != float('inf') and cost_nav_to_comm != float('inf'):
                        cost = cost_drop + cost_nav_to_sample + 1 + cost_nav_to_comm + 1
                        min_goal_cost = min(min_goal_cost, cost)

            # --- Case 3: Communicated Image Data ---
            elif pred == "communicated_image_data":
                obj_goal, mode_goal = args[0], args[1]
                # Option A: Rover already has image
                rovers_with_data = {r for r, o, m in have_image if o == obj_goal and m == mode_goal}
                for r in rovers_with_data:
                    current_loc = rover_locations.get(r)
                    if current_loc:
                        cost_nav_comm = self.min_dist_to_comm.get(current_loc, float('inf'))
                        if cost_nav_comm != float('inf'):
                            min_goal_cost = min(min_goal_cost, cost_nav_comm + 1)

                # Option B: Rover needs to calibrate, take image, communicate
                capable_rovers = {r for r in self.all_rovers if "imaging" in self.rover_equipment.get(r, set())}
                for r in capable_rovers:
                    if r in rovers_with_data: continue
                    current_loc = rover_locations.get(r)
                    if not current_loc: continue

                    rover_cams = self.rover_cameras.get(r, [])
                    for cam, supported_modes, cal_target_obj in rover_cams:
                        if mode_goal not in supported_modes: continue

                        cost_calibrate = 0
                        loc_after_cal = current_loc

                        # Calibration Step
                        if (cam, r) not in calibrated_cameras:
                            if not cal_target_obj: continue
                            cal_wps = self.calibration_visibility.get(cal_target_obj, set())
                            if not cal_wps: continue

                            min_dist_cal = float('inf')
                            best_cal_wp = None
                            for cal_wp in cal_wps:
                                d = self._get_dist(current_loc, cal_wp)
                                if d < min_dist_cal: min_dist_cal, best_cal_wp = d, cal_wp
                            if best_cal_wp is None: continue

                            cost_calibrate = min_dist_cal + 1
                            loc_after_cal = best_cal_wp

                        # Imaging Step
                        img_wps = self.objective_visibility.get(obj_goal, set())
                        if not img_wps: continue

                        min_dist_img = float('inf')
                        best_img_wp = None
                        for img_wp in img_wps:
                            d = self._get_dist(loc_after_cal, img_wp)
                            if d < min_dist_img: min_dist_img, best_img_wp = d, img_wp
                        if best_img_wp is None: continue

                        cost_nav_to_img = min_dist_img
                        loc_after_img = best_img_wp

                        # Communication Step
                        cost_nav_to_comm = self.min_dist_to_comm.get(loc_after_img, float('inf'))
                        if cost_nav_to_comm == float('inf'): continue

                        # Total cost for this sequence
                        cost = cost_calibrate + cost_nav_to_img + 1 + cost_nav_to_comm + 1
                        min_goal_cost = min(min_goal_cost, cost)

            # --- Accumulate Cost ---
            if min_goal_cost == float('inf'):
                possible = False # Found an unreachable goal
                break # Stop processing goals
            else:
                total_heuristic_cost += min_goal_cost

        # Return final heuristic value
        return total_heuristic_cost if possible else float('inf')
