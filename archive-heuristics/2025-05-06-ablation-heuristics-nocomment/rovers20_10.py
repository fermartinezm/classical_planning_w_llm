from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """Removes parentheses and splits a PDDL fact string into parts."""
    # Assumes fact is like '(predicate arg1 arg2)'
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Handle unexpected format, though planner state should be consistent
        return []
    return fact[1:-1].split()

# BFS implementation
def bfs(graph, start_node):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., rover at isolated waypoint not in can_traverse)
         return distances

    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        # Handle nodes that might be in distances but not in graph keys if graph is sparse
        if current_node not in graph: continue

        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def precompute_distances(rover_graphs, all_waypoints):
    """Precomputes shortest path distances for each rover between all waypoints."""
    all_dists = {}
    # Ensure all waypoints are in the graph keys for BFS initialization
    full_graphs = {}
    for rover, graph in rover_graphs.items():
        full_graphs[rover] = {wp: set(neighbors) for wp, neighbors in graph.items()}
        for wp in all_waypoints:
             full_graphs[rover].setdefault(wp, set())

    for rover, graph in full_graphs.items():
        rover_dists = {}
        for start_wp in graph:
            rover_dists[start_wp] = bfs(graph, start_wp)
        all_dists[rover] = rover_dists
    return all_dists

def min_dist(dists_r, from_wp, to_wps):
    """Finds the minimum distance from from_wp to any waypoint in to_wps."""
    if dists_r is None or from_wp not in dists_r:
        return float('inf')
    min_d = float('inf')
    for to_wp in to_wps:
        if to_wp in dists_r.get(from_wp, {}):
            min_d = min(min_d, dists_r[from_wp][to_wp])
    return min_d

def find_closest_wp(dists_r, from_wp, target_wps):
    """Finds the waypoint in target_wps closest to from_wp."""
    if dists_r is None or from_wp not in dists_r:
        return None
    min_d = float('inf')
    closest_wp = None
    for target_wp in target_wps:
        if target_wp in dists_r.get(from_wp, {}):
            d = dists_r[from_wp][target_wp]
            if d < min_d:
                min_d = d
                closest_wp = target_wp
    return closest_wp


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    Estimates the cost to reach the goal by summing up the minimum estimated
    costs for each unsatisfied goal fact independently. The cost for each
    goal is estimated based on the required actions (sample/image, calibrate,
    communicate) and the minimum movement cost for a suitable rover/camera
    to reach the necessary locations (sample/image waypoint, calibration
    waypoint, communication waypoint).

    Assumptions:
    - Goals are independent and can be pursued in parallel by different rovers.
    - Resource constraints (like a single store per rover, or camera calibration
      state after taking an image) are partially relaxed or simplified when
      estimating the cost for a single goal (e.g., drop cost is added if store
      is full, but we don't track which rover's store is used for which sample).
    - Movement cost between waypoints for a rover is the shortest path distance
      in the graph defined by 'can_traverse' facts.
    - The cost of each action (sample, drop, calibrate, take_image, communicate)
      is 1. Movement cost is the number of navigation steps.
    - If a goal requires reaching a waypoint (sample, image, calibrate, comm)
      that is unreachable for all suitable rovers, the goal is considered
      unreachable and assigned infinite cost.
    - Initial locations of soil/rock samples are static and required for sampling.

    Heuristic Initialization:
    - Parses static facts from the task to identify:
        - All rovers, waypoints, stores, cameras, modes, landers, objectives.
        - Rover traversal graphs ('can_traverse').
        - Waypoint visibility graph ('visible').
        - Objective visibility from waypoints ('visible_from').
        - Rover equipment ('equipped_for_soil_analysis', etc.).
        - Store ownership ('store_of').
        - Camera properties ('on_board', 'supports', 'calibration_target').
        - Lander location ('at_lander').
        - Initial locations of soil/rock samples ('at_soil_sample', 'at_rock_sample').
    - Computes the set of communication waypoints (visible from the lander).
    - Precomputes shortest path distances between all pairs of waypoints for
      each rover based on their traversal graphs using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic `h = 0`.
    2. Get the current state facts and goal facts.
    3. Determine the current location of each rover and the state of each store.
    4. For each goal fact in `task.goals`:
        a. If the goal fact is already in the current state, continue to the next goal.
        b. If the goal is `(communicated_soil_data W)`:
            i. Initialize minimum cost for this goal `min_goal_cost = infinity`.
            ii. Check if `(have_soil_analysis R W)` exists for any equipped rover R in the current state.
            iii. If yes: For each such rover R, calculate the cost to move R from its current location to any communication waypoint X and communicate (cost = `min_dist(R's dists, R's loc, comm_wps) + 1`). Update `min_goal_cost` with the minimum such cost.
            iv. If no: Check if `(at_soil_sample W)` was true in the initial state (sample must exist). If not, the goal is impossible. If yes, for each equipped rover R, calculate the cost to:
                - Move R from its current location to W (`dist(R's loc, W)`).
                - Add 1 if R's store is full (for drop action).
                - Add 1 for the `sample_soil` action.
                - Move R from W to any communication waypoint X (`min_dist(W, comm_wps)`).
                - Add 1 for the `communicate_soil_data` action.
                Calculate the total cost for this rover and update `min_goal_cost` with the minimum such cost over all equipped rovers and communication waypoints.
            v. If `min_goal_cost` is still infinity, the goal is unreachable. Add infinity to `h`.
            vi. Otherwise, add `min_goal_cost` to `h`.
        c. If the goal is `(communicated_rock_data W)`: Follow a similar process as for soil data, using rock-specific predicates and equipped rovers and checking `(at_rock_sample W)` in the initial state.
        d. If the goal is `(communicated_image_data O M)`:
            i. Initialize minimum cost for this goal `min_goal_cost = infinity`.
            ii. Check if `(have_image R O M)` exists for any equipped rover R with a camera I supporting M in the current state.
            iii. If yes: For each such rover R, calculate the cost to move R from its current location to any communication waypoint X and communicate (cost = `min_dist(R's dists, R's loc, comm_wps) + 1`). Update `min_goal_cost` with the minimum such cost.
            iv. If no: Check if objective O is visible from any waypoint (`visible_from O P`). If not, the goal is impossible. If yes, for each equipped rover R with a camera I supporting M, calculate the cost to:
                - If camera I is not calibrated:
                    - Find the closest calibration waypoint W visible from I's calibration target T (`find_closest_wp(R's dists, R's loc, cal_wps)`). If no such W exists, this camera path is impossible.
                    - Add movement cost to W (`dist(R's loc, W)`).
                    - Add 1 for the `calibrate` action.
                    - The rover is now conceptually at W.
                - If camera I is calibrated: Cost to calibrate is 0, rover is at its current location.
                - Find the closest image waypoint P visible from O, reachable from the rover's location after calibration (or current location if already calibrated) (`find_closest_wp(R's dists, loc_after_cal, img_wps)`). If no such P exists, this camera path is impossible.
                - Add movement cost to P (`dist(loc_after_cal, P)`).
                - Add 1 for the `take_image` action.
                - The rover is now conceptually at P.
                - Find the closest communication waypoint X reachable from P (`find_closest_wp(R's dists, P, comm_wps)`). If no such X exists, this camera path is impossible.
                - Add movement cost to X (`dist(P, X)`).
                - Add 1 for the `communicate_image_data` action.
                Calculate the total cost for this rover/camera combination and update `min_goal_cost` with the minimum such cost over all suitable combinations and waypoints.
            v. If `min_goal_cost` is still infinity, the goal is unreachable. Add infinity to `h`.
            vi. Otherwise, add `min_goal_cost` to `h`.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__(task)

        # --- Parse Static Information ---
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        self.rover_graphs = {} # rover -> {wp -> {neighbor_wp, ...}}
        self.visible_wps = {} # wp -> {visible_wp, ...}
        self.visible_from_obj = {} # objective -> {waypoint, ...}
        self.equipped_soil_rovers = set()
        self.equipped_rock_rovers = set()
        self.equipped_imaging_rovers = set()
        self.store_of_rover = {} # store -> rover (assuming 1:1)
        self.camera_on_rover = {} # camera -> rover
        self.camera_supports = {} # camera -> {mode, ...}
        self.calibration_target = {} # camera -> objective
        self.lander_location = None # waypoint where lander is
        self.initial_soil_samples = set() # waypoint with soil sample initially
        self.initial_rock_samples = set() # waypoint with rock sample initially


        # Collect all objects and static predicates
        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact_str in all_facts:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            # Collect objects (simple approach based on argument position/predicate)
            if predicate in ['at', 'at_lander', 'can_traverse', 'sample_soil', 'sample_rock', 'take_image', 'communicate_soil_data', 'communicate_rock_data', 'communicate_image_data', 'equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging', 'store_of', 'on_board']:
                 if len(args) > 0: self.rovers.add(args[0]) # First arg is often rover
            if predicate in ['at', 'at_lander', 'can_traverse', 'sample_soil', 'sample_rock', 'calibrate', 'take_image', 'communicate_soil_data', 'communicate_rock_data', 'communicate_image_data', 'visible', 'visible_from', 'at_soil_sample', 'at_rock_sample']:
                 for arg in args: self.waypoints.add(arg) # Waypoints appear in many positions
            if predicate in ['sample_soil', 'sample_rock', 'drop', 'empty', 'full', 'store_of']:
                 if len(args) > 0: self.stores.add(args[0]) # First arg is store
            if predicate in ['calibrate', 'take_image', 'on_board', 'supports', 'calibration_target', 'calibrated']:
                 if len(args) > 0: self.cameras.add(args[0]) # First arg is camera
            if predicate in ['supports', 'take_image', 'communicate_image_data']:
                 if len(args) > 1: self.modes.add(args[-1]) # Last arg is mode
            if predicate == 'at_lander':
                 self.landers.add(args[0])
            if predicate in ['take_image', 'communicate_image_data', 'calibration_target', 'visible_from', 'have_image']:
                 if len(args) > 1: self.objectives.add(args[1]) # Second arg is objective


            # Parse static predicates (from task.static)
            if fact_str in task.static:
                if predicate == 'at_lander':
                    self.lander_location = args[1]
                elif predicate == 'can_traverse':
                    r, x, y = args
                    self.rover_graphs.setdefault(r, {}).setdefault(x, set()).add(y)
                elif predicate == 'equipped_for_soil_analysis':
                    self.equipped_soil_rovers.add(args[0])
                elif predicate == 'equipped_for_rock_analysis':
                    self.equipped_rock_rovers.add(args[0])
                elif predicate == 'equipped_for_imaging':
                    self.equipped_imaging_rovers.add(args[0])
                elif predicate == 'store_of':
                    s, r = args
                    self.store_of_rover[s] = r
                elif predicate == 'visible':
                    w, p = args
                    self.visible_wps.setdefault(w, set()).add(p)
                elif predicate == 'on_board':
                    i, r = args
                    self.camera_on_rover[i] = r
                elif predicate == 'supports':
                    c, m = args
                    self.camera_supports.setdefault(c, set()).add(m)
                elif predicate == 'calibration_target':
                    i, t = args
                    self.calibration_target[i] = t
                elif predicate == 'visible_from':
                    o, w = args
                    self.visible_from_obj.setdefault(o, set()).add(w)
                # Initial sample locations are static in many instances,
                # but defined in :init block. Check initial state for these.

        # Parse initial sample locations from task.initial_state
        for fact_str in task.initial_state:
             parts = get_parts(fact_str)
             if not parts: continue
             predicate = parts[0]
             args = parts[1:]
             if predicate == 'at_soil_sample' and len(args) == 1:
                  self.initial_soil_samples.add(args[0])
             elif predicate == 'at_rock_sample' and len(args) == 1:
                  self.initial_rock_samples.add(args[0])


        # Add waypoints visible *to* the lander location to comm_wps
        self.comm_wps = set()
        if self.lander_location:
            # Find all wps X such that (visible X lander_location) is true
            for wp1, visible_set in self.visible_wps.items():
                if self.lander_location in visible_set:
                    self.comm_wps.add(wp1)
            # The lander location itself is also a communication point
            self.comm_wps.add(self.lander_location)

        # Ensure all known waypoints are in the graph keys for BFS
        for r in self.rovers:
             self.rover_graphs.setdefault(r, {})
             for wp in self.waypoints:
                 self.rover_graphs[r].setdefault(wp, set())

        # Precompute distances
        self.rover_dist = precompute_distances(self.rover_graphs, self.waypoints)

        # Store goals for easy access
        self.goals = task.goals

    def __call__(self, node):
        state = node.state
        h = 0

        # Get current dynamic state information
        rover_locations = {}
        store_state = {} # store -> 'empty' or 'full'
        calibrated_cameras = {} # (camera, rover) -> True/False
        have_soil = {} # (rover, waypoint) -> True/False
        have_rock = {} # (rover, waypoint) -> True/False
        have_image = {} # (rover, objective, mode) -> True/False

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2 and args[0] in self.rovers:
                rover_locations[args[0]] = args[1]
            elif predicate == 'empty' and len(args) == 1 and args[0] in self.stores:
                store_state[args[0]] = 'empty'
            elif predicate == 'full' and len(args) == 1 and args[0] in self.stores:
                store_state[args[0]] = 'full'
            elif predicate == 'calibrated' and len(args) == 2 and args[0] in self.cameras and args[1] in self.rovers:
                 calibrated_cameras[(args[0], args[1])] = True
            elif predicate == 'have_soil_analysis' and len(args) == 2 and args[0] in self.rovers and args[1] in self.waypoints:
                 have_soil[(args[0], args[1])] = True
            elif predicate == 'have_rock_analysis' and len(args) == 2 and args[0] in self.rovers and args[1] in self.waypoints:
                 have_rock[(args[0], args[1])] = True
            elif predicate == 'have_image' and len(args) == 3 and args[0] in self.rovers and args[1] in self.objectives and args[2] in self.modes:
                 have_image[(args[0], args[1], args[2])] = True

        # Ensure all rovers have a location recorded (should be in state if valid)
        # If a rover is not in rover_locations, min_dist/find_closest_wp will handle it.

        # Ensure all stores have a state recorded (assume full if state not specified - conservative)
        for s in self.stores:
             store_state.setdefault(s, 'full')

        # Ensure all cameras have a calibration state recorded (assume not calibrated if state not specified)
        for c in self.cameras:
             r = self.camera_on_rover.get(c)
             if r:
                 calibrated_cameras.setdefault((c, r), False)


        # Calculate cost for each unsatisfied goal
        for goal_str in self.goals:
            if goal_str in state:
                continue # Goal already satisfied

            parts = get_parts(goal_str)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            min_goal_cost = float('inf')

            if predicate == 'communicated_soil_data' and len(args) == 1:
                W = args[0]

                # Option 1: Soil analysis already exists
                analysis_exists = False
                for r in self.equipped_soil_rovers:
                    if have_soil.get((r, W), False):
                        analysis_exists = True
                        current_loc = rover_locations.get(r)
                        if current_loc:
                            cost = min_dist(self.rover_dist.get(r), current_loc, self.comm_wps)
                            if cost != float('inf'):
                                min_goal_cost = min(min_goal_cost, cost + 1) # move + communicate

                if analysis_exists and min_goal_cost != float('inf'):
                    h += min_goal_cost
                    continue # Found a path using existing analysis

                # Option 2: Need to sample
                if W not in self.initial_soil_samples:
                     # Soil sample is not available at W initially, goal is impossible
                     h += float('inf')
                     continue

                for r in self.equipped_soil_rovers:
                    current_loc = rover_locations.get(r)
                    if not current_loc: continue # Cannot use this rover if location unknown

                    store = None
                    for s, rv in self.store_of_rover.items():
                         if rv == r:
                              store = s
                              break
                    if not store: continue # Rover has no store?

                    drop_cost = 1 if store_state.get(store, 'full') == 'full' else 0

                    cost_to_sample_loc = self.rover_dist.get(r, {}).get(current_loc, {}).get(W, float('inf'))

                    if cost_to_sample_loc != float('inf'):
                        cost_after_sample = cost_to_sample_loc + drop_cost + 1 # move to W + drop + sample

                        cost_to_comm_loc = min_dist(self.rover_dist.get(r), W, self.comm_wps)

                        if cost_to_comm_loc != float('inf'):
                            total_cost = cost_after_sample + cost_to_comm_loc + 1 # move to comm + communicate
                            min_goal_cost = min(min_goal_cost, total_cost)

                h += min_goal_cost # Add infinity if unreachable

            elif predicate == 'communicated_rock_data' and len(args) == 1:
                W = args[0]

                # Option 1: Rock analysis already exists
                analysis_exists = False
                for r in self.equipped_rock_rovers:
                    if have_rock.get((r, W), False):
                        analysis_exists = True
                        current_loc = rover_locations.get(r)
                        if current_loc:
                            cost = min_dist(self.rover_dist.get(r), current_loc, self.comm_wps)
                            if cost != float('inf'):
                                min_goal_cost = min(min_goal_cost, cost + 1) # move + communicate

                if analysis_exists and min_goal_cost != float('inf'):
                    h += min_goal_cost
                    continue # Found a path using existing analysis

                # Option 2: Need to sample
                if W not in self.initial_rock_samples:
                     # Rock sample is not available at W initially, goal is impossible
                     h += float('inf')
                     continue

                for r in self.equipped_rock_rovers:
                    current_loc = rover_locations.get(r)
                    if not current_loc: continue # Cannot use this rover if location unknown

                    store = None
                    for s, rv in self.store_of_rover.items():
                         if rv == r:
                              store = s
                              break
                    if not store: continue # Rover has no store?

                    drop_cost = 1 if store_state.get(store, 'full') == 'full' else 0

                    cost_to_sample_loc = self.rover_dist.get(r, {}).get(current_loc, {}).get(W, float('inf'))

                    if cost_to_sample_loc != float('inf'):
                        cost_after_sample = cost_to_sample_loc + drop_cost + 1 # move to W + drop + sample

                        cost_to_comm_loc = min_dist(self.rover_dist.get(r), W, self.comm_wps)

                        if cost_to_comm_loc != float('inf'):
                            total_cost = cost_after_sample + cost_to_comm_loc + 1 # move to comm + communicate
                            min_goal_cost = min(min_goal_cost, total_cost)

                h += min_goal_cost # Add infinity if unreachable

            elif predicate == 'communicated_image_data' and len(args) == 2:
                O, M = args

                # Option 1: Image already exists
                image_exists = False
                for r in self.equipped_imaging_rovers:
                    # Find cameras on this rover supporting this mode
                    suitable_cameras = [c for c, rv in self.camera_on_rover.items() if rv == r and M in self.camera_supports.get(c, set())]
                    for i in suitable_cameras:
                        if have_image.get((r, O, M), False):
                            image_exists = True
                            current_loc = rover_locations.get(r)
                            if current_loc:
                                cost = min_dist(self.rover_dist.get(r), current_loc, self.comm_wps)
                                if cost != float('inf'):
                                    min_goal_cost = min(min_goal_cost, cost + 1) # move + communicate

                if image_exists and min_goal_cost != float('inf'):
                    h += min_goal_cost
                    continue # Found a path using existing image

                # Option 2: Need to take image
                img_wps = self.visible_from_obj.get(O, set())
                if not img_wps:
                    # Cannot see objective O from anywhere, goal impossible
                    h += float('inf')
                    continue

                for r in self.equipped_imaging_rovers:
                    current_loc = rover_locations.get(r)
                    if not current_loc: continue # Cannot use this rover

                    # Find cameras on this rover supporting this mode
                    suitable_cameras = [c for c, rv in self.camera_on_rover.items() if rv == r and M in self.camera_supports.get(c, set())]

                    for i in suitable_cameras:
                        cal_target = self.calibration_target.get(i)
                        cal_wps = self.visible_from_obj.get(cal_target, set()) if cal_target else set()

                        # Cost to calibrate (if needed) + move to cal point
                        cost_to_calibrate = float('inf')
                        loc_after_cal = current_loc # Default if no calibration needed/possible

                        if calibrated_cameras.get((i, r), False):
                            cost_to_calibrate = 0 # Already calibrated
                        elif cal_wps:
                            # Need to calibrate
                            best_cal_wp = find_closest_wp(self.rover_dist.get(r), current_loc, cal_wps)
                            if best_cal_wp:
                                cost_to_calibrate = self.rover_dist[r][current_loc][best_cal_wp] + 1 # move + calibrate
                                loc_after_cal = best_cal_wp # Rover is now conceptually at the calibration waypoint

                        if cost_to_calibrate == float('inf'):
                             continue # Cannot calibrate using this camera/rover

                        # Cost to take image + move to image point (from location after calibration)
                        cost_to_image = float('inf')
                        loc_after_image = loc_after_cal # Default if no image taken

                        best_img_wp = find_closest_wp(self.rover_dist.get(r), loc_after_cal, img_wps)
                        if best_img_wp:
                            cost_to_image = self.rover_dist[r][loc_after_cal][best_img_wp] + 1 # move + take_image
                            loc_after_image = best_img_wp # Rover is now conceptually at the image waypoint

                        if cost_to_image == float('inf'):
                             continue # Cannot reach image waypoint using this rover

                        # Cost to communicate + move to comm point (from location after image)
                        cost_to_comm = float('inf')
                        best_comm_wp = find_closest_wp(self.rover_dist.get(r), loc_after_image, self.comm_wps)
                        if best_comm_wp:
                             cost_to_comm = self.rover_dist[r][loc_after_image][best_comm_wp] + 1 # move + communicate

                        if cost_to_comm != float('inf'):
                            total_cost = cost_to_calibrate + cost_to_image + cost_to_comm
                            min_goal_cost = min(min_goal_cost, total_cost)

                h += min_goal_cost # Add infinity if unreachable

            # Add other goal types if any (only communication goals in this domain)

        # Return infinity if any goal is unreachable, otherwise return the sum
        return h if h != float('inf') else float('inf')

