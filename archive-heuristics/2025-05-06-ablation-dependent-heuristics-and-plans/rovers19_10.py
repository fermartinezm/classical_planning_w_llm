import collections
import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Use math.inf for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise an error
        # For robustness, return an empty list or handle appropriately
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_navigation_graph(static_facts, rovers, waypoints):
    """
    Builds an adjacency list representation of the navigation graph.
    Assumes 'can_traverse' is symmetric and the same for all rovers.
    """
    graph = collections.defaultdict(set)
    # Collect all waypoints explicitly mentioned in can_traverse or as waypoints
    all_waypoints = set(waypoints)
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'can_traverse':
            # (can_traverse ?r ?x ?y)
            wp1 = parts[2]
            wp2 = parts[3]
            graph[wp1].add(wp2)
            graph[wp2].add(wp1) # Assuming symmetric traversal
            all_waypoints.add(wp1)
            all_waypoints.add(wp2)

    # Ensure all known waypoints are keys in the graph, even if isolated
    for wp in all_waypoints:
        if wp not in graph:
            graph[wp] = set()

    return graph, list(all_waypoints) # Return graph and list of all waypoints

def compute_shortest_paths(graph, waypoints):
    """
    Computes all-pairs shortest paths using BFS.
    Returns a dictionary shortest_paths[start_wp][end_wp] = distance.
    """
    shortest_paths = {}
    for start_node in waypoints:
        shortest_paths[start_node] = {}
        queue = collections.deque([(start_node, 0)])
        visited = {start_node}

        while queue:
            current_node, dist = queue.popleft()
            shortest_paths[start_node][current_node] = dist

            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    # Fill in unreachable pairs with infinity
    for wp1 in waypoints:
        for wp2 in waypoints:
            if wp2 not in shortest_paths[wp1]:
                 shortest_paths[wp1][wp2] = math.inf

    return shortest_paths

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve
    each uncommunicated goal (soil data, rock data, image data).
    The total heuristic value is the sum of the minimum estimated costs
    for each individual unachieved goal.

    # Assumptions
    - All actions have a unit cost of 1.
    - Navigation cost between waypoints is the shortest path length in the
      traversal graph (computed using BFS).
    - 'can_traverse' predicates are symmetric and apply to all rovers.
    - Samples (soil/rock) are consumed upon sampling.
    - Camera calibration is consumed upon taking an image.
    - Stores are single-use for samples; dropping makes them empty again.
    - Goals requiring samples assume the sample was initially present at the waypoint.
    - Goals requiring images assume the objective is visible from at least one waypoint,
      the camera has a calibration target, and the target is visible from at least one waypoint.
    - The lander location is static and visible from at least one waypoint.

    # Heuristic Initialization
    - Parses static facts to identify:
        - Lander location.
        - All waypoints.
        - Navigation graph (from 'can_traverse').
        - Shortest path distances between all pairs of waypoints.
        - Waypoints visible from the lander.
        - Rovers equipped for soil, rock, and imaging.
        - Store ownership ('store_of').
        - Camera capabilities ('on_board', 'supports', 'calibration_target').
        - Objective visibility ('visible_from').
    - Precomputes minimum navigation costs from any waypoint to:
        - Any waypoint visible from the lander.
        - Any waypoint visible from a specific objective (for imaging).
        - Any waypoint visible from a specific calibration target (for calibration).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to get dynamic information (rover locations,
       sample locations, store status, collected data/images, communicated goals).
    3. Iterate through each goal fact defined in the task.
    4. For an unachieved goal `(communicated_soil_data W)`:
       - Estimate the minimum cost to get the sample and communicate it.
       - Option A: A rover R already has `(have_soil_analysis R W)`.
         - Cost = minimum navigation from R's current location to any waypoint
           visible from the lander + 1 (communicate). Minimize over such rovers R.
       - Option B: No rover has the sample, but a soil-equipped rover R can get it.
         - Requires `(at_soil_sample W)` in the initial state (checked implicitly
           by whether W was listed as a soil sample location in the initial state).
         - Cost = minimum over soil-equipped rovers R of (
           navigation from R's current location to W +
           1 (if R's store is full, need to drop) +
           1 (sample_soil) +
           minimum navigation from W to any waypoint visible from the lander +
           1 (communicate_soil_data)
         ).
       - The heuristic contribution for this goal is the minimum cost between Option A (if applicable) and Option B (if applicable).
    5. For an unachieved goal `(communicated_rock_data W)`: Similar logic as soil data, using rock-specific predicates and rovers.
    6. For an unachieved goal `(communicated_image_data O M)`:
       - Estimate the minimum cost to take the image and communicate it.
       - Option A: A rover R already has `(have_image R O M)`.
         - Cost = minimum navigation from R's current location to any waypoint
           visible from the lander + 1 (communicate). Minimize over such rovers R.
       - Option B: No rover has the image, but an imaging-equipped rover R with
         camera I supporting mode M can get it.
         - Requires I to have a calibration target T, O to be visible from some
           waypoint(s) P, and T to be visible from some waypoint(s) W.
         - Cost = minimum over suitable R, I of (
           minimum navigation from R's current location to any calibration waypoint W
           visible from I's target T +
           1 (calibrate) +
           minimum navigation from any such W to any imaging waypoint P visible from O +
           1 (take_image) +
           minimum navigation from any such P to any communication waypoint X visible
           from the lander +
           1 (communicate_image_data)
         ).
         - This navigation cost sequence (R_pos -> Cal_WP -> Img_WP -> Comm_WP) is
           estimated using precomputed shortest paths between sets of waypoints.
       - The heuristic contribution for this goal is the minimum cost between Option A (if applicable) and Option B (if applicable).
    7. The total heuristic value is the sum of contributions for all unachieved goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # --- Parse Static Facts ---
        self.lander_location = None
        self.rover_types = set()
        self.waypoint_types = set()
        self.visible_waypoints = set() # (wp1, wp2) pairs
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.imaging_rovers = set()
        self.store_to_rover = {} # store -> rover
        self.camera_supports = collections.defaultdict(set) # camera -> {modes}
        self.objective_visible_from = collections.defaultdict(set) # objective -> {waypoints}
        self.camera_calibration_target = {} # camera -> objective
        self.camera_on_board = {} # camera -> rover

        # Collect all objects by type first
        object_types = collections.defaultdict(set)
        # This requires parsing the problem file objects section, which is not
        # directly available in task.static. We can infer types from predicates.
        # A more robust parser would be needed for general PDDL.
        # For this domain, we can infer types from predicate arguments.
        # Let's assume we get object types from task somehow, or infer them
        # approximately from predicate usage in static facts.
        # A simpler approach for this specific heuristic is to just collect
        # all unique objects appearing in relevant static predicates and assume their type.
        all_objects = set()
        for fact in static_facts:
             all_objects.update(get_parts(fact)) # Add all parts as potential objects

        # Filter objects by type based on predicate usage (heuristic-specific inference)
        # This is a simplification; a proper parser would provide object types.
        # We'll rely on the structure of the predicates.
        # Let's collect waypoints and rovers from can_traverse/at_lander/at predicates
        waypoints_set = set()
        rovers_set = set()
        stores_set = set()
        cameras_set = set()
        objectives_set = set()
        modes_set = set()
        landers_set = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at_lander': landers_set.add(parts[1]); waypoints_set.add(parts[2]); self.lander_location = parts[2]
            elif pred == 'at': rovers_set.add(parts[1]); waypoints_set.add(parts[2]) # Assuming 'at' in static is for rovers
            elif pred == 'can_traverse': rovers_set.add(parts[1]); waypoints_set.add(parts[2]); waypoints_set.add(parts[3])
            elif pred == 'equipped_for_soil_analysis': rovers_set.add(parts[1]); self.soil_rovers.add(parts[1])
            elif pred == 'equipped_for_rock_analysis': rovers_set.add(parts[1]); self.rock_rovers.add(parts[1])
            elif pred == 'equipped_for_imaging': rovers_set.add(parts[1]); self.imaging_rovers.add(parts[1])
            elif pred == 'store_of': stores_set.add(parts[1]); rovers_set.add(parts[2]); self.store_to_rover[parts[1]] = parts[2]
            elif pred == 'calibrated': cameras_set.add(parts[1]); rovers_set.add(parts[2])
            elif pred == 'supports': cameras_set.add(parts[1]); modes_set.add(parts[2]); self.camera_supports[parts[1]].add(parts[2])
            elif pred == 'visible': waypoints_set.add(parts[1]); waypoints_set.add(parts[2]); self.visible_waypoints.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis': rovers_set.add(parts[1]); waypoints_set.add(parts[2])
            elif pred == 'have_soil_analysis': rovers_set.add(parts[1]); waypoints_set.add(parts[2])
            elif pred == 'full': stores_set.add(parts[1])
            elif pred == 'empty': stores_set.add(parts[1])
            elif pred == 'have_image': rovers_set.add(parts[1]); objectives_set.add(parts[2]); modes_set.add(parts[3])
            elif pred == 'communicated_soil_data': waypoints_set.add(parts[1])
            elif pred == 'communicated_rock_data': waypoints_set.add(parts[1])
            elif pred == 'communicated_image_data': objectives_set.add(parts[1]); modes_set.add(parts[2])
            elif pred == 'at_soil_sample': waypoints_set.add(parts[1])
            elif pred == 'at_rock_sample': waypoints_set.add(parts[1])
            elif pred == 'visible_from': objectives_set.add(parts[1]); waypoints_set.add(parts[2]); self.objective_visible_from[parts[1]].add(parts[2])
            elif pred == 'calibration_target': cameras_set.add(parts[1]); objectives_set.add(parts[2]); self.camera_calibration_target[parts[1]] = parts[2]
            elif pred == 'on_board': cameras_set.add(parts[1]); rovers_set.add(parts[2]); self.camera_on_board[parts[1]] = parts[2]

        self.waypoint_types = waypoints_set
        self.rover_types = rovers_set

        # --- Precompute Navigation ---
        self.navigation_graph, all_waypoints_list = build_navigation_graph(static_facts, self.rover_types, self.waypoint_types)
        self.shortest_paths = compute_shortest_paths(self.navigation_graph, all_waypoints_list)

        # --- Precompute Minimum Navigation to Key Locations ---
        self.waypoints_visible_from_lander = {
            wp1 for wp1, wp2 in self.visible_waypoints if wp2 == self.lander_location
        }

        self.min_nav_to_any_comm_point = {}
        for wp in all_waypoints_list:
            self.min_nav_to_any_comm_point[wp] = self._min_nav_wp_set(wp, self.waypoints_visible_from_lander)

        self.min_nav_to_any_imaging_point = collections.defaultdict(dict)
        for wp in all_waypoints_list:
            for obj in objectives_set: # Iterate through all potential objectives
                 img_points = self.objective_visible_from.get(obj, set())
                 self.min_nav_to_any_imaging_point[wp][obj] = self._min_nav_wp_set(wp, img_points)

        self.min_nav_to_any_calibration_point = collections.defaultdict(dict)
        for wp in all_waypoints_list:
            for obj in objectives_set: # Iterate through all potential calibration targets (objectives)
                 cal_points = self.objective_visible_from.get(obj, set())
                 self.min_nav_to_any_calibration_point[wp][obj] = self._min_nav_wp_set(wp, cal_points)


    def _get_store_for_rover(self, rover):
        """Helper to find the store associated with a rover."""
        # Invert store_to_rover mapping if needed, or iterate
        for store, r in self.store_to_rover.items():
            if r == rover:
                return store
        return None # Should not happen in valid instances

    def _min_nav_wp_set(self, start_wp, target_set):
        """Min navigation cost from a single waypoint to any in a set."""
        if start_wp not in self.shortest_paths or not target_set:
            return math.inf
        min_dist = math.inf
        for target_wp in target_set:
            if target_wp in self.shortest_paths[start_wp]:
                 min_dist = min(min_dist, self.shortest_paths[start_wp][target_wp])
        return min_dist

    def _min_nav_set_set(self, start_set, target_set):
        """Min navigation cost from any waypoint in start_set to any in target_set."""
        if not start_set or not target_set:
            return math.inf
        min_dist = math.inf
        for start_wp in start_set:
            min_dist_from_wp = self._min_nav_wp_set(start_wp, target_set)
            min_dist = min(min_dist, min_dist_from_wp)
        return min_dist

    def _min_nav_path_3_sets(self, start_wp, set_mid1, set_mid2, set_end):
        """Min navigation cost for path start_wp -> m1 -> m2 -> end."""
        if not set_mid1 or not set_mid2 or not set_end:
            return math.inf
        min_dist = math.inf
        for m1 in set_mid1:
            if m1 not in self.shortest_paths.get(start_wp, {}): continue
            dist_start_to_m1 = self.shortest_paths[start_wp][m1]
            if dist_start_to_m1 == math.inf: continue

            for m2 in set_mid2:
                if m2 not in self.shortest_paths.get(m1, {}): continue
                dist_m1_to_m2 = self.shortest_paths[m1][m2]
                if dist_m1_to_m2 == math.inf: continue

                dist_m2_to_end = self._min_nav_wp_set(m2, set_end)
                if dist_m2_to_end != math.inf:
                    min_dist = min(min_dist, dist_start_to_m1 + dist_m1_to_m2 + dist_m2_to_end)
        return min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse State Facts ---
        rover_locations = {} # rover -> waypoint
        soil_samples_at = set() # {waypoint}
        rock_samples_at = set() # {waypoint}
        empty_stores = set() # {store}
        full_stores = set() # {store}
        calibrated_cameras = set() # {(camera, rover)}
        have_soil_data = set() # {(rover, waypoint)}
        have_rock_data = set() # {(rover, waypoint)}
        have_image_data = set() # {(rover, objective, mode)}
        communicated_soil_data_goals = set() # {waypoint}
        communicated_rock_data_goals = set() # {waypoint}
        communicated_image_data_goals = set() # {(objective, mode)}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and parts[1] in self.rover_types: rover_locations[parts[1]] = parts[2]
            elif pred == 'at_soil_sample': soil_samples_at.add(parts[1])
            elif pred == 'at_rock_sample': rock_samples_at.add(parts[1])
            elif pred == 'empty': empty_stores.add(parts[1])
            elif pred == 'full': full_stores.add(parts[1])
            elif pred == 'calibrated': calibrated_cameras.add((parts[1], parts[2]))
            elif pred == 'have_soil_analysis': have_soil_data.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis': have_rock_data.add((parts[1], parts[2]))
            elif pred == 'have_image': have_image_data.add((parts[1], parts[2], parts[3]))
            elif pred == 'communicated_soil_data': communicated_soil_data_goals.add(parts[1])
            elif pred == 'communicated_rock_data': communicated_rock_data_goals.add(parts[1])
            elif pred == 'communicated_image_data': communicated_image_data_goals.add((parts[1], parts[2]))

        total_heuristic_cost = 0

        # --- Estimate Cost for Each Goal ---
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint in communicated_soil_data_goals:
                    continue # Goal already achieved

                min_cost_soil = math.inf

                # Option 1: Communicate existing sample
                rovers_with_sample = {r for (r, w) in have_soil_data if w == waypoint}
                if rovers_with_sample:
                    costs = []
                    for r in rovers_with_sample:
                        r_pos = rover_locations.get(r)
                        if r_pos:
                            # Cost = Nav(R_pos, Comm_Point) + 1
                            nav_cost = self.min_nav_to_any_comm_point.get(r_pos, math.inf)
                            if nav_cost != math.inf:
                                costs.append(nav_cost + 1)
                    if costs:
                        min_cost_soil = min(min_cost_soil, min(costs))

                # Option 2: Sample and communicate
                # Check if the sample is still available at the waypoint
                if waypoint in soil_samples_at:
                     costs = []
                     for r in self.soil_rovers:
                         r_pos = rover_locations.get(r)
                         store = self._get_store_for_rover(r)
                         if r_pos and store:
                             store_full = store in full_stores
                             # Cost = Nav(R_pos, W) + (1 if store full) + 1 (sample) + Nav(W, Comm_Point) + 1 (communicate)
                             nav_to_sample = self.shortest_paths.get(r_pos, {}).get(waypoint, math.inf)
                             nav_from_sample_to_comm = self.min_nav_to_any_comm_point.get(waypoint, math.inf)

                             if nav_to_sample != math.inf and nav_from_sample_to_comm != math.inf:
                                 cost = nav_to_sample + (1 if store_full else 0) + 1 + nav_from_sample_to_comm + 1
                                 costs.append(cost)
                     if costs:
                         min_cost_soil = min(min_cost_soil, min(costs))

                if min_cost_soil != math.inf:
                    total_heuristic_cost += min_cost_soil
                # else: goal is unreachable, heuristic should reflect this (infinity already handled)


            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint in communicated_rock_data_goals:
                    continue # Goal already achieved

                min_cost_rock = math.inf

                # Option 1: Communicate existing sample
                rovers_with_sample = {r for (r, w) in have_rock_data if w == waypoint}
                if rovers_with_sample:
                    costs = []
                    for r in rovers_with_sample:
                        r_pos = rover_locations.get(r)
                        if r_pos:
                            nav_cost = self.min_nav_to_any_comm_point.get(r_pos, math.inf)
                            if nav_cost != math.inf:
                                costs.append(nav_cost + 1)
                    if costs:
                        min_cost_rock = min(min_cost_rock, min(costs))

                # Option 2: Sample and communicate
                if waypoint in rock_samples_at:
                     costs = []
                     for r in self.rock_rovers:
                         r_pos = rover_locations.get(r)
                         store = self._get_store_for_rover(r)
                         if r_pos and store:
                             store_full = store in full_stores
                             nav_to_sample = self.shortest_paths.get(r_pos, {}).get(waypoint, math.inf)
                             nav_from_sample_to_comm = self.min_nav_to_any_comm_point.get(waypoint, math.inf)

                             if nav_to_sample != math.inf and nav_from_sample_to_comm != math.inf:
                                 cost = nav_to_sample + (1 if store_full else 0) + 1 + nav_from_sample_to_comm + 1
                                 costs.append(cost)
                     if costs:
                         min_cost_rock = min(min_cost_rock, min(costs))

                if min_cost_rock != math.inf:
                    total_heuristic_cost += min_cost_rock


            elif pred == 'communicated_image_data':
                objective = parts[1]
                mode = parts[2]
                if (objective, mode) in communicated_image_data_goals:
                    continue # Goal already achieved

                min_cost_image = math.inf

                # Option 1: Communicate existing image
                rovers_with_image = {r for (r, o, m) in have_image_data if o == objective and m == mode}
                if rovers_with_image:
                    costs = []
                    for r in rovers_with_image:
                        r_pos = rover_locations.get(r)
                        if r_pos:
                            nav_cost = self.min_nav_to_any_comm_point.get(r_pos, math.inf)
                            if nav_cost != math.inf:
                                costs.append(nav_cost + 1)
                    if costs:
                        min_cost_image = min(min_cost_image, min(costs))

                # Option 2: Take image and communicate
                costs = []
                for r in self.imaging_rovers:
                    r_pos = rover_locations.get(r)
                    if not r_pos: continue # Rover location unknown

                    # Find cameras on this rover that support the mode
                    cameras_on_R_supporting_M = {
                        i for i, r_on in self.camera_on_board.items()
                        if r_on == r and mode in self.camera_supports.get(i, set())
                    }
                    if not cameras_on_R_supporting_M: continue

                    for i in cameras_on_R_supporting_M:
                        t = self.camera_calibration_target.get(i)
                        if not t: continue # Camera has no calibration target

                        cal_points = self.objective_visible_from.get(t, set())
                        img_points = self.objective_visible_from.get(objective, set())
                        comm_points = self.waypoints_visible_from_lander

                        # Cost = Nav(R_pos, Cal_WP) + 1 + Nav(Cal_WP, Img_WP) + 1 + Nav(Img_WP, Comm_WP) + 1
                        # Minimize over Cal_WP in cal_points, Img_WP in img_points, Comm_WP in comm_points
                        nav_cost_sequence = self._min_nav_path_3_sets(r_pos, cal_points, img_points, comm_points)

                        if nav_cost_sequence != math.inf:
                            cost = nav_cost_sequence + 3 # 1 calibrate + 1 take_image + 1 communicate
                            costs.append(cost)

                if costs:
                    min_cost_image = min(min_cost_image, min(costs))

                if min_cost_image != math.inf:
                    total_heuristic_cost += min_cost_image
                # else: goal is unreachable


        return total_heuristic_cost

