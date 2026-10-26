import math
from collections import deque

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function for BFS
def bfs(graph, start_node):
    """Performs BFS to find shortest distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph, cannot reach anything
         return distances

    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Use .get() to handle nodes that might be in all_waypoints but have no outgoing edges in the graph
        for neighbor in graph.get(current_node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing the
    estimated minimum costs for each individual unachieved goal fact. It
    considers the steps required to achieve each goal (e.g., navigate, sample,
    calibrate, take image, communicate) and uses precomputed shortest paths
    for navigation costs.

    Assumptions:
    1.  The heuristic assumes that achieving each unachieved goal can be
        considered independently. It sums the minimum costs for each goal,
        ignoring potential positive or negative interactions (e.g., one
        navigation action serving multiple goals, or resource contention).
    2.  For image goals requiring calibration and imaging, the heuristic
        greedily selects the closest calibration waypoint from the current
        location, then the closest image waypoint from the calibration
        waypoint, and finally the closest communication waypoint from the
        image waypoint.
    3.  If a sample (soil or rock) is no longer at its initial location
        (`at_soil_sample` or `at_rock_sample` fact is false) and no rover
        currently holds that sample (`have_soil_analysis` or
        `have_rock_analysis` fact is false for all rovers), the heuristic
        consideration for sampling that specific waypoint is skipped. The goal
        is only considered reachable if a rover already holds the sample.
        This assumes samples cannot reappear or be transferred.
    4.  All rovers are assumed to have a store if they are equipped for
        soil or rock analysis.

    Heuristic Initialization:
    In the __init__ method, the heuristic processes the static information
    from the planning task. This involves:
    -   Storing the goal facts and initial state facts.
    -   Parsing static predicates like `at_lander`, `equipped_for_soil_analysis`,
        `store_of`, `on_board`, `supports`, `calibration_target`, `visible_from`,
        `visible`, and `can_traverse` into structured data (dictionaries and sets).
    -   Collecting all unique waypoints mentioned in the static or initial state facts.
    -   Building a navigation graph (adjacency list) for each rover based on the
        `can_traverse` facts.
    -   Precomputing all-pairs shortest paths for each rover on its respective
        navigation graph using Breadth-First Search (BFS). These distances are
        stored for quick lookup.
    -   Identifying the set of waypoints from which a rover can communicate
        (waypoints visible from the lander's location).

    Step-By-Step Thinking for Computing Heuristic:
    The __call__ method computes the heuristic value for a given state:
    1.  It first checks if the state is a goal state. If so, the heuristic is 0.
    2.  It identifies the set of goal facts that are not yet true in the current state.
    3.  It parses relevant dynamic facts from the current state, such as rover locations,
        store status (full/empty implied by absence of empty), held samples,
        held images, and camera calibration status.
    4.  It initializes the total heuristic value `h` to 0.
    5.  For each unachieved goal fact:
        a.  **For `(communicated_soil_data ?w)`:**
            -   Calculate the minimum cost `cost_w` to achieve this goal.
            -   Option 1: Communicate existing sample. Check if any rover `r_have` has the sample `?w`. If yes, calculate the cost: minimum navigation from `at(r_have)` to a communication point + 1 (communicate). Update `cost_w` with the minimum found.
            -   Option 2: Sample and Communicate. Check if the sample is still available at `?w` (`(at_soil_sample ?w)` is in the state). If yes, consider all soil-equipped rovers `r_sample`. Calculate the cost for each: minimum navigation from `at(r_sample)` to `?w` + 1 (sample) + (1 for drop if store is full) + minimum navigation from `?w` to a communication point + 1 (communicate). The minimum cost over all suitable rovers is taken.
            -   If `cost_w` remains infinity after checking both options, the goal is unreachable from this state, and the heuristic returns infinity.
        b.  **For `(communicated_rock_data ?w)`:**
            -   Similar logic as for soil data, but applied to rock samples and rock-equipped rovers.
        c.  **For `(communicated_image_data ?o ?m)`:**
            -   Calculate the minimum cost `cost_om` to achieve this goal.
            -   Option 1: Communicate existing image. Check if any rover `r_have` has the image `(?o ?m)`. If yes, calculate the cost: minimum navigation from `at(r_have)` to a communication point + 1 (communicate). Update `cost_om` with the minimum found.
            -   Option 2: Get Image and Communicate. Consider all imaging-equipped rovers `r_img` with cameras `i` supporting mode `?m`. For each suitable pair `(r_img, i)`:
                -   Calculate the cost to get the image and communicate it.
                -   If camera `i` is calibrated on `r_img`, the cost is estimated as: min Nav(`at(r_img)`, image_wp) + 1 (Image) + min Nav(image_wp, comm_wp) + 1 (Comm). The best image_wp is chosen greedily based on the first navigation.
                -   If camera `i` is not calibrated, the cost is estimated as: min Nav(`at(r_img)`, calib_wp) + 1 (Calib) + min Nav(calib_wp, image_wp) + 1 (Image) + min Nav(image_wp, comm_wp) + 1 (Comm). The best calib_wp and image_wp are chosen greedily based on the respective navigations.
                -   Update `cost_om` with the minimum found over all suitable `(r_img, i)` pairs and options.
            -   If `cost_om` remains infinity, the goal is unreachable, and the heuristic returns infinity.
    6.  The estimated minimum cost for the current goal fact is added to the total heuristic `h`.
    7.  After processing all unachieved goals, the total heuristic value `h` is returned. If any goal was found to be unreachable, infinity is returned early.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.initial_state = task.initial_state

        self.lander_location = {}
        self.rover_capabilities = {}
        self.store_owner = {}
        self.camera_info = {}
        self.objective_visibility = {}
        self.rover_graphs = {}
        self.all_waypoints = set()

        # Parse static facts
        for fact_string in task.static:
            parts = parse_fact(fact_string)
            pred = parts[0]

            if pred == 'at_lander':
                lander, wp = parts[1:]
                self.lander_location[lander] = wp
                self.all_waypoints.add(wp)
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif pred == 'store_of':
                store, rover = parts[1:]
                self.store_owner[store] = rover
            elif pred == 'on_board':
                camera, rover = parts[1:]
                self.camera_info.setdefault(camera, {})['rover'] = rover
            elif pred == 'supports':
                camera, mode = parts[1:]
                self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
            elif pred == 'calibration_target':
                camera, target = parts[1:]
                self.camera_info.setdefault(camera, {})['calib_target'] = target
            elif pred == 'visible_from':
                objective, wp = parts[1:]
                self.objective_visibility.setdefault(objective, set()).add(wp)
                self.all_waypoints.add(wp)
            elif pred == 'visible':
                 wp1, wp2 = parts[1:]
                 self.all_waypoints.add(wp1)
                 self.all_waypoints.add(wp2)
            elif pred == 'can_traverse':
                 rover, wp1, wp2 = parts[1:]
                 self.rover_graphs.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
                 self.all_waypoints.add(wp1)
                 self.all_waypoints.add(wp2)

        # Add waypoints from initial state facts that might not be in static facts
        for fact_string in task.initial_state:
             parts = parse_fact(fact_string)
             pred = parts[0]
             # Check predicates that involve a waypoint as the last argument
             if pred in ('at', 'at_lander', 'at_soil_sample', 'at_rock_sample'):
                 if len(parts) > 1:
                     self.all_waypoints.add(parts[-1])

        # Ensure all rovers and waypoints are represented in the graph structure
        all_rovers = set(self.rover_capabilities.keys()) | set(r for info in self.camera_info.values() for r in [info.get('rover')] if r) | set(self.store_owner.values())
        for rover in all_rovers:
             self.rover_graphs.setdefault(rover, {})
             for wp in self.all_waypoints:
                 self.rover_graphs[rover].setdefault(wp, set())

        # Precompute shortest paths for each rover
        self.rover_shortest_paths = {}
        for rover, graph in self.rover_graphs.items():
            self.rover_shortest_paths[rover] = {}
            for start_w in self.all_waypoints:
                distances = bfs(graph, start_w)
                for end_w, dist in distances.items():
                    if dist != float('inf'):
                        self.rover_shortest_paths[rover][(start_w, end_w)] = dist

        # Determine communication points visible from the lander
        self.lander_comm_points = set()
        lander_w = list(self.lander_location.values())[0] if self.lander_location else None
        if lander_w:
            # Find all waypoints X such that (visible X lander_w) is true
            for fact_string in task.static:
                parts = parse_fact(fact_string)
                if parts[0] == 'visible' and len(parts) == 3:
                    w1, w2 = parts[1:]
                    if w2 == lander_w:
                        self.lander_comm_points.add(w1)
            # Add the lander waypoint itself if it's in the set of all waypoints
            if lander_w in self.all_waypoints:
                 self.lander_comm_points.add(lander_w)


    def __call__(self, state):
        if self.goals <= state:
            return 0

        h = 0
        unachieved_goals = self.goals - state

        # Parse dynamic facts from the current state
        rover_at = {}
        store_full = set()
        have_soil = set()
        have_rock = set()
        calibrated_cameras = set()
        have_image = set()

        for fact_string in state:
            parts = parse_fact(fact_string)
            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                rover, wp = parts[1:]
                rover_at[rover] = wp
            elif pred == 'full' and len(parts) == 2:
                store = parts[1]
                store_full.add(store)
            elif pred == 'have_soil_analysis' and len(parts) == 3:
                rover, wp = parts[1:]
                have_soil.add((rover, wp))
            elif pred == 'have_rock_analysis' and len(parts) == 3:
                rover, wp = parts[1:]
                have_rock.add((rover, wp))
            elif pred == 'calibrated' and len(parts) == 3:
                camera, rover = parts[1:]
                calibrated_cameras.add((camera, rover))
            elif pred == 'have_image' and len(parts) == 4:
                rover, obj, mode = parts[1:]
                have_image.add((rover, obj, mode))

        comm_points = self.lander_comm_points

        for goal_fact_string in unachieved_goals:
            parts = parse_fact(goal_fact_string)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                w = parts[1]
                cost_w = float('inf')

                # Option 1: Communicate existing sample
                rovers_with_sample = [r for r, sample_w in have_soil if sample_w == w]
                for r_have in rovers_with_sample:
                    curr_w = rover_at.get(r_have)
                    if curr_w is not None:
                        nav_cost = min((self.rover_shortest_paths[r_have].get((curr_w, comm_wp), float('inf')) for comm_wp in comm_points), default=float('inf'))
                        if nav_cost != float('inf'):
                            cost_w = min(cost_w, nav_cost + 1) # nav + communicate

                # Option 2: Sample and Communicate (only if sample is still available)
                if f'(at_soil_sample {w})' in state:
                    soil_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
                    for r_sample in soil_rovers:
                        curr_w = rover_at.get(r_sample)
                        if curr_w is not None:
                            store = next((s for s, owner in self.store_owner.items() if owner == r_sample), None)
                            if store is not None:
                                sample_nav_cost = self.rover_shortest_paths[r_sample].get((curr_w, w), float('inf'))
                                if sample_nav_cost != float('inf'):
                                    sample_cost = sample_nav_cost + 1 # nav + sample
                                    if f'(full {store})' in state:
                                        sample_cost += 1 # drop

                                    comm_nav_cost = min((self.rover_shortest_paths[r_sample].get((w, comm_wp), float('inf')) for comm_wp in comm_points), default=float('inf'))
                                    if comm_nav_cost != float('inf'):
                                        comm_cost = comm_nav_cost + 1 # nav + communicate
                                        cost_w = min(cost_w, sample_cost + comm_cost)

                if cost_w == float('inf'):
                    return float('inf')
                h += cost_w

            elif pred == 'communicated_rock_data':
                w = parts[1]
                cost_w = float('inf')

                # Option 1: Communicate existing sample
                rovers_with_sample = [r for r, sample_w in have_rock if sample_w == w]
                for r_have in rovers_with_sample:
                    curr_w = rover_at.get(r_have)
                    if curr_w is not None:
                        nav_cost = min((self.rover_shortest_paths[r_have].get((curr_w, comm_wp), float('inf')) for comm_wp in comm_points), default=float('inf'))
                        if nav_cost != float('inf'):
                            cost_w = min(cost_w, nav_cost + 1) # nav + communicate

                # Option 2: Sample and Communicate (only if sample is still available)
                if f'(at_rock_sample {w})' in state:
                    rock_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
                    for r_sample in rock_rovers:
                        curr_w = rover_at.get(r_sample)
                        if curr_w is not None:
                            store = next((s for s, owner in self.store_owner.items() if owner == r_sample), None)
                            if store is not None:
                                sample_nav_cost = self.rover_shortest_paths[r_sample].get((curr_w, w), float('inf'))
                                if sample_nav_cost != float('inf'):
                                    sample_cost = sample_nav_cost + 1 # nav + sample
                                    if f'(full {store})' in state:
                                        sample_cost += 1 # drop

                                    comm_nav_cost = min((self.rover_shortest_paths[r_sample].get((w, comm_wp), float('inf')) for comm_wp in comm_points), default=float('inf'))
                                    if comm_nav_cost != float('inf'):
                                        comm_cost = comm_nav_cost + 1 # nav + communicate
                                        cost_w = min(cost_w, sample_cost + comm_cost)

                if cost_w == float('inf'):
                    return float('inf')
                h += cost_w

            elif pred == 'communicated_image_data':
                o, m = parts[1:]
                cost_om = float('inf')

                # Option 1: Communicate existing image
                rovers_with_image = [r for r, img_o, img_m in have_image if img_o == o and img_m == m]
                for r_have in rovers_with_image:
                    curr_w = rover_at.get(r_have)
                    if curr_w is not None:
                        nav_cost = min((self.rover_shortest_paths[r_have].get((curr_w, comm_wp), float('inf')) for comm_wp in comm_points), default=float('inf'))
                        if nav_cost != float('inf'):
                            cost_om = min(cost_om, nav_cost + 1) # nav + communicate

                # Option 2: Get Image and Communicate
                suitable_rover_camera_pairs = []
                imaging_rovers = [r for r, caps in self.rover_capabilities.items() if 'imaging' in caps]
                for r_img in imaging_rovers:
                    for cam, info in self.camera_info.items():
                        if info.get('rover') == r_img and m in info.get('modes', set()):
                            suitable_rover_camera_pairs.append((r_img, cam))

                for r_img, i in suitable_rover_camera_pairs:
                    curr_w = rover_at.get(r_img)
                    if curr_w is None: continue

                    image_points = self.objective_visibility.get(o, set())
                    calib_target = self.camera_info[i].get('calib_target')
                    calib_points = self.objective_visibility.get(calib_target, set()) if calib_target else set()

                    if not image_points or not comm_points: continue

                    # Cost to get image + Cost to communicate sequence
                    cost_get_image_and_comm = float('inf')

                    # Sub-Option 2a: Image -> Communicate (if calibrated)
                    if (i, r_img) in calibrated_cameras:
                        min_nav_curr_to_img = float('inf')
                        best_img_wp = None
                        for img_wp in image_points:
                            nav = self.rover_shortest_paths[r_img].get((curr_w, img_wp), float('inf'))
                            if nav < min_nav_curr_to_img:
                                min_nav_curr_to_img = nav
                                best_img_wp = img_wp

                        if best_img_wp is not None and min_nav_curr_to_img != float('inf'):
                            min_nav_img_to_comm = min((self.rover_shortest_paths[r_img].get((best_img_wp, comm_wp), float('inf')) for comm_wp in comm_points), default=float('inf'))
                            if min_nav_img_to_comm != float('inf'):
                                cost_get_image_and_comm = min(cost_get_image_and_comm, min_nav_curr_to_img + 1 + min_nav_img_to_comm + 1) # Nav + Image + Nav + Comm

                    # Sub-Option 2b: Calibrate -> Image -> Communicate (always possible if points exist)
                    if calib_points:
                        min_nav_curr_to_calib = float('inf')
                        best_calib_wp = None
                        for calib_wp in calib_points:
                            nav = self.rover_shortest_paths[r_img].get((curr_w, calib_wp), float('inf'))
                            if nav < min_nav_curr_to_calib:
                                min_nav_curr_to_calib = nav
                                best_calib_wp = calib_wp

                        if best_calib_wp is not None and min_nav_curr_to_calib != float('inf'):
                            min_nav_calib_to_img = float('inf')
                            best_img_wp_after_calib = None
                            for img_wp in image_points:
                                nav = self.rover_shortest_paths[r_img].get((best_calib_wp, img_wp), float('inf'))
                                if nav < min_nav_calib_to_img:
                                    min_nav_calib_to_img = nav
                                    best_img_wp_after_calib = img_wp

                            if best_img_wp_after_calib is not None and min_nav_calib_to_img != float('inf'):
                                min_nav_img_to_comm_after_calib = min((self.rover_shortest_paths[r_img].get((best_img_wp_after_calib, comm_wp), float('inf')) for comm_wp in comm_points), default=float('inf'))
                                if min_nav_img_to_comm_after_calib != float('inf'):
                                    cost_get_image_and_comm = min(cost_get_image_and_comm, min_nav_curr_to_calib + 1 + min_nav_calib_to_img + 1 + min_nav_img_to_comm_after_calib + 1) # Nav + Calib + Nav + Image + Nav + Comm

                    cost_om = min(cost_om, cost_get_image_and_comm)

                if cost_om == float('inf'):
                    return float('inf')
                h += cost_om

        return h
