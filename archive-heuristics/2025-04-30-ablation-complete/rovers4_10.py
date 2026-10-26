from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define a large value to represent infinite cost or unreachable states
INF = float('inf')

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve
    each unachieved goal fact independently and sums these estimates.
    It considers the steps needed to collect data (sample or image),
    including navigation and calibration (for images), and the steps
    needed to communicate the data, including navigation to a communication
    waypoint. Shortest path navigation costs are precomputed using BFS.

    # Assumptions
    - Each unachieved goal can be pursued independently.
    - Resource constraints like store capacity (beyond needing one empty slot for sampling)
      and camera calibration being consumed are simplified or ignored.
    - The heuristic estimates the minimum cost for *any* capable rover/camera
      to achieve the goal component (collect data, communicate data).
    - Navigation cost between waypoints is the shortest path distance (number of `navigate` actions).
    - Other actions (sample, drop, calibrate, take_image, communicate) cost 1.
    - If a goal component (like collecting data or reaching a communication point)
      is impossible for any available rover, a large penalty is added.

    # Heuristic Initialization
    - Parse static facts and initial state to identify all objects by type.
    - Parse static facts to extract:
        - Lander location.
        - Rover equipment (soil, rock, imaging).
        - Rover stores.
        - Camera capabilities (modes, calibration targets, on which rover).
        - Objective visibility from waypoints.
        - Waypoint visibility (for communication).
        - Rover traversal capabilities between waypoints.
    - Build traversal graphs for each rover based on `can_traverse` facts.
    - Precompute all-pairs shortest paths for each rover's traversal graph using BFS.
    - Store the set of goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to get the current location of each rover,
       which data/images have been collected, which cameras are calibrated,
       which stores are full, and where samples are located.
    3. Identify the set of waypoints visible *from* the lander (communication waypoints).
    4. For each goal fact specified in the problem:
       - If the goal fact is already true in the current state, add 0 to the total cost.
       - If the goal fact is `(communicated_soil_data ?w)`:
         - Estimate the minimum cost to achieve this goal by considering two options:
           - Option A: Soil data `(have_soil_analysis ?r ?w)` already exists for some rover `r`. Calculate the minimum cost for any such rover to navigate from its current location to a communication waypoint and communicate.
           - Option B: Soil data needs to be collected. Calculate the minimum cost for any soil-equipped rover to navigate from its current location to `w`, sample, navigate from waypoint `w` to a communication waypoint, and communicate.
         - The cost for this goal is the minimum of Option A (if applicable) and Option B. Add a large penalty if neither is possible.
       - If the goal fact is `(communicated_rock_data ?w)`:
         - Estimate cost similarly to soil data, using rock-specific predicates and equipment.
       - If the goal fact is `(communicated_image_data ?o ?m)`:
         - Estimate the minimum cost to achieve this goal by considering two options:
           - Option A: Image data `(have_image ?r ?o ?m)` already exists for some rover `r`. Calculate the minimum cost for any such rover to navigate from its current location to a communication waypoint and communicate.
           - Option B: Image data needs to be taken. Calculate the minimum cost for any imaging-equipped rover with a suitable camera to navigate from its current location to an image waypoint `p` visible from `o`, calibrate the camera (including navigation if needed), take the image, navigate from `p` to a communication waypoint, and communicate.
         - The cost for this goal is the minimum of Option A (if applicable) and Option B. Add a large penalty if neither is possible.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Collect all objects by type ---
        # This is a simplified approach assuming relevant objects appear in facts.
        self.objects_by_type = defaultdict(set)
        # Define mapping from predicate argument position to type
        # based on the PDDL domain definition.
        predicate_arg_types_map = {
            'at': {0: 'rover', 1: 'waypoint'},
            'at_lander': {0: 'lander', 1: 'waypoint'},
            'can_traverse': {0: 'rover', 1: 'waypoint', 2: 'waypoint'},
            'equipped_for_soil_analysis': {0: 'rover'},
            'equipped_for_rock_analysis': {0: 'rover'},
            'equipped_for_imaging': {0: 'rover'},
            'empty': {0: 'store'},
            'have_rock_analysis': {0: 'rover', 1: 'waypoint'},
            'have_soil_analysis': {0: 'rover', 1: 'waypoint'},
            'full': {0: 'store'},
            'calibrated': {0: 'camera', 1: 'rover'},
            'supports': {0: 'camera', 1: 'mode'},
            'visible': {0: 'waypoint', 1: 'waypoint'},
            'have_image': {0: 'rover', 1: 'objective', 2: 'mode'},
            'communicated_soil_data': {0: 'waypoint'},
            'communicated_rock_data': {0: 'waypoint'},
            'communicated_image_data': {0: 'objective', 1: 'mode'},
            'at_soil_sample': {0: 'waypoint'},
            'at_rock_sample': {0: 'waypoint'},
            'visible_from': {0: 'objective', 1: 'waypoint'},
            'store_of': {0: 'store', 1: 'rover'},
            'calibration_target': {0: 'camera', 1: 'objective'},
            'on_board': {0: 'camera', 1: 'rover'},
        }

        for fact in list(static_facts) + list(initial_state):
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]
            if predicate in predicate_arg_types_map:
                type_map = predicate_arg_types_map[predicate]
                for i, arg in enumerate(args):
                    if i in type_map:
                        self.objects_by_type[type_map[i]].add(arg)

        self.rovers = list(self.objects_by_type['rover'])
        self.waypoints = list(self.objects_by_type['waypoint'])
        self.cameras = list(self.objects_by_type['camera'])
        self.objectives = list(self.objects_by_type['objective'])
        self.modes = list(self.objects_by_type['mode'])
        self.stores = list(self.objects_by_type['store'])
        self.landers = list(self.objects_by_type['lander'])


        # --- Extract Static Information ---
        self.lander_waypoint = None
        self.rover_equipment = defaultdict(set) # rover -> {equipment_type_str}
        self.rover_stores = defaultdict(list) # rover -> [store]
        self.camera_info = defaultdict(dict) # camera -> {'rover': rover, 'modes': {mode}, 'cal_target': objective}
        self.objective_visibility = defaultdict(set) # objective -> {waypoint}
        self.waypoint_visibility = defaultdict(set) # waypoint -> {waypoint} # Stores (w1, w2) if (visible w1 w2)
        self.rover_traversal_graph = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> {waypoint} # Stores (w_from, w_to) if (can_traverse r w_from w_to)

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at_lander' and len(args) == 2:
                self.lander_waypoint = args[1]
            elif predicate.startswith('equipped_for_') and len(args) == 1:
                equipment_type = predicate.split('_')[2] # 'soil', 'rock', 'imaging'
                self.rover_equipment[args[0]].add(equipment_type)
            elif predicate == 'store_of' and len(args) == 2:
                self.rover_stores[args[1]].append(args[0])
            elif predicate == 'supports' and len(args) == 2:
                camera, mode = args
                if 'modes' not in self.camera_info[camera]:
                    self.camera_info[camera]['modes'] = set()
                self.camera_info[camera]['modes'].add(mode)
            elif predicate == 'calibration_target' and len(args) == 2:
                camera, target = args
                self.camera_info[camera]['cal_target'] = target
            elif predicate == 'on_board' and len(args) == 2:
                camera, rover = args
                self.camera_info[camera]['rover'] = rover
            elif predicate == 'visible_from' and len(args) == 2:
                objective, waypoint = args
                self.objective_visibility[objective].add(waypoint)
            elif predicate == 'visible' and len(args) == 2:
                w1, w2 = args
                self.waypoint_visibility[w1].add(w2)
                # Domain says visible is bidirectional, but predicate is (visible w1 w2).
                # Communicate action uses (visible rover_pos lander_pos).
                # So we need waypoints X such that (visible X lander_waypoint) is true.
                # This means lander_waypoint is in the set of waypoints visible *from* X.
                # Our waypoint_visibility stores w -> {points visible from w}.
                # This parsing is correct for that.

            elif predicate == 'can_traverse' and len(args) == 3:
                rover, w_from, w_to = args
                self.rover_traversal_graph[rover][w_from].add(w_to)


        # --- Precompute Shortest Paths for each rover ---
        self.rover_shortest_paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: INF))) # rover -> start_w -> end_w -> dist

        for rover in self.rovers:
            # Use all waypoints defined in the problem as potential nodes in the graph
            rover_waypoints = set(self.waypoints)

            for start_w in rover_waypoints:
                # Ensure start_w is a valid waypoint
                if start_w not in self.waypoints: continue

                if start_w not in self.rover_shortest_paths[rover]:
                     self.rover_shortest_paths[rover][start_w] = {} # Initialize dict for this start

                dist = {w: INF for w in rover_waypoints}
                dist[start_w] = 0
                q = deque([start_w])

                while q:
                    curr_w = q.popleft()

                    # Find neighbors: waypoints reachable *from* curr_w by this rover
                    neighbors = self.rover_traversal_graph.get(rover, {}).get(curr_w, set())

                    for next_w in neighbors:
                        if dist[next_w] == INF:
                            dist[next_w] = dist[curr_w] + 1
                            q.append(next_w)

                # Store distances from start_w to all reachable waypoints
                for w in rover_waypoints:
                     self.rover_shortest_paths[rover][start_w][w] = dist[w]


        # --- Store Goal Facts ---
        self.goal_facts = set(self.goals)

    def get_min_dist(self, rover, start_w, target_waypoints):
        """Helper to find min distance from start_w to any waypoint in target_waypoints for a rover."""
        if rover not in self.rover_shortest_paths or start_w not in self.rover_shortest_paths[rover]:
            return INF # Rover cannot navigate from start_w (or rover doesn't exist/no graph built)

        min_d = INF
        distances_from_start = self.rover_shortest_paths[rover].get(start_w, {}) # Use .get for safety

        for target_w in target_waypoints:
            min_d = min(min_d, distances_from_start.get(target_w, INF))
        return min_d

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Parse Current State ---
        current_rover_pos = {} # rover -> waypoint
        current_have_soil = set() # (rover, waypoint)
        current_have_rock = set() # (rover, waypoint)
        current_have_image = set() # (rover, objective, mode)
        current_calibrated = set() # (camera, rover)
        current_store_full = set() # store
        current_soil_samples_at = set() # waypoint
        current_rock_samples_at = set() # waypoint

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                current_rover_pos[args[0]] = args[1]
            elif predicate == 'have_soil_analysis' and len(args) == 2:
                current_have_soil.add(tuple(args))
            elif predicate == 'have_rock_analysis' and len(args) == 2:
                current_have_rock.add(tuple(args))
            elif predicate == 'have_image' and len(args) == 3:
                current_have_image.add(tuple(args))
            elif predicate == 'calibrated' and len(args) == 2:
                current_calibrated.add(tuple(args))
            elif predicate == 'full' and len(args) == 1:
                current_store_full.add(args[0])
            elif predicate == 'at_soil_sample' and len(args) == 1:
                current_soil_samples_at.add(args[0])
            elif predicate == 'at_rock_sample' and len(args) == 1:
                current_rock_samples_at.add(args[0])

        # --- Identify Communication Waypoints ---
        # Waypoints X such that (visible X lander_waypoint) is true.
        # This means lander_waypoint is in the set of waypoints visible *from* X.
        comm_waypoints = {w for w, neighbors in self.waypoint_visibility.items() if self.lander_waypoint in neighbors}


        # --- Estimate Cost for Each Ungoaled Fact ---
        for goal_fact in self.goal_facts:
            if goal_fact in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact)
            goal_type = parts[0]
            goal_params = parts[1:]

            current_goal_cost = INF # Cost estimate for this specific goal fact

            if goal_type == 'communicated_soil_data' and len(goal_params) == 1:
                w = goal_params[0]
                # Option 1: Data already exists
                rovers_with_data = [r for r in self.rovers if (r, w) in current_have_soil]
                if rovers_with_data:
                    min_comm_nav = INF
                    for r in rovers_with_data:
                        rover_pos = current_rover_pos.get(r)
                        if rover_pos:
                            min_comm_nav = min(min_comm_nav, self.get_min_dist(r, rover_pos, comm_waypoints))
                    if min_comm_nav != INF:
                        current_goal_cost = min(current_goal_cost, min_comm_nav + 1) # +1 for communicate

                # Option 2: Data needs to be collected
                soil_equipped_rovers = [r for r in self.rovers if 'soil' in self.rover_equipment.get(r, set())]
                if soil_equipped_rovers:
                    min_total_collect_comm_cost = INF
                    for r in soil_equipped_rovers:
                        rover_pos = current_rover_pos.get(r)
                        if rover_pos:
                            dist_to_sample = self.get_min_dist(r, rover_pos, {w})
                            if dist_to_sample != INF:
                                # Cost to sample: nav + sample
                                cost_to_sample = dist_to_sample + 1
                                # Cost to communicate from sample location (w): nav + communicate
                                dist_from_sample_to_comm = self.get_min_dist(r, w, comm_waypoints)
                                if dist_from_sample_to_comm != INF:
                                    cost_to_communicate = dist_from_sample_to_comm + 1
                                    min_total_collect_comm_cost = min(min_total_collect_comm_cost, cost_to_sample + cost_to_communicate)
                    current_goal_cost = min(current_goal_cost, min_total_collect_comm_cost)

            elif goal_type == 'communicated_rock_data' and len(goal_params) == 1:
                w = goal_params[0]
                # Option 1: Data already exists
                rovers_with_data = [r for r in self.rovers if (r, w) in current_have_rock]
                if rovers_with_data:
                    min_comm_nav = INF
                    for r in rovers_with_data:
                        rover_pos = current_rover_pos.get(r)
                        if rover_pos:
                            min_comm_nav = min(min_comm_nav, self.get_min_dist(r, rover_pos, comm_waypoints))
                    if min_comm_nav != INF:
                        current_goal_cost = min(current_goal_cost, min_comm_nav + 1) # +1 for communicate

                # Option 2: Data needs to be collected
                rock_equipped_rovers = [r for r in self.rovers if 'rock' in self.rover_equipment.get(r, set())]
                if rock_equipped_rovers:
                    min_total_collect_comm_cost = INF
                    for r in rock_equipped_rovers:
                        rover_pos = current_rover_pos.get(r)
                        if rover_pos:
                            dist_to_sample = self.get_min_dist(r, rover_pos, {w})
                            if dist_to_sample != INF:
                                # Cost to sample: nav + sample
                                cost_to_sample = dist_to_sample + 1
                                # Cost to communicate from sample location (w): nav + communicate
                                dist_from_sample_to_comm = self.get_min_dist(r, w, comm_waypoints)
                                if dist_from_sample_to_comm != INF:
                                    cost_to_communicate = dist_from_sample_to_comm + 1
                                    min_total_collect_comm_cost = min(min_total_collect_comm_cost, cost_to_sample + cost_to_communicate)
                    current_goal_cost = min(current_goal_cost, min_total_collect_comm_cost)

            elif goal_type == 'communicated_image_data' and len(goal_params) == 2:
                o, m = goal_params
                # Option 1: Image already exists
                rovers_with_image = [r for r in self.rovers if (r, o, m) in current_have_image]
                if rovers_with_image:
                    min_comm_nav = INF
                    for r in rovers_with_image:
                        rover_pos = current_rover_pos.get(r)
                        if rover_pos:
                            min_comm_nav = min(min_comm_nav, self.get_min_dist(r, rover_pos, comm_waypoints))
                    if min_comm_nav != INF:
                        current_goal_cost = min(current_goal_cost, min_comm_nav + 1) # +1 for communicate

                # Option 2: Image needs to be taken
                imaging_rovers = [r for r in self.rovers if 'imaging' in self.rover_equipment.get(r, set())]
                if imaging_rovers:
                    min_total_take_comm_cost = INF
                    image_waypoints = self.objective_visibility.get(o, set())

                    if image_waypoints:
                        for r in imaging_rovers:
                            rover_pos = current_rover_pos.get(r)
                            if not rover_pos: continue # Rover location unknown

                            # Find camera supporting mode m on this rover
                            rover_cameras = [cam for cam in self.cameras if self.camera_info.get(cam, {}).get('rover') == r and m in self.camera_info.get(cam, {}).get('modes', set())]
                            if not rover_cameras: continue # Rover cannot take this image

                            for cam in rover_cameras:
                                # Find best waypoint 'p' to take image from
                                min_nav_to_image_waypoint = INF
                                best_p = None
                                for p in image_waypoints:
                                    dist = self.get_min_dist(r, rover_pos, {p})
                                    if dist != INF and dist < min_nav_to_image_waypoint:
                                        min_nav_to_image_waypoint = dist
                                        best_p = p

                                if best_p is None: continue # Cannot reach any image waypoint for this rover/camera

                                # Cost to take image: nav + (calibration if needed) + take_image
                                cost_to_take = min_nav_to_image_waypoint + 1 # +1 for take_image

                                # Add calibration cost if needed
                                if (cam, r) not in current_calibrated:
                                    cost_to_take += 1 # +1 for calibrate
                                    cal_target = self.camera_info.get(cam, {}).get('cal_target')
                                    if cal_target:
                                        cal_waypoints = self.objective_visibility.get(cal_target, set())
                                        if cal_waypoints:
                                            # Find the cal_waypoint 'w' that minimizes dist(p, w) + dist(w, p)
                                            min_cal_nav_cost = INF
                                            for w_cal in cal_waypoints:
                                                dist_p_w = self.get_min_dist(r, best_p, {w_cal})
                                                dist_w_p = self.get_min_dist(r, w_cal, {best_p})
                                                if dist_p_w != INF and dist_w_p != INF:
                                                    min_cal_nav_cost = min(min_cal_nav_cost, dist_p_w + dist_w_p)

                                            if min_cal_nav_cost != INF:
                                                 cost_to_take += min_cal_nav_cost
                                            else:
                                                 cost_to_take = INF # Cannot calibrate (unreachable cal waypoint)
                                        else:
                                            cost_to_take = INF # Cannot calibrate (no visible spots for target)
                                    else:
                                        cost_to_take = INF # Cannot calibrate (no target defined)

                                if cost_to_take == INF: continue

                                # Cost to communicate from image location (best_p): nav + communicate
                                dist_from_image_to_comm = self.get_min_dist(r, best_p, comm_waypoints)
                                if dist_from_image_to_comm != INF:
                                    cost_to_communicate = dist_from_image_to_comm + 1
                                    min_total_take_comm_cost = min(min_total_take_comm_cost, cost_to_take + cost_to_communicate)

                    current_goal_cost = min(current_goal_cost, min_total_take_comm_cost)

            # Add the cost for this goal to the total, using a large penalty if impossible
            total_cost += current_goal_cost if current_goal_cost != INF else 1000 # Penalty for impossible goals

        return total_cost
