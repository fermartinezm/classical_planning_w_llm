from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
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

def build_distance_graph(waypoints, can_traverse_facts, rover_name):
    """Build adjacency list for BFS based on can_traverse facts for a specific rover."""
    graph = {wp: set() for wp in waypoints}
    for fact in can_traverse_facts:
        parts = get_parts(fact)
        if len(parts) == 4 and parts[0] == 'can_traverse' and parts[1] == rover_name:
            _, r, wp1, wp2 = parts
            graph[wp1].add(wp2)
    return graph

def bfs(graph, start_node):
    """Perform BFS from start_node and return distances to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

def precompute_distances(waypoints, can_traverse_facts, rovers):
    """Precompute shortest path distances for each rover."""
    rover_dist_graphs = {}
    for rover in rovers:
        graph = build_distance_graph(waypoints, can_traverse_facts, rover)
        rover_dist_graphs[rover] = {}
        for start_wp in waypoints:
            rover_dist_graphs[rover][start_wp] = bfs(graph, start_wp)
    return rover_dist_graphs


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    ungratified goal conditions. It does this by summing the minimum estimated
    cost for each ungratified goal independently. The cost for each goal
    includes the necessary sampling/imaging actions, communication actions,
    and estimated travel costs (shortest path distance) between relevant
    waypoints.

    # Assumptions
    - The heuristic is additive: the total cost is the sum of costs for
      individual goals.
    - Resource constraints (like store capacity beyond the immediate need
      to drop before sampling) and concurrent rover activities are largely
      ignored when calculating the cost for a single goal.
    - Travel cost between waypoints for a specific rover is the shortest
      path distance using its `can_traverse` capabilities.
    - For soil/rock goals, if the sample is not yet 'have_soil_analysis'
      or 'have_rock_analysis' for *any* rover, it must be sampled. This
      requires the sample to exist at the waypoint in the current state.
    - For image goals, if the image is not yet 'have_image' for *any*
      rover, it must be taken. This requires calibration if the camera
      chosen for the task is not currently calibrated. Taking an image
      uncalibrates the camera. The heuristic simplifies this by only
      counting the calibration cost if the camera is not calibrated
      *in the current state* when considering the need to take the image
      for this specific goal.
    - If the required sample/image data already exists (held by any rover),
      the sampling/imaging/calibration costs are zero for that goal. The
      cost is then just travel to a communication point + communication.

    # Heuristic Initialization
    - Extracts static information from the task:
        - Rover capabilities (soil, rock, imaging).
        - Rover store mapping.
        - Camera information (on-board rover, supported modes, calibration target).
        - Waypoint visibility for objectives and calibration targets.
        - Lander location and waypoints visible from the lander.
        - Waypoints and `can_traverse` facts for each rover.
    - Precomputes all-pairs shortest path distances between waypoints for each
      rover based on their `can_traverse` facts using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to get dynamic facts (rover locations, store states,
       have_analysis, have_image, calibrated status, sample locations).
    2. Initialize total heuristic cost to 0.
    3. For each goal condition in the task's goals:
        a. If the goal condition is already true in the current state, continue
           to the next goal.
        b. If the goal is `(communicated_soil_data ?w)`:
            - Calculate the minimum cost to achieve this goal from the current state.
            - Check if `(have_soil_analysis ?r' ?w)` is true for *any* rover `?r'`.
            - Find the best rover `?r` (equipped for soil), the best
              communication waypoint `?x` (visible from lander).
            - The cost includes:
                - If no rover has `(have_soil_analysis ?w)`:
                    - Cost of `sample_soil` (1 action).
                    - Cost of `drop` (1 action) if the chosen rover's store is full.
                    - Travel cost from the chosen rover's current location to `?w`.
                    - Travel cost from `?w` to `?x`.
                - Else (some rover has `(have_soil_analysis ?w)`):
                    - Travel cost from the chosen rover's current location to `?x`.
                - Cost of `communicate_soil_data` (1 action).
            - Minimize this cost over all suitable rovers and communication waypoints.
            - If no path exists or sampling is impossible, the cost is infinity.
            - Add the minimum cost to the total heuristic.
        c. If the goal is `(communicated_rock_data ?w)`:
            - Calculate the minimum cost similarly to the soil data goal, using
              rock-specific predicates and capabilities and checking if
              `(have_rock_analysis ?r' ?w)` is true for *any* rover `?r'`.
            - Add the minimum cost to the total heuristic.
        d. If the goal is `(communicated_image_data ?o ?m)`:
            - Calculate the minimum cost to achieve this goal from the current state.
            - Check if `(have_image ?r' ?o ?m)` is true for *any* rover `?r'`.
            - Find the best rover `?r` (equipped for imaging), best
              camera `?i` (on board, supports mode), best calibration waypoint `?w`,
              best image waypoint `?p`, and best communication waypoint `?x`.
            - The cost includes:
                - If no rover has `(have_image ?o ?m)`:
                    - Cost of `take_image` (1 action).
                    - If the chosen camera `?i` is not calibrated in the current state:
                        - Cost of `calibrate` (1 action).
                        - Travel cost from the chosen rover's current location to `?w`.
                        - Travel cost from `?w` to `?p`.
                    - Else (camera is calibrated):
                        - Travel cost from the chosen rover's current location to `?p`.
                    - Travel cost from `?p` to `?x`.
                - Else (some rover has `(have_image ?o ?m)`):
                    - Travel cost from the chosen rover's current location to `?x`.
                - Cost of `communicate_image_data` (1 action).
            - Minimize this cost over all suitable rovers, cameras, calibration,
              image, and communication waypoints.
            - If no path exists or imaging/calibration is impossible, the cost is infinity.
            - Add the minimum cost to the total heuristic.
    4. Return the total heuristic cost. If any individual goal cost was infinity,
       return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract all object types mentioned in static facts and goals
        waypoints = set()
        rovers = set()
landers = set()
        stores = set()
        cameras = set()
        modes = set()
        objectives = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at_lander': landers.add(parts[1]); waypoints.add(parts[2])
            elif pred == 'can_traverse': rovers.add(parts[1]); waypoints.add(parts[2]); waypoints.add(parts[3])
            elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']: rovers.add(parts[1])
            elif pred == 'supports': cameras.add(parts[1]); modes.add(parts[2])
            elif pred == 'visible': waypoints.add(parts[1]); waypoints.add(parts[2])
            elif pred == 'visible_from': objectives.add(parts[1]); waypoints.add(parts[2])
            elif pred == 'store_of': stores.add(parts[1]); rovers.add(parts[2])
            elif pred == 'calibration_target': cameras.add(parts[1]); objectives.add(parts[2])
            elif pred == 'on_board': cameras.add(parts[1]); rovers.add(parts[2])
            # Include types from initial state facts that might be in static (like at_soil_sample)
            elif pred in ['at_soil_sample', 'at_rock_sample']: waypoints.add(parts[1])
            # Include types from initial state facts that might be in static (like at) - though they are dynamic
            elif pred == 'at': rovers.add(parts[1]); waypoints.add(parts[2])


        # Ensure all objects mentioned in goals are considered
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             pred = parts[0]
             if pred == 'communicated_soil_data': waypoints.add(parts[1])
             elif pred == 'communicated_rock_data': waypoints.add(parts[1])
             elif pred == 'communicated_image_data': objectives.add(parts[1]); modes.add(parts[2])

        self.waypoints = list(waypoints)
        self.rovers = list(rovers)
        self.objectives = list(objectives)
        self.modes = list(modes)
        self.cameras = list(cameras)
        self.landers = list(landers) # Assuming only one lander

        # Store static information
        self.rover_capabilities = {r: set() for r in self.rovers}
        self.rover_stores = {} # Maps rover -> store
        self.rover_cameras = {r: [] for r in self.rovers} # Maps rover -> list of cameras
        self.camera_modes = {c: set() for c in self.cameras} # Maps camera -> set of modes
        self.camera_cal_target = {} # Maps camera -> calibration target objective
        self.objective_visible_from = {o: set() for o in self.objectives} # Maps objective -> set of visible waypoints
        self.target_visible_from = {o: set() for o in self.objectives} # Maps objective (target) -> set of visible waypoints
        self.lander_location = None
        self.comm_waypoint_visibility = set() # Waypoints visible from lander location

        can_traverse_facts = [fact for fact in static_facts if match(fact, 'can_traverse', '*', '*', '*')]
        visible_facts = [fact for fact in static_facts if match(fact, 'visible', '*', '*')]
        visible_from_facts = [fact for fact in static_facts if match(fact, 'visible_from', '*', '*')]

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'equipped_for_soil_analysis': self.rover_capabilities[parts[1]].add('soil')
            elif pred == 'equipped_for_rock_analysis': self.rover_capabilities[parts[1]].add('rock')
            elif pred == 'equipped_for_imaging': self.rover_capabilities[parts[1]].add('imaging')
            elif pred == 'store_of': self.rover_stores[parts[2]] = parts[1] # rover -> store
            elif pred == 'on_board': self.rover_cameras[parts[2]].append(parts[1]) # rover -> camera list
            elif pred == 'supports': self.camera_modes[parts[1]].add(parts[2])
            elif pred == 'calibration_target': self.camera_cal_target[parts[1]] = parts[2] # camera -> target objective
            elif pred == 'visible_from':
                 obj, wp = parts[1], parts[2]
                 if obj in self.objective_visible_from: # It's an objective for imaging
                     self.objective_visible_from[obj].add(wp)
                 # Check if it's a calibration target for any camera
                 if obj in self.camera_cal_target.values():
                      self.target_visible_from[obj].add(wp)
            elif pred == 'at_lander': self.lander_location = parts[2]

        # Precompute communication waypoints visible from lander
        if self.lander_location:
             for fact in visible_facts:
                 wp1, wp2 = get_parts(fact)[1], get_parts(fact)[2]
                 if wp1 == self.lander_location: self.comm_waypoint_visibility.add(wp2)
                 if wp2 == self.lander_location: self.comm_waypoint_visibility.add(wp1)

        # Precompute distances
        self.rover_dist_graphs = precompute_distances(self.waypoints, can_traverse_facts, self.rovers)


    def get_distance(self, rover, start_wp, end_wp):
         """Wrapper to get precomputed distance, handling potential missing data."""
         if rover not in self.rover_dist_graphs or start_wp not in self.rover_dist_graphs[rover]:
             return float('inf')
         return self.rover_dist_graphs[rover][start_wp].get(end_wp, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse relevant facts from the current state
        state_parts = [get_parts(fact) for fact in state]
        state_set = set(state) # Use set for fast membership checking

        # Get dynamic state information
        rover_locs = {parts[1]: parts[2] for parts in state_parts if parts and parts[0] == 'at' and parts[1] in self.rovers}
        store_states = {parts[1]: parts[0] for parts in state_parts if parts and parts[0] in ['empty', 'full'] and parts[1] in self.rover_stores.values()}
        have_soil = {(parts[1], parts[2]) for parts in state_parts if parts and parts[0] == 'have_soil_analysis'}
        have_rock = {(parts[1], parts[2]) for parts in state_parts if parts and parts[0] == 'have_rock_analysis'}
        have_img = {(parts[1], parts[2], parts[3]) for parts in state_parts if parts and parts[0] == 'have_image'}
        calibrated_cams = {(parts[1], parts[2]) for parts in state_parts if parts and parts[0] == 'calibrated'}
        soil_samples = {parts[1] for parts in state_parts if parts and parts[0] == 'at_soil_sample'}
        rock_samples = {parts[1] for parts in state_parts if parts and parts[0] == 'at_rock_sample'}

        total_cost = 0

        for goal in self.goals:
            if goal in state_set:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue
            pred = parts[0]

            min_cost_for_goal = float('inf')

            if pred == 'communicated_soil_data':
                w = parts[1]
                # Need to communicate soil data from w
                soil_analysis_exists_anywhere = any(hw == w for (hr, hw) in have_soil)

                # Find best rover and comm waypoint
                for r in self.rovers:
                    if 'soil' not in self.rover_capabilities.get(r, set()): continue # Rover not capable
                    current_r_loc = rover_locs.get(r)
                    if current_r_loc is None: continue # Rover location unknown

                    for x in self.comm_waypoint_visibility:
                        cost = 0
                        effective_start_loc_for_comm = current_r_loc # Default: start from current loc

                        if not soil_analysis_exists_anywhere:
                            # Need to sample
                            if w not in soil_samples:
                                cost = float('inf') # Cannot sample if no sample exists
                            else:
                                cost += 1 # sample_soil
                                r_store = self.rover_stores.get(r)
                                if r_store is None:
                                     cost = float('inf') # Rover has no store?
                                elif store_states.get(r_store, 'empty') == 'full':
                                    cost += 1 # drop

                                if cost != float('inf'):
                                    dist_to_sample = self.get_distance(r, current_r_loc, w)
                                    if dist_to_sample == float('inf'): cost = float('inf')
                                    else: cost += dist_to_sample
                                    effective_start_loc_for_comm = w # After sampling, rover is at w
                        else:
                            # Sample exists somewhere. Assume the chosen rover `r` can get it.
                            # Cost is just getting rover `r` to the communication point.
                            effective_start_loc_for_comm = current_r_loc


                        if cost != float('inf'):
                            # Need to communicate
                            cost += 1 # communicate_soil_data
                            dist_to_comm = self.get_distance(r, effective_start_loc_for_comm, x)
                            if dist_to_comm == float('inf'): cost = float('inf')
                            else: cost += dist_to_comm

                        min_cost_for_goal = min(min_cost_for_goal, cost)

            elif pred == 'communicated_rock_data':
                w = parts[1]
                 # Need to communicate rock data from w
                rock_analysis_exists_anywhere = any(hw == w for (hr, hw) in have_rock)

                # Find best rover and comm waypoint
                for r in self.rovers:
                    if 'rock' not in self.rover_capabilities.get(r, set()): continue # Rover not capable
                    current_r_loc = rover_locs.get(r)
                    if current_r_loc is None: continue # Rover location unknown

                    for x in self.comm_waypoint_visibility:
                        cost = 0
                        effective_start_loc_for_comm = current_r_loc # Default: start from current loc

                        if not rock_analysis_exists_anywhere:
                            # Need to sample
                            if w not in rock_samples:
                                cost = float('inf') # Cannot sample if no sample exists
                            else:
                                cost += 1 # sample_rock
                                r_store = self.rover_stores.get(r)
                                if r_store is None:
                                     cost = float('inf') # Rover has no store?
                                elif store_states.get(r_store, 'empty') == 'full':
                                    cost += 1 # drop

                                if cost != float('inf'):
                                    dist_to_sample = self.get_distance(r, current_r_loc, w)
                                    if dist_to_sample == float('inf'): cost = float('inf')
                                    else: cost += dist_to_sample
                                    effective_start_loc_for_comm = w # After sampling, rover is at w
                        else:
                            # Sample exists somewhere. Assume the chosen rover `r` can get it.
                            # Cost is just getting rover `r` to the communication point.
                            effective_start_loc_for_comm = current_r_loc


                        if cost != float('inf'):
                            # Need to communicate
                            cost += 1 # communicate_rock_data
                            dist_to_comm = self.get_distance(r, effective_start_loc_for_comm, x)
                            if dist_to_comm == float('inf'): cost = float('inf')
                            else: cost += dist_to_comm

                        min_cost_for_goal = min(min_cost_for_goal, cost)

            elif pred == 'communicated_image_data':
                o, m = parts[1], parts[2]
                # Need to communicate image data for objective o, mode m
                image_exists_anywhere = any((hi_o, hi_m) == (o, m) for (hi_r, hi_o, hi_m) in have_img)

                # Find best rover, camera, cal_wp, img_wp, comm_wp
                for r in self.rovers:
                    if 'imaging' not in self.rover_capabilities.get(r, set()): continue # Rover not capable
                    current_r_loc = rover_locs.get(r)
                    if current_r_loc is None: continue # Rover location unknown

                    for i in self.rover_cameras.get(r, []):
                        if m not in self.camera_modes.get(i, set()): continue # Camera doesn't support mode
                        cal_target = self.camera_cal_target.get(i)
                        if cal_target is None: continue # Camera has no calibration target

                        # Find possible calibration waypoints
                        cal_wps = self.target_visible_from.get(cal_target, set())
                        if not cal_wps: continue # No waypoint to calibrate from

                        # Find possible image waypoints
                        img_wps = self.objective_visible_from.get(o, set())
                        if not img_wps: continue # No waypoint to image from

                        # Find possible communication waypoints
                        comm_wps = self.comm_waypoint_visibility
                        if not comm_wps: continue # No waypoint to communicate from

                        for w in cal_wps:
                            for p in img_wps:
                                for x in comm_wps:
                                    cost = 0
                                    effective_start_loc_for_comm = current_r_loc # Default

                                    if not image_exists_anywhere:
                                        # Need to take image
                                        cost += 1 # take_image

                                        calibrated_now = (i, r) in calibrated_cams

                                        if not calibrated_now:
                                            # Need to calibrate
                                            cost += 1 # calibrate
                                            dist_to_cal = self.get_distance(r, current_r_loc, w)
                                            if dist_to_cal == float('inf'): cost = float('inf')
                                            else: cost += dist_to_cal
                                            current_loc_after_cal = w
                                        else:
                                            # Already calibrated
                                            current_loc_after_cal = current_r_loc

                                        if cost != float('inf'):
                                            dist_to_image = self.get_distance(r, current_loc_after_cal, p)
                                            if dist_to_image == float('inf'): cost = float('inf')
                                            else: cost += dist_to_image
                                            effective_start_loc_for_comm = p # After taking image, rover is at p

                                    else:
                                         # Image exists somewhere. Assume the chosen rover `r` can get it.
                                         # Cost is just getting rover `r` to the communication point.
                                         effective_start_loc_for_comm = current_r_loc


                                    if cost != float('inf'):
                                        # Need to communicate
                                        cost += 1 # communicate_image_data
                                        dist_to_comm = self.get_distance(r, effective_start_loc_for_comm, x)
                                        if dist_to_comm == float('inf'): cost = float('inf')
                                        else: cost += dist_to_comm

                                    min_cost_for_goal = min(min_cost_for_goal, cost)

            # If min_cost_for_goal is still infinity, the goal is unreachable
            if min_cost_for_goal == float('inf'):
                return float('inf') # Problem is unsolvable from this state

            total_cost += min_cost_for_goal

        return total_cost
