import collections

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach a goal state by summing up the
    estimated costs for each unachieved goal fact. For each unachieved goal
    (communicating soil data, rock data, or image data), it finds the minimum
    cost for any suitable rover to complete the necessary steps:
    1. If the required data (soil sample, rock sample, or image) is not yet
       collected by any rover, estimate the cost for a suitable rover to
       collect it. This involves navigation to the sample/image location,
       potentially dropping a full store (for samples), calibration (for images),
       and the sample/take_image action.
    2. If the required data is already collected by a specific rover, or after
       estimating the cost to collect it, estimate the cost for that rover
       to communicate the data. This involves navigation to a communication
       waypoint (visible from the lander) and the communicate action.
    The estimated cost for a goal is the minimum of these costs over all
    suitable rovers (and cameras for imaging). Navigation costs are estimated
    using Breadth-First Search (BFS) on the rover's traversal graph.

    Assumptions:
    - The PDDL instance is well-formed and solvable.
    - Samples are only removed by the corresponding sample action.
    - Rovers do not lose capabilities or cameras.
    - The structure of facts is consistent with the provided examples.
    - The heuristic uses a large constant (1000) to represent impossibility
      or very high cost for a sub-task; summing these up is intended.
    - Waypoint visibility is symmetric (if A visible from B, B visible from A).
    - Calibration targets are objectives.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the task definition
    to build data structures used for efficient heuristic computation:
    - Mapping of rovers to capabilities (soil, rock, imaging).
    - Mapping of rovers to their stores.
    - Mapping of rovers to cameras on board.
    - Mapping of cameras to supported modes.
    - Mapping of cameras to calibration targets (objectives).
    - Adjacency lists for waypoint visibility.
    - Mapping of objectives/targets to waypoints visible from them.
    - Navigation graphs (adjacency lists) for each rover based on traversability.
    - Set of communication points (waypoints visible from the lander).
    - Set of all rover names.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Initialize the total heuristic value `h` to 0.
    3. Identify the set of goal facts that are not true in the current state (`unachieved_goals`).
    4. For each `goal` in `unachieved_goals`:
        a. Initialize `min_total_cost_for_goal` to a large value (1000).
        b. Determine the type of goal (soil, rock, or image communication).
        c. If it's a soil communication goal `(communicated_soil_data ?w)`:
            i. Check if `(have_soil_analysis ?r ?w)` is true for any rover `?r` in the current state.
            ii. If data is collected by `rover_with_sample`:
                - Calculate the cost for `rover_with_sample` to navigate from its current location to a communication point and communicate (navigation cost + 1). Minimize over all communication points.
                - Update `min_total_cost_for_goal` with this communication cost.
            iii. If data is not collected:
                - Iterate through all rovers `?r` equipped for soil analysis.
                - For each such rover, estimate the cost to collect the sample at `?w`:
                    - Navigation from current location to `?w`.
                    - Cost to drop store if full (1 if full, 0 if empty).
                    - Sample action cost (1).
                    - Sum these costs (`cost_sample`).
                - If sampling is possible (`cost_sample` is finite):
                    - Estimate the cost for the same rover `?r` to navigate from `?w` (where it is after sampling) to a communication point and communicate (navigation cost + 1). Minimize over all communication points.
                    - Sum `cost_sample` and `cost_communicate` to get the total cost for this rover.
                    - Update `min_total_cost_for_goal` with the minimum total cost found across all suitable rovers.
        d. If it's a rock communication goal `(communicated_rock_data ?w)`:
            i. Similar logic as soil data, using rock-specific predicates and requirements.
        e. If it's an image communication goal `(communicated_image_data ?o ?m)`:
            i. Check if `(have_image ?r ?o ?m)` is true for any rover `?r` in the current state.
            ii. If data is collected by `rover_with_image`:
                - Calculate the cost for `rover_with_image` to navigate from its current location to a communication point and communicate (navigation cost + 1). Minimize over all communication points.
                - Update `min_total_cost_for_goal` with this communication cost.
            iii. If data is not collected:
                - Iterate through all rovers `?r` equipped for imaging and cameras `?i` on board `?r` supporting mode `?m`.
                - For each such (rover, camera) pair, estimate the cost to take the image:
                    - Find the best waypoint `?p` visible from `?o` (minimizing navigation from current location). Cost is `nav(current, p)`.
                    - If calibration is needed (camera `?i` has a calibration target `?t`):
                        - Find the best waypoint `?w` visible from `?t` (minimizing navigation from `?p`). Cost is `nav(p, w)`.
                        - Add calibration action cost (1).
                        - Add navigation cost back from `?w` to `?p` (`nav(w, p)`).
                        - Total calibration sequence cost is `nav(p, w) + 1 + nav(w, p)`.
                    - Add take_image action cost (1).
                    - Total imaging cost is `nav(current, p) + cost_calibration_sequence + 1`.
                - If imaging is possible (`cost_image` is finite):
                    - The rover is at `?p` after taking the image. Estimate the cost for the same rover `?r` to navigate from `?p` to a communication point and communicate (navigation cost + 1). Minimize over all communication points.
                    - Sum `cost_image` and `cost_communicate` to get the total cost for this (rover, camera) pair.
                    - Update `min_total_cost_for_goal` with the minimum total cost found across all suitable (rover, camera) pairs.
        f. Add `min_total_cost_for_goal` to the total heuristic value `h`.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task, static_info):
        """
        Pre-processes static information from the task.
        """
        self.task = task
        self.static_info = static_info
        self.large_cost = 1000 # Represents impossibility or very high cost

        # Data structures for static info
        self.lander_location = None
        self.rover_capabilities = collections.defaultdict(set)
        self.store_map = {} # rover -> store
        self.rover_cameras = collections.defaultdict(set) # rover -> set of cameras
        self.camera_modes = collections.defaultdict(set) # camera -> set of modes
        self.calibration_targets = {} # camera -> target (objective)
        self.visible_waypoints = collections.defaultdict(set) # waypoint -> set of visible waypoints
        self.visible_from_objective = collections.defaultdict(set) # objective -> set of visible waypoints
        self.rover_traversal_graphs = collections.defaultdict(lambda: collections.defaultdict(set)) # rover -> waypoint -> set of traversable waypoints
        self.communication_points = set() # set of waypoints visible from lander
        self.all_rovers = set() # set of all rover names

        self._parse_static_info(static_info)
        self._parse_initial_state_for_rovers(task.initial_state) # Ensure all rovers are captured
        self._precompute_communication_points()

    def _parse_static_info(self, static_info):
        """Parses the static facts."""
        for fact_str in static_info:
            parts = self._parse_fact(fact_str)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]

            if predicate == 'at_lander':
                # (at_lander ?x - lander ?y - waypoint)
                if len(parts) > 2:
                    self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                # (equipped_for_soil_analysis ?r - rover)
                if len(parts) > 1:
                    self.rover_capabilities[parts[1]].add('soil')
                    self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                # (equipped_for_rock_analysis ?r - rover)
                if len(parts) > 1:
                    self.rover_capabilities[parts[1]].add('rock')
                    self.all_rovers.add(parts[1])
            elif predicate == 'equipped_for_imaging':
                # (equipped_for_imaging ?r - rover)
                if len(parts) > 1:
                    self.rover_capabilities[parts[1]].add('imaging')
                    self.all_rovers.add(parts[1])
            elif predicate == 'store_of':
                # (store_of ?s - store ?r - rover)
                if len(parts) > 2:
                    self.store_map[parts[2]] = parts[1] # rover -> store
            elif predicate == 'on_board':
                # (on_board ?i - camera ?r - rover)
                if len(parts) > 2:
                    self.rover_cameras[parts[2]].add(parts[1]) # rover -> camera
            elif predicate == 'supports':
                # (supports ?c - camera ?m - mode)
                if len(parts) > 2:
                    self.camera_modes[parts[1]].add(parts[2]) # camera -> mode
            elif predicate == 'calibration_target':
                # (calibration_target ?i - camera ?o - objective)
                if len(parts) > 2:
                    self.calibration_targets[parts[1]] = parts[2] # camera -> target (objective)
            elif predicate == 'visible':
                # (visible ?w - waypoint ?p - waypoint)
                if len(parts) > 2:
                    self.visible_waypoints[parts[1]].add(parts[2])
                    self.visible_waypoints[parts[2]].add(parts[1]) # Assuming visible is symmetric
            elif predicate == 'visible_from':
                # (visible_from ?o - objective ?w - waypoint)
                if len(parts) > 2:
                    self.visible_from_objective[parts[1]].add(parts[2]) # objective -> waypoint
            elif predicate == 'can_traverse':
                # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                if len(parts) > 3:
                    self.rover_traversal_graphs[parts[1]][parts[2]].add(parts[3]) # rover -> waypoint -> set of traversable waypoints
                    self.all_rovers.add(parts[1])

    def _parse_initial_state_for_rovers(self, initial_state):
         """Adds rovers found in the initial state to the all_rovers set."""
         for fact_str in initial_state:
             parts = self._parse_fact(fact_str)
             if not parts: continue
             if parts[0] == 'at' and len(parts) > 1:
                 # (at ?x - rover ?y - waypoint)
                 self.all_rovers.add(parts[1])


    def _precompute_communication_points(self):
        """Finds all waypoints visible from the lander location."""
        if self.lander_location and self.lander_location in self.visible_waypoints:
             # Communication requires (visible ?x ?y) where ?x is rover loc and ?y is lander loc.
             # So communication points are waypoints ?x such that (visible ?x lander_loc).
             # Since visible is assumed symmetric, this is the set of waypoints visible from lander_loc.
             self.communication_points = set(self.visible_waypoints[self.lander_location])

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a list of strings."""
        # Remove surrounding parentheses and split by spaces
        fact_string = fact_string.strip()
        if not fact_string.startswith('(') or not fact_string.endswith(')'):
             return [] # Not a valid fact string format
        return fact_string[1:-1].split()

    def _get_object_name(self, fact_string, index):
        """Extracts the object name at a specific index from a fact string."""
        parts = self._parse_fact(fact_string)
        if 0 <= index < len(parts):
            return parts[index]
        return None

    def _get_rover_location(self, state, rover):
        """Finds the current waypoint of a given rover in the state."""
        for fact in state:
            if fact.startswith(f'(at {rover} '):
                # Fact is like '(at rover1 waypoint1)'
                parts = self._parse_fact(fact)
                if len(parts) > 2 and parts[0] == 'at' and parts[1] == rover:
                    return parts[2]
        return None # Rover location not found in state

    def shortest_path(self, graph, start, end):
        """
        Computes the shortest path distance between two waypoints using BFS.
        Returns self.large_cost if end is unreachable from start or inputs are invalid.
        """
        if start is None or end is None or start not in graph:
             return self.large_cost
        if start == end:
            return 0

        queue = collections.deque([(start, 0)])
        visited = {start}

        while queue:
            current_wp, dist = queue.popleft()

            if current_wp == end:
                return dist

            if current_wp in graph:
                for neighbor in graph[current_wp]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return self.large_cost # Unreachable

    def __call__(self, state, task):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        if task.goal_reached(state):
            return 0

        h = 0
        unachieved_goals = task.goals - state

        for goal in unachieved_goals:
            min_total_cost_for_goal = self.large_cost

            parts = self._parse_fact(goal)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'communicated_soil_data' and len(parts) > 1:
                w = parts[1]
                # Check if sample is already collected by any rover
                rover_with_sample = None
                for r in self.all_rovers:
                    if f'(have_soil_analysis {r} {w})' in state:
                        rover_with_sample = r
                        break

                if rover_with_sample:
                    # Sample already collected by this specific rover
                    r = rover_with_sample
                    current_loc = self._get_rover_location(state, r)
                    if current_loc is None: continue # Should not happen

                    # Cost to communicate sample
                    min_nav_to_comm = self.large_cost
                    for comm_pt in self.communication_points:
                        nav_cost = self.shortest_path(self.rover_traversal_graphs.get(r, {}), current_loc, comm_pt)
                        min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                    if min_nav_to_comm < self.large_cost:
                        cost_communicate = min_nav_to_comm + 1
                        min_total_cost_for_goal = min(min_total_cost_for_goal, cost_communicate)
                else:
                    # Sample not collected by any rover, need to collect and communicate
                    for r in self.all_rovers:
                        if 'soil' in self.rover_capabilities.get(r, set()):
                            current_loc = self._get_rover_location(state, r)
                            if current_loc is None: continue

                            # Cost to collect sample
                            nav_to_sample = self.shortest_path(self.rover_traversal_graphs.get(r, {}), current_loc, w)
                            if nav_to_sample >= self.large_cost: continue # Cannot reach sample location

                            store = self.store_map.get(r)
                            drop_cost = 1 if store and (f'(full {store})' in state) else 0
                            cost_sample = nav_to_sample + drop_cost + 1 # nav + drop + sample
                            location_after_sample = w # Rover is at sample location after sampling

                            # Cost to communicate sample
                            min_nav_to_comm = self.large_cost
                            for comm_pt in self.communication_points:
                                nav_cost = self.shortest_path(self.rover_traversal_graphs.get(r, {}), location_after_sample, comm_pt)
                                min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                            if min_nav_to_comm < self.large_cost: # Can reach communication point
                                cost_communicate = min_nav_to_comm + 1 # nav + communicate
                                total_cost_this_rover = cost_sample + cost_communicate
                                min_total_cost_for_goal = min(min_total_cost_for_goal, total_cost_this_rover)

            elif predicate == 'communicated_rock_data' and len(parts) > 1:
                w = parts[1]
                # Check if sample is already collected by any rover
                rover_with_sample = None
                for r in self.all_rovers:
                    if f'(have_rock_analysis {r} {w})' in state:
                        rover_with_sample = r
                        break

                if rover_with_sample:
                    # Sample already collected by this specific rover
                    r = rover_with_sample
                    current_loc = self._get_rover_location(state, r)
                    if current_loc is None: continue

                    # Cost to communicate sample
                    min_nav_to_comm = self.large_cost
                    for comm_pt in self.communication_points:
                        nav_cost = self.shortest_path(self.rover_traversal_graphs.get(r, {}), current_loc, comm_pt)
                        min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                    if min_nav_to_comm < self.large_cost:
                        cost_communicate = min_nav_to_comm + 1
                        min_total_cost_for_goal = min(min_total_cost_for_goal, cost_communicate)
                else:
                    # Sample not collected by any rover, need to collect and communicate
                    for r in self.all_rovers:
                        if 'rock' in self.rover_capabilities.get(r, set()):
                            current_loc = self._get_rover_location(state, r)
                            if current_loc is None: continue

                            # Cost to collect sample
                            nav_to_sample = self.shortest_path(self.rover_traversal_graphs.get(r, {}), current_loc, w)
                            if nav_to_sample >= self.large_cost: continue # Cannot reach sample location

                            store = self.store_map.get(r)
                            drop_cost = 1 if store and (f'(full {store})' in state) else 0
                            cost_sample = nav_to_sample + drop_cost + 1 # nav + drop + sample
                            location_after_sample = w # Rover is at sample location after sampling

                            # Cost to communicate sample
                            min_nav_to_comm = self.large_cost
                            for comm_pt in self.communication_points:
                                nav_cost = self.shortest_path(self.rover_traversal_graphs.get(r, {}), location_after_sample, comm_pt)
                                min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                            if min_nav_to_comm < self.large_cost: # Can reach communication point
                                cost_communicate = min_nav_to_comm + 1 # nav + communicate
                                total_cost_this_rover = cost_sample + cost_communicate
                                min_total_cost_for_goal = min(min_total_cost_for_goal, total_cost_this_rover)

            elif predicate == 'communicated_image_data' and len(parts) > 2:
                o = parts[1]
                m = parts[2]
                # Check if image is already collected by any rover
                rover_with_image = None
                for r in self.all_rovers:
                    if f'(have_image {r} {o} {m})' in state:
                        rover_with_image = r
                        break

                if rover_with_image:
                    # Image already collected by this specific rover
                    r = rover_with_image
                    current_loc = self._get_rover_location(state, r)
                    if current_loc is None: continue

                    # Cost to communicate image
                    min_nav_to_comm = self.large_cost
                    for comm_pt in self.communication_points:
                        nav_cost = self.shortest_path(self.rover_traversal_graphs.get(r, {}), current_loc, comm_pt)
                        min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                    if min_nav_to_comm < self.large_cost:
                        cost_communicate = min_nav_to_comm + 1
                        min_total_cost_for_goal = min(min_total_cost_for_goal, cost_communicate)
                else:
                    # Image not collected by any rover, need to collect and communicate
                    for r in self.all_rovers:
                        if 'imaging' in self.rover_capabilities.get(r, set()):
                            for i in self.rover_cameras.get(r, set()):
                                if m in self.camera_modes.get(i, set()):
                                    current_loc = self._get_rover_location(state, r)
                                    if current_loc is None: continue

                                    # Cost to take image
                                    min_image_sequence_cost = self.large_cost

                                    image_pts = self.visible_from_objective.get(o, set())
                                    if not image_pts: continue # Cannot view objective

                                    # Find best waypoint p visible from o
                                    min_nav_to_image_pt = self.large_cost
                                    best_p = None
                                    for p in image_pts:
                                        nav_cost = self.shortest_path(self.rover_traversal_graphs.get(r, {}), current_loc, p)
                                        if nav_cost < min_nav_to_image_pt:
                                             min_nav_to_image_pt = nav_cost
                                             best_p = p

                                    if best_p is None or min_nav_to_image_pt >= self.large_cost: continue # Cannot reach any image point

                                    cost_calibration_sequence = 0
                                    cal_target = self.calibration_targets.get(i)
                                    best_w = None

                                    if cal_target:
                                        cal_pts = self.visible_from_objective.get(cal_target, set()) # Use visible_from_objective for targets
                                        if not cal_pts: continue # Cannot view calibration target

                                        # Find best waypoint w visible from target (navigating from best_p)
                                        min_nav_p_to_cal_pt = self.large_cost
                                        for w in cal_pts:
                                            nav_cost = self.shortest_path(self.rover_traversal_graphs.get(r, {}), best_p, w)
                                            if nav_cost < min_nav_p_to_cal_pt:
                                                 min_nav_p_to_cal_pt = nav_cost
                                                 best_w = w

                                        if best_w is None or min_nav_p_to_cal_pt >= self.large_cost: continue # Cannot reach any calibration point from image_pt

                                        # Calibration sequence: nav(p, w) + calibrate + nav(w, p)
                                        nav_w_to_p = self.shortest_path(self.rover_traversal_graphs.get(r, {}), best_w, best_p)
                                        if nav_w_to_p >= self.large_cost: continue # Cannot navigate back

                                        cost_calibration_sequence = min_nav_p_to_cal_pt + 1 + nav_w_to_p

                                    # Total cost to get image: nav(current, p) + calibration sequence cost (if needed) + take_image action
                                    cost_image_this_rover_camera = min_nav_to_image_pt + cost_calibration_sequence + 1 # nav(curr, p) + nav(p, w) + cal + nav(w, p) + take_image

                                    # Location after imaging sequence is best_p
                                    location_after_image = best_p

                                    # Cost to communicate image
                                    min_nav_to_comm = self.large_cost
                                    for comm_pt in self.communication_points:
                                        nav_cost = self.shortest_path(self.rover_traversal_graphs.get(r, {}), location_after_image, comm_pt)
                                        min_nav_to_comm = min(min_nav_to_comm, nav_cost)

                                    if min_nav_to_comm < self.large_cost: # Can reach communication point
                                        cost_communicate = min_nav_to_comm + 1 # nav + communicate
                                        total_cost_this_rover_camera = cost_image_this_rover_camera + cost_communicate
                                        min_total_cost_for_goal = min(min_total_cost_for_goal, total_cost_this_rover_camera)


            # Add the minimum cost found for this goal to the total heuristic
            h += min_total_cost_for_goal

        return h
