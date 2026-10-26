import collections

class roversHeuristic:
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach a goal state by summing the
    estimated costs for each unachieved goal fact. For each unachieved goal,
    it finds the minimum cost required by any capable rover (and camera, for
    image goals) to achieve that specific goal fact in isolation, assuming
    other goals do not interfere. The cost calculation for a single goal
    involves estimating navigation costs (using precomputed shortest paths)
    and action costs (sampling, dropping, calibrating, taking image,
    communicating).

    Assumptions:
    - The domain is unit-cost.
    - Navigation is possible between waypoints if `(can_traverse r u v)`
      is true for the rover `r`. The `(visible u v)` precondition for
      `navigate` is assumed to be consistent with `can_traverse` or
      handled by the graph construction if necessary (using `can_traverse`
      is simpler and common in benchmarks).
    - The shortest path distance is a reasonable estimate for navigation cost.
    - The cost of achieving independent goals can be summed (additive heuristic).
    - If a sample (`at_soil_sample` or `at_rock_sample`) is no longer at its
      initial location and no rover currently holds that sample data, it is
      considered lost and the corresponding communication goal is unreachable
      via sampling by any rover that doesn't already have the sample.

    Heuristic Initialization:
    The heuristic constructor precomputes static information from the task,
    including:
    - Mapping rovers to their equipment (soil, rock, imaging).
    - Mapping stores to their owning rovers.
    - Mapping rovers to their cameras and camera capabilities (supports,
      calibration target).
    - Mapping objectives to waypoints from which they are visible.
    - Identifying the lander location and waypoints visible from the lander.
    - Building a navigation graph for each rover based on `can_traverse` facts.
    - Computing all-pairs shortest paths for each rover on its navigation graph
      using BFS.
    - Storing initial locations of soil and rock samples.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Initialize the total heuristic value `h` to 0.
    3. Extract relevant dynamic facts from the current state (rover positions,
       store status, `have_` facts, `calibrated` status, `at_sample` status,
       `communicated_` status).
    4. Iterate through each goal fact specified in the task.
    5. If a goal fact is already true in the current state, skip it.
    6. For an unachieved goal fact `g`:
       a. Initialize the minimum cost for this goal `cost_g` to infinity.
       b. If `g` is `(communicated_soil_data ?w)`:
          - Iterate through all rovers `r`.
          - If `r` is equipped for soil analysis:
            - Calculate the cost for rover `r` to achieve this goal:
              - Start with cost 0 at `current_pos[r]`.
              - If `(have_soil_analysis r w)` is false:
                - If `(at_soil_sample w)` is true in the current state:
                  - Add navigation cost `dist(r, current_pos[r], w)`.
                  - Add drop cost (1) if rover's store is full.
                  - Add sample cost (1).
                  - The rover is now at waypoint `w`.
                - Else if `w` was an initial soil sample location:
                  - This sample is gone and this rover doesn't have it. This path is impossible for this rover. Set cost to infinity.
                - Else:
                  - Sample never existed here. This path is impossible. Set cost to infinity.
              - If the cost so far is not infinity:
                - Add navigation cost from the rover's current location (either `current_pos[r]` if sample was held, or `w` if sample was just taken) to the closest lander-visible waypoint.
                - Add communication cost (1).
              - Update `cost_g` with the minimum finite cost found for rover `r`.
       c. If `g` is `(communicated_rock_data ?w)`: (Similar logic as soil data)
          - Iterate through all rovers `r`.
          - If `r` is equipped for rock analysis:
            - Calculate the cost for rover `r` to achieve this goal (similar steps as soil, using rock-specific facts and actions).
            - Update `cost_g` with the minimum finite cost found for rover `r`.
       d. If `g` is `(communicated_image_data ?o ?m)`:
          - Iterate through all rovers `r` equipped for imaging.
          - Iterate through all cameras `i` on board `r` that support mode `m`.
          - If camera `i` has a calibration target `t` and there are waypoints visible from `t` (`cal_waypoints`) and waypoints visible from objective `o` (`img_waypoints`):
            - Calculate the cost for rover `r` with camera `i` to achieve this goal:
              - Start with cost 0 at `current_pos[r]`.
              - If `(have_image r o m)` is false:
                - If `(calibrated i r)` is true:
                  - Add navigation cost from `current_pos[r]` to the closest `img_waypoints`.
                  - Add take_image cost (1).
                  - The rover is now at the chosen `img_waypoints`.
                - Else:
                  - Find the minimum cost path that visits a `cal_waypoints` (for calibration) and then an `img_waypoints` (for taking the image). This involves navigation `current_pos[r]` -> `w_cal` -> `p_img`.
                  - Add navigation cost `dist(r, current_pos[r], w_cal) + dist(r, w_cal, p_img)`.
                  - Add calibrate cost (1).
                  - Add take_image cost (1).
                  - The rover is now at the chosen `p_img`.
              - If the cost so far is not infinity:
                - Add navigation cost from the rover's current location (either `current_pos[r]` if image was held, or the chosen `p_img` if image was just taken) to the closest lander-visible waypoint.
                - Add communication cost (1).
              - Update `cost_g` with the minimum finite cost found for rover `r` and camera `i`.
       e. If `cost_g` is still infinity after checking all options (rovers/cameras), the goal is unreachable. Return infinity for the total heuristic.
       f. Add `cost_g` to the total heuristic `h`.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        self.task = task
        self.infinity = float('inf')

        # --- Parse Objects ---
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        # Helper to extract objects from facts
        def extract_objects(fact_string):
            parts = fact_string.strip('()').split()
            # Skip predicate (parts[0])
            return [p for p in parts[1:]]

        # Collect all potential object names from initial state and static facts
        # A more robust parser would get types from the PDDL definition,
        # but we can infer some from predicates.
        all_facts = list(task.initial_state) + list(task.static)
        typed_objects = {}
        for fact in all_facts:
            parts = fact.strip('()').split()
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) == 3: # (at ?x - rover ?y - waypoint)
                 typed_objects[parts[1]] = 'rover'
                 typed_objects[parts[2]] = 'waypoint'
            elif pred == 'at_lander' and len(parts) == 3: # (at_lander ?x - lander ?y - waypoint)
                 typed_objects[parts[1]] = 'lander'
                 typed_objects[parts[2]] = 'waypoint'
            elif pred == 'can_traverse' and len(parts) == 4: # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                 typed_objects[parts[1]] = 'rover'
                 typed_objects[parts[2]] = 'waypoint'
                 typed_objects[parts[3]] = 'waypoint'
            elif pred in ('equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging') and len(parts) == 2: # (?r - rover)
                 typed_objects[parts[1]] = 'rover'
            elif pred in ('have_rock_analysis', 'have_soil_analysis') and len(parts) == 3: # (?r - rover ?w - waypoint)
                 typed_objects[parts[1]] = 'rover'
                 typed_objects[parts[2]] = 'waypoint'
            elif pred == 'calibrated' and len(parts) == 3: # (?c - camera ?r - rover)
                 typed_objects[parts[1]] = 'camera'
                 typed_objects[parts[2]] = 'rover'
            elif pred == 'supports' and len(parts) == 3: # (?c - camera ?m - mode)
                 typed_objects[parts[1]] = 'camera'
                 typed_objects[parts[2]] = 'mode'
            elif pred == 'visible' and len(parts) == 3: # (?w - waypoint ?p - waypoint)
                 typed_objects[parts[1]] = 'waypoint'
                 typed_objects[parts[2]] = 'waypoint'
            elif pred == 'have_image' and len(parts) == 4: # (?r - rover ?o - objective ?m - mode)
                 typed_objects[parts[1]] = 'rover'
                 typed_objects[parts[2]] = 'objective'
                 typed_objects[parts[3]] = 'mode'
            elif pred in ('communicated_soil_data', 'communicated_rock_data') and len(parts) == 2: # (?w - waypoint)
                 typed_objects[parts[1]] = 'waypoint'
            elif pred == 'communicated_image_data' and len(parts) == 3: # (?o - objective ?m - mode)
                 typed_objects[parts[1]] = 'objective'
                 typed_objects[parts[2]] = 'mode'
            elif pred in ('at_soil_sample', 'at_rock_sample') and len(parts) == 2: # (?w - waypoint)
                 typed_objects[parts[1]] = 'waypoint'
            elif pred == 'visible_from' and len(parts) == 3: # (?o - objective ?w - waypoint)
                 typed_objects[parts[1]] = 'objective'
                 typed_objects[parts[2]] = 'waypoint'
            elif pred == 'store_of' and len(parts) == 3: # (?s - store ?r - rover)
                 typed_objects[parts[1]] = 'store'
                 typed_objects[parts[2]] = 'rover'
            elif pred == 'calibration_target' and len(parts) == 3: # (?i - camera ?t - objective)
                 typed_objects[parts[1]] = 'camera'
                 typed_objects[parts[2]] = 'objective'
            elif pred == 'on_board' and len(parts) == 3: # (?i - camera ?r - rover)
                 typed_objects[parts[1]] = 'camera'
                 typed_objects[parts[2]] = 'rover'
            # Handle 'empty' and 'full' predicates to identify stores
            elif pred in ('empty', 'full') and len(parts) == 2: # (?s - store)
                 typed_objects[parts[1]] = 'store'


        for obj, obj_type in typed_objects.items():
            if obj_type == 'rover': self.rovers.add(obj)
            elif obj_type == 'waypoint': self.waypoints.add(obj)
            elif obj_type == 'store': self.stores.add(obj)
            elif obj_type == 'camera': self.cameras.add(obj)
            elif obj_type == 'mode': self.modes.add(obj)
            elif obj_type == 'lander': self.landers.add(obj)
            elif obj_type == 'objective': self.objectives.add(obj)

        # --- Parse Static Facts ---
        self.rover_equipment = {r: {'soil': False, 'rock': False, 'imaging': False} for r in self.rovers}
        self.store_owner = {} # store -> rover
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {r: set() for r in self.rovers} # rover -> set(camera)
        self.camera_info = {c: {'on_board': None, 'supports': set(), 'calibration_target': None} for c in self.cameras} # camera -> info
        self.objective_visible_from = {o: set() for o in self.objectives} # objective -> set(waypoint)
        self.lander_waypoint = None
        self.lander_visible_waypoints = set()
        self.rover_nav_graph = {r: {w: set() for w in self.waypoints} for r in self.rovers} # rover -> waypoint -> set(waypoint)

        for fact in task.static:
            parts = fact.strip('()').split()
            if not parts: continue
            pred = parts[0]
            if pred == 'equipped_for_soil_analysis' and len(parts) == 2 and parts[1] in self.rovers:
                self.rover_equipment[parts[1]]['soil'] = True
            elif pred == 'equipped_for_rock_analysis' and len(parts) == 2 and parts[1] in self.rovers:
                self.rover_equipment[parts[1]]['rock'] = True
            elif pred == 'equipped_for_imaging' and len(parts) == 2 and parts[1] in self.rovers:
                self.rover_equipment[parts[1]]['imaging'] = True
            elif pred == 'store_of' and len(parts) == 3 and parts[1] in self.stores and parts[2] in self.rovers:
                store, rover = parts[1], parts[2]
                self.store_owner[store] = rover
                self.rover_stores[rover] = store
            elif pred == 'on_board' and len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.rovers:
                camera, rover = parts[1], parts[2]
                self.rover_cameras[rover].add(camera)
                self.camera_info[camera]['on_board'] = rover
            elif pred == 'supports' and len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.modes:
                camera, mode = parts[1], parts[2]
                self.camera_info[camera]['supports'].add(mode)
            elif pred == 'calibration_target' and len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.objectives:
                camera, objective = parts[1], parts[2]
                self.camera_info[camera]['calibration_target'] = objective
            elif pred == 'visible_from' and len(parts) == 3 and parts[1] in self.objectives and parts[2] in self.waypoints:
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from[objective].add(waypoint)
            elif pred == 'at_lander' and len(parts) == 3 and parts[1] in self.landers and parts[2] in self.waypoints:
                # Assuming only one lander
                self.lander_waypoint = parts[2]
            elif pred == 'can_traverse' and len(parts) == 4 and parts[1] in self.rovers and parts[2] in self.waypoints and parts[3] in self.waypoints:
                 rover, w1, w2 = parts[1], parts[2], parts[3]
                 self.rover_nav_graph[rover][w1].add(w2)

        # Determine lander visible waypoints
        # Need a general visibility graph first
        visibility_graph = {w: set() for w in self.waypoints}
        for fact in task.static:
             parts = fact.strip('()').split()
             if not parts: continue
             if parts[0] == 'visible' and len(parts) == 3 and parts[1] in self.waypoints and parts[2] in self.waypoints:
                 w1, w2 = parts[1], parts[2]
                 visibility_graph[w1].add(w2)

        if self.lander_waypoint:
             # Waypoints x from which lander_waypoint is visible
             self.lander_visible_waypoints = {x for x in self.waypoints if self.lander_waypoint in visibility_graph.get(x, set())}


        # --- Compute All-Pairs Shortest Paths for each rover ---
        self.dist = {r: {} for r in self.rovers}
        for rover in self.rovers:
            graph = self.rover_nav_graph.get(rover, {})
            for start_wp in self.waypoints:
                self.dist[rover][start_wp] = self._bfs(graph, start_wp)

        # --- Parse Initial State Facts ---
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        for fact in task.initial_state:
            parts = fact.strip('()').split()
            if not parts: continue
            if parts[0] == 'at_soil_sample' and len(parts) == 2 and parts[1] in self.waypoints:
                self.initial_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample' and len(parts) == 2 and parts[1] in self.waypoints:
                self.initial_rock_samples.add(parts[1])

    def _bfs(self, graph, start_node):
        """Computes shortest path distances from start_node to all reachable nodes."""
        dist = {node: self.infinity for node in graph}
        if start_node in dist: # Ensure start_node is a valid key in the graph's nodes
            dist[start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                u = queue.popleft()
                if u in graph: # Check if u has outgoing edges defined
                    for v in graph[u]:
                        if dist[v] is self.infinity:
                            dist[v] = dist[u] + 1
                            queue.append(v)
        return dist


    def __call__(self, state, task):
        """
        Computes the heuristic value for the given state.

        @param state The current state (frozenset of facts).
        @param task The planning task object.
        @return The estimated number of actions to reach a goal state.
        """
        if task.goal_reached(state):
            return 0

        # --- Extract Dynamic Facts from State ---
        current_pos = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        calibrated = set() # (camera, rover)
        have_image = set() # (rover, objective, mode)
        communicated_soil = set() # waypoint
        communicated_rock = set() # waypoint
        communicated_image = set() # (objective, mode)
        at_soil_sample_state = set() # waypoint
        at_rock_sample_state = set() # waypoint

        for fact in state:
            parts = fact.strip('()').split()
            if not parts: continue # Skip empty strings from parsing errors
            pred = parts[0]
            if pred == 'at' and len(parts) == 3 and parts[1] in self.rovers:
                current_pos[parts[1]] = parts[2]
            elif pred == 'empty' and len(parts) == 2 and parts[1] in self.stores:
                store_status[parts[1]] = 'empty'
            elif pred == 'full' and len(parts) == 2 and parts[1] in self.stores:
                store_status[parts[1]] = 'full'
            elif pred == 'have_soil_analysis' and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                have_soil.add((parts[1], parts[2]))
            elif pred == 'have_rock_analysis' and len(parts) == 3 and parts[1] in self.rovers and parts[2] in self.waypoints:
                have_rock.add((parts[1], parts[2]))
            elif pred == 'calibrated' and len(parts) == 3 and parts[1] in self.cameras and parts[2] in self.rovers:
                calibrated.add((parts[1], parts[2]))
            elif pred == 'have_image' and len(parts) == 4 and parts[1] in self.rovers and parts[2] in self.objectives and parts[3] in self.modes:
                have_image.add((parts[1], parts[2], parts[3]))
            elif pred == 'communicated_soil_data' and len(parts) == 2 and parts[1] in self.waypoints:
                communicated_soil.add(parts[1])
            elif pred == 'communicated_rock_data' and len(parts) == 2 and parts[1] in self.waypoints:
                communicated_rock.add(parts[1])
            elif pred == 'communicated_image_data' and len(parts) == 3 and parts[1] in self.objectives and parts[2] in self.modes:
                communicated_image.add((parts[1], parts[2]))
            elif pred == 'at_soil_sample' and len(parts) == 2 and parts[1] in self.waypoints:
                at_soil_sample_state.add(parts[1])
            elif pred == 'at_rock_sample' and len(parts) == 2 and parts[1] in self.waypoints:
                at_rock_sample_state.add(parts[1])

        total_heuristic = 0

        # --- Calculate Cost for Each Unachieved Goal ---
        for goal in task.goals:
            parts = goal.strip('()').split()
            if not parts: continue
            pred = parts[0]

            cost_g = self.infinity

            if pred == 'communicated_soil_data' and len(parts) == 2:
                waypoint = parts[1]
                if waypoint in communicated_soil:
                    cost_g = 0
                else:
                    # Find min cost over equipped rovers
                    min_rover_cost = self.infinity
                    for rover in self.rovers:
                        if self.rover_equipment.get(rover, {}).get('soil', False):
                            curr = current_pos.get(rover)
                            if curr is None: continue # Rover location unknown

                            store = self.rover_stores.get(rover)
                            store_full = (store is not None and store_status.get(store) == 'full')

                            cost_to_get_sample = self.infinity
                            pos_after_sample = None

                            if (rover, waypoint) in have_soil:
                                cost_to_get_sample = 0
                                pos_after_sample = curr
                            elif waypoint in at_soil_sample_state: # Sample available now
                                nav_cost = self.dist.get(rover, {}).get(curr, {}).get(waypoint, self.infinity)
                                if nav_cost is not self.infinity:
                                    cost_to_get_sample = nav_cost + (1 if store_full else 0) + 1 # nav + drop (if needed) + sample
                                    pos_after_sample = waypoint
                            elif waypoint in self.initial_soil_samples: # Sample was there initially but is gone now
                                # Assume another rover sampled it. This rover cannot sample it.
                                cost_to_get_sample = self.infinity # This rover cannot get the sample part
                            # Else: Sample never existed here. cost_to_get_sample remains infinity.

                            total_cost_r = self.infinity
                            if cost_to_get_sample is not self.infinity:
                                # Need to communicate from pos_after_sample
                                min_nav_lander = self.infinity
                                if pos_after_sample is not None:
                                    for lander_wp in self.lander_visible_waypoints:
                                        d = self.dist.get(rover, {}).get(pos_after_sample, {}).get(lander_wp, self.infinity)
                                        if d is not self.infinity: min_nav_lander = min(min_nav_lander, d)

                                if min_nav_lander is not self.infinity:
                                    total_cost_r = cost_to_get_sample + min_nav_lander + 1 # communicate
                                else:
                                    total_cost_r = self.infinity # Cannot reach lander
                            # Else: total_cost_r remains infinity

                            min_rover_cost = min(min_rover_cost, total_cost_r)

                    cost_g = min_rover_cost

            elif pred == 'communicated_rock_data' and len(parts) == 2:
                waypoint = parts[1]
                if waypoint in communicated_rock:
                    cost_g = 0
                else:
                    # Find min cost over equipped rovers
                    min_rover_cost = self.infinity
                    for rover in self.rovers:
                        if self.rover_equipment.get(rover, {}).get('rock', False):
                            curr = current_pos.get(rover)
                            if curr is None: continue # Rover location unknown

                            store = self.rover_stores.get(rover)
                            store_full = (store is not None and store_status.get(store) == 'full')

                            cost_to_get_sample = self.infinity
                            pos_after_sample = None

                            if (rover, waypoint) in have_rock:
                                cost_to_get_sample = 0
                                pos_after_sample = curr
                            elif waypoint in at_rock_sample_state: # Sample available now
                                nav_cost = self.dist.get(rover, {}).get(curr, {}).get(waypoint, self.infinity)
                                if nav_cost is not self.infinity:
                                    cost_to_get_sample = nav_cost + (1 if store_full else 0) + 1 # nav + drop (if needed) + sample
                                    pos_after_sample = waypoint
                            elif waypoint in self.initial_rock_samples: # Sample was there initially but is gone now
                                # Assume another rover sampled it. This rover cannot sample it.
                                cost_to_get_sample = self.infinity # This rover cannot get the sample part
                            # Else: Sample never existed here. cost_to_get_sample remains infinity.

                            total_cost_r = self.infinity
                            if cost_to_get_sample is not self.infinity:
                                # Need to communicate from pos_after_sample
                                min_nav_lander = self.infinity
                                if pos_after_sample is not None:
                                    for lander_wp in self.lander_visible_waypoints:
                                        d = self.dist.get(rover, {}).get(pos_after_sample, {}).get(lander_wp, self.infinity)
                                        if d is not self.infinity: min_nav_lander = min(min_nav_lander, d)

                                if min_nav_lander is not self.infinity:
                                    total_cost_r = cost_to_get_sample + min_nav_lander + 1 # communicate
                                else:
                                    total_cost_r = self.infinity # Cannot reach lander
                            # Else: total_cost_r remains infinity

                            min_rover_cost = min(min_rover_cost, total_cost_r)

                    cost_g = min_rover_cost

            elif pred == 'communicated_image_data' and len(parts) == 3:
                objective, mode = parts[1], parts[2]
                if (objective, mode) in communicated_image:
                    cost_g = 0
                else:
                    # Find min cost over equipped rovers and cameras
                    min_rover_cam_cost = self.infinity
                    for rover in self.rovers:
                        if self.rover_equipment.get(rover, {}).get('imaging', False):
                            curr = current_pos.get(rover)
                            if curr is None: continue # Rover location unknown

                            for camera in self.rover_cameras.get(rover, set()):
                                cam_info = self.camera_info.get(camera)
                                if cam_info and mode in cam_info.get('supports', set()):
                                    cal_target = cam_info.get('calibration_target')
                                    cal_waypoints = self.objective_visible_from.get(cal_target, set()) if cal_target else set()
                                    img_waypoints = self.objective_visible_from.get(objective, set())

                                    if not cal_waypoints or not img_waypoints:
                                        continue # Cannot achieve with this camera/rover

                                    cost_to_get_image = self.infinity
                                    pos_after_image = None

                                    if (rover, objective, mode) in have_image:
                                        cost_to_get_image = 0
                                        pos_after_image = curr
                                    else: # Need to take image
                                        min_cost_img_part = self.infinity
                                        best_p_img = None # Track best waypoint to end up at

                                        if (camera, rover) in calibrated:
                                            # Need to reach p_img, take image.
                                            min_nav_img = self.infinity
                                            for p_img in img_waypoints:
                                                d = self.dist.get(rover, {}).get(curr, {}).get(p_img, self.infinity)
                                                if d is not self.infinity:
                                                    if d < min_nav_img:
                                                        min_nav_img = d
                                                        best_p_img = p_img
                                            if min_nav_img is not self.infinity: min_cost_img_part = min_nav_img + 1
                                        else:
                                            # Need to reach w_cal, calibrate, reach p_img, take image.
                                            min_nav_cal_img = self.infinity
                                            for w_cal in cal_waypoints:
                                                for p_img in img_waypoints:
                                                    d_curr_wcal = self.dist.get(rover, {}).get(curr, {}).get(w_cal, self.infinity)
                                                    d_wcal_pimg = self.dist.get(rover, {}).get(w_cal, {}).get(p_img, self.infinity)
                                                    if d_curr_wcal is not self.infinity and d_wcal_pimg is not self.infinity:
                                                        total_cost = d_curr_wcal + 1 + d_wcal_pimg + 1 # nav1 + calibrate + nav2 + take_image
                                                        if total_cost < min_nav_cal_img:
                                                            min_nav_cal_img = total_cost
                                                            best_p_img = p_img # Store the p_img that achieved this minimum
                                            if min_nav_cal_img is not self.infinity: min_cost_img_part = min_nav_cal_img

                                        cost_to_get_image = min_cost_img_part
                                        pos_after_image = best_p_img # Rover is at this p_img after taking image

                                    total_cost_ri = self.infinity
                                    if cost_to_get_image is not self.infinity and pos_after_image is not None:
                                        # Need to communicate from pos_after_image
                                        min_nav_lander = self.infinity
                                        for lander_wp in self.lander_visible_waypoints:
                                            d = self.dist.get(rover, {}).get(pos_after_image, {}).get(lander_wp, self.infinity)
                                            if d is not self.infinity: min_nav_lander = min(min_nav_lander, d)

                                        if min_nav_lander is not self.infinity:
                                            total_cost_ri = cost_to_get_image + min_nav_lander + 1 # communicate
                                        else:
                                            total_cost_ri = self.infinity # Cannot reach lander
                                    # Else: total_cost_ri remains infinity (cannot get image or no valid p_img found)

                                    min_rover_cam_cost = min(min_rover_cam_cost, total_cost_ri)

                    cost_g = min_rover_cam_cost

            # Add cost of this goal to total heuristic
            if cost_g is self.infinity:
                # If any unachieved goal is unreachable, the whole state is unreachable.
                return self.infinity
            total_heuristic += cost_g

        return total_heuristic
