from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Rovers domain. The heuristic estimates the
    number of actions required to satisfy all remaining communication goals
    (communicated_image_data, communicated_soil_data, communicated_rock_data).
    It reasons about the key subgoals that must precede communication:
      - For image goals: taking an image (possibly calibrating a camera first) and
        then getting to a location that can communicate with a lander.
      - For soil/rock goals: sampling at a sample waypoint (ensuring an empty store)
        and then getting to a location that can communicate with a lander.

    Assumptions
    -----------
    - Navigation between waypoints for a specific rover follows the directed graph
      given by (can_traverse ?r ?from ?to) facts in the static information; each
      graph edge costs 1 (one navigate action).
    - Calibration and take_image each cost 1 action (unless the camera is already
      calibrated for that rover and camera).
    - Sampling (sample_soil/sample_rock), dropping a full store (drop) and
      communicating (communicate_*) each cost 1 action.
    - If a required capability (e.g. equipped_for_imaging, suitable camera, or
      an on-board camera that supports a required mode) is missing for a rover,
      that rover is considered unable to perform the respective image capture.
    - The heuristic is not required to be admissible; it aims to be informative
      and efficiently computable (using shortest-path BFS on the waypoint graphs).

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
      - The set of goal facts (task.goals).
      - Static mappings:
        - adjacency graph per rover from can_traverse facts.
        - visible edges between waypoints (visible).
        - visible_from: objectives -> set of waypoints where they can be imaged.
        - lander waypoints (at_lander).
        - comm_spots: waypoints that can see a lander (suitable for communication).
        - on_board cameras per rover.
        - supports mapping camera -> set(modes).
        - calibration_target mapping camera -> set(objectives).
        - which rovers are equipped for imaging/soil/rock.
        - store_of mapping rover -> store.
    These structures are used to quickly estimate travel distances and feasibility.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic does the following:

    1. Quick goal check:
       - If the state's facts already satisfy all goal facts, return 0.

    2. Parse dynamic facts from the state:
       - Current rover locations (at ...)
       - Which stores are empty/full
       - Which cameras are calibrated for which rover (calibrated ...)
       - Which analyses or images are already obtained (have_image, have_soil_analysis, have_rock_analysis)
       - Which communications are already performed

    3. For each goal fact still missing, estimate the minimal number of actions
       required to achieve it. We consider all rovers and take the best (minimum)
       cost among them for achieving that single goal. The estimate for a single
       goal is the sum of:
         - Necessary navigation steps (shortest path counts on a rover-specific graph)
         - Required action steps (calibrate, take_image, sample, drop, communicate)
       Details per goal type:
         - communicated_image_data (objective o, mode m):
             For each rover that has a usable on-board camera (camera i with
             calibration_target i o and supports i m and equipped_for_imaging):
               - Move to an imaging waypoint w (one from visible_from[o]) -> nav cost.
               - If camera i is not already calibrated for that rover, add 1 (calibrate).
               - Add 1 for take_image.
               - Move from w to a communication waypoint x (one that can see a lander) -> nav cost.
               - Add 1 for communicate_image_data.
             Also handle the case where have_image already holds (skip calibr/take).
             Choose minimum across all rovers, cameras and possible w/x pairs.
         - communicated_soil_data / communicated_rock_data (waypoint p):
             For each rover equipped for soil/rock analysis:
               - If rover already has have_*_analysis for p: only move to comm spot and communicate (nav + 1).
               - Else:
                 - If rover's store is full in the current state, add 1 (drop) to empty it.
                 - Move to the sample waypoint p (nav).
                 - Add 1 for sample_*.
                 - Move to comm spot (nav).
                 - Add 1 for communicate_*_data.
             Choose minimum over rovers.

    4. Sum the minimal per-goal estimates to form the heuristic value.

    Notes on robustness
    -------------------
    - If no rover can feasibly achieve a particular goal due to missing capabilities
      or unreachable waypoints, the heuristic inserts a large finite penalty for that
      single goal (we use PENALTY). This keeps the heuristic finite while strongly
      preferring states where goals are feasible.
    - The BFS-based shortest-path is per rover and efficient since waypoint
      graphs are typically small; if a start waypoint equals a target, distance is 0.
    """

    PENALTY = 1000  # large finite penalty for infeasible subgoals (keeps heuristic finite)

    def __init__(self, task):
        """Extract goals and static domain information needed by the heuristic."""
        self.goals = set(task.goals)
        static = task.static

        # adjacency per rover: rover -> { waypoint_from: set(waypoint_to, ...) }
        self.adjacency = {}
        # visible edges: waypoint -> set(waypoint_visible)
        self.visible = {}
        # visible_from: objective -> set(waypoints)
        self.visible_from = {}
        # lander waypoints
        self.lander_waypoints = set()
        # communication-capable waypoints (those that can see a lander)
        self.comm_spots = set()
        # on_board: rover -> set(cameras)
        self.on_board = {}
        # supports: camera -> set(modes)
        self.supports = {}
        # calibration_target: camera -> set(objectives)
        self.calibration_target = {}
        # stores: rover -> store
        self.store_of = {}
        # capabilities
        self.equipped_for_imaging = set()
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()

        # collect waypoints and other static info
        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "can_traverse":
                # (can_traverse ?r ?from ?to)
                _, r, frm, to = parts
                self.adjacency.setdefault(r, {}).setdefault(frm, set()).add(to)
            elif pred == "visible":
                # (visible ?a ?b)
                _, a, b = parts
                self.visible.setdefault(a, set()).add(b)
            elif pred == "visible_from":
                # (visible_from ?o ?w)
                _, o, w = parts
                self.visible_from.setdefault(o, set()).add(w)
            elif pred == "at_lander":
                # (at_lander ?l ?w)
                _, l, w = parts
                self.lander_waypoints.add(w)
            elif pred == "on_board":
                # (on_board ?camera ?rover)
                _, cam, r = parts
                self.on_board.setdefault(r, set()).add(cam)
            elif pred == "supports":
                # (supports ?camera ?mode)
                _, cam, mode = parts
                self.supports.setdefault(cam, set()).add(mode)
            elif pred == "calibration_target":
                # (calibration_target ?camera ?objective)
                _, cam, obj = parts
                self.calibration_target.setdefault(cam, set()).add(obj)
            elif pred == "store_of":
                # (store_of ?store ?rover)
                _, store, r = parts
                self.store_of[r] = store
            elif pred == "equipped_for_imaging":
                _, r = parts
                self.equipped_for_imaging.add(r)
            elif pred == "equipped_for_soil_analysis":
                _, r = parts
                self.equipped_for_soil.add(r)
            elif pred == "equipped_for_rock_analysis":
                _, r = parts
                self.equipped_for_rock.add(r)

        # compute comm_spots as those waypoints that can see any lander waypoint
        for wp, visibles in self.visible.items():
            if any(lander_wp in visibles for lander_wp in self.lander_waypoints):
                self.comm_spots.add(wp)

        # Ensure every rover has an adjacency entry (graph may be empty)
        # Also ensure adjacency dicts contain empty adjacency sets for nodes without outgoing edges.
        for r in list(self.adjacency.keys()):
            # already set for rovers present
            pass

    def shortest_distance(self, rover, start, targets):
        """
        BFS on the rover-specific adjacency graph to compute the shortest number
        of navigate steps from start to any waypoint in 'targets'. Returns None
        if unreachable or adjacency info missing.
        """
        if start is None:
            return None
        if targets is None or len(targets) == 0:
            return None
        if start in targets:
            return 0
        graph = self.adjacency.get(rover)
        if not graph:
            # If there is no adjacency info for this rover, we cannot navigate (treat as unreachable).
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            # consider outgoing edges (if none, continue)
            for neigh in graph.get(node, ()):
                if neigh in visited:
                    continue
                if neigh in targets:
                    return dist + 1
                visited.add(neigh)
                q.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        """Compute heuristic estimate (number of actions) from the current state."""
        state = node.state

        # 1) If goals are already satisfied, return 0.
        if self.goals <= state:
            return 0

        # Parse dynamic state facts
        # rover_locations: rover -> waypoint
        rover_locations = {}
        # calibrated cameras: set of (camera, rover) tuples
        calibrated = set()
        # store empty/full
        empty_stores = set()
        full_stores = set()
        # have_image facts: set of tuples (rover, objective, mode)
        have_images = set()
        # have soil/rock analysis: (rover, waypoint)
        have_soil = set()
        have_rock = set()
        # communicated facts already present
        communicated = set()

        # also track which waypoints have samples (at_soil_sample / at_rock_sample) from state
        at_soil_sample = set()
        at_rock_sample = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at":
                # could be at rover waypoint
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # 'at' is overloaded; we only treat rover locations here by checking rover name prefixes is not reliable,
                    # so better to check for 'at <rover> <waypoint>' by seeing if a rover has adjacency in static
                    # but we can simply capture all "at" facts; rover_locations will be only ones where rover exists in adjacency or store_of
                    # We conservatively add any 'at' and later only consider rovers that exist in adjacency or on_board info.
                    # However to be explicit: domain uses 'at ?x - rover ?y - waypoint' so 'at' facts with two args are rover locations.
                    rover_locations[obj] = loc
            elif pred == "calibrated":
                # (calibrated camera rover)
                _, cam, r = parts
                calibrated.add((cam, r))
            elif pred == "empty":
                _, store = parts
                empty_stores.add(store)
            elif pred == "full":
                _, store = parts
                full_stores.add(store)
            elif pred == "have_image":
                # (have_image rover objective mode)
                _, r, o, m = parts
                have_images.add((r, o, m))
            elif pred == "have_soil_analysis":
                _, r, w = parts
                have_soil.add((r, w))
            elif pred == "have_rock_analysis":
                _, r, w = parts
                have_rock.add((r, w))
            elif pred == "communicated_image_data":
                _, o, m = parts
                communicated.add(("image", o, m))
            elif pred == "communicated_soil_data":
                _, w = parts
                communicated.add(("soil", w))
            elif pred == "communicated_rock_data":
                _, w = parts
                communicated.add(("rock", w))
            elif pred == "at_soil_sample":
                _, w = parts
                at_soil_sample.add(w)
            elif pred == "at_rock_sample":
                _, w = parts
                at_rock_sample.add(w)

        total_estimate = 0

        # Helper to compute minimal navigation distance for a rover to any of given targets
        def nav_dist(r, start_wp, target_wps):
            if start_wp is None:
                return None
            return self.shortest_distance(r, start_wp, set(target_wps))

        # Iterate over each goal and compute minimal estimated cost to satisfy it
        for goal in self.goals:
            if goal in state:
                # already satisfied
                continue
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]

            # For each goal compute minimal cost across rovers
            best_cost_for_goal = None

            if pred == "communicated_image_data":
                # goal: (communicated_image_data ?o ?m)
                _, obj, mode = parts

                # If some state fact already indicates the communication, skip (should not be here).
                if ("image", obj, mode) in communicated:
                    continue

                # Evaluate every rover for feasibility and cost
                # Comm spots are precomputed self.comm_spots
                if not self.comm_spots:
                    # no comm spots known statically; infeasible
                    best_cost_for_goal = self.PENALTY
                else:
                    for r in set(list(self.adjacency.keys()) + list(self.on_board.keys()) + list(self.store_of.keys()) + list(rover_locations.keys())):
                        # rover must be equipped for imaging (static)
                        if r not in self.equipped_for_imaging:
                            continue
                        # does this rover currently already have the image?
                        if (r, obj, mode) in have_images:
                            # just need to navigate to comm spot and communicate
                            start = rover_locations.get(r)
                            d = nav_dist(r, start, self.comm_spots)
                            if d is None:
                                # cannot reach any comm spot
                                continue
                            cost = d + 1  # +1 for communicate
                            if best_cost_for_goal is None or cost < best_cost_for_goal:
                                best_cost_for_goal = cost
                            continue

                        # else need to find a camera on-board that supports mode and has calibration_target for objective
                        cams = self.on_board.get(r, set())
                        feasible_camera_found = False
                        for cam in cams:
                            if mode not in self.supports.get(cam, set()):
                                continue
                            if obj not in self.calibration_target.get(cam, set()):
                                continue
                            feasible_camera_found = True
                            # Consider all capture waypoints for this objective
                            capture_wps = self.visible_from.get(obj, set())
                            if not capture_wps:
                                continue
                            start = rover_locations.get(r)
                            # For each capture waypoint compute nav to capture, calibration (if needed), take_image, nav to comm, communicate
                            for w in capture_wps:
                                d1 = nav_dist(r, start, {w})
                                if d1 is None:
                                    continue
                                # calibrate if (cam, r) not already in calibrated
                                calibr_needed = 0 if (cam, r) in calibrated else 1
                                take_cost = 1
                                # after taking picture we are at w; need to go to comm spot
                                d2 = nav_dist(r, w, self.comm_spots)
                                if d2 is None:
                                    continue
                                cost = d1 + calibr_needed + take_cost + d2 + 1  # +1 communicate
                                if best_cost_for_goal is None or cost < best_cost_for_goal:
                                    best_cost_for_goal = cost
                        if not feasible_camera_found:
                            # this rover cannot take the required image; skip
                            continue

            elif pred == "communicated_soil_data":
                # goal: (communicated_soil_data ?w)
                _, wp = parts

                if ("soil", wp) in communicated:
                    continue

                # Evaluate rovers
                for r in set(list(self.adjacency.keys()) + list(self.store_of.keys()) + list(rover_locations.keys())):
                    if r not in self.equipped_for_soil:
                        continue
                    start = rover_locations.get(r)
                    # If rover already has the analysis
                    if (r, wp) in have_soil:
                        d = nav_dist(r, start, self.comm_spots)
                        if d is None:
                            continue
                        cost = d + 1  # communicate
                        if best_cost_for_goal is None or cost < best_cost_for_goal:
                            best_cost_for_goal = cost
                        continue
                    # Else rover must sample at wp. Check that wp currently has a soil sample
                    if wp not in at_soil_sample:
                        # If no soil sample at wp in state, cannot sample -> skip this rover (problem may be unsolvable)
                        continue
                    # Check store for rover
                    store = self.store_of.get(r)
                    if store is None:
                        continue
                    drop_needed = 1 if store in full_stores else 0
                    # Need to navigate to wp to sample
                    d1 = nav_dist(r, start, {wp})
                    if d1 is None:
                        continue
                    # after sampling, need to go to comm spot
                    d2 = nav_dist(r, wp, self.comm_spots)
                    if d2 is None:
                        continue
                    cost = drop_needed + d1 + 1 + d2 + 1  # drop(if needed) + nav to sample + sample + nav to comm + communicate
                    if best_cost_for_goal is None or cost < best_cost_for_goal:
                        best_cost_for_goal = cost

            elif pred == "communicated_rock_data":
                # goal: (communicated_rock_data ?w)
                _, wp = parts

                if ("rock", wp) in communicated:
                    continue

                # Evaluate rovers
                for r in set(list(self.adjacency.keys()) + list(self.store_of.keys()) + list(rover_locations.keys())):
                    if r not in self.equipped_for_rock:
                        continue
                    start = rover_locations.get(r)
                    # If rover already has the analysis
                    if (r, wp) in have_rock:
                        d = nav_dist(r, start, self.comm_spots)
                        if d is None:
                            continue
                        cost = d + 1  # communicate
                        if best_cost_for_goal is None or cost < best_cost_for_goal:
                            best_cost_for_goal = cost
                        continue
                    # Else rover must sample at wp. Check that wp currently has a rock sample
                    if wp not in at_rock_sample:
                        # If no rock sample at wp in state, cannot sample -> skip this rover
                        continue
                    # Check store for rover
                    store = self.store_of.get(r)
                    if store is None:
                        continue
                    drop_needed = 1 if store in full_stores else 0
                    # Need to navigate to wp to sample
                    d1 = nav_dist(r, start, {wp})
                    if d1 is None:
                        continue
                    # after sampling, need to go to comm spot
                    d2 = nav_dist(r, wp, self.comm_spots)
                    if d2 is None:
                        continue
                    cost = drop_needed + d1 + 1 + d2 + 1  # drop(if needed) + nav to sample + sample + nav to comm + communicate
                    if best_cost_for_goal is None or cost < best_cost_for_goal:
                        best_cost_for_goal = cost

            else:
                # Unknown goal predicate - give small penalty to be conservative
                best_cost_for_goal = 1 if best_cost_for_goal is None else best_cost_for_goal

            # If no rover could feasibly achieve the goal (best_cost_for_goal is None),
            # mark as infeasible with a large finite penalty.
            if best_cost_for_goal is None:
                total_estimate += self.PENALTY
            else:
                total_estimate += best_cost_for_goal

        return total_estimate