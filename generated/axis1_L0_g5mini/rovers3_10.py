from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Remove surrounding parentheses and split into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the PDDL 'rover' domain. The heuristic estimates
    the (non-admissible) number of actions required to achieve all remaining goals
    (communications of images, soil analyses and rock analyses). It reasons about
    the following required steps for each goal:
      - acquiring the required data (sampling or taking images),
      - ensuring cameras are calibrated when needed,
      - navigating rovers between waypoints (calibration sites, imaging/sample sites,
        and a waypoint from which the lander is visible),
      - communicating the data to the lander, and
      - occasional store management (drop) if the rover's store is full.

    Assumptions
    -----------
    - Each navigate action moves a rover along one can_traverse edge and counts as 1.
    - Calibrate, take_image, sample_soil, sample_rock, communicate_* and drop each
      count as 1 action.
    - The heuristic treats goals independently and sums per-goal costs (it ignores
      some interdependencies and possible synergies between goals to remain simple
      and fast).
    - If a rover already has a required datum (e.g., have_image or have_soil_analysis),
      the heuristic counts only the actions necessary to communicate it.
    - Calibrations are modeled precisely: if a specific camera-on-rover pair is
      calibrated in the current state, calibrate is not required for the very next
      take_image using that camera. (take_image deletes the calibration, so multiple
      images with the same camera need recalibration each time).
    - Drop actions can be executed anywhere (matching the domain definition) and do
      not require movement. Therefore the heuristic charges 1 action when a drop is needed
      to make a store empty before a sample action.
    - If no rover/camera is suitable for a goal or required navigation is impossible,
      a large finite penalty is returned for that goal (so heuristic remains finite).

    Heuristic Initialization
    ------------------------
    The constructor extracts useful static information from task.static:
      - can_traverse graph per rover (adjacency lists),
      - visibility facts (visible waypoint -> waypoint),
      - visible_from mapping (objective -> waypoints from which that objective is visible),
      - camera calibration targets (camera -> objective),
      - which camera is on which rover (on_board),
      - camera mode support (camera -> set(modes)),
      - which store belongs to which rover (store_of and rover->store),
      - which rovers are equipped for imaging/soil/rock analysis,
      - lander waypoint(s) (from at_lander facts).
    It also stores the task goals for quick lookup.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
      1. If the state's facts already satisfy all goals -> return 0.
      2. Parse the current state's dynamic facts:
         - rover positions (at rover waypoint),
         - store empty/full flags,
         - calibrated camera-rover pairs,
         - which rover already has which images/soil/rock analyses.
      3. Precompute the set of waypoints from which a rover can communicate with the lander:
         - any waypoint X for which (visible X Y) holds and Y is a lander waypoint.
      4. For each goal (only those not yet satisfied in the state) compute a minimal
         local plan cost and sum them. Supported goal types considered:
         - (communicated_image_data OBJECTIVE MODE)
           * If some rover already has (have_image rover OBJECTIVE MODE): cost is
             minimal navigation from that rover's current waypoint to a lander-visible
             waypoint + 1 for the communicate action.
           * Otherwise, consider each rover-camera pair where the camera is on the rover,
             the rover is equipped_for_imaging, and the camera supports the required mode.
             For that pair:
               - If the camera-rover pair is already calibrated in the state, skip calibration step.
               - Else, find a calibration waypoint (a waypoint that is in visible_from for the
                 camera's calibration_target objective). Add navigation cost (rover -> calibration_wp)
                 + 1 calibrate.
               - Find an imaging waypoint (a waypoint in visible_from for the target objective).
                 Add navigation cost (from current/last waypoint to imaging_wp) + 1 take_image.
               - After imaging, add navigation cost to any lander-visible waypoint + 1 communicate.
             Choose the minimal cost across cameras and rovers.
         - (communicated_soil_data WAYPOINT) and (communicated_rock_data WAYPOINT)
           * If some rover already has have_soil_analysis/have_rock_analysis for that waypoint:
             cost is navigation from that rover's position to any lander-visible waypoint + 1 communicate.
           * Otherwise, consider rovers equipped for the corresponding analysis type, each with a store:
             - If the store is full in the current state, add cost 1 for drop (no movement).
             - Add navigation cost from rover position to the sample waypoint + 1 sample action.
             - Add navigation cost from sample waypoint to a lander-visible waypoint + 1 communicate action.
             Choose the minimal cost across rovers.
      5. Sum the per-goal minimal costs and return that sum.
    """

    def __init__(self, task):
        # Goals we need to reach (set of goal fact strings).
        self.goals = task.goals
        static_facts = task.static

        # Build can_traverse graph per rover (directed edges as given).
        self.can_traverse = {}  # rover -> dict waypoint -> set(neighbors)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "can_traverse":
                rover, a, b = parts[1], parts[2], parts[3]
                self.can_traverse.setdefault(rover, {}).setdefault(a, set()).add(b)
                # Ensure nodes exist
                self.can_traverse.setdefault(rover, {}).setdefault(b, set())

        # visible relations: wp1 -> set(wp2) meaning (visible wp1 wp2)
        self.visible = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "visible":
                a, b = parts[1], parts[2]
                self.visible.setdefault(a, set()).add(b)

        # visible_from: objective -> set(waypoints)
        self.visible_from = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "visible_from":
                obj, wp = parts[1], parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)

        # calibration_target: camera -> objective (camera may have single target in these domains)
        self.calibration_target = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "calibration_target":
                cam, obj = parts[1], parts[2]
                self.calibration_target[cam] = obj

        # on_board: camera -> rover
        self.on_board = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "on_board":
                cam, rover = parts[1], parts[2]
                self.on_board[cam] = rover

        # supports: camera -> set(modes)
        self.supports = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "supports":
                cam, mode = parts[1], parts[2]
                self.supports.setdefault(cam, set()).add(mode)

        # store_of: store -> rover  and rover -> store (assume one store per rover in domain instances)
        self.store_of = {}
        self.rover_store = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "store_of":
                store, rover = parts[1], parts[2]
                self.store_of[store] = rover
                self.rover_store[rover] = store

        # equipped capabilities for rovers
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "equipped_for_soil_analysis":
                self.equipped_for_soil.add(parts[1])
            elif parts[0] == "equipped_for_rock_analysis":
                self.equipped_for_rock.add(parts[1])
            elif parts[0] == "equipped_for_imaging":
                self.equipped_for_imaging.add(parts[1])

        # lander waypoints (there may be one)
        self.lander_waypoints = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "at_lander":
                # (at_lander lander waypoint)
                self.lander_waypoints.add(parts[2])

        # Precompute which waypoints can be used to communicate with the lander:
        # any waypoint X such that (visible X LW) for some lander waypoint LW
        self.lander_visible_waypoints = set()
        if self.lander_waypoints:
            for wp1, targets in self.visible.items():
                if any(t in self.lander_waypoints for t in targets):
                    self.lander_visible_waypoints.add(wp1)

    def _shortest_path_len(self, rover, start, targets):
        """
        BFS to compute shortest path length (in number of navigate actions) for
        rover from start waypoint to any waypoint in targets using can_traverse edges.
        Returns INF if no path exists or rover has no traversal info.
        """
        if start in targets:
            return 0
        if rover not in self.can_traverse:
            return INF
        adj = self.can_traverse[rover]
        # if start not even in adj keys, still try (maybe isolated)
        if start not in adj:
            return INF
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nb in adj.get(node, ()):
                if nb in visited:
                    continue
                if nb in targets:
                    return d + 1
                visited.add(nb)
                q.append((nb, d + 1))
        return INF

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse dynamic facts from state
        rover_position = {}  # rover -> waypoint
        calibrated = set()  # set of (camera, rover) pairs
        empty_stores = set()
        full_stores = set()
        have_image = set()  # triples (rover, objective, mode)
        have_soil = set()  # pairs (rover, waypoint)
        have_rock = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                # Could be (at rover waypoint) or other at facts (like at waypoint for lander is in static)
                if len(parts) == 3:
                    obj, wp = parts[1], parts[2]
                    # detect rover by presence in can_traverse or by naming convention; safer: rovers exist in can_traverse keys or equipped sets
                    # We'll assume anything that also appears in can_traverse mapping is a rover.
                    if obj in self.can_traverse or obj in self.rover_store or obj in self.equipped_for_imaging or obj in self.equipped_for_soil or obj in self.equipped_for_rock:
                        rover_position[obj] = wp
            elif pred == "calibrated":
                cam, rov = parts[1], parts[2]
                calibrated.add((cam, rov))
            elif pred == "empty":
                empty_stores.add(parts[1])
            elif pred == "full":
                full_stores.add(parts[1])
            elif pred == "have_image":
                rov, obj, mode = parts[1], parts[2], parts[3]
                have_image.add((rov, obj, mode))
            elif pred == "have_soil_analysis":
                rov, wp = parts[1], parts[2]
                have_soil.add((rov, wp))
            elif pred == "have_rock_analysis":
                rov, wp = parts[1], parts[2]
                have_rock.add((rov, wp))

        total_cost = 0

        # Helper: candidate communication waypoints are precomputed self.lander_visible_waypoints.
        comm_wps = self.lander_visible_waypoints
        if not comm_wps:
            # If no waypoint can communicate to lander (should be rare), treat communication impossible
            # but return finite large cost for each remaining goal.
            missing_goals = [g for g in self.goals if g not in state]
            return len(missing_goals) * 50

        # Process each goal (only those not yet satisfied in the state)
        for goal in self.goals:
            if goal in state:
                continue  # already satisfied

            parts = get_parts(goal)
            pred = parts[0]

            if pred == "communicated_image_data":
                objective, mode = parts[1], parts[2]

                # If some rover already has the image (have_image), choose best communicator
                best_cost = INF

                # Option A: some rover already has the image -> only need to navigate and communicate
                for (rov, obj, m) in list(have_image):
                    if obj == objective and m == mode:
                        # cost = nav from rov position to a comm_wp + 1 communicate
                        start = rover_position.get(rov)
                        if start is None:
                            continue
                        d = self._shortest_path_len(rov, start, comm_wps)
                        cost = d + 1 if d < INF else INF
                        if cost < best_cost:
                            best_cost = cost

                # Option B: take an image then communicate. Consider all rover-camera pairs that can capture mode.
                # For each camera on a rover that supports the mode and whose rover is equipped_for_imaging.
                for cam, rov in self.on_board.items():
                    if rov not in self.equipped_for_imaging:
                        continue
                    if mode not in self.supports.get(cam, set()):
                        continue
                    # rover must exist in traversal info to navigate; if not, still allow but with INF distances
                    start = rover_position.get(rov)
                    if start is None:
                        continue

                    # If camera already calibrated on this rover, no calibrate step needed.
                    cam_calibrated = (cam, rov) in calibrated

                    # Calibration waypoint candidates: waypoints that can see the calibration target objective
                    if cam in self.calibration_target:
                        calib_obj = self.calibration_target[cam]
                        calib_wps = self.visible_from.get(calib_obj, set())
                    else:
                        calib_wps = set()

                    # Imaging waypoints: from which objective is visible
                    img_wps = self.visible_from.get(objective, set())
                    if not img_wps:
                        # can't image this objective
                        continue

                    # Consider all choices:
                    # If already calibrated: start -> imaging_wp -> comm_wp
                    if cam_calibrated:
                        for img_wp in img_wps:
                            # nav from start to img_wp
                            d1 = self._shortest_path_len(rov, start, {img_wp})
                            if d1 == INF:
                                continue
                            # after take_image, need nav from img_wp to comm_wp
                            d2 = self._shortest_path_len(rov, img_wp, comm_wps)
                            if d2 == INF:
                                continue
                            # cost: move to img_wp + take_image + move to comm_wp + communicate
                            cost = d1 + 1 + d2 + 1
                            if cost < best_cost:
                                best_cost = cost
                    else:
                        # need to calibrate at some calib_wp first: start -> calib_wp + calibrate
                        if not calib_wps:
                            # cannot calibrate this camera -> skip
                            continue
                        for calib_wp in calib_wps:
                            d0 = self._shortest_path_len(rov, start, {calib_wp})
                            if d0 == INF:
                                continue
                            # after calibration, must go to imaging wp
                            for img_wp in img_wps:
                                d1 = self._shortest_path_len(rov, calib_wp, {img_wp})
                                if d1 == INF:
                                    continue
                                d2 = self._shortest_path_len(rov, img_wp, comm_wps)
                                if d2 == INF:
                                    continue
                                # cost: move to calib + calibrate + move to img + take_image + move to comm + communicate
                                cost = d0 + 1 + d1 + 1 + d2 + 1
                                if cost < best_cost:
                                    best_cost = cost

                # If no viable rover-camera found, penalize but finite.
                if best_cost >= INF:
                    best_cost = 50
                total_cost += best_cost

            elif pred == "communicated_soil_data":
                sample_wp = parts[1]

                # If communicated already in state we skipped; else compute minimal cost.
                best_cost = INF

                # Option A: some rover already has soil analysis at that waypoint
                for (rov, wp) in have_soil:
                    if wp == sample_wp:
                        start = rover_position.get(rov)
                        if start is None:
                            continue
                        d = self._shortest_path_len(rov, start, comm_wps)
                        cost = d + 1 if d < INF else INF  # +1 communicate
                        if cost < best_cost:
                            best_cost = cost

                # Option B: need to sample then communicate; consider rovers equipped for soil analysis
                for rov in self.equipped_for_soil:
                    start = rover_position.get(rov)
                    if start is None:
                        continue

                    # store management: the rover's store must be empty for sample. If not empty, add 1 drop.
                    store = self.rover_store.get(rov)
                    drop_cost = 0
                    if store:
                        if store in full_stores:
                            drop_cost = 1
                        # if empty or not mentioned, assume empty unless full flag present
                    # nav to sample_wp
                    d1 = self._shortest_path_len(rov, start, {sample_wp})
                    if d1 == INF:
                        continue
                    # sample action
                    # then nav to comm wp from sample_wp
                    d2 = self._shortest_path_len(rov, sample_wp, comm_wps)
                    if d2 == INF:
                        continue
                    cost = drop_cost + d1 + 1 + d2 + 1
                    if cost < best_cost:
                        best_cost = cost

                if best_cost >= INF:
                    best_cost = 50
                total_cost += best_cost

            elif pred == "communicated_rock_data":
                sample_wp = parts[1]

                best_cost = INF

                # Option A: some rover already has rock analysis at that waypoint
                for (rov, wp) in have_rock:
                    if wp == sample_wp:
                        start = rover_position.get(rov)
                        if start is None:
                            continue
                        d = self._shortest_path_len(rov, start, comm_wps)
                        cost = d + 1 if d < INF else INF
                        if cost < best_cost:
                            best_cost = cost

                # Option B: need to sample then communicate; consider rovers equipped for rock analysis
                for rov in self.equipped_for_rock:
                    start = rover_position.get(rov)
                    if start is None:
                        continue

                    # store management
                    store = self.rover_store.get(rov)
                    drop_cost = 0
                    if store:
                        if store in full_stores:
                            drop_cost = 1

                    d1 = self._shortest_path_len(rov, start, {sample_wp})
                    if d1 == INF:
                        continue
                    d2 = self._shortest_path_len(rov, sample_wp, comm_wps)
                    if d2 == INF:
                        continue
                    cost = drop_cost + d1 + 1 + d2 + 1
                    if cost < best_cost:
                        best_cost = cost

                if best_cost >= INF:
                    best_cost = 50
                total_cost += best_cost

            else:
                # Other goal types (not expected) -> small penalty if not satisfied.
                total_cost += 5

        return int(total_cost)