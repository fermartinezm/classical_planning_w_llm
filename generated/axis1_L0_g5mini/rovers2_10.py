from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the parts of a fact string '(pred a b ...)' as a list [pred, a, b, ...]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the PDDL rovers domain. The heuristic produces
    an estimate (number of actions) to reach the goals that request communicated
    data: communicated_image_data, communicated_soil_data, and communicated_rock_data.
    The estimate is computed by selecting, for each unmet goal, a (domain-feasible)
    rover (and camera/store when needed) and summing up a small plan-cost estimate:
    number of navigation steps (navigate actions) plus the required domain actions
    (calibrate, take_image, sample_soil/sample_rock, communicate, drop when necessary).
    The heuristic is intentionally optimistic about some resource re-use but accounts
    for critical prerequisites (equipment, store emptiness, camera supports, calibration).

    Assumptions
    -----------
    - Navigation cost: each 'navigate' operator counts as 1.
    - Other operators (sample_soil, sample_rock, calibrate, take_image, communicate, drop)
      each count as 1.
    - Cameras are mounted on rovers as indicated by static 'on_board' facts and support
      modes as given by static 'supports' facts.
    - Calibration targets and visible_from static facts determine possible waypoints
      where calibrate/take_image can be executed.
    - A rover can navigate along directed edges (can_traverse) but only if the pair
      (from,to) is also marked 'visible' (the domain's navigate precondition).
    - For sampling, the rover chosen must be equipped for the sample type and must
      have access to its store (store_of). If the store is not empty in the current
      state, one 'drop' action is pessimistically added to empty it (since a drop
      will make the store empty).
    - Communication requires the rover to be at some waypoint that is visible to a
      lander waypoint; any lander in the domain may be used.

    Heuristic Initialization
    ------------------------
    In the constructor we extract static/domain-wide facts into efficient structures:
    - self.goals: the set of goal fact strings.
    - can_traverse_by_rover: dict rover -> adjacency dict waypoint -> set(neighbor_waypoints)
      (edges only included if both can_traverse and visible (from,to) exist in static).
    - visible_pairs: set of (from, to) visibility pairs between waypoints.
    - visible_from: dict objective -> set(waypoints) that can observe that objective.
    - supports: dict camera -> set(modes) it supports.
    - on_board: dict camera -> rover (camera mounted on rover).
    - calibration_target: dict camera -> objective (the objective used to calibrate this camera).
    - store_of: dict store -> rover, and rover_to_store: rover -> store (if any).
    - equipped_for_imaging/soil/rock: sets of rovers with these capabilities.
    - lander_waypoints: set of waypoints where landers are located (from at_lander facts).
    These structures permit fast per-state computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goals already true in the state, return 0 (goal state).
    2. Parse the state to obtain:
       - rover_positions: dict rover -> waypoint (from '(at rover waypoint)').
       - which stores are empty or full.
       - which have_* facts already exist (have_image, have_soil_analysis, have_rock_analysis).
       - which cameras are currently calibrated for which rover (calibrated camera rover).
    3. Precompute for every rover a shortest-path distance function (BFS) on that rover's
       can_traverse graph from the rover's current waypoint to any target waypoint. Distances
       count number of navigate actions.
    4. For each unmet goal fact:
       - communicated_image_data objective mode:
         * If some rover already has have_image (o,m), cost is minimal navigation from that
           rover's position to any waypoint visible to any lander + 1 communicate action.
         * Else, consider each camera that supports the desired mode and is on some rover:
           - compute minimal cost:
             distance(rover -> calibration_waypoint) + 1 (calibrate, unless already calibrated
             on that camera/rover) +
             distance(calibration_waypoint -> image_waypoint) + 1 (take_image) +
             distance(image_waypoint -> a lander-visible waypoint) + 1 (communicate).
           - distances use the rover's can_traverse graph. If any needed distance is unreachable
             for that rover, that candidate is discarded.
         * Choose minimum candidate cost across rovers/cameras. If no candidate exists, use a
           large finite fallback cost to keep heuristic finite (this indicates unsolvable or
           unreachable from current assumptions).
       - communicated_soil_data / communicated_rock_data waypoint:
         * If already communicated in state, cost 0.
         * Else, if some rover already has have_soil_analysis/have_rock_analysis for that waypoint,
           cost = distance(rover -> any lander-visible waypoint) + 1 communicate.
         * Else, for each rover equipped for that analysis:
           - cost = distance(rover -> sample_waypoint) + 1 (sample action)
             + extra_cost_if_store_not_empty (0 if its store is empty, else 1 to 'drop' first)
             + distance(sample_waypoint -> any lander-visible waypoint) + 1 communicate.
           - Choose minimal across rovers.
    5. Sum the minimal per-goal estimates to produce the heuristic value. The heuristic is
       admissible in trivial cases but not required to be admissible; it is constructed to
       be informative and efficiently computable.
    """

    def __init__(self, task):
        """Extract static information and goal set from the task."""
        self.goals = task.goals
        static = task.static

        # Visibility pairs between waypoints: (from, to)
        self.visible_pairs = {(get_parts(f)[1], get_parts(f)[2]) for f in static if get_parts(f)[0] == "visible"}

        # Build per-rover traversal graphs: edges only valid if both 'can_traverse rover a b'
        # exists and 'visible a b' exists (because navigate precondition contains both).
        self.can_traverse_by_rover = {}
        for f in static:
            parts = get_parts(f)
            if parts[0] == "can_traverse":
                rover, a, b = parts[1], parts[2], parts[3]
                # Only include edge if visibility a->b also exists in static
                if (a, b) in self.visible_pairs:
                    self.can_traverse_by_rover.setdefault(rover, {}).setdefault(a, set()).add(b)

        # visible_from: objective -> set(waypoints)
        self.visible_from = {}
        for f in static:
            parts = get_parts(f)
            if parts[0] == "visible_from":
                obj, wp = parts[1], parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)

        # supports: camera -> set(modes)
        self.supports = {}
        for f in static:
            parts = get_parts(f)
            if parts[0] == "supports":
                cam, mode = parts[1], parts[2]
                self.supports.setdefault(cam, set()).add(mode)

        # on_board: camera -> rover
        self.on_board = {}
        for f in static:
            parts = get_parts(f)
            if parts[0] == "on_board":
                cam, rover = parts[1], parts[2]
                self.on_board[cam] = rover

        # calibration_target: camera -> objective
        self.calibration_target = {}
        for f in static:
            parts = get_parts(f)
            if parts[0] == "calibration_target":
                cam, obj = parts[1], parts[2]
                self.calibration_target[cam] = obj

        # store_of: store -> rover and rover_to_store
        self.store_of = {}
        self.rover_to_store = {}
        for f in static:
            parts = get_parts(f)
            if parts[0] == "store_of":
                store, rover = parts[1], parts[2]
                self.store_of[store] = rover
                # assume one store per rover (as in typical rovers problems)
                self.rover_to_store[rover] = store

        # equipped sets
        self.equipped_for_soil = {get_parts(f)[1] for f in static if get_parts(f)[0] == "equipped_for_soil_analysis"}
        self.equipped_for_rock = {get_parts(f)[1] for f in static if get_parts(f)[0] == "equipped_for_rock_analysis"}
        self.equipped_for_imaging = {get_parts(f)[1] for f in static if get_parts(f)[0] == "equipped_for_imaging"}

        # lander waypoints: set of waypoints where a lander is located (from at_lander facts)
        self.lander_waypoints = {get_parts(f)[2] for f in static if get_parts(f)[0] == "at_lander"}

        # Precompute the set of waypoints that are acceptable for communication:
        # any waypoint x such that (visible x y) for some y in lander_waypoints
        self.communication_waypoints = {x for (x, y) in self.visible_pairs if y in self.lander_waypoints}

        # Default large finite penalty used when a subgoal looks unreachable for all rovers.
        self.large_penalty = 1000

    def _bfs_distance(self, rover, start, targets):
        """
        BFS to compute minimal number of navigate steps from 'start' to any waypoint in 'targets'
        using the can_traverse graph for the given rover.
        Returns integer distance (0 if start in targets), or None if unreachable.
        """
        if start is None:
            return None
        if not targets:
            return None
        if start in targets:
            return 0
        graph = self.can_traverse_by_rover.get(rover, {})
        # standard BFS
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, d = queue.popleft()
            for nbr in graph.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return d + 1
                visited.add(nbr)
                queue.append((nbr, d + 1))
        return None

    def __call__(self, node):
        """Return estimated number of actions to reach the goals from node.state."""
        state = node.state

        # If all goals satisfied, return 0
        if self.goals <= state:
            return 0

        # Parse state facts for dynamic information
        rover_positions = {}          # rover -> waypoint
        empty_stores = set()          # stores currently empty
        full_stores = set()           # stores currently full
        have_image = set()            # tuples (rover, objective, mode)
        have_soil = set()             # tuples (rover, waypoint)
        have_rock = set()             # tuples (rover, waypoint)
        calibrated = set()            # tuples (camera, rover)

        for f in state:
            parts = get_parts(f)
            pred = parts[0]
            if pred == "at" and len(parts) == 3:
                # could be (at rover waypoint) or other objects; we only take rovers
                obj, loc = parts[1], parts[2]
                # Rover names are known via can_traverse_by_rover keys or equipment sets; include any that appear
                # Use heuristic: if this object appears in can_traverse_by_rover or in any equip sets, treat as rover
                if obj in self.can_traverse_by_rover or obj in self.equipped_for_imaging or obj in self.equipped_for_soil or obj in self.equipped_for_rock:
                    rover_positions[obj] = loc
            elif pred == "empty":
                empty_stores.add(parts[1])
            elif pred == "full":
                full_stores.add(parts[1])
            elif pred == "have_image":
                # (have_image rover objective mode)
                if len(parts) >= 4:
                    have_image.add((parts[1], parts[2], parts[3]))
            elif pred == "have_soil_analysis":
                if len(parts) >= 3:
                    have_soil.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis":
                if len(parts) >= 3:
                    have_rock.add((parts[1], parts[2]))
            elif pred == "calibrated":
                if len(parts) >= 3:
                    calibrated.add((parts[1], parts[2]))

        total_estimate = 0

        # Helper to compute minimal navigation distance from a rover to any of 'targets' waypoints
        def min_nav(rover, targets):
            start = rover_positions.get(rover)
            if start is None:
                return None
            return self._bfs_distance(rover, start, set(targets))

        # For each goal, compute minimal cost and sum
        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]

            # Already achieved?
            if goal in state:
                continue

            if pred == "communicated_image_data":
                # goal form: (communicated_image_data objective mode)
                obj, mode = parts[1], parts[2]
                # If some rover already has the image (but not communicated), only need communication
                best_cost = None

                for (r, o, m) in have_image:
                    if o == obj and m == mode:
                        # need to navigate to a communication waypoint and communicate
                        dist = min_nav(r, self.communication_waypoints)
                        if dist is None:
                            continue
                        cost = dist + 1  # communicate
                        if best_cost is None or cost < best_cost:
                            best_cost = cost

                if best_cost is not None:
                    total_estimate += best_cost
                    continue

                # Else, we must take the image first then communicate.
                # Consider each camera that supports the mode and is on some rover and rover equipped for imaging.
                candidate_cost = None
                for cam, modes in self.supports.items():
                    if mode not in modes:
                        continue
                    if cam not in self.on_board:
                        continue
                    rover = self.on_board[cam]
                    if rover not in self.equipped_for_imaging:
                        continue
                    # calibration objective for this camera
                    cal_obj = self.calibration_target.get(cam)
                    # calibration waypoints
                    cal_wps = self.visible_from.get(cal_obj, set()) if cal_obj is not None else set()
                    # image waypoints (where objective obj is visible from)
                    img_wps = self.visible_from.get(obj, set())
                    if not img_wps:
                        continue  # cannot take image if no visible_from entries
                    # distances:
                    #  - rover -> any cal waypoint
                    #  - cal waypoint -> any image waypoint (we use can_traverse graph of rover)
                    #  - image waypoint -> any communication waypoint
                    # If camera already calibrated on this rover, skip calibrate step
                    need_calibrate = (cam, rover) not in calibrated
                    # Evaluate minimal across actual intermediate waypoints: we approximate by taking minimal
                    # sum of distances: rover->cal + cal->img + img->comm. To keep computation light, compute:
                    # dist_rover_to_cal = min_nav(rover, cal_wps) or 0 if no cal required and rover at some img wp
                    # After calibrate rover stays at calibration wp. We then need distance from that wp to an image wp.
                    # We'll test combinations: for each cal_wp in cal_wps (or None if calibration skipped), find distances.
                    # To keep efficient, attempt these steps:
                    # If no calibration waypoints exist but calibrate required, we cannot use this camera.
                    cal_wp_candidates = cal_wps if cal_wps else (set() if need_calibrate else {rover_positions.get(rover)})

                    if need_calibrate and not cal_wp_candidates:
                        continue

                    # If no calibration required, we can start from rover's start position as 'cal_wp'
                    if not need_calibrate:
                        cal_wp_candidates = {rover_positions.get(rover)}

                    # For each candidate cal waypoint, compute distances
                    for cal_wp in cal_wp_candidates:
                        if cal_wp is None:
                            continue
                        # distance rover -> cal_wp
                        d1 = self._bfs_distance(rover, rover_positions.get(rover), {cal_wp})
                        if d1 is None:
                            continue
                        # If calibration is needed, add 1 for calibrate
                        d1_total = d1 + (1 if need_calibrate else 0)

                        # Now from cal_wp to any image waypoint
                        # Note: after calibrate rover is at cal_wp
                        # compute dist cal_wp -> any img_wp using rover graph (temporarily set start cal_wp)
                        # Use BFS from cal_wp: to do that, create a helper BFS that takes explicit start
                        def bfs_from(start_wp, rover_name, target_set):
                            if start_wp in target_set:
                                return 0
                            graph = self.can_traverse_by_rover.get(rover_name, {})
                            visited = {start_wp}
                            q = deque([(start_wp, 0)])
                            while q:
                                node, dd = q.popleft()
                                for nb in graph.get(node, ()):
                                    if nb in visited:
                                        continue
                                    if nb in target_set:
                                        return dd + 1
                                    visited.add(nb)
                                    q.append((nb, dd + 1))
                            return None

                        d2 = bfs_from(cal_wp, rover, img_wps)
                        if d2 is None:
                            continue

                        # After take_image, camera becomes uncalibrated again (domain effect), but rover is at image_wp.
                        # Add 1 for take_image.
                        d2_total = d2 + 1

                        # From whichever image waypoint we reach, go to communication waypoint and communicate
                        # Instead of enumerating actual image wp reached, approximate by computing minimal distance
                        # from any img_wp to communication waypoint using rover graph starting from img_wp.
                        d3 = None
                        # We'll compute minimal over img_wps of bfs_from(img_wp, rover, comm_wps)
                        for img_wp in img_wps:
                            # Only consider img_wp reachable from cal_wp (we tested existence above), but still check
                            dist_img_to_comm = bfs_from(img_wp, rover, self.communication_waypoints)
                            if dist_img_to_comm is not None:
                                if d3 is None or dist_img_to_comm < d3:
                                    d3 = dist_img_to_comm
                        if d3 is None:
                            continue

                        # +1 for communicate
                        total_cost = d1_total + d2_total + d3 + 1
                        if candidate_cost is None or total_cost < candidate_cost:
                            candidate_cost = total_cost

                if candidate_cost is None:
                    # No feasible camera/rover found; use large finite penalty
                    total_estimate += self.large_penalty
                else:
                    total_estimate += candidate_cost

            elif pred == "communicated_soil_data" or pred == "communicated_rock_data":
                # goal form: (communicated_soil_data waypoint) or (communicated_rock_data waypoint)
                sample_wp = parts[1]
                is_soil = pred == "communicated_soil_data"

                # If some rover already has the analysis of this sample, only need to communicate
                best_cost = None
                if is_soil:
                    for (r, wp) in have_soil:
                        if wp == sample_wp:
                            d = min_nav(r, self.communication_waypoints)
                            if d is None:
                                continue
                            cost = d + 1  # communicate
                            if best_cost is None or cost < best_cost:
                                best_cost = cost
                else:
                    for (r, wp) in have_rock:
                        if wp == sample_wp:
                            d = min_nav(r, self.communication_waypoints)
                            if d is None:
                                continue
                            cost = d + 1
                            if best_cost is None or cost < best_cost:
                                best_cost = cost

                if best_cost is not None:
                    total_estimate += best_cost
                    continue

                # Else, need to sample then communicate. Consider every rover equipped for the sample type.
                candidate_cost = None
                rover_candidates = self.equipped_for_soil if is_soil else self.equipped_for_rock
                for rover in rover_candidates:
                    # Must have a store associated
                    store = self.rover_to_store.get(rover)
                    if store is None:
                        continue
                    # distance rover -> sample_wp
                    d_to_sample = min_nav(rover, {sample_wp})
                    if d_to_sample is None:
                        continue
                    # sample action cost = 1
                    # check store empty
                    extra_drop = 0
                    if store not in empty_stores:
                        # If the store is full, a drop action can empty it. If neither full nor empty is present,
                        # assume not empty and treat it as needing a drop to make it empty (pessimistic).
                        if store in full_stores:
                            extra_drop = 1
                        else:
                            extra_drop = 1
                    # after sampling rover at sample_wp
                    # go to communication waypoint
                    # distance sample_wp -> communication waypoint (using rover's graph)
                    d_to_comm = self._bfs_distance(rover, sample_wp, self.communication_waypoints)
                    if d_to_comm is None:
                        continue
                    cost = d_to_sample + 1 + extra_drop + d_to_comm + 1  # navigate + sample + drop(if any) + nav + communicate
                    if candidate_cost is None or cost < candidate_cost:
                        candidate_cost = cost

                if candidate_cost is None:
                    total_estimate += self.large_penalty
                else:
                    total_estimate += candidate_cost

            else:
                # Unknown goal predicate type: assign small penalty (or skip). Keep heuristic finite.
                total_estimate += 1

        return int(total_estimate)