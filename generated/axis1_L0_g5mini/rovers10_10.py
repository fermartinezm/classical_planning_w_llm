from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

INF = 100000  # Large finite penalty for unreachable goals


def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into parts."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact matches the pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic tailored for the PDDL "rover" domain.
    The heuristic estimates the number of actions still required to satisfy
    all communication goals:
      - communicated_image_data <objective> <mode>
      - communicated_soil_data <waypoint>
      - communicated_rock_data <waypoint>

    The heuristic is not required to be admissible. It aims to provide a
    reasonably accurate and cheaply computable estimate by reasoning about
    (1) the sensing (sampling / imaging) actions required, (2) calibration
    when imaging is necessary, (3) the communication step, and (4)
    navigation steps estimated via BFS on the per-rover traversal graph.

    Assumptions
    -----------
    - Static facts (task.static) contain can_traverse, visible, visible_from,
      calibration_target, supports, on_board, store_of, and equipped_for_*
      information. If some of these appear in the initial state instead,
      this heuristic also looks in the state's facts when necessary.
    - Navigation cost is the number of 'navigate' actions (each counts as 1).
      We assume the rover traverse graph (can_traverse facts) provides
      directed edges between waypoints for each rover.
    - Imaging requires calibrate + take_image (unless the camera is already
      calibrated or the have_image fact already holds).
    - Sampling requires a store of the rover to be empty; if not empty, one
      drop action (if available) is counted to free it.
    - Communication requires the rover to be at some waypoint that is
      visible to the lander waypoint (at_lander facts). We consider any such
      waypoint reachable for communication.
    - Interactions between different goals (e.g., sharing a calibration or a
      single navigation trip serving multiple goals) are ignored to keep
      computation cheap. The heuristic therefore sums per-goal optimistic
      single-rover costs.

    Heuristic Initialization
    ------------------------
    The constructor extracts relevant static information into convenient
    data structures:
      - per-rover adjacency lists from can_traverse facts (only edges for
        that rover)
      - visible pairs (waypoint -> waypoint)
      - visible_from: mapping objective -> set(waypoints)
      - cameras on-board each rover, supports, and calibration_target mapping
      - store_of mapping store -> rover
      - sets of rovers and waypoints seen in static facts
    The global set of lander waypoints is gathered from union of static and
    initial/state facts (some problems declare at_lander in init, others
    in static). The heuristic will check both the static facts and the
    dynamic state when computing per-node costs.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given planning node (state), the heuristic does the following:

    1. If all goal fluents are already true in the state, return 0.

    2. Parse current dynamic facts:
       - Rover positions (from "(at <rover> <waypoint>)")
       - Which stores are empty/full
       - Which cameras are currently calibrated
       - Which images / analyses have already been collected or communicated

    3. For each unsatisfied goal, estimate a minimal cost to satisfy that
       goal by considering all rovers (and cameras for image goals) capable
       of achieving it. The estimate contains:
         - Navigation steps (computed via BFS on the rover-specific graph)
           between the sequence of required waypoints (start -> calibration
           -> imaging -> communication). Distances are counted as number of
           navigate actions.
         - Action steps: number of required domain actions (calibrate,
           take_image, sample_soil/sample_rock, drop, communicate), each
           counted as 1.
       Important simplifications:
         - For image goals, we consider cameras on-board the rover that
           support the requested mode and have a declared calibration_target.
           If the required have_image fact already exists for that (rover,
           objective, mode), the cost reduces to (navigate-to-comm + communicate).
           If a camera is already calibrated for that rover (calibrated fact
           present), the calibrate action is omitted.
         - For soil/rock goals, we check whether the rover has an empty
           store; if not, but it has a full store, we account for one drop
           action to free it. Sampling is then counted as 1 action.
         - Communication requires being at a waypoint visible to the lander.
           We try to minimize navigation to any such waypoint.

    4. For each goal we take the minimal cost across all rovers (and cameras
       for image goals). The heuristic value is the sum of these per-goal
       minimal costs. If a goal appears unreachable by any rover given the
       available static connectivity and equipment, a large finite penalty
       (INF) is used for that goal.

    5. The heuristic returns an integer cost estimate.

    Notes on correctness and computational complexity:
    - Computing BFS distances on the (typically small) waypoint graph is
      efficient. Per-node work scales roughly with (number of rovers) *
      (BFS on waypoint graph) * (number of unsatisfied goals).
    - The heuristic always returns 0 for goal states and returns finite
      values (possibly large) even if parts of the state appear temporarily
      unreachable. This keeps the heuristic usable in greedy best-first search.
    """

    def __init__(self, task):
        # Goals that must be reached (set of fact strings).
        self.goals = task.goals
        # Static facts (unchanging).
        self.static = set(task.static)

        # Build per-rover adjacency lists from can_traverse facts.
        # adjacency[rover][from_wp] = set(to_wp, ...)
        self.adjacency = {}
        self.rovers = set()
        self.waypoints = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "can_traverse":
                # fact: (can_traverse rover from to)
                _, rover, from_wp, to_wp = parts
                self.rovers.add(rover)
                self.waypoints.add(from_wp)
                self.waypoints.add(to_wp)
                self.adjacency.setdefault(rover, {}).setdefault(from_wp, set()).add(to_wp)

        # visible pairs (may be required for navigation precondition and communication)
        self.visible = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "visible":
                _, a, b = parts
                self.visible.add((a, b))
                self.waypoints.add(a)
                self.waypoints.add(b)

        # visible_from: objective -> set(waypoints)
        self.visible_from = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "visible_from":
                _, obj, wp = parts
                self.visible_from.setdefault(obj, set()).add(wp)
                self.waypoints.add(wp)

        # calibration_target: camera -> objective
        self.calibration_target = {}
        # on_board: camera -> rover
        self.on_board = {}
        # supports: camera -> set(modes)
        self.supports = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "calibration_target":
                _, cam, obj = parts
                self.calibration_target[cam] = obj
            elif parts[0] == "on_board":
                _, cam, rover = parts
                self.on_board[cam] = rover
                self.rovers.add(rover)
            elif parts[0] == "supports":
                _, cam, mode = parts
                self.supports.setdefault(cam, set()).add(mode)

        # equipped sets (imaging/soil/rock)
        self.equipped_for_imaging = set()
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "equipped_for_imaging":
                self.equipped_for_imaging.add(parts[1])
                self.rovers.add(parts[1])
            elif parts[0] == "equipped_for_soil_analysis":
                self.equipped_for_soil.add(parts[1])
                self.rovers.add(parts[1])
            elif parts[0] == "equipped_for_rock_analysis":
                self.equipped_for_rock.add(parts[1])
                self.rovers.add(parts[1])

        # store_of: store -> rover
        self.store_of = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "store_of":
                _, store, rover = parts
                self.store_of[store] = rover
                self.rovers.add(rover)

        # gather cameras per rover for quick lookup
        self.cameras_on_rover = {}
        for cam, rover in self.on_board.items():
            self.cameras_on_rover.setdefault(rover, set()).add(cam)

        # Gather waypoints from adjacency/visible stuff already done; if empty, leave as seen.

    def bfs_min_distance(self, rover, start, targets):
        """
        BFS on rover-specific adjacency graph to compute minimal number of
        navigation steps (navigate actions) from start to any waypoint in
        targets. Returns INF if not reachable.
        """
        if start is None:
            return INF
        if start in targets:
            return 0
        adj = self.adjacency.get(rover, {})
        # Quick fail if adjacency empty
        if not adj:
            return INF

        visited = set()
        queue = deque()
        queue.append((start, 0))
        visited.add(start)
        while queue:
            cur, dist = queue.popleft()
            # neighbors: adjacency.get(cur, set()) but we must ensure navigate precondition
            # requires both can_traverse and visible(cur, nei). We built adjacency only from
            # can_traverse; so also check visible.
            for nei in adj.get(cur, set()):
                if (cur, nei) not in self.visible:
                    # if the visible relation is not present in static, navigation impossible
                    continue
                if nei in visited:
                    continue
                if nei in targets:
                    return dist + 1
                visited.add(nei)
                queue.append((nei, dist + 1))
        return INF

    def find_landers(self, state):
        """
        Return a list of lander waypoints found in static or state facts:
        facts of the form (at_lander <lander> <waypoint>).
        """
        lander_wps = set()
        for fact in self.static | set(state):
            parts = get_parts(fact)
            if parts[0] == "at_lander":
                _, _, wp = parts
                lander_wps.add(wp)
                self.waypoints.add(wp)
        return list(lander_wps)

    def rover_positions(self, state):
        """
        Parse state to find rover positions: (at <rover> <waypoint>)
        Returns dict rover -> waypoint (if not present, rover not included).
        """
        pos = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj = parts[1]
                wp = parts[2]
                if obj in self.rovers:
                    pos[obj] = wp
                    self.waypoints.add(wp)
        return pos

    def __call__(self, node):
        state = node.state

        # 1) If all goal facts are already true, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Dynamic data from state
        rover_pos = self.rover_positions(state)  # rover -> waypoint
        empty_stores = {get_parts(f)[1] for f in state if get_parts(f)[0] == "empty"}
        full_stores = {get_parts(f)[1] for f in state if get_parts(f)[0] == "full"}
        calibrated = {(get_parts(f)[1], get_parts(f)[2]) for f in state if get_parts(f)[0] == "calibrated"}
        have_image = {(get_parts(f)[1], get_parts(f)[2], get_parts(f)[3]) for f in state if get_parts(f)[0] == "have_image"}
        have_soil = {(get_parts(f)[1], get_parts(f)[2]) for f in state if get_parts(f)[0] == "have_soil_analysis"}
        have_rock = {(get_parts(f)[1], get_parts(f)[2]) for f in state if get_parts(f)[0] == "have_rock_analysis"}
        communicated_image = {(get_parts(f)[1], get_parts(f)[2]) for f in state if get_parts(f)[0] == "communicated_image_data"}
        communicated_soil = {get_parts(f)[1] for f in state if get_parts(f)[0] == "communicated_soil_data"}
        communicated_rock = {get_parts(f)[1] for f in state if get_parts(f)[0] == "communicated_rock_data"}

        lander_wps = self.find_landers(state)
        # If no lander is known, communication is impossible (but we use INF penalty).
        if not lander_wps:
            # All communication goals will be unreachable -> finite large cost
            # Sum INF for each goal not yet satisfied.
            unsat_goals = [g for g in self.goals if g not in state]
            return len(unsat_goals) * INF

        # Build visible relation possibly extended with dynamic facts (if any)
        visible_pairs = set(self.visible)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "visible":
                visible_pairs.add((parts[1], parts[2]))

        # Helper to get any waypoint set that can communicate to a lander waypoint
        # (i.e., wp such that visible(wp, lander_wp) holds).
        def communication_waypoints():
            comm_wps = set()
            for lander_wp in lander_wps:
                for wp in self.waypoints:
                    if (wp, lander_wp) in visible_pairs:
                        comm_wps.add(wp)
            return comm_wps

        comm_wps = communication_waypoints()

        total_cost = 0

        # Iterate through each goal to compute a per-goal minimal estimate
        for goal_fact in self.goals:
            if goal_fact in state:
                continue  # already satisfied

            parts = get_parts(goal_fact)
            predicate = parts[0]

            best_cost = INF

            if predicate == "communicated_image_data":
                # Goal: (communicated_image_data objective mode)
                _, objective, mode = parts

                # If already communicated as discovered above, skip (already handled).
                if (objective, mode) in communicated_image:
                    continue

                # imaging waypoints where objective is visible from
                img_wps = self.visible_from.get(objective, set())

                # For each rover that can perform imaging (equipped and has cameras on board)
                for rover in self.rovers:
                    # Rover must be equipped for imaging
                    if rover not in self.equipped_for_imaging and rover not in self.equipped_for_imaging:
                        # if not present in static equipped set, skip (can't image)
                        # note: some domain variants might put equipped in state; check there
                        pass  # continue checking since we may still discover equipment in state
                    # But we will check presence in state as well; if neither, skip
                    equipped_in_state = any(get_parts(f)[0] == "equipped_for_imaging" and get_parts(f)[1] == rover for f in state)
                    if (rover not in self.equipped_for_imaging) and not equipped_in_state:
                        continue

                    start = rover_pos.get(rover, None)
                    if start is None:
                        continue

                    # If some rover already has the required image, cost might be only communication
                    if (rover, objective, mode) in have_image:
                        # Only need to navigate to a comm waypoint and communicate
                        nav = self.bfs_min_distance(rover, start, comm_wps)
                        if nav < INF:
                            best_cost = min(best_cost, nav + 1)  # +1 for communicate_image_data
                        continue

                    # Consider cameras on this rover that support the mode
                    cams = self.cameras_on_rover.get(rover, set())
                    for cam in cams:
                        if mode not in self.supports.get(cam, set()):
                            continue
                        # Camera must have a calibration target declared
                        if cam not in self.calibration_target:
                            continue
                        calib_obj = self.calibration_target[cam]
                        # calibration positions are waypoints visible_from calib_obj
                        cal_wps = self.visible_from.get(calib_obj, set())
                        if not cal_wps:
                            continue
                        # imaging waypoints
                        if not img_wps:
                            continue

                        # Case: if camera already calibrated for (cam, rover) in state
                        cam_calibrated = (cam, rover) in calibrated

                        # Distances: start -> cal_wp (if needed), cal_wp -> img_wp, img_wp -> comm_wp
                        # We try all combinations of cal_wp and img_wp and comm_wp, choose minimal
                        # If camera calibrated, we can skip cal_wp and start from start -> img_wp directly.
                        if cam_calibrated:
                            # start -> img -> comm
                            for img_wp in img_wps:
                                d1 = self.bfs_min_distance(rover, start, {img_wp})
                                if d1 >= INF:
                                    continue
                                # after take_image the camera becomes not calibrated (but that's irrelevant here)
                                d2 = self.bfs_min_distance(rover, img_wp, comm_wps)
                                if d2 >= INF:
                                    continue
                                cost = d1 + 1 + d2 + 1  # navigate to img + take_image + navigate to comm + communicate
                                best_cost = min(best_cost, cost)
                        else:
                            # Need calibrate first: start -> cal -> img -> comm (plus actions: calibrate, take_image, communicate)
                            for cal_wp in cal_wps:
                                d0 = self.bfs_min_distance(rover, start, {cal_wp})
                                if d0 >= INF:
                                    continue
                                for img_wp in img_wps:
                                    d1_candidates = self.bfs_min_distance(rover, cal_wp, {img_wp})
                                    if d1_candidates >= INF:
                                        continue
                                    d2 = self.bfs_min_distance(rover, img_wp, comm_wps)
                                    if d2 >= INF:
                                        continue
                                    cost = d0 + 1 + d1_candidates + 1 + d2 + 1
                                    # d0: nav start->cal, +1 calibrate, d1 nav cal->img, +1 take_image, d2 nav ->comm, +1 communicate
                                    best_cost = min(best_cost, cost)

            elif predicate == "communicated_soil_data":
                # Goal: (communicated_soil_data waypoint)
                _, sample_wp = parts

                if sample_wp in communicated_soil:
                    continue

                # For each rover that can perform soil analysis
                for rover in self.rovers:
                    # Must be equipped for soil either in static or state
                    equipped_static = rover in self.equipped_for_soil
                    equipped_state = any(get_parts(f)[0] == "equipped_for_soil_analysis" and get_parts(f)[1] == rover for f in state)
                    if not (equipped_static or equipped_state):
                        continue

                    start = rover_pos.get(rover, None)
                    if start is None:
                        continue

                    # If rover already has soil analysis at that waypoint, only need to communicate
                    if (rover, sample_wp) in have_soil:
                        # need to navigate to comm and communicate
                        nav = self.bfs_min_distance(rover, start, comm_wps)
                        if nav < INF:
                            best_cost = min(best_cost, nav + 1)
                        continue

                    # Need to go to sample_wp, possibly drop to free a store, sample, go to comm, communicate
                    # Check if this rover has an empty store
                    rover_has_empty_store = any(s for s, r in self.store_of.items() if r == rover and s in empty_stores)
                    rover_has_full_store = any(s for s, r in self.store_of.items() if r == rover and s in full_stores)
                    # nav start -> sample_wp
                    d0 = self.bfs_min_distance(rover, start, {sample_wp})
                    if d0 >= INF:
                        continue
                    extra_drop = 0
                    if not rover_has_empty_store:
                        # If rover has a full store, we can drop it (cost 1) to free
                        if rover_has_full_store:
                            extra_drop = 1
                        else:
                            # No store belonging to the rover at all -> cannot sample
                            continue
                    # sample action = 1
                    # nav sample_wp -> comm
                    d1 = self.bfs_min_distance(rover, sample_wp, comm_wps)
                    if d1 >= INF:
                        continue
                    cost = d0 + extra_drop + 1 + d1 + 1  # nav + optional drop + sample + nav + communicate
                    best_cost = min(best_cost, cost)

            elif predicate == "communicated_rock_data":
                # Goal: (communicated_rock_data waypoint)
                _, sample_wp = parts

                if sample_wp in communicated_rock:
                    continue

                # For each rover that can perform rock analysis
                for rover in self.rovers:
                    # Must be equipped for rock either in static or state
                    equipped_static = rover in self.equipped_for_rock
                    equipped_state = any(get_parts(f)[0] == "equipped_for_rock_analysis" and get_parts(f)[1] == rover for f in state)
                    if not (equipped_static or equipped_state):
                        continue

                    start = rover_pos.get(rover, None)
                    if start is None:
                        continue

                    # If rover already has rock analysis at that waypoint, only need to communicate
                    if (rover, sample_wp) in have_rock:
                        nav = self.bfs_min_distance(rover, start, comm_wps)
                        if nav < INF:
                            best_cost = min(best_cost, nav + 1)
                        continue

                    # Need to go to sample_wp, possibly drop to free a store, sample, go to comm, communicate
                    rover_has_empty_store = any(s for s, r in self.store_of.items() if r == rover and s in empty_stores)
                    rover_has_full_store = any(s for s, r in self.store_of.items() if r == rover and s in full_stores)
                    d0 = self.bfs_min_distance(rover, start, {sample_wp})
                    if d0 >= INF:
                        continue
                    extra_drop = 0
                    if not rover_has_empty_store:
                        if rover_has_full_store:
                            extra_drop = 1
                        else:
                            continue
                    d1 = self.bfs_min_distance(rover, sample_wp, comm_wps)
                    if d1 >= INF:
                        continue
                    cost = d0 + extra_drop + 1 + d1 + 1  # nav + optional drop + sample + nav + communicate
                    best_cost = min(best_cost, cost)

            else:
                # For any unknown goal predicate, add a small penalty (shouldn't happen often).
                best_cost = min(best_cost, 1)

            # If the best cost for this goal is still INF, treat as unreachable -> add INF
            total_cost += best_cost if best_cost < INF else INF

        return int(total_cost)