from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Remove surrounding parentheses and split a PDDL fact string into parts.
    Example: "(at rover1 waypoint2)" -> ["at", "rover1", "waypoint2"]
    """
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Match a fact against a pattern using shell-style wildcards.
    The fact is given as a string like "(at rover1 waypoint2)".
    Pattern example: ("at", "*", "waypoint2")
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "rovers" domain. It estimates the number
    of actions needed to achieve the remaining (unsatisfied) goals by
    constructing simple task-specific subplans for each goal and summing their
    estimated costs. The heuristic is intentionally not admissible but aims to
    be informative and cheaply computable.

    Assumptions
    -----------
    - Navigation cost between waypoints is measured as the minimum number of
      `navigate` actions (each costs 1). The graph of traversable arcs is
      determined by static `can_traverse` facts for each rover and filtered
      by static `visible` preconditions that must hold for each traverse.
    - Sampling actions (sample_soil/sample_rock), calibrate, take_image,
      communicate_* and drop each count as a single action.
    - A rover's store is either `empty` or `full` in a state. If the store is
      `full` and the rover needs to sample, a `drop` action (cost 1) is needed
      first.
    - Communication always requires the rover to stand at a waypoint `x` such
      that there exists a lander waypoint `y` with both `(at_lander lander y)`
      (static) and `(visible x y)` (static). We precompute the set of such
      communication-capable waypoints.
    - Imaging requires a suitable camera on board the rover supporting the
      requested mode. If the needed image already exists (have_image) then only
      communication (and navigation to a communication waypoint) is required.
    - Calibration is necessary unless a `(calibrated camera rover)` fact is
      already present in the state for the camera-rover pair. A single
      calibrate action suffices before taking a single image (calibration
      targets are considered via their `visible_from` static facts).
    - The heuristic treats goals independently and sums per-goal estimates.
      This disregards inter-goal synergies (e.g., a single navigation satisfying
      multiple goals), which keeps computation cheap.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and prepare the following static structures:
    - adjacency per rover (waypoint graph) using static `can_traverse` and
      `visible` facts (navigate requires both).
    - mapping of lander -> waypoint from `at_lander` and the derived set of
      communication-capable waypoints from which any lander waypoint is visible.
    - `visible_from` mapping: objective -> set of waypoints where that objective
      is visible (used for image capture and calibration).
    - `calibration_target` mapping: camera -> objective that serves as its
      calibration target (then we use `visible_from` for that objective to get
      calibration waypoints).
    - `supports` mapping: camera -> set of modes it supports.
    - `on_board` mapping: camera -> rover (to find which camera is on which rover).
    - `equipped_for_*` sets for quick capability checks (soil, rock, imaging).
    - `store_of` mapping: store -> rover and rover -> store.
    We also prepare caches for BFS shortest-path lookups per rover to keep the
    heuristic efficient.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state:
    1. If all goal facts are already satisfied in the state, return 0.
    2. Parse the state to obtain dynamic facts:
       - Rover positions: (at rover waypoint)
       - Store states: which stores are `empty` or `full`
       - Which images/analyses have already been gathered (have_image / have_*_analysis)
       - Which cameras are currently calibrated: (calibrated camera rover)
    3. For each unsatisfied goal fact, compute a small estimated subplan cost:
       - communicated_soil_data waypointW:
         * If some rover already has `have_soil_analysis rover waypointW`, cost
           is navigation from that rover's current location to any
           communication-capable waypoint + 1 (communicate).
         * Otherwise, for each rover r that is statically `equipped_for_soil_analysis`:
           - If r's store is `full`, add cost 1 for `drop`.
           - Add navigation cost from r's current location to waypointW.
           - Add 1 for `sample_soil`.
           - Add navigation cost from waypointW to a communication waypoint.
           - Add 1 for `communicate_soil_data`.
           Choose the rover with minimum resulting cost. If no rover can reach
           the sample or communication location, consider the cost large.
       - communicated_rock_data waypointW: analogous to soil but uses rock
         equipment.
       - communicated_image_data objective mode:
         * If there's already some `have_image rover objective mode`, cost is
           navigation from that rover's location to a communication waypoint +
           1 (communicate).
         * Otherwise consider rovers `r` that have an on-board camera `c`
           supporting `mode` and are `equipped_for_imaging`:
           - Determine calibration waypoints for the camera `c` (from its
             calibration target's `visible_from` entries). Calibration costs 1
             (unless `(calibrated c r)` is already true).
           - Determine image-capture waypoints for the objective (`visible_from`).
           - Determine navigation route visiting required waypoints (calibrate
             and capture) in the best order, then navigate to a communication
             waypoint and add costs for actions:
             calibrate (if needed) + take_image + communicate.
           - Use BFS to get navigation costs between waypoints for the rover.
           Pick the minimal rover-camera combination.
    4. Sum minimal per-goal costs to obtain the final heuristic estimate.
    5. If some goal is provably impossible (no capable rover/camera or no path),
       a large finite penalty is added for that goal so the heuristic remains
       finite but strongly disfavors the branch.

    The heuristic intentionally trades off admissibility for informativeness and
    runtime: it builds cheap, decoupled subplans per goal and uses BFS on the
    static traverse graph to estimate navigation steps.
    """

    # A finite large penalty for impossible/unreachable subplans (keeps heuristic finite).
    LARGE_PENALTY = 1000

    def __init__(self, task):
        # Goals and static facts
        self.goals = task.goals
        static = task.static

        # Build adjacency graphs per rover for navigation.
        # An edge (y -> z) for rover r exists iff (can_traverse r y z) AND (visible y z).
        self.rover_adjacency = {}  # rover -> {waypoint: set(neighbors)}
        self.rovers = set()

        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "can_traverse":
                _, rover, a, b = parts
                self.rovers.add(rover)
                # check visibility: need (visible a b) static too
                if ("(visible %s %s)" % (a, b)) in static:
                    self.rover_adjacency.setdefault(rover, {}).setdefault(a, set()).add(b)

        # Ensure rovers with no can_traverse facts still appear (from other static facts)
        for fact in static:
            parts = get_parts(fact)
            # on_board, equipped, store_of can reveal rovers
            if parts[0] in ("on_board", "equipped_for_imaging", "equipped_for_soil_analysis", "equipped_for_rock_analysis", "store_of"):
                # rover name may be at different position depending on predicate
                for part in parts[1:]:
                    if part.startswith("rover"):
                        self.rovers.add(part)
                        self.rover_adjacency.setdefault(part, {})

        # Precompute set of communication-capable waypoints:
        # comm_waypoints = {x | exists lander waypoint y with (at_lander L y) and (visible x y)}
        lander_waypoints = set()
        for fact in static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, wp = get_parts(fact)
                lander_waypoints.add(wp)
        self.comm_waypoints = set()
        for fact in static:
            if match(fact, "visible", "*", "*"):
                _, src, dst = get_parts(fact)
                if dst in lander_waypoints:
                    self.comm_waypoints.add(src)

        # visible_from mapping: objective -> set(waypoints)
        self.visible_from = {}
        for fact in static:
            if match(fact, "visible_from", "*", "*"):
                _, objective, wp = get_parts(fact)
                self.visible_from.setdefault(objective, set()).add(wp)

        # calibration_target: camera -> objective
        self.calibration_target = {}
        for fact in static:
            if match(fact, "calibration_target", "*", "*"):
                _, cam, obj = get_parts(fact)
                self.calibration_target[cam] = obj

        # supports: camera -> set(modes)
        self.camera_supports = {}
        for fact in static:
            if match(fact, "supports", "*", "*"):
                _, cam, mode = get_parts(fact)
                self.camera_supports.setdefault(cam, set()).add(mode)

        # on_board: camera -> rover
        self.on_board = {}
        for fact in static:
            if match(fact, "on_board", "*", "*"):
                _, cam, rover = get_parts(fact)
                self.on_board[cam] = rover
                self.rovers.add(rover)
                self.rover_adjacency.setdefault(rover, {})

        # equipment sets
        self.equipped_for_soil = {get_parts(f)[1] for f in static if match(f, "equipped_for_soil_analysis", "*")}
        self.equipped_for_rock = {get_parts(f)[1] for f in static if match(f, "equipped_for_rock_analysis", "*")}
        self.equipped_for_imaging = {get_parts(f)[1] for f in static if match(f, "equipped_for_imaging", "*")}

        # store_of and reverse mapping rover -> store
        self.store_of = {}
        self.rover_store = {}
        for fact in static:
            if match(fact, "store_of", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_of[store] = rover
                self.rover_store[rover] = store

        # BFS caches: (rover, start, goal) -> distance (int) or None if unreachable
        self._bfs_cache = {}

    def _bfs_distance(self, rover, start, goal):
        """
        Shortest number of navigate steps for `rover` from `start` to `goal`.
        Returns an integer distance or None if unreachable. Uses and populates cache.
        """
        if start == goal:
            return 0
        key = (rover, start, goal)
        if key in self._bfs_cache:
            return self._bfs_cache[key]

        adjacency = self.rover_adjacency.get(rover, {})
        visited = {start}
        q = deque([(start, 0)])
        found = None
        while q:
            cur, d = q.popleft()
            neighbors = adjacency.get(cur, ())
            for n in neighbors:
                if n in visited:
                    continue
                if n == goal:
                    found = d + 1
                    q.clear()
                    break
                visited.add(n)
                q.append((n, d + 1))
        self._bfs_cache[key] = found
        return found

    def __call__(self, node):
        state = node.state

        # If already at goal return 0
        if self.goals <= state:
            return 0

        # Dynamic information from state
        # - rover positions
        rover_pos = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, subject, wp = get_parts(fact)
                # only keep if subject looks like a rover
                if subject.startswith("rover"):
                    rover_pos[subject] = wp

        # - store statuses (empty/full)
        empty_stores = {get_parts(f)[1] for f in state if match(f, "empty", "*")}
        full_stores = {get_parts(f)[1] for f in state if match(f, "full", "*")}

        # - have images and analyses
        have_image = set()  # tuples (rover, objective, mode)
        for fact in state:
            if match(fact, "have_image", "*", "*", "*"):
                _, r, o, m = get_parts(fact)
                have_image.add((r, o, m))
        have_soil = set()
        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                _, r, w = get_parts(fact)
                have_soil.add((r, w))
        have_rock = set()
        for fact in state:
            if match(fact, "have_rock_analysis", "*", "*"):
                _, r, w = get_parts(fact)
                have_rock.add((r, w))

        # calibrated facts
        calibrated = { (get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "calibrated", "*", "*") }
        # e.g., set of (camera, rover)

        total_cost = 0

        # Helper to find minimal nav cost for a rover to any comm waypoint
        def min_nav_to_comm(rover, start_wp):
            best = None
            for cw in self.comm_waypoints:
                d = self._bfs_distance(rover, start_wp, cw)
                if d is not None:
                    if best is None or d < best:
                        best = d
            return best

        # Process each missing goal
        for goal in self.goals:
            if goal in state:
                continue  # already satisfied

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                _, sample_wp = parts
                best_cost = None
                # If some rover already has the soil analysis, only need to communicate
                for (r, w) in have_soil:
                    if w == sample_wp:
                        # nav from rover_pos[r] to comm + 1
                        start = rover_pos.get(r)
                        if start is None:
                            continue
                        nav = min_nav_to_comm(r, start)
                        if nav is None:
                            continue
                        cost = nav + 1
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                # Otherwise, try to pick a rover to sample
                for r in list(self.rovers):
                    if r not in self.equipped_for_soil:
                        continue
                    start = rover_pos.get(r)
                    if start is None:
                        continue
                    # check store
                    store = self.rover_store.get(r)
                    drop_cost = 0
                    if store is None:
                        # No store known for this rover; cannot sample
                        continue
                    if store in full_stores:
                        drop_cost = 1
                    # nav to sample location
                    d1 = self._bfs_distance(r, start, sample_wp)
                    if d1 is None:
                        continue
                    # sample cost 1
                    # then nav from sample_wp to comm waypoint
                    nav_to_comm = min_nav_to_comm(r, sample_wp)
                    if nav_to_comm is None:
                        continue
                    cost = drop_cost + d1 + 1 + nav_to_comm + 1  # drop? + nav to sample + sample + nav to comm + communicate
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                if best_cost is None:
                    total_cost += self.LARGE_PENALTY
                else:
                    total_cost += best_cost

            elif predicate == "communicated_rock_data":
                _, sample_wp = parts
                best_cost = None
                # If some rover already has rock analysis
                for (r, w) in have_rock:
                    if w == sample_wp:
                        start = rover_pos.get(r)
                        if start is None:
                            continue
                        nav = min_nav_to_comm(r, start)
                        if nav is None:
                            continue
                        cost = nav + 1
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                # Otherwise, search for rover to sample
                for r in list(self.rovers):
                    if r not in self.equipped_for_rock:
                        continue
                    start = rover_pos.get(r)
                    if start is None:
                        continue
                    store = self.rover_store.get(r)
                    drop_cost = 0
                    if store is None:
                        continue
                    if store in full_stores:
                        drop_cost = 1
                    d1 = self._bfs_distance(r, start, sample_wp)
                    if d1 is None:
                        continue
                    nav_to_comm = min_nav_to_comm(r, sample_wp)
                    if nav_to_comm is None:
                        continue
                    cost = drop_cost + d1 + 1 + nav_to_comm + 1
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                if best_cost is None:
                    total_cost += self.LARGE_PENALTY
                else:
                    total_cost += best_cost

            elif predicate == "communicated_image_data":
                _, objective, mode = parts
                best_cost = None
                # If some rover already has the image (but not communicated), only need to nav+communicate
                for (r, o, m) in have_image:
                    if o == objective and m == mode:
                        start = rover_pos.get(r)
                        if start is None:
                            continue
                        nav = min_nav_to_comm(r, start)
                        if nav is None:
                            continue
                        cost = nav + 1
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                # Otherwise, evaluate rovers with cameras that support mode and are equipped
                # Consider each camera on a rover
                for cam, r in self.on_board.items():
                    # camera must support mode and rover must be equipped for imaging
                    if mode not in self.camera_supports.get(cam, set()):
                        continue
                    if r not in self.equipped_for_imaging:
                        continue
                    start = rover_pos.get(r)
                    if start is None:
                        continue
                    # image capture waypoints for objective
                    image_wps = self.visible_from.get(objective, set())
                    if not image_wps:
                        continue
                    # calibration objective and its waypoints
                    calib_obj = self.calibration_target.get(cam)
                    calib_wps = set()
                    if calib_obj is not None:
                        calib_wps = self.visible_from.get(calib_obj, set())
                    # If no calibration waypoints exist for this camera, calibration impossible
                    # but calibration might not be necessary if camera is already calibrated for this rover
                    camera_calibrated = (cam, r) in calibrated
                    # We need to plan a route that visits needed points:
                    # - a calibration waypoint (unless already calibrated)
                    # - an image capture waypoint
                    # - a communication waypoint
                    # We will iterate over combinations of candidate waypoints (small sets) and
                    # try permutations of visit order to approximate minimal nav cost.
                    # Candidate sets:
                    cand_calibs = calib_wps if not camera_calibrated else {start}  # if calibrated, no need to travel to calib points
                    cand_images = image_wps
                    cand_comms = self.comm_waypoints
                    if not cand_calibs and not camera_calibrated:
                        # Camera requires calibration but no calibration waypoints -> impossible
                        continue
                    # Limit candidate sizes for performance (they're usually small)
                    # Build small lists
                    cand_calibs = list(cand_calibs)[:6]
                    cand_images = list(cand_images)[:6]
                    cand_comms = list(cand_comms)[:6]
                    # For each choice of cal, img, comm compute minimal nav route visiting them in any order
                    for cal_wp in cand_calibs:
                        for img_wp in cand_images:
                            for comm_wp in cand_comms:
                                # We need distances between points with respect to this rover
                                d_start_cal = 0 if (camera_calibrated and cal_wp == start) else self._bfs_distance(r, start, cal_wp)
                                if d_start_cal is None:
                                    continue
                                # When camera already calibrated, we may set cal action cost to 0
                                d_cal_img = self._bfs_distance(r, cal_wp, img_wp)
                                if d_cal_img is None:
                                    continue
                                d_img_comm = self._bfs_distance(r, img_wp, comm_wp)
                                if d_img_comm is None:
                                    continue
                                # Consider orders:
                                # If we are already calibrated, we can skip calibration and go start->img->comm
                                if camera_calibrated:
                                    nav_cost = self._bfs_distance(r, start, img_wp)
                                    if nav_cost is None:
                                        continue
                                    action_cost = 1  # take_image
                                    total_here = nav_cost + action_cost + d_img_comm + 1  # nav to img + take + nav to comm + communicate
                                else:
                                    # Need calibrate (1), take_image (1), communicate (1)
                                    # Best ordering typically: start->calibrate->image->comm
                                    nav1 = d_start_cal
                                    nav2 = d_cal_img
                                    nav3 = d_img_comm
                                    if nav1 is None or nav2 is None or nav3 is None:
                                        continue
                                    total_here = nav1 + 1 + nav2 + 1 + nav3 + 1  # nav to cal + calibrate + nav to img + take + nav to comm + communicate
                                # update best
                                if best_cost is None or total_here < best_cost:
                                    best_cost = total_here
                if best_cost is None:
                    total_cost += self.LARGE_PENALTY
                else:
                    total_cost += best_cost

            else:
                # Unknown goal predicate type: fallback large penalty
                total_cost += self.LARGE_PENALTY

        return int(total_cost)