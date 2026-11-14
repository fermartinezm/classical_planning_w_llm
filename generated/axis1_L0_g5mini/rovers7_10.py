from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string by removing parentheses and splitting."""
    return fact.strip()[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the PDDL "rover" domain.

    The heuristic estimates the number of actions required to satisfy all goal
    communication facts (communicated_image_data, communicated_soil_data,
    communicated_rock_data). For each goal that is not yet achieved the heuristic
    estimates the minimal sequence of actions needed:
      - for images: (navigate)*, calibrate, (navigate)*, take_image, (navigate)*, communicate
      - for soil/rock: (navigate)*, (drop if store full), sample_soil/sample_rock, (navigate)*, communicate

    Assumptions
    -----------
    - Static facts (task.static) contain:
        * can_traverse facts per rover that define directed navigation edges.
        * which cameras are on which rover (on_board).
        * which camera supports which modes (supports).
        * calibration_target(camera, objective).
        * visible and visible_from facts.
        * store_of relations and at_lander facts.
        * which rovers are equipped for imaging/soil/rock (treated as static).
    - We treat each required communicated datum independently and compute a minimal
      cost for each; the heuristic then sums these minimal per-goal costs. This
      may double-count shared travel but gives a strong guiding signal.
    - Distances are counted in number of navigate actions (each navigate = 1).
    - When a required fact is already present in the state (e.g., have_image or
      have_soil_analysis), the heuristic omits the cost of producing that fact
      and only counts subsequent steps (e.g., communication).
    - If some needed path or capability is unavailable in the static graph, a
      large finite penalty is used (keeps heuristic finite).

    Heuristic Initialization
    ------------------------
    In the constructor we extract and index static facts:
      - adjacency lists per rover from can_traverse facts
      - visible edges between waypoints
      - visible_from mapping objective -> set(waypoints)
      - calibration_target mapping camera -> objective
      - supports mapping camera -> set(modes)
      - on_board mapping rover -> set(cameras)
      - store_of mapping rover -> set(stores) (usually one)
      - set of waypoints that are visible to any lander (communication target waypoints)
      - sets of rovers equipped for imaging/soil/rock

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal fact not yet true in the current state:

    1. communicated_image_data(objective, mode)
       - If already communicated -> cost 0.
       - If some rover already has have_image(rover, objective, mode) in state:
         estimate minimal navigation distance for that rover to a waypoint that is
         visible to a lander, plus 1 for the communicate action.
       - Otherwise consider each rover that is equipped_for_imaging and has an
         on-board camera supporting the mode. For each such rover/camera:
           a) Choose a calibration waypoint: any waypoint from visible_from(calibration_target(camera))
           b) Choose an imaging waypoint: any waypoint from visible_from(objective)
           c) Compute navigation distances:
               - from rover current location -> calibration waypoint (dist1)
               - calibration action (1)
               - from calibration waypoint -> imaging waypoint (dist2)
               - take_image action (1)
               - from imaging waypoint -> a waypoint visible to a lander (dist3)
               - communicate action (1)
           d) cost candidate = dist1 + 1 + dist2 + 1 + dist3 + 1
         Take minimal candidate across rovers/cameras. If any distance is unreachable,
         treat that candidate as infeasible.
       - If no feasible rover/camera/path exists, use a fixed large penalty.

    2. communicated_soil_data(waypoint) and communicated_rock_data(waypoint)
       - If already communicated -> cost 0.
       - If some rover already has have_soil_analysis/have_rock_analysis for that waypoint:
         cost = minimal navigation from that rover location to a lander-visible waypoint + 1 (communicate).
       - Otherwise, for each rover equipped to do the sampling:
           a) find the store_of store for that rover (static)
           b) if that store is not empty in the current state, then dropping is required (+1)
           c) navigation to the sample waypoint + 1 for sample action
           d) navigation from sample waypoint to a lander-visible waypoint + 1 for communicate
         Choose minimal cost across eligible rovers.
       - If infeasible for all rovers, use a finite large penalty.

    3. The heuristic value returned is the sum of the per-goal minimal costs. If
       the whole goal set is already satisfied the heuristic returns 0.

    Efficiency notes
    ----------------
    - Shortest-path calculations per rover are done by BFS over the static
      can_traverse graph. BFS results are cached (during a single __call__)
      for repeated queries from the same start node.
    - The heuristic uses a relatively coarse model (it sums independent goal
      costs) which keeps computation fast while being informative.
    """

    # large finite penalty for unreachable/infeasible goals (keeps heuristic finite)
    UNREACHABLE_PENALTY = 100

    def __init__(self, task):
        """Extract static information and goal list."""
        self.goals = task.goals
        static = task.static

        # adjacency per rover: rover -> {from_wp: set(to_wp)}
        self.adjacency = defaultdict(lambda: defaultdict(set))
        # visible edges between waypoints (from -> set(to))
        self.visible = defaultdict(set)
        # visible_from: objective -> set(waypoints)
        self.visible_from = defaultdict(set)
        # calibration_target: camera -> objective
        self.calibration_target = {}
        # supports: camera -> set(modes)
        self.supports = defaultdict(set)
        # on_board: rover -> set(cameras)
        self.on_board = defaultdict(set)
        # store_of: rover -> set(stores)
        self.store_of = defaultdict(set)
        # at_lander: set(lander_waypoints)
        lander_waypoints = set()
        # rovers equipped sets
        self.equipped_for_imaging = set()
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()

        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]
            if pred == "can_traverse":
                # (can_traverse rover from to)
                rover, frm, to = args
                self.adjacency[rover][frm].add(to)
            elif pred == "visible":
                frm, to = args
                self.visible[frm].add(to)
            elif pred == "visible_from":
                objective, wp = args
                self.visible_from[objective].add(wp)
            elif pred == "calibration_target":
                camera, objective = args
                self.calibration_target[camera] = objective
            elif pred == "supports":
                camera, mode = args
                self.supports[camera].add(mode)
            elif pred == "on_board":
                camera, rover = args
                self.on_board[rover].add(camera)
            elif pred == "store_of":
                store, rover = args
                self.store_of[rover].add(store)
            elif pred == "at_lander":
                lander, wp = args
                lander_waypoints.add(wp)
            elif pred == "equipped_for_imaging":
                rover = args[0]
                self.equipped_for_imaging.add(rover)
            elif pred == "equipped_for_soil_analysis":
                rover = args[0]
                self.equipped_for_soil_analysis.add(rover)
            elif pred == "equipped_for_rock_analysis":
                rover = args[0]
                self.equipped_for_rock_analysis.add(rover)

        # Precompute set of waypoints that are visible to any lander waypoint:
        self.lander_visible_sources = set()
        if lander_waypoints:
            # For each static visible edge x -> y where y is a lander waypoint, add x
            for src, dests in self.visible.items():
                for d in dests:
                    if d in lander_waypoints:
                        self.lander_visible_sources.add(src)

    def _bfs_distances(self, rover, start):
        """
        Return dict: waypoint -> shortest navigation distance (number of navigates)
        from start for the given rover using static adjacency.
        """
        # If rover has no adjacency info, the only reachable node is start itself.
        adj = self.adjacency.get(rover, {})
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in adj.get(cur, ()):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    def __call__(self, node):
        """Compute the heuristic value for the given search node."""

        state = node.state

        # If goal already reached, return 0
        if self.goals <= state:
            return 0

        # Parse dynamic facts from state
        # rover locations: at ?r ?wp
        rover_location = {}
        # stores empty/full
        empty_stores = set()
        full_stores = set()
        # sample location facts in state
        at_soil_sample = set()
        at_rock_sample = set()
        # have_image facts: set of tuples (rover, objective, mode)
        have_image = set()
        # have_soil_analysis facts: (rover, waypoint)
        have_soil_analysis = set()
        have_rock_analysis = set()
        # communicated facts (we will rely on goal checking, but track for convenience)
        communicated_images = set()
        communicated_soil = set()
        communicated_rock = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]
            if pred == "at" and len(args) == 2:
                # could be at rover waypoint or other objects; domain uses at rover waypoint
                obj, wp = args
                # heuristically treat rovers only
                if obj.startswith("rover"):
                    rover_location[obj] = wp
            elif pred == "empty":
                empty_stores.add(args[0])
            elif pred == "full":
                full_stores.add(args[0])
            elif pred == "at_soil_sample":
                at_soil_sample.add(args[0])
            elif pred == "at_rock_sample":
                at_rock_sample.add(args[0])
            elif pred == "have_image" and len(args) == 3:
                r, o, m = args
                have_image.add((r, o, m))
            elif pred == "have_soil_analysis" and len(args) == 2:
                r, w = args
                have_soil_analysis.add((r, w))
            elif pred == "have_rock_analysis" and len(args) == 2:
                r, w = args
                have_rock_analysis.add((r, w))
            elif pred == "communicated_image_data" and len(args) == 2:
                o, m = args
                communicated_images.add((o, m))
            elif pred == "communicated_soil_data" and len(args) == 1:
                w = args[0]
                communicated_soil.add(w)
            elif pred == "communicated_rock_data" and len(args) == 1:
                w = args[0]
                communicated_rock.add(w)

        # Cache BFS distance maps during this heuristic call to avoid recomputing
        dist_cache = {}

        def dist(rover, start, target):
            """Return navigation distance or None if unreachable."""
            if start == target:
                return 0
            key = (rover, start)
            if key not in dist_cache:
                dist_cache[key] = self._bfs_distances(rover, start)
            return dist_cache[key].get(target, None)

        total_cost = 0

        # Helper: minimal distance from a waypoint 'src' to any lander-visible waypoint
        def min_to_comm(src, rover):
            # If there are no lander-visible waypoints, communication may still be
            # feasible if some visible relation to a lander's waypoint exists in static,
            # but we precomputed lander_visible_sources. If empty, return None.
            if not self.lander_visible_sources:
                return None
            # If src itself is a lander-visible source, distance is 0
            if src in self.lander_visible_sources:
                return 0
            # BFS-based distance from src to any waypoint in lander_visible_sources
            distances = self._bfs_distances(rover, src)
            best = None
            for target in self.lander_visible_sources:
                if target in distances:
                    d = distances[target]
                    if best is None or d < best:
                        best = d
            return best

        # Process each goal independently and sum estimates
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]

            # communicated_image_data objective mode
            if pred == "communicated_image_data":
                o, m = args
                if (o, m) in communicated_images:
                    continue  # already satisfied

                best_cost = None

                # Option A: some rover already has the image -> only communicate
                for (r, obj, mode) in have_image:
                    if obj == o and mode == m:
                        r_loc = rover_location.get(r)
                        if r_loc is None:
                            continue
                        d = min_to_comm(r_loc, r)
                        if d is not None:
                            cost = d + 1  # navigate(s) + communicate
                            if best_cost is None or cost < best_cost:
                                best_cost = cost

                # Option B: need to take image then communicate
                # Iterate over rovers that can image
                for r in self.equipped_for_imaging:
                    r_loc = rover_location.get(r)
                    if r_loc is None:
                        continue
                    cameras = self.on_board.get(r, set())
                    # find cameras on this rover that support the mode
                    for cam in cameras:
                        if m not in self.supports.get(cam, ()):
                            continue
                        # calibration objective for this camera
                        calib_obj = self.calibration_target.get(cam)
                        # calibration waypoints (where calibration is possible)
                        calib_wps = set()
                        if calib_obj:
                            calib_wps = self.visible_from.get(calib_obj, set())
                        # imaging waypoints (where objective o can be imaged)
                        img_wps = self.visible_from.get(o, set())
                        if not img_wps:
                            continue
                        if not calib_wps:
                            # still possible if calibration_target missing in static,
                            # but then calibrate cannot be applied -> camera unusable
                            continue

                        # For efficiency, compute distances from start once
                        dist_from_start = self._bfs_distances(r, r_loc)

                        # Try combinations of calibration waypoint and imaging waypoint
                        for cal_wp in calib_wps:
                            if cal_wp not in dist_from_start:
                                continue
                            d1 = dist_from_start[cal_wp]
                            # compute distances from cal_wp (we need cal_wp -> img_wp)
                            dist_from_cal = self._bfs_distances(r, cal_wp)
                            for img_wp in img_wps:
                                if img_wp not in dist_from_cal:
                                    continue
                                d2 = dist_from_cal[img_wp]
                                # now from img_wp to a lander-visible waypoint
                                d3 = min_to_comm(img_wp, r)
                                if d3 is None:
                                    continue
                                # costs: d1 (to cal) + 1 (calibrate) + d2 (to img) +
                                # 1 (take_image) + d3 (to comm spot) + 1 (communicate)
                                cost = d1 + 1 + d2 + 1 + d3 + 1
                                if best_cost is None or cost < best_cost:
                                    best_cost = cost

                if best_cost is None:
                    best_cost = self.UNREACHABLE_PENALTY
                total_cost += best_cost

            # communicated_soil_data waypoint
            elif pred == "communicated_soil_data":
                w = args[0]
                if w in communicated_soil:
                    continue
                best_cost = None

                # Option A: some rover already has the soil analysis -> only communicate
                for (r, wp) in have_soil_analysis:
                    if wp == w:
                        r_loc = rover_location.get(r)
                        if r_loc is None:
                            continue
                        d = min_to_comm(r_loc, r)
                        if d is not None:
                            cost = d + 1
                            if best_cost is None or cost < best_cost:
                                best_cost = cost

                # Option B: need to sample then communicate
                # sample must be performed at waypoint w, and it must currently have at_soil_sample
                if w in at_soil_sample:
                    for r in self.equipped_for_soil_analysis:
                        r_loc = rover_location.get(r)
                        if r_loc is None:
                            continue
                        # find store of rover
                        stores = self.store_of.get(r, set())
                        # typically one store; if none, rover cannot sample
                        if not stores:
                            continue
                        # choose any store (they are static)
                        # check if that store is empty in the current state
                        extra = 0
                        store_ok = False
                        for s in stores:
                            if s in empty_stores:
                                store_ok = True
                                break
                            if s in full_stores:
                                # need drop first
                                store_ok = True
                                extra = 1
                                break
                        if not store_ok:
                            # no store facts present -> be conservative and assume drop needed
                            extra = 1

                        # compute distances
                        d_to_sample = dist(r, r_loc, w)
                        if d_to_sample is None:
                            continue
                        d_sample_to_comm = min_to_comm(w, r)
                        if d_sample_to_comm is None:
                            continue
                        cost = extra + d_to_sample + 1 + d_sample_to_comm + 1
                        # extra (drop if needed) + navigate + sample + navigate + communicate
                        if best_cost is None or cost < best_cost:
                            best_cost = cost

                if best_cost is None:
                    best_cost = self.UNREACHABLE_PENALTY
                total_cost += best_cost

            # communicated_rock_data waypoint
            elif pred == "communicated_rock_data":
                w = args[0]
                if w in communicated_rock:
                    continue
                best_cost = None

                # Option A: some rover already has the rock analysis -> only communicate
                for (r, wp) in have_rock_analysis:
                    if wp == w:
                        r_loc = rover_location.get(r)
                        if r_loc is None:
                            continue
                        d = min_to_comm(r_loc, r)
                        if d is not None:
                            cost = d + 1
                            if best_cost is None or cost < best_cost:
                                best_cost = cost

                # Option B: need to sample then communicate
                if w in at_rock_sample:
                    for r in self.equipped_for_rock_analysis:
                        r_loc = rover_location.get(r)
                        if r_loc is None:
                            continue
                        # store handling
                        stores = self.store_of.get(r, set())
                        if not stores:
                            continue
                        extra = 0
                        store_ok = False
                        for s in stores:
                            if s in empty_stores:
                                store_ok = True
                                break
                            if s in full_stores:
                                store_ok = True
                                extra = 1
                                break
                        if not store_ok:
                            extra = 1

                        d_to_sample = dist(r, r_loc, w)
                        if d_to_sample is None:
                            continue
                        d_sample_to_comm = min_to_comm(w, r)
                        if d_sample_to_comm is None:
                            continue
                        cost = extra + d_to_sample + 1 + d_sample_to_comm + 1
                        if best_cost is None or cost < best_cost:
                            best_cost = cost

                if best_cost is None:
                    best_cost = self.UNREACHABLE_PENALTY
                total_cost += best_cost

            else:
                # ignore other goal types (domain primarily contains communicated_*)
                # but keep finite by not adding cost
                continue

        return total_cost