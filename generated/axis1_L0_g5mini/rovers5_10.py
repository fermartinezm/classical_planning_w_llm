from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

INF = 10 ** 6


def get_parts(fact):
    """Remove surrounding parentheses and split fact into parts."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Check whether a fact matches a pattern with wildcards (*) (fact as string)."""
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'rovers' domain. It estimates the number of
    actions (navigation, sampling/calibration/take-image, communication and drops)
    required to achieve the remaining communication goals:
      - communicated_soil_data <waypoint>
      - communicated_rock_data <waypoint>
      - communicated_image_data <objective> <mode>

    The heuristic is intentionally not admissible; it aims to be informative and
    fast to compute to guide greedy best-first search.

    Assumptions
    -----------
    - Static facts provided in task.static do not change; they include:
      - can_traverse per rover (directed edges between waypoints)
      - visible and visible_from relations
      - camera on_board, supports, and calibration_target facts
      - store_of mapping from store to rover
      - equipped_for_* flags (usually static)
      - at_lander facts (where the lander is located)
    - Sampling requires an empty store; if a store is full, a single DROP action
      (cost 1) will make it empty (DROP has no location precondition).
    - Calibration is required before the first take_image with a camera, and a
      take_image action removes the calibrated fact, so each image (mode) needs
      a calibration beforehand unless the camera is currently calibrated for
      that rover-camera pair in the state.
    - Navigation cost is measured as number of NAVIGATE actions (each costs 1).
    - Communication (soil/rock/image) always requires a COMMUNICATE_* action (cost 1)
      and the rover to be at some waypoint from which the lander is visible.
    - If multiple rovers/cameras could service a goal, we choose the cheapest option
      (min over rovers/cameras). We then sum costs for all unsatisfied goals.
      This may overcount shared work but remains an effective informed heuristic
      for greedy search.

    Heuristic Initialization
    ------------------------
    In the constructor we extract static relations into efficient structures:
      - self.can_traverse[rover] : adjacency dict mapping waypoint -> set(neighbors)
      - self.visible_to[wp] : set of waypoints v such that (visible wp v) holds
                          (we use reverse mapping to find communication points to lander)
      - self.visible_from_obj[objective] : set of waypoints p where (visible_from objective p)
      - self.calibration_target[camera] : objective used to calibrate the camera
      - self.supports[camera] : set of modes supported by camera
      - self.on_board[camera] : rover the camera is on (static)
      - self.store_of[store] : rover that owns the store
      - self.at_lander_waypoints : set of waypoints where any lander is located
    Also keep self.goals for quick access.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node (state), the heuristic does:
    1. If all goals are already satisfied in the state, return 0.
    2. Parse the current state to extract:
       - rover positions (from (at rover waypoint))
       - which stores are empty/full
       - which 'have_*' and 'have_image' facts are already true
       - which cameras are currently calibrated (calibrated camera rover)
       - equipment flags present in state (union with static equipment info)
    3. For each unsatisfied goal, compute a cheapest approximate plan cost:
       - communicated_soil_data wp:
         * If some rover already has have_soil_analysis rover wp in state:
             cost = min_{such rover r} (dist(r.loc, comm_wp) + 1) where comm_wp
                    is any waypoint that can see the lander.
         * Else: choose a rover r that is (or can be) used for soil sampling:
             cost = drop_cost_if_store_full + dist(r.loc, sample_wp) + 1(sample)
                    + dist(sample_wp, comm_wp) + 1(communicate)
         * drop_cost_if_store_full is 1 if r's store is full in the current state,
           else 0. If a rover lacks equipment, it is not considered.
       - communicated_rock_data wp: analogous to soil but using rock-analysis-capable rovers.
       - communicated_image_data objective mode:
         * If have_image rover objective mode already exists in state: just navigate
           to a comm_wp and communicate => dist + 1.
         * Else: for each rover r that is equipped_for_imaging and has a camera i
           on_board that supports the requested mode:
             - Determine calibration objective t for camera i.
             - Consider waypoints p_img in visible_from[objective] (where the image must
               be taken) and p_cal in visible_from[t] (where the camera can be calibrated).
             - Navigation cost approximated as min over p_cal,p_img of:
                 dist(r.loc, p_cal) + dist(p_cal, p_img)
               (this accounts for moving to calibrate and then moving to imaging site;
               they may be equal if a waypoint can serve both roles).
             - If camera i is already calibrated for this rover in the state, omit
               the calibration action (cost 0); otherwise add calibrate cost 1.
             - Add take_image cost 1, then navigation to a comm_wp (dist p_img -> comm_wp)
               and communication cost 1.
    4. If at any point distances are unreachable in the rover's traverse graph, treat
       that option as infinitely costly and try other rovers/options.
    5. Sum the minimal cost for each unsatisfied goal and return this sum.
    Notes
    -----
    - The heuristic sums per-goal minimal costs (it does not attempt to optimally share
      actions across goals). This makes it fast and informative but not admissible.
    - Distances are shortest-path lengths in the rover-specific can_traverse graph
      (computed with BFS on demand).
    """

    def __init__(self, task):
        # Goals (a set of goal fact strings)
        self.goals = task.goals

        # Extract static facts from task.static
        static = task.static

        # can_traverse: rover -> adjacency dict of waypoint -> set(neighbours)
        self.can_traverse = {}
        for fact in static:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, rover, a, b = get_parts(fact)
                self.can_traverse.setdefault(rover, {}).setdefault(a, set()).add(b)

        # visible: mapping waypoint -> set of waypoints it can see (visible a b)
        self.visible = {}
        for fact in static:
            if match(fact, "visible", "*", "*"):
                _, a, b = get_parts(fact)
                self.visible.setdefault(a, set()).add(b)

        # visible_from: objective -> set of waypoints p where (visible_from objective p)
        self.visible_from_obj = {}
        for fact in static:
            if match(fact, "visible_from", "*", "*"):
                _, obj, wp = get_parts(fact)
                self.visible_from_obj.setdefault(obj, set()).add(wp)

        # calibration_target: camera -> objective
        self.calibration_target = {}
        for fact in static:
            if match(fact, "calibration_target", "*", "*"):
                _, cam, obj = get_parts(fact)
                self.calibration_target[cam] = obj

        # supports: camera -> set(modes)
        self.supports = {}
        for fact in static:
            if match(fact, "supports", "*", "*"):
                _, cam, mode = get_parts(fact)
                self.supports.setdefault(cam, set()).add(mode)

        # on_board: camera -> rover
        self.on_board = {}
        for fact in static:
            if match(fact, "on_board", "*", "*"):
                _, cam, rover = get_parts(fact)
                self.on_board[cam] = rover

        # store_of: store -> rover
        self.store_of = {}
        for fact in static:
            if match(fact, "store_of", "*", "*"):
                _, store, rover = get_parts(fact)
                self.store_of[store] = rover

        # equipped_for_* flags present in static (sets)
        self.static_equipped_for_soil = {
            get_parts(f)[1] for f in static if match(f, "equipped_for_soil_analysis", "*")
        }
        self.static_equipped_for_rock = {
            get_parts(f)[1] for f in static if match(f, "equipped_for_rock_analysis", "*")
        }
        self.static_equipped_for_imaging = {
            get_parts(f)[1] for f in static if match(f, "equipped_for_imaging", "*")
        }

        # where landers are located (waypoints)
        self.at_lander_waypoints = {
            get_parts(f)[2] for f in static if match(f, "at_lander", "*", "*")
        }

        # Gather list of all waypoints from static can_traverse/visible/visible_from facts
        waypoints = set()
        for fact in static:
            parts = get_parts(fact)
            if parts[0] in {"can_traverse", "visible"} and len(parts) >= 3:
                waypoints.add(parts[2])
                waypoints.add(parts[3]) if len(parts) > 3 else None
            if parts[0] in {"visible_from"} and len(parts) >= 3:
                waypoints.add(parts[2])
        self.waypoints = waypoints

    # --- Utility methods used in __call__ ---

    def _bfs_distance(self, rover, start, targets):
        """
        BFS on the rover-specific can_traverse graph to get minimum distance
        from start to any waypoint in targets. Returns INF if unreachable.
        """
        if start in targets:
            return 0
        if rover not in self.can_traverse:
            return INF
        adj = self.can_traverse[rover]
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            # neighbors may be empty
            for neigh in adj.get(node, ()):
                if neigh in visited:
                    continue
                if neigh in targets:
                    return d + 1
                visited.add(neigh)
                q.append((neigh, d + 1))
        return INF

    def _all_pairs_min_distance(self, rover, starts, targets):
        """
        Compute minimal distance between any start in `starts` and any target in `targets`
        using rover-specific traversal. Returns INF if no pair is connected.
        """
        best = INF
        for s in starts:
            d = self._bfs_distance(rover, s, targets)
            if d < best:
                best = d
        return best

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If already goal, return 0
        if self.goals <= state:
            return 0

        # ----- Parse current (non-static) info from the state -----
        # rover positions: rover -> waypoint (from (at rover waypoint))
        rover_pos = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                # only rovers are relevant here; ignore other "at" facts (packages in other domains)
                # but we can add all - rover names are known by presence in can_traverse keys or equipment.
                rover_pos[obj] = loc

        # store emptiness: store -> True if empty else False if full (if neither in state, assume empty)
        store_is_empty = {}
        for fact in state:
            if match(fact, "empty", "*"):
                _, s = get_parts(fact)
                store_is_empty[s] = True
            if match(fact, "full", "*"):
                _, s = get_parts(fact)
                store_is_empty[s] = False

        # find have_soil_analysis and have_rock_analysis facts: set of tuples
        have_soil = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "have_soil_analysis", "*", "*")}
        have_rock = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "have_rock_analysis", "*", "*")}

        # have_image facts: set of tuples (rover, objective, mode)
        have_image = {(get_parts(f)[1], get_parts(f)[2], get_parts(f)[3]) for f in state if match(f, "have_image", "*", "*", "*")}

        # calibrated cameras in current state: set of (camera, rover)
        calibrated = {tuple(get_parts(f)[1:]) for f in state if match(f, "calibrated", "*", "*")}

        # Equipment flags possibly in state (combine with statics)
        equipped_for_soil = set(self.static_equipped_for_soil)
        equipped_for_rock = set(self.static_equipped_for_rock)
        equipped_for_imaging = set(self.static_equipped_for_imaging)
        for fact in state:
            if match(fact, "equipped_for_soil_analysis", "*"):
                equipped_for_soil.add(get_parts(fact)[1])
            if match(fact, "equipped_for_rock_analysis", "*"):
                equipped_for_rock.add(get_parts(fact)[1])
            if match(fact, "equipped_for_imaging", "*"):
                equipped_for_imaging.add(get_parts(fact)[1])

        # at_lander waypoints: also consider if some at_lander facts are in the state (rare)
        at_lander_wps = set(self.at_lander_waypoints)
        for fact in state:
            if match(fact, "at_lander", "*", "*"):
                at_lander_wps.add(get_parts(fact)[2])

        # Precompute the set of communication waypoints (any waypoint x such that visible x lander_wp)
        comm_waypoints = set()
        for x, vs in self.visible.items():
            for land_wp in at_lander_wps:
                if land_wp in vs:
                    comm_waypoints.add(x)
        # If comm_waypoints empty but at_lander exists, fallback: try any waypoint visible mapping (makes heuristic weaker)
        if not comm_waypoints and at_lander_wps:
            for x in self.visible:
                comm_waypoints.add(x)

        # Helper to compute min navigation distance for a rover to reach any waypoint in a set
        def nav_dist(rover, start, targets):
            if start is None:
                return INF
            return self._bfs_distance(rover, start, targets)

        total_cost = 0

        # Evaluate each unsatisfied goal and accumulate costs
        for g in self.goals:
            if g in state:
                continue  # already satisfied

            parts = get_parts(g)
            pred = parts[0]

            # --- communicated_soil_data waypoint ---
            if pred == "communicated_soil_data":
                wp = parts[1]

                # If some rover already has the soil analysis for that waypoint, only communication remains.
                candidates = [(r, p) for (r, p) in have_soil if p == wp]
                best_cost = INF

                # If already have soil, choose cheapest communication path
                for r, _ in candidates:
                    start = rover_pos.get(r)
                    d_to_comm = nav_dist(r, start, comm_waypoints)
                    if d_to_comm >= INF:
                        continue
                    cost = d_to_comm + 1  # communicate
                    if cost < best_cost:
                        best_cost = cost

                # Otherwise, need to sample + communicate
                # Try rovers equipped for soil analysis
                for r in equipped_for_soil:
                    start = rover_pos.get(r)
                    # must have a store for this rover
                    # find store(s) of rover
                    stores = [s for s, rr in self.store_of.items() if rr == r]
                    if not stores:
                        continue
                    # use first store (there may be only one); choose the cheapest among the rover's stores
                    store_cost = INF
                    for s in stores:
                        empty = store_is_empty.get(s, True)  # missing info -> assume empty
                        drop_cost = 0 if empty else 1
                        # nav to sample location wp
                        d_to_sample = nav_dist(r, start, {wp})
                        if d_to_sample >= INF:
                            continue
                        # After sampling at wp, need to go to comm waypoint and communicate
                        d_sample_to_comm = self._bfs_distance(r, wp, comm_waypoints)
                        if d_sample_to_comm >= INF:
                            continue
                        cost = drop_cost + d_to_sample + 1 + d_sample_to_comm + 1  # drop(optional)+nav+sample+nav+comm
                        if cost < store_cost:
                            store_cost = cost
                    if store_cost < best_cost:
                        best_cost = store_cost

                if best_cost >= INF:
                    best_cost = INF // 10  # unreachable = large but finite to keep heuristic finite

                total_cost += best_cost

            # --- communicated_rock_data waypoint ---
            elif pred == "communicated_rock_data":
                wp = parts[1]
                candidates = [(r, p) for (r, p) in have_rock if p == wp]
                best_cost = INF

                for r, _ in candidates:
                    start = rover_pos.get(r)
                    d_to_comm = nav_dist(r, start, comm_waypoints)
                    if d_to_comm >= INF:
                        continue
                    cost = d_to_comm + 1
                    if cost < best_cost:
                        best_cost = cost

                for r in equipped_for_rock:
                    start = rover_pos.get(r)
                    stores = [s for s, rr in self.store_of.items() if rr == r]
                    if not stores:
                        continue
                    store_cost = INF
                    for s in stores:
                        empty = store_is_empty.get(s, True)
                        drop_cost = 0 if empty else 1
                        d_to_sample = nav_dist(r, start, {wp})
                        if d_to_sample >= INF:
                            continue
                        d_sample_to_comm = self._bfs_distance(r, wp, comm_waypoints)
                        if d_sample_to_comm >= INF:
                            continue
                        cost = drop_cost + d_to_sample + 1 + d_sample_to_comm + 1
                        if cost < store_cost:
                            store_cost = cost
                    if store_cost < best_cost:
                        best_cost = store_cost

                if best_cost >= INF:
                    best_cost = INF // 10

                total_cost += best_cost

            # --- communicated_image_data objective mode ---
            elif pred == "communicated_image_data":
                obj = parts[1]
                mode = parts[2]
                best_cost = INF

                # If some rover already has the image (have_image), only communication needed
                for (r, o, m) in have_image:
                    if o == obj and m == mode:
                        start = rover_pos.get(r)
                        d_to_comm = nav_dist(r, start, comm_waypoints)
                        if d_to_comm >= INF:
                            continue
                        cost = d_to_comm + 1
                        if cost < best_cost:
                            best_cost = cost

                # Otherwise, must take image and then communicate
                # Iterate over cameras on board rovers and check support + calibration target
                # camera -> rover mapping in self.on_board
                for cam, rover in self.on_board.items():
                    if mode not in self.supports.get(cam, set()):
                        continue
                    if rover not in equipped_for_imaging:
                        continue
                    start = rover_pos.get(rover)
                    # waypoints where image can be taken
                    img_wps = self.visible_from_obj.get(obj, set())
                    if not img_wps:
                        continue
                    # calibration objective for cam
                    calib_obj = self.calibration_target.get(cam)
                    if calib_obj is None:
                        continue
                    calib_wps = self.visible_from_obj.get(calib_obj, set())
                    if not calib_wps:
                        continue

                    # Precompute distances:
                    # We want min over p_cal in calib_wps, p_img in img_wps of dist(start->p_cal) + dist(p_cal->p_img)
                    # We will compute by trying all p_cal (may be small)
                    nav_min = INF
                    for p_cal in calib_wps:
                        d_start_to_pcal = nav_dist(rover, start, {p_cal})
                        if d_start_to_pcal >= INF:
                            continue
                        # BFS from p_cal to any p_img using rover's graph
                        d_pcal_to_pimg = self._bfs_distance(rover, p_cal, img_wps)
                        if d_pcal_to_pimg >= INF:
                            continue
                        total_nav = d_start_to_pcal + d_pcal_to_pimg
                        if total_nav < nav_min:
                            nav_min = total_nav

                    # Also consider doing calibration at the imaging location if it is also a calibration location
                    # (covered above when p_cal equals p_img via calib_wps intersection img_wps), but we still ensure it

                    if nav_min >= INF:
                        continue

                    # After imaging at p_img, need to go to a comm waypoint
                    # approximate by computing minimal distance from any p_img to any comm_waypoint
                    # For this we compute min_{p_img in img_wps} dist(p_img, comm_waypoints)
                    d_img_to_comm = INF
                    for p_img in img_wps:
                        d = self._bfs_distance(rover, p_img, comm_waypoints)
                        if d < d_img_to_comm:
                            d_img_to_comm = d
                    if d_img_to_comm >= INF:
                        continue

                    # calibration action cost: 0 if (calibrated cam rover) in state, else 1
                    calibr_cost = 0 if (cam, rover) in calibrated else 1
                    # take_image cost 1, communicate cost 1
                    cost = nav_min + calibr_cost + 1 + d_img_to_comm + 1
                    if cost < best_cost:
                        best_cost = cost

                if best_cost >= INF:
                    best_cost = INF // 10

                total_cost += best_cost

            else:
                # Unknown predicate in goals: treat as 0 (conservative) or small penalty
                total_cost += 0

        # ensure finite value (cap)
        if total_cost >= INF:
            total_cost = INF // 5
        return int(total_cost)