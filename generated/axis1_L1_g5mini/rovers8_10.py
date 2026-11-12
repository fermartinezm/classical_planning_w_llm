from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the list of tokens for a PDDL fact string like "(pred arg1 arg2)"."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (a rover/camera-like domain).
    The heuristic estimates the number of actions required to achieve all remaining
    goals by focusing on the domain's two main goal types:
      - lk059nof <objective> <mode>  (image/data for an objective in a given mode)
      - chfbweyv <waypoint>          (a separate "visit/transmit" style fact for waypoints)
    The heuristic is not guaranteed admissible, but aims to produce informative,
    efficiently-computable estimates to guide greedy best-first search.

    Assumptions
    -----------
    - The primary way to achieve lk059nof(objective, mode) is:
        1) move a suitable rover to a waypoint where that objective is located,
        2) attach a compatible camera to that rover (jx1xpj7f),
        3) take the image/sensor reading (tkek73dx),
        4) perform the finalizing action (bypa1zm7) that produces lk059nof.
      We approximate this with cost = (rover move distance) + 3 actions (attach, take, finalize).
    - The primary way to achieve chfbweyv(waypoint) is:
        1) move a rover to that waypoint,
        2) run a store/processing action (cp8otkvs or wyhd5xzi) depending on waypoint flags,
        3) run a finalization action (a7xj6iq1 or n84tdn2k).
      We approximate this with cost = (rover move distance) + 3 actions (store-op, final-op, small margin).
    - Static compatibility facts (camera supports mode, camera supports objective, camera<->rover compatibility,
      rover movement graph entries) are encoded in the task.static set and used to determine feasible
      camera/rover combinations and rover movement edges.
    - Rovers that lack required static predicates (e.g., jy67mgmv) are considered unable to perform certain steps.
    - If no feasible rover/camera combination or path exists for a goal according to the static info,
      a large but finite penalty is used so the heuristic remains finite.

    Heuristic Initialization
    ------------------------
    In the constructor (called once per task) we extract and index static facts to accelerate
    repeated heuristic calls:
      - adjacency_by_rover: movement edges (from xuwncpn8 facts) keyed by rover -> {wp: [neighbors]}
      - objective_waypoints: mapping objective -> set(waypoints) from wecz86om facts
      - camera_objectives: map camera -> set(objectives) from g52pa0bi facts
      - camera_modes: map camera -> set(modes) from xq9py0w8 facts
      - camera_rover_compat: map camera -> set(rovers) from uss6vmx6 facts
      - waypoint_flags: sets for 'czmzagzw' and 'hfrxxy8q' waypoints
    We also store the goal set for quick goal-checking.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node (whose .state is a frozenset of predicate strings):
      1) If all goals are already satisfied in the node's state, return 0.
      2) Build a combined "facts" view = node.state U static to allow checking preconditions that
         may be encoded as static facts.
      3) Determine current rover locations by scanning facts that match (ilkg22gp rover waypoint)
         in the combined facts. If a rover's current location is unknown, treat distance as large.
      4) For each unmet goal:
         - If it is an lk059nof(objective, mode) goal:
             * For each rover that appears usable (has jy67mgmv in combined facts or is present in the movement index),
               check if there exists a camera that:
                 - supports that objective (camera_objectives),
                 - supports that mode (camera_modes),
                 - is compatible with the rover (camera_rover_compat).
               * For each feasible rover-camera pair, compute BFS distance (using adjacency_by_rover)
                 from the rover's current waypoint to any waypoint containing the objective (objective_waypoints).
               * Estimated cost for that goal is min_over_pairs( distance + 3 ). If none feasible => large penalty.
         - If it is a chfbweyv(waypoint) goal:
             * Check whether the waypoint has czmzagzw or hfrxxy8q flags (in static or state).
               If it has neither, we conservatively assign a larger penalty (but still finite).
             * For rovers present in combined facts, compute distance to that waypoint; feasible if rover has
               required rover-mode flag (dp3yzgup or vg9nzfll) as needed by cp8otkvs/wyhd5xzi.
             * Estimated cost for that goal is min_over_rovers(distance + 3). If none feasible => large penalty.
      5) The total heuristic is the sum of the estimated costs for all currently unsatisfied goal facts.
         (This optimistic sum assumes goals can be addressed in parallel by different rovers/cameras where possible,
         but still gives a reasonable prioritized measure of how many actions remain.)
      6) All distances and penalties are capped to ensure a finite integer value is returned.

    Note on scalability and admissibility:
    - The heuristic is designed to be inexpensive (BFS on small waypoint graphs per rover where needed)
      and informative by using domain structure (camera compatibility, objective locations).
    - It is not admissible (it may underestimate or overestimate depending on resource contention),
      but it is suitable for greedy best-first search where admissibility is not required.
    """

    # A conservative finite penalty for unreachable situations (keeps heuristic finite).
    UNREACHABLE_PENALTY = 200

    def __init__(self, task):
        # store goals and static facts
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Data structures extracted from static facts
        # adjacency_by_rover: rover -> { wp_from: [wp_to, ...], ... }
        self.adjacency_by_rover = {}
        # objective -> set(waypoints)
        self.objective_waypoints = {}
        # camera -> set(objectives)
        self.camera_objectives = {}
        # camera -> set(modes)
        self.camera_modes = {}
        # camera -> set(rovers) compatibility
        self.camera_rover_compat = {}
        # waypoint flags
        self.waypoint_czmzagzw = set()  # waypoint has czmzagzw
        self.waypoint_hfrxxy8q = set()  # waypoint has hfrxxy8q

        # Other static predicates helpful for feasibility checks
        self.jy67_rovers = set()   # rovers that have jy67mgmv (required by jx1xpj7f)
        self.dp3_rovers = set()    # rovers with dp3yzgup (used by cp8otkvs)
        self.vg9_rovers = set()    # rovers with vg9nzfll (used by wyhd5xzi)
        self.fa5tam_pairs = set()  # (store, rover) pairs from fa5tam1q
        self.pmcrx_stores = set()  # stores that currently have pmcrx641 static (availability)
        self.kx_adjacency = set()  # kxvuge0a waypoint->waypoint pairs (may be useful)

        # Parse static facts
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "xuwncpn8":
                # (xuwncpn8 rover wp_from wp_to)
                rover, fr, to = parts[1], parts[2], parts[3]
                self.adjacency_by_rover.setdefault(rover, {}).setdefault(fr, []).append(to)
            elif pred == "wecz86om":
                # (wecz86om objective waypoint)
                obj, wp = parts[1], parts[2]
                self.objective_waypoints.setdefault(obj, set()).add(wp)
            elif pred == "g52pa0bi":
                # (g52pa0bi camera objective)
                cam, obj = parts[1], parts[2]
                self.camera_objectives.setdefault(cam, set()).add(obj)
            elif pred == "xq9py0w8":
                # (xq9py0w8 camera mode)
                cam, mode = parts[1], parts[2]
                self.camera_modes.setdefault(cam, set()).add(mode)
            elif pred == "uss6vmx6":
                # (uss6vmx6 camera rover)
                cam, rover = parts[1], parts[2]
                self.camera_rover_compat.setdefault(cam, set()).add(rover)
            elif pred == "czmzagzw":
                # (czmzagzw waypoint)
                self.waypoint_czmzagzw.add(parts[1])
            elif pred == "hfrxxy8q":
                # (hfrxxy8q waypoint)
                self.waypoint_hfrxxy8q.add(parts[1])
            elif pred == "jy67mgmv":
                self.jy67_rovers.add(parts[1])
            elif pred == "dp3yzgup":
                self.dp3_rovers.add(parts[1])
            elif pred == "vg9nzfll":
                self.vg9_rovers.add(parts[1])
            elif pred == "fa5tam1q":
                # (fa5tam1q store rover)
                store, rover = parts[1], parts[2]
                self.fa5tam_pairs.add((store, rover))
            elif pred == "pmcrx641":
                # (pmcrx641 store)
                self.pmcrx_stores.add(parts[1])
            elif pred == "kxvuge0a":
                # (kxvuge0a wp1 wp2)
                self.kx_adjacency.add((parts[1], parts[2]))

    def _bfs_distance(self, rover, start_wp, target_wps):
        """
        BFS that returns the minimal number of move-steps (azu4u5xz actions) for `rover`
        to reach any waypoint in `target_wps` from `start_wp` using the rover-specific
        adjacency (xuwncpn8). Returns an integer distance (0 if already at target).
        If no path is found or rover has no adjacency info, returns None.
        """
        if start_wp in target_wps:
            return 0
        if rover not in self.adjacency_by_rover:
            return None
        adj = self.adjacency_by_rover[rover]
        visited = {start_wp}
        q = deque([(start_wp, 0)])
        while q:
            cur, d = q.popleft()
            for neigh in adj.get(cur, []):
                if neigh in visited:
                    continue
                if neigh in target_wps:
                    return d + 1
                visited.add(neigh)
                q.append((neigh, d + 1))
        return None

    def __call__(self, node):
        """
        Compute heuristic estimate for node.state (a frozenset of fact strings).
        """
        state = set(node.state)  # convert to mutable set
        # Quick goal test: if all goals are in the current state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Combined facts used for checking both dynamic and static preconditions.
        combined = state | self.static

        # Discover rovers and their current waypoints from combined facts:
        # ilkg22gp rover waypoint  (rover at waypoint)
        rover_locations = {}  # rover -> waypoint
        for fact in combined:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ilkg22gp":
                rover_locations[parts[1]] = parts[2]

        # Also consider rovers known from adjacency_by_rover even if location unknown
        known_rovers = set(self.adjacency_by_rover.keys()) | set(rover_locations.keys())
        # Also include rovers that appear in other static predicates
        known_rovers |= self.jy67_rovers | self.dp3_rovers | self.vg9_rovers
        # heuristic accumulates per-goal estimates
        total_cost = 0

        # For faster queries, invert camera structures: for each objective+mode, which cameras are feasible?
        # camera feasible if supports objective and mode
        cameras_for_obj_mode = {}
        for cam in set(list(self.camera_objectives.keys()) + list(self.camera_modes.keys())):
            objs = self.camera_objectives.get(cam, set())
            modes = self.camera_modes.get(cam, set())
            for obj in objs:
                for mode in modes:
                    cameras_for_obj_mode.setdefault((obj, mode), set()).add(cam)

        # For each unsatisfied goal, estimate minimal cost
        for goal in self.goals:
            if goal in state:
                continue  # already satisfied

            parts = get_parts(goal)
            pred = parts[0]

            if pred == "lk059nof":
                # Goal: (lk059nof objective mode)
                objective, mode = parts[1], parts[2]
                # candidate waypoints for objective
                target_wps = self.objective_waypoints.get(objective, set())
                if not target_wps:
                    # if we don't know where the objective is, penalize heavily but finite
                    total_cost += self.UNREACHABLE_PENALTY
                    continue

                # cameras that can both image this objective and support the mode
                candidate_cameras = cameras_for_obj_mode.get((objective, mode), set())
                if not candidate_cameras:
                    # no camera supports both objective and mode
                    total_cost += self.UNREACHABLE_PENALTY
                    continue

                best_cost = None
                # consider each rover that could perform the task
                for rover in known_rovers:
                    # rover must have the jy67mgmv flag (precondition in jx1xpj7f)
                    if rover not in self.jy67_rovers and ("(jy67mgmv %s)" % rover) not in combined:
                        # skip rover if it doesn't satisfy the required flag
                        continue

                    # rover current location
                    start_wp = rover_locations.get(rover)
                    if start_wp is None:
                        # we don't know rover location => treat as unreachable for now
                        continue

                    # for each camera check compatibility camera<->rover (uss6vmx6)
                    for cam in candidate_cameras:
                        compat_rovers = self.camera_rover_compat.get(cam, set())
                        if rover not in compat_rovers:
                            continue
                        # BFS distance for rover to any waypoint where objective is present
                        dist = self._bfs_distance(rover, start_wp, target_wps)
                        if dist is None:
                            continue
                        # attach(1) + take(1) + finalize(1) = 3 extra actions
                        est = dist + 3
                        if best_cost is None or est < best_cost:
                            best_cost = est

                if best_cost is None:
                    total_cost += self.UNREACHABLE_PENALTY
                else:
                    total_cost += best_cost

            elif pred == "chfbweyv":
                # Goal: (chfbweyv waypoint)
                waypoint = parts[1]
                # If waypoint already has required flag in combined facts maybe cheaper
                has_czm = (waypoint in self.waypoint_czmzagzw) or ("(czmzagzw %s)" % waypoint) in combined
                has_hfr = (waypoint in self.waypoint_hfrxxy8q) or ("(hfrxxy8q %s)" % waypoint) in combined

                # We will try both processes: cp8otkvs (czm) requiring dp3yzgup rover,
                # or wyhd5xzi (hfr) requiring vg9nzfll rover.
                best_cost = None
                for rover in known_rovers:
                    start_wp = rover_locations.get(rover)
                    if start_wp is None:
                        continue

                    # Option cp8: needs czmzagzw on waypoint and rover in dp3_rovers and store availability
                    if has_czm and (rover in self.dp3_rovers or ("(dp3yzgup %s)" % rover) in combined):
                        # also require a store with pmcrx641 and mapping fa5tam1q (store->rover)
                        feasible_store = False
                        for store, r in self.fa5tam_pairs:
                            if r == rover and store in self.pmcrx_stores:
                                feasible_store = True
                                break
                            # also consider static facts that might encode pmcrx641 or fa5tam1q in combined
                        if not feasible_store:
                            # try to detect if any (pmcrx641 store) and (fa5tam1q store rover) are in combined
                            for fact in combined:
                                p = get_parts(fact)
                                if p[0] == "pmcrx641":
                                    st = p[1]
                                    if ("(fa5tam1q %s %s)" % (st, rover)) in combined:
                                        feasible_store = True
                                        break
                        if feasible_store:
                            dist = self._bfs_distance(rover, start_wp, {waypoint})
                            if dist is not None:
                                est = dist + 3  # store-op + finalize + margin
                                if best_cost is None or est < best_cost:
                                    best_cost = est

                    # Option wyhd: needs hfrxxy8q on waypoint and rover in vg9_rovers and store availability
                    if has_hfr and (rover in self.vg9_rovers or ("(vg9nzfll %s)" % rover) in combined):
                        feasible_store = False
                        for store, r in self.fa5tam_pairs:
                            if r == rover and store in self.pmcrx_stores:
                                feasible_store = True
                                break
                        if not feasible_store:
                            for fact in combined:
                                p = get_parts(fact)
                                if p[0] == "pmcrx641":
                                    st = p[1]
                                    if ("(fa5tam1q %s %s)" % (st, rover)) in combined:
                                        feasible_store = True
                                        break
                        if feasible_store:
                            dist = self._bfs_distance(rover, start_wp, {waypoint})
                            if dist is not None:
                                est = dist + 3
                                if best_cost is None or est < best_cost:
                                    best_cost = est

                if best_cost is None:
                    # If waypoint lacks both flags or no feasible rover was found, assign penalty
                    total_cost += self.UNREACHABLE_PENALTY
                else:
                    total_cost += best_cost

            else:
                # For any other goal predicates not explicitly modeled, apply a small default penalty
                # to encourage progress but remain finite.
                total_cost += 2

        # final heuristic is integer
        return int(total_cost)