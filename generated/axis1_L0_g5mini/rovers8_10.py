from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

INF_COST = 9999


def get_parts(fact):
    """Remove surrounding parentheses and split into components."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Match a fact string against a pattern (wildcards `*` supported).
    Example: match("(at rover1 wp1)", "at", "*", "wp1") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the PDDL "rover" domain. The heuristic
    estimates the number of actions required to satisfy outstanding
    communication goals (communicated_soil_data, communicated_rock_data,
    communicated_image_data). The estimate is computed by considering the
    minimal sequence of operations a rover must perform to collect the
    required data (sample or image), optionally calibrate a camera, move
    to an uplink (a waypoint that can "see" a lander), and finally
    communicate the data. Distances between waypoints are measured as the
    minimum number of navigate actions (BFS on static can_traverse graph
    for each rover).

    Assumptions
    -----------
    - Each data goal will be achieved by a single rover (we take the best
      rover for each goal independently).
    - Stores belong to a single rover (the domain enforces store_of facts).
    - Camera on-board relationships, supports, calibration_target, can_traverse,
      and visible/visible_from facts are static and available in task.static.
    - Communicate actions require the rover to be at a waypoint that is
      visible to a waypoint with the lander (at_lander).
    - We approximate cost by summing minimal independent costs per goal;
      interactions (e.g., one rover doing multiple goals in one trip) are
      not explicitly accounted for, so the heuristic is not admissible but
      aims to be informative and efficiently computable.

    Heuristic Initialization
    ------------------------
    In the constructor we extract static structures that are used to compute
    costs quickly:
      - adjacency lists per rover (directed) from "can_traverse" facts,
        used for BFS distance computations.
      - mapping store -> rover from "store_of".
      - sets for equipped capabilities: imaging, soil, rock per rover.
      - camera on-board mapping camera -> rover, and supports mapping camera -> set(modes).
      - calibration_target mapping camera -> objective.
      - visible mapping (wp -> set of wps it can see) from "visible".
      - visible_from mapping objective -> set of waypoints where that objective
        is visible from (used for imaging and calibration).
      - the set of waypoints that are "lander" waypoints from static at_lander facts.
      - list of all rovers discovered statically (from can_traverse / equipped facts).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state) the heuristic proceeds as follows:

    1. If the node state already satisfies all goals, return 0.

    2. Extract dynamic state tidbits needed:
       - current waypoint of each rover (facts of the form "(at rover wp)").
       - which stores are empty/full in the current state.
       - which analyses or images are already available (have_soil_analysis,
         have_rock_analysis, have_image).
       - which cameras are currently calibrated (calibrated facts).
       - optionally, at_lander facts if they are not static.

    3. Precompute the set of uplink waypoints:
       - uplinks = {x | exists y (at_lander l y in static or state) and visible(x,y) }.
       These are the waypoints where a rover may perform the communicate_* action.

    4. For each unsatisfied goal fact:
       - If it is communicated_soil_data at wp:
            * For each rover capable of soil analysis (equipped_for_soil_analysis)
              and for which we can compute distances:
              - If have_soil_analysis already true for (r,wp), cost =
                dist(r,current_wp,nearest_uplink) + 1 (communicate).
              - Else cost = dist(current->wp) + (drop_cost if store full) + 1 (sample)
                + dist(wp->nearest_uplink) + 1 (communicate).
            * Take minimal cost among all rovers. If none suitable, return a large cost.
       - If it is communicated_rock_data at wp: analogous to soil but using rock capability.
       - If it is communicated_image_data (objective, mode):
            * For each rover that has at least one camera on-board supporting the mode:
              - If have_image already exists, cost = dist(current->nearest_uplink) + 1.
              - Else consider:
                  - possible image waypoints P = visible_from(objective)
                  - for chosen camera i with calibration_target t, possible calibration
                    waypoints W = visible_from(t)
                  - If camera already calibrated at this state -> no calibrate step.
                    otherwise calibration adds +1 and requires visiting some w in W.
                  - total cost = dist(current->w) + (1 if calibrate) + dist(w->p)
                                 + 1 (take_image) + dist(p->nearest_uplink) + 1 (communicate)
                    where p in P, w in W (they may be equal). Choose minimal feasible pair.
            * Take minimal cost across cameras and rovers.

    5. Sum the minimal costs for all unsatisfied goals and return that sum.

    Notes on costs:
    - Each navigate action costs 1 (we count number of edges in shortest path).
    - sample_soil/sample_rock/calibrate/take_image/communicate/drop each cost 1.
    - If a necessary path does not exist for a rover (unreachable), that rover
      is ignored for that particular goal. If no rover can accomplish a goal,
      the heuristic uses a large finite penalty.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # adjacency per rover for can_traverse edges: rover -> {wp: [wps...]}
        self.adj = {}
        # rovers set (collect from static facts)
        self.rovers = set()

        # stores: store -> rover
        self.store_of = {}
        # equip sets
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()

        # cameras: on_board camera -> rover, supports camera -> set(modes)
        self.camera_on_board = {}
        self.camera_supports = {}
        # camera calibration target: camera -> objective
        self.calibration_target = {}

        # visible and visible_from maps
        # visible_map: wp -> set(wp2) where (visible wp wp2)
        self.visible_map = {}
        # visible_from: objective -> set(waypoints)
        self.visible_from = {}

        # statically-known lander waypoints
        self.static_lander_waypoints = set()

        # Parse static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "can_traverse":
                # (can_traverse rover wp1 wp2)
                _, rover, wp1, wp2 = parts
                self.rovers.add(rover)
                self.adj.setdefault(rover, {}).setdefault(wp1, []).append(wp2)
            elif pred == "store_of":
                _, store, rover = parts
                self.store_of[store] = rover
            elif pred == "equipped_for_soil_analysis":
                _, rover = parts
                self.equipped_for_soil.add(rover)
            elif pred == "equipped_for_rock_analysis":
                _, rover = parts
                self.equipped_for_rock.add(rover)
            elif pred == "equipped_for_imaging":
                _, rover = parts
                self.equipped_for_imaging.add(rover)
            elif pred == "on_board":
                _, camera, rover = parts
                self.camera_on_board[camera] = rover
            elif pred == "supports":
                _, camera, mode = parts
                self.camera_supports.setdefault(camera, set()).add(mode)
            elif pred == "calibration_target":
                _, camera, objective = parts
                self.calibration_target[camera] = objective
            elif pred == "visible":
                _, a, b = parts
                self.visible_map.setdefault(a, set()).add(b)
            elif pred == "visible_from":
                _, obj, wp = parts
                self.visible_from.setdefault(obj, set()).add(wp)
            elif pred == "at_lander":
                # (at_lander lander waypoint)
                _, lander, wp = parts
                self.static_lander_waypoints.add(wp)
            else:
                # some other static facts ignored here
                pass

        # Build reverse map: rover -> list of stores (usually one)
        self.rover_stores = {}
        for store, rover in self.store_of.items():
            self.rover_stores.setdefault(rover, []).append(store)

    def _bfs_distance(self, rover, start, goal):
        """
        BFS to compute the minimal number of 'navigate' actions for a rover
        to go from start to goal. Uses the rover-specific adjacency map.
        Returns INF_COST if unreachable or adjacency missing (but if start==goal returns 0).
        """
        if start == goal:
            return 0
        if rover not in self.adj:
            # If no traversal info for this rover, it's immobile (cannot move)
            return INF_COST
        adj = self.adj[rover]
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            cur, d = q.popleft()
            for nbr in adj.get(cur, []):
                if nbr == goal:
                    return d + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, d + 1))
        return INF_COST

    def _nearest_uplink_distance(self, rover, start, uplinks):
        """
        Distance from start to nearest uplink waypoint (set uplinks). Returns INF_COST
        if none reachable.
        """
        if not uplinks:
            return INF_COST
        best = INF_COST
        for upl in uplinks:
            d = self._bfs_distance(rover, start, upl)
            if d < best:
                best = d
        return best

    def __call__(self, node):
        state = node.state

        # If the goal already reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract dynamic information from the current state
        # rover positions: rover -> waypoint
        rover_pos = {}
        # stores emptiness: store -> True if empty, False if full (if neither present, treat as empty)
        store_empty = {}
        # have flags
        have_soil = set()      # tuples (rover, wp)
        have_rock = set()      # tuples (rover, wp)
        have_image = set()     # tuples (rover, objective, mode)
        calibrated = set()     # tuples (camera, rover)

        # include dynamic at_lander facts in case lander positions are dynamic
        dynamic_lander_waypoints = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at":
                # (at rover waypoint)
                _, rover, wp = parts
                if rover.startswith("rover"):
                    rover_pos[rover] = wp
                    self.rovers.add(rover)
            elif pred == "empty":
                _, store = parts
                store_empty[store] = True
            elif pred == "full":
                _, store = parts
                store_empty[store] = False
            elif pred == "have_soil_analysis":
                _, rover, wp = parts
                have_soil.add((rover, wp))
            elif pred == "have_rock_analysis":
                _, rover, wp = parts
                have_rock.add((rover, wp))
            elif pred == "have_image":
                _, rover, obj, mode = parts
                have_image.add((rover, obj, mode))
            elif pred == "calibrated":
                _, camera, rover = parts
                calibrated.add((camera, rover))
            elif pred == "at_lander":
                _, lander, wp = parts
                dynamic_lander_waypoints.add(wp)
            else:
                # other facts not needed here explicitly
                pass

        # uplink waypoints: waypoints x such that exists lander waypoint y with visible(x,y)
        lander_waypoints = set(self.static_lander_waypoints) | dynamic_lander_waypoints
        uplinks = set()
        if lander_waypoints:
            for x, visibles in self.visible_map.items():
                if any(y in lander_waypoints for y in visibles):
                    uplinks.add(x)

        total_cost = 0

        # Helper to find stores of a rover and whether they are empty in current state
        def rover_store_status(rover):
            stores = self.rover_stores.get(rover, [])
            # Domain typically has a single store per rover, pick the first or None
            if not stores:
                return None, True
            s = stores[0]
            # If store_empty doesn't include s, assume empty (safe default for planning)
            return s, store_empty.get(s, True)

        # Evaluate each goal
        for goal in self.goals:
            if goal in state:
                # already satisfied
                continue
            parts = get_parts(goal)
            pred = parts[0]

            best_for_goal = INF_COST

            # Helper: iterate candidate rovers (rover list from discovered static/dynamic)
            candidate_rovers = list(self.rovers)

            if pred == "communicated_soil_data":
                # goal: (communicated_soil_data waypoint)
                _, wp = parts
                for r in candidate_rovers:
                    # must have rover position to plan moves
                    if r not in rover_pos:
                        continue
                    cur_wp = rover_pos[r]
                    # only rovers equipped for soil are useful
                    if r not in self.equipped_for_soil:
                        continue

                    # if rover already has soil analysis for wp
                    if (r, wp) in have_soil:
                        d_to_uplink = self._nearest_uplink_distance(r, cur_wp, uplinks)
                        if d_to_uplink >= INF_COST:
                            continue
                        cost = d_to_uplink + 1  # communicate
                        if cost < best_for_goal:
                            best_for_goal = cost
                        continue

                    # else must sample: go to wp, possibly drop store, sample, go to uplink, communicate
                    d1 = self._bfs_distance(r, cur_wp, wp)
                    if d1 >= INF_COST:
                        continue
                    store, is_empty = rover_store_status(r)
                    drop_cost = 0
                    if store is None:
                        # no store associated -> cannot sample
                        continue
                    if not is_empty:
                        # if full -> must drop once
                        drop_cost = 1
                    sample_cost = 1
                    d2 = self._nearest_uplink_distance(r, wp, uplinks)
                    if d2 >= INF_COST:
                        continue
                    comm_cost = 1
                    cost = d1 + drop_cost + sample_cost + d2 + comm_cost
                    if cost < best_for_goal:
                        best_for_goal = cost

            elif pred == "communicated_rock_data":
                # goal: (communicated_rock_data waypoint)
                _, wp = parts
                for r in candidate_rovers:
                    if r not in rover_pos:
                        continue
                    cur_wp = rover_pos[r]
                    if r not in self.equipped_for_rock:
                        continue

                    if (r, wp) in have_rock:
                        d_to_uplink = self._nearest_uplink_distance(r, cur_wp, uplinks)
                        if d_to_uplink >= INF_COST:
                            continue
                        cost = d_to_uplink + 1
                        if cost < best_for_goal:
                            best_for_goal = cost
                        continue

                    d1 = self._bfs_distance(r, cur_wp, wp)
                    if d1 >= INF_COST:
                        continue
                    store, is_empty = rover_store_status(r)
                    if store is None:
                        continue
                    drop_cost = 0
                    if not is_empty:
                        drop_cost = 1
                    sample_cost = 1
                    d2 = self._nearest_uplink_distance(r, wp, uplinks)
                    if d2 >= INF_COST:
                        continue
                    comm_cost = 1
                    cost = d1 + drop_cost + sample_cost + d2 + comm_cost
                    if cost < best_for_goal:
                        best_for_goal = cost

            elif pred == "communicated_image_data":
                # goal: (communicated_image_data objective mode)
                _, obj, mode = parts
                # possible image waypoints
                image_wps = self.visible_from.get(obj, set())
                if not image_wps:
                    # no static information about where the objective is visible - skip
                    best_for_goal = min(best_for_goal, INF_COST)
                for r in candidate_rovers:
                    if r not in rover_pos:
                        continue
                    cur_wp = rover_pos[r]
                    # rover must be capable of imaging and/or have a camera on board
                    if r not in self.equipped_for_imaging:
                        # maybe cameras on board belong to this rover but not equipped flag
                        # still check cameras on board
                        pass

                    # gather cameras on this rover supporting the required mode
                    cams = [
                        cam for cam, rv in self.camera_on_board.items() if rv == r and mode in self.camera_supports.get(cam, set())
                    ]
                    if not cams:
                        # no suitable camera on this rover
                        continue

                    # if rover already has the requested image just need to communicate
                    # check any rover-specific have_image fact
                    if any((r, obj, mode) == triple for triple in have_image):
                        d_to_uplink = self._nearest_uplink_distance(r, cur_wp, uplinks)
                        if d_to_uplink >= INF_COST:
                            continue
                        cost = d_to_uplink + 1
                        if cost < best_for_goal:
                            best_for_goal = cost
                        continue

                    # else compute minimal sequence: (optional calibrate) -> take_image -> communicate
                    for cam in cams:
                        # calibration target objective for this camera (if any)
                        calib_obj = self.calibration_target.get(cam, None)
                        calib_wps = self.visible_from.get(calib_obj, set()) if calib_obj else set()
                        # If camera already calibrated for this rover in this state, we can skip calibrate
                        is_calibrated = (cam, r) in calibrated

                        # If no image waypoints known, skip camera
                        if not image_wps:
                            continue

                        # Consider combinations of calibration waypoint w and image waypoint p.
                        # If already calibrated, treat w = None.
                        candidate_pairs = []
                        if is_calibrated:
                            # only need to go to an image viewpoint p
                            for p in image_wps:
                                candidate_pairs.append((None, p))
                        else:
                            # need to pick w in calib_wps and p in image_wps
                            # allow w==p if some waypoint sees both the calibration target and the objective
                            if calib_wps:
                                for w in calib_wps:
                                    for p in image_wps:
                                        candidate_pairs.append((w, p))
                            else:
                                # No calibration waypoints known for this camera; treat as impossible for this camera
                                continue

                        for w, p in candidate_pairs:
                            # compute distance sequence
                            if w is None:
                                # already calibrated: go cur -> p
                                d1 = self._bfs_distance(r, cur_wp, p)
                                if d1 >= INF_COST:
                                    continue
                                calib_cost = 0
                                d2 = 0
                            else:
                                # need cur->w, calibrate, w->p
                                d1 = self._bfs_distance(r, cur_wp, w)
                                if d1 >= INF_COST:
                                    continue
                                calib_cost = 1  # calibrate action
                                d2 = self._bfs_distance(r, w, p)
                                if d2 >= INF_COST:
                                    continue
                            take_cost = 1  # take_image
                            d3 = self._nearest_uplink_distance(r, p, uplinks)
                            if d3 >= INF_COST:
                                continue
                            comm_cost = 1
                            cost = d1 + calib_cost + d2 + take_cost + d3 + comm_cost
                            if cost < best_for_goal:
                                best_for_goal = cost

            else:
                # For any other goal predicate we do a simple fallback: if fact not in state,
                # assign a small penalty (1) to push planner to satisfy it; this domain's goals
                # are only communication predicates in the provided problems, so this is safe.
                best_for_goal = min(best_for_goal, 1)

            # If we couldn't find any feasible rover/camera combination for this goal,
            # penalize heavily but keep finite.
            if best_for_goal >= INF_COST:
                best_for_goal = 200  # large finite penalty signaling difficulty/unreachable

            total_cost += best_for_goal

        # Ensure finite return value
        return int(min(total_cost, INF_COST))