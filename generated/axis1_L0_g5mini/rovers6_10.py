from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "rovers" domain. The heuristic estimates the
    number of actions still required to satisfy all goal facts (communications of soil,
    rock and image data). It reasons about the required sensing (sampling/calibration/take
    image), movements (navigate actions between waypoints), and the communications that
    must be performed at a waypoint that can see a lander. The heuristic is not required
    to be admissible; its goal is to produce informative estimates efficiently to guide
    greedy best-first search.

    Assumptions
    -----------
    - Each navigation between adjacent waypoints costs 1 (one navigate action).
    - Sampling (sample_soil/sample_rock), calibrate, take_image, communicate_*, and drop
      are each counted as cost 1 per action used by the planner.
    - After a take_image action the camera becomes uncalibrated (as in the domain).
    - Stores are single-capacity. A sample requires the associated store to be empty;
      if it is not empty in the current state, we count one "drop" action to empty it.
    - The heuristic treats tasks independently (it sums minimal per-goal estimated costs),
      so it does not try to optimally interleave actions across goals; this keeps the
      heuristic efficient while still producing useful guidance.
    - If a goal is already satisfied in the state, it contributes 0 cost.

    Heuristic Initialization
    ------------------------
    In the constructor we extract static information to accelerate computations in calls:
    - Graph of traversability per rover (directed edges from waypoint -> waypoint).
    - visible_from mapping: for each objective the waypoints from which it is visible.
    - comm_waypoints: waypoints from which some lander is visible (these are candidate
      positions from which communication actions can be executed).
    - on_board: map camera -> rover (static).
    - supports: map camera -> set of supported modes.
    - calibration_target: map camera -> objective it can calibrate for.
    - store_of: map store -> rover, and reverse mapping rover -> its store.
    - equipped_for_* sets (imaging/soil/rock) available to rovers.
    - The set of goal facts is stored to quickly detect already satisfied goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goals are already satisfied in the given state, return 0.
    2. Parse dynamic state facts to extract:
       - Rover positions: map rover -> current waypoint.
       - Store states (empty/full).
       - Which rovers already have soil/rock analyses for which waypoints.
       - Which rovers already have images for (objective, mode).
       - Which cameras are currently calibrated on which rover.
    3. Precompute BFS distances between waypoints for each rover using the rover's
       traversability graph (from static can_traverse facts). Distances are integer
       counts of navigate actions. If some waypoint is unreachable by a rover, treat the
       distance as a large penalty (but finite).
    4. For each remaining goal fact (one of communicated_soil_data, communicated_rock_data,
       communicated_image_data), compute the minimal estimated cost across all rovers
       that could possibly achieve it.
       - communicated_soil_data at_wp:
         * If some rover already has have_soil_analysis for that waypoint, estimate:
           distance(rover_pos -> some_comm_wp) + 1 (communicate).
         * Otherwise, for each rover equipped_for_soil_analysis: estimate:
           distance(rover_pos -> sample_wp) + (1 if its store is not empty then +1 drop)
           + 1 (sample) + distance(sample_wp -> comm_wp) + 1 (communicate).
       - communicated_rock_data is analogous, using equipped_for_rock_analysis and have_rock_analysis.
       - communicated_image_data objective mode:
         * If some rover already has have_image for (objective,mode), estimate:
           distance(rover_pos -> comm_wp) + 1 (communicate).
         * Otherwise, for each candidate camera (static: on_board camera rover, camera supports mode,
           calibration_target camera objective) and rover equipped_for_imaging: estimate
           distance(rover_pos -> image_wp) + calibrate_cost + 1 (take_image)
           + distance(image_wp -> comm_wp) + 1 (communicate).
           calibrate_cost is 0 if that camera is already calibrated on that rover in state,
           else 1 (calibrate). Note that after take_image the camera will be uncalibrated,
           so calibrate is required for each image separately (this heuristic respects that).
    5. Sum the minimal estimated cost contributions for all unsatisfied goals to produce
       the final heuristic value.
    """

    def __init__(self, task):
        # Save goal facts (set of strings).
        self.goals = task.goals
        static_facts = task.static

        # Traversability graph per rover: rover -> {wp: set(neighbor_wps)}
        self.traversal = {}
        # visible_from: objective -> set(waypoints)
        self.visible_from = {}
        # comm_waypoints: waypoints that can see a lander (candidates for communications)
        self.comm_waypoints = set()
        # camera -> rover (on_board)
        self.on_board = {}
        # camera -> set(modes)
        self.supports = {}
        # camera -> calibration_target objective
        self.calibration_target = {}
        # store -> rover and rover -> store
        self.store_of = {}
        self.rover_store = {}
        # equipped_for sets
        self.equipped_imaging = set()
        self.equipped_soil = set()
        self.equipped_rock = set()

        # collect lander waypoints
        lander_waypoints = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "can_traverse":
                rover, a, b = parts[1], parts[2], parts[3]
                if rover not in self.traversal:
                    self.traversal[rover] = {}
                self.traversal[rover].setdefault(a, set()).add(b)
            elif pred == "visible_from":
                obj, wp = parts[1], parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)
            elif pred == "visible":
                a, b = parts[1], parts[2]
                # We treat visible facts generically; comm_waypoints are determined later
                # when we know lander waypoints.
                # For now we store these visible relations in a helper structure by string.
                # But to be efficient we will reuse static_facts scanning later to add to comm_waypoints.
                pass
            elif pred == "at_lander":
                # (at_lander lander waypoint)
                _, wp = parts[1], parts[2]
                lander_waypoints.add(wp)
            elif pred == "on_board":
                camera, rover = parts[1], parts[2]
                self.on_board[camera] = rover
            elif pred == "supports":
                camera, mode = parts[1], parts[2]
                self.supports.setdefault(camera, set()).add(mode)
            elif pred == "calibration_target":
                camera, obj = parts[1], parts[2]
                self.calibration_target[camera] = obj
            elif pred == "store_of":
                store, rover = parts[1], parts[2]
                self.store_of[store] = rover
                self.rover_store[rover] = store
            elif pred == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
            elif pred == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])

        # Build comm_waypoints by scanning visible facts in static_facts:
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "visible":
                a, b = parts[1], parts[2]
                if b in lander_waypoints:
                    self.comm_waypoints.add(a)

        # For any rover that has no explicit adjacency in traversal, ensure an empty dict
        # so __call__ can iterate consistently.
        # Also collect known rover names from traversal and on_board and equipped sets.
        known_rovers = set(self.traversal.keys()) | set(self.on_board.values()) | self.equipped_imaging | self.equipped_soil | self.equipped_rock | set(self.rover_store.keys())
        for r in known_rovers:
            self.traversal.setdefault(r, {})

        # Precompute list of cameras that are valid for imaging (calibration_target present and supports present and on_board present)
        self.cameras = set(self.on_board.keys())

        # A large but finite penalty for unreachable waypoint combinations.
        self._INF = 1000

    def _bfs_distances(self, rover, start):
        """
        Compute shortest distances (in number of navigations) from start waypoint to all reachable waypoints
        according to this rover's can_traverse graph. Returns dict wp -> distance.
        """
        # If the rover has no traversal info, return empty mapping (unreachable).
        graph = self.traversal.get(rover, {})
        dist = {}
        if start is None:
            return dist
        q = deque()
        dist[start] = 0
        q.append(start)
        while q:
            node = q.popleft()
            for nbr in graph.get(node, ()):
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    q.append(nbr)
        return dist

    def __call__(self, node):
        state = node.state

        # If all goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract dynamic state info.
        rover_pos = {}  # rover -> waypoint
        store_empty = {}  # store -> True/False (True if empty)
        have_soil = set()  # tuples (rover, waypoint)
        have_rock = set()
        have_image = set()  # tuples (rover, objective, mode)
        calibrated = set()  # tuples (camera, rover)

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                # (at rover waypoint)
                # We only consider rovers here; other at facts exist (e.g., objects) but domain uses at for rovers.
                obj, wp = parts[1], parts[2]
                # If this object is a rover we encountered in traversal or known rovers, record it.
                # Otherwise we don't need it.
                if obj in self.traversal:
                    rover_pos[obj] = wp
            elif pred == "empty":
                store_empty[parts[1]] = True
            elif pred == "full":
                store_empty[parts[1]] = False
            elif pred == "have_soil_analysis":
                have_soil.add((parts[1], parts[2]))  # (rover, waypoint)
            elif pred == "have_rock_analysis":
                have_rock.add((parts[1], parts[2]))
            elif pred == "have_image":
                have_image.add((parts[1], parts[2], parts[3]))  # (rover, objective, mode)
            elif pred == "calibrated":
                calibrated.add((parts[1], parts[2]))  # (camera, rover)

        # Helper to test whether a particular communicated goal is already in the state.
        def goal_satisfied(goal_fact):
            return goal_fact in state

        total_cost = 0

        # Precompute BFS distances from each rover's current position.
        rover_distances = {}
        for rover in self.traversal:
            start = rover_pos.get(rover, None)
            rover_distances[rover] = self._bfs_distances(rover, start)

        # Helper to get distance (moves) between two waypoints for a rover.
        def distance(rover, a, b):
            if a is None or b is None:
                return self._INF
            # If a == b no move needed.
            if a == b:
                return 0
            # If we have BFS from a? Our precomputed BFS is from rover's start only.
            # For moves between arbitrary waypoints (e.g., sample_wp -> comm_wp) we do a BFS from sample_wp.
            # To keep it efficient we will run BFS from a when needed (cache per rover and start).
            # Implement a small cache:
            if ("_bfs_cache" not in rover_distances[rover]) or (rover_distances[rover].get("_bfs_cache_start") != a):
                # Run BFS from a and store under special key
                graph = self.traversal.get(rover, {})
                dist = {}
                q = deque()
                dist[a] = 0
                q.append(a)
                while q:
                    node = q.popleft()
                    for nbr in graph.get(node, ()):
                        if nbr not in dist:
                            dist[nbr] = dist[node] + 1
                            q.append(nbr)
                rover_distances[rover]["_bfs_cache"] = dist
                rover_distances[rover]["_bfs_cache_start"] = a
            else:
                dist = rover_distances[rover]["_bfs_cache"]
            return dist.get(b, self._INF)

        # Evaluate each goal individually and sum minimal estimated costs.
        for goal in self.goals:
            if goal_satisfied(goal):
                continue  # contributes 0

            parts = get_parts(goal)
            pred = parts[0]

            if pred == "communicated_soil_data":
                wp = parts[1]  # waypoint whose soil data must be communicated
                best = self._INF

                # Option: some rover may already have the analysis -> only communicate needed.
                for (r, p) in list(have_soil):
                    if p == wp:
                        # needs to move to any comm waypoint and communicate
                        start = rover_pos.get(r, None)
                        if start is None:
                            continue
                        # choose best comm waypoint (min distance)
                        d_comm = min(
                            (rover_distances[r].get(cwp, self._INF) for cwp in self.comm_waypoints),
                            default=self._INF,
                        )
                        # If the rover is not at start BFS from start was computed earlier
                        cost = d_comm + 1  # communicate
                        best = min(best, cost)

                # Option: a rover must sample and then communicate.
                for r in self.traversal:
                    if r not in self.equipped_soil:
                        continue
                    start = rover_pos.get(r, None)
                    if start is None:
                        continue
                    # distance to sample location wp
                    d_to_sample = rover_distances[r].get(wp, self._INF)
                    # optionally drop if its store is not empty
                    store = self.rover_store.get(r, None)
                    drop_cost = 0
                    if store is not None:
                        # if we don't know its state, assume not empty? Prefer to treat unknown as empty.
                        if store in store_empty:
                            if not store_empty[store]:
                                drop_cost = 1
                        else:
                            # If no info, assume empty to avoid over-penalizing (heuristic can be optimistic)
                            drop_cost = 0
                    # sample cost
                    sample_cost = 1
                    # after sampling at wp, move to nearest comm waypoint and communicate
                    best_comm_move = self._INF
                    for cwp in self.comm_waypoints:
                        d_sample_to_comm = distance(r, wp, cwp)
                        if d_sample_to_comm < best_comm_move:
                            best_comm_move = d_sample_to_comm
                    if d_to_sample >= self._INF or best_comm_move >= self._INF:
                        continue  # unreachable by this rover
                    cost = d_to_sample + drop_cost + sample_cost + best_comm_move + 1  # communicate
                    best = min(best, cost)

                total_cost += min(best, self._INF)

            elif pred == "communicated_rock_data":
                wp = parts[1]
                best = self._INF

                for (r, p) in list(have_rock):
                    if p == wp:
                        start = rover_pos.get(r, None)
                        if start is None:
                            continue
                        d_comm = min(
                            (rover_distances[r].get(cwp, self._INF) for cwp in self.comm_waypoints),
                            default=self._INF,
                        )
                        cost = d_comm + 1
                        best = min(best, cost)

                for r in self.traversal:
                    if r not in self.equipped_rock:
                        continue
                    start = rover_pos.get(r, None)
                    if start is None:
                        continue
                    d_to_sample = rover_distances[r].get(wp, self._INF)
                    store = self.rover_store.get(r, None)
                    drop_cost = 0
                    if store is not None:
                        if store in store_empty:
                            if not store_empty[store]:
                                drop_cost = 1
                    sample_cost = 1
                    best_comm_move = self._INF
                    for cwp in self.comm_waypoints:
                        d_sample_to_comm = distance(r, wp, cwp)
                        if d_sample_to_comm < best_comm_move:
                            best_comm_move = d_sample_to_comm
                    if d_to_sample >= self._INF or best_comm_move >= self._INF:
                        continue
                    cost = d_to_sample + drop_cost + sample_cost + best_comm_move + 1
                    best = min(best, cost)

                total_cost += min(best, self._INF)

            elif pred == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                best = self._INF

                # If some rover already has the image (have_image), only communication needed.
                for (r, o, m) in list(have_image):
                    if o == obj and m == mode:
                        start = rover_pos.get(r, None)
                        if start is None:
                            continue
                        d_comm = min(
                            (rover_distances[r].get(cwp, self._INF) for cwp in self.comm_waypoints),
                            default=self._INF,
                        )
                        cost = d_comm + 1
                        best = min(best, cost)

                # Otherwise, for each camera that can take this image for this objective and mode:
                for camera in self.cameras:
                    # camera must support the requested mode
                    if mode not in self.supports.get(camera, ()):
                        continue
                    # camera must have calibration_target objective equal to obj
                    if self.calibration_target.get(camera, None) != obj:
                        continue
                    # camera must be on some rover
                    rover = self.on_board.get(camera, None)
                    if rover is None:
                        continue
                    # rover must be equipped for imaging
                    if rover not in self.equipped_imaging:
                        continue
                    start = rover_pos.get(rover, None)
                    if start is None:
                        continue
                    # consider all image waypoints (visible_from objective)
                    image_wps = self.visible_from.get(obj, set())
                    if not image_wps:
                        continue
                    # For each image waypoint compute:
                    for img_wp in image_wps:
                        # move to image waypoint
                        d_to_image = rover_distances[rover].get(img_wp, self._INF)
                        if d_to_image >= self._INF:
                            continue
                        # calibrate cost: 0 if this camera is already calibrated on this rover, else 1
                        calibrate_cost = 0 if (camera, rover) in calibrated else 1
                        # take_image cost = 1
                        take_cost = 1
                        # after taking image, need to move to comm waypoint and communicate
                        best_comm_move = self._INF
                        for cwp in self.comm_waypoints:
                            d_img_to_comm = distance(rover, img_wp, cwp)
                            if d_img_to_comm < best_comm_move:
                                best_comm_move = d_img_to_comm
                        if best_comm_move >= self._INF:
                            continue
                        cost = d_to_image + calibrate_cost + take_cost + best_comm_move + 1
                        best = min(best, cost)

                total_cost += min(best, self._INF)

            else:
                # Unknown goal predicate: fall back to cost 1 (safe finite estimate).
                total_cost += 1

        return int(total_cost)