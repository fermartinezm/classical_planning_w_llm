from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into its tokens."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether a fact (string) matches a pattern of tokens (wildcards '*' allowed).
    Example: match('(at rover1 wp1)', 'at', 'rover*', '*') -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (rover/camera-like) domain.
    The heuristic estimates the (approximate) number of actions required to
    achieve all goal facts from the current state by reasoning about two main
    goal patterns that appear in the domain files provided:
      - (lk059nof <objective> <resolution>) : image-delivery goals
      - (chfbweyv <waypoint>)                : waypoint-processing goals

    Assumptions
    -----------
    - Goals are delivered by operators that require a rover (or a rover+camera)
      to be at specific waypoints and then perform 1-2 domain actions (e.g.,
      take-image, process-sample).
    - Cameras have fixed capabilities (resolutions) encoded in static facts.
    - Cameras are statically assignable to rovers (uss6vmx6 facts).
    - Rover movement options are encoded by xuwncpn8 facts (directed edges).
    - Some preconditions (e.g., camera-mount existence v25mqvwj) may already
      hold in the state; otherwise a single mounting action is assumed.
    - The heuristic is not required to be admissible; it may overcount and it
      sums estimated costs for independent goals.
    - If multiple rovers or cameras could achieve a goal, we pick the least
      costly option (min over rovers/cameras).

    Heuristic Initialization
    ------------------------
    In the constructor we extract relevant static information into structures:
      - self.adjacency[rover][from_wp] -> list of to_wp   (from xuwncpn8)
      - self.obj_waypoints[objective] -> set(waypoints)   (from wecz86om)
      - self.cam_objectives[camera] -> set(objectives)    (from g52pa0bi)
      - self.cam_resolutions[camera] -> set(resolutions)  (from xq9py0w8)
      - self.camera_to_rover[camera] -> set(rovers)       (from uss6vmx6)
      - self.jy_rovers -> set(rovers with jy67mgmv)
      - self.dp3_rovers -> set(rovers with dp3yzgup)
      - self.czm_waypoints -> set(waypoints with czmzagzw)
    We also store the goal facts (task.goals) to check goal satisfaction quickly.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given node/state the heuristic proceeds as follows:

    1. Return 0 immediately if all goals are present in the state.

    2. Parse dynamic facts of the current state to obtain:
       - rover_positions: mapping rover -> current waypoint (from ilkg22gp facts).
       - mounted_pairs: set of tuples (camera, rover) for which v25mqvwj holds.

    3. For each unsatisfied goal fact in the goal set:
       - If the goal is of the form (lk059nof <obj> <res>):
         * Intuition: producing this fact typically needs (a) a rover with a
           camera capable of <res> to be at one of the objective's waypoints,
           (b) possibly a camera-mount action (jx1xpj7f), (c) the image-taking
           action (tkek73dx), and (d) the postprocessing action (bypa1zm7).
         * Compute candidates:
           - For every camera that supports <res>, and for every rover
             that the camera can attach to (camera->rover static mapping)
             and that has the rover-capability flag (jy67mgmv), compute:
               candidate_cost = move_dist(rover_pos, any objective waypoint)
                                + mount_cost (0 if mounted already else 1)
                                + 1 (take-image action)
                                + 1 (finalize action)
             where move_dist is the BFS shortest path length on xuwncpn8 edges
             specific to that rover. If no route exists or there is missing
             static info, a conservative finite penalty is used.
           - Take the minimum candidate cost as the estimate for this goal.
       - If the goal is of the form (chfbweyv <wp>):
         * Intuition: this kind of goal is typically produced by first running
           a store/collect action (cp8otkvs or wyhd5xzi) at some waypoint
           (requires czmzagzw/hfrxxy8q) and then a processing action that
           requires the rover to be at a related waypoint. We approximate the
           required steps as "move + collect + move + process" but try to
           choose the rover that minimizes travel.
         * Compute candidates:
           - For each rover that looks able to run the collect action
             (e.g., present in dp3yzgup or having other static flags), compute
               candidate_cost = move_dist(rover_pos, collect_wp)
                                + 1 (collect action)
                                + move_dist(collect_wp, process_wp)  (often 0)
                                + 1 (process action)
             For efficiency and robustness we choose collect_wp from the
             set of waypoints with czmzagzw (static). If no czm point exists,
             fallback to a small finite penalty.
           - Take the minimum candidate cost among rovers as the estimate.

    4. Sum the per-goal estimates. This sum is returned as the heuristic value.
       Because we sum per-goal estimates we may overcount shared actions; this
       is acceptable because the objective is to reduce search effort (not to
       be admissible).

    Practical notes on implementation:
    - BFS for path length uses the per-rover xuwncpn8 edges extracted from
      static facts. If no path can be found we return a bounded penalty so the
      heuristic remains finite.
    - The heuristic relies primarily on static connectivity + current rover
      positions and the presence/absence of mounts, keeping the computation
      efficient (simple graph searches and lookups).
    """

    def __init__(self, task):
        # store goal facts set (strings)
        self.goals = task.goals

        static_facts = task.static

        # adjacency per rover: xuwncpn8 rover from to
        self.adjacency = {}
        for fact in static_facts:
            if match(fact, "xuwncpn8", "*", "*", "*"):
                _, rover, frm, to = get_parts(fact)
                self.adjacency.setdefault(rover, {}).setdefault(frm, []).append(to)

        # objective -> waypoints (wecz86om objective waypoint)
        self.obj_waypoints = {}
        for fact in static_facts:
            if match(fact, "wecz86om", "*", "*"):
                _, obj, wp = get_parts(fact)
                self.obj_waypoints.setdefault(obj, set()).add(wp)

        # camera -> objectives (g52pa0bi camera objective)
        self.cam_objectives = {}
        for fact in static_facts:
            if match(fact, "g52pa0bi", "*", "*"):
                _, cam, obj = get_parts(fact)
                self.cam_objectives.setdefault(cam, set()).add(obj)

        # camera -> resolutions (xq9py0w8 camera resolution)
        self.cam_resolutions = {}
        for fact in static_facts:
            if match(fact, "xq9py0w8", "*", "*"):
                _, cam, res = get_parts(fact)
                self.cam_resolutions.setdefault(cam, set()).add(res)

        # camera -> rovers (uss6vmx6 camera rover)
        self.camera_to_rover = {}
        for fact in static_facts:
            if match(fact, "uss6vmx6", "*", "*"):
                _, cam, rover = get_parts(fact)
                self.camera_to_rover.setdefault(cam, set()).add(rover)

        # rovers that have jy67mgmv (capability flag)
        self.jy_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "jy67mgmv", "*")}

        # rovers that have dp3yzgup (used for certain collect actions)
        self.dp3_rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "dp3yzgup", "*")}

        # czmzagzw waypoints (places where cp8otkvs can operate)
        self.czm_waypoints = {get_parts(fact)[1] for fact in static_facts if match(fact, "czmzagzw", "*")}

        # hfrxxy8q waypoints (alternative collect spots)
        self.hfrx_waypoints = {get_parts(fact)[1] for fact in static_facts if match(fact, "hfrxxy8q", "*")}

        # Precompute small limit for unreachable distances
        # Choose a reasonable upper bound (depends on domain size). Use 50 to be safe.
        self.MAX_DISTANCE_PENALTY = 50

    def bfs_distance(self, rover, start, goal):
        """
        Compute shortest path length from start to goal using the rover-specific
        adjacency edges. Returns an integer >= 0. If no path found, returns
        a bounded penalty to keep heuristic finite.
        """
        if start == goal:
            return 0
        # If we have no adjacency info for this rover, return penalty
        if rover not in self.adjacency:
            return min(self.MAX_DISTANCE_PENALTY, 5)
        adj = self.adjacency[rover]
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nb in adj.get(node, []):
                if nb == goal:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        # unreachable: return bounded penalty
        return self.MAX_DISTANCE_PENALTY

    def __call__(self, node):
        state = node.state

        # If goal already reached -> heuristic 0
        if self.goals <= state:
            return 0

        # Current rover positions from ilkg22gp rover waypoint
        rover_pos = {}
        for fact in state:
            if match(fact, "ilkg22gp", "*", "*"):
                _, rover, wp = get_parts(fact)
                # Keep the last seen position if multiple exist (shouldn't happen)
                rover_pos[rover] = wp

        # Mounted camera-rover pairs present in state (v25mqvwj camera rover)
        mounted = {tuple(get_parts(fact)[1:3]) for fact in state if match(fact, "v25mqvwj", "*", "*")}

        total_cost = 0

        # Helper to estimate cost for lk059nof goals (image deliver)
        def estimate_lk059nof(obj, res):
            candidates = []

            # candidate rovers/cameras that can satisfy resolution
            for cam, resolutions in self.cam_resolutions.items():
                if res not in resolutions:
                    continue
                # camera must be capable of objective too (not strictly necessary but reasonable)
                # find objectives this camera can serve; if camera not linked to obj via g52pa0bi, still allow it
                # since static data sometimes lists which object camera is useful for.
                # We'll allow camera even if not explicitly mapped, but prefer mapped ones.
                cam_objs = self.cam_objectives.get(cam, set())
                prefers_obj = (obj in cam_objs)
                rovers_for_cam = self.camera_to_rover.get(cam, set())
                for rover in rovers_for_cam:
                    if rover not in self.jy_rovers:
                        # rover lacks required capability flag in many preconditions
                        continue
                    # compute minimal distance from rover's current pos to any waypoint for this objective
                    waypoints = self.obj_waypoints.get(obj, None)
                    if not waypoints:
                        # No static waypoint for objective -> assume small fixed cost
                        move_cost = 2
                    else:
                        # If rover has a known position, compute min distance; else penalize mildly
                        if rover in rover_pos:
                            start = rover_pos[rover]
                            move_cost = min(self.bfs_distance(rover, start, wp) for wp in waypoints)
                        else:
                            move_cost = 5
                    # mount cost: if (cam, rover) already mounted then 0 else 1
                    mount_cost = 0 if (cam, rover) in mounted else 1
                    # 1 for take-image (tkek73dx) and 1 for finalization (bypa1zm7)
                    base = 2
                    score = move_cost + mount_cost + base
                    # prefer candidates that are explicitly mapped to objective
                    if prefers_obj:
                        score -= 1  # small preference
                    candidates.append(score)

            if candidates:
                return int(min(candidates))
            # fallback: no camera with required resolution found; return conservative estimate
            return 6

        # Helper to estimate cost for chfbweyv waypoint goals
        def estimate_chfbweyv(waypoint):
            candidates = []
            # consider rovers that appear capable (dp3 flag preferred, else any jy rover)
            candidate_rovers = set(self.dp3_rovers) | set(self.jy_rovers) | set(rover_pos.keys())
            # use collect waypoints as czm or hfrx if present; otherwise use target waypoint directly
            collect_points = set(self.czm_waypoints) or set(self.hfrx_waypoints) or {waypoint}
            for rover in candidate_rovers:
                # if rover position known, compute move cost to a collect point
                if rover in rover_pos:
                    start = rover_pos[rover]
                    # pick best collect point minimizing start->collect + collect->process distances
                    best_local = None
                    for collect_wp in collect_points:
                        d1 = self.bfs_distance(rover, start, collect_wp)
                        # assume processing happens at waypoint or close; estimate d2 as distance from collect_wp to waypoint
                        # We can't use adjacency between different rovers, so use rover's adjacency if available
                        d2 = self.bfs_distance(rover, collect_wp, waypoint)
                        candidate_cost = d1 + 1 + d2 + 1  # move to collect + collect + move + process
                        if best_local is None or candidate_cost < best_local:
                            best_local = candidate_cost
                    if best_local is not None:
                        candidates.append(best_local)
                else:
                    # unknown rover position -> moderate penalty
                    candidates.append(6)
            if candidates:
                return int(min(candidates))
            # fallback conservative cost
            return 6

        # Process each goal in the task
        for goal in self.goals:
            if goal in state:
                continue
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "lk059nof":
                # goal form: (lk059nof objective resolution)
                try:
                    _, objective, resolution = parts
                except ValueError:
                    # malformed goal -> small penalty
                    total_cost += 5
                    continue
                total_cost += estimate_lk059nof(objective, resolution)
            elif pred == "chfbweyv":
                # goal form: (chfbweyv waypoint)
                try:
                    _, wp = parts
                except ValueError:
                    total_cost += 3
                    continue
                total_cost += estimate_chfbweyv(wp)
            else:
                # Generic fallback: estimate 2 actions per unsatisfied goal
                total_cost += 2

        # Ensure heuristic is integer and finite
        if total_cost is None:
            return 0
        return int(total_cost)