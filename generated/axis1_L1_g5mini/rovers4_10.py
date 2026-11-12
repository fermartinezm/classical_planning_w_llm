from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact (string) matches a pattern of tokens (wildcards '*' not supported
    here because facts are matched exactly by token positions in this heuristic).
    """
    parts = get_parts(fact)
    if len(parts) != len(pattern):
        return False
    return all(pat == "*" or pat == part for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the obfuscated "secret_domain" (a rover-like domain).
    The domain's goals typically require either:
      - acquiring images of objectives in certain image formats (predicates lk059nof objective type),
      - producing some waypoint-specific data predicate (e.g. chfbweyv waypoint).
    This heuristic estimates the number of actions required to reach the goal from a
    given state by reasoning about:
      - rover current positions,
      - waypoint connectivity (shortest path in the waypoint graph),
      - camera->objective compatibility and camera->rover relationships,
      - whether cameras are already "mounted/ready" (v25mqvwj facts),
      - whether intermediate image facts (l3943stc) or sensor-output facts (i0yg0wob / s33vuot5) are
        already present in the state and can reduce required actions.

    Assumptions
    -----------
    - Actions that produce images follow a three-step pattern:
        1) (optionally) mount/prepare the camera for the rover (v25mqvwj),
        2) perform the imaging action that produces an intermediate fact (l3943stc),
        3) a finalize action converts the intermediate fact into the goal fact (lk059nof).
      Each of steps (1)-(3) corresponds to approximately one operator in the domain.
    - Movement between waypoints is accomplished by repeated applications of the single-step
      move operator (azu4u5xz). We approximate travel cost by the shortest-path (edge count)
      in the static adjacency graph (kxvuge0a facts).
    - Many static relationships (camera supports types, camera-objective relationships,
      objective->waypoint mapping) are present in the task static facts and are used
      to identify feasible rover+camera candidates for imaging an objective.
    - For producing waypoint data facts (e.g., chfbweyv), a short sequence involving
      (1) moving a rover into position, (2) an operation that produces an intermediate
      local fact (i0yg0wob or s33vuot5), and (3) a finalize action is typical.
      We estimate such goals with a small fixed number of actions + travel cost.
    - The heuristic is not admissible; it is meant to be informative and cheap to compute.

    Heuristic Initialization
    ------------------------
    The constructor extracts static information into convenient structures:
      - adjacency: waypoint -> set(neighbour_waypoints), from kxvuge0a facts
      - objective_to_waypoint: mapping from objective -> waypoint (wecz86om)
      - camera_to_objectives: mapping camera -> set(objectives) (g52pa0bi)
      - camera_types: mapping camera -> set(types) (xq9py0w8)
      - camera_to_rovers: mapping camera -> set(rovers) (uss6vmx6)
      - waypoint_static_flags: sets of waypoints that have czmzagzw or hfrxxy8q,
        which inform whether cp8otkvs / wyhd5xzi-like actions are usable.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node (with node.state as a frozenset of fact strings):

    1) If all goals already hold in the state (task.goal set is subset of state), return 0.

    2) Parse the state to extract:
         - which lk059nof goals are already true,
         - which intermediate l3943stc imaging facts are present,
         - which cameras are already mounted (v25mqvwj),
         - current rover positions (ilkg22gp),
         - other intermediate facts like i0yg0wob / s33vuot5 (possible precursors).

    3) For each goal fact in the task.goals:
       - If the goal is already present in state -> cost 0.
       - If goal predicate is 'lk059nof' (objective + type):
         a) If an intermediate l3943stc for that objective & type exists -> cost 1 (finalize).
         b) Otherwise, consider all static camera candidates that are associated with the objective
            (camera -> objective mapping) and that support the required type.
            For each camera candidate and for each rover the camera can be with (camera->rover),
            compute:
              - travel_cost = shortest path (edge count) from that rover's current waypoint
                              to the objective's waypoint (if rover position unknown, assume small default).
              - mount_cost = 0 if camera is already mounted for that rover in the current state;
                             otherwise 1 (to represent jx1xpj7f-like operation).
              - imaging_cost = 1 (tkek73dx-like)
              - finalize_cost = 1 (bypa1zm7-like)
            Candidate cost = travel_cost + mount_cost + imaging_cost + finalize_cost.
            Select the minimum candidate cost for the goal. If no candidate exists, use a conservative
            fallback cost (small finite number).
       - If goal predicate is 'chfbweyv' (waypoint data):
         a) If it is already present -> 0.
         b) If an intermediate i0yg0wob for that waypoint exists in the state -> 1 (finalize action).
         c) Otherwise compute minimal travel distance from any rover to the waypoint and:
              - if the waypoint currently has czmzagzw or hfrxxy8q (either static or in state),
                assume two actions will create the intermediate fact and then finalize it:
                  cost = travel_dist + 2 (intermediate + finalize)
              - otherwise assume a slightly larger small fixed cost: travel_dist + 3.

    4) Sum the per-goal estimates to produce the heuristic value. This is a simple, fast
       additive approximation. For efficiency, BFS distances are computed on-demand and
       memoized for each source waypoint.

    Note: The heuristic tries to capture the primary contributors to cost:
          travel distance and the canonical number of domain operators needed (mount / take / finalize).
          It intentionally simplifies many domain constraints (e.g., store availability,
          dp3yzgup flags, exact parameter typing) to maintain speed and robustness.
    """

    def __init__(self, task):
        # store goals for quick access
        self.goals = set(task.goals)
        static_facts = set(task.static)

        # adjacency graph from kxvuge0a facts
        self.adjacency = {}
        for fact in static_facts:
            if match(fact, "kxvuge0a", "*", "*"):
                _, a, b = get_parts(fact)
                self.adjacency.setdefault(a, set()).add(b)
        # Make sure reversed edges not assumed unless present in static facts.

        # objective -> waypoint mapping (wecz86om)
        self.objective_to_waypoint = {}
        for fact in static_facts:
            if match(fact, "wecz86om", "*", "*"):
                _, obj, wp = get_parts(fact)
                # If multiple waypoints, keep one (the static facts usually enumerate all valid ones).
                # We'll keep the first seen; BFS will handle distance even if there are multiple valid wps.
                if obj not in self.objective_to_waypoint:
                    self.objective_to_waypoint[obj] = wp

        # camera -> objectives mapping (g52pa0bi)
        self.camera_to_objectives = {}
        for fact in static_facts:
            if match(fact, "g52pa0bi", "*", "*"):
                _, cam, obj = get_parts(fact)
                self.camera_to_objectives.setdefault(cam, set()).add(obj)

        # camera -> supported types (xq9py0w8)
        self.camera_types = {}
        for fact in static_facts:
            if match(fact, "xq9py0w8", "*", "*"):
                _, cam, typ = get_parts(fact)
                self.camera_types.setdefault(cam, set()).add(typ)

        # camera -> rover mapping (uss6vmx6)
        self.camera_to_rovers = {}
        for fact in static_facts:
            if match(fact, "uss6vmx6", "*", "*"):
                _, cam, rov = get_parts(fact)
                self.camera_to_rovers.setdefault(cam, set()).add(rov)

        # waypoint flags that indicate availability of special site actions (czmzagzw, hfrxxy8q)
        self.cz_sites = set()
        self.hf_sites = set()
        for fact in static_facts:
            if match(fact, "czmzagzw", "*"):
                _, wp = get_parts(fact)
                self.cz_sites.add(wp)
            if match(fact, "hfrxxy8q", "*"):
                _, wp = get_parts(fact)
                self.hf_sites.add(wp)

        # memoization for BFS distances (source -> dict(dest -> dist))
        self._bfs_cache = {}

    def _shortest_distance(self, src, dest):
        """Return shortest path edge-count from src to dest using static adjacency (BFS)."""
        if src == dest:
            return 0
        if src in self._bfs_cache and dest in self._bfs_cache[src]:
            return self._bfs_cache[src][dest]
        # BFS
        dist = {}
        dq = deque([src])
        dist[src] = 0
        while dq:
            u = dq.popleft()
            for v in self.adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if v == dest:
                        # cache partial results
                        self._bfs_cache.setdefault(src, {})[dest] = dist[v]
                        return dist[v]
                    dq.append(v)
        # not reachable: use a conservative finite penalty
        penalty = 5
        self._bfs_cache.setdefault(src, {})[dest] = penalty
        return penalty

    def __call__(self, node):
        state = set(node.state)

        # If all goals already hold, heuristic is zero.
        if self.goals <= state:
            return 0

        # Gather relevant facts from the state
        # current lk059nof goals already satisfied
        present_lk059nof = set()
        present_l3943stc = set()  # intermediate imaging facts: (l3943stc rover objective type)
        mounted = set()  # v25mqvwj camera rover
        rover_pos = {}  # rover -> waypoint (from ilkg22gp)
        present_i0yg0wob = set()  # i0yg0wob rover waypoint
        present_s33vuot5 = set()  # s33vuot5 rover waypoint

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "lk059nof":
                _, obj, typ = parts
                present_lk059nof.add((obj, typ))
            elif pred == "l3943stc":
                # l3943stc ?wgox9 - cdmmbhnd ?ms09y - zi8ouvk1 ?hzpb6 - kad6swsx
                _, rov, obj, typ = parts
                present_l3943stc.add((obj, typ))
            elif pred == "v25mqvwj":
                _, cam, rov = parts
                mounted.add((cam, rov))
            elif pred == "ilkg22gp":
                _, rov, wp = parts
                rover_pos[rov] = wp
            elif pred == "i0yg0wob":
                _, rov, wp = parts
                present_i0yg0wob.add((rov, wp))
            elif pred == "s33vuot5":
                _, rov, wp = parts
                present_s33vuot5.add((rov, wp))

        total_cost = 0

        # Precompute list of rovers (from state and static adjacency)
        rovers = set(rover_pos.keys())
        # also include camera_to_rovers from static as possible rovers to consider
        for cam in self.camera_to_rovers:
            rovers.update(self.camera_to_rovers[cam])

        # Helper: obtain objective's waypoint (from static info if available)
        def objective_waypoint(obj):
            return self.objective_to_waypoint.get(obj, None)

        # For each goal, compute an estimate and add to total_cost
        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]

            if pred == "lk059nof":
                _, obj, typ = parts
                # already satisfied?
                if (obj, typ) in present_lk059nof:
                    continue
                # intermediate present?
                if (obj, typ) in present_l3943stc:
                    total_cost += 1  # finalize (bypa1zm7)
                    continue

                # identify candidate (camera, rover) pairs that can image this objective in typ
                candidates = []
                # find cameras that are statically linked to this objective
                for cam, objs in self.camera_to_objectives.items():
                    if obj not in objs:
                        continue
                    # camera must support the requested type
                    cam_types = self.camera_types.get(cam, set())
                    if typ not in cam_types:
                        continue
                    # which rovers can operate this camera?
                    rovers_for_cam = self.camera_to_rovers.get(cam, set())
                    if not rovers_for_cam:
                        # no static camera->rover mapping; consider any rover as fallback
                        rovers_for_cam = rovers.copy()
                    for rov in rovers_for_cam:
                        # travel distance from rover current position to objective waypoint
                        wp = objective_waypoint(obj)
                        if wp is None:
                            travel = 1  # small fallback
                        else:
                            rov_pos = rover_pos.get(rov, None)
                            if rov_pos is None:
                                travel = 1  # assume small travel if unknown
                            else:
                                travel = self._shortest_distance(rov_pos, wp)
                        mount_needed = 0 if (cam, rov) in mounted else 1
                        # imaging + finalize costs assumed 1 each
                        cost = travel + mount_needed + 1 + 1
                        candidates.append(((cam, rov), cost, mount_needed, travel))

                if candidates:
                    # choose cheapest candidate
                    best = min(candidates, key=lambda x: x[1])
                    total_cost += best[1]
                else:
                    # No suitable static candidate - fallback to a conservative finite cost
                    total_cost += 6

            elif pred == "chfbweyv":
                # waypoint data goal
                _, wp_goal = parts
                # already satisfied?
                if any(match(f, "chfbweyv", wp_goal) for f in state):
                    continue
                # if any i0yg0wob present for this waypoint, only finalize needed
                if any(pwp == wp_goal for (_r, pwp) in present_i0yg0wob):
                    total_cost += 1
                    continue
                # else compute minimal travel distance from any rover to this waypoint
                min_travel = None
                for rov in rovers:
                    pos = rover_pos.get(rov, None)
                    if pos is None:
                        continue
                    d = self._shortest_distance(pos, wp_goal)
                    if min_travel is None or d < min_travel:
                        min_travel = d
                if min_travel is None:
                    # no rover positions known, assume a small penalty
                    min_travel = 1
                # if waypoint has a special site flag, assume two actions create intermediate and then finalize
                if (wp_goal in self.cz_sites) or (wp_goal in self.hf_sites) or any(match(f, "czmzagzw", wp_goal) or match(f, "hfrxxy8q", wp_goal) for f in state):
                    total_cost += min_travel + 2
                else:
                    # otherwise assume a slightly larger sequence
                    total_cost += min_travel + 3

            else:
                # Generic fallback for unexpected goal predicates: count them as 1 each if not already present
                if not any(match(f, *get_parts(goal)) for f in state):
                    total_cost += 1

        # ensure heuristic is finite and non-negative
        if total_cost < 0:
            total_cost = 0
        return int(total_cost)