from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain ("cdmmbhnd").
    The domain models rovers, cameras, objectives, and waypoints. Goals require
    that an objective has an image of a given resolution available (predicate
    `lk059nof objective resolution`). The typical plan to achieve such a goal:
      - move a rover to a waypoint associated with the objective (wecz86om objective waypoint),
      - (optionally) mount a suitable camera on that rover (jx1xpj7f),
      - capture the image with the camera (tkek73dx) producing l3943stc (rover-objective-resolution),
      - post-process / send the image (bypa1zm7) to obtain lk059nof.

    This heuristic estimates the number of actions still required to get every
    missing `lk059nof` goal. The estimate is not required to be admissible; it
    should be informative and efficiently computable.

    Assumptions
    -----------
    - The static facts contain:
        * adjacency between waypoints (kxvuge0a wp1 wp2).
        * mapping from objective -> waypoint(s) via wecz86om.
        * camera capabilities xq9py0w8 (camera resolution).
        * camera<->rover relation uss6vmx6 (camera usable by rover).
        * camera objective suitability g52pa0bi (camera <-> objective).
    - A camera can be mounted on a rover using jx1xpj7f when rover is at the
      objective's waypoint (this matches the domain operator structure).
    - Capturing an image (tkek73dx) requires the camera to be mounted (v25mqvwj)
      and the rover to be at the objective waypoint and the camera to support
      the requested resolution.
    - After capturing, a single post-processing action (bypa1zm7) is used to
      obtain `lk059nof`. (We conservatively count one extra action for that.)
    - Movement cost between adjacent waypoints is counted as 1 (operator azu4u5xz).
    - Mounting and capturing each cost 1 action, and post-processing costs 1 action.
      (These costs are chosen to reflect the number of operators involved in the domain.)

    Heuristic Initialization
    ------------------------
    In the constructor we parse static facts from the task and build fast lookup
    structures:
    - adjacency graph of waypoints from kxvuge0a (for shortest-path BFS).
    - objective -> set(waypoints) from wecz86om.
    - camera -> set(resolutions) from xq9py0w8.
    - camera -> set(objectives) from g52pa0bi.
    - camera -> rover mapping from uss6vmx6.
    We also keep the set of goal facts (task.goals) for quick checks.

    Step-By-Step Thinking for Computing Heuristic (for a given state)
    ----------------------------------------------------------------
    1. If the state already satisfies all goals (self.goals subset of state),
       return 0.

    2. Parse the current state to obtain:
       - rover positions (ilkg22gp rover waypoint),
       - which cameras are currently mounted on which rovers (v25mqvwj camera rover),
       - which l3943stc facts (raw captured images) are already present,
       - which lk059nof facts (goal facts) are already present.

    3. For each goal fact that is missing (a goal is of form (lk059nof objective res)):
       a. If a corresponding l3943stc (captured image) for that objective+res
          is already present in the state, estimate cost 1 (the post-processing
          action bypa1zm7) to turn it into the final `lk059nof`.
       b. Otherwise, find the best rover-camera pair that can produce that
          objective+resolution:
            - Candidate cameras must support the required resolution (xq9py0w8).
            - Candidate cameras must be usable by some rover (uss6vmx6).
            - Candidate cameras are preferred if they are listed as suitable
              for the objective (g52pa0bi); however, in the absence of such
              mapping we still consider any camera that supports the resolution
              and is usable by a rover (to keep the heuristic finite).
            - For each rover that can use a suitable camera, compute the
              shortest-path distance from the rover's current waypoint to any
              waypoint associated with the objective (wecz86om), using the
              adjacency graph (kxvuge0a). If no path exists to any objective
              waypoint, the candidate is considered infeasible.
            - The estimated cost for that candidate is:
                distance_to_objective_waypoint
                + mount_cost (0 if camera already mounted on that rover in state,
                              else 1 for jx1xpj7f)
                + 1 (capture via tkek73dx)
                + 1 (post-process via bypa1zm7)
            - The heuristic uses the minimum such cost over all candidates.

       c. If no candidate (camera+rover+path) can be found, fall back to a
          conservative finite estimate (a fixed penalty) so heuristic remains finite.

    4. Return the sum of the per-goal estimates. The heuristic assumes goals are
       independent (overestimates in case of shared work but provides an
       informative guidance for greedy best-first search).

    Notes on Design Choices
    -----------------------
    - The heuristic intentionally reasons about concrete rovers, cameras and
      waypoint distances to produce a more informative estimate than simply
      counting remaining goals.
    - Approximations/simplifications (e.g., counting post-processing as a
      single action) make the heuristic fast to compute while retaining useful
      structure from the domain.
    """

    def __init__(self, task):
        # Store goals (set of goal fact strings)
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency graph between waypoints from kxvuge0a facts.
        self.adj = {}  # waypoint -> set(neighbor_waypoints)
        # objective -> set(waypoints)
        self.obj_waypoints = {}
        # camera -> set(resolutions)
        self.camera_resolutions = {}
        # camera -> set(objectives) (suitability)
        self.camera_objectives = {}
        # camera -> set(rovers) (usable by)
        self.camera_rovers = {}

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "kxvuge0a":
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
            elif pred == "wecz86om":
                # (wecz86om objective waypoint)
                obj, wp = parts[1], parts[2]
                self.obj_waypoints.setdefault(obj, set()).add(wp)
            elif pred == "xq9py0w8":
                # (xq9py0w8 camera resolution)
                cam, res = parts[1], parts[2]
                self.camera_resolutions.setdefault(cam, set()).add(res)
            elif pred == "g52pa0bi":
                # (g52pa0bi camera objective)
                cam, obj = parts[1], parts[2]
                self.camera_objectives.setdefault(cam, set()).add(obj)
            elif pred == "uss6vmx6":
                # (uss6vmx6 camera rover)
                cam, rover = parts[1], parts[2]
                self.camera_rovers.setdefault(cam, set()).add(rover)

        # A conservative penalty used when no feasible candidate is found.
        # Must be finite to ensure the heuristic returns a finite number.
        self.infeasible_penalty = 20

    def _bfs_distance(self, start, targets):
        """
        BFS on the waypoint adjacency graph to find shortest distance from
        start waypoint to any waypoint in targets. Returns integer distance or
        None if no path exists.
        """
        if start in targets:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return d + 1
                visited.add(nbr)
                q.append((nbr, d + 1))
        return None

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goal facts already in the state, return 0.
        if self.goals <= state:
            return 0

        # Parse state facts of interest.
        rover_pos = {}  # rover -> waypoint (from ilkg22gp)
        mounted = set()  # set of (camera, rover) for v25mqvwj
        captured = set()  # set of (rover, objective, res) from l3943stc
        achieved = set()  # lk059nof facts present as (objective, res)

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ilkg22gp":
                # (ilkg22gp rover waypoint)
                rover, wp = parts[1], parts[2]
                rover_pos[rover] = wp
            elif pred == "v25mqvwj":
                # (v25mqvwj camera rover)
                cam, rover = parts[1], parts[2]
                mounted.add((cam, rover))
            elif pred == "l3943stc":
                # (l3943stc rover objective resolution)
                rover, obj, res = parts[1], parts[2], parts[3]
                captured.add((rover, obj, res))
            elif pred == "lk059nof":
                # (lk059nof objective resolution)
                obj, res = parts[1], parts[2]
                achieved.add((obj, res))

        total_estimate = 0

        # For each goal fact, estimate cost if not already achieved.
        for goal in self.goals:
            gparts = get_parts(goal)
            if gparts[0] != "lk059nof":
                # Unknown/unexpected goal predicate; skip (or treat as small penalty).
                if goal not in state:
                    total_estimate += 1
                continue

            obj, res = gparts[1], gparts[2]
            if (obj, res) in achieved:
                continue  # already satisfied

            # If a captured image with matching objective & resolution exists,
            # we only need the post-processing (bypa1zm7).
            any_captured = any((cobj == obj and cres == res) for (_, cobj, cres) in captured)
            if any_captured:
                total_estimate += 1  # one post-processing action
                continue

            # Otherwise, we need to pick a rover & camera to produce the captured image.
            # Candidate cameras are those that support the resolution.
            candidates = []
            for cam, res_set in self.camera_resolutions.items():
                if res in res_set:
                    # cameras usable by rovers:
                    for rover in self.camera_rovers.get(cam, ()):
                        # Only consider rovers that have a known position in this state.
                        if rover in rover_pos:
                            candidates.append((cam, rover))

            # If none found by direct capability, attempt a weaker candidate selection:
            # consider any camera that supports res even if camera_rovers mapping is missing.
            if not candidates:
                for cam, res_set in self.camera_resolutions.items():
                    if res in res_set:
                        # attempt to use any rover we know about (fallback)
                        for rover in rover_pos.keys():
                            candidates.append((cam, rover))

            best_cost = None

            # Determine the set of waypoints where the objective can be serviced.
            target_waypoints = self.obj_waypoints.get(obj, set())
            if not target_waypoints:
                # If we don't know any waypoint for the objective, use penalty.
                best_cost = self.infeasible_penalty

            else:
                # Evaluate each candidate (camera, rover) pairing.
                for cam, rover in candidates:
                    # Distance from rover current position to any of the objective's waypoints.
                    start_wp = rover_pos.get(rover)
                    if start_wp is None:
                        continue
                    dist = self._bfs_distance(start_wp, target_waypoints)
                    if dist is None:
                        # no path to reach objective's waypoints with this rover
                        continue

                    # Mount cost: if camera already mounted on this rover, 0 else 1.
                    mount_cost = 0 if (cam, rover) in mounted else 1

                    # Check camera-objective suitability (g52pa0bi). If camera is
                    # explicitly suitable for the objective, slightly prefer it by
                    # not increasing cost; otherwise give a small penalty (0 or 0.5).
                    suitability_penalty = 0
                    cam_objs = self.camera_objectives.get(cam)
                    if cam_objs is not None and obj not in cam_objs:
                        # If camera mapping exists but not for this objective, it's a weaker match.
                        suitability_penalty = 0.5

                    # Capture cost + post-processing cost = 2
                    cost = dist + mount_cost + 2 + suitability_penalty

                    # Keep best found cost
                    if best_cost is None or cost < best_cost:
                        best_cost = cost

            # If still no candidate (best_cost None), use fallback penalty
            if best_cost is None:
                best_cost = self.infeasible_penalty

            # Round up to integer to reflect discrete actions.
            total_estimate += int(best_cost + 0.9999)

        return int(total_estimate)