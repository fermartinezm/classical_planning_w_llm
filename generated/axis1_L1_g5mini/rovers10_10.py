from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the individual tokens of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (cdmmbhnd). The main goal in
    instance files of this domain is to produce facts of the form
    (lk059nof <objective> <mode>) for a set of objectives and modes. The typical
    achievement pattern in this domain is:
      - mount/activate a camera on a rover (jx1xpj7f),
      - take/store an image (tkek73dx -> produces l3943stc),
      - post-process/transfer the stored image into the final objective fact (bypa1zm7 ->
        produces lk059nof).
    Movement of rovers between waypoints uses azu4u5xz and requires rover-specific
    traversability facts (xuwncpn8) plus global connectivity facts (kxvuge0a).

    This heuristic estimates the number of actions remaining to achieve all
    (lk059nof objective mode) goals by:
      - For each missing goal, computing the cheapest rover-based plan:
          moves_to_relevant_waypoint + (mount action) + (take-image action) + (finalize action)
        where moves_to_relevant_waypoint is the shortest-path length (number of azu4u5xz moves)
        for that rover from its current location to any waypoint associated with the objective.
      - If an l3943stc fact (already taken image stored on a rover for the objective+mode)
        is already present in the current state, assume only the finalizing action remains.
      - If no rover/camera combination appears capable (based on static information),
        use a small fallback constant per goal to keep the heuristic finite.

    Assumptions
    -----------
    - The relevant sequence of actions that produces lk059nof is:
        jx1xpj7f (mount camera) -> tkek73dx (take image) -> bypa1zm7 (finalize)
      and therefore we approximate these three domain operators as three required actions
      whenever an image must still be taken and processed.
    - Movement cost is 1 per azu4u5xz action; the shortest path on the rover-specific
      directed traversability graph approximates the required number of movement actions.
    - Cameras that can be used for an objective are those that have a static
      g52pa0bi relation to the objective; the camera must support the required mode
      (xq9py0w8) and must be associated to the rover (uss6vmx6).
    - If l3943stc already exists for an objective+mode in the state, only the final
      action (bypa1zm7) is counted (approximated as one action).
    - The heuristic is not guaranteed admissible but is designed to be informative
      and cheaply computable.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores the static relations needed to compute the
    heuristic quickly on every node evaluation:
      - self.objective_waypoints: objective -> set of waypoints (from wecz86om).
      - self.cameras_for_objective: objective -> set of cameras (from g52pa0bi).
      - self.camera_modes: camera -> set of modes (from xq9py0w8).
      - self.camera_on_rover: camera -> rover (from uss6vmx6).
      - self.rovers_with_mount: set of rovers that can mount/use cameras (from jy67mgmv).
      - self.kx_edges: set of (from, to) waypoint edges from kxvuge0a (global connectivity).
      - self.xuwn_edges_by_rover: rover -> set of (from, to) edges from xuwncpn8 (rover-specific)
        (these edges are intersected with global kxvuge0a when computing traversable edges).
      - self.goals: set of goal fact strings to allow goal check.

    Step-By-Step Thinking for Computing Heuristic (__call__)
    -------------------------------------------------------
    1. If all goal facts are already present in the current state, return 0.
    2. Parse the current state to extract:
       - rover_locations: mapping rover -> waypoint from ilkg22gp facts.
       - present_l3943stc: set of (objective, mode) for which l3943stc is already true.
    3. For each required goal fact "(lk059nof <objective> <mode>)" that is not in the state:
       a) If (objective, mode) in present_l3943stc, estimate cost 1 (single finalize action).
       b) Otherwise, for each rover that:
          - is listed as mountable (jy67mgmv),
          - has a camera attached (uss6vmx6) that both (i) is declared for the objective
            (g52pa0bi) and (ii) supports the required mode (xq9py0w8),
          compute the rover-specific shortest path length (BFS) from its current location
          to any waypoint listed for the objective (wecz86om) using directed edges that are
          present both in xuwncpn8 (for that rover) and in kxvuge0a (global connectivity).
          The estimated cost for that rover is: path_length + 3 (mount + take + finalize).
       c) Take the minimal estimated cost among all rovers. If no rover/camera provides a path
          (or there is no suitable camera), use a fallback cost (here 5) for that goal to keep
          the heuristic finite.
    4. Return the sum of the per-goal minimal estimates.

    Notes on Complexity and Effectiveness
    ------------------------------------
    - The heuristic runs BFS for each rover/goal pair but only on the small waypoint graph,
      so it remains efficient on typical problem sizes of this domain.
    - Although optimistic in some respects (ignores some additional preconditions needed by
      bypa1zm7 and others), the heuristic focuses on the dominant costs: rover movement and
      the three core imaging actions, and therefore helps guide greedy best-first search
      effectively toward goals.
    """

    def __init__(self, task):
        """Extract static structures and goals from the task for fast use in __call__."""
        self.goals = set(task.goals)
        static = task.static

        # objective -> {waypoint, ...}
        self.objective_waypoints = {}
        # objective -> {camera, ...}
        self.cameras_for_objective = {}
        # camera -> {mode, ...}
        self.camera_modes = {}
        # camera -> rover
        self.camera_on_rover = {}
        # rovers that have mount capability
        self.rovers_with_mount = set()
        # global waypoint edges: set of (from, to)
        self.kx_edges = set()
        # rover-specific xuwn edges: rover -> set((from, to))
        self.xuwn_edges_by_rover = {}

        for fact in static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "wecz86om":
                # (wecz86om objective waypoint)
                _, obj, wp = parts
                self.objective_waypoints.setdefault(obj, set()).add(wp)

            elif pred == "g52pa0bi":
                # (g52pa0bi camera objective)
                _, camera, obj = parts
                self.cameras_for_objective.setdefault(obj, set()).add(camera)

            elif pred == "xq9py0w8":
                # (xq9py0w8 camera mode)
                _, camera, mode = parts
                self.camera_modes.setdefault(camera, set()).add(mode)

            elif pred == "uss6vmx6":
                # (uss6vmx6 camera rover)
                _, camera, rover = parts
                self.camera_on_rover[camera] = rover

            elif pred == "jy67mgmv":
                # (jy67mgmv rover)
                _, rover = parts
                self.rovers_with_mount.add(rover)

            elif pred == "kxvuge0a":
                # (kxvuge0a from to)
                _, a, b = parts
                self.kx_edges.add((a, b))

            elif pred == "xuwncpn8":
                # (xuwncpn8 rover from to)
                _, rover, a, b = parts
                self.xuwn_edges_by_rover.setdefault(rover, set()).add((a, b))

        # Precompute rover-specific traversable adjacency by intersecting with kx edges.
        # For each rover, keep only edges that exist also in global kxvuge0a (movement requires both).
        self.traversable_by_rover = {}
        for rover, edges in self.xuwn_edges_by_rover.items():
            self.traversable_by_rover[rover] = {e for e in edges if e in self.kx_edges}

    def __call__(self, node):
        """Return the estimated number of actions to achieve all remaining lk059nof goals."""
        state = node.state

        # 1) If we already reached all goals, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Parse state for rover locations and any already taken images (l3943stc).
        rover_locations = {}  # rover -> waypoint
        present_l3943 = set()  # set of (objective, mode) that have l3943stc
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ilkg22gp":
                # (ilkg22gp rover waypoint)
                _, rover, waypoint = parts
                rover_locations[rover] = waypoint
            elif pred == "l3943stc":
                # (l3943stc rover objective mode)
                _, _rover, objective, mode = parts
                present_l3943.add((objective, mode))

        total_estimate = 0

        # Evaluate each goal fact. Goals are strings like "(lk059nof objective1 colour)"
        for goal in self.goals:
            if goal in state:
                continue  # already satisfied

            parts = get_parts(goal)
            if parts[0] != "lk059nof" or len(parts) < 3:
                # If the goal is something else, conservatively contribute a small constant.
                total_estimate += 2
                continue

            _, objective, mode = parts
            # If an image has already been taken and stored, assume only one final action required.
            if (objective, mode) in present_l3943:
                total_estimate += 1
                continue

            # Otherwise, search for the best rover that can accomplish the imaging.
            best_cost_for_goal = None

            # Candidate cameras for this objective:
            cameras = self.cameras_for_objective.get(objective, set())

            # waypoints associated with this objective (locations where imaging is possible)
            target_waypoints = self.objective_waypoints.get(objective, set())
            if not target_waypoints:
                # no known waypoint for objective -> fallback cost
                total_estimate += 5
                continue

            # For each rover available in the domain, evaluate feasibility and path cost.
            # We consider only rovers that are mountable and appear in state if possible.
            candidate_rovers = set(self.traversable_by_rover.keys()) | set(rover_locations.keys()) | self.rovers_with_mount
            for rover in candidate_rovers:
                # Rover must be mountable (jy67mgmv), otherwise it cannot perform jx1xpj7f/tkek73dx reliably.
                if rover not in self.rovers_with_mount:
                    continue

                # Must have a current location in the state to be useful
                if rover not in rover_locations:
                    continue

                rover_wp = rover_locations[rover]

                # For this rover, find if there's a camera attached that supports the needed mode and is associated with objective.
                suitable_camera_found = False
                for cam in cameras:
                    cam_rover = self.camera_on_rover.get(cam)
                    if cam_rover != rover:
                        continue
                    if mode not in self.camera_modes.get(cam, set()):
                        continue
                    suitable_camera_found = True
                    break
                if not suitable_camera_found:
                    # no suitable camera attached to this rover for this objective/mode
                    continue

                # BFS on rover-specific traversable graph to find distance to any target waypoint
                traversable = self.traversable_by_rover.get(rover, set())
                # If no traversable edges, allow zero-move if already at target waypoint
                if rover_wp in target_waypoints:
                    moves = 0
                else:
                    # Build adjacency dict for BFS
                    adj = {}
                    for a, b in traversable:
                        adj.setdefault(a, []).append(b)
                    # BFS
                    q = deque()
                    q.append((rover_wp, 0))
                    visited = {rover_wp}
                    found = False
                    moves = None
                    while q:
                        cur, dist = q.popleft()
                        if cur in target_waypoints:
                            moves = dist
                            found = True
                            break
                        for nb in adj.get(cur, ()):
                            if nb not in visited:
                                visited.add(nb)
                                q.append((nb, dist + 1))
                    if not found:
                        # No path for this rover to any objective waypoint
                        continue

                # Estimated cost: moves + 3 actions (mount, take image, finalize)
                cost = moves + 3
                if best_cost_for_goal is None or cost < best_cost_for_goal:
                    best_cost_for_goal = cost

            if best_cost_for_goal is None:
                # No rover could be matched; fallback finite estimate
                total_estimate += 5
            else:
                total_estimate += best_cost_for_goal

        # Ensure the heuristic is finite and at least 0
        return int(max(0, total_estimate))