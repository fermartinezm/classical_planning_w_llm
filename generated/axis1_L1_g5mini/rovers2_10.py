from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return list of tokens inside a fact string '(pred a b ...)'"""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (rover/camera-like) domain.
    The goals in this domain are facts of the form (lk059nof <objective> <resolution>).
    To achieve such a goal the domain requires capturing an image (l3943stc) of the
    objective with a camera that supports the requested resolution and then performing
    a finishing action (bypa1zm7) that consumes the image to produce the lk059nof goal fact.
    Cameras are statically assigned to rovers and support specific resolutions.
    Rovers can move along edges (xuwncpn8 facts) to reach the waypoint(s) where the
    objective can be observed (wecz86om facts).

    Assumptions
    -----------
    - Static facts encode:
        * which camera can image which objective (g52pa0bi camera objective).
        * which camera supports which resolutions (xq9py0w8 camera resolution).
        * camera <-> rover assignment (uss6vmx6 camera rover).
        * objective <-> waypoint associations (wecz86om objective waypoint).
        * rover waypoint adjacency (xuwncpn8 rover wp_from wp_to) used for movement.
    - Moving a rover across one xuwncpn8 edge costs 1 action (azu4u5xz).
    - Creating a camera-on-rover link (v25mqvwj) costs 1 action (jx1xpj7f).
    - Taking the picture to produce the intermediate image fact (l3943stc) costs 1 action (tkek73dx).
    - The final processing action that yields (lk059nof ...) costs 1 action (bypa1zm7).
    - If an intermediate fact already exists in the state (v25mqvwj or l3943stc), the
      corresponding action is not needed (we subtract it from the estimate).
    - If no compatible camera/rover can produce the required image, the goal is
      considered costly and assigned a high finite penalty to keep the heuristic finite.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores the following information from the task's
    static facts and goals:
    - goals: the set of goal facts (task.goals).
    - objective_to_waypoints: mapping objective -> set(waypoints) from wecz86om facts.
    - camera_to_objectives: mapping camera -> set(objectives) from g52pa0bi facts.
    - camera_to_resolutions: mapping camera -> set(resolutions) from xq9py0w8 facts.
    - camera_to_rover: mapping camera -> rover from uss6vmx6 facts.
    - rover_graphs: mapping rover -> adjacency dict of waypoint -> set(neighbor_waypoints)
      built from xuwncpn8 facts. These graphs represent per-rover allowed moves.
    """

    def __init__(self, task):
        # store goals (set of strings)
        self.goals = set(task.goals)

        # initialize maps
        self.objective_to_waypoints = {}  # objective -> set(waypoints)
        self.camera_to_objectives = {}  # camera -> set(objectives)
        self.camera_to_resolutions = {}  # camera -> set(resolutions)
        self.camera_to_rover = {}  # camera -> rover
        self.rover_graphs = {}  # rover -> { waypoint: set(neighbors) }

        static_facts = task.static

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "wecz86om" and len(parts) >= 3:
                obj, wp = parts[1], parts[2]
                self.objective_to_waypoints.setdefault(obj, set()).add(wp)
            elif pred == "g52pa0bi" and len(parts) >= 3:
                cam, obj = parts[1], parts[2]
                self.camera_to_objectives.setdefault(cam, set()).add(obj)
            elif pred == "xq9py0w8" and len(parts) >= 3:
                cam, res = parts[1], parts[2]
                self.camera_to_resolutions.setdefault(cam, set()).add(res)
            elif pred == "uss6vmx6" and len(parts) >= 3:
                cam, rover = parts[1], parts[2]
                self.camera_to_rover[cam] = rover
            elif pred == "xuwncpn8" and len(parts) >= 4:
                rover, frm, to = parts[1], parts[2], parts[3]
                if rover not in self.rover_graphs:
                    self.rover_graphs[rover] = {}
                self.rover_graphs[rover].setdefault(frm, set()).add(to)

        # A conservative large penalty for unreachable/unserviceable goals
        self.unreachable_penalty = 100

    def _bfs_distance(self, rover, start_wp, target_wps):
        """
        BFS shortest-path in the per-rover movement graph (xuwncpn8 edges).
        Returns minimal number of moves (edges) from start_wp to any in target_wps.
        If start_wp already in target_wps returns 0.
        If graph or path missing returns None.
        """
        if start_wp in target_wps:
            return 0
        graph = self.rover_graphs.get(rover)
        if not graph:
            return None
        visited = set()
        q = deque()
        q.append((start_wp, 0))
        visited.add(start_wp)
        while q:
            wp, dist = q.popleft()
            # neighbors can be empty
            for nb in graph.get(wp, ()):
                if nb in visited:
                    continue
                if nb in target_wps:
                    return dist + 1
                visited.add(nb)
                q.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Step-By-Step Thinking for Computing Heuristic
        ---------------------------------------------
        For a given state (node.state), the heuristic estimates the number of actions
        needed to achieve all missing (lk059nof <objective> <resolution>) goal facts.

        For each goal fact (objective, resolution):
        1. If the goal fact is already true in the state -> cost 0.
        2. Else, check whether an intermediate image fact l3943stc <rover> <objective> <resolution>
           already exists in the state:
           - If l3943stc exists for that objective+resolution, then we still may need the
             final bypa1zm7 action (cost +1) unless lk059nof already present.
        3. If l3943stc does not exist: we attempt to find a camera+rover pair capable of
           producing it. A camera can produce the image for the given objective+resolution
           if:
             - camera_to_objectives[camera] contains the objective (g52pa0bi),
             - camera_to_resolutions[camera] contains the resolution (xq9py0w8),
             - camera_to_rover[camera] gives a rover that can be used (uss6vmx6).
           For each such camera assigned to a rover:
             - Find the current waypoint of that rover from the state's (ilkg22gp rover wp)
               facts. If missing, that rover cannot be used.
             - From the objective_to_waypoints mapping (wecz86om), get the waypoint(s)
               where the objective is observable.
             - Compute BFS distance (number of move actions) for that rover to reach any
               of those waypoints (using xuwncpn8 edges for that rover).
             - If reachable, the cost to produce l3943stc is estimated as:
                   moves_to_reach + cost_of_jx1xpj7f_if_needed + cost_of_tkek73dx
               where:
                   - moves_to_reach: number of azu4u5xz actions to move rover to the required wp.
                   - cost_of_jx1xpj7f_if_needed: 1 if (v25mqvwj <camera> <rover>) not already in the state,
                     else 0.
                   - cost_of_tkek73dx: 1 (take-picture action) unless l3943stc already exists.
             - After l3943stc exists (either already or we estimate creating it), we assume
               one final bypa1zm7 action is necessary to produce lk059nof (cost +1) unless
               lk059nof is already present.
        4. Among all camera+rover options we pick the minimal estimated cost for that goal.
           If no option is available but the intermediate l3943stc already exists in the state,
           we still count the final bypa1zm7 (cost +1). If neither l3943stc exists nor any
           camera+rover can produce it, we add a finite large penalty to keep the heuristic finite.
        5. The heuristic value is the sum of the minimal estimates for each missing goal.
        """

        state = set(node.state)

        # Quick goal-check: if all goals satisfied -> heuristic 0
        if self.goals <= state:
            return 0

        # extract rover current waypoints from state: ilkg22gp rover waypoint
        rover_wp = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ilkg22gp" and len(parts) >= 3:
                rover, wp = parts[1], parts[2]
                rover_wp[rover] = wp

        # track existing intermediate facts in the state
        existing_v25 = set()  # (camera, rover) pairs with v25mqvwj
        existing_l3943 = set()  # tuples (rover, objective, resolution)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v25mqvwj" and len(parts) >= 3:
                cam, rover = parts[1], parts[2]
                existing_v25.add((cam, rover))
            elif parts[0] == "l3943stc" and len(parts) >= 4:
                rover, obj, res = parts[1], parts[2], parts[3]
                existing_l3943.add((rover, obj, res))

        total_cost = 0

        # compute costs for each goal that is not yet satisfied
        for goal in self.goals:
            if goal in state:
                continue  # already achieved

            parts = get_parts(goal)
            if parts[0] != "lk059nof" or len(parts) < 3:
                # heuristic only handles lk059nof style goals; if something else,
                # assign a small default (shouldn't happen in this domain).
                total_cost += 1
                continue

            obj, res = parts[1], parts[2]

            # If some l3943stc for this objective+res already exists, we only need final action
            # (bypa1zm7) unless lk059nof already true (we checked). Find any rover with that l3943.
            if any((r, obj, res) in existing_l3943 for r, obj, res in existing_l3943):
                # Add 1 for the final processing action
                total_cost += 1
                continue

            # Try all cameras that can image this objective and support this resolution
            best_cost_for_goal = None

            # Build list of candidate cameras that can image objective and support resolution
            candidate_cameras = []
            for cam, objs in self.camera_to_objectives.items():
                if obj in objs and res in self.camera_to_resolutions.get(cam, set()):
                    candidate_cameras.append(cam)

            for cam in candidate_cameras:
                rover = self.camera_to_rover.get(cam)
                if rover is None:
                    continue  # camera not assigned to a rover -> cannot use

                start_wp = rover_wp.get(rover)
                if start_wp is None:
                    continue  # rover has no known position in state -> skip

                target_wps = self.objective_to_waypoints.get(obj, set())
                if not target_wps:
                    continue  # objective has no known waypoints -> skip

                dist = self._bfs_distance(rover, start_wp, target_wps)
                if dist is None:
                    continue  # unreachable with this rover

                # cost components:
                # - moves: dist (azu actions)
                # - jx1xpj7f to create v25 if not already present (camera not at rover)
                v25_needed = 0 if (cam, rover) in existing_v25 else 1
                # - tkek73dx: take-picture action to create l3943stc (1)
                take_picture_cost = 1
                # - final bypa1zm7 to create lk059nof (1)
                finalize_cost = 1

                cost = dist + v25_needed + take_picture_cost + finalize_cost

                if best_cost_for_goal is None or cost < best_cost_for_goal:
                    best_cost_for_goal = cost

            if best_cost_for_goal is not None:
                total_cost += best_cost_for_goal
            else:
                # No viable camera/rover option found. If some l3943stc exists that we missed,
                # we would have handled it earlier. Otherwise assign a finite large penalty.
                total_cost += self.unreachable_penalty

        return total_cost