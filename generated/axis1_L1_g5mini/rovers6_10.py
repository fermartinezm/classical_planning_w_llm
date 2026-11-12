from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract parts from a PDDL fact string like '(pred a b)' -> ['pred','a','b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic tailored for the secret_domain (rover/camera style) domain.
    It estimates the number of actions required to achieve all goal facts
    of the form (lk059nof <objective> <mode>) by reasoning about the minimal
    pipeline required to capture and store an image of a given objective
    in the requested mode:
      - Move a suitable rover to a waypoint where the objective is located.
      - (If necessary) Mount a compatible camera on that rover.
      - Take the image with the camera in the requested mode.
      - Store/transfer the taken image to obtain (lk059nof ...).

    Assumptions
    -----------
    - Movement (azu4u5xz) cost is treated as 1 per hop between adjacent waypoints.
      Movement is only possible along edges present both in xuwncpn8 (rover-specific)
      and kxvuge0a (global connectivity) facts. We compute shortest path lengths
      per rover using those allowed edges.
    - Mounting (jx1xpj7f), capturing (tkek73dx), and storing (bypa1zm7) are each
      counted as one action when needed.
    - If a camera is already mounted on a rover (v25mqvwj in the current state),
      mounting is not required for that (camera, rover) pair.
    - Camera-objective compatibility (g52pa0bi) and camera mode support (xq9py0w8)
      are required to take an image in a given mode. If the static facts do not
      indicate camera-rover compatibility (uss6vmx6), we allow the pairing (optimistic)
      unless explicit compatibility facts exist and exclude it.
    - This heuristic is not required to be admissible. It is intentionally optimistic
      about reusing resources and ignores many auxiliary actions (e.g., preparing stores,
      toggling store availability), but it still accounts for movement and the three
      core actions (mount, take, store).

    Heuristic Initialization
    ------------------------
    In the constructor we extract and index static facts to allow quick lookup:
    - objective_waypoints: map objective -> set of waypoints where the objective can be observed (wecz86om).
    - camera_modes: map camera -> set of supported modes (xq9py0w8).
    - camera_objectives: map camera -> set of objectives it can image (g52pa0bi).
    - camera_rover_compat: map camera -> set of rovers it is explicitly compatible with (uss6vmx6).
    - kx_edges: set of global waypoint connectivity edges (kxvuge0a).
    - xuwncpn8 per-rover edges: used together with kx_edges to build the actual per-rover adjacency graph
      used to compute shortest paths for each rover.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state (node.state) the heuristic follows these steps:

    1. Quick goal check:
       - If all goal facts are already present in the state, return 0.

    2. Parse the current state to obtain:
       - Current rover positions (ilkg22gp facts).
       - Currently mounted cameras (v25mqvwj facts).
       - Any other present facts are not directly required for the simplified approximation.

    3. For each goal fact (we focus on (lk059nof <objective> <mode>) goals):
       - If the goal is already satisfied in the current state, it contributes 0.
       - Otherwise, compute a lower-bound cost to produce that lk059nof:
         a) An (objective, mode) requires an image capture (l3943stc) and then storing (bypa1zm7).
         b) To capture an image in the requested mode we need:
            - A rover at a waypoint where the objective is observable (wecz86om).
            - A camera that can image that objective (g52pa0bi) and supports the requested mode (xq9py0w8).
            - The camera must be (or become) mounted on the chosen rover (v25mqvwj or jx1xpj7f).
         c) Compute the minimal movement cost: shortest path length (in hops) for any rover from its
            current waypoint to any waypoint where the objective is located. Movement is only allowed
            along per-rover edges that also exist as global kxvuge0a edges.
         d) For each feasible (rover, camera, waypoint) combination that satisfies compatibility and mode
            support, estimate:
                cost = movement_hops + mount_needed + take_cost + store_cost
               where mount_needed is 0 if (camera, rover) already mounted in state, else 1.
               take_cost = 1 (tkek73dx) and store_cost = 1 (bypa1zm7).
         e) The minimal such cost is the contribution of that goal to the heuristic. If no feasible
            combination is found, the heuristic uses a large finite penalty (to keep the heuristic
            finite while signaling difficulty/unreachability).

    4. Sum the per-goal minimal costs to get the final heuristic value. Summing makes the
       heuristic more aggressive (less admissible) but often gives stronger guidance for
       greedy best-first search in this domain.

    Note
    ----
    - The heuristic purposefully ignores some action preconditions and auxiliary
      toggling steps for simplicity and speed. It aims to produce a reasonable,
      efficiently computable estimate of the remaining effort (in number of actions)
      to reach the goals.
    """

    def __init__(self, task):
        # Goals we must reach (set of fact strings)
        self.goals = task.goals

        # Extract static facts into indexes for fast lookup.
        static = task.static

        # objective -> set(waypoint)
        self.objective_waypoints = {}
        # camera -> set(modes)
        self.camera_modes = {}
        # camera -> set(objectives)
        self.camera_objectives = {}
        # camera -> set(rovers) (explicit compatibility)
        self.camera_rover_compat = {}
        # global connectivity edges (from, to)
        self.kx_edges = set()
        # xuwncpn8 facts: per-rover allowed edges as (from,to)
        self.xu_edges_per_rover = {}

        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "wecz86om" and len(parts) >= 3:
                obj, wp = parts[1], parts[2]
                self.objective_waypoints.setdefault(obj, set()).add(wp)
            elif pred == "xq9py0w8" and len(parts) >= 3:
                cam, mode = parts[1], parts[2]
                self.camera_modes.setdefault(cam, set()).add(mode)
            elif pred == "g52pa0bi" and len(parts) >= 3:
                cam, obj = parts[1], parts[2]
                self.camera_objectives.setdefault(cam, set()).add(obj)
            elif pred == "uss6vmx6" and len(parts) >= 3:
                cam, rover = parts[1], parts[2]
                self.camera_rover_compat.setdefault(cam, set()).add(rover)
            elif pred == "kxvuge0a" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.kx_edges.add((a, b))
            elif pred == "xuwncpn8" and len(parts) >= 4:
                rover, a, b = parts[1], parts[2], parts[3]
                self.xu_edges_per_rover.setdefault(rover, set()).add((a, b))

        # Build per-rover adjacency by intersecting xuwncpn8 edges with global kx edges.
        # This models the azu4u5xz precondition which requires both predicates.
        self.rover_adj = {}
        for rover, xu_edges in self.xu_edges_per_rover.items():
            adj = {}
            for (a, b) in xu_edges:
                if (a, b) in self.kx_edges:
                    adj.setdefault(a, []).append(b)
            self.rover_adj[rover] = adj

    def _shortest_path_hops(self, rover, start, goal):
        """Return number of hops from start to goal for a given rover, or None if unreachable."""
        if start == goal:
            return 0
        if rover not in self.rover_adj:
            return None
        adj = self.rover_adj[rover]
        # BFS
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, d = queue.popleft()
            for nbr in adj.get(node, []):
                if nbr == goal:
                    return d + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, d + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If already a goal state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse current rover locations and current camera mounts from the state.
        rover_location = {}        # rover -> waypoint (from ilkg22gp)
        mounted_camera_pairs = set()  # (camera, rover) from v25mqvwj

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "ilkg22gp" and len(parts) >= 3:
                rover, wp = parts[1], parts[2]
                rover_location[rover] = wp
            elif pred == "v25mqvwj" and len(parts) >= 3:
                cam, rover = parts[1], parts[2]
                mounted_camera_pairs.add((cam, rover))

        # Cost parameters (each counted as 1 action)
        move_cost = 1
        mount_cost = 1
        take_cost = 1
        store_cost = 1

        total_estimated = 0
        LARGE_PENALTY = 1000  # finite but large penalty for unreachable goals

        # Consider only lk059nof goals explicitly, fall back to 1 for unknown goal patterns.
        for goal in self.goals:
            if goal in state:
                continue  # already satisfied

            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]

            # We provide a focused estimate for (lk059nof <objective> <mode>) goals.
            if pred != "lk059nof" or len(parts) < 3:
                # Unknown goal pattern: give a small positive estimate (optimistic).
                total_estimated += 1
                continue

            objective, mode = parts[1], parts[2]

            # Find candidate waypoints where this objective can be observed.
            candidate_waypoints = self.objective_waypoints.get(objective, set())
            if not candidate_waypoints:
                # No static information where objective is; penalize but finite.
                total_estimated += LARGE_PENALTY
                continue

            best_cost = float("inf")

            # Iterate over rovers and try to find minimal-cost plan to achieve the goal.
            for rover, start_wp in rover_location.items():
                for wp in candidate_waypoints:
                    hops = self._shortest_path_hops(rover, start_wp, wp)
                    if hops is None:
                        # This rover cannot reach this waypoint (according to static move constraints).
                        continue

                    # For each camera that can image this objective in the required mode:
                    for cam, cam_objs in self.camera_objectives.items():
                        if objective not in cam_objs:
                            continue
                        # Camera must support the required mode.
                        if mode not in self.camera_modes.get(cam, set()):
                            continue
                        # If compatibility facts exist for this camera, require rover to be listed;
                        # otherwise assume optimistic compatibility.
                        compat_set = self.camera_rover_compat.get(cam, None)
                        if compat_set is not None and rover not in compat_set:
                            continue

                        # Mount is needed unless the exact (cam, rover) pair is already mounted
                        mount_needed = 0 if (cam, rover) in mounted_camera_pairs else 1

                        # Total estimated actions:
                        cost = hops * move_cost + mount_needed * mount_cost + take_cost + store_cost

                        if cost < best_cost:
                            best_cost = cost

            if best_cost == float("inf"):
                # No feasible combination found -> large penalty to indicate difficulty.
                total_estimated += LARGE_PENALTY
            else:
                total_estimated += best_cost

        return int(total_estimated)