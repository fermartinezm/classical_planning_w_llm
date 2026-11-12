from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the transport domain that estimates the number of actions
    (drive, pick-up, drop) required to reach the goal arrangement of packages.
    It reasonably accounts for driving distances (number of drive actions along
    roads) and the required pick-up and drop actions. The heuristic computes,
    for each package that is not yet at its goal, the minimum-cost plan that
    moves that package to its goal using a single vehicle (drive -> pick ->
    drive -> drop or drive -> drop if already in vehicle). The overall
    heuristic value is the sum of those per-package estimates.

    Assumptions
    -----------
    - Each package ends up being moved by some vehicle (no other means of motion).
    - Picking up a package costs 1 action; dropping a package costs 1 action.
    - Driving between two adjacent locations costs 1 action per road step.
    - Capacity bookkeeping is approximated: the heuristic ignores complex
      re-shuffling of packages between vehicles and assumes vehicles can be
      used for the package after optionally dropping one carried package if
      necessary. (The true domain enforces capacity via capacity-predecessor
      facts; this heuristic uses those facts only to compute capacity ranks,
      but does not simulate complex multi-package load/unload sequences.)
    - The heuristic is not required to be admissible; it aims to be informative
      and efficiently computable to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - The set of goal facts (task.goals) to decide which packages matter.
    - All capacity-predecessor facts from static facts and computes a numeric
      rank for each size token. The rank gives a notion of "free slots"
      associated with a size token (rank 0 = smallest size, larger ranks =
      more free capacity).
    - The adjacency graph of locations from static "road" facts and precomputes
      nothing else eagerly; BFS is used on demand to compute shortest path
      lengths (number of drive actions) between locations. The set of vehicle
      symbols is derived from capacity facts in states (capacity facts name
      vehicles).
    - The mapping of package -> goal location (from task.goals).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner state (node.state) the heuristic does the following:

    1. Parse the state facts to obtain:
       - Current location of each vehicle (facts of the form "(at vX lY)").
       - For each package of interest (those with goal conditions), whether it
         is at some location "(at p l)" or inside a vehicle "(in p v)".

    2. Build or reuse the roads adjacency list from static facts. Roads are
       directed edges; the number of drive actions equals the directed
       shortest-path length between two locations.

    3. For every package that is not currently at its goal:
       - If the package is inside vehicle v:
         * Cost estimate = distance(vehicle_location, goal_location) + 1 (drop).
           If no path exists, use a large finite penalty.
       - If the package is at location L (not in any vehicle):
         * For each vehicle v, compute the cost:
           distance(v_location, L) + 1 (pick) + distance(L, goal_location) + 1 (drop).
           Choose the minimum across vehicles as the per-package estimate.
         * If no vehicle has a path to L or L cannot reach the goal, use a
           large finite penalty to keep the value finite.
       - If the package is already at the goal location, cost is 0.

    4. The final heuristic value is the sum of the per-package estimates.

    Notes on design trade-offs:
    - The heuristic treats packages independently and sums per-package costs.
      This ignores possible savings from multi-package trips by the same
      vehicle. While this may overestimate in some cases, it is simple,
      informative, fast to compute, and usually guides search well for the
      transport domain used here.
    - Distances are computed using BFS on the road graph; BFS is efficient
      for the typical size of transport instances.
    """

    def __init__(self, task):
        # Goal facts (frozenset of strings like "(at p1 l2)")
        self.goals = set(task.goals)

        # Static facts (roads and capacity-predecessor)
        static = set(task.static)

        # Build adjacency list for roads (directed)
        self.adj = {}
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "road":
                _, a, b = parts
                self.adj.setdefault(a, set()).add(b)
        # Ensure all locations have an entry
        for s in list(self.adj.keys()):
            self.adj.setdefault(s, set())

        # Parse capacity-predecessor relations from static to build size-ranks.
        # capacity-predecessor s1 s2 means s1 is predecessor (smaller) of s2.
        predecessors = {}   # s2 -> list of s1 (predecessors)
        successors = {}     # s1 -> list of s2 (successors)
        sizes = set()
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "capacity-predecessor":
                _, s1, s2 = parts
                sizes.add(s1); sizes.add(s2)
                successors.setdefault(s1, []).append(s2)
                predecessors.setdefault(s2, []).append(s1)
        # Also include any size symbols that might appear in task facts (safe)
        for fact in task.facts:
            parts = get_parts(fact)
            if parts[0] == "capacity" and len(parts) >= 3:
                sizes.add(parts[2])

        # Compute ranks: minimal sizes (no predecessor) get rank 0; successors get rank+1
        # Use topological-like BFS from minimal nodes.
        rank = {}
        # nodes with no predecessor have rank 0
        queue = deque()
        for s in sizes:
            if s not in predecessors:
                rank[s] = 0
                queue.append(s)
        # propagate ranks to successors, using max to handle DAGs
        while queue:
            s = queue.popleft()
            for succ in successors.get(s, []):
                new_rank = rank[s] + 1
                if succ not in rank or new_rank > rank[succ]:
                    rank[succ] = new_rank
                    queue.append(succ)
        # store rank mapping (if no sizes found, rank stays empty)
        self.size_rank = rank  # size_token -> integer rank (free-slot count)

        # Extract package -> goal_location mapping from goals
        self.package_goals = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "at" and len(parts) >= 3:
                _, p, loc = parts
                self.package_goals[p] = loc

        # A large finite penalty for unreachable routing situations
        self.INF_COST = 10000

    def _bfs_distance(self, start):
        """
        Return a dict mapping locations reachable from `start` to the shortest
        number of drive actions (road edges) required to reach them. If start
        is not known in the adjacency list, return empty dict.
        """
        if start not in self.adj:
            return {}
        dist = {start: 0}
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist

    def __call__(self, node):
        state = set(node.state)  # frozenset -> set for membership operations

        # Quick goal test: if all goal facts are satisfied, return 0.
        if self.goals <= state:
            return 0

        # Identify vehicles by capacity facts in the state: (capacity v s)
        vehicles = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "capacity" and len(parts) >= 3:
                _, v, s = parts
                # vehicle v has capacity size s
                vehicles[v] = s

        # Map vehicle -> location using (at v loc)
        vehicle_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in vehicles:
                    vehicle_loc[obj] = loc

        # Map packages: whether at location or in vehicle
        package_loc = {}   # package -> location (if on ground) or vehicle (if inside)
        package_in = {}    # package -> vehicle if in, else None
        # Initialize packages of interest (from goals)
        for p in self.package_goals:
            package_loc[p] = None
            package_in[p] = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in package_loc:
                    package_loc[obj] = loc
                    package_in[obj] = None
            elif parts[0] == "in" and len(parts) >= 3:
                obj, v = parts[1], parts[2]
                if obj in package_loc:
                    package_in[obj] = v
                    package_loc[obj] = None

        # Precompute BFS distances from all vehicle locations and all package locations
        # (only for relevant unique starts).
        loc_starts = set()
        for v in vehicle_loc.values():
            loc_starts.add(v)
        for loc in package_loc.values():
            if loc is not None:
                loc_starts.add(loc)
        # Also include goal locations (for BFS from package/vehicle starts we don't need goal as start)
        # Compute BFS distances mapping: start_location -> {dest:dist}
        distances = {}
        for start in loc_starts:
            distances[start] = self._bfs_distance(start)

        total_cost = 0

        # For each package with a goal, compute minimal estimated cost to reach its goal.
        for pkg, goal_loc in self.package_goals.items():
            # If package is already at goal, cost 0.
            if package_loc.get(pkg) == goal_loc:
                continue
            # If package is inside a vehicle
            if package_in.get(pkg) is not None:
                v = package_in[pkg]
                # If we do not know vehicle location or vehicle is not in vehicles, can't route -> penalty.
                v_loc = vehicle_loc.get(v)
                if v_loc is None:
                    total_cost += self.INF_COST
                    continue
                # distance from vehicle location to goal
                dist_map = distances.get(v_loc)
                if dist_map is None:
                    dist_map = self._bfs_distance(v_loc)
                    distances[v_loc] = dist_map
                d = dist_map.get(goal_loc)
                if d is None:
                    total_cost += self.INF_COST
                else:
                    # one drop action
                    total_cost += d + 1
                continue

            # Package is on the ground at some location
            p_loc = package_loc.get(pkg)
            if p_loc is None:
                # Unknown position for the package; treat as unreachable
                total_cost += self.INF_COST
                continue

            # For each vehicle, compute cost: drive(v_loc -> p_loc) + 1 pick + drive(p_loc -> goal) + 1 drop
            best = None
            # Compute distance map from p_loc if needed
            if p_loc not in distances:
                distances[p_loc] = self._bfs_distance(p_loc)
            p_dist_map = distances[p_loc]
            for v, s in vehicles.items():
                v_loc = vehicle_loc.get(v)
                if v_loc is None:
                    continue
                # Ensure v_loc distances are known
                v_dist_map = distances.get(v_loc)
                if v_dist_map is None:
                    v_dist_map = self._bfs_distance(v_loc)
                    distances[v_loc] = v_dist_map

                # distance v -> p
                d1 = v_dist_map.get(p_loc)
                # distance p -> goal
                d2 = p_dist_map.get(goal_loc)
                if d1 is None or d2 is None:
                    # If either leg unreachable by this vehicle, skip it
                    continue
                cost = d1 + 1 + d2 + 1  # drive to package, pick, drive to goal, drop
                # We could adjust cost slightly if vehicle already at p_loc (d1 == 0) or at goal (d2 == 0)
                if best is None or cost < best:
                    best = cost

            if best is None:
                # No vehicle can reach p_loc or p_loc cannot reach goal -> large penalty
                total_cost += self.INF_COST
            else:
                total_cost += best

        return int(total_cost)