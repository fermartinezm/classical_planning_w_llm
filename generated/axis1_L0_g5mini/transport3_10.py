from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the transport domain that estimates the number of actions
    required to deliver all packages to their goal locations. The estimate
    is computed as a sum of per-package transport costs under a simple,
    optimistic model that ignores complex interactions between packages
    (e.g. picking multiple packages in one route). The heuristic uses
    shortest-path distances on the road graph (from static facts) to
    estimate required drive actions, and counts pick-up and drop actions
    explicitly.

    Assumptions
    -----------
    - Each 'drive' action moves a vehicle along a single road fact (one edge).
    - A pick-up action costs 1 and a drop action costs 1 (matching domain operator counts).
    - Any vehicle can pick up any package (we do not enforce detailed capacity
      constraints for the heuristic). This simplifies the estimate and keeps it
      efficiently computable. (Capacity predecessor information is extracted
      and available but not strictly enforced in the per-package cost.)
    - If a package is currently inside a vehicle, only the drive actions to the
      goal and a single drop are necessary (we assume the vehicle can drop it).
    - If a package is on the ground, we consider the best vehicle to serve it
      (including a hypothetical vehicle initially at the package location). The
      cost is: drive(vehicle -> package) + pick + drive(package -> goal) + drop.
    - When the road graph does not provide a path between two locations, the
      heuristic uses a large finite penalty to keep the heuristic value finite.

    Heuristic Initialization
    ------------------------
    In __init__, the heuristic:
    - Stores the goal location for each goal package (extracted from task.goals).
    - Extracts the directed road graph from the static facts (task.static).
    - Extracts the capacity-predecessor relations from static facts to provide
      size ordering information (not strictly required by the simple cost
      model, but kept for completeness and possible extensions).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already satisfied in the state, return 0.
    2. Extract the current state information:
       - Vehicle locations via (at <vehicle> <loc>) facts (vehicles are those
         that appear in capacity facts in the state).
       - Vehicle capacities (capacity <vehicle> <size>) -- stored for potential use.
       - Package positions: either (in <package> <vehicle>) or (at <package> <loc>).
    3. For each package that must be placed at some goal location:
       - If it is already at the goal location, cost = 0.
       - If it is inside vehicle v:
         * Find the shortest-path length (in drive actions) from v's current
           location to the package's goal location using the road graph.
         * Cost = drive_distance + 1 (single drop action).
       - If it is on the ground at location L:
         * Consider all actual vehicles v and also a hypothetical vehicle
           located at L (this models an immediately-available vehicle).
         * For each candidate vehicle position Vpos, compute:
             drive(Vpos -> L) + 1 (pick) + drive(L -> goal) + 1 (drop)
           where drive(...) is the shortest-path length in the road graph.
         * Choose the minimum cost among candidates as the package's cost.
    4. Sum the per-package costs to produce the heuristic value.
    5. If any required road distance is unreachable, use a large finite penalty
       for that term to ensure the heuristic remains finite (but large).
    """

    def __init__(self, task):
        # store goal mapping: package -> goal location (only 'at' goals are relevant)
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) >= 3:
                pkg = parts[1]
                loc = parts[2]
                self.goal_locations[pkg] = loc

        # extract static road graph and capacity-predecessor relations
        self.road_graph = {}  # adjacency: loc -> set(of neighbor locs)
        self.predecessor = {}  # size_successor -> size_predecessor (s2 -> s1)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) >= 3:
                frm, to = parts[1], parts[2]
                self.road_graph.setdefault(frm, set()).add(to)
                # ensure nodes are present even if no outgoing edges
                self.road_graph.setdefault(to, set())
            elif parts[0] == "capacity-predecessor" and len(parts) >= 3:
                s1, s2 = parts[1], parts[2]
                # s1 is predecessor of s2 (picking moves from s2 -> s1)
                self.predecessor[s2] = s1

        # determine all known locations (nodes in graph)
        self.locations = set(self.road_graph.keys())

        # large penalty for unreachable distances
        self.LARGE = 10000

    def shortest_path_length(self, start, goal):
        """Return the shortest distance in edges from start to goal on the road graph.
        If start==goal returns 0. If either node unknown or no path exists, return None."""
        if start == goal:
            return 0
        if start not in self.road_graph or goal not in self.road_graph:
            return None
        # BFS
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.road_graph.get(node, ()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If already a goal state, heuristic is zero (requirement 2).
        if self.goal_locations:
            all_goal_facts = {f"(at {pkg} {loc})" for pkg, loc in self.goal_locations.items()}
            if all_goal_facts <= state:
                return 0

        # Extract vehicle capacities (vehicles are those mentioned in capacity facts).
        vehicle_capacity = {}  # v -> size
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "capacity" and len(parts) >= 3:
                vehicle_capacity[parts[1]] = parts[2]
        vehicles = set(vehicle_capacity.keys())

        # Extract vehicle locations
        vehicle_locations = {}  # v -> location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in vehicles:
                    vehicle_locations[obj] = loc

        # Extract package current positions: either (in p v) or (at p l)
        package_pos = {}  # pkg -> ("in", vehicle) or ("at", location)
        # initialize packages mentioned in goals to ensure they are considered
        for pkg in self.goal_locations:
            package_pos[pkg] = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "in" and len(parts) >= 3:
                pkg, v = parts[1], parts[2]
                if pkg in package_pos:
                    package_pos[pkg] = ("in", v)
            elif parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in package_pos:
                    package_pos[obj] = ("at", loc)

        total_cost = 0

        # For each goal package compute estimated minimal cost to place it at goal.
        for pkg, goal_loc in self.goal_locations.items():
            pos = package_pos.get(pkg)
            # If unknown in state, assume worst-case finite penalty
            if pos is None:
                total_cost += self.LARGE
                continue

            # If already at goal, cost 0
            if pos[0] == "at" and pos[1] == goal_loc:
                continue

            # If inside a vehicle:
            if pos[0] == "in":
                v = pos[1]
                v_loc = vehicle_locations.get(v)
                if v_loc is None:
                    # vehicle location unknown -> give large penalty
                    drop_cost = self.LARGE
                else:
                    d = self.shortest_path_length(v_loc, goal_loc)
                    drop_cost = (d if d is not None else None)
                    if drop_cost is None:
                        drop_cost = self.LARGE
                # one drop action
                total_cost += drop_cost + 1
            else:
                # package on ground at location L
                pkg_loc = pos[1]

                # cost if a hypothetical vehicle is already at pkg_loc (no initial drive)
                d_pkg_to_goal = self.shortest_path_length(pkg_loc, goal_loc)
                if d_pkg_to_goal is None:
                    base_cost = self.LARGE
                else:
                    base_cost = d_pkg_to_goal
                best_cost = 1 + base_cost + 1  # pick + drive(pkg->goal) + drop

                # consider all real vehicles
                for v in vehicles:
                    v_loc = vehicle_locations.get(v)
                    if v_loc is None:
                        continue
                    d_v_to_pkg = self.shortest_path_length(v_loc, pkg_loc)
                    d_pkg_to_goal = self.shortest_path_length(pkg_loc, goal_loc)
                    # If any distance missing, treat as unreachable for this vehicle
                    if d_v_to_pkg is None or d_pkg_to_goal is None:
                        continue
                    candidate = d_v_to_pkg + 1 + d_pkg_to_goal + 1
                    if candidate < best_cost:
                        best_cost = candidate

                # If we could not find any finite candidate (e.g., no vehicle path),
                # fall back to large penalty but finite.
                if best_cost >= self.LARGE:
                    total_cost += self.LARGE
                else:
                    total_cost += best_cost

        # final heuristic is integer (sum of action counts)
        return int(total_cost)