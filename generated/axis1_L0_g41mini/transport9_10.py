from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    Summary:
    ----------
    This heuristic estimates the number of actions required to deliver all packages
    to their goal locations. It takes into account the packages' current positions,
    vehicles' locations, and allowed roads to approximate minimal driving and loading actions.
    It is designed to guide a greedy best-first search efficiently (not necessarily admissible).

    Assumptions:
    ----------
    - Vehicles have a capacity hierarchy denoted by sizes and capacity-predecessor relations.
    - Packages must be picked up from their current location and delivered to their goal location.
    - Vehicles move only along roads connecting locations.
    - Packages can be carried inside vehicles or placed at locations.
    - Capacity constraints restrict how many packages a vehicle can carry simultaneously.
    - Actions are: drive (move vehicle between adjacent locations), pick-up (load package),
      and drop (unload package).
    - The heuristic does not compute exact shortest paths but approximates distances by the minimal road
      steps between locations using BFS.
    - Loading and dropping each count as one action.
    - Driving between adjacent locations counts as one action per road segment.

    Heuristic Initialization:
    ----------
    - Extract all road edges from static facts and build an undirected graph of locations.
    - Extract capacity-predecessor relations to determine capacity "levels".
    - Extract goal package locations from goal facts.
    - Store initial vehicle capacities.
    - Cache minimal distances between all pairs of locations using BFS for fast lookup.

    Step-By-Step Thinking for Computing Heuristic:
    ----------
    For each package:
      1. Identify the current location of the package:
         - If it is inside a vehicle, identify the vehicle's location.
         - Otherwise, package is at a ground location.
      2. Determine the goal location of the package.
      3. Compute the minimal driving distance(s) needed:
         a) If package is on ground:
            - Distance between package location and a vehicle that can carry it (minimum over all vehicles that can fit).
            - Distance from that vehicle's location to the package's goal location.
         b) If package is in a vehicle:
            - Distance from vehicle's location to package's goal location.
      4. Add 1 pick-up and 1 drop action (if package currently on ground), or just 1 drop action if already in vehicle.
         Because pick-up is needed only if package is not inside a vehicle.
      5. Sum these costs as an estimate of actions needed for the package.

    Additionally:
    - We use the capacity-predecessor chain to determine if vehicle's current capacity suffices for the package size.
      If not, that vehicle can't carry that package.
    - Heuristic sums costs for all packages independently, ignoring possible synergies.
    - This heuristic is not admissible but aims to be informative and fast.
    """

    def __init__(self, task):
        # Static facts and goals from task
        self.goals = task.goals
        static_facts = task.static

        # Parse roads into adjacency list (undirected)
        self.road_graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)

        # Parse capacity-predecessor relations into predecessor->successor map and build size order
        self.cap_pred = {}
        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, c1, c2 = get_parts(fact)
                self.cap_pred[c2] = c1  # c1 is predecessor of c2

        # Build map of size "levels": smallest size has level 0, its successor level 1, etc.
        # This helps quickly check if capacity c2 >= c1 by comparing levels.
        # We infer levels by walking backwards from larger sizes until no predecessor.
        self.size_levels = {}
        def get_level(size):
            if size in self.size_levels:
                return self.size_levels[size]
            if size not in self.cap_pred:
                self.size_levels[size] = 0
                return 0
            level = get_level(self.cap_pred[size]) + 1
            self.size_levels[size] = level
            return level

        # Precompute sizes levels for all sizes appearing in capacity or capacity-predecessor
        sizes_in_domain = set()
        for fact in static_facts:
            parts = get_parts(fact)
            for part in parts[1:]:
                if part.startswith('c'):  # likely a size object
                    sizes_in_domain.add(part)
        for s in sizes_in_domain:
            get_level(s)

        # Extract the goal location for each package (from goal facts)
        self.package_goals = {}
        for fact in self.goals:
            if match(fact, "at", "*", "*"):
                _, package, loc = get_parts(fact)
                self.package_goals[package] = loc

        # Cache for minimal distances between locations (BFS)
        # We'll compute on demand and store results for efficiency.
        self.dist_cache = {}

    def min_distance(self, start, goal):
        """Compute shortest distance between two locations in the road graph using BFS."""
        if (start, goal) in self.dist_cache:
            return self.dist_cache[(start, goal)]
        if start == goal:
            self.dist_cache[(start, goal)] = 0
            return 0
        visited = {start}
        frontier = [start]
        distance = 0
        while frontier:
            distance += 1
            next_frontier = []
            for loc in frontier:
                for nbr in self.road_graph.get(loc, []):
                    if nbr == goal:
                        self.dist_cache[(start, goal)] = distance
                        self.dist_cache[(goal, start)] = distance
                        return distance
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.append(nbr)
            frontier = next_frontier
        # If disconnected (should not happen in valid instances), return large number
        large_dist = 1000
        self.dist_cache[(start, goal)] = large_dist
        self.dist_cache[(goal, start)] = large_dist
        return large_dist

    def capacity_geq(self, cap1, cap2):
        """
        Check if capacity cap1 >= cap2.
        I.e. vehicle capacity cap1 can hold package size cap2.
        Uses size levels computed from capacity-predecessor chain.
        """
        level1 = self.size_levels.get(cap1, -1)
        level2 = self.size_levels.get(cap2, -1)
        return level1 >= level2

    def __call__(self, node):
        """
        Estimate number of actions to reach the goal from the current state.

        The heuristic sums, for each package not yet at its goal location, an estimate of:
          - Needed drive actions to pick or carry the package to goal.
          - Pick-up and drop actions.
        """

        state = node.state

        # Quick check: if all goals satisfied, heuristic is 0
        if self.goals <= state:
            return 0

        # Parse current package locations and if in vehicle
        package_positions = {}  # package -> ("in_vehicle", vehicle) or ("at_loc", location)
        # Parse vehicle locations and capacities
        vehicle_locations = {}  # vehicle -> location
        vehicle_capacities = {}  # vehicle -> current capacity size

        # Also parse packages' sizes by checking paired pick/drop actions in operators is complex,
        # so we approximate by capacity relations: This domain doesn't explicitly encode sizes on packages,
        # so we assume all packages require smallest size c0. We do not have explicit package sizes from state.
        # We'll assume all packages can be carried by any vehicle capacity c1 or above.
        # (In real scenario, package sizes might be in facts, but here missing.)
        # So we just check capacity constraints conservatively.

        # Extract capacities of vehicles
        for fact in state:
            if match(fact, "capacity", "*", "*"):
                _, vehicle, capacity = get_parts(fact)
                vehicle_capacities[vehicle] = capacity

        # Extract locations of vehicles and packages
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                # Determine if obj is vehicle or package by type heuristics
                # Vehicles start with "v" and packages with "p" in examples, but safer to use predicates
                # We'll consider vehicles are keys of vehicle_capacities; else packages.

                if obj in vehicle_capacities:
                    vehicle_locations[obj] = loc
                else:
                    package_positions[obj] = ("at_loc", loc)
            elif match(fact, "in", "*", "*"):
                # package in vehicle
                _, package, vehicle = get_parts(fact)
                package_positions[package] = ("in_vehicle", vehicle)

        total_estimate = 0

        # For each package with a goal, compute cost to reach goal
        for package, goal_loc in self.package_goals.items():
            # If already at goal location, cost is 0 for this package
            pos_info = package_positions.get(package, None)
            if pos_info is None:
                # Package not found in state? Possibly error or already delivered.
                # Treat as zero cost.
                continue

            # Check if package already at goal location (either at location or inside vehicle at goal)
            if pos_info[0] == "at_loc":
                if pos_info[1] == goal_loc:
                    continue  # no cost for this package
                # Package on ground, needs to be picked up and delivered
                package_loc = pos_info[1]

                # To pick up package, vehicle must have enough capacity
                # We'll find the vehicle with minimal cost to pick and deliver this package

                min_cost_for_package = None
                # Because package sizes unknown, assume smallest: c0
                pkg_size = "c0"

                # If no vehicle capacity info available (should not happen), estimate large cost
                if not vehicle_capacities:
                    # We guess one pick, one drop, and distance driving
                    dist = self.min_distance(package_loc, goal_loc)
                    min_cost_for_package = dist + 2  # pick + drop
                else:
                    # Try all vehicles that can carry package
                    for v, v_loc in vehicle_locations.items():
                        v_cap = vehicle_capacities.get(v, None)
                        if v_cap is None:
                            continue
                        if not self.capacity_geq(v_cap, pkg_size):
                            continue  # vehicle cannot carry package
                        # Estimate: drive vehicle from v_loc to package_loc (if not there)
                        drive_to_package = self.min_distance(v_loc, package_loc) if v_loc != package_loc else 0
                        # pick-up: 1 action
                        # drive vehicle from package_loc to goal_loc (may be same location)
                        drive_to_goal = self.min_distance(package_loc, goal_loc) if package_loc != goal_loc else 0
                        # drop: 1 action
                        cost = drive_to_package + 1 + drive_to_goal + 1
                        if (min_cost_for_package is None) or (cost < min_cost_for_package):
                            min_cost_for_package = cost

                    # If no suitable vehicle found, assign large cost
                    if min_cost_for_package is None:
                        min_cost_for_package = 1000

                total_estimate += min_cost_for_package
            else:
                # package is in a vehicle
                vehicle = pos_info[1]
                v_loc = vehicle_locations.get(vehicle, None)
                if v_loc is None:
                    # vehicle location unknown, assign large cost
                    total_estimate += 1000
                    continue

                # If vehicle at goal location, only drop needed
                if v_loc == goal_loc:
                    total_estimate += 1  # drop
                else:
                    # need to drive vehicle from current location to goal location + drop
                    dist = self.min_distance(v_loc, goal_loc)
                    total_estimate += dist + 1  # drive + drop

        return total_estimate
