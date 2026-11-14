from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque


def get_parts(fact):
    """Remove parentheses and split fact into parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern, where patterns may include wildcards '*'.
    
    Args:
        fact: String representing a fact, e.g. "(at p1 l1)".
        args: Pattern elements, e.g. "at", "*", "l1".
    
    Returns:
        True if fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.
    
    Summary
    -------
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations.
    It accounts for:
     - distances vehicles must drive to pick up and deliver packages,
     - capacity constraints of vehicles limiting how many packages can be carried concurrently,
     - the necessity to pick up and drop each package at least once.
    
    Assumptions
    -----------
    - Packages can only be moved by vehicles that transport them.
    - Each package must be picked up at its current location and dropped at its goal location.
    - Vehicle capacity decreases by one size level for each package carried, represented by capacity-predecessor facts.
    - Road connections are bidirectional as given by static facts "road".
    - Driving between any two locations counts as the number of steps equal to the shortest path length in the road graph.
    - Picking up or dropping a package costs one action each.
    - The heuristic is not necessarily admissible but aims to be informative and efficient.
    
    Heuristic Initialization
    ------------------------
    - Extract the goal locations for all packages from the goal facts.
    - Extract all road connections and build an undirected graph of locations.
    - Extract capacity-predecessor relations to understand vehicle capacities and how they reduce when picking up packages.
    - Build a lookup for capacity levels (integers) so that capacity can be efficiently tracked.
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Parse the current state to determine:
       - Where each package currently is (at location or inside which vehicle).
       - Where each vehicle currently is.
       - The current capacity of each vehicle.
    2. For each package not at its goal location:
       a. If the package is already inside a vehicle, estimate steps for that vehicle to drive to the package's goal location plus one drop action.
          (Ignore intermediate capacity changes for heuristic simplicity.)
       b. If the package is at a location:
          - Estimate minimal driving cost for some vehicle to drive from its current location to the package's location,
            then from the package's location to its goal location.
          - Add pick-up and drop costs.
          - Consider vehicle capacity: only vehicles with sufficient capacity can pick up the package.
    3. Since exact assignment and route planning is complex, approximate by:
       - For each package on the ground, assign the minimal cost vehicle that can currently pick it up.
       - For packages inside vehicles, consider only delivery cost.
    4. Sum the minimal estimated costs (drive + pick-up + drop) for all packages not yet at their goal locations.
    5. Return the summed estimate as the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal condition facts, e.g. (at p1 l2)
        self.static = task.static  # static facts such as road and capacity-predecessor

        # Extract all locations from static roads
        self.locations = set()
        self.road_map = {}  # undirected graph adjacency list: location -> set of adjacent locations
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_map.setdefault(l1, set()).add(l2)
                self.road_map.setdefault(l2, set()).add(l1)

        # Extract capacity-predecessor relations and assign numeric levels for comparing vehicle capacities
        self.capacity_levels = {}
        # We'll build levels gradually: c0 < c1 < c2 < ...
        # Find all capacity objects:
        capacity_objs = set()
        capacity_pred_pairs = set()
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, c1, c2 = get_parts(fact)
                capacity_pred_pairs.add((c1, c2))
                capacity_objs.add(c1)
                capacity_objs.add(c2)

        # Find minimal elements (no predecessor) to start levels
        successors_of = {}
        predecessors_of = {}
        for c1, c2 in capacity_pred_pairs:
            successors_of.setdefault(c1, set()).add(c2)
            predecessors_of.setdefault(c2, set()).add(c1)
        minimal_caps = capacity_objs - set(predecessors_of.keys())

        # Assign levels by topological order
        level = 0
        to_visit = list(minimal_caps)
        assigned_levels = {}
        while to_visit:
            next_level_nodes = []
            for c in to_visit:
                if c not in assigned_levels:
                    assigned_levels[c] = level
                    # enqueue successors if all their predecessors are assigned
                    for succ in successors_of.get(c, []):
                        preds = predecessors_of.get(succ, set())
                        if preds and all(p in assigned_levels for p in preds):
                            next_level_nodes.append(succ)
            to_visit = next_level_nodes
            level += 1
        self.capacity_levels = assigned_levels  # e.g. {'c0':0,'c1':1,'c2':2,...}

        # Extract goal locations per package
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, loc = get_parts(goal)
                self.goal_locations[package] = loc

        # Action costs in the heuristic for clarity
        self.PICKUP_COST = 1
        self.DROP_COST = 1
        self.DRIVE_COST_PER_EDGE = 1

        # Cache shortest paths between locations for efficient repeated queries
        # We'll compute all shortest paths once with BFS from each location (since graph is unweighted)
        self.shortest_paths_cache = {}

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between two locations in the road graph using BFS."""
        if start == goal:
            return 0
        if (start, goal) in self.shortest_paths_cache:
            return self.shortest_paths_cache[(start, goal)]
        if (goal, start) in self.shortest_paths_cache:
            return self.shortest_paths_cache[(goal, start)]

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.road_map.get(current, []):
                if neighbor == goal:
                    self.shortest_paths_cache[(start, goal)] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # If no path found (disconnected), return a large number to discourage
        large_dist = 1000000
        self.shortest_paths_cache[(start, goal)] = large_dist
        return large_dist

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is 0 by definition
        if self.goals <= state:
            return 0

        # Parse current state info:

        # 1. Package locations: either (at package location) or (in package vehicle)
        package_locations = {}  # package -> location or vehicle
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, pkg, loc = get_parts(fact)
                if pkg in self.goal_locations:
                    package_locations[pkg] = loc
            elif match(fact, "in", "*", "*"):
                _, pkg, vehicle = get_parts(fact)
                if pkg in self.goal_locations:
                    package_locations[pkg] = vehicle

        # 2. Vehicle locations: (at vehicle location)
        vehicle_locations = {}  # vehicle -> location
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                if obj.startswith("v"):  # vehicle names start with "v"
                    vehicle_locations[obj] = loc

        # 3. Current capacity of each vehicle (capacity fact present)
        vehicle_capacities = {}  # vehicle -> capacity object string
        for fact in state:
            if match(fact, "capacity", "*", "*"):
                _, v, c = get_parts(fact)
                vehicle_capacities[v] = c

        # For capacity, lookup numeric level to compare capacity size
        def can_carry(vehicle, packages_on_vehicle_count):
            """
            Checks if vehicle still has capacity left for 1 more package.
            Capacity reduces by one level per package onboard.
            """
            if vehicle not in vehicle_capacities:
                return False
            current_capacity_level = self.capacity_levels.get(vehicle_capacities[vehicle], -1)
            # Number of packages currently in vehicle:
            # Since capacity decreases by one level per package according to capacity-predecessor,
            # we estimate capacity by difference between initial capacity and current
            # But we do not know initial capacity here, so approximate by:
            # count packages currently in vehicle:
            # We'll count packages in vehicle, then see if capacity level >= number of onboard packages + 1
            # To simplify, since only one predecessor step is known per package, assume capacity level counts packages
            return current_capacity_level > packages_on_vehicle_count

        # Precompute packages currently inside each vehicle for capacity counting
        packages_in_vehicle_count = {}
        for fact in state:
            if match(fact, "in", "*", "*"):
                _, pkg, vehicle = get_parts(fact)
                packages_in_vehicle_count[vehicle] = packages_in_vehicle_count.get(vehicle, 0) + 1

        # For vehicles without packages, default count zero
        for v in vehicle_locations:
            if v not in packages_in_vehicle_count:
                packages_in_vehicle_count[v] = 0

        # 4. Identify packages that are not yet at their goal locations
        # Packages already at goal: if package_locations[pkg] == goal_locations[pkg]
        # If inside a vehicle, need delivery step to goal location

        total_estimated_cost = 0

        for pkg, goal_loc in self.goal_locations.items():
            if pkg not in package_locations:
                # Should not happen in consistent states, but ignore if so
                continue
            curr_pos = package_locations[pkg]
            if curr_pos == goal_loc:
                # Already at goal, no cost
                continue

            # Case 1: package inside a vehicle
            if curr_pos.startswith("v"):
                vehicle = curr_pos
                if vehicle not in vehicle_locations:
                    # Unknown vehicle position, pessimistically assign high cost
                    cost = 10
                else:
                    vehicle_loc = vehicle_locations[vehicle]
                    drive_cost = self.shortest_path_length(vehicle_loc, goal_loc) * self.DRIVE_COST_PER_EDGE
                    cost = drive_cost + self.DROP_COST
                total_estimated_cost += cost

            else:
                # package on ground at location curr_pos

                # Find all vehicles that can currently carry this package (have enough capacity)
                candidate_vehicles = []
                for vehicle, vehicle_loc in vehicle_locations.items():
                    # Check if capacity allows pickup
                    if can_carry(vehicle, packages_in_vehicle_count.get(vehicle, 0)):
                        candidate_vehicles.append(vehicle)

                if not candidate_vehicles:
                    # No vehicle can carry now (should be rare or impossible)
                    # Assign a large cost to reflect hard to move
                    total_estimated_cost += 100
                    continue

                # For each candidate vehicle, compute cost:
                # drive from vehicle_loc to package_loc + pickup + drive package_loc to goal_loc + drop
                best_cost = None
                for v in candidate_vehicles:
                    vehicle_loc = vehicle_locations[v]

                    dist_to_package = self.shortest_path_length(vehicle_loc, curr_pos) * self.DRIVE_COST_PER_EDGE
                    dist_to_goal = self.shortest_path_length(curr_pos, goal_loc) * self.DRIVE_COST_PER_EDGE

                    cost = dist_to_package + self.PICKUP_COST + dist_to_goal + self.DROP_COST

                    if best_cost is None or cost < best_cost:
                        best_cost = cost

                total_estimated_cost += best_cost if best_cost is not None else 100

        return total_estimated_cost
