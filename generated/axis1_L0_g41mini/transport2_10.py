from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern with wildcards (*).

    Args:
        fact (str): PDDL fact string e.g. "(at p1 l1)".
        *args: sequence of strings/patterns to match fact components.

    Returns:
        bool: True if all components match the given patterns.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Summary
    -------
    Estimates the number of actions needed to deliver all packages to their goal
    locations. The heuristic accounts for vehicle travel on roads, picking up and
    dropping packages considering vehicle capacities.

    Assumptions
    -----------
    - Packages start at some locations or inside vehicles.
    - Vehicles have capacities represented by size levels with predecessor relations.
    - Vehicles move only on directly connected roads.
    - Capacity is properly adjusted upon pick-up and drop actions.
    - Each pick-up and drop corresponds to one package moved, and each drive moves a vehicle to an adjacent location.
    - The heuristic is not guaranteed admissible but aims to guide search efficiently.
    - Road connections are bidirectional as given in static facts.

    Heuristic Initialization
    --------------------------
    - Extract the road network as an adjacency map for quick shortest path calculations.
    - Extract capacity-predecessor relations to determine capacity levels and package size fitting.
    - Extract the goal conditions to identify the desired final package locations.
    - Store all static information necessary to quickly query road reachability and distances.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    1. Identify for each package:
       a) Its current location or if it is inside a vehicle (along with that vehicle's location).
       b) Its goal location from the goal conditions.
    2. Compute the shortest road distance (number of drives) between:
       - Vehicle's current location and package pickup location (if package is not inside any vehicle).
       - Package current location to its goal location (if package is not inside a vehicle).
       - Vehicle's current location to package goal location (if package is inside that vehicle).
    3. For packages not yet at goal, estimate:
       - Number of required picks (1 per package).
       - Number of required drops (1 per package).
       - Number of drives needed to pick and deliver the package:
         * Travel from vehicle to package (if necessary).
         * Travel from package pickup to delivery location.
         * Consider that vehicles must carry packages respecting capacities.
    4. Approximate the load of packages into vehicles assuming no complex rearranging; focus on minimum needed moves.
    5. Sum the estimated number of drive, pick-up, and drop actions over all packages.
    6. Return the total as heuristic value; zero if all goals are met.

    This heuristic balances simplicity and informativeness to effectively guide a greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize heuristic with static information:

        - Build adjacency list for roads.
        - Build a map for capacity predecessors (size hierarchy).
        - Collect goal package locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build road adjacency map as a dict: location -> set of connected locations
        self.roads = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.roads.setdefault(l1, set()).add(l2)

        # Build capacity predecessor chain as a dict: s2 -> s1 such that capacity-predecessor s1 s2
        # This allows to determine "smaller" capacity from a "larger"
        self.capacity_pred = {}
        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

        # Extract goal package locations from goals
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.package_goals[package] = location

        # Cache known locations for shortest path computations
        self.locations = set(self.roads.keys())
        for locs in self.roads.values():
            self.locations.update(locs)

    def shortest_distance(self, start, goal):
        """
        Compute shortest distance between two locations on the road graph using BFS.

        Args:
            start (str): Start location.
            goal (str): Goal location.

        Returns:
            int: Shortest number of drives from start to goal (0 if same), or large number if unreachable.
        """
        if start == goal:
            return 0
        if start not in self.locations or goal not in self.locations:
            return 10000  # Arbitrarily large to indicate no path.

        from collections import deque

        visited = set([start])
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.roads.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 10000

    def get_vehicle_capacity_level(self, vehicle, state):
        """
        Determine the capacity size of a vehicle from the current state.

        Args:
            vehicle (str): Vehicle identifier.
            state (frozenset): Current state facts.

        Returns:
            str or None: The capacity size predicate string for the vehicle (e.g. 'c1'), or None if unknown.
        """
        # capacity facts have format: (capacity v1 c1)
        for fact in state:
            if match(fact, "capacity", vehicle, "*"):
                return get_parts(fact)[2]
        return None

    def can_fit_package(self, vehicle_capacity, package_size, capacity_pred):
        """
        Check if package size fits into vehicle capacity size considering capacity-predecessor relation.

        Args:
            vehicle_capacity (str): current capacity size of vehicle
            package_size (str): size of package to fit
            capacity_pred (dict): capacity predecessor mapping s2 -> s1

        Returns:
            bool: True if package fits, False otherwise.
        """
        # The domain uses capacity-predecessor to represent allowed capacities after pickup/drop.
        # Since package sizes are not explicitly given in state, assume packages reduce capacity by one step.
        # For heuristic, assume all packages fit; this can be refined if package sizes were known.
        # Here we just assume package fits for heuristic simplification.
        return True

    def __call__(self, node):
        """
        Compute heuristic estimate of number of actions (drive, pick-up, drop) remaining.

        Args:
            node: Search node with attribute 'state' representing current facts.

        Returns:
            int: Estimated cost to reach goal.
        """
        state = node.state

        # If goal already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Extract package current locations or vehicles they are inside.
        package_locations = {}
        # Extract vehicle current locations.
        vehicle_locations = {}
        # Extract vehicle capacities.
        vehicle_capacities = {}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                obj = parts[1]
                loc = parts[2]
                if obj.startswith("v"):  # vehicle
                    vehicle_locations[obj] = loc
                else:  # package or other locatable object
                    package_locations[obj] = loc
            elif pred == "in":
                pkg = parts[1]
                veh = parts[2]
                package_locations[pkg] = veh  # package is inside vehicle
            elif pred == "capacity":
                veh = parts[1]
                cap = parts[2]
                vehicle_capacities[veh] = cap

        # Initialize total cost
        total_cost = 0

        # Cost constants per action
        COST_DRIVE = 1
        COST_PICKUP = 1
        COST_DROP = 1

        # For each package, estimate actions needed to deliver
        for package, goal_loc in self.package_goals.items():
            # Current package location or vehicle
            current_loc = package_locations.get(package)
            if current_loc is None:
                # Package location unknown, assume zero to be safe
                continue

            # Check if package already at goal location:
            # A package at the goal location must be (at package goal_location)
            # But package might be inside a vehicle at goal location, which is not goal.
            if current_loc == goal_loc:
                # Package is at goal location (on ground) => done, no cost
                continue

            # Determine if package is inside a vehicle or at a location
            if current_loc.startswith("v"):  # package inside vehicle
                veh = current_loc
                veh_loc = vehicle_locations.get(veh)
                if veh_loc is None:
                    # Vehicle location unknown, skip package cost estimate
                    continue

                # If vehicle is at package goal location, only drop needed
                if veh_loc == goal_loc:
                    total_cost += COST_DROP
                    # We assume drop also releases a capacity slot, but we do not model capacity explicitly here
                    continue

                # Otherwise:
                # 1. Drive vehicle from current vehicle location to goal location
                dist_drive = self.shortest_distance(veh_loc, goal_loc)
                total_cost += dist_drive * COST_DRIVE
                # 2. Drop package at goal
                total_cost += COST_DROP
                continue

            else:
                # package at a location (not in vehicle)
                pkg_loc = current_loc

                # If package already at goal, cost 0 (already checked above)

                # Try to find a vehicle that can pick and deliver this package
                # For heuristic we assume there is at least one vehicle (simplification)
                # We'll estimate minimal drive distances considering closest vehicle

                # Find vehicles that can pick this package (we don't have package size in the state)
                # So assume all vehicles can carry it for heuristic
                min_cost = 100000

                for veh, veh_loc in vehicle_locations.items():
                    # Drives needed: from vehicle to package + from package to goal
                    dist_to_pkg = self.shortest_distance(veh_loc, pkg_loc)
                    dist_pkg_to_goal = self.shortest_distance(pkg_loc, goal_loc)

                    # pick + drop actions assumed 1 each
                    cost_estimate = (
                        dist_to_pkg * COST_DRIVE
                        + COST_PICKUP
                        + dist_pkg_to_goal * COST_DRIVE
                        + COST_DROP
                    )

                    if cost_estimate < min_cost:
                        min_cost = cost_estimate

                # Add minimal estimated cost for this package
                total_cost += min_cost

        return total_cost
