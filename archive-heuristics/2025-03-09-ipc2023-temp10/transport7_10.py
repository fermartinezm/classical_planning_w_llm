from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transport7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the need to move vehicles to packages, pick up packages, move vehicles to goal locations, and drop packages.

    # Assumptions
    - Each package needs to be picked up by a vehicle, transported to its destination, and dropped.
    - Vehicles need to move to the packages and to the destination locations.
    - The heuristic does not consider capacity constraints or size constraints.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Extract the road network from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages and vehicles from the state.
    2. For each package, determine its goal location.
    3. If a package is not at its goal location, estimate the cost as follows:
        a. Find the closest vehicle to the package's current location.
        b. Estimate the cost to move the vehicle to the package (number of road actions).
        c. Add the cost of picking up the package (1 action).
        d. Estimate the cost to move the vehicle to the package's goal location (number of road actions).
        e. Add the cost of dropping off the package (1 action).
    4. Sum the estimated costs for all packages to obtain the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Extract road network information.
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal state from the given state.
        """
        state = node.state

        # Extract current locations of packages and vehicles.
        package_locations = {}
        vehicle_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                if any(match(fact, "at", obj, t) for t in ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l12', 'l13', 'l14', 'l15', 'l16', 'l17']):
                    if any(match(fact, "at", t, location) for t in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']):
                        vehicle_locations[obj] = location
                    elif any(match(fact, "at", t, location) for t in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17']):
                        package_locations[obj] = location

        total_cost = 0
        for package, goal_location in self.goal_locations.items():
            if package_locations.get(package) != goal_location:
                # Find the closest vehicle to the package.
                min_distance = float('inf')
                closest_vehicle = None
                for vehicle, vehicle_location in vehicle_locations.items():
                    distance = self.shortest_path_length(vehicle_location, package_locations.get(package), self.roads)
                    if distance < min_distance:
                        min_distance = distance
                        closest_vehicle = vehicle

                # Estimate the cost to move the vehicle to the package, pick up the package,
                # move the vehicle to the goal location, and drop off the package.
                vehicle_to_package_cost = self.shortest_path_length(vehicle_locations.get(closest_vehicle), package_locations.get(package), self.roads)
                package_to_goal_cost = self.shortest_path_length(package_locations.get(package), goal_location, self.roads)

                total_cost += vehicle_to_package_cost + 1 + package_to_goal_cost + 1

        return total_cost

    def shortest_path_length(self, start, end, roads):
        """
        Compute the shortest path length between two locations using a simple breadth-first search.
        """
        if start == end:
            return 0

        queue = [(start, 0)]  # (location, distance)
        visited = {start}

        while queue:
            location, distance = queue.pop(0)

            # Check neighbors
            for l1, l2 in roads:
                if l1 == location and l2 == end:
                    return distance + 1
                if l2 == location and l1 == end:
                    return distance + 1

            for l1, l2 in roads:
                if l1 == location and l2 not in visited:
                    queue.append((l2, distance + 1))
                    visited.add(l2)
                if l2 == location and l1 not in visited:
                    queue.append((l1, distance + 1))
                    visited.add(l1)

        return float('inf')  # No path found
