from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(road l1 l2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    to their goal locations. It considers:
    - The distance packages need to be moved (using a simple count of required road segments)
    - The capacity constraints of vehicles
    - The need to pick up and drop off packages

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - Vehicles can carry multiple packages if their capacity allows
    - The road network is bidirectional (though this is handled by the static facts)
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package from task.goals
    - Extract road network from static facts
    - Extract capacity predecessor relationships from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a. Find its current location (either directly or via the vehicle it's in)
        b. Find the closest vehicle that can carry it (considering capacity)
        c. Calculate the distance from vehicle to package (if not already in vehicle)
        d. Calculate the distance from package's location to goal location
        e. Add costs for pick-up and drop-off actions
    2. Sum all these costs across all packages
    3. Add a penalty for vehicles that need to reposition to pick up packages
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build road network graph
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)

        # Store capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current locations of packages and vehicles
        current_locations = {}
        vehicle_capacities = {}
        packages_in_vehicles = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                packages_in_vehicles.setdefault(vehicle, set()).add(package)
            elif parts[0] == "capacity":
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            # Skip if package is already at goal
            if package in current_locations and current_locations[package] == goal_loc:
                continue

            # Find current location of package (either directly or via vehicle)
            if package in current_locations:
                # Package is at a location
                current_loc = current_locations[package]
                in_vehicle = False
                carrying_vehicle = None
            else:
                # Package is in a vehicle
                carrying_vehicle = next(
                    (v for v, pkgs in packages_in_vehicles.items() if package in pkgs),
                    None,
                )
                if carrying_vehicle is None:
                    # Package location unknown - this shouldn't happen in valid states
                    continue
                current_loc = current_locations[carrying_vehicle]
                in_vehicle = True

            # Find all vehicles that could carry this package
            suitable_vehicles = []
            for vehicle, capacity in vehicle_capacities.items():
                # Check if vehicle has capacity to carry this package
                # (simplified - assumes any capacity > c0 can carry)
                if capacity != "c0":
                    suitable_vehicles.append(vehicle)

            # Calculate minimal distance to move package to goal
            min_package_cost = float("inf")

            for vehicle in suitable_vehicles:
                vehicle_loc = current_locations[vehicle]

                # If package is already in this vehicle
                if vehicle == carrying_vehicle:
                    drive_cost = self._estimate_distance(vehicle_loc, goal_loc)
                    total_cost += drive_cost + 1  # +1 for drop action
                    break

                # If package is not in this vehicle
                else:
                    # Cost to get vehicle to package
                    cost1 = self._estimate_distance(vehicle_loc, current_loc)
                    # Cost to drive package to goal
                    cost2 = self._estimate_distance(current_loc, goal_loc)
                    total_cost += cost1 + cost2 + 2  # +2 for pick and drop actions

            else:  # No suitable vehicles found
                # Just estimate distance from current to goal
                total_cost += self._estimate_distance(current_loc, goal_loc) + 1

        return total_cost

    def _estimate_distance(self, loc1, loc2):
        """
        Estimate the minimal number of drive actions needed to get from loc1 to loc2.
        Uses a simple BFS since we don't need the actual path.
        """
        if loc1 == loc2:
            return 0

        visited = {loc1}
        queue = [(loc1, 0)]

        while queue:
            current, distance = queue.pop(0)
            for neighbor in self.road_graph.get(current, []):
                if neighbor == loc2:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        # If no path found (shouldn't happen in valid problems)
        return float("inf")
