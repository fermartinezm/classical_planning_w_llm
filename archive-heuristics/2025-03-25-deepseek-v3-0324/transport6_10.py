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
    - The distance packages need to be moved (using a simple graph distance)
    - Whether vehicles need to pick up/drop off packages
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - The road network is bidirectional (though this is handled via static facts)
    - Vehicles can carry multiple packages if capacity allows
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package from task.goals
    - Build a graph representation of the road network from static facts
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) Find its current location (either directly or via vehicle location)
        b) Find the closest vehicle that can carry it (considering capacity)
        c) Calculate distance from vehicle to package (if not already in vehicle)
        d) Calculate distance from package's location to goal location
        e) Add costs for pick-up and drop-off actions
    2. Sum all these costs to get the total heuristic estimate
    3. Special cases:
        - If package is already at goal: 0 cost
        - If package is in a vehicle not at goal: need to drop it
        - If no vehicle can reach package: use max distance as fallback
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build road graph from static facts
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)  # roads are bidirectional

        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

        # Extract capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state

        # If we're in a goal state, heuristic is 0
        if self.goals <= state:
            return 0

        # Track current locations of packages and vehicles
        current_locations = {}
        vehicle_capacities = {}
        package_locations = {}
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                if obj.startswith("p"):  # package
                    package_locations[obj] = loc
                elif obj.startswith("v"):  # vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                current_locations[package] = vehicle  # package is in vehicle
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            # Get current location of package (either direct or via vehicle)
            current_loc = current_locations.get(package)
            if current_loc is None:
                continue  # package not found in state (shouldn't happen)

            # If package is already at goal, no cost
            if current_loc == goal_loc and package in package_locations:
                continue

            # If package is in a vehicle, we need to drop it first
            if current_loc in vehicle_locations:
                vehicle = current_loc
                vehicle_loc = vehicle_locations[vehicle]
                # If vehicle is already at goal, just need to drop (1 action)
                if vehicle_loc == goal_loc:
                    total_cost += 1
                    continue
                # Otherwise, need to drive to goal and drop (distance + 1)
                distance = self._get_distance(vehicle_loc, goal_loc)
                total_cost += distance + 1
                continue

            # Package is not in a vehicle - find best vehicle to transport it
            min_package_cost = float('inf')

            for vehicle, vehicle_loc in vehicle_locations.items():
                # Check if vehicle has capacity to pick up package
                if vehicle not in vehicle_capacities:
                    continue  # vehicle has no capacity (shouldn't happen)

                # Calculate cost to pick up package with this vehicle
                # 1. Drive to package location (if not already there)
                if vehicle_loc != current_loc:
                    drive_cost = self._get_distance(vehicle_loc, current_loc)
                else:
                    drive_cost = 0

                # 2. Pick up package (1 action)
                # 3. Drive to goal location
                to_goal_cost = self._get_distance(current_loc, goal_loc)
                # 4. Drop package (1 action)
                total_vehicle_cost = drive_cost + 1 + to_goal_cost + 1

                if total_vehicle_cost < min_package_cost:
                    min_package_cost = total_vehicle_cost

            # If we found a vehicle, use that cost, otherwise use fallback
            if min_package_cost != float('inf'):
                total_cost += min_package_cost
            else:
                # Fallback: just use distance from current to goal location
                distance = self._get_distance(current_loc, goal_loc)
                total_cost += distance * 2  # approximate with 2 actions per step

        return total_cost

    def _get_distance(self, start, end):
        """Calculate shortest path distance between two locations using BFS."""
        if start == end:
            return 0

        visited = {start}
        queue = [(start, 0)]

        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.road_graph.get(current, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If no path found, return large number (shouldn't happen for valid problems)
        return 1000
