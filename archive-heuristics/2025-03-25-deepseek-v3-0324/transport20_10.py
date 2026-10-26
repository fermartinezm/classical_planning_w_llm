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
    - Whether packages need to be picked up or dropped
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - The road network is bidirectional (though this is encoded in the static facts)
    - Vehicles can carry multiple packages if they have sufficient capacity
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal locations for each package from task.goals
    - Build a road graph from static facts for path distance estimation
    - Extract capacity information from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a. If the package is in a vehicle:
            - Add cost for dropping it at current vehicle location
            - Treat it as being at the vehicle's location
        b. Find the shortest path distance from current location to goal
        c. Estimate required actions:
            - Drive actions: equal to path distance
            - Pick-up action if not already in a vehicle
            - Drop action if goal location isn't current vehicle location
    2. For vehicles that need to move to pick up packages:
        a. Estimate distance from current location to package location
        b. Add corresponding drive actions
    3. Sum all estimated actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build road graph
        self.road_graph = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = set()
                if l2 not in self.road_graph:
                    self.road_graph[l2] = set()
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1)

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

    def _bfs_distance(self, start, goal):
        """Compute shortest path distance between two locations using BFS."""
        if start == goal:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current == goal:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.road_graph.get(current, []):
                queue.append((neighbor, dist + 1))
        return float('inf')  # No path exists

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state

        # If all goals are satisfied, return 0
        if self.goals <= state:
            return 0

        # Track package locations and vehicle locations
        package_locations = {}
        vehicle_locations = {}
        in_vehicle = {}
        vehicle_capacities = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):  # Package
                    package_locations[obj] = loc
                else:  # Vehicle
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size

        total_cost = 0

        # Process each package that's not at its goal
        for package, goal_loc in self.goal_locations.items():
            if package in in_vehicle:
                # Package is in a vehicle - need to drop it first
                vehicle = in_vehicle[package]
                current_loc = vehicle_locations[vehicle]
                total_cost += 1  # Drop action
            else:
                current_loc = package_locations.get(package, None)

            if current_loc != goal_loc:
                # Estimate drive distance from current to goal location
                distance = self._bfs_distance(current_loc, goal_loc)
                if distance == float('inf'):
                    return float('inf')  # Unreachable goal
                total_cost += distance

                # If package isn't in a vehicle, add pick-up cost
                if package not in in_vehicle:
                    total_cost += 1  # Pick-up action

        return total_cost
