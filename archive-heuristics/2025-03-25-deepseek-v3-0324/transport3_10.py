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
    - The distance packages need to be moved (using road network distances)
    - Whether vehicles need to pick up or drop packages
    - Vehicle capacity constraints

    # Assumptions:
    - Packages can be transported by any vehicle with sufficient capacity
    - Vehicles can be routed optimally through the road network
    - Each pick-up and drop action counts as 1 unit
    - Each drive action between locations counts as 1 unit

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a road network graph for distance calculations
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a) If package is in a vehicle:
            - Calculate distance from vehicle's current location to package's goal
            - Add 1 for drop action
        b) If package is at a location:
            - Find nearest vehicle with sufficient capacity
            - Calculate distance from vehicle to package
            - Add 1 for pick-up action
            - Calculate distance from package location to goal
            - Add 1 for drop action
    2. Sum all required actions across all packages
    3. Add any necessary empty vehicle movements (if vehicles need to reposition)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and parts[1].startswith("p"):
                self.goal_locations[parts[1]] = parts[2]

        # Build road network graph
        self.road_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)

        # Extract capacity information
        self.capacity_predecessors = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "capacity-predecessor":
                self.capacity_predecessors[parts[2]] = parts[1]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current locations of packages and vehicles
        current_locations = {}
        in_vehicle = {}
        vehicle_capacities = {}
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in":
                pkg, vehicle = parts[1], parts[2]
                in_vehicle[pkg] = vehicle
            elif parts[0] == "capacity":
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size

        total_cost = 0

        # BFS function to find shortest path between locations
        def shortest_path(start, end):
            if start == end:
                return 0
            visited = set()
            queue = [(start, 0)]
            while queue:
                loc, dist = queue.pop(0)
                if loc == end:
                    return dist
                if loc in visited:
                    continue
                visited.add(loc)
                for neighbor in self.road_graph.get(loc, []):
                    queue.append((neighbor, dist + 1))
            return float('inf')  # No path found

        for package, goal_loc in self.goal_locations.items():
            if package in in_vehicle:
                # Package is in a vehicle - need to drop it at goal
                vehicle = in_vehicle[package]
                vehicle_loc = current_locations.get(vehicle, None)
                if vehicle_loc:
                    path_cost = shortest_path(vehicle_loc, goal_loc)
                    total_cost += path_cost + 1  # +1 for drop action
            else:
                # Package is at a location - need to pick it up first
                pkg_loc = current_locations.get(package, None)
                if pkg_loc and pkg_loc != goal_loc:
                    # Find nearest vehicle with capacity
                    min_vehicle_cost = float('inf')
                    for vehicle, capacity in vehicle_capacities.items():
                        if capacity != "c0":  # Vehicle has some capacity
                            vehicle_loc = current_locations.get(vehicle, None)
                            if vehicle_loc:
                                # Cost to get to package
                                to_pkg = shortest_path(vehicle_loc, pkg_loc)
                                # Cost to get from package to goal
                                to_goal = shortest_path(pkg_loc, goal_loc)
                                total_vehicle_cost = to_pkg + to_goal + 2  # +1 pick, +1 drop
                                if total_vehicle_cost < min_vehicle_cost:
                                    min_vehicle_cost = total_vehicle_cost
                    if min_vehicle_cost != float('inf'):
                        total_cost += min_vehicle_cost

        return total_cost
