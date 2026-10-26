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
    - The distance packages need to be transported (using road network distances)
    - Whether packages need to be picked up or dropped
    - Vehicle capacities and their current locations

    # Assumptions:
    - The road network is bidirectional (if (road l1 l2) exists, movement is possible both ways)
    - Vehicles can carry multiple packages if they have sufficient capacity
    - Packages can only be transported by vehicles (not on their own)

    # Heuristic Initialization
    - Extract goal locations for each package
    - Build a graph representation of the road network for pathfinding
    - Extract capacity information for vehicles

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal location:
        a. If the package is in a vehicle:
            - Add cost for unloading it at current vehicle location
            - Then treat as if it's at that location
        b. Find the nearest vehicle that can carry the package (considering capacity)
        c. Calculate the distance from vehicle's current location to package location
        d. Calculate the distance from package location to goal location
        e. Add costs for:
            - Vehicle moving to package (drive actions)
            - Picking up package (pick-up action)
            - Vehicle moving to goal (drive actions)
            - Dropping package (drop action)
    2. Sum all these costs across all packages
    3. For overlapping routes (same vehicle transporting multiple packages), 
       the heuristic will naturally account for shared segments
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

        # Build road network graph
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

        # Extract capacity predecessor relationships
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def _bfs_distance(self, start, goal):
        """Calculate shortest path distance between two locations using BFS."""
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
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If we're in a goal state, heuristic should be 0
        if self.goals <= state:
            return 0

        # Track current locations of packages and vehicles
        current_locations = {}
        in_vehicle = {}
        vehicle_capacities = {}
        vehicles = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                if obj.startswith("v"):  # It's a vehicle
                    vehicles.add(obj)
            elif match(fact, "in", "*", "*"):
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif match(fact, "capacity", "*", "*"):
                vehicle, size = parts[1], parts[2]
                vehicle_capacities[vehicle] = size

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            # If package is already at goal, no cost
            if package in current_locations and current_locations[package] == goal_loc:
                continue

            # If package is in a vehicle, we need to drop it first
            if package in in_vehicle:
                vehicle = in_vehicle[package]
                current_loc = current_locations[vehicle]
                total_cost += 1  # drop action
                # Now treat as if package is at vehicle's current location
            else:
                current_loc = current_locations.get(package, None)
                if not current_loc:
                    continue  # package location unknown, skip

            # Find nearest vehicle that can carry this package
            min_vehicle_cost = float('inf')
            best_vehicle = None

            for vehicle in vehicles:
                # Check if vehicle has capacity to pick up package
                if vehicle not in vehicle_capacities:
                    continue  # vehicle at full capacity
                
                vehicle_loc = current_locations[vehicle]
                
                # Cost for vehicle to reach package and then go to goal
                dist_to_pkg = self._bfs_distance(vehicle_loc, current_loc)
                dist_to_goal = self._bfs_distance(current_loc, goal_loc)
                
                if dist_to_pkg == float('inf') or dist_to_goal == float('inf'):
                    continue  # no valid path
                
                vehicle_cost = dist_to_pkg + dist_to_goal + 2  # +2 for pick-up and drop
                
                if vehicle_cost < min_vehicle_cost:
                    min_vehicle_cost = vehicle_cost
                    best_vehicle = vehicle

            if best_vehicle is not None:
                total_cost += min_vehicle_cost
            else:
                # No vehicle can reach this package, return large number
                return float('inf')

        return total_cost
