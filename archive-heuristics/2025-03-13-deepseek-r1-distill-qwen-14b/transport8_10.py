from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from itertools import product
import heapq

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their goal locations. It considers the shortest path between locations and vehicle capacities.

    # Assumptions:
    - Packages must be transported from their current location to a goal location.
    - Vehicles can move along roads and carry packages based on their capacity.
    - The heuristic assumes that vehicles can move optimally between locations.

    # Heuristic Initialization
    - Extracts static facts including road connections, package goals, vehicle capacities, and capacity predecessors.
    - Constructs a graph representation of the road network for efficient shortest path computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, determine its current location and whether it's being carried by a vehicle.
    2. If the package is in a vehicle, find the vehicle's current location.
    3. Compute the shortest path from the vehicle's location to the package's goal location.
    4. If the package is not in a vehicle, find the nearest vehicle that can pick it up.
    5. Sum the number of drive actions needed for all packages, considering vehicle capacities and movement constraints.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts and goal information."""
        super().__init__(task)
        self.goals = task.goals
        self.static = task.static

        # Extract road connections
        self.roads = {}
        for fact in self.static:
            if fact.startswith("(road"):
                parts = fact[1:-1].split()
                l1, l2 = parts[1], parts[2]
                if l1 not in self.roads:
                    self.roads[l1] = []
                self.roads[l1].append(l2)
                if l2 not in self.roads:
                    self.roads[l2] = []
                self.roads[l2].append(l1)

        # Extract package goals
        self.package_goals = {}
        for goal in self.goals:
            if goal.startswith("(at"):
                package = goal.split()[1]
                loc = goal.split()[2]
                self.package_goals[package] = loc

        # Extract vehicle capacities and capacity predecessors
        self.vehicle_capacities = {}
        self.capacity_graph = {}
        for fact in self.static:
            if fact.startswith("(capacity"):
                vehicle, size = fact[1:-1].split()[1], fact[1:-1].split()[2]
                self.vehicle_capacities[vehicle] = size
            elif fact.startswith("(capacity-predecessor"):
                s1, s2 = fact[1:-1].split()[1], fact[1:-1].split()[2]
                if s1 not in self.capacity_graph:
                    self.capacity_graph[s1] = []
                self.capacity_graph[s1].append(s2)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_locations = {}
        packages = []
        vehicles = []

        # Extract current locations of packages and vehicles
        for fact in state:
            if fact.startswith("(at"):
                obj, loc = fact[1:-1].split()[1], fact[1:-1].split()[2]
                if obj == "package":
                    packages.append((obj, loc))
                else:
                    vehicles.append((obj, loc))
            elif fact.startswith("(in"):
                package, vehicle = fact[1:-1].split()[1], fact[1:-1].split()[2]
                current_locations[package] = vehicle

        # Function to find shortest path between locations
        def shortest_path(start, end):
            if start == end:
                return 0
            visited = {}
            heap = [(0, start)]
            while heap:
                dist, node = heapq.heappop(heap)
                if node in visited:
                    continue
                visited[node] = dist
                for neighbor in self.roads.get(node, []):
                    if neighbor not in visited:
                        heapq.heappush(heap, (dist + 1, neighbor))
            return visited.get(end, float('inf'))

        total_actions = 0

        # Process each package
        for package, goal in self.package_goals.items():
            if package not in current_locations:
                # Package is at a location, not in a vehicle
                package_loc = None
                for fact in state:
                    if fact.startswith("(at") and fact[1:-1].split()[1] == package:
                        package_loc = fact[1:-1].split()[2]
                        break
                if package_loc is None:
                    continue  # Package not present in state

                # Find nearest vehicle
                nearest_vehicle = None
                min_distance = float('inf')
                for vehicle, vehicle_loc in vehicles:
                    distance = shortest_path(package_loc, vehicle_loc)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_vehicle = vehicle_loc

                # Vehicle needs to drive to package location
                drive_actions = shortest_path(nearest_vehicle, package_loc)
                total_actions += drive_actions

                # Vehicle picks up package
                total_actions += 1

                # Vehicle drives to goal location
                drive_actions = shortest_path(package_loc, goal)
                total_actions += drive_actions

            else:
                # Package is in a vehicle
                vehicle = current_locations[package]
                vehicle_loc = None
                for fact in state:
                    if fact.startswith("(at") and fact[1:-1].split()[1] == vehicle:
                        vehicle_loc = fact[1:-1].split()[2]
                        break
                if vehicle_loc is None:
                    continue  # Vehicle not present in state

                # Compute drive actions needed to reach goal
                drive_actions = shortest_path(vehicle_loc, goal)
                total_actions += drive_actions

        return total_actions
