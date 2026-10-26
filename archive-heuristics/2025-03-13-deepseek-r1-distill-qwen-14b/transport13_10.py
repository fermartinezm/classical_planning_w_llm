from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    to their target locations. It considers the current state of each package,
    whether it is in a vehicle or on the ground, and calculates the necessary
    actions to move the package to its destination.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - Each vehicle has a specific capacity that determines how many packages it can carry.
    - The goal is to have all packages at their respective target locations.

    # Heuristic Initialization
    - Extracts static facts including road connections and vehicle capacities.
    - Maps each location to its connected locations for efficient lookup.

    # Step-by-Step Thinking for Computing Heuristic Value
    1. For each package, determine its current location and whether it is inside a vehicle.
    2. If the package is already at its target location, no actions are needed.
    3. If the package is inside a vehicle, find the vehicle's current location.
    4. Calculate the number of drive actions required to move the vehicle to the package's location.
    5. If the vehicle is not at the package's location, add actions to drive the vehicle there.
    6. If the package is not in a vehicle, add actions to pick up the package.
    7. Calculate the number of drive actions required to move the vehicle to the target location.
    8. Add actions to drop off the package at the target location.
    9. Sum the actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and preparing data structures."""
        super().__init__(task)
        self.static_facts = task.static
        self.road_connections = self._extract_road_connections()
        self.capacity_info = self._extract_capacity_info()
        self.vehicle_locations = {}

    def _extract_road_connections(self):
        """Extract and store all road connections from static facts."""
        roads = {}
        for fact in self.static_facts:
            if fnmatch(fact, '(road * *)'):
                l1, l2 = fact[6:-1].split()
                if l1 not in roads:
                    roads[l1] = []
                roads[l1].append(l2)
                if l2 not in roads:
                    roads[l2] = []
                roads[l2].append(l1)
        return roads

    def _extract_capacity_info(self):
        """Extract and store vehicle capacities and capacity predecessor relationships."""
        capacities = {}
        for fact in self.static_facts:
            if fnmatch(fact, '(capacity * *)'):
                vehicle, size = fact[9:-1].split()
                capacities[vehicle] = size
            elif fnmatch(fact, '(capacity-predecessor * *)'):
                s1, s2 = fact[18:-1].split()
                if s1 not in capacities:
                    capacities[s1] = s2
        return capacities

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_actions = 0

        # Extract current locations of all objects
        current_locations = {}
        for fact in state:
            if fnmatch(fact, '(at * * )'):
                obj, loc = fact[4:-1].split()
                current_locations[obj] = loc
            elif fnmatch(fact, '(in * * )'):
                package, vehicle = fact[4:-1].split()
                current_locations[package] = vehicle

        # Extract goal locations for each package
        goal_locations = {}
        for goal in self.task.goals:
            if fnmatch(goal, '(at * * )'):
                package, loc = goal[4:-1].split()
                goal_locations[package] = loc

        for package, target in goal_locations.items():
            current_loc = current_locations.get(package, None)
            if current_loc == target:
                continue  # Package is already at the goal

            if current_loc is None:
                continue  # Package not found in state (should not happen)

            # Check if the package is in a vehicle
            in_vehicle = current_loc not in self.road_connections
            if in_vehicle:
                # Find the vehicle's current location
                vehicle = current_loc
                vehicle_loc = current_locations.get(vehicle, None)
                if vehicle_loc is None:
                    continue  # Vehicle not found (should not happen)
                current_loc = vehicle_loc

            # Calculate the number of drive actions needed
            from_loc = current_loc
            to_loc = target if not in_vehicle else self._get_vehicle_dropoff_location(package, target)

            path = self._find_shortest_path(from_loc, to_loc)
            if path is None:
                continue  # No path exists (should not happen in solvable states)

            total_actions += len(path)

            if not in_vehicle:
                # Add actions to pick up the package
                total_actions += 2  # pick-up and drive actions

            # Add actions to drop off the package
            total_actions += 2  # drop and drive actions

        return total_actions

    def _find_shortest_path(self, start, end):
        """Find the shortest path using BFS."""
        from collections import deque
        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.road_connections.get(current, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        return None

    def _get_vehicle_dropoff_location(self, package, target):
        """Determine the vehicle's drop-off location based on capacity."""
        vehicle = current_locations.get(package, None)
        if vehicle is None:
            return target
        # Check if the vehicle can carry the package
        if vehicle in self.capacity_info:
            current_capacity = self.capacity_info[vehicle]
            if current_capacity >= target:
                return target
        # If not, find the nearest location with sufficient capacity
        for loc in self.road_connections.get(target, []):
            if loc in self.capacity_info and self.capacity_info[loc] >= target:
                return loc
        return target  # Fallback to target location
