from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class Transport21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations. For each package, it calculates the minimal drive actions needed for a vehicle to pick it up, transport it to the goal, and drop it, considering current capacities and road distances.

    # Assumptions
    - Each package can be transported independently by the best possible vehicle.
    - Vehicles can pick up a package if their current capacity allows (based on capacity-predecessor chain).
    - Drive actions between locations use the shortest path based on the road network.

    # Heuristic Initialization
    - Extract static road information to build a graph for shortest path calculations.
    - Extract capacity-predecessor relationships to determine valid pick-ups.
    - Store goal locations for each package.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal:
        a. If in a vehicle: Sum drive steps from vehicle's current location to goal plus a drop action.
        b. If not in a vehicle: Find the closest vehicle that can pick it up (considering capacity), sum drive steps to package, pick-up, drive to goal, and drop.
    2. Sum the minimal actions for all packages.
    """

    def __init__(self, task):
        self.goal_locations = {}
        # Extract goal locations for each package
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('p'):
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        # Build road graph
        self.road_graph = defaultdict(list)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                from_loc = parts[1]
                to_loc = parts[2]
                self.road_graph[from_loc].append(to_loc)

        # Build capacity predecessor hierarchy
        self.capacity_predecessors = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'capacity-predecessor':
                s1 = parts[1]
                s2 = parts[2]
                self.capacity_predecessors[s2] = s1

    def _shortest_path(self, start, end):
        """Compute the shortest path between start and end using BFS."""
        if start == end:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == end:
                return dist
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.road_graph.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        # If no path found, return a large number (assuming solvable)
        return 1000  # Arbitrary large value

    def __call__(self, node):
        state = node.state
        current_locations = {}
        in_vehicles = {}
        vehicles = set()
        vehicle_capacities = {}

        # Parse state to get current locations and capacities
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj = parts[1]
                loc = parts[2]
                current_locations[obj] = loc
            elif parts[0] == 'in':
                package = parts[1]
                vehicle = parts[2]
                in_vehicles[package] = vehicle
            elif parts[0] == 'capacity':
                vehicle = parts[1]
                capacity = parts[2]
                vehicles.add(vehicle)
                vehicle_capacities[vehicle] = capacity

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            if package in in_vehicles:
                # Package is in a vehicle
                vehicle = in_vehicles[package]
                vehicle_loc = current_locations.get(vehicle, None)
                if not vehicle_loc:
                    # Vehicle location unknown, assume worst case
                    total_cost += 1000
                    continue
                if vehicle_loc == goal_loc:
                    # Already at goal, no cost
                    continue
                drive_steps = self._shortest_path(vehicle_loc, goal_loc)
                total_cost += drive_steps + 1  # drive + drop
            else:
                # Package is not in a vehicle
                package_loc = current_locations.get(package, None)
                if package_loc == goal_loc:
                    continue  # already at goal
                min_cost = float('inf')
                for vehicle in vehicles:
                    vehicle_loc = current_locations.get(vehicle, None)
                    if not vehicle_loc:
                        continue
                    capacity = vehicle_capacities.get(vehicle, None)
                    if not capacity or capacity not in self.capacity_predecessors:
                        continue  # cannot pick up
                    # Calculate drive steps to package and to goal
                    drive_to_package = self._shortest_path(vehicle_loc, package_loc)
                    drive_to_goal = self._shortest_path(package_loc, goal_loc)
                    cost = drive_to_package + drive_to_goal + 2  # pick-up and drop
                    if cost < min_cost:
                        min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost
                else:
                    # No vehicle can pick up, add large penalty
                    total_cost += 1000

        return total_cost
