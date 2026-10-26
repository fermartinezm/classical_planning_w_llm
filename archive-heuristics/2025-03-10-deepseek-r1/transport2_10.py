from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Transport2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the shortest path for driving, necessary pick-up and drop actions, and vehicle capacities.

    # Assumptions
    - Roads are directed, and the shortest path is computed using BFS.
    - Vehicles can carry one package at a time (capacity constraints based on size hierarchy).
    - If a package is in a vehicle, the vehicle must drive to the goal location and drop it.
    - If a package is not in a vehicle, the closest available vehicle (with capacity) will pick it up.

    # Heuristic Initialization
    - Extracts static road and capacity-predecessor information.
    - Determines goal locations for each package from the problem's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package not at its goal:
        a. If in a vehicle, compute drive distance from vehicle's location to goal and add a drop action.
        b. If not in a vehicle, find the closest vehicle that can pick it up (considering capacity):
            i. Compute drive distance from vehicle's current location to package's location.
            ii. Compute drive distance from package's location to goal.
            iii. Add pick-up and drop actions.
    2. Sum the estimated actions for all packages.
    """

    def __init__(self, task):
        self.roads = defaultdict(list)
        self.capacity_predecessors = {}  # Maps s2 to s1 (capacity-predecessor s1 s2)
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.roads[l1].append(l2)
            elif parts[0] == 'capacity-predecessor':
                s1, s2 = parts[1], parts[2]
                self.capacity_predecessors[s2] = s1

        self.goal_locations = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

    def get_shortest_path_length(self, start, end):
        if start == end:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.roads.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Collect vehicles and their data
        vehicles = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'capacity':
                vehicle = parts[1]
                capacity = parts[2]
                vehicles[vehicle] = {
                    'capacity': capacity,
                    'location': None,
                }
            elif parts[0] == 'at' and parts[1] in vehicles:
                vehicle = parts[1]
                location = parts[2]
                vehicles[vehicle]['location'] = location

        for package, goal_loc in self.goal_locations.items():
            # Check if package is already at goal
            current_at_goal = any(fact == f'(at {package} {goal_loc})' for fact in state)
            if current_at_goal:
                continue

            # Check if package is in a vehicle
            in_vehicle = None
            for fact in state:
                if fact.startswith(f'(in {package} '):
                    parts = fact[1:-1].split()
                    in_vehicle = parts[2]
                    break

            if in_vehicle:
                # Package is in a vehicle, need to drive to goal and drop
                vehicle = in_vehicle
                if vehicle not in vehicles or not vehicles[vehicle]['location']:
                    continue
                vehicle_loc = vehicles[vehicle]['location']
                drive_distance = self.get_shortest_path_length(vehicle_loc, goal_loc)
                if drive_distance == float('inf'):
                    continue
                # Check if vehicle can drop (current capacity has a predecessor)
                current_cap = vehicles[vehicle]['capacity']
                if current_cap in self.capacity_predecessors:
                    total_cost += drive_distance + 1  # drive + drop
            else:
                # Package is not in a vehicle, find its current location
                current_loc = None
                for fact in state:
                    if fact.startswith(f'(at {package} '):
                        parts = fact[1:-1].split()
                        current_loc = parts[2]
                        break
                if not current_loc:
                    continue

                min_steps = float('inf')
                for vehicle, data in vehicles.items():
                    veh_cap = data['capacity']
                    veh_loc = data['location']
                    if not veh_loc or veh_cap not in self.capacity_predecessors:
                        continue  # Cannot pick up

                    d1 = self.get_shortest_path_length(veh_loc, current_loc)
                    if d1 == float('inf'):
                        continue
                    d2 = self.get_shortest_path_length(current_loc, goal_loc)
                    if d2 == float('inf'):
                        continue
                    steps = d1 + d2 + 2  # pick-up and drop
                    if steps < min_steps:
                        min_steps = steps

                if min_steps != float('inf'):
                    total_cost += min_steps
                else:
                    # Penalize if no vehicle can transport
                    total_cost += 1000  # Large penalty for unreachable

        return total_cost
