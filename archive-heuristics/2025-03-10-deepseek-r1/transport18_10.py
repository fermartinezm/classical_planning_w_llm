from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict


class Transport18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations. 
    For each package, it considers whether it is in a vehicle or not and calculates the minimal drive actions 
    needed along with necessary pick-up and drop actions.

    # Assumptions
    - Roads are directed, and the shortest path between locations is computed using BFS.
    - Vehicles can always pick up and drop packages regardless of current capacity (ignores capacity constraints for efficiency).
    - The problem is solvable, so all necessary paths exist.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task.
    - Builds a directed graph of roads from static facts.
    - Precomputes shortest paths between all pairs of locations using BFS.
    - Identifies all vehicles from initial capacity facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package:
        a. If the package is at the goal, contribute 0 to the heuristic.
        b. If the package is in a vehicle:
            i. Compute the shortest path from the vehicle's current location to the goal.
            ii. Add 1 action for dropping the package.
        c. If the package is not in a vehicle:
            i. For each vehicle, compute the shortest path from the vehicle's location to the package's location.
            ii. Compute the shortest path from the package's location to the goal.
            iii. Take the minimal total actions (drive to package, drive to goal, pick-up, and drop) across all vehicles.
    2. Sum the estimated actions for all packages.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            if goal.startswith('(at '):
                parts = goal[1:-1].split()
                package = parts[1]
                loc = parts[2]
                self.goal_locations[package] = loc

        self.roads = defaultdict(list)
        for fact in task.static:
            if fact.startswith('(road '):
                parts = fact[1:-1].split()
                l1, l2 = parts[1], parts[2]
                self.roads[l1].append(l2)

        self.shortest_paths = {}
        locations = set(self.roads.keys())
        for l1 in self.roads:
            for l2 in self.roads[l1]:
                locations.add(l2)
        locations = list(locations)
        for loc in locations:
            self.shortest_paths[loc] = self.bfs(loc)

        self.vehicles = set()
        for fact in task.initial_state:
            if fact.startswith('(capacity '):
                parts = fact[1:-1].split()
                self.vehicles.add(parts[1])

    def bfs(self, start):
        visited = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.roads.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        return visited

    def get_distance(self, from_loc, to_loc):
        if from_loc not in self.shortest_paths:
            return float('inf')
        return self.shortest_paths[from_loc].get(to_loc, float('inf'))

    def __call__(self, node):
        state = node.state
        current_locations = {}
        packages_in_vehicles = {}

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                packages_in_vehicles[package] = vehicle

        current_vehicle_locs = {v: current_locations[v] for v in self.vehicles if v in current_locations}

        total_cost = 0
        for package, goal_loc in self.goal_locations.items():
            if package in packages_in_vehicles:
                vehicle = packages_in_vehicles[package]
                if vehicle not in current_vehicle_locs:
                    continue
                vehicle_loc = current_vehicle_locs[vehicle]
                distance = self.get_distance(vehicle_loc, goal_loc)
                total_cost += distance + 1 if distance != float('inf') else 0
            else:
                if package not in current_locations:
                    continue
                package_loc = current_locations[package]
                if package_loc == goal_loc:
                    continue
                min_cost = float('inf')
                for vehicle in current_vehicle_locs:
                    vehicle_loc = current_vehicle_locs[vehicle]
                    d1 = self.get_distance(vehicle_loc, package_loc)
                    d2 = self.get_distance(package_loc, goal_loc)
                    if d1 != float('inf') and d2 != float('inf'):
                        total = d1 + d2 + 2
                        if total < min_cost:
                            min_cost = total
                if min_cost != float('inf'):
                    total_cost += min_cost

        return total_cost
