from fnmatch import fnmatch
from collections import deque

class Transport25Heuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations using vehicles. It considers the shortest path for driving actions and assumes vehicles can adjust their capacity as needed.

    # Assumptions
    - Vehicles can always pick up and drop packages regardless of their current capacity.
    - Roads are bidirectional and precomputed shortest paths are used for drive actions.
    - Each pick-up and drop action is counted as one step each.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds a road graph from static road facts.
    - Precomputes shortest paths between all pairs of locations using BFS.
    - Extracts capacity predecessor relationships from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package in the goal conditions:
        a. If the package is already at its goal location, cost is 0.
        b. If the package is in a vehicle, calculate the drive distance from the vehicle's current location to the goal and add a drop action.
        c. If the package is at a location, find the nearest vehicle to pick it up, drive to the goal, and drop it. Sum the drive distances and actions.
    2. Sum the costs for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        self.goal_locations = self._extract_goal_locations(task.goals)
        self.static_roads = self._build_road_graph(task.static)
        self.location_distances = self._precompute_distances()
        self.predecessor_map = self._extract_predecessors(task.static)

    def _extract_goal_locations(self, goals):
        goal_locs = {}
        for goal in goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and parts[1].startswith('p'):
                package = parts[1]
                location = parts[2]
                goal_locs[package] = location
        return goal_locs

    def _build_road_graph(self, static):
        graph = {}
        for fact in static:
            if fnmatch(fact, '(road * *)'):
                parts = fact[1:-1].split()
                l1, l2 = parts[1], parts[2]
                if l1 not in graph:
                    graph[l1] = set()
                graph[l1].add(l2)
                if l2 not in graph:
                    graph[l2] = set()
                graph[l2].add(l1)
        return graph

    def _precompute_distances(self):
        distances = {}
        locations = self.static_roads.keys()
        for loc in locations:
            distances[loc] = self._bfs(loc)
        return distances

    def _bfs(self, start):
        visited = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.static_roads.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        return visited

    def _extract_predecessors(self, static):
        predecessors = {}
        for fact in static:
            if fnmatch(fact, '(capacity-predecessor * *)'):
                parts = fact[1:-1].split()
                s1, s2 = parts[1], parts[2]
                predecessors[s2] = s1
        return predecessors

    def __call__(self, node):
        state = node.state
        current_locations = {}
        in_vehicle = {}
        capacities = {}

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif parts[0] == 'capacity':
                vehicle, cap = parts[1], parts[2]
                capacities[vehicle] = cap

        total_cost = 0
        vehicles = {parts[1] for fact in state if fnmatch(fact, '(capacity * *)') for parts in [fact[1:-1].split()]}

        for package, goal_loc in self.goal_locations.items():
            if package in in_vehicle:
                vehicle = in_vehicle[package]
                vehicle_loc = current_locations.get(vehicle)
                if vehicle_loc:
                    distance = self._get_distance(vehicle_loc, goal_loc)
                    total_cost += distance + 1  # Drop action
            else:
                package_loc = current_locations.get(package)
                if package_loc == goal_loc:
                    continue
                if not package_loc:
                    continue
                min_steps = float('inf')
                for vehicle in vehicles:
                    vehicle_loc = current_locations.get(vehicle)
                    if not vehicle_loc:
                        continue
                    distance_a = self._get_distance(vehicle_loc, package_loc)
                    distance_b = self._get_distance(package_loc, goal_loc)
                    if distance_a == float('inf') or distance_b == float('inf'):
                        continue
                    total = distance_a + distance_b + 2  # Pick-up and drop
                    if total < min_steps:
                        min_steps = total
                if min_steps != float('inf'):
                    total_cost += min_steps
                else:
                    total_cost += 0  # Fallback if no path found (assumes problem is solvable)

        return total_cost

    def _get_distance(self, from_loc, to_loc):
        if from_loc == to_loc:
            return 0
        return self.location_distances.get(from_loc, {}).get(to_loc, float('inf'))
