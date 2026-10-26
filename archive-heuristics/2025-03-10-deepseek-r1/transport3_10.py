from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class Transport3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions required to move all packages to their goal locations. 
    For each package, it calculates the minimal drive steps needed for a vehicle to pick it up, 
    transport it to the goal, and considers vehicle capacities and road directions.

    # Assumptions
    - Roads are directed; shortest paths are precomputed using BFS.
    - Vehicles can carry packages if their current capacity allows (not c0).
    - Each package's cost is calculated independently, which may overcount shared vehicle movements.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds a directed road graph from static facts.
    - Precomputes shortest paths between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package:
        a. If at goal: cost += 0.
        b. If in a vehicle: cost += drive steps from vehicle's location to goal + 1 (drop).
        c. If at a location and not at goal:
            i. Find vehicles with capacity > c0.
            ii. For each vehicle, compute drive steps from its location to package's location (A) and to goal (B).
            iii. Take minimal (A + B) + 2 (pick-up and drop).
    2. Sum costs for all packages.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('p'):
                self.goal_locations[parts[1]] = parts[2]

        # Build directed road graph
        self.road_graph = defaultdict(list)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                self.road_graph[parts[1]].append(parts[2])

        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        all_locations = set(self.road_graph.keys())
        for destinations in self.road_graph.values():
            all_locations.update(destinations)
        all_locations = list(all_locations)

        for start in all_locations:
            queue = deque([(start, 0)])
            visited = {start: 0}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.road_graph.get(current, []):
                    if neighbor not in visited or visited[neighbor] > dist + 1:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            self.shortest_paths[start] = visited

    def __call__(self, node):
        state = node.state
        package_locs = {}
        vehicle_locs = {}
        capacities = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    package_locs[obj] = loc
                elif obj.startswith('v'):
                    vehicle_locs[obj] = loc
            elif parts[0] == 'in':
                package_locs[parts[1]] = parts[2]
            elif parts[0] == 'capacity':
                capacities[parts[1]] = parts[2]

        total_cost = 0
        for package, goal_loc in self.goal_locations.items():
            current = package_locs.get(package)
            if not current:
                continue

            if current.startswith('v'):
                # In a vehicle
                vehicle = current
                veh_loc = vehicle_locs.get(vehicle)
                if not veh_loc:
                    continue
                sp = self.shortest_paths.get(veh_loc, {})
                distance = sp.get(goal_loc, float('inf'))
                total_cost += distance + 1 if distance != float('inf') else 1000
            else:
                # At a location
                if current == goal_loc:
                    continue
                min_cost = float('inf')
                for veh, veh_loc in vehicle_locs.items():
                    if capacities.get(veh, 'c0') == 'c0':
                        continue
                    sp_veh = self.shortest_paths.get(veh_loc, {})
                    dist_a = sp_veh.get(current, float('inf'))
                    if dist_a == float('inf'):
                        continue
                    sp_pkg = self.shortest_paths.get(current, {})
                    dist_b = sp_pkg.get(goal_loc, float('inf'))
                    if dist_b == float('inf'):
                        continue
                    cost = dist_a + dist_b + 2
                    if cost < min_cost:
                        min_cost = cost
                total_cost += min_cost if min_cost != float('inf') else 1000

        return total_cost
