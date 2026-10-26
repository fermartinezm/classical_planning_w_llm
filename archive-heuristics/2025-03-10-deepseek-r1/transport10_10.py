from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class Transport10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their goal locations. 
    For each package, it considers whether it is already in a vehicle or needs to be picked up, computes the 
    shortest path for required movements, and accounts for pick-up and drop actions.

    # Assumptions
    - Roads are directed, and shortest paths are computed using BFS.
    - Vehicles can carry packages if their current capacity allows (i.e., there exists a capacity-predecessor).
    - If a package cannot be picked up by any vehicle, a large penalty (1000) is added, assuming future actions will resolve this.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds a directed graph of roads from static facts.
    - Preprocesses capacity-predecessor relationships to determine valid pick-ups.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, check if it is already at its goal location.
    2. If the package is in a vehicle:
        a. Calculate the shortest path from the vehicle's current location to the goal.
        b. Add drive steps and a drop action.
    3. If the package is not in a vehicle:
        a. For each vehicle, check if it can pick up the package (capacity allows).
        b. Compute drive steps from the vehicle's location to the package, then to the goal.
        c. Add pick-up and drop actions, track the minimal steps across all vehicles.
    4. Sum the estimated steps for all packages.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                package, loc = parts[1], parts[2]
                self.goal_locations[package] = loc

        self.road_graph = defaultdict(list)
        self.capacity_predecessors = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].append(l2)
            elif parts[0] == 'capacity-predecessor' and len(parts) == 3:
                s_prev, s = parts[1], parts[2]
                self.capacity_predecessors[s] = s_prev

        self.memo_paths = {}

    def shortest_path(self, start, end):
        if start == end:
            return 0
        key = (start, end)
        if key in self.memo_paths:
            return self.memo_paths[key]
        
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current == end:
                self.memo_paths[key] = dist
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        
        self.memo_paths[key] = float('inf')
        return float('inf')

    def __call__(self, node):
        state = node.state
        current_locations = {}
        in_vehicle = {}
        vehicle_locations = {}
        capacities = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                if obj.startswith('v'):
                    vehicle_locations[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif parts[0] == 'capacity':
                vehicle, cap = parts[1], parts[2]
                capacities[vehicle] = cap

        total = 0
        for package, goal_loc in self.goal_locations.items():
            if package not in current_locations and package not in in_vehicle:
                continue

            if package in current_locations and current_locations[package] == goal_loc:
                continue

            if package in in_vehicle:
                vehicle = in_vehicle[package]
                vehicle_loc = vehicle_locations.get(vehicle, None)
                if vehicle_loc == goal_loc:
                    total += 1
                    continue
                if vehicle_loc is None:
                    total += 1000
                    continue
                drive_steps = self.shortest_path(vehicle_loc, goal_loc)
                if drive_steps == float('inf'):
                    total += 1000
                else:
                    total += drive_steps + 1
            else:
                package_loc = current_locations.get(package, None)
                if package_loc is None:
                    total += 1000
                    continue
                min_steps = float('inf')
                for vehicle in vehicle_locations:
                    vehicle_cap = capacities.get(vehicle, None)
                    if vehicle_cap is None or vehicle_cap not in self.capacity_predecessors:
                        continue
                    vehicle_loc = vehicle_locations[vehicle]
                    drive1 = self.shortest_path(vehicle_loc, package_loc)
                    if drive1 == float('inf'):
                        continue
                    drive2 = self.shortest_path(package_loc, goal_loc)
                    if drive2 == float('inf'):
                        continue
                    total_steps = drive1 + 1 + drive2 + 1
                    if total_steps < min_steps:
                        min_steps = total_steps
                if min_steps == float('inf'):
                    total += 1000
                else:
                    total += min_steps

        return total
