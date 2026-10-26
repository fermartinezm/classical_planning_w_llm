from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to transport all packages to their goal locations. It considers the minimal drive steps between locations (using precomputed shortest paths), the need to pick-up and drop packages, and vehicle capacities.

    # Assumptions
    - Roads form a directed graph, and shortest paths are precomputed.
    - Each package's goal is an 'at' predicate.
    - Vehicles can carry packages if their capacity allows (current size has a predecessor).
    - The state is solvable, so paths exist and vehicles are available for all packages.

    # Heuristic Initialization
    1. Extract road network from static facts and precompute all-pairs shortest paths (APSP) using BFS.
    2. Extract capacity-predecessor hierarchy to determine valid vehicle capacities.
    3. Extract goal locations for each package from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package:
        a. If already at goal, cost is 0.
        b. If in a vehicle:
            i. Drive from current vehicle location to package's goal location.
            ii. Add drive steps + 1 (drop action).
        c. If not in a vehicle:
            i. Find the best vehicle (minimal total drive steps to pick-up and deliver).
            ii. Drive steps from vehicle's location to package's current location.
            iii. Drive steps from package's location to goal.
            iv. Add 2 actions (pick-up and drop) and sum all steps.
    2. Sum costs for all packages to get total heuristic value.
    """

    def __init__(self, task):
        # Extract roads and build adjacency list
        roads = set()
        for fact in task.static:
            if match(fact, 'road', '*', '*'):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                roads.add((l1, l2))
        
        # Build adjacency list for directed graph
        adjacency = defaultdict(list)
        locations = set()
        for l1, l2 in roads:
            adjacency[l1].append(l2)
            locations.add(l1)
            locations.add(l2)
        locations = list(locations)
        
        # Precompute all-pairs shortest paths using BFS
        self.apsp = {}
        for start in locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for end in locations:
                self.apsp[(start, end)] = visited.get(end, float('inf'))
        
        # Extract capacity-predecessor hierarchy
        self.predecessor_map = {}
        for fact in task.static:
            if match(fact, 'capacity-predecessor', '*', '*'):
                parts = get_parts(fact)
                s1, s2 = parts[1], parts[2]
                self.predecessor_map[s2] = s1
        
        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in task.goals:
            if match(goal, 'at', '*', '*'):
                parts = get_parts(goal)
                package, loc = parts[1], parts[2]
                self.goal_locations[package] = loc

    def __call__(self, node):
        state = node.state
        vehicle_capacities = {}
        vehicle_locations = {}
        package_in_vehicle = {}
        package_locations = {}
        
        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 0:
                continue
            if parts[0] == 'capacity' and len(parts) == 3:
                vehicle = parts[1]
                size = parts[2]
                vehicle_capacities[vehicle] = size
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in vehicle_capacities:
                    vehicle_locations[obj] = loc
                else:
                    package_locations[obj] = loc
            elif parts[0] == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
        
        total_cost = 0
        for package, goal_loc in self.goal_locations.items():
            if package in package_in_vehicle:
                # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                current_loc = vehicle_locations.get(vehicle, None)
                if current_loc is None:
                    continue  # Assuming valid state, skip if not found
                drive_steps = self.apsp.get((current_loc, goal_loc), float('inf'))
                total_cost += drive_steps + 1  # drop action
            else:
                # Package is not in a vehicle
                current_loc = package_locations.get(package, None)
                if current_loc is None or current_loc == goal_loc:
                    continue  # Already at goal or invalid state
                min_cost = float('inf')
                for vehicle in vehicle_capacities:
                    capacity = vehicle_capacities[vehicle]
                    if capacity not in self.predecessor_map:
                        continue  # Vehicle cannot pick up
                    vehicle_loc = vehicle_locations.get(vehicle, None)
                    if vehicle_loc is None:
                        continue  # Vehicle location unknown
                    drive_to_package = self.apsp.get((vehicle_loc, current_loc), float('inf'))
                    drive_to_goal = self.apsp.get((current_loc, goal_loc), float('inf'))
                    if drive_to_package == float('inf') or drive_to_goal == float('inf'):
                        continue
                    cost = drive_to_package + drive_to_goal + 2  # pick-up and drop
                    if cost < min_cost:
                        min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost
                else:
                    # Penalty for unreachable, assuming state is solvable
                    total_cost += 1000
        return total_cost
