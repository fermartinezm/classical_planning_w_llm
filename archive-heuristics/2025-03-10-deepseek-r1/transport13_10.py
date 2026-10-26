from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Transport13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions required to move all packages to their goal locations. 
    Considers drive distances for vehicles to pick up and deliver packages, including necessary 
    pick-up and drop actions. Uses precomputed shortest paths for efficient distance calculation.

    # Assumptions
    - Roads are bidirectional.
    - Vehicles can pick up a package if their capacity allows (exists a predecessor size).
    - Vehicles can drop a package if their capacity allows (exists a successor size).
    - Each package is handled independently, potentially overestimating drive steps.

    # Heuristic Initialization
    - Extracts road network to compute shortest paths between all locations.
    - Extracts capacity-predecessor relationships to check pick-up/drop feasibility.
    - Extracts goal locations for each package from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package at goal: contribute 0.
    2. For each package in a vehicle:
        a. Compute drive distance from vehicle's current location to goal.
        b. Check if vehicle can drop (current capacity has a successor).
        c. Add drive distance + 1 (drop action).
    3. For each package not in a vehicle:
        a. For each vehicle, check if it can pick up (current capacity has a predecessor).
        b. Compute drive distance from vehicle to package and package to goal.
        c. Choose minimal total distance (A + B + 2 actions).
        d. If no vehicle can pick up, assign a large penalty.
    4. Sum all package costs for the heuristic value.
    """

    def __init__(self, task):
        self.goal_locations = {}
        self.capacity_predecessors = set()
        self.road_graph = {}
        self.shortest_paths = {}

        # Extract goal locations for each package
        for goal in task.goals:
            if goal.startswith('(at'):
                parts = goal[1:-1].split()
                package, loc = parts[1], parts[2]
                self.goal_locations[package] = loc

        # Extract static road and capacity-predecessor facts
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
            elif parts[0] == 'capacity-predecessor':
                s1, s2 = parts[1], parts[2]
                self.capacity_predecessors.add((s1, s2))

        # Precompute shortest paths between all locations using BFS
        all_locations = set(self.road_graph.keys())
        self.shortest_paths = {loc: {} for loc in all_locations}
        for start in all_locations:
            queue = deque([(start, 0)])
            visited = {start: 0}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.road_graph.get(current, []):
                    if neighbor not in visited or dist + 1 < visited[neighbor]:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            self.shortest_paths[start] = visited

    def __call__(self, node):
        state = node.state
        package_in_vehicle = {}
        package_loc = {}
        vehicle_loc = {}
        vehicle_cap = {}

        # Parse current state
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    package_loc[obj] = loc
                elif obj.startswith('v'):
                    vehicle_loc[obj] = loc
            elif parts[0] == 'in':
                pkg, veh = parts[1], parts[2]
                package_in_vehicle[pkg] = veh
            elif parts[0] == 'capacity':
                veh, cap = parts[1], parts[2]
                vehicle_cap[veh] = cap

        heuristic_value = 0

        for package, goal in self.goal_locations.items():
            current = package_loc.get(package)
            if package in package_in_vehicle:
                # Package is in a vehicle
                veh = package_in_vehicle[package]
                veh_current = vehicle_loc.get(veh)
                if not veh_current:
                    continue
                # Calculate drive distance from vehicle's location to goal
                distance = self.shortest_paths.get(veh_current, {}).get(goal, float('inf'))
                # Check if vehicle can drop (current capacity has a successor)
                cap = vehicle_cap.get(veh)
                can_drop = any(s1 == cap for (s1, s2) in self.capacity_predecessors)
                if can_drop:
                    heuristic_value += distance + 1  # drive + drop
                else:
                    heuristic_value += distance + 1  # assume can drop despite capacity
            else:
                # Package is not in a vehicle
                current_pkg_loc = package_loc.get(package)
                if current_pkg_loc == goal:
                    continue
                min_cost = float('inf')
                for veh in vehicle_loc:
                    # Check if vehicle can pick up (current capacity has a predecessor)
                    cap = vehicle_cap.get(veh)
                    can_pick = any(s2 == cap for (s1, s2) in self.capacity_predecessors)
                    if not can_pick:
                        continue
                    veh_pos = vehicle_loc[veh]
                    # Distance from vehicle to package and package to goal
                    dist_veh_to_pkg = self.shortest_paths.get(veh_pos, {}).get(current_pkg_loc, float('inf'))
                    dist_pkg_to_goal = self.shortest_paths.get(current_pkg_loc, {}).get(goal, float('inf'))
                    total = dist_veh_to_pkg + dist_pkg_to_goal + 2  # pick + drop
                    if total < min_cost:
                        min_cost = total
                if min_cost == float('inf'):
                    min_cost = 1000  # penalty if no vehicle can pick up
                heuristic_value += min_cost

        return heuristic_value
