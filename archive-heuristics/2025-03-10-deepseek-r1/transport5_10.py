from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

class Transport5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions required to move all packages to their goal locations by considering the minimal drive steps for each package, along with necessary pick-up and drop actions, taking into account vehicle capacities.

    # Assumptions
    - Each package must be transported by a vehicle that can reach its current and goal locations.
    - Vehicles can only pick up packages if their current capacity allows (i.e., has a predecessor in the capacity hierarchy).
    - Roads are directed, and drive steps are computed using the shortest path in the directed road graph.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goals.
    - Builds a capacity predecessor hierarchy from static facts.
    - Precomputes shortest paths between all locations using BFS on the road graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, check if it is already at its goal location. If yes, cost is 0.
    2. If the package is in a vehicle:
        a. Compute drive steps from the vehicle's current location to the goal.
        b. Add 1 action for dropping the package.
    3. If the package is not in a vehicle:
        a. For each vehicle capable of picking up (based on capacity):
            i. Compute drive steps from the vehicle's location to the package's location.
            ii. Add 1 action for picking up.
            iii. Compute drive steps from the package's location to the goal.
            iv. Add 1 action for dropping.
        b. Take the minimal cost from all capable vehicles.
        c. If no capable vehicle is found, add a high penalty.
    4. Sum the costs for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('p'):
                self.goal_locations[parts[1]] = parts[2]

        # Build capacity predecessor hierarchy
        self.capacity_predecessors = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'capacity-predecessor':
                self.capacity_predecessors[parts[2]] = parts[1]

        # Build road graph and precompute shortest paths
        self.road_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                self.road_graph[l1].append(l2)

        # Precompute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        locations = set(self.road_graph.keys())
        for l in self.road_graph.values():
            locations.update(l)
        for loc in locations:
            self.shortest_paths[loc] = self.bfs(loc)

    def bfs(self, start):
        """Compute shortest paths from 'start' using BFS."""
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def get_drive_steps(self, from_loc, to_loc):
        """Return minimal drive steps between locations, or infinity if unreachable."""
        if from_loc not in self.shortest_paths:
            return float('inf')
        return self.shortest_paths[from_loc].get(to_loc, float('inf'))

    def __call__(self, node):
        state = node.state
        package_info = {}  # {package: {location: str | None, vehicle: str | None}}
        vehicle_info = {}  # {vehicle: {location: str, capacity: str}}

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    package_info[obj] = {'location': loc, 'vehicle': None}
                elif obj.startswith('v'):
                    if obj not in vehicle_info:
                        vehicle_info[obj] = {'location': loc, 'capacity': None}
                    else:
                        vehicle_info[obj]['location'] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                package_info[package] = {'location': None, 'vehicle': vehicle}
            elif parts[0] == 'capacity':
                vehicle, cap = parts[1], parts[2]
                if vehicle not in vehicle_info:
                    vehicle_info[vehicle] = {'location': None, 'capacity': cap}
                else:
                    vehicle_info[vehicle]['capacity'] = cap

        total_cost = 0
        for package, goal_loc in self.goal_locations.items():
            if package not in package_info:
                total_cost += 1000  # Penalty for missing package info
                continue

            current = package_info[package]
            if current['vehicle'] is None:
                current_loc = current['location']
                if current_loc == goal_loc:
                    continue  # Already at goal

                # Package needs to be picked up and transported
                min_cost = float('inf')
                for vehicle, info in vehicle_info.items():
                    veh_loc = info['location']
                    capacity = info['capacity']
                    if capacity is None or capacity not in self.capacity_predecessors:
                        continue  # Cannot pick up

                    drive_to_package = self.get_drive_steps(veh_loc, current_loc)
                    drive_to_goal = self.get_drive_steps(current_loc, goal_loc)
                    if drive_to_package == float('inf') or drive_to_goal == float('inf'):
                        continue

                    cost = drive_to_package + 1 + drive_to_goal + 1
                    if cost < min_cost:
                        min_cost = cost

                if min_cost == float('inf'):
                    total_cost += 1000  # No valid vehicle found
                else:
                    total_cost += min_cost
            else:
                # Package is in a vehicle
                vehicle = current['vehicle']
                if vehicle not in vehicle_info:
                    total_cost += 1000  # Vehicle not found
                    continue

                veh_loc = vehicle_info[vehicle]['location']
                if veh_loc == goal_loc:
                    total_cost += 1  # Only drop needed
                else:
                    drive_steps = self.get_drive_steps(veh_loc, goal_loc)
                    if drive_steps == float('inf'):
                        total_cost += 1000
                    else:
                        total_cost += drive_steps + 1  # Drive and drop

        return total_cost
