from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class Transport24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their goal locations. 
    It considers the shortest path for vehicle movements, pick-up/drop actions, and adjusts for vehicle capacities.

    # Assumptions
    - Vehicles can drop existing packages to increase their capacity if needed.
    - Roads are bidirectional for pathfinding.
    - The heuristic may overestimate steps when vehicles need to adjust capacity but ensures finite values for solvable states.

    # Heuristic Initialization
    - Extracts road network and capacity hierarchy from static facts.
    - Preprocesses shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, check if it's at the goal. If not, calculate steps based on its current state.
    2. If the package is in a vehicle, compute drive steps from the vehicle's location to the goal plus a drop action.
    3. If the package is not in a vehicle, evaluate all vehicles:
        a. Check if the vehicle can pick up the package directly (current capacity allows).
        b. If not, calculate steps to drop existing packages, then pick up and deliver.
    4. Sum the minimal steps for all packages to get the heuristic value.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                self.goal_locations[parts[1]] = parts[2]

        self.road_graph = {}
        self.capacity_predecessors = {}
        self.capacity_successors = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1)
            elif parts[0] == 'capacity-predecessor':
                s1, s2 = parts[1], parts[2]
                self.capacity_predecessors[s2] = s1
                self.capacity_successors[s1] = s2

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
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        state = node.state
        current_package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        packages_in_vehicle = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    current_package_locations[obj] = loc
                elif obj.startswith('v'):
                    vehicle_locations[obj] = loc
            elif parts[0] == 'in':
                pkg, veh = parts[1], parts[2]
                current_package_locations[pkg] = veh
                packages_in_vehicle.setdefault(veh, []).append(pkg)
            elif parts[0] == 'capacity':
                veh, cap = parts[1], parts[2]
                vehicle_capacities[veh] = cap

        total = 0
        for package, goal_loc in self.goal_locations.items():
            current = current_package_locations.get(package)
            if not current:
                continue

            if current == goal_loc:
                continue

            if current.startswith('v'):
                vehicle = current
                veh_loc = vehicle_locations.get(vehicle)
                if not veh_loc:
                    continue
                path_length = self.get_shortest_path_length(veh_loc, goal_loc)
                if path_length is None:
                    return float('inf')
                total += path_length + 1
            else:
                current_loc = current
                min_steps = float('inf')
                for vehicle in vehicle_locations:
                    veh_loc = vehicle_locations[vehicle]
                    capacity = vehicle_capacities.get(vehicle)
                    pkg_in_veh = packages_in_vehicle.get(vehicle, [])

                    steps = 0
                    valid = True

                    if capacity in self.capacity_predecessors:
                        path1 = self.get_shortest_path_length(veh_loc, current_loc)
                        path2 = self.get_shortest_path_length(current_loc, goal_loc)
                        if path1 is None or path2 is None:
                            continue
                        steps_candidate = path1 + path2 + 2
                        if steps_candidate < min_steps:
                            min_steps = steps_candidate
                    else:
                        if not pkg_in_veh:
                            continue

                        new_capacity = capacity
                        for _ in range(len(pkg_in_veh)):
                            new_capacity = self.capacity_successors.get(new_capacity)
                            if not new_capacity:
                                valid = False
                                break
                        if not valid or new_capacity not in self.capacity_predecessors:
                            continue

                        steps_drop = len(pkg_in_veh)
                        path1 = self.get_shortest_path_length(veh_loc, current_loc)
                        path2 = self.get_shortest_path_length(current_loc, goal_loc)
                        if path1 is None or path2 is None:
                            continue
                        steps_candidate = steps_drop + path1 + path2 + 2
                        if steps_candidate < min_steps:
                            min_steps = steps_candidate

                if min_steps == float('inf'):
                    return float('inf')
                total += min_steps

        return total
