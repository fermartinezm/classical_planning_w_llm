from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions required to transport all packages to their goals. 
    For each package, computes minimal drives needed considering current locations and 
    vehicle capacities, using precomputed shortest paths between locations.

    # Assumptions:
    - Roads are directed, requiring explicit drive actions for each direction.
    - Vehicles can pick up packages only if their capacity allows.
    - Each package's transport is considered independently, potentially overestimating steps.

    # Heuristic Initialization
    1. Extracts goal locations for each package.
    2. Builds a directed graph from static roads and precomputes shortest paths using BFS.
    3. Collects capacity-predecessor relationships to determine valid pick-ups.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package in the goal:
        a. If in a vehicle, add drive steps from vehicle's location to goal and a drop action.
        b. If not in a vehicle, find the closest vehicle that can pick it up (based on capacity and location), then compute steps for pickup, drive to goal, and drop.
    2. Sum the minimal steps for all packages, handling each independently.
    """

    def __init__(self, task):
        self.goal_locations = {}
        # Extract goal locations for each package
        for fact in task.goals:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('p'):
                self.goal_locations[parts[1]] = parts[2]

        # Build directed graph from static roads
        roads = []
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                roads.append((parts[1], parts[2]))

        self.graph = {}
        for l1, l2 in roads:
            if l1 not in self.graph:
                self.graph[l1] = []
            self.graph[l1].append(l2)

        # Collect all unique locations and precompute shortest paths
        all_locations = {l1 for l1, _ in roads} | {l2 for _, l2 in roads}
        self.shortest_paths = {loc: {} for loc in all_locations}
        for loc in all_locations:
            visited = {loc: 0}
            queue = deque([loc])
            while queue:
                current = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            self.shortest_paths[loc] = visited

        # Extract capacity predecessors
        self.capacity_predecessors = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'capacity-predecessor':
                self.capacity_predecessors[parts[2]] = parts[1]

    def __call__(self, node):
        state = node.state
        cost = 0

        package_to_vehicle = {}
        package_locs = {}
        vehicle_locs = {}
        vehicle_caps = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'in' and parts[1].startswith('p'):
                package_to_vehicle[parts[1]] = parts[2]
            elif parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    package_locs[obj] = loc
                else:
                    vehicle_locs[obj] = loc
            elif parts[0] == 'capacity':
                vehicle_caps[parts[1]] = parts[2]

        for pkg, goal_loc in self.goal_locations.items():
            if pkg in package_to_vehicle:
                vehicle = package_to_vehicle[pkg]
                vehicle_loc = vehicle_locs.get(vehicle)
                if not vehicle_loc:
                    continue
                steps = self.shortest_paths.get(vehicle_loc, {}).get(goal_loc, float('inf'))
                cost += steps + 1  # drive and drop
            else:
                current_loc = package_locs.get(pkg)
                if current_loc == goal_loc:
                    continue
                min_steps = float('inf')
                for veh in vehicle_locs:
                    cap = vehicle_caps.get(veh)
                    if not cap or cap not in self.capacity_predecessors:
                        continue
                    veh_loc = vehicle_locs[veh]
                    steps_to_p = self.shortest_paths.get(veh_loc, {}).get(current_loc, float('inf'))
                    steps_to_g = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
                    total = steps_to_p + 1 + steps_to_g + 1
                    if total < min_steps:
                        min_steps = total
                cost += min_steps if min_steps != float('inf') else 1000  # penalty if no vehicle

        return cost
