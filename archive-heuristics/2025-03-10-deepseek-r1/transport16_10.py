from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class Transport16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions needed to move all packages to their goal locations. 
    Considers drive actions for vehicle movement, pick-up/drop actions, and vehicle capacities.

    # Assumptions
    - Roads are directed; shortest paths are precomputed using BFS.
    - Vehicles can carry packages if their capacity level allows.
    - Each pick-up/drop action is 1 step; drive steps are based on shortest path.

    # Heuristic Initialization
    1. Extracts goal locations for each package.
    2. Builds a road graph and precomputes shortest paths between all locations.
    3. Constructs capacity hierarchy to determine vehicle capacities.

    # Step-By-Step Thinking
    1. For each package not at its goal:
        a. If in a vehicle: Calculate drive steps from vehicle's location to goal, add 1 for drop.
        b. If at a location: Find the best vehicle (min drive steps) to pick up and deliver, adding 2 actions.
    2. Sum all actions for all packages.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('p'):
                self.goal_locations[parts[1]] = parts[2]

        self.road_map = defaultdict(list)
        roads = set()
        for fact in task.static:
            if match(fact, 'road', '*', '*'):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.road_map[l1].append(l2)
                roads.add((l1, l2))

        self.locations = {l for l1, l2 in roads for l in (l1, l2)}
        self.shortest_paths = defaultdict(dict)
        for start in self.locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.road_map.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for node, dist in visited.items():
                self.shortest_paths[(start, node)] = dist

        self.capacity_predecessors = {}
        for fact in task.static:
            if match(fact, 'capacity-predecessor', '*', '*'):
                s1, s2 = get_parts(fact)[1], get_parts(fact)[2]
                self.capacity_predecessors[s2] = s1

        self.capacity_levels = {'c0': 0}
        current_level = 0
        current_cap = 'c0'
        while True:
            next_caps = [s for s, p in self.capacity_predecessors.items() if p == current_cap]
            if not next_caps:
                break
            current_level += 1
            for cap in next_caps:
                self.capacity_levels[cap] = current_level
            current_cap = next_caps[0]

    def __call__(self, node):
        state = node.state
        packages = {}
        vehicles = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    packages[obj] = loc
                else:
                    vehicles[obj] = {'location': loc, 'capacity_level': 0}
            elif parts[0] == 'in':
                packages[parts[1]] = parts[2]
            elif parts[0] == 'capacity':
                vehicle, cap = parts[1], parts[2]
                if vehicle in vehicles:
                    vehicles[vehicle]['capacity_level'] = self.capacity_levels.get(cap, 0)

        heuristic_value = 0
        for package, goal_loc in self.goal_locations.items():
            current = packages.get(package)
            if current == goal_loc:
                continue

            if current in vehicles:  # Package is in a vehicle
                vehicle = vehicles[current]
                distance = self.shortest_paths.get((vehicle['location'], goal_loc), float('inf'))
                heuristic_value += distance + 1 if distance != float('inf') else 0
            else:  # Package is at a location
                current_loc = current
                min_steps = float('inf')
                for veh in vehicles.values():
                    if veh['capacity_level'] < 1:
                        continue
                    veh_loc = veh['location']
                    to_package = self.shortest_paths.get((veh_loc, current_loc), float('inf'))
                    to_goal = self.shortest_paths.get((current_loc, goal_loc), float('inf'))
                    total = to_package + to_goal + 2
                    if total < min_steps:
                        min_steps = total
                if min_steps != float('inf'):
                    heuristic_value += min_steps

        return heuristic_value
