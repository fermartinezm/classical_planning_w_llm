from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class Transport17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations. It considers drive actions for vehicles, pick-up/drop actions, and vehicle capacity constraints.

    # Assumptions
    - Roads are directed, and the shortest path between locations is precomputed.
    - Vehicles can only pick up packages if their current capacity allows (i.e., they have a predecessor size).
    - Each package's goal is handled independently, ignoring potential synergies between vehicle loads.

    # Heuristic Initialization
    - Extracts static road and capacity-predecessor facts to build a road graph and predecessor map.
    - Precomputes all-pairs shortest paths for the road network using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package in the goal:
        a. If already at the goal, contribute 0.
        b. If in a vehicle, compute drive steps from vehicle's location to goal + 1 drop action.
        c. If not in a vehicle:
            i. Find all vehicles that can pick it up (current capacity allows).
            ii. For each valid vehicle, compute drive steps to package, pick-up, drive to goal, and drop.
            iii. Take the minimal steps from all valid vehicles.
            iv. If no valid vehicle, assign a high penalty (1e6).
    2. Sum the minimal steps for all packages.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('p'):
                self.goal_locations[parts[1]] = parts[2]

        # Build road graph and predecessor map from static facts
        self.predecessor_map = {}
        road_graph = defaultdict(set)
        for fact in task.static:
            if match(fact, 'capacity-predecessor', '*', '*'):
                s1, s2 = get_parts(fact)[1], get_parts(fact)[2]
                self.predecessor_map[s2] = s1
            elif match(fact, 'road', '*', '*'):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                road_graph[l1].add(l2)

        # Precompute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        all_locations = set(road_graph.keys())
        for neighbors in road_graph.values():
            all_locations.update(neighbors)
        all_locations = list(all_locations)

        for start in all_locations:
            distances = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in road_graph.get(current, []):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            for end in all_locations:
                self.shortest_paths[(start, end)] = distances.get(end, float('inf'))

    def __call__(self, node):
        state = node.state
        current_package_locs = {}
        current_vehicle_locs = {}
        current_vehicle_caps = {}
        packages_in_vehicles = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    current_package_locs[obj] = loc
                elif obj.startswith('v'):
                    current_vehicle_locs[obj] = loc
            elif parts[0] == 'in':
                pkg, veh = parts[1], parts[2]
                packages_in_vehicles[pkg] = veh
            elif parts[0] == 'capacity':
                veh, cap = parts[1], parts[2]
                current_vehicle_caps[veh] = cap

        total = 0
        INF = float('inf')
        for package, goal_loc in self.goal_locations.items():
            if package in packages_in_vehicles:
                # Package is in a vehicle
                vehicle = packages_in_vehicles[package]
                veh_loc = current_vehicle_locs.get(vehicle, None)
                if veh_loc is None:
                    total += 1e6
                    continue
                distance = self.shortest_paths.get((veh_loc, goal_loc), INF)
                total += distance + 1  # drive + drop
            else:
                # Package is not in a vehicle
                pkg_loc = current_package_locs.get(package, None)
                if pkg_loc == goal_loc:
                    continue
                if pkg_loc is None:
                    total += 1e6
                    continue
                min_steps = INF
                for vehicle, veh_loc in current_vehicle_locs.items():
                    cap = current_vehicle_caps.get(vehicle, None)
                    if cap not in self.predecessor_map:
                        continue  # Cannot pick up
                    # Drive to package, then to goal
                    dist_pickup = self.shortest_paths.get((veh_loc, pkg_loc), INF)
                    dist_deliver = self.shortest_paths.get((pkg_loc, goal_loc), INF)
                    steps = dist_pickup + dist_deliver + 2  # pick-up and drop
                    if steps < min_steps:
                        min_steps = steps
                total += min_steps if min_steps != INF else 1e6
        return total
