from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transport11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions needed to move all packages to their goal locations.
    For each package, it calculates the minimal drive steps required for a vehicle to pick it up,
    transport it to the goal, and perform necessary load/unload actions, considering vehicle capacities.

    # Assumptions
    - Each package can be handled by a single vehicle optimally.
    - Vehicles can be rerouted instantly without conflicting routes.
    - Road networks are static and fully connected (solvable states only).

    # Heuristic Initialization
    - Precompute shortest path distances between all locations using BFS on road network.
    - Extract capacity predecessor relationships to determine valid pickup actions.
    - Store goal locations for each package from task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package:
        a. If already at goal, contribute 0.
        b. If in a vehicle: calculate drive steps from vehicle's current location to goal + 1 (drop).
        c. If at a location: find the best vehicle (min drive steps to pickup, then to goal) + 2 (pickup/drop).
    2. Sum all package costs, assuming optimal parallel vehicle usage.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                self.goal_locations[parts[1]] = parts[2]

        # Build road graph and precompute distances
        roads = defaultdict(list)
        for fact in task.static:
            if match(fact, 'road', '*', '*'):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                roads[l1].append(l2)
        self.distances = {}
        for source in roads:
            visited = {}
            queue = deque([(source, 0)])
            while queue:
                loc, dist = queue.popleft()
                if loc in visited:
                    continue
                visited[loc] = dist
                for neighbor in roads[loc]:
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
            self.distances[source] = visited

        # Build capacity predecessor map
        self.cap_pred = {}
        for fact in task.static:
            if match(fact, 'capacity-predecessor', '*', '*'):
                s1, s2 = get_parts(fact)[1], get_parts(fact)[2]
                self.cap_pred[s2] = s1

    def __call__(self, node):
        state = node.state
        vehicle_caps = {}
        vehicle_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'capacity':
                vehicle_caps[parts[1]] = parts[2]
            elif parts[0] == 'at' and parts[1] in vehicle_caps:
                vehicle_locs[parts[1]] = parts[2]

        total = 0
        for pkg, goal_loc in self.goal_locations.items():
            # Find current location and if in vehicle
            in_veh = False
            current_loc = None
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'in' and parts[1] == pkg:
                    veh = parts[2]
                    in_veh = True
                    current_loc = vehicle_locs.get(veh, None)
                    break
            if not in_veh:
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'at' and parts[1] == pkg:
                        current_loc = parts[2]
                        break

            if current_loc == goal_loc:
                continue

            if in_veh:
                # Drive from vehicle's current location to goal
                drive = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                total += drive + 1 if drive != float('inf') else 0
            else:
                min_cost = float('inf')
                pkg_loc = current_loc
                for veh, cap in vehicle_caps.items():
                    if cap not in self.cap_pred:
                        continue  # Can't pick up
                    veh_loc = vehicle_locs.get(veh, None)
                    if not veh_loc:
                        continue
                    # Drive to pkg then to goal
                    d1 = self.distances.get(veh_loc, {}).get(pkg_loc, float('inf'))
                    d2 = self.distances.get(pkg_loc, {}).get(goal_loc, float('inf'))
                    if d1 != float('inf') and d2 != float('inf'):
                        min_cost = min(min_cost, d1 + d2 + 2)
                total += min_cost if min_cost != float('inf') else 0

        return total
