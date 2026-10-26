from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions required to move all packages to their goal locations.
    For each package, considers drive actions needed for vehicle movement, pick-up/drop costs,
    and vehicle capacity constraints. Roads are treated as directed edges for shortest path calculations.

    # Assumptions
    - Roads are directed; drive actions require explicit road connections.
    - Vehicles can only pick up packages if their current capacity allows (based on capacity-predecessor).
    - Each package is handled independently; potential parallel actions are not considered.
    - Static facts (roads, capacity hierarchy) are invariant across all states.

    # Heuristic Initialization
    1. Extract road network and build directed adjacency list.
    2. Precompute shortest paths between all location pairs using BFS.
    3. Extract capacity-predecessor relationships to determine valid pickup capacities.
    4. Parse goal locations for each package from task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package:
        a. If already at goal: cost += 0.
        b. If in a vehicle:
            i. Calculate drive steps from vehicle's current location to goal.
            ii. Add 1 for drop action.
        c. If not in a vehicle:
            i. Find all vehicles that can pick it up (valid capacity).
            ii. For each valid vehicle, compute:
                - Drive steps from vehicle to package.
                - Pick-up (1 action).
                - Drive steps from package location to goal.
                - Drop (1 action).
            iii. Take minimum cost across all valid vehicles.
    2. Sum costs for all packages.
    """

    def __init__(self, task):
        # Extract static road and capacity information
        self.roads = set()
        self.capacity_predecessor = {}
        allowed_pickup_capacities = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "road":
                self.roads.add((parts[1], parts[2]))
            elif parts[0] == "capacity-predecessor":
                s1, s2 = parts[1], parts[2]
                self.capacity_predecessor[s1] = s2
                allowed_pickup_capacities.add(s2)

        self.allowed_pickup_capacities = allowed_pickup_capacities

        # Build directed adjacency list and locations set
        self.adj = defaultdict(list)
        self.locations = set()
        for l1, l2 in self.roads:
            self.adj[l1].append(l2)
            self.locations.update({l1, l2})

        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        for start in self.locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                current_dist = visited[current]
                for neighbor in self.adj[current]:
                    if neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        queue.append(neighbor)
            # Store distances (inf if unreachable)
            for loc in self.locations:
                self.shortest_paths[(start, loc)] = visited.get(loc, float("inf"))

        # Extract package goal locations
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and parts[2].startswith("l"):
                self.goal_locations[parts[1]] = parts[2]

    def __call__(self, node):
        state = node.state
        current_package_locs = {}
        in_vehicle = {}
        vehicle_locs = {}
        vehicle_caps = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj.startswith("p"):
                    current_package_locs[obj] = loc
                elif obj.startswith("v"):
                    vehicle_locs[obj] = loc
            elif parts[0] == "in":
                pkg, veh = parts[1], parts[2]
                in_vehicle[pkg] = veh
            elif parts[0] == "capacity":
                veh, cap = parts[1], parts[2]
                vehicle_caps[veh] = cap

        total_cost = 0
        for pkg, goal_loc in self.goal_locations.items():
            if pkg in in_vehicle:
                # Package is in a vehicle
                veh = in_vehicle[pkg]
                veh_loc = vehicle_locs.get(veh, None)
                if not veh_loc:
                    continue  # invalid state
                dist = self.shortest_paths.get((veh_loc, goal_loc), float("inf"))
                total_cost += dist + 1  # drive + drop
            else:
                # Package is not in a vehicle
                current_loc = current_package_locs.get(pkg)
                if not current_loc or current_loc == goal_loc:
                    continue  # already at goal or invalid

                min_cost = float("inf")
                for veh, veh_loc in vehicle_locs.items():
                    if vehicle_caps.get(veh) not in self.allowed_pickup_capacities:
                        continue  # vehicle can't pick up

                    # Cost: drive to pkg + pick-up + drive to goal + drop
                    to_pkg = self.shortest_paths.get((veh_loc, current_loc), float("inf"))
                    to_goal = self.shortest_paths.get((current_loc, goal_loc), float("inf"))
                    cost = to_pkg + 1 + to_goal + 1
                    if cost < min_cost:
                        min_cost = cost

                if min_cost != float("inf"):
                    total_cost += min_cost

        return total_cost
