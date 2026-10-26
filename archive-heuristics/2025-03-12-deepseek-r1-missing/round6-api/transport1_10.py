from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions required to move all packages to their goal locations by considering the shortest driving paths and necessary pick-up/drop actions. Vehicles' capacities are checked to ensure they can carry packages.

    # Assumptions
    - Roads are bidirectional.
    - Vehicles can carry packages if their current capacity allows (exists a predecessor in capacity hierarchy).
    - The shortest path between locations is used for driving steps.

    # Heuristic Initialization
    - Extracts static information (roads, capacity hierarchy, vehicles, and goals).
    - Builds a graph of locations and precomputes all-pairs shortest paths using BFS.
    - Identifies goal locations for each package.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package in the goal:
        a. If already at the goal, skip.
        b. If in a vehicle, calculate driving distance from vehicle's current location to goal plus a drop action.
        c. If not in a vehicle, find the closest vehicle (considering capacity) and calculate:
            i. Distance from vehicle to package's location.
            ii. Distance from package's location to goal.
            iii. Add pick-up and drop actions.
    2. Sum all calculated actions for an overall heuristic estimate.
    """

    def __init__(self, task):
        """Initialize with static data, precompute shortest paths, and extract goals."""
        self.static = task.static
        self.goals = task.goals

        # Extract capacity predecessors
        self.cap_pred = {}
        for fact in self.static:
            parts = self._get_parts(fact)
            if parts[0] == 'capacity-predecessor':
                s1, s2 = parts[1], parts[2]
                self.cap_pred[s2] = s1

        # Collect all vehicles from initial state and static
        self.vehicles = set()
        for fact in task.initial_state | self.static:
            parts = self._get_parts(fact)
            if parts[0] == 'capacity':
                self.vehicles.add(parts[1])

        # Build road graph and compute shortest paths
        roads = set()
        locations = set()
        for fact in self.static:
            parts = self._get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                roads.add((l1, l2))
                roads.add((l2, l1))
                locations.update({l1, l2})

        self.graph = defaultdict(list)
        for l1, l2 in roads:
            self.graph[l1].append(l2)

        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        for start in locations:
            visited = {start: 0}
            queue = deque([(start, 0)])
            while queue:
                curr, dist = queue.popleft()
                for neighbor in self.graph.get(curr, []):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            for end in locations:
                self.shortest_paths[(start, end)] = visited.get(end, float('inf'))

        # Extract goal locations for each package
        self.goal_locs = {}
        for goal in self.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                self.goal_locs[parts[1]] = parts[2]

    def _get_parts(self, fact):
        """Split a PDDL fact into components."""
        return fact[1:-1].split()

    def __call__(self, node):
        """Compute heuristic value for the current state."""
        state = node.state
        total = 0

        # Current capacities of vehicles
        capacities = {}
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'capacity':
                capacities[parts[1]] = parts[2]

        # Vehicle locations
        vehicle_loc = {}
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.vehicles:
                vehicle_loc[parts[1]] = parts[2]

        # Packages in vehicles
        in_veh = {}
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'in':
                in_veh[parts[1]] = parts[2]

        for pkg, goal in self.goal_locs.items():
            if pkg in in_veh:
                # Package is in a vehicle
                veh = in_veh[pkg]
                curr = vehicle_loc.get(veh, None)
                if not curr:
                    total += 1000  # Penalty if vehicle location unknown
                    continue
                dist = self.shortest_paths.get((curr, goal), float('inf'))
                total += dist + 1  # Drive to goal + drop
            else:
                # Package is not in a vehicle
                curr_pkg_loc = None
                for fact in state:
                    parts = self._get_parts(fact)
                    if parts[0] == 'at' and parts[1] == pkg:
                        curr_pkg_loc = parts[2]
                        break
                if not curr_pkg_loc or curr_pkg_loc == goal:
                    continue  # Already at goal or invalid
                min_cost = float('inf')
                for veh, veh_loc in vehicle_loc.items():
                    cap = capacities.get(veh, None)
                    if cap not in self.cap_pred:
                        continue  # Cannot pick up
                    dist_to_pkg = self.shortest_paths.get((veh_loc, curr_pkg_loc), float('inf'))
                    dist_to_goal = self.shortest_paths.get((curr_pkg_loc, goal), float('inf'))
                    cost = dist_to_pkg + dist_to_goal + 2  # pick + drop
                    if cost < min_cost:
                        min_cost = cost
                if min_cost == float('inf'):
                    min_cost = 1000  # Penalty if no valid vehicle
                total += min_cost

        return total
