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
    Estimates the number of actions required to move all packages to their goal locations. For each package, it calculates the minimal drive actions needed along with necessary pick-up and drop actions.

    # Assumptions
    - Each package can be handled independently by a suitable vehicle.
    - Vehicles can adjust their capacity as needed (ignoring other packages).
    - Roads form a directed graph, using the shortest path for drive actions.

    # Heuristic Initialization
    1. Extract goals and static road information.
    2. Precompute shortest paths between all pairs of locations using BFS.
    3. Process capacity predecessors to determine valid pick-up/drop actions.

    # Step-By-Step Thinking
    1. For each package:
        a. If at goal: no cost.
        b. If in a vehicle: check if it can be dropped at the goal, add drive and drop actions.
        c. If at a location: find the best vehicle to pick it up, calculate drive actions, pick-up, and drop.
    2. Sum costs for all packages.
    """

    def __init__(self, task):
        self.goal_locations = self._extract_goals(task.goals)
        self.road_graph = self._build_road_graph(task.static)
        self.cap_predecessors, self.successors = self._process_capacity_predecessors(task.static)
        self.shortest_paths = self._precompute_shortest_paths()

    def _extract_goals(self, goals):
        goal_locs = {}
        for goal in goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                goal_locs[parts[1]] = parts[2]
        return goal_locs

    def _build_road_graph(self, static):
        road_graph = defaultdict(list)
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                road_graph[parts[1]].append(parts[2])
        return road_graph

    def _process_capacity_predecessors(self, static):
        cap_pre = {}
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == 'capacity-predecessor' and len(parts) == 3:
                cap_pre[parts[2]] = parts[1]
        successors = defaultdict(list)
        for s2, s1 in cap_pre.items():
            successors[s1].append(s2)
        return cap_pre, successors

    def _precompute_shortest_paths(self):
        locations = set(self.road_graph.keys())
        for neighbors in self.road_graph.values():
            locations.update(neighbors)
        shortest_paths = {}
        for loc in locations:
            shortest_paths[loc] = self._bfs(loc)
        return shortest_paths

    def _bfs(self, start):
        visited = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        return visited

    def __call__(self, node):
        state = node.state
        cost = 0

        vehicles = set()
        vehicle_caps = {}
        vehicle_locs = {}
        package_info = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'capacity' and len(parts) == 3:
                vehicles.add(parts[1])
                vehicle_caps[parts[1]] = parts[2]
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in vehicles:
                    vehicle_locs[obj] = loc
                else:
                    package_info[obj] = ('at', loc)
            elif parts[0] == 'in' and len(parts) == 3:
                package_info[parts[1]] = ('in', parts[2])

        for package, goal_loc in self.goal_locations.items():
            current = package_info.get(package, ('at', None))
            if current[1] == goal_loc:
                continue

            if current[0] == 'in':
                vehicle = current[1]
                if vehicle not in vehicle_caps or vehicle not in vehicle_locs:
                    cost += 1000
                    continue
                cap = vehicle_caps[vehicle]
                if cap not in self.successors or not self.successors[cap]:
                    cost += 1000
                    continue
                current_loc = vehicle_locs[vehicle]
                dist = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
                if dist == float('inf'):
                    cost += 1000
                else:
                    cost += dist + 1
            else:
                current_loc = current[1]
                min_cost = float('inf')
                for veh in vehicles:
                    if veh not in vehicle_caps or veh not in vehicle_locs:
                        continue
                    cap = vehicle_caps[veh]
                    if cap not in self.cap_predecessors:
                        continue
                    veh_loc = vehicle_locs[veh]
                    d1 = self.shortest_paths.get(veh_loc, {}).get(current_loc, float('inf'))
                    d2 = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
                    if d1 == float('inf') or d2 == float('inf'):
                        continue
                    total = d1 + d2 + 2
                    if total < min_cost:
                        min_cost = total
                cost += min_cost if min_cost != float('inf') else 1000

        return cost
