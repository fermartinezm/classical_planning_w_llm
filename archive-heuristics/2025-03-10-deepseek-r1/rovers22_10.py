from fnmatch import fnmatch
from collections import defaultdict
import math
from heapq import heappush, heappop

class rovers22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goals by considering the minimal steps needed for each rover to collect samples, take images, and communicate data. It accounts for navigation, equipment requirements, and static constraints.

    # Assumptions
    - Each rover can handle multiple tasks, but the heuristic sums individual minimal costs.
    - The shortest path between waypoints is precomputed for each rover.
    - Static facts (like rover capabilities and camera details) are preprocessed for efficiency.

    # Heuristic Initialization
    - Extracts static information: rover paths, lander positions, camera details, and visibility.
    - Precomputes shortest paths between waypoints for each rover using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Soil/Rock Data**: For each uncommunicated sample, calculate the cost to navigate, sample, and communicate.
    2. **Image Data**: For each uncommunicated image, calculate calibration, navigation, image capture, and communication costs.
    3. **Sum Costs**: Aggregate minimal costs for all unachieved goals.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract can_traverse for each rover and build shortest paths
        self.rovers = set()
        self.graph = defaultdict(dict)  # rover -> {from_wp: [to_wps]}
        for fact in self.static:
            parts = self.get_parts(fact)
            if parts[0] == 'can_traverse':
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                if from_wp not in self.graph[rover]:
                    self.graph[rover][from_wp] = []
                self.graph[rover][from_wp].append(to_wp)

        # Precompute shortest paths for each rover
        self.shortest_paths = defaultdict(dict)  # rover -> {from: {to: dist}}
        for rover in self.rovers:
            self.shortest_paths[rover] = self.bfs_all(rover)

        # Extract lander positions
        self.lander_waypoints = {}
        for fact in self.static:
            if self.match(fact, 'at_lander', '*', '*'):
                parts = self.get_parts(fact)
                self.lander_waypoints[parts[1]] = parts[2]

        # Build visible_from for objectives and visible for lander communication
        self.visible_from = defaultdict(list)  # objective -> [waypoints]
        self.visible = set()  # (from_wp, to_wp)
        for fact in self.static:
            parts = self.get_parts(fact)
            if parts[0] == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.visible_from[obj].append(wp)
            elif parts[0] == 'visible':
                self.visible.add((parts[1], parts[2]))

        # For each lander waypoint, collect waypoints visible to it
        self.visible_to_lander = defaultdict(list)
        for lander, wp in self.lander_waypoints.items():
            self.visible_to_lander[lander] = [f for f, t in self.visible if t == wp]

        # Camera and rover equipment info
        self.calibration_target = {}
        self.supports = defaultdict(list)
        self.on_board = defaultdict(list)
        self.store_of = {}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        for fact in self.static:
            parts = self.get_parts(fact)
            if parts[0] == 'calibration_target':
                self.calibration_target[parts[1]] = parts[2]
            elif parts[0] == 'supports':
                self.supports[parts[1]].append(parts[2])
            elif parts[0] == 'on_board':
                self.on_board[parts[2]].append(parts[1])
            elif parts[0] == 'store_of':
                self.store_of[parts[2]] = parts[1]
            elif parts[0] == 'equipped_for_soil_analysis':
                self.equipped_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_rock.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_imaging.add(parts[1])

    def get_parts(self, fact):
        return fact[1:-1].split()

    def match(self, fact, *pattern):
        parts = self.get_parts(fact)
        return len(parts) == len(pattern) and all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

    def bfs_all(self, rover):
        graph = self.graph[rover]
        nodes = set(graph.keys())
        for tos in graph.values():
            nodes.update(tos)
        nodes = list(nodes)
        distances = {}
        for node in nodes:
            distances[node] = {}
            queue = [(0, node)]
            visited = set()
            while queue:
                dist, current = heappop(queue)
                if current in visited:
                    continue
                visited.add(current)
                distances[node][current] = dist
                for neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        heappush(queue, (dist + 1, neighbor))
        return distances

    def min_distance(self, rover, from_wp, to_wps):
        if rover not in self.shortest_paths or from_wp not in self.shortest_paths[rover]:
            return math.inf
        dists = self.shortest_paths[rover][from_wp]
        min_dist = math.inf
        for wp in to_wps:
            if wp in dists:
                min_dist = min(min_dist, dists[wp])
        return min_dist

    def __call__(self, node):
        state = node.state
        current_pos = {}
        has_soil = defaultdict(set)  # rover -> {waypoint}
        has_rock = defaultdict(set)
        has_image = defaultdict(set)  # rover -> {(objective, mode)}
        calibrated = defaultdict(set)  # rover -> {camera}
        stored = {}  # rover -> store state

        for fact in state:
            parts = self.get_parts(fact)
            if parts[0] == 'at':
                current_pos[parts[1]] = parts[2]
            elif parts[0] == 'have_soil_analysis':
                has_soil[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis':
                has_rock[parts[1]].add(parts[2])
            elif parts[0] == 'have_image':
                has_image[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                calibrated[parts[2]].add(parts[1])
            elif parts[0] == 'empty':
                stored[self.store_of[parts[1]]] = True
            elif parts[0] == 'full':
                stored[self.store_of[parts[1]]] = False

        remaining_goals = [g for g in self.goals if g not in state]
        total_cost = 0

        for goal in remaining_goals:
            g_parts = self.get_parts(goal)
            if g_parts[0] == 'communicated_soil_data':
                wp = g_parts[1]
                min_cost = math.inf
                for lander, l_wp in self.lander_waypoints.items():
                    comm_wps = self.visible_to_lander[lander]
                    for rover in self.rovers:
                        if rover not in self.equipped_soil:
                            continue
                        store = self.store_of.get(rover, None)
                        if store is None or not stored.get(rover, True):
                            continue
                        if wp in has_soil[rover]:
                            pos = current_pos.get(rover, None)
                            if pos is None:
                                continue
                            cost = self.min_distance(rover, pos, comm_wps)
                            if cost != math.inf:
                                min_cost = min(min_cost, cost + 1)
                        else:
                            if not any(self.match(f, 'at_soil_sample', wp) for f in state):
                                continue
                            r_pos = current_pos.get(rover, None)
                            if r_pos is None:
                                continue
                            cost_to_wp = self.min_distance(rover, r_pos, [wp])
                            if cost_to_wp == math.inf:
                                continue
                            cost_from_wp = self.min_distance(rover, wp, comm_wps)
                            if cost_from_wp == math.inf:
                                continue
                            total = cost_to_wp + 1 + cost_from_wp + 1
                            min_cost = min(min_cost, total)
                if min_cost != math.inf:
                    total_cost += min_cost
            elif g_parts[0] == 'communicated_rock_data':
                wp = g_parts[1]
                min_cost = math.inf
                for lander, l_wp in self.lander_waypoints.items():
                    comm_wps = self.visible_to_lander[lander]
                    for rover in self.rovers:
                        if rover not in self.equipped_rock:
                            continue
                        store = self.store_of.get(rover, None)
                        if store is None or not stored.get(rover, True):
                            continue
                        if wp in has_rock[rover]:
                            pos = current_pos.get(rover, None)
                            if pos is None:
                                continue
                            cost = self.min_distance(rover, pos, comm_wps)
                            if cost != math.inf:
                                min_cost = min(min_cost, cost + 1)
                        else:
                            if not any(self.match(f, 'at_rock_sample', wp) for f in state):
                                continue
                            r_pos = current_pos.get(rover, None)
                            if r_pos is None:
                                continue
                            cost_to_wp = self.min_distance(rover, r_pos, [wp])
                            if cost_to_wp == math.inf:
                                continue
                            cost_from_wp = self.min_distance(rover, wp, comm_wps)
                            if cost_from_wp == math.inf:
                                continue
                            total = cost_to_wp + 1 + cost_from_wp + 1
                            min_cost = min(min_cost, total)
                if min_cost != math.inf:
                    total_cost += min_cost
            elif g_parts[0] == 'communicated_image_data':
                obj, mode = g_parts[1], g_parts[2]
                min_cost = math.inf
                for rover in self.rovers:
                    if rover not in self.equipped_imaging:
                        continue
                    for cam in self.on_board[rover]:
                        if mode not in self.supports[cam]:
                            continue
                        target_obj = self.calibration_target.get(cam, None)
                        if target_obj is None:
                            continue
                        cal_wps = self.visible_from.get(target_obj, [])
                        if not cal_wps:
                            continue
                        r_pos = current_pos.get(rover, None)
                        if r_pos is None:
                            continue
                        if cam in calibrated[rover]:
                            cal_cost = 0
                            cal_pos = r_pos
                        else:
                            cal_cost = self.min_distance(rover, r_pos, cal_wps)
                            if cal_cost == math.inf:
                                continue
                            cal_cost += 1
                            cal_pos = cal_wps[0]
                        img_wps = self.visible_from.get(obj, [])
                        if not img_wps:
                            continue
                        img_cost = self.min_distance(rover, cal_pos, img_wps)
                        if img_cost == math.inf:
                            continue
                        img_cost += 1
                        img_pos = img_wps[0]
                        comm_cost = math.inf
                        for lander, l_wp in self.lander_waypoints.items():
                            comm_wps = self.visible_to_lander[lander]
                            cost = self.min_distance(rover, img_pos, comm_wps)
                            if cost != math.inf:
                                comm_cost = min(comm_cost, cost + 1)
                        if comm_cost == math.inf:
                            continue
                        total = cal_cost + img_cost + comm_cost
                        if total < min_cost:
                            min_cost = total
                if min_cost != math.inf:
                    total_cost += min_cost

        return total_cost if total_cost != math.inf else 0
