from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, pattern):
    parts = fact[1:-1].split()
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern_parts))

class rovers23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the number of actions required for rovers to collect samples, take images, and communicate data. It considers movement between waypoints, calibration of cameras, and task assignments based on rover capabilities.

    Assumptions:
    - Each rover can handle multiple tasks but must be properly equipped.
    - The lander's position is static and known.
    - Movement between waypoints uses the shortest path based on can_traverse.

    Heuristic Initialization:
    - Extracts static information including lander position, rover equipment, camera details, and waypoint visibility.
    - Precomputes shortest paths for each rover's possible movements.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify unmet goals for soil, rock, and image data.
    2. For each unmet goal, determine the minimal steps required:
        a. Soil/Rock: Navigate to sample location, collect sample, communicate.
        b. Images: Calibrate camera, navigate to imaging location, take image, communicate.
    3. Estimate movement costs using precomputed shortest paths.
    4. Sum all required actions for a total heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.lander_waypoint = None
        self.visible_to_lander = set()
        self.rover_graphs = defaultdict(dict)
        self.rover_equipment = defaultdict(lambda: defaultdict(bool))
        self.camera_info = defaultdict(dict)
        self.objective_visible_waypoints = defaultdict(set)
        self.store_to_rover = {}
        self.calibration_targets = {}
        self.visible_from = defaultdict(set)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_waypoint = parts[2]
            elif parts[0] == 'visible':
                from_wp, to_wp = parts[1], parts[2]
                if to_wp == self.lander_waypoint:
                    self.visible_to_lander.add(from_wp)
            elif parts[0] == 'can_traverse':
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                if from_wp not in self.rover_graphs[rover]:
                    self.rover_graphs[rover][from_wp] = set()
                self.rover_graphs[rover][from_wp].add(to_wp)
            elif parts[0] == 'equipped_for_soil_analysis':
                self.rover_equipment[parts[1]]['soil'] = True
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_equipment[parts[1]]['rock'] = True
            elif parts[0] == 'equipped_for_imaging':
                self.rover_equipment[parts[1]]['imaging'] = True
            elif parts[0] == 'on_board':
                self.camera_info[parts[1]]['on_rover'] = parts[2]
            elif parts[0] == 'supports':
                cam, mode = parts[1], parts[2]
                if 'supports' not in self.camera_info[cam]:
                    self.camera_info[cam]['supports'] = set()
                self.camera_info[cam]['supports'].add(mode)
            elif parts[0] == 'calibration_target':
                cam, obj = parts[1], parts[2]
                self.camera_info[cam]['calibration_target'] = obj
                self.calibration_targets[cam] = obj
            elif parts[0] == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.objective_visible_waypoints[obj].add(wp)
                self.visible_from[obj].add(wp)
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover

    def get_distance(self, rover, start, end, graph):
        if start == end:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            if node == end:
                return dist
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph.get(node, []):
                queue.append((neighbor, dist + 1))
        return float('inf')

    def __call__(self, node):
        state = node.state
        total_cost = 0

        current_rovers = {}
        have_soil = defaultdict(set)
        have_rock = defaultdict(set)
        have_image = defaultdict(set)
        calibrated = defaultdict(set)
        stores = defaultdict(lambda: 'empty')

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[2] == 'rover':
                rover, wp = parts[1], parts[3]
                current_rovers[rover] = wp
            elif parts[0] == 'at' and parts[2] == 'waypoint':
                rover, wp = parts[1], parts[3]
                current_rovers[rover] = wp
            elif parts[0] == 'have_soil_analysis':
                have_soil[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis':
                have_rock[parts[1]].add(parts[2])
            elif parts[0] == 'have_image':
                have_image[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                calibrated[parts[2]].add(parts[1])
            elif parts[0] == 'empty':
                stores[self.store_to_rover[parts[1]]] = 'empty'
            elif parts[0] == 'full':
                stores[self.store_to_rover[parts[1]]] = 'full'

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                wp = parts[1]
                if goal in state:
                    continue
                min_cost = float('inf')
                for rover in have_soil:
                    if wp in have_soil[rover]:
                        current_pos = current_rovers.get(rover)
                        if current_pos:
                            distances = [self.get_distance(rover, current_pos, vwp, self.rover_graphs[rover]) for vwp in self.visible_to_lander]
                            if distances:
                                cost = min(distances) + 1
                                min_cost = min(min_cost, cost)
                if min_cost != float('inf'):
                    total_cost += min_cost
                    continue
                if f'(at_soil_sample {wp})' in state:
                    min_cost_collect = float('inf')
                    for rover in self.rover_equipment:
                        if self.rover_equipment[rover].get('soil', False) and stores[rover] == 'empty':
                            current_pos = current_rovers.get(rover)
                            if current_pos:
                                dist_to_wp = self.get_distance(rover, current_pos, wp, self.rover_graphs[rover])
                                distances_comm = [self.get_distance(rover, wp, vwp, self.rover_graphs[rover]) for vwp in self.visible_to_lander]
                                if distances_comm:
                                    cost = dist_to_wp + 1 + min(distances_comm) + 1
                                    min_cost_collect = min(min_cost_collect, cost)
                    if min_cost_collect != float('inf'):
                        total_cost += min_cost_collect
                    else:
                        total_cost += 10
                else:
                    total_cost += 10
            elif parts[0] == 'communicated_rock_data':
                wp = parts[1]
                if goal in state:
                    continue
                min_cost = float('inf')
                for rover in have_rock:
                    if wp in have_rock[rover]:
                        current_pos = current_rovers.get(rover)
                        if current_pos:
                            distances = [self.get_distance(rover, current_pos, vwp, self.rover_graphs[rover]) for vwp in self.visible_to_lander]
                            if distances:
                                cost = min(distances) + 1
                                min_cost = min(min_cost, cost)
                if min_cost != float('inf'):
                    total_cost += min_cost
                    continue
                if f'(at_rock_sample {wp})' in state:
                    min_cost_collect = float('inf')
                    for rover in self.rover_equipment:
                        if self.rover_equipment[rover].get('rock', False) and stores[rover] == 'empty':
                            current_pos = current_rovers.get(rover)
                            if current_pos:
                                dist_to_wp = self.get_distance(rover, current_pos, wp, self.rover_graphs[rover])
                                distances_comm = [self.get_distance(rover, wp, vwp, self.rover_graphs[rover]) for vwp in self.visible_to_lander]
                                if distances_comm:
                                    cost = dist_to_wp + 1 + min(distances_comm) + 1
                                    min_cost_collect = min(min_cost_collect, cost)
                    if min_cost_collect != float('inf'):
                        total_cost += min_cost_collect
                    else:
                        total_cost += 10
                else:
                    total_cost += 10
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                if goal in state:
                    continue
                min_cost_comm = float('inf')
                for rover in have_image:
                    if (obj, mode) in have_image[rover]:
                        current_pos = current_rovers.get(rover)
                        if current_pos:
                            distances = [self.get_distance(rover, current_pos, vwp, self.rover_graphs[rover]) for vwp in self.visible_to_lander]
                            if distances:
                                cost = min(distances) + 1
                                min_cost_comm = min(min_cost_comm, cost)
                if min_cost_comm != float('inf'):
                    total_cost += min_cost_comm
                    continue
                min_cost_take = float('inf')
                for cam in self.camera_info:
                    if mode not in self.camera_info[cam].get('supports', set()):
                        continue
                    rover = self.camera_info[cam].get('on_rover')
                    if not rover or not self.rover_equipment[rover].get('imaging', False):
                        continue
                    cal_obj = self.camera_info[cam].get('calibration_target')
                    if not cal_obj:
                        continue
                    cal_wps = self.visible_from.get(cal_obj, set())
                    img_wps = self.visible_from.get(obj, set())
                    if not img_wps:
                        continue
                    current_pos = current_rovers.get(rover)
                    if not current_pos:
                        continue
                    is_calibrated = cam in calibrated.get(rover, set())
                    cal_cost = 0
                    if not is_calibrated:
                        if not cal_wps:
                            continue
                        cal_distances = [self.get_distance(rover, current_pos, wp, self.rover_graphs[rover]) for wp in cal_wps]
                        if not cal_distances:
                            continue
                        cal_cost = min(cal_distances) + 1
                        cal_pos = cal_wps[cal_distances.index(min(cal_distances))]
                    else:
                        cal_pos = current_pos
                    img_distances = [self.get_distance(rover, cal_pos, wp, self.rover_graphs[rover]) for wp in img_wps]
                    if not img_distances:
                        continue
                    img_cost = min(img_distances) + 1
                    comm_distances = [self.get_distance(rover, wp, vwp, self.rover_graphs[rover]) for wp in img_wps for vwp in self.visible_to_lander]
                    if not comm_distances:
                        continue
                    comm_cost = min(comm_distances) + 1
                    total_cost_cam = cal_cost + img_cost + comm_cost
                    min_cost_take = min(min_cost_take, total_cost_cam)
                if min_cost_take != float('inf'):
                    total_cost += min_cost_take
                else:
                    total_cost += 10
        return total_cost
