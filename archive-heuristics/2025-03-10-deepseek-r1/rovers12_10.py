from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rovers12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the number of actions required for rovers to achieve all communication goals, including soil data, rock data, and image data. It considers navigation, sampling, calibration, imaging, and communication actions, using precomputed shortest paths for efficient calculation.

    Assumptions:
    - Each rover can traverse between waypoints as defined by static 'can_traverse' predicates.
    - Soil and rock samples are available until collected.
    - Cameras can be calibrated if the rover is at the correct waypoint.
    - Communication requires the rover to be at a waypoint visible from the lander's location.

    Heuristic Initialization:
    - Preprocesses static information including rover traversal graphs, camera details, and visibility.
    - Computes shortest paths for each rover's navigation capabilities.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each unachieved goal, determine the minimal actions required.
    2. For soil/rock data: Check if collected, compute navigation and communication costs.
    3. For image data: Check if taken, compute calibration, imaging, and communication costs.
    4. Sum the minimal costs for all goals, assuming optimal rover assignments.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract lander's location
        self.lander_location = None
        # Extract store_of mappings
        self.store_to_rover = {}
        # Extract visible predicates
        self.visible = defaultdict(set)
        # Rover traversal graphs and shortest paths
        self.rover_paths = defaultdict(lambda: {'graph': defaultdict(list), 'shortest_paths': {}})
        # Camera information: calibration target, on_board rover, supported modes
        self.camera_info = defaultdict(lambda: {'calibration_target': None, 'on_board': None, 'supports': []})
        # Objective visible_from waypoints
        self.objective_visible_from = defaultdict(list)

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'at_lander' and parts[1] == 'general':
                self.lander_location = parts[2]
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
            elif parts[0] == 'visible':
                x, y = parts[1], parts[2]
                self.visible[x].add(y)
            elif parts[0] == 'can_traverse':
                rover, x, y = parts[1], parts[2], parts[3]
                self.rover_paths[rover]['graph'][x].append(y)
            elif parts[0] == 'calibration_target':
                camera, obj = parts[1], parts[2]
                self.camera_info[camera]['calibration_target'] = obj
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info[camera]['on_board'] = rover
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info[camera]['supports'].append(mode)
            elif parts[0] == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.objective_visible_from[obj].append(wp)

        # Precompute shortest paths for each rover
        for rover in self.rover_paths:
            graph = self.rover_paths[rover]['graph']
            waypoints = set()
            for x in graph:
                waypoints.add(x)
                for y in graph[x]:
                    waypoints.add(y)
            waypoints = list(waypoints)
            shortest_paths = {}
            for start in waypoints:
                visited = {start: 0}
                queue = deque([start])
                while queue:
                    current = queue.popleft()
                    for neighbor in graph.get(current, []):
                        if neighbor not in visited:
                            visited[neighbor] = visited[current] + 1
                            queue.append(neighbor)
                shortest_paths[start] = visited
            self.rover_paths[rover]['shortest_paths'] = shortest_paths

    def __call__(self, node):
        state = node.state
        if self.lander_location is None:
            return 0

        # Extract current state information
        current_positions = {}
        equipped_for_soil = set()
        equipped_for_rock = set()
        equipped_for_imaging = set()
        store_status = {}
        have_soil = defaultdict(set)
        have_rock = defaultdict(set)
        have_image = defaultdict(lambda: defaultdict(set))
        calibrated_cameras = defaultdict(set)
        at_soil_samples = set()
        at_rock_samples = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('rover'):
                rover, wp = parts[1], parts[2]
                current_positions[rover] = wp
            elif parts[0] == 'equipped_for_soil_analysis':
                equipped_for_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                equipped_for_rock.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                equipped_for_imaging.add(parts[1])
            elif parts[0] == 'empty':
                store = parts[1]
                rover = self.store_to_rover.get(store)
                if rover is not None:
                    store_status[rover] = 'empty'
            elif parts[0] == 'full':
                store = parts[1]
                rover = self.store_to_rover.get(store)
                if rover is not None:
                    store_status[rover] = 'full'
            elif parts[0] == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                have_soil[rover].add(wp)
            elif parts[0] == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                have_rock[rover].add(wp)
            elif parts[0] == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image[rover][obj].add(mode)
            elif parts[0] == 'calibrated':
                camera, rover = parts[1], parts[2]
                calibrated_cameras[rover].add(camera)
            elif parts[0] == 'at_soil_sample':
                at_soil_samples.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                at_rock_samples.add(parts[1])

        total_cost = 0
        lander_wp = self.lander_location
        visible_to_lander = [x for x in self.visible if lander_wp in self.visible[x]]

        for goal in self.goals:
            if goal in state:
                continue
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                wp = parts[1]
                min_cost = float('inf')
                # Check if any rover has the sample
                for rover in have_soil:
                    if wp in have_soil[rover]:
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        rover_paths = self.rover_paths[rover]['shortest_paths'].get(current_pos, {})
                        min_nav = min((rover_paths.get(x, float('inf')) for x in visible_to_lander), default=float('inf'))
                        if min_nav != float('inf'):
                            min_cost = min(min_cost, min_nav + 1)
                # If not, check if sample is present and needs collection
                if min_cost == float('inf') and wp in at_soil_samples:
                    for rover in equipped_for_soil:
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        store_stat = store_status.get(rover, 'empty')
                        drop_cost = 1 if store_stat == 'full' else 0
                        rover_paths = self.rover_paths[rover]['shortest_paths']
                        from_current = rover_paths.get(current_pos, {})
                        to_wp_cost = from_current.get(wp, float('inf'))
                        if to_wp_cost == float('inf'):
                            continue
                        from_wp = rover_paths.get(wp, {})
                        to_lander_cost = min((from_wp.get(x, float('inf')) for x in visible_to_lander), default=float('inf'))
                        if to_lander_cost == float('inf'):
                            continue
                        cost = drop_cost + to_wp_cost + 1 + to_lander_cost + 1
                        min_cost = min(min_cost, cost)
                if min_cost == float('inf'):
                    min_cost = 0
                total_cost += min_cost
            elif parts[0] == 'communicated_rock_data':
                wp = parts[1]
                min_cost = float('inf')
                for rover in have_rock:
                    if wp in have_rock[rover]:
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        rover_paths = self.rover_paths[rover]['shortest_paths'].get(current_pos, {})
                        min_nav = min((rover_paths.get(x, float('inf')) for x in visible_to_lander), default=float('inf'))
                        if min_nav != float('inf'):
                            min_cost = min(min_cost, min_nav + 1)
                if min_cost == float('inf') and wp in at_rock_samples:
                    for rover in equipped_for_rock:
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        store_stat = store_status.get(rover, 'empty')
                        drop_cost = 1 if store_stat == 'full' else 0
                        rover_paths = self.rover_paths[rover]['shortest_paths']
                        from_current = rover_paths.get(current_pos, {})
                        to_wp_cost = from_current.get(wp, float('inf'))
                        if to_wp_cost == float('inf'):
                            continue
                        from_wp = rover_paths.get(wp, {})
                        to_lander_cost = min((from_wp.get(x, float('inf')) for x in visible_to_lander), default=float('inf'))
                        if to_lander_cost == float('inf'):
                            continue
                        cost = drop_cost + to_wp_cost + 1 + to_lander_cost + 1
                        min_cost = min(min_cost, cost)
                if min_cost == float('inf'):
                    min_cost = 0
                total_cost += min_cost
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                min_cost = float('inf')
                # Check if any rover has the image
                for rover in have_image:
                    if obj in have_image[rover] and mode in have_image[rover][obj]:
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        rover_paths = self.rover_paths[rover]['shortest_paths'].get(current_pos, {})
                        min_nav = min((rover_paths.get(x, float('inf')) for x in visible_to_lander), default=float('inf'))
                        if min_nav != float('inf'):
                            min_cost = min(min_cost, min_nav + 1)
                # If not, find suitable camera and compute cost
                if min_cost == float('inf'):
                    for camera in self.camera_info:
                        cam = self.camera_info[camera]
                        if mode not in cam['supports']:
                            continue
                        rover = cam['on_board']
                        if not rover or rover not in equipped_for_imaging:
                            continue
                        cal_target = cam['calibration_target']
                        cal_wps = self.objective_visible_from.get(cal_target, [])
                        current_pos = current_positions.get(rover)
                        if not current_pos or not cal_wps:
                            continue
                        # Calibration cost
                        if camera not in calibrated_cameras.get(rover, set()):
                            rover_paths = self.rover_paths[rover]['shortest_paths']
                            cal_dist = min((rover_paths.get(current_pos, {}).get(wp, float('inf')) for wp in cal_wps), default=float('inf'))
                            if cal_dist == float('inf'):
                                continue
                            cal_cost = cal_dist + 1
                            cal_location = min(cal_wps, key=lambda wp: rover_paths.get(current_pos, {}).get(wp, float('inf')))
                        else:
                            cal_cost = 0
                            cal_location = current_pos
                        # Take image at visible_from waypoint
                        img_wps = self.objective_visible_from.get(obj, [])
                        if not img_wps:
                            continue
                        rover_paths = self.rover_paths[rover]['shortest_paths']
                        if cal_cost > 0:
                            img_dist = min((rover_paths.get(cal_location, {}).get(wp, float('inf')) for wp in img_wps), default=float('inf'))
                        else:
                            img_dist = min((rover_paths.get(current_pos, {}).get(wp, float('inf')) for wp in img_wps), default=float('inf'))
                        if img_dist == float('inf'):
                            continue
                        # Navigate to lander-visible waypoint
                        from_img = rover_paths.get(wp, {}) if (wp := img_wps[0]) else {}
                        comm_dist = min((from_img.get(x, float('inf')) for x in visible_to_lander), default=float('inf'))
                        if comm_dist == float('inf'):
                            continue
                        total_cost_cam = cal_cost + img_dist + 1 + comm_dist + 1
                        min_cost = min(min_cost, total_cost_cam)
                if min_cost == float('inf'):
                    min_cost = 0
                total_cost += min_cost

        return total_cost
