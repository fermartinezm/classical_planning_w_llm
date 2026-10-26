from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    for p, pat in zip(parts, pattern_parts):
        if not fnmatch(p, pat):
            return False
    return True

class rovers7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goals by considering navigation, sample collection, imaging, and communication. It breaks down each goal into required steps and sums the minimal actions needed.

    # Assumptions
    - Rovers can traverse between waypoints as per precomputed shortest paths.
    - Soil/rock samples not present are assumed to be already collected.
    - Optimistic handling of rover stores (assuming drops can be done if needed).
    - Cameras can be calibrated if needed for imaging.

    # Heuristic Initialization
    - Extracts static information: lander's waypoint, rover traversal graphs, camera calibration targets, and objective visibility.
    - Precomputes shortest paths for each rover's navigation.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each uncommunicated soil/rock data:
        a. If a rover has the analysis, estimate navigation to communicate.
        b. If sample is present, estimate collection and communication.
        c. If sample is missing, assume analysis exists and communicate.
    2. For each uncommunicated image data:
        a. If a rover has the image, estimate navigation to communicate.
        b. If not, calibrate camera, take image, and communicate.
    3. Sum all actions, considering the minimal paths and required steps.
    """

    def __init__(self, task):
        self.static = task.static
        self.goals = task.goals

        # Extract lander's waypoint
        self.lander_waypoint = None
        for fact in self.static:
            if match(fact, 'at_lander * *'):
                parts = get_parts(fact)
                self.lander_waypoint = parts[2]
                break

        # Communication waypoints (visible from lander's waypoint)
        self.communication_waypoints = set()
        for fact in self.static:
            if match(fact, 'visible * *'):
                parts = get_parts(fact)
                from_wp, to_wp = parts[1], parts[2]
                if to_wp == self.lander_waypoint:
                    self.communication_waypoints.add(from_wp)

        # Build rover graphs and precompute distances
        self.rovers = set()
        self.rover_graphs = defaultdict(dict)
        for fact in self.static:
            if match(fact, 'can_traverse * * *'):
                parts = get_parts(fact)
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                self.rovers.add(rover)
                if from_wp not in self.rover_graphs[rover]:
                    self.rover_graphs[rover][from_wp] = []
                self.rover_graphs[rover][from_wp].append(to_wp)

        # Precompute shortest paths for each rover
        self.rover_distances = {}
        for rover in self.rovers:
            graph = self.rover_graphs[rover]
            distances = defaultdict(dict)
            all_waypoints = set()
            for wp in graph:
                all_waypoints.add(wp)
                for neighbor in graph[wp]:
                    all_waypoints.add(neighbor)
            for wp in all_waypoints:
                visited = {}
                queue = deque([(wp, 0)])
                while queue:
                    current, dist = queue.popleft()
                    if current in visited:
                        continue
                    visited[current] = dist
                    for neighbor in graph.get(current, []):
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))
                for node in visited:
                    distances[wp][node] = visited[node]
            self.rover_distances[rover] = distances

        # Camera calibration targets and objective visibility
        self.calibration_target = {}
        self.visible_from = defaultdict(list)
        for fact in self.static:
            if match(fact, 'calibration_target * *'):
                parts = get_parts(fact)
                self.calibration_target[parts[1]] = parts[2]
            elif match(fact, 'visible_from * *'):
                parts = get_parts(fact)
                self.visible_from[parts[1]].append(parts[2])

        # Cameras on rovers and their supported modes
        self.on_board = defaultdict(list)
        self.supports = defaultdict(list)
        for fact in self.static:
            if match(fact, 'on_board * *'):
                parts = get_parts(fact)
                self.on_board[parts[2]].append(parts[1])
            elif match(fact, 'supports * *'):
                parts = get_parts(fact)
                self.supports[parts[1]].append(parts[2])

    def __call__(self, node):
        state = node.state
        current_pos = {}
        stores = {}
        store_status = {}
        have_soil = defaultdict(set)
        have_rock = defaultdict(set)
        have_image = defaultdict(lambda: defaultdict(set))
        calibrated = defaultdict(set)

        for fact in state:
            parts = get_parts(fact)
            if match(fact, 'at * *'):
                current_pos[parts[1]] = parts[2]
            elif match(fact, 'store_of * *'):
                stores[parts[2]] = parts[1]
            elif match(fact, 'empty *'):
                store_status[parts[1]] = 'empty'
            elif match(fact, 'full *'):
                store_status[parts[1]] = 'full'
            elif match(fact, 'have_soil_analysis * *'):
                have_soil[parts[2]].add(parts[1])
            elif match(fact, 'have_rock_analysis * *'):
                have_rock[parts[2]].add(parts[1])
            elif match(fact, 'have_image * * *'):
                have_image[parts[2]][parts[3]].add(parts[1])
            elif match(fact, 'calibrated * *'):
                calibrated[parts[2]].add(parts[1])

        cost = 0

        # Helper to get minimal distance for a rover to any target waypoints
        def min_distance(rover, start, targets):
            if rover not in self.rover_distances or start not in self.rover_distances[rover]:
                return float('inf')
            distances = []
            for target in targets:
                if target in self.rover_distances[rover][start]:
                    distances.append(self.rover_distances[rover][start][target])
            return min(distances) if distances else float('inf')

        # Process soil data goals
        for goal in self.goals:
            if not match(goal, 'communicated_soil_data *'):
                continue
            wp = get_parts(goal)[1]
            if f'(communicated_soil_data {wp})' in state:
                continue
            if wp in have_soil:
                min_steps = float('inf')
                for rover in have_soil[wp]:
                    if rover not in current_pos:
                        continue
                    dist = min_distance(rover, current_pos[rover], self.communication_waypoints)
                    if dist != float('inf'):
                        min_steps = min(min_steps, dist + 1)
                if min_steps != float('inf'):
                    cost += min_steps
                else:
                    cost += 1000
            else:
                if f'(at_soil_sample {wp})' in state:
                    min_steps = float('inf')
                    for rover in self.rovers:
                        if f'(equipped_for_soil_analysis {rover})' not in self.static:
                            continue
                        if rover not in current_pos:
                            continue
                        store = stores.get(rover, None)
                        steps_drop = 0
                        if store and store_status.get(store, 'empty') == 'full':
                            steps_drop = 1
                        dist_to_wp = min_distance(rover, current_pos[rover], [wp])
                        if dist_to_wp == float('inf'):
                            continue
                        dist_to_comm = min_distance(rover, wp, self.communication_waypoints)
                        if dist_to_comm == float('inf'):
                            continue
                        total = steps_drop + dist_to_wp + 1 + dist_to_comm + 1
                        if total < min_steps:
                            min_steps = total
                    if min_steps != float('inf'):
                        cost += min_steps
                    else:
                        cost += 1000
                else:
                    min_steps = float('inf')
                    for rover in self.rovers:
                        if f'(equipped_for_soil_analysis {rover})' not in self.static:
                            continue
                        if rover not in current_pos:
                            continue
                        dist = min_distance(rover, current_pos[rover], self.communication_waypoints)
                        if dist != float('inf'):
                            min_steps = min(min_steps, dist + 1)
                    if min_steps != float('inf'):
                        cost += min_steps
                    else:
                        cost += 1000

        # Process rock data goals (similar to soil)
        for goal in self.goals:
            if not match(goal, 'communicated_rock_data *'):
                continue
            wp = get_parts(goal)[1]
            if f'(communicated_rock_data {wp})' in state:
                continue
            if wp in have_rock:
                min_steps = float('inf')
                for rover in have_rock[wp]:
                    if rover not in current_pos:
                        continue
                    dist = min_distance(rover, current_pos[rover], self.communication_waypoints)
                    if dist != float('inf'):
                        min_steps = min(min_steps, dist + 1)
                if min_steps != float('inf'):
                    cost += min_steps
                else:
                    cost += 1000
            else:
                if f'(at_rock_sample {wp})' in state:
                    min_steps = float('inf')
                    for rover in self.rovers:
                        if f'(equipped_for_rock_analysis {rover})' not in self.static:
                            continue
                        if rover not in current_pos:
                            continue
                        store = stores.get(rover, None)
                        steps_drop = 0
                        if store and store_status.get(store, 'empty') == 'full':
                            steps_drop = 1
                        dist_to_wp = min_distance(rover, current_pos[rover], [wp])
                        if dist_to_wp == float('inf'):
                            continue
                        dist_to_comm = min_distance(rover, wp, self.communication_waypoints)
                        if dist_to_comm == float('inf'):
                            continue
                        total = steps_drop + dist_to_wp + 1 + dist_to_comm + 1
                        if total < min_steps:
                            min_steps = total
                    if min_steps != float('inf'):
                        cost += min_steps
                    else:
                        cost += 1000
                else:
                    min_steps = float('inf')
                    for rover in self.rovers:
                        if f'(equipped_for_rock_analysis {rover})' not in self.static:
                            continue
                        if rover not in current_pos:
                            continue
                        dist = min_distance(rover, current_pos[rover], self.communication_waypoints)
                        if dist != float('inf'):
                            min_steps = min(min_steps, dist + 1)
                    if min_steps != float('inf'):
                        cost += min_steps
                    else:
                        cost += 1000

        # Process image data goals
        for goal in self.goals:
            if not match(goal, 'communicated_image_data * *'):
                continue
            obj, mode = get_parts(goal)[1], get_parts(goal)[2]
            if f'(communicated_image_data {obj} {mode})' in state:
                continue
            if obj in have_image and mode in have_image[obj]:
                min_steps = float('inf')
                for rover in have_image[obj][mode]:
                    if rover not in current_pos:
                        continue
                    dist = min_distance(rover, current_pos[rover], self.communication_waypoints)
                    if dist != float('inf'):
                        min_steps = min(min_steps, dist + 1)
                if min_steps != float('inf'):
                    cost += min_steps
                else:
                    cost += 1000
            else:
                min_steps = float('inf')
                for camera in self.supports:
                    if mode not in self.supports[camera]:
                        continue
                    if camera not in self.calibration_target:
                        continue
                    calibration_obj = self.calibration_target[camera]
                    cal_waypoints = self.visible_from.get(calibration_obj, [])
                    img_waypoints = self.visible_from.get(obj, [])
                    if not cal_waypoints or not img_waypoints:
                        continue
                    for rover in self.on_board:
                        if camera not in self.on_board[rover]:
                            continue
                        if f'(equipped_for_imaging {rover})' not in self.static:
                            continue
                        current_pos_rover = current_pos.get(rover, None)
                        if not current_pos_rover:
                            continue
                        # Calibration steps
                        if camera in calibrated[rover]:
                            cal_cost = 0
                            cal_pos = current_pos_rover
                        else:
                            cal_dist = min_distance(rover, current_pos_rover, cal_waypoints)
                            if cal_dist == float('inf'):
                                continue
                            cal_cost = cal_dist + 1
                            cal_pos = cal_waypoints[0] if cal_waypoints else None
                        # Imaging steps
                        img_dist = min_distance(rover, cal_pos, img_waypoints)
                        if img_dist == float('inf'):
                            continue
                        img_cost = img_dist + 1
                        # Communication steps
                        comm_dist = min_distance(rover, img_waypoints[0], self.communication_waypoints)
                        if comm_dist == float('inf'):
                            continue
                        comm_cost = comm_dist + 1
                        total = cal_cost + img_cost + comm_cost
                        if total < min_steps:
                            min_steps = total
                if min_steps != float('inf'):
                    cost += min_steps
                else:
                    cost += 1000

        return cost
