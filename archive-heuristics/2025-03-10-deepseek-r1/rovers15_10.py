from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern_parts))

class rovers15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the number of actions required to achieve all communication goals by summing the minimal steps needed for each unachieved goal. It considers soil/rock sampling, image capturing, calibration, and communication actions, taking into account rover movement distances based on precomputed shortest paths.

    Assumptions:
    - Each goal is handled by the most efficient rover and path available.
    - Rovers can navigate using precomputed shortest paths between waypoints.
    - Static facts (like can_traverse, visible, calibration targets) are known and preprocessed.
    - The lander is at a single waypoint (general case handled if multiple).

    Heuristic Initialization:
    - Precompute shortest paths for each rover's navigation graph.
    - Extract static information: lander positions, calibration targets, visible waypoints, and rover capabilities.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each uncommunicated soil/rock data goal:
        a. If a rover has the sample, compute steps to communicate from nearest visible waypoint.
        b. Else, compute steps to sample and communicate, including store management.
    2. For each uncommunicated image data goal:
        a. If a rover has the image, compute steps to communicate.
        b. Else, compute steps to calibrate camera, capture image, and communicate.
    3. Sum all minimal steps, assuming optimal rover assignment and parallel goal achievement.
    """

    def __init__(self, task):
        self.task = task
        self.static = task.static
        self.goals = task.goals

        # Extract lander waypoints
        self.lander_waypoints = {}
        for fact in self.static:
            if match(fact, 'at_lander * *'):
                parts = get_parts(fact)
                self.lander_waypoints[parts[1]] = parts[2]

        # Extract calibration targets
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, 'calibration_target * *'):
                parts = get_parts(fact)
                self.calibration_targets[parts[1]] = parts[2]

        # Extract visible_from relationships
        self.visible_from = defaultdict(list)
        for fact in self.static:
            if match(fact, 'visible_from * *'):
                parts = get_parts(fact)
                self.visible_from[parts[1]].append(parts[2])

        # Build rover navigation graphs and precompute shortest paths
        self.rover_graphs = defaultdict(lambda: defaultdict(list))
        visible_pairs = set()
        for fact in self.static:
            if match(fact, 'visible * *'):
                parts = get_parts(fact)
                visible_pairs.add((parts[1], parts[2]))
            elif match(fact, 'can_traverse * * *'):
                parts = get_parts(fact)
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                if (from_wp, to_wp) in visible_pairs:
                    self.rover_graphs[rover][from_wp].append(to_wp)

        # Precompute shortest paths for each rover
        self.rover_shortest = defaultdict(dict)
        for rover in self.rover_graphs:
            graph = self.rover_graphs[rover]
            nodes = set(graph.keys())
            for neighbors in graph.values():
                nodes.update(neighbors)
            self.rover_shortest[rover] = {}
            for start in nodes:
                distances = {start: 0}
                queue = deque([start])
                while queue:
                    current = queue.popleft()
                    for neighbor in graph.get(current, []):
                        if neighbor not in distances:
                            distances[neighbor] = distances[current] + 1
                            queue.append(neighbor)
                self.rover_shortest[rover][start] = distances

        # Store rover equipment and stores from static
        self.equipped = defaultdict(lambda: defaultdict(bool))
        self.store_map = {}
        for fact in self.static:
            if match(fact, 'equipped_for_*_analysis *'):
                parts = get_parts(fact)
                self.equipped[parts[0].split('_')[2]][parts[1]] = True
            elif match(fact, 'store_of * *'):
                parts = get_parts(fact)
                self.store_map[parts[2]] = parts[1]

    def __call__(self, node):
        state = node.state
        h_val = 0

        # Helper functions
        def has_fact(pattern):
            return any(match(fact, pattern) for fact in state)

        def get_rover_position(rover):
            for fact in state:
                if match(fact, f'at {rover} *'):
                    return get_parts(fact)[2]
            return None

        # Process goals
        uncommunicated_soil = set()
        uncommunicated_rock = set()
        uncommunicated_images = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data' and goal not in state:
                uncommunicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data' and goal not in state:
                uncommunicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data' and goal not in state:
                uncommunicated_images.add((parts[1], parts[2]))

        lander_wp = next(iter(self.lander_waypoints.values()))

        # Helper to find communication steps
        def get_comm_steps(rover, curr_pos):
            if rover not in self.rover_shortest or curr_pos not in self.rover_shortest[rover]:
                return float('inf')
            distances = self.rover_shortest[rover][curr_pos]
            comm_points = [wp_from for (wp_from, wp_to) in self.rover_graphs['visible'] if wp_to == lander_wp]
            return min([distances.get(wp, float('inf')) for wp in comm_points], default=float('inf')) + 1

        # Process soil data goals
        for wp in uncommunicated_soil:
            min_cost = float('inf')
            # Check existing samples
            for fact in state:
                if match(fact, f'have_soil_analysis * {wp}'):
                    rover = get_parts(fact)[1]
                    curr_pos = get_rover_position(rover)
                    if curr_pos:
                        steps = get_comm_steps(rover, curr_pos)
                        min_cost = min(min_cost, steps)
            # Check sampling needed
            if f'(at_soil_sample {wp})' in state:
                for rover in self.store_map:
                    if not self.equipped['soil'][rover]:
                        continue
                    store = self.store_map[rover]
                    if f'(full {store})' in state:
                        drop_cost = 1
                    else:
                        drop_cost = 0
                    curr_pos = get_rover_position(rover)
                    if not curr_pos:
                        continue
                    # Move to wp
                    dist_to_wp = self.rover_shortest[rover][curr_pos].get(wp, float('inf'))
                    # Move to comm point
                    dist_from_wp = get_comm_steps(rover, wp) - 1  # subtract communicate action
                    total = drop_cost + dist_to_wp + 1 + dist_from_wp
                    min_cost = min(min_cost, total)
            h_val += min_cost if min_cost != float('inf') else 1000

        # Process rock data goals (similar to soil)
        for wp in uncommunicated_rock:
            min_cost = float('inf')
            for fact in state:
                if match(fact, f'have_rock_analysis * {wp}'):
                    rover = get_parts(fact)[1]
                    curr_pos = get_rover_position(rover)
                    if curr_pos:
                        steps = get_comm_steps(rover, curr_pos)
                        min_cost = min(min_cost, steps)
            if f'(at_rock_sample {wp})' in state:
                for rover in self.store_map:
                    if not self.equipped['rock'][rover]:
                        continue
                    store = self.store_map[rover]
                    if f'(full {store})' in state:
                        drop_cost = 1
                    else:
                        drop_cost = 0
                    curr_pos = get_rover_position(rover)
                    if not curr_pos:
                        continue
                    dist_to_wp = self.rover_shortest[rover][curr_pos].get(wp, float('inf'))
                    dist_from_wp = get_comm_steps(rover, wp) - 1
                    total = drop_cost + dist_to_wp + 1 + dist_from_wp
                    min_cost = min(min_cost, total)
            h_val += min_cost if min_cost != float('inf') else 1000

        # Process image goals
        for (obj, mode) in uncommunicated_images:
            min_cost = float('inf')
            # Check existing images
            for fact in state:
                if match(fact, f'have_image * {obj} {mode}'):
                    rover = get_parts(fact)[1]
                    curr_pos = get_rover_position(rover)
                    if curr_pos:
                        steps = get_comm_steps(rover, curr_pos)
                        min_cost = min(min_cost, steps)
            # Check image capture needed
            for rover in self.rover_shortest:
                if not any(match(fact, f'equipped_for_imaging {rover}') for fact in self.static):
                    continue
                # Find supported cameras
                for cam in self.calibration_targets:
                    if not any(match(fact, f'supports {cam} {mode}') for fact in self.static):
                        continue
                    # Calibration needed?
                    calibrated = f'(calibrated {cam} {rover})' in state
                    target_obj = self.calibration_targets[cam]
                    calib_wps = self.visible_from.get(target_obj, [])
                    curr_pos = get_rover_position(rover)
                    if not curr_pos or not calib_wps:
                        continue
                    # Calibration steps
                    calib_dist = min([self.rover_shortest[rover][curr_pos].get(wp, float('inf')) for wp in calib_wps], default=float('inf'))
                    if calib_dist == float('inf') and not calibrated:
                        continue
                    calib_cost = 0 if calibrated else calib_dist + 1
                    new_pos = calib_wps[0] if not calibrated else curr_pos  # Simplified
                    # Capture image
                    img_wps = self.visible_from.get(obj, [])
                    img_dist = min([self.rover_shortest[rover][new_pos].get(wp, float('inf')) for wp in img_wps], default=float('inf'))
                    if img_dist == float('inf'):
                        continue
                    # Communicate
                    comm_dist = get_comm_steps(rover, img_wps[0]) if img_wps else float('inf')
                    total = calib_cost + img_dist + 1 + comm_dist
                    min_cost = min(min_cost, total)
            h_val += min_cost if min_cost != float('inf') else 1000

        return h_val
