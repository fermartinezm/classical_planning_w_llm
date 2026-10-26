from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    return fact[1:-1].split()


def match(fact, *patterns):
    parts = get_parts(fact)
    return len(parts) == len(patterns) and all(fnmatch(part, pattern) for part, pattern in zip(parts, patterns))


def bfs_shortest_paths(adjacency):
    shortest_paths = {}
    for start in adjacency:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        shortest_paths[start] = distances
    return shortest_paths


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    Estimates the number of actions required to achieve all goals by calculating minimal steps for each unachieved goal. 
    Considers optimal rover assignments for soil/rock sampling, image capturing, and communication tasks.

    # Assumptions
    - Landers' positions are static.
    - Can_traverse relationships are precomputed for shortest paths.
    - Rovers can be optimally assigned to different tasks (ignoring conflicts).
    - Camera calibration persists until used (optimistic assumption).

    # Heuristic Initialization
    - Extracts static information: lander positions, rover capabilities, camera specs, waypoint visibility.
    - Precomputes shortest paths between waypoints for each rover using BFS.

    # Step-By-Step Thinking
    1. For each unachieved soil/rock data goal:
        a. If sample collected: estimate communication steps.
        b. Else: estimate sampling + communication steps.
    2. For each unachieved image data goal:
        a. If image exists: estimate communication steps.
        b. Else: estimate calibration (if needed), imaging, and communication steps.
    3. Sum minimal costs across all goals assuming optimal rover assignments.
    """

    def __init__(self, task):
        self.goals = task.goals
        static = task.static

        # Extract landers
        self.landers = {}
        for fact in static:
            if match(fact, 'at_lander', '*', '*'):
                parts = get_parts(fact)
                self.landers[parts[1]] = parts[2]

        # Rover equipment
        self.rover_equipment = {}
        for fact in static:
            if match(fact, 'equipped_for_soil_analysis', '*'):
                rover = get_parts(fact)[1]
                self.rover_equipment.setdefault(rover, {}).update(soil=True)
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                rover = get_parts(fact)[1]
                self.rover_equipment.setdefault(rover, {}).update(rock=True)
            elif match(fact, 'equipped_for_imaging', '*'):
                rover = get_parts(fact)[1]
                self.rover_equipment.setdefault(rover, {}).update(imaging=True)

        # Rover stores
        self.rover_stores = {}
        for fact in static:
            if match(fact, 'store_of', '*', '*'):
                store, rover = get_parts(fact)[1], get_parts(fact)[2]
                self.rover_stores.setdefault(rover, []).append(store)

        # Cameras
        self.cameras = []
        cams = {}
        for fact in static:
            if match(fact, 'on_board', '*', '*'):
                cam, rover = get_parts(fact)[1], get_parts(fact)[2]
                cams[cam] = {'rover': rover, 'modes': [], 'calibration_target': None}
            elif match(fact, 'supports', '*', '*'):
                cam, mode = get_parts(fact)[1], get_parts(fact)[2]
                if cam in cams:
                    cams[cam]['modes'].append(mode)
            elif match(fact, 'calibration_target', '*', '*'):
                cam, target = get_parts(fact)[1], get_parts(fact)[2]
                if cam in cams:
                    cams[cam]['calibration_target'] = target
        self.cameras = list(cams.values())

        # Rover movement graphs
        self.rover_graphs = {}
        for fact in static:
            if match(fact, 'can_traverse', '*', '*', '*'):
                rover, from_wp, to_wp = get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]
                self.rover_graphs.setdefault(rover, {}).setdefault(from_wp, []).append(to_wp)

        # Precompute shortest paths
        self.rover_paths = {}
        for rover, graph in self.rover_graphs.items():
            self.rover_paths[rover] = bfs_shortest_paths(graph)

        # Visibility and visible_from
        self.visible = set()
        self.visible_from = {}
        for fact in static:
            if match(fact, 'visible', '*', '*'):
                from_wp, to_wp = get_parts(fact)[1], get_parts(fact)[2]
                self.visible.add((from_wp, to_wp))
            elif match(fact, 'visible_from', '*', '*'):
                obj, wp = get_parts(fact)[1], get_parts(fact)[2]
                self.visible_from.setdefault(obj, set()).add(wp)

    def __call__(self, node):
        state = node.state
        total_cost = 0

        def has(fact_pattern):
            return any(match(fact, *fact_pattern.split()) for fact in state)

        def rover_pos(rover):
            for fact in state:
                if match(fact, 'at', rover, '*'):
                    return get_parts(fact)[2]
            return None

        lander_wp = next(iter(self.landers.values())) if self.landers else None

        for goal in self.goals:
            if goal in state:
                continue
            goal_parts = get_parts(goal)

            # Soil data goal
            if goal_parts[0] == 'communicated_soil_data':
                wp = goal_parts[1]
                min_cost = float('inf')
                # Check existing samples
                for rover in self.rover_equipment:
                    if has(f'have_soil_analysis {rover} {wp}'):
                        pos = rover_pos(rover)
                        if pos and lander_wp:
                            visible = [to_wp for (f, t) in self.visible if f == lander_wp]
                            for x in visible:
                                steps = self.rover_paths.get(rover, {}).get(pos, {}).get(x, float('inf'))
                                min_cost = min(min_cost, steps + 1)
                # Check sampling needed
                if min_cost == float('inf') and has(f'at_soil_sample {wp}'):
                    for rover in self.rover_equipment:
                        if self.rover_equipment[rover].get('soil') and any(has(f'empty {s}') for s in self.rover_stores.get(rover, [])):
                            pos = rover_pos(rover)
                            if pos:
                                steps_to_wp = self.rover_paths.get(rover, {}).get(pos, {}).get(wp, float('inf'))
                                if steps_to_wp != float('inf') and lander_wp:
                                    visible = [to_wp for (f, t) in self.visible if f == lander_wp]
                                    for x in visible:
                                        steps_comm = self.rover_paths.get(rover, {}).get(wp, {}).get(x, float('inf'))
                                        total = steps_to_wp + 1 + steps_comm + 1
                                        min_cost = min(min_cost, total)
                if min_cost != float('inf'):
                    total_cost += min_cost

            # Rock data goal (similar to soil)
            elif goal_parts[0] == 'communicated_rock_data':
                wp = goal_parts[1]
                min_cost = float('inf')
                for rover in self.rover_equipment:
                    if has(f'have_rock_analysis {rover} {wp}'):
                        pos = rover_pos(rover)
                        if pos and lander_wp:
                            visible = [to_wp for (f, t) in self.visible if f == lander_wp]
                            for x in visible:
                                steps = self.rover_paths.get(rover, {}).get(pos, {}).get(x, float('inf'))
                                min_cost = min(min_cost, steps + 1)
                if min_cost == float('inf') and has(f'at_rock_sample {wp}'):
                    for rover in self.rover_equipment:
                        if self.rover_equipment[rover].get('rock') and any(has(f'empty {s}') for s in self.rover_stores.get(rover, [])):
                            pos = rover_pos(rover)
                            if pos:
                                steps_to_wp = self.rover_paths.get(rover, {}).get(pos, {}).get(wp, float('inf'))
                                if steps_to_wp != float('inf') and lander_wp:
                                    visible = [to_wp for (f, t) in self.visible if f == lander_wp]
                                    for x in visible:
                                        steps_comm = self.rover_paths.get(rover, {}).get(wp, {}).get(x, float('inf'))
                                        total = steps_to_wp + 1 + steps_comm + 1
                                        min_cost = min(min_cost, total)
                if min_cost != float('inf'):
                    total_cost += min_cost

            # Image data goal
            elif goal_parts[0] == 'communicated_image_data':
                obj, mode = goal_parts[1], goal_parts[2]
                min_cost = float('inf')
                # Check existing images
                for rover in self.rover_equipment:
                    if has(f'have_image {rover} {obj} {mode}'):
                        pos = rover_pos(rover)
                        if pos and lander_wp:
                            visible = [to_wp for (f, t) in self.visible if f == lander_wp]
                            for x in visible:
                                steps = self.rover_paths.get(rover, {}).get(pos, {}).get(x, float('inf'))
                                min_cost = min(min_cost, steps + 1)
                # Check image capture needed
                if min_cost == float('inf'):
                    for cam in self.cameras:
                        if mode not in cam['modes']:
                            continue
                        rover = cam['rover']
                        if not self.rover_equipment.get(rover, {}).get('imaging'):
                            continue
                        # Calibration check
                        calibrated = has(f'calibrated {cam["camera"]} {rover}')
                        cal_target = cam['calibration_target']
                        cal_steps = 0
                        if not calibrated and cal_target in self.visible_from:
                            cal_wps = self.visible_from[cal_target]
                            pos = rover_pos(rover)
                            if pos:
                                min_cal = min([self.rover_paths.get(rover, {}).get(pos, {}).get(wp, float('inf')) for wp in cal_wps], default=float('inf'))
                                if min_cal != float('inf'):
                                    cal_steps = min_cal + 1  # navigate + calibrate
                        # Imaging steps
                        if obj in self.visible_from:
                            img_wps = self.visible_from[obj]
                            pos_after_cal = next(iter(cal_wps), None) if not calibrated else rover_pos(rover)
                            if pos_after_cal:
                                min_img = min([self.rover_paths.get(rover, {}).get(pos_after_cal, {}).get(wp, float('inf')) for wp in img_wps], default=float('inf'))
                                if min_img != float('inf'):
                                    img_steps = min_img + 1  # navigate + image
                                    # Communication steps
                                    if lander_wp:
                                        visible = [to_wp for (f, t) in self.visible if f == lander_wp]
                                        min_comm = min([self.rover_paths.get(rover, {}).get(wp, {}).get(x, float('inf')) for wp in img_wps for x in visible], default=float('inf'))
                                        if min_comm != float('inf'):
                                            total = cal_steps + img_steps + min_comm + 1
                                            min_cost = min(min_cost, total)
                if min_cost != float('inf'):
                    total_cost += min_cost

        return total_cost if total_cost != 0 else 0
