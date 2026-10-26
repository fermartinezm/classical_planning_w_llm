from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

class rovers14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    Estimates the number of actions needed to achieve all communication goals by considering minimal steps for each goal, assuming optimal rover assignments and paths.

    # Assumptions
    - Rovers can handle multiple goals with summed steps.
    - Stores can be emptied when needed (additional drop actions if full).
    - Cameras can be recalibrated as needed.
    - Navigate actions use precomputed shortest paths between waypoints.

    # Heuristic Initialization
    - Extracts static info on rovers' equipment, stores, cameras, lander positions, and waypoint visibility.
    - Precomputes shortest paths between waypoints for each rover using BFS.

    # Step-By-Step Thinking
    1. For each unachieved goal:
        a. Soil/Rock Data: Find minimal steps for a rover to collect and communicate the sample.
        b. Image Data: Find steps to calibrate, capture, and communicate the image.
    2. Sum minimal steps for all goals, assuming parallel rover operations.
    """

    def __init__(self, task):
        self.static = task.static
        self.goals = task.goals

        self.rover_equipment = {}
        self.rover_store = {}
        self.camera_info = {}
        self.lander_positions = {}
        self.visible_from = {}
        self.calibration_targets = {}
        self.supports_modes = {}
        self.can_traverse = {}
        self.waypoints = set()

        for fact in self.static:
            parts = fact[1:-1].split()
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('rock')
            elif predicate == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('imaging')
            elif predicate == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_store[rover] = store
            elif predicate == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['rover'] = rover
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                self.supports_modes.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target':
                camera, objective = parts[1], parts[2]
                self.calibration_targets[camera] = objective
            elif predicate == 'at_lander':
                lander, wp = parts[1], parts[2]
                self.lander_positions[lander] = wp
            elif predicate == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.visible_from.setdefault(obj, []).append(wp)
            elif predicate == 'can_traverse':
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                self.can_traverse.setdefault(rover, {}).setdefault(from_wp, []).append(to_wp)
                self.waypoints.update([from_wp, to_wp])

        self.shortest_paths = {}
        for rover in self.can_traverse:
            self.shortest_paths[rover] = {}
            for wp in self.waypoints:
                self.shortest_paths[rover][wp] = {}
                queue = deque([(wp, 0)])
                visited = set()
                while queue:
                    current, dist = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    self.shortest_paths[rover][wp][current] = dist
                    for neighbor in self.can_traverse[rover].get(current, []):
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state
        total = 0
        current_positions = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                rover, wp = parts[1], parts[2]
                current_positions[rover] = wp

        for goal in self.goals:
            if goal in state:
                continue
            parts = goal[1:-1].split()
            if parts[0] == 'communicated_soil_data':
                wp = parts[1]
                min_cost = float('inf')
                for rover in self.rover_equipment.get('soil', []):
                    if 'soil' not in self.rover_equipment.get(rover, set()):
                        continue
                    store = self.rover_store.get(rover)
                    if not store:
                        continue
                    has_sample = f'(have_soil_analysis {rover} {wp})' in state
                    if has_sample:
                        lander_wp = next(iter(self.lander_positions.values()), None)
                        if not lander_wp:
                            continue
                        visible_wps = [fact.split()[1] for fact in self.static if fact.startswith(f'(visible ') and fact.endswith(f' {lander_wp})')]
                        if not visible_wps:
                            continue
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        min_dist = min((self.shortest_paths[rover][current_pos].get(w, float('inf')) for w in visible_wps), default=float('inf'))
                        if min_dist == float('inf'):
                            continue
                        cost = min_dist + 1
                        min_cost = min(min_cost, cost)
                    else:
                        store_full = f'(full {store})' in state
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        dist_to_wp = self.shortest_paths[rover][current_pos].get(wp, float('inf'))
                        if dist_to_wp == float('inf'):
                            continue
                        lander_wp = next(iter(self.lander_positions.values()), None)
                        visible_wps = [fact.split()[1] for fact in self.static if fact.startswith(f'(visible ') and fact.endswith(f' {lander_wp})')]
                        if not visible_wps:
                            continue
                        dist_from_wp = min((self.shortest_paths[rover][wp].get(w, float('inf')) for w in visible_wps), default=float('inf'))
                        if dist_from_wp == float('inf'):
                            continue
                        cost = dist_to_wp + 1 + dist_from_wp + 1
                        if store_full:
                            cost += 1
                        min_cost = min(min_cost, cost)
                if min_cost != float('inf'):
                    total += min_cost
            elif parts[0] == 'communicated_rock_data':
                wp = parts[1]
                min_cost = float('inf')
                for rover in self.rover_equipment.get('rock', []):
                    if 'rock' not in self.rover_equipment.get(rover, set()):
                        continue
                    store = self.rover_store.get(rover)
                    if not store:
                        continue
                    has_sample = f'(have_rock_analysis {rover} {wp})' in state
                    if has_sample:
                        lander_wp = next(iter(self.lander_positions.values()), None)
                        if not lander_wp:
                            continue
                        visible_wps = [fact.split()[1] for fact in self.static if fact.startswith(f'(visible ') and fact.endswith(f' {lander_wp})')]
                        if not visible_wps:
                            continue
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        min_dist = min((self.shortest_paths[rover][current_pos].get(w, float('inf')) for w in visible_wps), default=float('inf'))
                        if min_dist == float('inf'):
                            continue
                        cost = min_dist + 1
                        min_cost = min(min_cost, cost)
                    else:
                        store_full = f'(full {store})' in state
                        current_pos = current_positions.get(rover)
                        if not current_pos:
                            continue
                        dist_to_wp = self.shortest_paths[rover][current_pos].get(wp, float('inf'))
                        if dist_to_wp == float('inf'):
                            continue
                        lander_wp = next(iter(self.lander_positions.values()), None)
                        visible_wps = [fact.split()[1] for fact in self.static if fact.startswith(f'(visible ') and fact.endswith(f' {lander_wp})')]
                        if not visible_wps:
                            continue
                        dist_from_wp = min((self.shortest_paths[rover][wp].get(w, float('inf')) for w in visible_wps), default=float('inf'))
                        if dist_from_wp == float('inf'):
                            continue
                        cost = dist_to_wp + 1 + dist_from_wp + 1
                        if store_full:
                            cost += 1
                        min_cost = min(min_cost, cost)
                if min_cost != float('inf'):
                    total += min_cost
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                min_cost = float('inf')
                for camera in self.camera_info:
                    if mode not in self.supports_modes.get(camera, set()):
                        continue
                    rover = self.camera_info[camera].get('rover')
                    if not rover or 'imaging' not in self.rover_equipment.get(rover, set()):
                        continue
                    cal_obj = self.calibration_targets.get(camera)
                    if not cal_obj:
                        continue
                    cal_visible = self.visible_from.get(cal_obj, [])
                    if not cal_visible:
                        continue
                    current_pos = current_positions.get(rover)
                    if not current_pos:
                        continue
                    calibrated = f'(calibrated {camera} {rover})' in state
                    cal_steps = 0
                    if not calibrated:
                        cal_dist = min((self.shortest_paths[rover][current_pos].get(w, float('inf')) for w in cal_visible), default=float('inf'))
                        if cal_dist == float('inf'):
                            continue
                        cal_steps = cal_dist + 1
                        new_pos = cal_visible[0]
                    else:
                        new_pos = current_pos
                    obj_visible = self.visible_from.get(obj, [])
                    if not obj_visible:
                        continue
                    img_dist = min((self.shortest_paths[rover][new_pos].get(w, float('inf')) for w in obj_visible), default=float('inf'))
                    if img_dist == float('inf'):
                        continue
                    img_steps = img_dist + 1
                    lander_wp = next(iter(self.lander_positions.values()), None)
                    if not lander_wp:
                        continue
                    comm_visible = [fact.split()[1] for fact in self.static if fact.startswith(f'(visible ') and fact.endswith(f' {lander_wp})')]
                    if not comm_visible:
                        continue
                    comm_dist = min((self.shortest_paths[rover][w].get(x, float('inf')) for w in obj_visible for x in comm_visible), default=float('inf'))
                    if comm_dist == float('inf'):
                        continue
                    comm_steps = comm_dist + 1
                    total_steps = cal_steps + img_steps + comm_steps
                    min_cost = min(min_cost, total_steps)
                if min_cost != float('inf'):
                    total += min_cost
        return total
