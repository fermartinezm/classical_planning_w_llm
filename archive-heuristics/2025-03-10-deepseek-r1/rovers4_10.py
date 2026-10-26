from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class rovers4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goals by considering:
    - Navigation between waypoints for sample collection, imaging, and communication.
    - Soil and rock sample analysis and communication.
    - Image capture requiring camera calibration and subsequent communication.

    # Assumptions
    - Each rover can handle multiple tasks optimally.
    - Navigation distances are precomputed using BFS on the rover's traversable graph.
    - Cameras need calibration only once per image capture session.

    # Heuristic Initialization
    - Extracts static information: lander positions, rover traversal graphs, camera capabilities, and visibility.
    - Precomputes shortest path distances for each rover's navigation graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine unachieved goals (soil, rock, image data not communicated).
    2. For each goal, find the minimal actions required using the closest rover:
        a. **Soil/Rock Data**: Navigate to sample location, collect sample, navigate to communication point.
        b. **Image Data**: Calibrate camera (if needed), navigate to imaging location, capture image, communicate.
    3. Sum the minimal actions for all goals, assuming optimal parallel task handling by rovers.
    """

    def __init__(self, task):
        self.goals = task.goals
        static = task.static

        # Extract lander's waypoint
        self.lander_waypoint = None
        for fact in static:
            if fnmatch(fact, '(at_lander * *)'):
                parts = fact[1:-1].split()
                self.lander_waypoint = parts[2]

        # Extract visible waypoints for each target waypoint
        self.visible = defaultdict(set)
        for fact in static:
            if fnmatch(fact, '(visible * *)'):
                parts = fact[1:-1].split()
                x, y = parts[1], parts[2]
                self.visible[y].add(x)

        # Extract camera information
        self.calibration_targets = {}
        self.supports_modes = defaultdict(set)
        self.on_board_cameras = defaultdict(set)
        for fact in static:
            if fnmatch(fact, '(calibration_target * *)'):
                parts = fact[1:-1].split()
                self.calibration_targets[parts[1]] = parts[2]
            elif fnmatch(fact, '(supports * *)'):
                parts = fact[1:-1].split()
                self.supports_modes[parts[1]].add(parts[2])
            elif fnmatch(fact, '(on_board * *)'):
                parts = fact[1:-1].split()
                self.on_board_cameras[parts[2]].add(parts[1])

        # Extract rover equipment and stores
        self.equipped_for = defaultdict(set)
        self.store_of = {}
        for fact in static:
            if fnmatch(fact, '(equipped_for_*_analysis *)') or fnmatch(fact, '(equipped_for_imaging *)'):
                parts = fact[1:-1].split()
                capability = parts[0].split('_')[2]
                self.equipped_for[parts[1]].add(capability)
            elif fnmatch(fact, '(store_of * *)'):
                parts = fact[1:-1].split()
                self.store_of[parts[1]] = parts[2]

        # Build rover navigation graphs and precompute distances
        self.rover_distances = defaultdict(dict)
        for rover in set().union(*[self.on_board_cameras, self.equipped_for]):
            graph = defaultdict(list)
            for fact in static:
                if fnmatch(fact, '(can_traverse %s * *)' % rover):
                    parts = fact[1:-1].split()
                    from_wp, to_wp = parts[2], parts[3]
                    graph[from_wp].append(to_wp)
            distances = {}
            for start in graph:
                visited = {start: 0}
                queue = deque([(start, 0)])
                while queue:
                    current, dist = queue.popleft()
                    for neighbor in graph.get(current, []):
                        if neighbor not in visited:
                            visited[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
                distances[start] = visited
            self.rover_distances[rover] = distances

        # Extract visible_from for objectives
        self.visible_from = defaultdict(set)
        for fact in static:
            if fnmatch(fact, '(visible_from * *)'):
                parts = fact[1:-1].split()
                self.visible_from[parts[1]].add(parts[2])

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Extract current rover positions
        rover_pos = {}
        for fact in state:
            if fnmatch(fact, '(at * *)'):
                parts = fact[1:-1].split()
                rover_pos[parts[1]] = parts[2]

        # Extract available samples
        available_soil = set()
        available_rock = set()
        for fact in state:
            if fnmatch(fact, '(at_soil_sample *)'):
                available_soil.add(fact[1:-1].split()[1])
            elif fnmatch(fact, '(at_rock_sample *)'):
                available_rock.add(fact[1:-1].split()[1])

        # Extract stored analyses and images
        have_soil = defaultdict(set)
        have_rock = defaultdict(set)
        have_image = defaultdict(set)
        calibrated = defaultdict(set)
        store_status = defaultdict(str)
        for fact in state:
            if fnmatch(fact, '(have_soil_analysis * *)'):
                r, wp = fact[1:-1].split()[1], fact[1:-1].split()[2]
                have_soil[r].add(wp)
            elif fnmatch(fact, '(have_rock_analysis * *)'):
                r, wp = fact[1:-1].split()[1], fact[1:-1].split()[2]
                have_rock[r].add(wp)
            elif fnmatch(fact, '(have_image * * *)'):
                r, obj, mode = fact[1:-1].split()[1], fact[1:-1].split()[2], fact[1:-1].split()[3]
                have_image[r].add((obj, mode))
            elif fnmatch(fact, '(calibrated * *)'):
                cam, r = fact[1:-1].split()[1], fact[1:-1].split()[2]
                calibrated[r].add(cam)
            elif fnmatch(fact, '(empty *)'):
                store_status[fact[1:-1].split()[1]] = 'empty'
            elif fnmatch(fact, '(full *)'):
                store_status[fact[1:-1].split()[1]] = 'full'

        # Process each unachieved goal
        for goal in self.goals:
            if goal in state:
                continue
            parts = goal[1:-1].split()
            if parts[0] == 'communicated_soil_data':
                wp = parts[1]
                min_cost = float('inf')
                # Check if any rover has the analysis
                for rover in have_soil:
                    if wp in have_soil[rover]:
                        current_pos = rover_pos.get(rover, None)
                        if not current_pos:
                            continue
                        distances = self.rover_distances[rover].get(current_pos, {})
                        comm_points = self.visible.get(self.lander_waypoint, [])
                        min_dist = min([distances.get(x, float('inf')) for x in comm_points], default=float('inf'))
                        if min_dist != float('inf'):
                            min_cost = min(min_cost, min_dist + 1)
                if min_cost != float('inf'):
                    total_cost += min_cost
                else:
                    # Collect sample
                    if wp in available_soil:
                        for rover in self.equipped_for:
                            if 'soil' in self.equipped_for[rover]:
                                store = next((s for s, r in self.store_of.items() if r == rover and store_status.get(s) == 'empty'), None)
                                if store and rover in rover_pos:
                                    current_pos = rover_pos[rover]
                                    dist_to_wp = self.rover_distances[rover].get(current_pos, {}).get(wp, float('inf'))
                                    if dist_to_wp == float('inf'):
                                        continue
                                    comm_points = self.visible.get(self.lander_waypoint, [])
                                    dist_comm = min([self.rover_distances[rover].get(wp, {}).get(x, float('inf')) for x in comm_points], default=float('inf'))
                                    if dist_comm != float('inf'):
                                        total_cost += dist_to_wp + 1 + dist_comm + 1
                                        break
            elif parts[0] == 'communicated_rock_data':
                wp = parts[1]
                min_cost = float('inf')
                for rover in have_rock:
                    if wp in have_rock[rover]:
                        current_pos = rover_pos.get(rover, None)
                        if not current_pos:
                            continue
                        distances = self.rover_distances[rover].get(current_pos, {})
                        comm_points = self.visible.get(self.lander_waypoint, [])
                        min_dist = min([distances.get(x, float('inf')) for x in comm_points], default=float('inf'))
                        if min_dist != float('inf'):
                            min_cost = min(min_cost, min_dist + 1)
                if min_cost != float('inf'):
                    total_cost += min_cost
                else:
                    if wp in available_rock:
                        for rover in self.equipped_for:
                            if 'rock' in self.equipped_for[rover]:
                                store = next((s for s, r in self.store_of.items() if r == rover and store_status.get(s) == 'empty'), None)
                                if store and rover in rover_pos:
                                    current_pos = rover_pos[rover]
                                    dist_to_wp = self.rover_distances[rover].get(current_pos, {}).get(wp, float('inf'))
                                    if dist_to_wp == float('inf'):
                                        continue
                                    comm_points = self.visible.get(self.lander_waypoint, [])
                                    dist_comm = min([self.rover_distances[rover].get(wp, {}).get(x, float('inf')) for x in comm_points], default=float('inf'))
                                    if dist_comm != float('inf'):
                                        total_cost += dist_to_wp + 1 + dist_comm + 1
                                        break
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                min_cost = float('inf')
                # Check if any rover has the image
                for rover in have_image:
                    if (obj, mode) in have_image[rover]:
                        current_pos = rover_pos.get(rover, None)
                        if not current_pos:
                            continue
                        comm_points = self.visible.get(self.lander_waypoint, [])
                        distances = self.rover_distances[rover].get(current_pos, {})
                        min_dist = min([distances.get(x, float('inf')) for x in comm_points], default=float('inf'))
                        if min_dist != float('inf'):
                            min_cost = min(min_cost, min_dist + 1)
                if min_cost != float('inf'):
                    total_cost += min_cost
                else:
                    # Capture image
                    for rover in self.on_board_cameras:
                        cams = [cam for cam in self.on_board_cameras[rover] if mode in self.supports_modes[cam]]
                        for cam in cams:
                            cal_target = self.calibration_targets.get(cam, None)
                            if not cal_target:
                                continue
                            cal_waypoints = self.visible_from.get(cal_target, [])
                            if not cal_waypoints:
                                continue
                            # Calibration cost
                            if cam not in calibrated.get(rover, set()):
                                current_pos = rover_pos.get(rover, None)
                                if not current_pos:
                                    continue
                                cal_dist = min([self.rover_distances[rover].get(current_pos, {}).get(w, float('inf')) for w in cal_waypoints], default=float('inf'))
                                if cal_dist == float('inf'):
                                    continue
                                cal_cost = cal_dist + 1
                                new_pos = min(cal_waypoints, key=lambda w: self.rover_distances[rover].get(current_pos, {}).get(w, float('inf')))
                            else:
                                cal_cost = 0
                                new_pos = rover_pos.get(rover, None)
                            if not new_pos:
                                continue
                            # Imaging waypoints
                            img_waypoints = self.visible_from.get(obj, [])
                            img_dist = min([self.rover_distances[rover].get(new_pos, {}).get(w, float('inf')) for w in img_waypoints], default=float('inf'))
                            if img_dist == float('inf'):
                                continue
                            # Communication after imaging
                            img_wp = min(img_waypoints, key=lambda w: self.rover_distances[rover].get(new_pos, {}).get(w, float('inf')))
                            comm_dist = min([self.rover_distances[rover].get(img_wp, {}).get(x, float('inf')) for x in self.visible.get(self.lander_waypoint, [])], default=float('inf'))
                            if comm_dist == float('inf'):
                                continue
                            total_cost += cal_cost + img_dist + 1 + comm_dist + 1
                            break
        return total_cost
