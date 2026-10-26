from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern_parts))

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the number of actions required to achieve all communication goals by considering the minimal steps needed for each unmet goal. It accounts for navigation, sampling, calibration, imaging, and communication actions.

    Assumptions:
    - Each rover can handle multiple goals, but the heuristic considers the most efficient assignment.
    - Navigation costs are precomputed using the shortest path between waypoints for each rover.
    - Static facts (like lander positions and calibration targets) are invariant and preprocessed.

    Heuristic Initialization:
    - Extracts static information such as lander positions, calibration targets, rover equipment, and waypoint visibility.
    - Precomputes shortest navigation paths for each rover using BFS on allowed traversals.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify unmet goals (soil, rock, image data not communicated).
    2. For each soil/rock goal:
        a. If sampled, compute navigation to a visible lander waypoint.
        b. If not sampled, compute navigation to sample location, then to a lander-visible waypoint.
    3. For each image goal:
        a. If image exists, compute navigation to communicate.
        b. If not, compute calibration (if needed), imaging, and communication steps.
    4. Sum minimal actions for all unmet goals, considering optimal rover assignments.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract lander positions
        self.lander_positions = {}
        for fact in self.static:
            if match(fact, 'at_lander * *'):
                parts = get_parts(fact)
                self.lander_positions[parts[1]] = parts[2]

        # Extract calibration targets and supported modes
        self.calibration_targets = {}
        self.supports_modes = {}
        for fact in self.static:
            if match(fact, 'calibration_target * *'):
                parts = get_parts(fact)
                self.calibration_targets[parts[1]] = parts[2]
            elif match(fact, 'supports * *'):
                parts = get_parts(fact)
                cam, mode = parts[1], parts[2]
                if cam not in self.supports_modes:
                    self.supports_modes[cam] = set()
                self.supports_modes[cam].add(mode)

        # Rover equipment and stores
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.stores = {}
        for fact in self.static:
            if match(fact, 'equipped_for_soil_analysis *'):
                self.equipped_soil.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_rock_analysis *'):
                self.equipped_rock.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_imaging *'):
                self.equipped_imaging.add(get_parts(fact)[1])
            elif match(fact, 'store_of * *'):
                parts = get_parts(fact)
                self.stores[parts[2]] = parts[1]

        # Visible_from: {objective: set(waypoints)}
        self.visible_from = {}
        for fact in self.static:
            if match(fact, 'visible_from * *'):
                parts = get_parts(fact)
                obj, wp = parts[1], parts[2]
                if obj not in self.visible_from:
                    self.visible_from[obj] = set()
                self.visible_from[obj].add(wp)

        # Precompute navigation distances for each rover
        self.distances = {}
        waypoints = set()
        visible = set()
        can_traverse = {}

        # Collect visible and can_traverse facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'visible' and len(parts) == 3:
                visible.add((parts[1], parts[2]))
            elif parts[0] == 'can_traverse' and len(parts) == 4:
                rover, y, z = parts[1], parts[2], parts[3]
                if rover not in can_traverse:
                    can_traverse[rover] = {}
                if y not in can_traverse[rover]:
                    can_traverse[rover][y] = set()
                can_traverse[rover][y].add(z)
                waypoints.update([y, z])

        # Build navigation graph for each rover and compute shortest paths
        for rover in can_traverse:
            graph = {}
            for y in can_traverse[rover]:
                for z in can_traverse[rover][y]:
                    if (y, z) in visible:
                        if y not in graph:
                            graph[y] = []
                        graph[y].append(z)
            self.distances[rover] = {}
            for start in waypoints:
                self.distances[rover][start] = {}
                visited = {start: 0}
                queue = deque([(start, 0)])
                while queue:
                    current, steps = queue.popleft()
                    for neighbor in graph.get(current, []):
                        if neighbor not in visited:
                            visited[neighbor] = steps + 1
                            queue.append((neighbor, steps + 1))
                for wp in waypoints:
                    self.distances[rover][start][wp] = visited.get(wp, float('inf'))

    def __call__(self, node):
        state = node.state
        unmet_goals = [g for g in self.goals if g not in state]
        total_cost = 0

        for goal in unmet_goals:
            g_parts = get_parts(goal)
            if g_parts[0] == 'communicated_soil_data':
                wp = g_parts[1]
                cost = self._soil_cost(wp, state)
            elif g_parts[0] == 'communicated_rock_data':
                wp = g_parts[1]
                cost = self._rock_cost(wp, state)
            elif g_parts[0] == 'communicated_image_data':
                obj, mode = g_parts[1], g_parts[2]
                cost = self._image_cost(obj, mode, state)
            else:
                cost = 0
            total_cost += cost if cost != float('inf') else 0

        return total_cost

    def _soil_cost(self, wp, state):
        min_cost = float('inf')
        lander = next(iter(self.lander_positions.keys()))
        lander_wp = self.lander_positions[lander]

        # Check if any rover already has the sample
        for r in self.equipped_soil:
            if f'(have_soil_analysis {r} {wp})' in state:
                rover_pos = self._current_rover_pos(r, state)
                if not rover_pos:
                    continue
                comm_cost = self._comm_cost(r, rover_pos, lander_wp)
                if comm_cost < min_cost:
                    min_cost = comm_cost

        # If not, check if sample is available and can be collected
        if f'(at_soil_sample {wp})' in state:
            for r in self.equipped_soil:
                rover_pos = self._current_rover_pos(r, state)
                if not rover_pos:
                    continue
                store = self.stores.get(r)
                drop_cost = 1 if store and f'(full {store})' in state else 0
                nav_to_sample = self.distances[r][rover_pos].get(wp, float('inf'))
                if nav_to_sample == float('inf'):
                    continue
                sample_cost = drop_cost + nav_to_sample + 1  # drop (if needed) + navigate + sample
                comm_cost = self._comm_cost(r, wp, lander_wp)
                total = sample_cost + comm_cost
                if total < min_cost:
                    min_cost = total

        return min_cost if min_cost != float('inf') else 0

    def _rock_cost(self, wp, state):
        min_cost = float('inf')
        lander = next(iter(self.lander_positions.keys()))
        lander_wp = self.lander_positions[lander]

        # Check if any rover already has the sample
        for r in self.equipped_rock:
            if f'(have_rock_analysis {r} {wp})' in state:
                rover_pos = self._current_rover_pos(r, state)
                if not rover_pos:
                    continue
                comm_cost = self._comm_cost(r, rover_pos, lander_wp)
                if comm_cost < min_cost:
                    min_cost = comm_cost

        # If not, check if sample is available and can be collected
        if f'(at_rock_sample {wp})' in state:
            for r in self.equipped_rock:
                rover_pos = self._current_rover_pos(r, state)
                if not rover_pos:
                    continue
                store = self.stores.get(r)
                drop_cost = 1 if store and f'(full {store})' in state else 0
                nav_to_sample = self.distances[r][rover_pos].get(wp, float('inf'))
                if nav_to_sample == float('inf'):
                    continue
                sample_cost = drop_cost + nav_to_sample + 1  # drop + navigate + sample
                comm_cost = self._comm_cost(r, wp, lander_wp)
                total = sample_cost + comm_cost
                if total < min_cost:
                    min_cost = total

        return min_cost if min_cost != float('inf') else 0

    def _image_cost(self, obj, mode, state):
        min_cost = float('inf')
        lander = next(iter(self.lander_positions.keys()))
        lander_wp = self.lander_positions[lander]

        # Check if any rover has the image
        for r in self.equipped_imaging:
            if f'(have_image {r} {obj} {mode})' in state:
                rover_pos = self._current_rover_pos(r, state)
                if not rover_pos:
                    continue
                comm_cost = self._comm_cost(r, rover_pos, lander_wp)
                if comm_cost < min_cost:
                    min_cost = comm_cost

        # If not, find a rover and camera to take the image
        for r in self.equipped_imaging:
            rover_pos = self._current_rover_pos(r, state)
            if not rover_pos:
                continue
            # Check available cameras on rover
            for cam in [c for c in self.calibration_targets if f'(on_board {c} {r})' in self.static]:
                if mode not in self.supports_modes.get(cam, []):
                    continue
                # Calibration required
                target = self.calibration_targets.get(cam)
                if not target:
                    continue
                calibrate_cost = 0
                if f'(calibrated {cam} {r})' not in state:
                    # Need to calibrate
                    cal_wps = self.visible_from.get(target, [])
                    if not cal_wps:
                        continue
                    min_cal_dist = min([self.distances[r][rover_pos].get(w, float('inf')) for w in cal_wps)
                    if min_cal_dist == float('inf'):
                        continue
                    calibrate_cost = min_cal_dist + 1  # navigate + calibrate
                    cal_pos = min(cal_wps, key=lambda w: self.distances[r][rover_pos].get(w, float('inf')))
                else:
                    cal_pos = rover_pos
                # Take image at visible_from obj
                img_wps = self.visible_from.get(obj, [])
                if not img_wps:
                    continue
                min_img_dist = min([self.distances[r][cal_pos].get(w, float('inf')) for w in img_wps)
                if min_img_dist == float('inf'):
                    continue
                take_image_cost = min_img_dist + 1  # navigate + take_image
                # Communicate
                comm_cost = self._comm_cost(r, img_wps[0], lander_wp)
                total = calibrate_cost + take_image_cost + comm_cost
                if total < min_cost:
                    min_cost = total

        return min_cost if min_cost != float('inf') else 0

    def _current_rover_pos(self, rover, state):
        for fact in state:
            if match(fact, f'at {rover} *'):
                return get_parts(fact)[2]
        return None

    def _comm_cost(self, rover, start_wp, lander_wp):
        # Find waypoints visible to lander's location
        visible_x = [x for x, y in [get_parts(f) for f in self.static if match(f, 'visible * *')] if y == lander_wp]
        if not visible_x:
            return float('inf')
        min_dist = min([self.distances[rover][start_wp].get(x, float('inf')) for x in visible_x])
        return min_dist + 1 if min_dist != float('inf') else float('inf')
