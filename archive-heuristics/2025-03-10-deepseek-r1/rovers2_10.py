from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class rovers2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals by considering the necessary steps for soil/rock sampling, image capturing, and data communication. It precomputes navigation paths for each rover to efficiently estimate movement costs.

    # Assumptions
    - Each rover has a single store for samples.
    - The lander is stationary at a fixed waypoint.
    - Navigation paths are precomputed for each rover to estimate minimal movement steps.
    - Calibration, sampling, and communication actions are considered as single steps.

    # Heuristic Initialization
    - Extracts static information including rover capabilities, camera supports, calibration targets, and waypoint visibility.
    - Precomputes shortest navigation paths between all waypoints for each rover using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Goals**: Identify unachieved communication goals for soil, rock, and image data.
    2. **Soil/Rock Data Handling**:
        - If a rover already has the sample analysis, compute the cost to navigate to a visible waypoint and communicate.
        - If not, check if the sample is present and compute the cost for sampling, navigating, and communicating.
    3. **Image Data Handling**:
        - If a rover has the image, compute communication cost.
        - If not, compute costs for calibration (if needed), image capture, and communication.
    4. **Sum Costs**: Aggregate the minimal costs for all unachieved goals to form the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.lander_waypoint = None
        self.calibration_target = {}
        self.supports = defaultdict(set)
        self.on_board = defaultdict(list)
        self.visible_from = defaultdict(set)
        self.rover_graphs = defaultdict(dict)
        self.rover_distances = {}
        self.rover_store = {}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()

        # Extract lander's waypoint
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'at_lander':
                self.lander_waypoint = parts[2]
                break

        # Extract calibration_target
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'calibration_target':
                self.calibration_target[parts[1]] = parts[2]

        # Extract supports
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'supports':
                self.supports[parts[1]].add(parts[2])

        # Extract on_board
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'on_board':
                self.on_board[parts[2]].append(parts[1])

        # Extract visible_from
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'visible_from':
                self.visible_from[parts[1]].add(parts[2])

        # Extract can_traverse and build rover graphs
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'can_traverse':
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                if from_wp not in self.rover_graphs[rover]:
                    self.rover_graphs[rover][from_wp] = []
                self.rover_graphs[rover][from_wp].append(to_wp)

        # Precompute shortest paths for each rover
        for rover in self.rover_graphs:
            graph = self.rover_graphs[rover]
            distances = {}
            all_waypoints = set()
            for from_wp in graph:
                all_waypoints.add(from_wp)
                for to_wp in graph[from_wp]:
                    all_waypoints.add(to_wp)
            for wp in all_waypoints:
                visited = {wp: 0}
                queue = deque([wp])
                while queue:
                    current = queue.popleft()
                    current_dist = visited[current]
                    for neighbor in graph.get(current, []):
                        if neighbor not in visited or visited[neighbor] > current_dist + 1:
                            visited[neighbor] = current_dist + 1
                            queue.append(neighbor)
                for dest in visited:
                    distances[(wp, dest)] = visited[dest]
            self.rover_distances[rover] = distances

        # Extract rover store_of
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'store_of':
                self.rover_store[parts[2]] = parts[1]

        # Extract equipped_for predicates
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'equipped_for_soil_analysis':
                self.equipped_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_rock.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_imaging.add(parts[1])

    def __call__(self, node):
        state = node.state
        unachieved = self.goals - state
        if not unachieved:
            return 0

        total_cost = 0
        visible_from_lander = self.get_visible_from_lander()

        for goal in unachieved:
            parts = goal[1:-1].split()
            if parts[0] == 'communicated_soil_data':
                wp = parts[1]
                cost = self.handle_soil_goal(wp, state, visible_from_lander)
                total_cost += cost
            elif parts[0] == 'communicated_rock_data':
                wp = parts[1]
                cost = self.handle_rock_goal(wp, state, visible_from_lander)
                total_cost += cost
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                cost = self.handle_image_goal(obj, mode, state, visible_from_lander)
                total_cost += cost

        return total_cost

    def handle_soil_goal(self, wp, state, visible_from_lander):
        min_cost = float('inf')
        for rover in self.equipped_soil:
            if f'(have_soil_analysis {rover} {wp})' in state:
                current_pos = self.get_current_position(rover, state)
                if current_pos is None:
                    continue
                comm_cost = self.compute_comm_cost(rover, current_pos, visible_from_lander)
                if comm_cost is not None and comm_cost < min_cost:
                    min_cost = comm_cost
        if min_cost != float('inf'):
            return min_cost

        if f'(at_soil_sample {wp})' not in state:
            return 0

        for rover in self.equipped_soil:
            current_pos = self.get_current_position(rover, state)
            if current_pos is None:
                continue
            store = self.rover_store.get(rover)
            if not store:
                continue
            store_full = f'(full {store})' in state
            distance_to_wp = self.rover_distances.get(rover, {}).get((current_pos, wp), float('inf'))
            if distance_to_wp == float('inf'):
                continue
            cost = distance_to_wp
            if store_full:
                cost += 1
            cost += 1  # sample
            comm_cost = self.compute_comm_cost(rover, wp, visible_from_lander)
            if comm_cost is None:
                continue
            cost += comm_cost
            if cost < min_cost:
                min_cost = cost

        return min_cost if min_cost != float('inf') else 0

    def handle_rock_goal(self, wp, state, visible_from_lander):
        min_cost = float('inf')
        for rover in self.equipped_rock:
            if f'(have_rock_analysis {rover} {wp})' in state:
                current_pos = self.get_current_position(rover, state)
                if current_pos is None:
                    continue
                comm_cost = self.compute_comm_cost(rover, current_pos, visible_from_lander)
                if comm_cost is not None and comm_cost < min_cost:
                    min_cost = comm_cost
        if min_cost != float('inf'):
            return min_cost

        if f'(at_rock_sample {wp})' not in state:
            return 0

        for rover in self.equipped_rock:
            current_pos = self.get_current_position(rover, state)
            if current_pos is None:
                continue
            store = self.rover_store.get(rover)
            if not store:
                continue
            store_full = f'(full {store})' in state
            distance_to_wp = self.rover_distances.get(rover, {}).get((current_pos, wp), float('inf'))
            if distance_to_wp == float('inf'):
                continue
            cost = distance_to_wp
            if store_full:
                cost += 1
            cost += 1  # sample
            comm_cost = self.compute_comm_cost(rover, wp, visible_from_lander)
            if comm_cost is None:
                continue
            cost += comm_cost
            if cost < min_cost:
                min_cost = cost

        return min_cost if min_cost != float('inf') else 0

    def handle_image_goal(self, obj, mode, state, visible_from_lander):
        min_cost = float('inf')
        for rover in self.equipped_imaging:
            if f'(have_image {rover} {obj} {mode})' in state:
                current_pos = self.get_current_position(rover, state)
                if current_pos is None:
                    continue
                comm_cost = self.compute_comm_cost(rover, current_pos, visible_from_lander)
                if comm_cost is not None and comm_cost < min_cost:
                    min_cost = comm_cost
        if min_cost != float('inf'):
            return min_cost

        for rover in self.equipped_imaging:
            current_pos = self.get_current_position(rover, state)
            if current_pos is None:
                continue
            for camera in self.on_board.get(rover, []):
                if mode not in self.supports[camera]:
                    continue
                cal_obj = self.calibration_target.get(camera)
                if not cal_obj:
                    continue
                calibrated = f'(calibrated {camera} {rover})' in state
                cal_wps = self.visible_from.get(cal_obj, set())
                image_wps = self.visible_from.get(obj, set())
                if not cal_wps or not image_wps:
                    continue

                cal_cost = 0
                if not calibrated:
                    min_cal_dist = min((self.rover_distances[rover].get((current_pos, wp), float('inf')) for wp in cal_wps), default=float('inf'))
                    if min_cal_dist == float('inf'):
                        continue
                    cal_cost += min_cal_dist + 1
                    cal_current_pos = next(wp for wp in cal_wps if self.rover_distances[rover].get((current_pos, wp), float('inf')) == min_cal_dist)
                else:
                    cal_current_pos = current_pos
                    min_cal_dist = 0

                min_image_dist = min((self.rover_distances[rover].get((cal_current_pos, wp), float('inf')) for wp in image_wps), default=float('inf'))
                if min_image_dist == float('inf'):
                    continue
                image_cost = min_image_dist + 1
                image_current_pos = next(wp for wp in image_wps if self.rover_distances[rover].get((cal_current_pos, wp), float('inf')) == min_image_dist)

                comm_cost = self.compute_comm_cost(rover, image_current_pos, visible_from_lander)
                if comm_cost is None:
                    continue

                total_cost = cal_cost + image_cost + comm_cost
                if total_cost < min_cost:
                    min_cost = total_cost

        return min_cost if min_cost != float('inf') else 0

    def get_current_position(self, rover, state):
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] == rover:
                return parts[2]
        return None

    def compute_comm_cost(self, rover, current_pos, visible_from_lander):
        if not visible_from_lander:
            return None
        min_dist = min((self.rover_distances.get(rover, {}).get((current_pos, wp), float('inf')) for wp in visible_from_lander), default=float('inf'))
        if min_dist == float('inf'):
            return None
        return min_dist + 1

    def get_visible_from_lander(self):
        visible = set()
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'visible' and parts[1] == self.lander_waypoint:
                visible.add(parts[2])
        return visible
