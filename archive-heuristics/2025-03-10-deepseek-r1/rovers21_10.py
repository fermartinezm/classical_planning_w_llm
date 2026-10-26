from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class rovers21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all communication goals (soil, rock, image data) by considering the minimal steps needed for each goal, including navigation, sampling, calibration, imaging, and communication.

    # Assumptions
    - Each goal is handled by the most efficient rover available.
    - Navigation steps are estimated using the shortest path based on can_traverse and visible predicates.
    - Rovers can drop samples if their store is full to collect new samples.
    - Cameras can be calibrated once per image capture.

    # Heuristic Initialization
    - Extracts static information including lander positions, rover equipment, camera details, and waypoint visibility.
    - Builds navigation graphs for each rover to compute shortest paths.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unachieved soil/rock data goal:
       a. Find rovers equipped to collect the sample.
       b. Estimate steps to navigate to the sample location, collect it, and communicate.
    2. For each unachieved image data goal:
       a. Find cameras supporting the required mode and their rovers.
       b. Estimate steps to calibrate (if needed), navigate to image location, capture, and communicate.
    3. Sum the minimal steps for all goals, assuming optimal rover assignment and path selection.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information from the task."""
        self.static = task.static
        self.goals = task.goals
        self.lander_locations = {}
        self.rover_stores = {}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.cameras = defaultdict(dict)
        self.visible_from = defaultdict(set)
        self.rover_graphs = defaultdict(lambda: defaultdict(list))
        self.waypoints = set()

        for fact in self.static:
            parts = self.get_parts(fact)
            if parts[0] == 'at_lander':
                self.lander_locations[parts[1]] = parts[2]
            elif parts[0] == 'store_of':
                self.rover_stores[parts[2]] = parts[1]
            elif parts[0] == 'equipped_for_soil_analysis':
                self.equipped_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_rock.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_imaging.add(parts[1])
            elif parts[0] == 'on_board':
                self.cameras[parts[1]]['on_board'] = parts[2]
            elif parts[0] == 'calibration_target':
                self.cameras[parts[1]]['calibration_target'] = parts[2]
            elif parts[0] == 'supports':
                if 'supports' not in self.cameras[parts[1]]:
                    self.cameras[parts[1]]['supports'] = set()
                self.cameras[parts[1]]['supports'].add(parts[2])
            elif parts[0] == 'visible_from':
                self.visible_from[parts[1]].add(parts[2])
            elif parts[0] == 'can_traverse':
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                if f'(visible {from_wp} {to_wp})' in self.static:
                    self.rover_graphs[rover][from_wp].append(to_wp)
            elif parts[0] == 'visible':
                self.waypoints.update(parts[1:3])

    def get_parts(self, fact):
        """Split a PDDL fact into its components."""
        return fact[1:-1].split()

    def shortest_path(self, adj, start, end):
        """Compute the shortest path between start and end using BFS."""
        if start == end:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == end:
                return dist
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        if self.goals <= state:
            return 0

        current_pos = {}
        store_status = defaultdict(bool)
        calibrated = defaultdict(bool)
        have_image = defaultdict(bool)

        for fact in state:
            parts = self.get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('rover'):
                current_pos[parts[1]] = parts[2]
            elif parts[0] == 'empty':
                store_status[parts[1]] = True
            elif parts[0] == 'calibrated':
                calibrated[(parts[1], parts[2])] = True
            elif parts[0] == 'have_image':
                have_image[(parts[1], parts[2], parts[3])] = True

        heuristic = 0

        # Soil data goals
        soil_goals = {p[2] for g in self.goals if (p := self.get_parts(g))[0] == 'communicated_soil_data'}
        for wp in soil_goals:
            if f'(communicated_soil_data {wp})' in state:
                continue
            min_cost = float('inf')
            for rover in self.equipped_soil:
                if rover not in current_pos or rover not in self.rover_stores:
                    continue
                store = self.rover_stores[rover]
                cost = 0 if store_status.get(store, False) else 1
                adj = self.rover_graphs[rover]
                start = current_pos[rover]
                steps = self.shortest_path(adj, start, wp)
                if steps == float('inf'):
                    continue
                cost += steps + 1
                lander_loc = next(iter(self.lander_locations.values()))
                possible_x = [x for x in self.waypoints if f'(visible {x} {lander_loc})' in self.static]
                min_comm = min((self.shortest_path(adj, wp, x) + 1 for x in possible_x), default=float('inf'))
                if min_comm == float('inf'):
                    continue
                cost += min_comm
                min_cost = min(min_cost, cost)
            if min_cost != float('inf'):
                heuristic += min_cost

        # Rock data goals
        rock_goals = {p[2] for g in self.goals if (p := self.get_parts(g))[0] == 'communicated_rock_data'}
        for wp in rock_goals:
            if f'(communicated_rock_data {wp})' in state:
                continue
            min_cost = float('inf')
            for rover in self.equipped_rock:
                if rover not in current_pos or rover not in self.rover_stores:
                    continue
                store = self.rover_stores[rover]
                cost = 0 if store_status.get(store, False) else 1
                adj = self.rover_graphs[rover]
                start = current_pos[rover]
                steps = self.shortest_path(adj, start, wp)
                if steps == float('inf'):
                    continue
                cost += steps + 1
                lander_loc = next(iter(self.lander_locations.values()))
                possible_x = [x for x in self.waypoints if f'(visible {x} {lander_loc})' in self.static]
                min_comm = min((self.shortest_path(adj, wp, x) + 1 for x in possible_x), default=float('inf'))
                if min_comm == float('inf'):
                    continue
                cost += min_comm
                min_cost = min(min_cost, cost)
            if min_cost != float('inf'):
                heuristic += min_cost

        # Image data goals
        image_goals = {(p[1], p[2]) for g in self.goals if (p := self.get_parts(g))[0] == 'communicated_image_data'}
        for (obj, mode) in image_goals:
            if f'(communicated_image_data {obj} {mode})' in state:
                continue
            min_cost = float('inf')
            for cam, cam_info in self.cameras.items():
                if mode not in cam_info.get('supports', set()):
                    continue
                rover = cam_info.get('on_board')
                if not rover or rover not in current_pos:
                    continue
                target = cam_info.get('calibration_target')
                if not target or target not in self.visible_from:
                    continue
                if have_image.get((rover, obj, mode), False):
                    adj = self.rover_graphs[rover]
                    start = current_pos[rover]
                    lander_loc = next(iter(self.lander_locations.values()))
                    possible_x = [x for x in self.waypoints if f'(visible {x} {lander_loc})' in self.static]
                    min_comm = min((self.shortest_path(adj, start, x) + 1 for x in possible_x), default=float('inf'))
                    if min_comm < min_cost:
                        min_cost = min_comm
                    continue
                cal_needed = not calibrated.get((cam, rover), False)
                possible_w1 = self.visible_from.get(target, set())
                cal_cost = 0
                if cal_needed:
                    adj = self.rover_graphs[rover]
                    start = current_pos[rover]
                    cal_steps = min((self.shortest_path(adj, start, w1) for w1 in possible_w1), default=float('inf'))
                    if cal_steps == float('inf'):
                        continue
                    cal_cost = cal_steps + 1
                    current_after_cal = min(possible_w1, key=lambda w: self.shortest_path(adj, start, w))
                else:
                    current_after_cal = current_pos[rover]
                possible_w2 = self.visible_from.get(obj, set())
                if not possible_w2:
                    continue
                adj = self.rover_graphs[rover]
                img_steps = min((self.shortest_path(adj, current_after_cal, w2) for w2 in possible_w2), default=float('inf'))
                if img_steps == float('inf'):
                    continue
                img_cost = img_steps + 1
                lander_loc = next(iter(self.lander_locations.values()))
                possible_x = [x for x in self.waypoints if f'(visible {x} {lander_loc})' in self.static]
                comm_steps = min((self.shortest_path(adj, min(possible_w2, key=lambda w: self.shortest_path(adj, current_after_cal, w)), x) for x in possible_x), default=float('inf'))
                if comm_steps == float('inf'):
                    continue
                comm_cost = comm_steps + 1
                total_cost = cal_cost + img_cost + comm_cost
                if total_cost < min_cost:
                    min_cost = total_cost
            if min_cost != float('inf'):
                heuristic += min_cost

        return heuristic
