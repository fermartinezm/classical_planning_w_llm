from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rovers1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required for rovers to achieve all goals, including navigating to collect samples, calibrating cameras, taking images, and communicating data.

    # Assumptions
    - Rovers can traverse between connected waypoints using the shortest path.
    - Cameras need calibration only once before taking an image.
    - Each sample collection requires an empty store, which is emptied after communication.

    # Heuristic Initialization
    - Extract static information such as lander location, rover capabilities, camera details, and waypoint connectivity.
    - Precompute graphs for each rover's navigable waypoints.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify unmet goals (uncommunicated data).
    2. For each unmet goal:
        a. Soil/Rock Data: Check if collected; if not, estimate navigation to sample, then to lander-visible waypoint.
        b. Image Data: Check if captured; if not, calibrate camera (if needed), navigate to objective's waypoint, take image, then communicate.
    3. Use BFS to find minimal navigation steps between waypoints for each rover.
    4. Sum the minimal actions for all unmet goals.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.lander_location = None
        self.calibration_target = {}
        self.visible_from = defaultdict(list)
        self.supports = defaultdict(list)
        self.on_board = defaultdict(list)
        self.equipped_for_imaging = set()
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.store_of = {}
        self.rover_graphs = defaultdict(lambda: defaultdict(list))
        self.lander_visible_waypoints = []
        self.rovers = set()

        # Extract static information
        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, 'at_lander', '*', '*'):
                self.lander_location = parts[2]
            elif match(fact, 'calibration_target', '*', '*'):
                self.calibration_target[parts[1]] = parts[2]
            elif match(fact, 'visible_from', '*', '*'):
                self.visible_from[parts[1]].append(parts[2])
            elif match(fact, 'supports', '*', '*'):
                self.supports[parts[1]].append(parts[2])
            elif match(fact, 'on_board', '*', '*'):
                self.on_board[parts[2]].append(parts[1])
            elif match(fact, 'equipped_for_imaging', '*'):
                self.equipped_for_imaging.add(parts[1])
                self.rovers.add(parts[1])
            elif match(fact, 'equipped_for_soil_analysis', '*'):
                self.equipped_for_soil.add(parts[1])
                self.rovers.add(parts[1])
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                self.equipped_for_rock.add(parts[1])
                self.rovers.add(parts[1])
            elif match(fact, 'store_of', '*', '*'):
                self.store_of[parts[1]] = parts[2]
            elif match(fact, 'can_traverse', '*', '*', '*'):
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                self.rover_graphs[rover][from_wp].append(to_wp)
            elif match(fact, 'visible', '*', self.lander_location):
                self.lander_visible_waypoints.append(parts[1])

    def minimal_navigate_steps(self, rover, start, end):
        if start == end:
            return 0
        graph = self.rover_graphs.get(rover, {})
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, steps = queue.popleft()
            if current == end:
                return steps
            if current in visited:
                continue
            visited.add(current)
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, steps + 1))
        return float('inf')

    def __call__(self, node):
        state = node.state
        unmet_goals = [g for g in self.goals if g not in state]
        if not unmet_goals:
            return 0

        current_rover_pos = {}
        calibrated = set()
        have_soil = defaultdict(set)
        have_rock = defaultdict(set)
        have_image = defaultdict(set)
        stores = defaultdict(lambda: 'empty')

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.rovers:
                current_rover_pos[parts[1]] = parts[2]
            elif parts[0] == 'calibrated':
                calibrated.add((parts[1], parts[2]))
            elif parts[0] == 'have_soil_analysis':
                have_soil[parts[1]].add(parts[2])
            elif parts[0] == 'have_rock_analysis':
                have_rock[parts[1]].add(parts[2])
            elif parts[0] == 'have_image':
                have_image[parts[1]].add((parts[2], parts[3]))
            elif parts[0] == 'empty':
                stores[self.store_of[parts[1]]] = 'empty'
            elif parts[0] == 'full':
                stores[self.store_of[parts[1]]] = 'full'

        total_cost = 0

        for goal in unmet_goals:
            g_parts = get_parts(goal)
            if g_parts[0] == 'communicated_soil_data':
                wp = g_parts[1]
                candidates = [r for r in self.rovers if wp in have_soil[r]]
                if candidates:
                    min_cost = min(
                        [self.minimal_navigate_steps(r, current_rover_pos[r], lv) + 1
                         for r in candidates
                         for lv in self.lander_visible_waypoints], default=float('inf'))
                    total_cost += min_cost if min_cost != float('inf') else 100
                else:
                    valid_rovers = [r for r in self.equipped_for_soil if stores[r] == 'empty']
                    min_cost = float('inf')
                    for r in valid_rovers:
                        if r not in current_rover_pos:
                            continue
                        to_sample = self.minimal_navigate_steps(r, current_rover_pos[r], wp)
                        if to_sample == float('inf'):
                            continue
                        to_lander = min([self.minimal_navigate_steps(r, wp, lv) for lv in self.lander_visible_waypoints], default=float('inf'))
                        if to_lander == float('inf'):
                            continue
                        cost = to_sample + 1 + to_lander + 1
                        min_cost = min(min_cost, cost)
                    total_cost += min_cost if min_cost != float('inf') else 100
            elif g_parts[0] == 'communicated_rock_data':
                wp = g_parts[1]
                candidates = [r for r in self.rovers if wp in have_rock[r]]
                if candidates:
                    min_cost = min(
                        [self.minimal_navigate_steps(r, current_rover_pos[r], lv) + 1
                         for r in candidates
                         for lv in self.lander_visible_waypoints], default=float('inf'))
                    total_cost += min_cost if min_cost != float('inf') else 100
                else:
                    valid_rovers = [r for r in self.equipped_for_rock if stores[r] == 'empty']
                    min_cost = float('inf')
                    for r in valid_rovers:
                        if r not in current_rover_pos:
                            continue
                        to_sample = self.minimal_navigate_steps(r, current_rover_pos[r], wp)
                        if to_sample == float('inf'):
                            continue
                        to_lander = min([self.minimal_navigate_steps(r, wp, lv) for lv in self.lander_visible_waypoints], default=float('inf'))
                        if to_lander == float('inf'):
                            continue
                        cost = to_sample + 1 + to_lander + 1
                        min_cost = min(min_cost, cost)
                    total_cost += min_cost if min_cost != float('inf') else 100
            elif g_parts[0] == 'communicated_image_data':
                obj, mode = g_parts[1], g_parts[2]
                candidates = [r for r in self.rovers if (obj, mode) in have_image[r]]
                if candidates:
                    min_cost = min(
                        [self.minimal_navigate_steps(r, current_rover_pos[r], lv) + 1
                         for r in candidates
                         for lv in self.lander_visible_waypoints], default=float('inf'))
                    total_cost += min_cost if min_cost != float('inf') else 100
                else:
                    min_cost = float('inf')
                    for r in self.equipped_for_imaging:
                        for cam in self.on_board[r]:
                            if mode not in self.supports[cam]:
                                continue
                            cal_obj = self.calibration_target.get(cam)
                            if not cal_obj:
                                continue
                            cal_wps = self.visible_from.get(cal_obj, [])
                            img_wps = self.visible_from.get(obj, [])
                            if not cal_wps or not img_wps:
                                continue
                            current_pos = current_rover_pos.get(r)
                            if not current_pos:
                                continue
                            is_cal = (cam, r) in calibrated
                            cal_steps = 0
                            if not is_cal:
                                cal_steps = min([self.minimal_navigate_steps(r, current_pos, wp) for wp in cal_wps], default=float('inf'))
                                if cal_steps == float('inf'):
                                    continue
                                cal_steps += 1
                                new_pos = cal_wps[0]
                            else:
                                new_pos = current_pos
                            img_steps = min([self.minimal_navigate_steps(r, new_pos, wp) for wp in img_wps], default=float('inf'))
                            if img_steps == float('inf'):
                                continue
                            comm_steps = min([self.minimal_navigate_steps(r, wp, lv) for wp in img_wps for lv in self.lander_visible_waypoints], default=float('inf'))
                            if comm_steps == float('inf'):
                                continue
                            total = cal_steps + img_steps + 1 + comm_steps + 1
                            min_cost = min(min_cost, total)
                    total_cost += min_cost if min_cost != float('inf') else 100

        return total_cost
