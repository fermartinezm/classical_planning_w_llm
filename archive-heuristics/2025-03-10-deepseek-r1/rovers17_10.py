from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *pattern):
    parts = get_parts(fact)
    if len(parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

class rovers17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all communication goals by considering the minimal steps needed for each unachieved goal. It accounts for soil/rock sampling, image capturing, calibration, and communication actions.

    # Assumptions
    - Each rover can navigate between waypoints in one action if can_traverse allows.
    - Communication requires being at a waypoint visible from any lander's location.
    - Calibration and image capture steps are based on current rover positions and camera statuses.
    - Sampling requires an empty store, and full stores may need to be dropped.

    # Heuristic Initialization
    - Extracts static information: lander positions, rover stores, camera calibration targets, on-board cameras, supported modes, visible waypoints, and can_traverse links.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each uncommunicated soil/rock data goal:
        a. If a rover has the sample, estimate steps to communicate.
        b. Else, estimate steps to sample and communicate.
    2. For each uncommunicated image data goal:
        a. If a rover has the image, estimate steps to communicate.
        b. Else, estimate steps to calibrate, capture image, and communicate.
    3. Sum the minimal steps for all goals.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_waypoints = {}
        self.rover_stores = {}
        self.calibration_target = {}
        self.on_board = defaultdict(list)
        self.supports = defaultdict(set)
        self.visible_from = defaultdict(set)
        self.can_traverse = defaultdict(set)
        self.visible = defaultdict(set)

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'at_lander':
                lander = parts[1]
                waypoint = parts[2]
                self.lander_waypoints[lander] = waypoint
            elif parts[0] == 'store_of':
                store = parts[1]
                rover = parts[2]
                self.rover_stores[rover] = store
            elif parts[0] == 'calibration_target':
                camera = parts[1]
                objective = parts[2]
                self.calibration_target[camera] = objective
            elif parts[0] == 'on_board':
                camera = parts[1]
                rover = parts[2]
                self.on_board[rover].append(camera)
            elif parts[0] == 'supports':
                camera = parts[1]
                mode = parts[2]
                self.supports[camera].add(mode)
            elif parts[0] == 'visible_from':
                objective = parts[1]
                waypoint = parts[2]
                self.visible_from[objective].add(waypoint)
            elif parts[0] == 'can_traverse':
                rover = parts[1]
                from_wp = parts[2]
                to_wp = parts[3]
                self.can_traverse[rover].add((from_wp, to_wp))
            elif parts[0] == 'visible':
                from_wp = parts[1]
                to_wp = parts[2]
                self.visible[from_wp].add(to_wp)

    def __call__(self, node):
        state = node.state
        if node.parent is None and self.goals <= state:
            return 0

        current_communicated_soil = set()
        current_communicated_rock = set()
        current_communicated_image = set()
        current_have_soil = defaultdict(set)
        current_have_rock = defaultdict(set)
        current_have_image = defaultdict(set)
        current_rover_pos = {}
        current_empty_stores = set()
        current_full_stores = set()
        current_calibrated = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'communicated_soil_data':
                current_communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                current_communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                current_communicated_image.add((parts[1], parts[2]))
            elif parts[0] == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                current_have_soil[rover].add(wp)
            elif parts[0] == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                current_have_rock[rover].add(wp)
            elif parts[0] == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                current_have_image[rover].add((obj, mode))
            elif parts[0] == 'at':
                rover, wp = parts[1], parts[2]
                current_rover_pos[rover] = wp
            elif parts[0] == 'empty':
                current_empty_stores.add(parts[1])
            elif parts[0] == 'full':
                current_full_stores.add(parts[1])
            elif parts[0] == 'calibrated':
                camera, rover = parts[1], parts[2]
                current_calibrated.add((camera, rover))

        total_cost = 0

        # Process soil data goals
        for goal in self.goals:
            if match(goal, 'communicated_soil_data', '*'):
                wp = get_parts(goal)[1]
                if wp in current_communicated_soil:
                    continue
                min_cost = float('inf')
                # Check rovers with the sample
                for rover in current_have_soil:
                    if wp in current_have_soil[rover]:
                        for lander_wp in self.lander_waypoints.values():
                            visible_waypoints = self.visible.get(lander_wp, set())
                            current_pos = current_rover_pos.get(rover, '')
                            if current_pos in visible_waypoints:
                                cost = 1
                            else:
                                cost = 2  # navigate + communicate
                            if cost < min_cost:
                                min_cost = cost
                # If no rover has sample, find one to collect
                if min_cost == float('inf'):
                    for rover in self.rover_stores:
                        if f'equipped_for_soil_analysis {rover}' not in state:
                            continue
                        store = self.rover_stores[rover]
                        drop_cost = 1 if store in current_full_stores else 0
                        # Assume can navigate to wp and back
                        cost = drop_cost + 4  # navigate, sample, navigate, communicate
                        if cost < min_cost:
                            min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost

        # Process rock data goals (similar to soil)
        for goal in self.goals:
            if match(goal, 'communicated_rock_data', '*'):
                wp = get_parts(goal)[1]
                if wp in current_communicated_rock:
                    continue
                min_cost = float('inf')
                for rover in current_have_rock:
                    if wp in current_have_rock[rover]:
                        for lander_wp in self.lander_waypoints.values():
                            visible_waypoints = self.visible.get(lander_wp, set())
                            current_pos = current_rover_pos.get(rover, '')
                            if current_pos in visible_waypoints:
                                cost = 1
                            else:
                                cost = 2
                            if cost < min_cost:
                                min_cost = cost
                if min_cost == float('inf'):
                    for rover in self.rover_stores:
                        if f'equipped_for_rock_analysis {rover}' not in state:
                            continue
                        store = self.rover_stores[rover]
                        drop_cost = 1 if store in current_full_stores else 0
                        cost = drop_cost + 4
                        if cost < min_cost:
                            min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost

        # Process image data goals
        for goal in self.goals:
            if match(goal, 'communicated_image_data', '*', '*'):
                obj, mode = get_parts(goal)[1], get_parts(goal)[2]
                if (obj, mode) in current_communicated_image:
                    continue
                min_cost = float('inf')
                # Check rovers with the image
                for rover in current_have_image:
                    if (obj, mode) in current_have_image[rover]:
                        for lander_wp in self.lander_waypoints.values():
                            visible_waypoints = self.visible.get(lander_wp, set())
                            current_pos = current_rover_pos.get(rover, '')
                            if current_pos in visible_waypoints:
                                cost = 1
                            else:
                                cost = 2
                            if cost < min_cost:
                                min_cost = cost
                # If no rover has image, find one to take it
                if min_cost == float('inf'):
                    for rover in self.on_board:
                        if f'equipped_for_imaging {rover}' not in state:
                            continue
                        for camera in self.on_board[rover]:
                            if mode not in self.supports[camera]:
                                continue
                            cal_obj = self.calibration_target.get(camera, None)
                            if not cal_obj:
                                continue
                            calibrated = (camera, rover) in current_calibrated
                            cal_visible = self.visible_from.get(cal_obj, set())
                            cal_cost = 0 if calibrated else 2  # navigate + calibrate
                            img_visible = self.visible_from.get(obj, set())
                            # Assume can navigate to img_visible waypoint
                            img_cost = 2  # navigate + take_image
                            comm_cost = 2  # navigate + communicate
                            total_cost_cam = cal_cost + img_cost + comm_cost
                            if total_cost_cam < min_cost:
                                min_cost = total_cost_cam
                if min_cost != float('inf'):
                    total_cost += min_cost

        return total_cost
