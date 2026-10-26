from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rovers20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all uncommunicated goals. It considers the steps needed for each rover to sample soil/rock, take images, and communicate data. The heuristic sums the minimal steps required for each goal, assuming optimal rover assignments.

    # Assumptions
    - Each rover can handle multiple goals independently.
    - Moving between waypoints takes 1 action per move.
    - Calibration is needed once per camera use.
    - Stores can be emptied by dropping samples if necessary.

    # Heuristic Initialization
    - Extracts static information: lander positions, camera calibration targets, supported modes, visible waypoints, and rover stores.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each uncommunicated soil/rock data goal:
        a. If a rover has the sample, estimate steps to communicate.
        b. If no rover has the sample, estimate steps to sample and communicate.
    2. For each uncommunicated image data goal:
        a. If a rover has the image, estimate steps to communicate.
        b. If no rover has the image, estimate steps to calibrate, take image, and communicate.
    3. Sum the minimal steps for all goals, considering optimal rover assignments.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Static data structures
        self.lander_positions = {}  # lander -> waypoint
        self.calibration_targets = {}  # camera -> objective
        self.supports_modes = defaultdict(set)  # camera -> set of modes
        self.visible_from = defaultdict(set)  # objective -> set of waypoints
        self.on_board = {}  # camera -> rover
        self.store_of = {}  # store -> rover
        self.visible = defaultdict(set)  # waypoint -> visible waypoints

        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, 'at_lander', '*', '*'):
                lander, wp = parts[1], parts[2]
                self.lander_positions[lander] = wp
            elif match(fact, 'calibration_target', '*', '*'):
                cam, obj = parts[1], parts[2]
                self.calibration_targets[cam] = obj
            elif match(fact, 'supports', '*', '*'):
                cam, mode = parts[1], parts[2]
                self.supports_modes[cam].add(mode)
            elif match(fact, 'visible_from', '*', '*'):
                obj, wp = parts[1], parts[2]
                self.visible_from[obj].add(wp)
            elif match(fact, 'on_board', '*', '*'):
                cam, rover = parts[1], parts[2]
                self.on_board[cam] = rover
            elif match(fact, 'store_of', '*', '*'):
                store, rover = parts[1], parts[2]
                self.store_of[store] = rover
            elif match(fact, 'visible', '*', '*'):
                wp1, wp2 = parts[1], parts[2]
                self.visible[wp2].add(wp1)

    def __call__(self, node):
        state = node.state
        total = 0

        # Extract communicated data
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()
        for fact in state:
            if match(fact, 'communicated_soil_data', '*'):
                communicated_soil.add(get_parts(fact)[1])
            elif match(fact, 'communicated_rock_data', '*'):
                communicated_rock.add(get_parts(fact)[1])
            elif match(fact, 'communicated_image_data', '*', '*'):
                parts = get_parts(fact)
                communicated_image.add((parts[1], parts[2]))

        # Extract rover info
        rover_info = defaultdict(lambda: {
            'position': None,
            'equipped_soil': False,
            'equipped_rock': False,
            'equipped_imaging': False,
            'store': None,
            'store_empty': False,
            'store_full': False,
            'soil_samples': set(),
            'rock_samples': set(),
            'calibrated_cams': set(),
            'images': set()
        })

        for fact in state:
            parts = get_parts(fact)
            if match(fact, 'at', 'rover*', '*'):
                rover, wp = parts[1], parts[2]
                rover_info[rover]['position'] = wp
            elif match(fact, 'equipped_for_soil_analysis', 'rover*'):
                rover = parts[1]
                rover_info[rover]['equipped_soil'] = True
            elif match(fact, 'equipped_for_rock_analysis', 'rover*'):
                rover = parts[1]
                rover_info[rover]['equipped_rock'] = True
            elif match(fact, 'equipped_for_imaging', 'rover*'):
                rover = parts[1]
                rover_info[rover]['equipped_imaging'] = True
            elif match(fact, 'store_of', '*', 'rover*'):
                store, rover = parts[1], parts[2]
                rover_info[rover]['store'] = store
            elif match(fact, 'empty', '*'):
                store = parts[1]
                rover = self.store_of.get(store, None)
                if rover:
                    rover_info[rover]['store_empty'] = True
                    rover_info[rover]['store_full'] = False
            elif match(fact, 'full', '*'):
                store = parts[1]
                rover = self.store_of.get(store, None)
                if rover:
                    rover_info[rover]['store_full'] = True
                    rover_info[rover]['store_empty'] = False
            elif match(fact, 'have_soil_analysis', 'rover*', '*'):
                rover, wp = parts[1], parts[2]
                rover_info[rover]['soil_samples'].add(wp)
            elif match(fact, 'have_rock_analysis', 'rover*', '*'):
                rover, wp = parts[1], parts[2]
                rover_info[rover]['rock_samples'].add(wp)
            elif match(fact, 'calibrated', '*', 'rover*'):
                cam, rover = parts[1], parts[2]
                rover_info[rover]['calibrated_cams'].add(cam)
            elif match(fact, 'have_image', 'rover*', '*', '*'):
                rover, obj, mode = parts[1], parts[2], parts[3]
                rover_info[rover]['images'].add((obj, mode))

        # Process soil data goals
        for goal in self.goals:
            if match(goal, 'communicated_soil_data', '*'):
                wp = get_parts(goal)[1]
                if wp in communicated_soil:
                    continue
                min_steps = float('inf')
                lander_wp = self.lander_positions.get('general', None)
                if not lander_wp:
                    continue
                visible_to_lander = self.visible.get(lander_wp, set())

                # Check rovers with sample
                for rover, info in rover_info.items():
                    if wp in info['soil_samples']:
                        pos = info['position']
                        if pos in visible_to_lander:
                            steps = 1  # communicate
                        else:
                            steps = 2  # move + communicate
                        if steps < min_steps:
                            min_steps = steps

                # Check rovers that can sample
                if min_steps == float('inf'):
                    for rover, info in rover_info.items():
                        if not info['equipped_soil']:
                            continue
                        store_full = info['store_full']
                        steps = 0
                        if store_full:
                            steps += 1  # drop
                        steps += 1  # move to wp
                        steps += 1  # sample
                        steps += 1  # move to visible
                        steps += 1  # communicate
                        if steps < min_steps:
                            min_steps = steps

                if min_steps != float('inf'):
                    total += min_steps

        # Process rock data goals (similar to soil)
        for goal in self.goals:
            if match(goal, 'communicated_rock_data', '*'):
                wp = get_parts(goal)[1]
                if wp in communicated_rock:
                    continue
                min_steps = float('inf')
                lander_wp = self.lander_positions.get('general', None)
                if not lander_wp:
                    continue
                visible_to_lander = self.visible.get(lander_wp, set())

                for rover, info in rover_info.items():
                    if wp in info['rock_samples']:
                        pos = info['position']
                        if pos in visible_to_lander:
                            steps = 1
                        else:
                            steps = 2
                        if steps < min_steps:
                            min_steps = steps

                if min_steps == float('inf'):
                    for rover, info in rover_info.items():
                        if not info['equipped_rock']:
                            continue
                        store_full = info['store_full']
                        steps = 0
                        if store_full:
                            steps += 1
                        steps += 1
                        steps += 1
                        steps += 1
                        steps += 1
                        if steps < min_steps:
                            min_steps = steps

                if min_steps != float('inf'):
                    total += min_steps

        # Process image data goals
        for goal in self.goals:
            if match(goal, 'communicated_image_data', '*', '*'):
                obj, mode = get_parts(goal)[1], get_parts(goal)[2]
                if (obj, mode) in communicated_image:
                    continue
                min_steps = float('inf')
                lander_wp = self.lander_positions.get('general', None)
                if not lander_wp:
                    continue
                visible_to_lander = self.visible.get(lander_wp, set())

                # Check rovers with image
                for rover, info in rover_info.items():
                    if (obj, mode) in info['images']:
                        pos = info['position']
                        if pos in visible_to_lander:
                            steps = 1
                        else:
                            steps = 2
                        if steps < min_steps:
                            min_steps = steps

                # Check rovers that can take image
                if min_steps == float('inf'):
                    for cam, cam_obj in self.calibration_targets.items():
                        rover = self.on_board.get(cam, None)
                        if not rover:
                            continue
                        info = rover_info[rover]
                        if not info['equipped_imaging']:
                            continue
                        if mode not in self.supports_modes[cam]:
                            continue
                        cal_wps = self.visible_from.get(cam_obj, set())
                        img_wps = self.visible_from.get(obj, set())
                        if not cal_wps or not img_wps:
                            continue

                        steps = 0
                        if cam not in info['calibrated_cams']:
                            steps += 2  # move to cal wp + calibrate
                        steps += 1  # move to img wp
                        steps += 1  # take image
                        steps += 1  # move to visible
                        steps += 1  # communicate
                        if steps < min_steps:
                            min_steps = steps

                if min_steps != float('inf'):
                    total += min_steps

        return total
