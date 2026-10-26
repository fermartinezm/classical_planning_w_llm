from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    return fact[1:-1].split()


def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern_parts))


class rovers16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goals by considering the minimal steps needed for each unmet goal. It accounts for navigation, sampling, calibration, imaging, and communication actions.

    # Assumptions
    - Each rover can handle multiple goals but may need to navigate between waypoints.
    - Soil and rock samples are available if not yet collected.
    - Cameras can be recalibrated as needed, and paths between waypoints exist if `can_traverse` allows.

    # Heuristic Initialization
    - Extracts static information including calibration targets, camera supports, visible waypoints, lander positions, rover capabilities, and traversal paths.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Static Information**: Parse static facts to build structures for quick lookup.
    2. **Parse Current State**: Track rover positions, samples, calibrated cameras, images, and store statuses.
    3. **Evaluate Unmet Goals**:
       - **Soil/Rock Data**: Check if sampled, else compute steps to sample, navigate, and communicate.
       - **Image Data**: Check if image exists, else compute steps to calibrate, take image, and communicate.
    4. **Sum Costs**: Aggregate minimal steps for each goal, considering the most efficient rover and path.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Static information
        self.calibration_target = {}  # camera -> objective
        self.on_board = {}  # camera -> rover
        self.supports = set()  # (camera, mode)
        self.visible_from = {}  # objective -> set(waypoints)
        self.at_lander = {}  # lander -> waypoint
        self.can_traverse = set()  # (rover, from, to)
        self.store_of = {}  # store -> rover
        self.equipped_imaging = set()  # rovers
        self.equipped_soil = set()  # rovers
        self.equipped_rock = set()  # rovers

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'calibration_target':
                self.calibration_target[parts[1]] = parts[2]
            elif parts[0] == 'on_board':
                self.on_board[parts[1]] = parts[2]
            elif parts[0] == 'supports':
                self.supports.add((parts[1], parts[2]))
            elif parts[0] == 'visible_from':
                obj, wp = parts[1], parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)
            elif parts[0] == 'at_lander':
                self.at_lander[parts[1]] = parts[2]
            elif parts[0] == 'can_traverse':
                self.can_traverse.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'store_of':
                self.store_of[parts[1]] = parts[2]
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_imaging.add(parts[1])
            elif parts[0] == 'equipped_for_soil_analysis':
                self.equipped_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_rock.add(parts[1])

    def __call__(self, node):
        state = node.state
        h_val = 0

        # Current communicated data
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()

        # Rover info
        rovers = self.equipped_imaging | self.equipped_soil | self.equipped_rock
        rover_info = {r: {'position': None, 'soil': set(), 'rock': set(), 'calibrated': set(), 'images': set(),
                          'store_empty': False, 'store_full': False} for r in rovers}

        # Current samples
        current_soil = set()
        current_rock = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'communicated_soil_data':
                communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                communicated_image.add((parts[1], parts[2]))
            elif parts[0] == 'at' and parts[1] in rover_info:
                rover_info[parts[1]]['position'] = parts[2]
            elif parts[0] == 'have_soil_analysis' and parts[1] in rover_info:
                rover_info[parts[1]]['soil'].add(parts[2])
            elif parts[0] == 'have_rock_analysis' and parts[1] in rover_info:
                rover_info[parts[1]]['rock'].add(parts[2])
            elif parts[0] == 'calibrated' and parts[2] in rover_info:
                rover_info[parts[2]]['calibrated'].add(parts[1])
            elif parts[0] == 'have_image' and parts[1] in rover_info:
                rover_info[parts[1]]['images'].add((parts[2], parts[3]))
            elif parts[0] == 'empty' and parts[1] in self.store_of:
                rover = self.store_of[parts[1]]
                if rover in rover_info:
                    rover_info[rover]['store_empty'] = True
                    rover_info[rover]['store_full'] = False
            elif parts[0] == 'full' and parts[1] in self.store_of:
                rover = self.store_of[parts[1]]
                if rover in rover_info:
                    rover_info[rover]['store_empty'] = False
                    rover_info[rover]['store_full'] = True
            elif parts[0] == 'at_soil_sample':
                current_soil.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                current_rock.add(parts[1])

        # Process each goal
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            if parts[0] == 'communicated_soil_data':
                wp = parts[1]
                if wp in communicated_soil:
                    continue
                min_cost = float('inf')
                # Check if any rover has the sample
                for rover in self.equipped_soil:
                    if wp in rover_info[rover]['soil']:
                        # Need to communicate
                        cost = 0
                        lander_wp = next(iter(self.at_lander.values()), None)
                        if not lander_wp:
                            continue
                        current_pos = rover_info[rover]['position']
                        if current_pos and any((rover, current_pos, l_wp) in self.can_traverse for l_wp in [lander_wp]):
                            cost += 1  # communicate
                        else:
                            cost += 2  # navigate + communicate
                        if cost < min_cost:
                            min_cost = cost
                # If no rover has sample, check sampling
                if min_cost == float('inf'):
                    for rover in self.equipped_soil:
                        if rover_info[rover]['store_full']:
                            store_cost = 1  # drop
                        else:
                            store_cost = 0
                        if wp not in current_soil:
                            continue  # Assume sample is available for heuristic
                        pos = rover_info[rover]['position']
                        navigate_sample = 0 if pos == wp else 1
                        navigate_comm = 1  # to lander visible wp
                        cost = store_cost + navigate_sample + 1 + navigate_comm + 1
                        if cost < min_cost:
                            min_cost = cost
                if min_cost != float('inf'):
                    h_val += min_cost
            elif parts[0] == 'communicated_rock_data':
                wp = parts[1]
                if wp in communicated_rock:
                    continue
                min_cost = float('inf')
                for rover in self.equipped_rock:
                    if wp in rover_info[rover]['rock']:
                        cost = 0
                        lander_wp = next(iter(self.at_lander.values()), None)
                        current_pos = rover_info[rover]['position']
                        if current_pos and any((rover, current_pos, l_wp) in self.can_traverse for l_wp in [lander_wp]):
                            cost += 1
                        else:
                            cost += 2
                        if cost < min_cost:
                            min_cost = cost
                if min_cost == float('inf'):
                    for rover in self.equipped_rock:
                        if rover_info[rover]['store_full']:
                            store_cost = 1
                        else:
                            store_cost = 0
                        if wp not in current_rock:
                            continue
                        pos = rover_info[rover]['position']
                        navigate_sample = 0 if pos == wp else 1
                        navigate_comm = 1
                        cost = store_cost + navigate_sample + 1 + navigate_comm + 1
                        if cost < min_cost:
                            min_cost = cost
                if min_cost != float('inf'):
                    h_val += min_cost
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                if (obj, mode) in communicated_image:
                    continue
                min_cost = float('inf')
                # Check existing images
                for rover in self.equipped_imaging:
                    if (obj, mode) in rover_info[rover]['images']:
                        cost = 0
                        lander_wp = next(iter(self.at_lander.values()), None)
                        current_pos = rover_info[rover]['position']
                        if current_pos and any((rover, current_pos, l_wp) in self.can_traverse for l_wp in [lander_wp]):
                            cost += 1
                        else:
                            cost += 2
                        if cost < min_cost:
                            min_cost = cost
                # If no image, find camera and steps
                if min_cost == float('inf'):
                    for camera, cam_mode in self.supports:
                        if cam_mode != mode:
                            continue
                        rover = self.on_board.get(camera, None)
                        if not rover or rover not in self.equipped_imaging:
                            continue
                        # Calibration needed?
                        cal_target = self.calibration_target.get(camera, None)
                        if not cal_target:
                            continue
                        cal_wps = self.visible_from.get(cal_target, set())
                        current_pos = rover_info[rover]['position']
                        cal_cost = 0
                        if camera not in rover_info[rover]['calibrated']:
                            if current_pos in cal_wps:
                                cal_cost = 1  # calibrate
                            else:
                                cal_cost = 2  # navigate + calibrate
                        # Take image
                        img_wps = self.visible_from.get(obj, set())
                        img_pos = current_pos if cal_cost == 0 else None
                        if not img_pos or img_pos not in img_wps:
                            img_cost = 2  # navigate + take_image
                        else:
                            img_cost = 1  # take_image
                        # Communicate
                        comm_cost = 2  # navigate + communicate
                        total_cost = cal_cost + img_cost + comm_cost
                        if total_cost < min_cost:
                            min_cost = total_cost
                if min_cost != float('inf'):
                    h_val += min_cost
        return h_val
