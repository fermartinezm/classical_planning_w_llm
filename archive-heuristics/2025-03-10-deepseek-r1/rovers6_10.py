from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class rovers6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goals by considering the steps needed for each unachieved goal, including movement, sampling, calibration, imaging, and communication.

    # Assumptions
    - Rovers can navigate between waypoints directly connected via 'can_traverse' and 'visible' in one move.
    - If a rover's store is full, it must drop before sampling (adds one action).
    - Calibration requires moving to a waypoint visible from the camera's target once.
    - Communication requires moving to a waypoint visible from any lander's location.

    # Heuristic Initialization
    - Extract static information: lander positions, camera calibration targets, supported modes, rover equipment, stores, and navigation/visibility graphs.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unachieved goal:
        a. Communicated soil/rock data:
            i. If a rover has the sample, estimate steps to communicate.
            ii. Else, estimate steps to sample and communicate.
        b. Communicated image data:
            i. If a rover has the image, estimate steps to communicate.
            ii. Else, estimate steps to calibrate, take image, and communicate.
    2. Sum the minimal steps for all goals, assuming optimal rover assignments.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract lander positions
        self.landers = {}
        # Camera calibration targets and supported modes
        self.calibration_targets = {}
        self.supports_modes = defaultdict(set)
        # Rover equipment and stores
        self.on_board = defaultdict(list)
        self.store_of = {}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        # Navigation and visibility
        self.can_traverse = defaultdict(set)
        self.visible = set()

        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'at_lander':
                self.landers[parts[1]] = parts[2]
            elif parts[0] == 'calibration_target':
                self.calibration_targets[parts[1]] = parts[2]
            elif parts[0] == 'supports':
                self.supports_modes[parts[1]].add(parts[2])
            elif parts[0] == 'on_board':
                self.on_board[parts[2]].append(parts[1])
            elif parts[0] == 'store_of':
                self.store_of[parts[1]] = parts[2]
            elif parts[0] == 'equipped_for_soil_analysis':
                self.equipped_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_rock.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_imaging.add(parts[1])
            elif parts[0] == 'can_traverse':
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                self.can_traverse[rover].add((from_wp, to_wp))
            elif parts[0] == 'visible':
                self.visible.add((parts[1], parts[2]))

        self.rover_store = {rover: store for store, rover in self.store_of.items()}

    def __call__(self, node):
        state = node.state
        unmet_goals = [g for g in self.goals if g not in state]
        if not unmet_goals:
            return 0

        heuristic = 0

        def get_rover_loc(rover):
            for fact in state:
                if fact.startswith(f'(at {rover} '):
                    return get_parts(fact)[2]
            return None

        for goal in unmet_goals:
            g_parts = get_parts(goal)
            if g_parts[0] == 'communicated_soil_data':
                wp = g_parts[1]
                has_soil = any(f'(have_soil_analysis {r} {wp})' in state for r in self.equipped_soil)
                if has_soil:
                    min_steps = 2  # Default move + communicate
                    for rover in self.equipped_soil:
                        if f'(have_soil_analysis {rover} {wp})' not in state:
                            continue
                        loc = get_rover_loc(rover)
                        if loc and any((loc, l_wp) in self.visible for l_wp in self.landers.values()):
                            min_steps = 1
                            break
                    heuristic += min_steps
                else:
                    min_steps = 5  # Assume worst case
                    for rover in self.equipped_soil:
                        store = self.rover_store.get(rover)
                        if not store:
                            continue
                        steps = 1 if f'(full {store})' in state else 0
                        steps += 4  # move, sample, move, communicate
                        min_steps = min(min_steps, steps)
                    heuristic += min_steps

            elif g_parts[0] == 'communicated_rock_data':
                wp = g_parts[1]
                has_rock = any(f'(have_rock_analysis {r} {wp})' in state for r in self.equipped_rock)
                if has_rock:
                    min_steps = 2
                    for rover in self.equipped_rock:
                        if f'(have_rock_analysis {rover} {wp})' not in state:
                            continue
                        loc = get_rover_loc(rover)
                        if loc and any((loc, l_wp) in self.visible for l_wp in self.landers.values()):
                            min_steps = 1
                            break
                    heuristic += min_steps
                else:
                    min_steps = 5
                    for rover in self.equipped_rock:
                        store = self.rover_store.get(rover)
                        if not store:
                            continue
                        steps = 1 if f'(full {store})' in state else 0
                        steps += 4
                        min_steps = min(min_steps, steps)
                    heuristic += min_steps

            elif g_parts[0] == 'communicated_image_data':
                obj, mode = g_parts[1], g_parts[2]
                has_image = any(f'(have_image {r} {obj} {mode})' in state for r in self.equipped_imaging)
                if has_image:
                    min_steps = 2
                    for rover in self.equipped_imaging:
                        if f'(have_image {rover} {obj} {mode})' not in state:
                            continue
                        loc = get_rover_loc(rover)
                        if loc and any((loc, l_wp) in self.visible for l_wp in self.landers.values()):
                            min_steps = 1
                            break
                    heuristic += min_steps
                else:
                    min_steps = 6  # calibrate (2) + image (2) + comm (2)
                    for rover in self.equipped_imaging:
                        for cam in self.on_board.get(rover, []):
                            if mode not in self.supports_modes[cam]:
                                continue
                            cal_target = self.calibration_targets.get(cam)
                            if not cal_target:
                                continue
                            calibrated = f'(calibrated {cam} {rover})' in state
                            steps = 0 if calibrated else 2
                            steps += 4  # move, image, move, communicate
                            min_steps = min(min_steps, steps)
                    heuristic += min_steps

        return heuristic
