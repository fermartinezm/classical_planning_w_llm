from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class rovers9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the number of actions required to achieve all goals by considering the minimal steps needed for each unachieved goal. It accounts for soil/rock sampling, image capturing, and data communication, leveraging rover capabilities and current state information.

    Assumptions:
    - Each rover can navigate directly between any two connected waypoints in one action.
    - Soil/rock samples can be collected if the rover has an empty store and appropriate equipment.
    - Cameras can be calibrated at any valid waypoint for their target.
    - Communication requires the rover to be at a waypoint visible to the lander's location.

    Heuristic Initialization:
    - Extracts static information such as lander position, rover equipment, camera details, and waypoint visibility.
    - Preprocesses static facts to build data structures for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each unachieved goal, determine the type (soil, rock, image).
    2. For soil/rock goals:
        a. Check if a rover has the sample. If yes, estimate steps to communicate.
        b. If not, estimate steps to navigate, sample, and communicate, considering store status.
    3. For image goals:
        a. Check if a rover has the image. If yes, estimate steps to communicate.
        b. If not, estimate steps to calibrate camera, navigate, capture image, and communicate.
    4. Sum the minimal steps for all unachieved goals to get the heuristic value.
    """

    def __init__(self, task):
        self.static = task.static
        self.goals = task.goals
        self.initial_state = task.initial_state

        self.lander_pos = None
        self.rover_equipment = {}
        self.rover_stores = {}
        self.camera_info = {}
        self.visible_from = {}
        self.can_traverse = {}
        self.visible = set()

        for fact in self.static:
            parts = fact[1:-1].split()
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'at_lander':
                self.lander_pos = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False})['soil'] = True
            elif predicate == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False})['rock'] = True
            elif predicate == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False})['imaging'] = True
            elif predicate == 'store_of':
                self.rover_stores[parts[2]] = parts[1]
            elif predicate == 'supports':
                camera = parts[1]
                mode = parts[2]
                self.camera_info.setdefault(camera, {'supports': set(), 'calibration_target': None})['supports'].add(mode)
            elif predicate == 'calibration_target':
                camera = parts[1]
                obj = parts[2]
                self.camera_info.setdefault(camera, {'supports': set(), 'calibration_target': None})['calibration_target'] = obj
            elif predicate == 'visible_from':
                obj = parts[1]
                wp = parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)
            elif predicate == 'can_traverse':
                rover = parts[1]
                self.can_traverse.setdefault(rover, set()).add((parts[2], parts[3]))
            elif predicate == 'visible':
                self.visible.add((parts[1], parts[2]))

        self.rovers = list(self.rover_equipment.keys())

    def __call__(self, node):
        state = node.state
        total = 0

        unachieved_goals = [g for g in self.goals if g not in state]
        for goal in unachieved_goals:
            parts = goal[1:-1].split()
            goal_type = parts[0]
            if goal_type == 'communicated_soil_data':
                wp = parts[1]
                total += self.estimate_soil_rock_goal(wp, state, 'soil')
            elif goal_type == 'communicated_rock_data':
                wp = parts[1]
                total += self.estimate_soil_rock_goal(wp, state, 'rock')
            elif goal_type == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                total += self.estimate_image_goal(obj, mode, state)

        return total

    def estimate_soil_rock_goal(self, wp, state, goal_type):
        min_steps = float('inf')
        lander_pos = self.lander_pos
        analysis_pred = f'have_{goal_type}_analysis'
        equipment_key = goal_type

        for rover in self.rovers:
            if not self.rover_equipment.get(rover, {}).get(equipment_key, False):
                continue
            store = self.rover_stores.get(rover)
            if not store:
                continue
            store_empty = f'(empty {store})' in state
            has_analysis = f'({analysis_pred} {rover} {wp})' in state

            if has_analysis:
                current_pos = self.get_rover_position(rover, state)
                if not current_pos:
                    continue
                steps = 1 if (current_pos, lander_pos) in self.visible else 2
                min_steps = min(min_steps, steps)
            else:
                steps = 0
                if not store_empty:
                    steps += 1
                current_pos = self.get_rover_position(rover, state)
                if current_pos != wp:
                    steps += 1
                steps += 1  # sample
                steps += 1  # navigate to x
                steps += 1  # communicate
                min_steps = min(min_steps, steps)

        return min_steps if min_steps != float('inf') else 0

    def estimate_image_goal(self, obj, mode, state):
        min_steps = float('inf')
        lander_pos = self.lander_pos

        for rover in self.rovers:
            if not self.rover_equipment.get(rover, {}).get('imaging', False):
                continue
            current_pos = self.get_rover_position(rover, state)
            if not current_pos:
                continue

            for camera in self.camera_info:
                if f'(on_board {camera} {rover})' not in self.static:
                    continue
                if mode not in self.camera_info[camera]['supports']:
                    continue
                cal_target = self.camera_info[camera]['calibration_target']
                cal_wps = self.visible_from.get(cal_target, set())
                if not cal_wps:
                    continue

                if f'(have_image {rover} {obj} {mode})' in state:
                    steps = 1 if (current_pos, lander_pos) in self.visible else 2
                    min_steps = min(min_steps, steps)
                    continue

                calibrated = f'(calibrated {camera} {rover})' in state
                steps = 0
                if not calibrated:
                    cal_wp = next(iter(cal_wps))
                    if current_pos != cal_wp:
                        steps += 1
                    steps += 1  # calibrate
                    current_pos = cal_wp

                img_wps = self.visible_from.get(obj, set())
                if not img_wps:
                    continue
                img_wp = next(iter(img_wps))
                if current_pos != img_wp:
                    steps += 1  # navigate to img_wp
                steps += 1  # take_image

                comm_pos = img_wp
                if (comm_pos, lander_pos) not in self.visible:
                    steps += 1  # navigate to visible waypoint
                    comm_pos = next((to_wp for (from_wp, to_wp) in self.visible if to_wp == lander_pos), None)
                    if not comm_pos:
                        continue
                steps += 1  # communicate

                min_steps = min(min_steps, steps)

        return min_steps if min_steps != float('inf') else 0

    def get_rover_position(self, rover, state):
        for fact in state:
            if fact.startswith(f'(at {rover} '):
                return fact[1:-1].split()[2]
        return None
