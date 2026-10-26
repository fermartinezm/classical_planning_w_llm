from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class Rovers11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    Estimates the number of actions required to achieve all communication goals by considering the minimal steps needed for each unachieved goal, including sampling, calibration, imaging, and communication actions.

    # Assumptions
    - Each goal can be handled by the most efficient rover/camera possible.
    - Navigation between waypoints takes one action, regardless of distance.
    - Stores can be emptied as needed (adding steps for drop actions if full).
    - Cameras can be calibrated once per image if needed.

    # Heuristic Initialization
    - Extract static information including lander's position, rover equipment, camera details, visibility, and traversal capabilities.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unachieved soil/rock communication goal:
        a. Find rovers capable of sampling.
        b. Calculate steps to sample (including navigation and store management).
        c. Calculate steps to navigate to a lander-visible waypoint and communicate.
    2. For each unachieved image communication goal:
        a. Find cameras supporting the required mode and their rovers.
        b. Calculate steps to calibrate (if needed), take image, and communicate.
    3. Sum the minimal steps for all goals.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract lander's position
        self.lander_wp = None
        for fact in self.static:
            if match(fact, 'at_lander', 'general', '*'):
                parts = get_parts(fact)
                self.lander_wp = parts[2]
                break

        # Rover equipment
        self.rover_equipment = defaultdict(set)
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) >= 2 and parts[0] in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                rover = parts[1]
                eq = parts[0].split('_')[-1]
                self.rover_equipment[rover].add(eq)

        # Camera info: {camera: {rover, modes, calibration_target}}
        self.cameras = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'on_board':
                cam, rover = parts[1], parts[2]
                self.cameras[cam] = {'rover': rover, 'modes': set(), 'calibration_target': None}
            elif parts[0] == 'supports' and parts[1] in self.cameras:
                self.cameras[parts[1]]['modes'].add(parts[2])
            elif parts[0] == 'calibration_target' and parts[1] in self.cameras:
                self.cameras[parts[1]]['calibration_target'] = parts[2]

        # Visible waypoints (from, to)
        self.visible = set()
        for fact in self.static:
            if match(fact, 'visible', '*', '*'):
                from_wp, to_wp = get_parts(fact)[1], get_parts(fact)[2]
                self.visible.add((from_wp, to_wp))

        # Visible_from objectives: {objective: set(waypoints)}
        self.visible_from = defaultdict(set)
        for fact in self.static:
            if match(fact, 'visible_from', '*', '*'):
                obj, wp = get_parts(fact)[1], get_parts(fact)[2]
                self.visible_from[obj].add(wp)

        # Store_of: {store: rover}
        self.store_of = {}
        for fact in self.static:
            if match(fact, 'store_of', '*', '*'):
                store, rover = get_parts(fact)[1], get_parts(fact)[2]
                self.store_of[store] = rover

    def __call__(self, node):
        state = node.state
        total = 0
        achieved_goals = set(g for g in self.goals if g in state)

        for goal in self.goals:
            if goal in achieved_goals:
                continue
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                wp = parts[1]
                total += self.handle_soil_goal(wp, state)
            elif parts[0] == 'communicated_rock_data':
                wp = parts[1]
                total += self.handle_rock_goal(wp, state)
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                total += self.handle_image_goal(obj, mode, state)
        return total

    def handle_soil_goal(self, wp, state):
        min_steps = float('inf')
        for rover, eqs in self.rover_equipment.items():
            if 'soil' not in eqs:
                continue
            steps = 0
            has_soil = f'(have_soil_analysis {rover} {wp})' in state
            if not has_soil:
                at_wp = f'(at {rover} {wp})' in state
                if not at_wp:
                    steps += 1
                store = next((s for s, r in self.store_of.items() if r == rover), None)
                if store and f'(full {store})' in state:
                    steps += 1
                steps += 1
            comm_steps = self.get_communication_steps(rover, state)
            steps += comm_steps
            min_steps = min(min_steps, steps)
        return min_steps if min_steps != float('inf') else 0

    def handle_rock_goal(self, wp, state):
        min_steps = float('inf')
        for rover, eqs in self.rover_equipment.items():
            if 'rock' not in eqs:
                continue
            steps = 0
            has_rock = f'(have_rock_analysis {rover} {wp})' in state
            if not has_rock:
                at_wp = f'(at {rover} {wp})' in state
                if not at_wp:
                    steps += 1
                store = next((s for s, r in self.store_of.items() if r == rover), None)
                if store and f'(full {store})' in state:
                    steps += 1
                steps += 1
            comm_steps = self.get_communication_steps(rover, state)
            steps += comm_steps
            min_steps = min(min_steps, steps)
        return min_steps if min_steps != float('inf') else 0

    def handle_image_goal(self, obj, mode, state):
        min_steps = float('inf')
        for cam, info in self.cameras.items():
            if mode not in info['modes']:
                continue
            rover = info['rover']
            if 'imaging' not in self.rover_equipment.get(rover, set()):
                continue
            steps = 0
            have_image = f'(have_image {rover} {obj} {mode})' in state
            if not have_image:
                calibrated = f'(calibrated {cam} {rover})' in state
                if not calibrated:
                    cal_target = info['calibration_target']
                    cal_wps = self.visible_from.get(cal_target, set())
                    current_pos = self.get_rover_position(rover, state)
                    if current_pos in cal_wps:
                        steps += 1
                    else:
                        steps += 2
                img_wps = self.visible_from.get(obj, set())
                current_pos = self.get_rover_position(rover, state)
                if current_pos not in img_wps:
                    steps += 2
                steps += 1
            comm_steps = self.get_communication_steps(rover, state)
            steps += comm_steps
            min_steps = min(min_steps, steps)
        return min_steps if min_steps != float('inf') else 0

    def get_communication_steps(self, rover, state):
        current_pos = self.get_rover_position(rover, state)
        if current_pos and (current_pos, self.lander_wp) in self.visible:
            return 1
        return 2

    def get_rover_position(self, rover, state):
        for fact in state:
            if match(fact, 'at', rover, '*'):
                return get_parts(fact)[2]
        return None
