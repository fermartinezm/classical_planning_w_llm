from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Rovers24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all communication goals for soil, rock, and image data. It considers navigation, sampling, calibration, imaging, and communication actions.

    # Assumptions
    - Each navigation between waypoints is estimated as 1 action, regardless of distance.
    - Rovers can handle their stores (e.g., dropping samples if necessary).
    - Calibration and imaging waypoints are reachable with one navigation step.
    - Communication requires being at a waypoint visible to the lander.

    # Heuristic Initialization
    - Extracts static information including lander's location, camera details, waypoint visibility, rover capabilities, and traversal paths.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Goals**: Identify unmet communication goals for soil, rock, and image data.
    2. **Soil/Rock Data**:
        - If sample is collected, check if rover can communicate; else, estimate sampling and communication steps.
    3. **Image Data**:
        - If image is taken, check if rover can communicate; else, estimate calibration, imaging, and communication steps.
    4. **Sum Actions**: Total heuristic is the sum of all estimated actions for unmet goals.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract lander's waypoint
        self.lander_waypoint = None
        for fact in self.static:
            if fact.startswith('(at_lander '):
                parts = fact[1:-1].split()
                if parts[1] == 'general':
                    self.lander_waypoint = parts[2]
                    break

        # Camera info: {cam: {calibration_target, on_board, supports}}
        self.camera_info = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'calibration_target':
                cam, obj = parts[1], parts[2]
                self.camera_info.setdefault(cam, {})['calibration_target'] = obj
            elif parts[0] == 'on_board':
                cam, rover = parts[1], parts[2]
                self.camera_info.setdefault(cam, {})['on_board'] = rover
            elif parts[0] == 'supports':
                cam, mode = parts[1], parts[2]
                self.camera_info.setdefault(cam, {}).setdefault('supports', set()).add(mode)

        # visible_from: {objective: set(waypoints)}
        self.visible_from = {}
        for fact in self.static:
            if fact.startswith('(visible_from '):
                parts = fact[1:-1].split()
                obj, wp = parts[1], parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)

        # Rover info: {rover: {soil, rock, imaging, store}}
        self.rovers = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rovers.setdefault(rover, {}).update(soil=True)
            elif parts[0] == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rovers.setdefault(rover, {}).update(rock=True)
            elif parts[0] == 'equipped_for_imaging':
                rover = parts[1]
                self.rovers.setdefault(rover, {}).update(imaging=True)
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.rovers.setdefault(rover, {}).update(store=store)

        # visible_to_lander: waypoints visible from lander's location
        self.visible_to_lander = set()
        for fact in self.static:
            if fact.startswith('(visible '):
                parts = fact[1:-1].split()
                from_wp, to_wp = parts[1], parts[2]
                if to_wp == self.lander_waypoint:
                    self.visible_to_lander.add(from_wp)

    def __call__(self, node):
        state = node.state
        h = 0

        unmet_goals = [g for g in self.goals if g not in state]

        for goal in unmet_goals:
            goal_parts = goal[1:-1].split()
            if goal_parts[0] == 'communicated_soil_data':
                wp = goal_parts[1]
                # Check if any rover has the analysis
                has_analysis = [f for f in state if f.startswith(f'(have_soil_analysis ') and f.endswith(f' {wp})')]
                if has_analysis:
                    for fact in has_analysis:
                        rover = fact.split()[1]
                        rover_pos = next((f.split()[2][:-1] for f in state if f.startswith(f'(at {rover} ')), None)
                        if rover_pos and rover_pos in self.visible_to_lander:
                            h += 1
                        else:
                            h += 2  # navigate + communicate
                    continue
                # Else, find a rover to sample
                for rover in self.rovers:
                    if self.rovers[rover].get('soil'):
                        store = self.rovers[rover].get('store')
                        if store and f'(empty {store})' not in state:
                            h += 1  # drop
                        h += 1  # navigate to wp
                        h += 1  # sample
                        h += 1  # navigate to visible
                        h += 1  # communicate
                        break  # Assume first capable rover handles it
            elif goal_parts[0] == 'communicated_rock_data':
                wp = goal_parts[1]
                has_analysis = [f for f in state if f.startswith(f'(have_rock_analysis ') and f.endswith(f' {wp})')]
                if has_analysis:
                    for fact in has_analysis:
                        rover = fact.split()[1]
                        rover_pos = next((f.split()[2][:-1] for f in state if f.startswith(f'(at {rover} ')), None)
                        if rover_pos and rover_pos in self.visible_to_lander:
                            h += 1
                        else:
                            h += 2
                    continue
                for rover in self.rovers:
                    if self.rovers[rover].get('rock'):
                        store = self.rovers[rover].get('store')
                        if store and f'(empty {store})' not in state:
                            h += 1
                        h += 1
                        h += 1
                        h += 1
                        h += 1
                        break
            elif goal_parts[0] == 'communicated_image_data':
                obj, mode = goal_parts[1], goal_parts[2]
                has_image = [f for f in state if f.startswith(f'(have_image ') and f.endswith(f' {obj} {mode})')]
                if has_image:
                    for fact in has_image:
                        rover = fact.split()[1]
                        rover_pos = next((f.split()[2][:-1] for f in state if f.startswith(f'(at {rover} ')), None)
                        if rover_pos and rover_pos in self.visible_to_lander:
                            h += 1
                        else:
                            h += 2
                    continue
                # Find camera and rover
                for cam, info in self.camera_info.items():
                    if info.get('calibration_target') == obj and mode in info.get('supports', set()):
                        rover = info.get('on_board')
                        if rover and self.rovers.get(rover, {}).get('imaging'):
                            calibrated = f'(calibrated {cam} {rover})' in state
                            if not calibrated:
                                h += 2  # navigate to wp + calibrate
                            h += 1  # navigate to image wp
                            h += 1  # take_image
                            h += 2  # navigate to lander visible + communicate
                            break  # Assume first valid camera
        return h
