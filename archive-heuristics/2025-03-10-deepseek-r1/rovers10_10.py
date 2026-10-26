from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rovers10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required for rovers to achieve all goals, which include communicating soil/rock data from specific waypoints and image data of objectives in certain modes. The heuristic considers the current state of rovers, their equipment, and static information about the environment to compute an estimate.

    # Assumptions
    - Each rover can carry only one sample at a time (due to a single store).
    - Cameras need calibration before taking images, which requires being at a waypoint visible from the calibration target.
    - Communication requires the rover to be at a waypoint visible from the lander's location.
    - Navigate actions between directly connected waypoints are assumed to take one step.

    # Heuristic Initialization
    - Extract static information including lander location, camera details, rover equipment, waypoint visibility, and traversal capabilities.
    - Preprocess goals to identify required soil/rock data and image data.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Soil/Rock Data**:
        a. If data is already communicated, skip.
        b. If a rover has the sample, compute steps to communicate from a visible waypoint.
        c. If no rover has the sample, compute steps to collect it and communicate.
    2. **Image Data**:
        a. If image is already communicated, skip.
        b. If a rover has the image, compute steps to communicate.
        c. If no rover has the image, compute steps to calibrate camera, take the image, and communicate.
    3. Sum the minimal steps for all unachieved goals.
    """

    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Extract lander's location
        self.lander_location = None
        for fact in self.static:
            if match(fact, 'at_lander', 'general', '*'):
                self.lander_location = get_parts(fact)[2]
                break

        # Camera info: {camera: {rover, modes, calibration_target}}
        self.camera_info = {}
        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, 'on_board', '*', '*'):
                cam, rover = parts[1], parts[2]
                if cam not in self.camera_info:
                    self.camera_info[cam] = {'rover': rover, 'modes': set(), 'calibration_target': None}
            elif match(fact, 'supports', '*', '*'):
                cam, mode = parts[1], parts[2]
                if cam in self.camera_info:
                    self.camera_info[cam]['modes'].add(mode)
            elif match(fact, 'calibration_target', '*', '*'):
                cam, target = parts[1], parts[2]
                if cam in self.camera_info:
                    self.camera_info[cam]['calibration_target'] = target

        # Rover equipment: {rover: {soil, rock, imaging}}
        self.rovers_equipment = {}
        for fact in self.static:
            if match(fact, 'equipped_for_soil_analysis', '*'):
                rover = get_parts(fact)[1]
                self.rovers_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False})['soil'] = True
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                rover = get_parts(fact)[1]
                self.rovers_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False})['rock'] = True
            elif match(fact, 'equipped_for_imaging', '*'):
                rover = get_parts(fact)[1]
                self.rovers_equipment.setdefault(rover, {'soil': False, 'rock': False, 'imaging': False})['imaging'] = True

        # Store assignments: {store: rover}
        self.store_of = {}
        for fact in self.static:
            if match(fact, 'store_of', '*', '*'):
                store, rover = get_parts(fact)[1], get_parts(fact)[2]
                self.store_of[store] = rover

        # Visible waypoint pairs
        self.visible_pairs = set()
        for fact in self.static:
            if match(fact, 'visible', '*', '*'):
                from_wp, to_wp = get_parts(fact)[1], get_parts(fact)[2]
                self.visible_pairs.add((from_wp, to_wp))

        # Visible from objectives: {objective: set(waypoints)}
        self.visible_from = {}
        for fact in self.static:
            if match(fact, 'visible_from', '*', '*'):
                obj, wp = get_parts(fact)[1], get_parts(fact)[2]
                self.visible_from.setdefault(obj, set()).add(wp)

        # Can traverse: {rover: set((from, to))}
        self.can_traverse = {}
        for fact in self.static:
            if match(fact, 'can_traverse', '*', '*', '*'):
                rover, from_wp, to_wp = get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]
                self.can_traverse.setdefault(rover, set()).add((from_wp, to_wp))

        # Preprocess goals into soil, rock, and image goals
        self.goal_soil = set()
        self.goal_rock = set()
        self.goal_images = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if match(goal, 'communicated_soil_data', '*'):
                self.goal_soil.add(parts[1])
            elif match(goal, 'communicated_rock_data', '*'):
                self.goal_rock.add(parts[1])
            elif match(goal, 'communicated_image_data', '*', '*'):
                self.goal_images.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Helper to check if a fact is in state
        has_fact = lambda f: any(match(sf, *get_parts(f)) for sf in state)

        # Process soil data goals
        for wp in self.goal_soil:
            goal_fact = f'(communicated_soil_data {wp})'
            if has_fact(goal_fact):
                continue

            # Check if any rover has the sample
            sample_holder = None
            for fact in state:
                if match(fact, 'have_soil_analysis', '*', wp):
                    sample_holder = get_parts(fact)[1]
                    break

            if sample_holder:
                # Find rover's current position
                rover_pos = None
                for fact in state:
                    if match(fact, 'at', sample_holder, '*'):
                        rover_pos = get_parts(fact)[2]
                        break
                if not rover_pos:
                    continue

                # Check if current position is visible to lander
                if (rover_pos, self.lander_location) in self.visible_pairs:
                    comm_cost = 1
                else:
                    comm_cost = 2  # navigate + communicate
                total_cost += comm_cost
            else:
                # Find a rover to collect the sample
                min_cost = float('inf')
                for rover in self.rovers_equipment:
                    if not self.rovers_equipment[rover]['soil']:
                        continue

                    # Find the rover's store
                    store = next((s for s, r in self.store_of.items() if r == rover), None)
                    if not store or not any(match(fact, 'empty', store) for fact in state):
                        continue

                    # Rover's current position
                    rover_pos = None
                    for fact in state:
                        if match(fact, 'at', rover, '*'):
                            rover_pos = get_parts(fact)[2]
                            break
                    if not rover_pos:
                        continue

                    # Navigate to wp cost
                    navigate_cost = 1 if rover_pos != wp else 0

                    # After sample, navigate to a waypoint visible to lander
                    if (wp, self.lander_location) in self.visible_pairs:
                        comm_steps = 1  # communicate from wp
                    else:
                        comm_steps = 2  # navigate + communicate

                    total_steps = navigate_cost + 1 + comm_steps
                    min_cost = min(min_cost, total_steps)

                if min_cost != float('inf'):
                    total_cost += min_cost

        # Process rock data goals (similar to soil)
        for wp in self.goal_rock:
            goal_fact = f'(communicated_rock_data {wp})'
            if has_fact(goal_fact):
                continue

            sample_holder = None
            for fact in state:
                if match(fact, 'have_rock_analysis', '*', wp):
                    sample_holder = get_parts(fact)[1]
                    break

            if sample_holder:
                rover_pos = None
                for fact in state:
                    if match(fact, 'at', sample_holder, '*'):
                        rover_pos = get_parts(fact)[2]
                        break
                if not rover_pos:
                    continue

                if (rover_pos, self.lander_location) in self.visible_pairs:
                    comm_cost = 1
                else:
                    comm_cost = 2
                total_cost += comm_cost
            else:
                min_cost = float('inf')
                for rover in self.rovers_equipment:
                    if not self.rovers_equipment[rover]['rock']:
                        continue

                    store = next((s for s, r in self.store_of.items() if r == rover), None)
                    if not store or not any(match(fact, 'empty', store) for fact in state):
                        continue

                    rover_pos = None
                    for fact in state:
                        if match(fact, 'at', rover, '*'):
                            rover_pos = get_parts(fact)[2]
                            break
                    if not rover_pos:
                        continue

                    navigate_cost = 1 if rover_pos != wp else 0

                    if (wp, self.lander_location) in self.visible_pairs:
                        comm_steps = 1
                    else:
                        comm_steps = 2

                    total_steps = navigate_cost + 1 + comm_steps
                    min_cost = min(min_cost, total_steps)

                if min_cost != float('inf'):
                    total_cost += min_cost

        # Process image data goals
        for (obj, mode) in self.goal_images:
            goal_fact = f'(communicated_image_data {obj} {mode})'
            if has_fact(goal_fact):
                continue

            # Check if any rover has the image
            image_holder = None
            for fact in state:
                if match(fact, 'have_image', '*', obj, mode):
                    image_holder = get_parts(fact)[1]
                    break

            if image_holder:
                rover_pos = None
                for fact in state:
                    if match(fact, 'at', image_holder, '*'):
                        rover_pos = get_parts(fact)[2]
                        break
                if not rover_pos:
                    continue

                if (rover_pos, self.lander_location) in self.visible_pairs:
                    comm_cost = 1
                else:
                    comm_cost = 2
                total_cost += comm_cost
            else:
                min_cost = float('inf')
                for cam in self.camera_info.values():
                    if mode not in cam['modes'] or cam['calibration_target'] != obj:
                        continue
                    rover = cam['rover']
                    if not self.rovers_equipment.get(rover, {}).get('imaging', False):
                        continue

                    # Check calibration
                    calibrated = any(match(fact, 'calibrated', cam['rover'], rover) for fact in state)
                    cal_wps = self.visible_from.get(obj, [])
                    if not cal_wps:
                        continue

                    # Current position
                    rover_pos = None
                    for fact in state:
                        if match(fact, 'at', rover, '*'):
                            rover_pos = get_parts(fact)[2]
                            break
                    if not rover_pos:
                        continue

                    cal_cost = 0
                    if not calibrated:
                        # Find any cal_wp
                        cal_wp = next(iter(cal_wps))
                        # Check if can traverse from current to cal_wp
                        can_traverse = self.can_traverse.get(rover, set())
                        if (rover_pos, cal_wp) in can_traverse and (rover_pos, cal_wp) in self.visible_pairs:
                            navigate_cal = 0 if rover_pos == cal_wp else 1
                        else:
                            navigate_cal = 1  # assume possible in 1 step
                        cal_cost = navigate_cal + 1

                    # Take image at any visible wp
                    image_wps = self.visible_from.get(obj, [])
                    if not image_wps:
                        continue
                    image_wp = next(iter(image_wps))
                    current_after_cal = cal_wp if not calibrated else rover_pos
                    if current_after_cal != image_wp:
                        navigate_image = 1
                    else:
                        navigate_image = 0

                    # Communicate
                    if (image_wp, self.lander_location) in self.visible_pairs:
                        comm_steps = 1
                    else:
                        comm_steps = 2

                    total_steps = cal_cost + navigate_image + 1 + comm_steps
                    min_cost = min(min_cost, total_steps)

                if min_cost != float('inf'):
                    total_cost += min_cost

        return total_cost
