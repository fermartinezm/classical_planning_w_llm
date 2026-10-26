from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, pattern):
    """Check if a fact matches a pattern with wildcards."""
    fact_parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(fact_parts) != len(pattern_parts):
        return False
    return all(fnmatch(fact_part, pattern_part) for fact_part, pattern_part in zip(fact_parts, pattern_parts))

class rovers8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goals by considering:
    - Soil/rock sample collection and communication.
    - Image capture (including camera calibration) and communication.
    - Movement between waypoints, assuming each navigate action takes 1 step.

    # Assumptions
    - Each navigate action between waypoints is approximated as 1 step.
    - Soil/rock samples can be collected if a rover has an empty store and appropriate equipment.
    - Cameras can be calibrated if the rover is at a suitable waypoint.
    - The lander's location is static and known.

    # Heuristic Initialization
    - Extract static information: lander location, rover equipment, stores, cameras, and their properties.
    - Preprocess calibration targets and supported camera modes.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unachieved goal:
        a. **Soil/Rock Data**: Check if collected or needs sampling. Account for store availability.
        b. **Image Data**: Check if taken. If not, compute steps for calibration, image capture, and communication.
    2. Sum the minimal steps for all goals, considering current state and static capabilities.
    """

    def __init__(self, task):
        """Extract static information and initialize data structures."""
        self.task = task
        self.static = task.static
        self.goals = task.goals

        # Landers location (assumes single lander 'general')
        self.lander_wp = None
        for fact in self.static:
            if match(fact, 'at_lander general *'):
                self.lander_wp = get_parts(fact)[2]
                break

        # Rover data: equipment, stores, cameras
        self.rovers = {}
        # Camera data: calibration targets, supported modes
        self.cameras = {}

        # Process static facts to populate rover and camera info
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'equipped_for_soil_analysis':
                rover = parts[1]
                if rover not in self.rovers:
                    self.rovers[rover] = {'soil': True, 'rock': False, 'imaging': False, 'stores': [], 'cameras': []}
                else:
                    self.rovers[rover]['soil'] = True
            elif parts[0] == 'equipped_for_rock_analysis':
                rover = parts[1]
                if rover not in self.rovers:
                    self.rovers[rover] = {'soil': False, 'rock': True, 'imaging': False, 'stores': [], 'cameras': []}
                else:
                    self.rovers[rover]['rock'] = True
            elif parts[0] == 'equipped_for_imaging':
                rover = parts[1]
                if rover not in self.rovers:
                    self.rovers[rover] = {'soil': False, 'rock': False, 'imaging': True, 'stores': [], 'cameras': []}
                else:
                    self.rovers[rover]['imaging'] = True
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                if rover not in self.rovers:
                    self.rovers[rover] = {'soil': False, 'rock': False, 'imaging': False, 'stores': [store], 'cameras': []}
                else:
                    self.rovers[rover]['stores'].append(store)
            elif parts[0] == 'on_board':
                cam, rover = parts[1], parts[2]
                if rover not in self.rovers:
                    self.rovers[rover] = {'soil': False, 'rock': False, 'imaging': False, 'stores': [], 'cameras': [cam]}
                else:
                    self.rovers[rover]['cameras'].append(cam)
            elif parts[0] == 'calibration_target':
                cam, target = parts[1], parts[2]
                if cam not in self.cameras:
                    self.cameras[cam] = {'target': target, 'supports': []}
                else:
                    self.cameras[cam]['target'] = target
            elif parts[0] == 'supports':
                cam, mode = parts[1], parts[2]
                if cam not in self.cameras:
                    self.cameras[cam] = {'target': None, 'supports': [mode]}
                else:
                    self.cameras[cam]['supports'].append(mode)

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        total = 0

        # Check each goal condition
        for goal in self.task.goals:
            g_parts = get_parts(goal)
            if goal in state:
                continue  # Already achieved

            # Communicated soil data
            if g_parts[0] == 'communicated_soil_data':
                wp = g_parts[1]
                # Check if any rover has the analysis
                has_analysis = any(f'(have_soil_analysis {rover} {wp})' in state for rover in self.rovers)
                if has_analysis:
                    total += 2  # navigate + communicate
                else:
                    # Check if sample is present and can be collected
                    if f'(at_soil_sample {wp})' in state:
                        # Check for rover with empty store
                        can_sample = False
                        for rover, data in self.rovers.items():
                            if data['soil'] and any(f'(empty {store})' in state for store in data['stores']):
                                can_sample = True
                                break
                        total += 4 if can_sample else 5  # sample steps or drop + sample
                    else:
                        total += 5  # assume possible but needs handling

            # Communicated rock data (similar to soil)
            elif g_parts[0] == 'communicated_rock_data':
                wp = g_parts[1]
                has_analysis = any(f'(have_rock_analysis {rover} {wp})' in state for rover in self.rovers)
                if has_analysis:
                    total += 2
                else:
                    if f'(at_rock_sample {wp})' in state:
                        can_sample = False
                        for rover, data in self.rovers.items():
                            if data['rock'] and any(f'(empty {store})' in state for store in data['stores']):
                                can_sample = True
                                break
                        total += 4 if can_sample else 5
                    else:
                        total += 5

            # Communicated image data
            elif g_parts[0] == 'communicated_image_data':
                obj, mode = g_parts[1], g_parts[2]
                # Check if any rover has the image
                has_image = any(f'(have_image {rover} {obj} {mode})' in state for rover in self.rovers)
                if has_image:
                    total += 2
                else:
                    min_steps = float('inf')
                    # Find suitable cameras
                    for rover_name, rover_data in self.rovers.items():
                        if not rover_data['imaging']:
                            continue
                        for cam in rover_data['cameras']:
                            if mode not in self.cameras.get(cam, {}).get('supports', []):
                                continue
                            # Check calibration
                            calibrated = f'(calibrated {cam} {rover_name})' in state
                            steps = 4 if calibrated else 6
                            min_steps = min(min_steps, steps)
                    total += min_steps if min_steps != float('inf') else 6  # default if no camera found

        return total
