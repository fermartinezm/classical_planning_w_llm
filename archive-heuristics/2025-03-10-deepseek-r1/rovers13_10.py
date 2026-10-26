from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class rovers13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Summary:
    This heuristic estimates the number of actions required to achieve all goals by considering the remaining steps for soil/rock sampling, image capture, and data communication. It accounts for rover equipment, store management, camera calibration, and navigation.

    Assumptions:
    - Each rover has a single store for soil/rock samples.
    - The lander's position is static and known from the initial state.
    - Cameras must be calibrated at specific waypoints visible from their objectives.
    - Navigation between waypoints is approximated as a single step for simplicity.

    Heuristic Initialization:
    - Extracts static information such as lander location, visible waypoints, rover equipment, camera details, and objective visibility.
    - Builds mappings for rover stores, calibration targets, supported camera modes, and visible waypoints for objectives.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each uncommunicated soil/rock sample:
       - If the sample is present, find a rover to sample it, considering store status and navigation.
       - If the sample is already collected, compute steps to communicate it.
    2. For each uncommunicated image:
       - If the image is taken, compute steps to communicate it.
       - If not, find a suitable camera, check calibration, and compute steps for calibration, capture, and communication.
    3. Sum the minimal steps for all goals, considering optimal rover and camera usage.
    """

    def __init__(self, task):
        self.goals = task.goals
        static = task.static

        # Extract lander's location
        self.lander_loc = None
        for fact in static:
            if fact.startswith('(at_lander '):
                parts = fact[1:-1].split()
                if parts[1] == 'general':
                    self.lander_loc = parts[2]
                    break

        # Waypoints visible to the lander
        self.visible_waypoints = set()
        for fact in static:
            if fact.startswith('(visible '):
                parts = fact[1:-1].split()
                if parts[2] == self.lander_loc:
                    self.visible_waypoints.add(parts[1])

        # Rover equipment and stores
        self.rover_equipment = {}  # {rover: {'soil', 'rock', 'imaging'}}
        self.rover_stores = {}     # {rover: store}
        for fact in static:
            parts = fact[1:-1].split()
            if parts[0] == 'equipped_for_soil_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {}).update(soil=True)
            elif parts[0] == 'equipped_for_rock_analysis':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {}).update(rock=True)
            elif parts[0] == 'equipped_for_imaging':
                rover = parts[1]
                self.rover_equipment.setdefault(rover, {}).update(imaging=True)
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store

        # Camera information
        self.calibration_targets = {}  # {camera: objective}
        self.supports_modes = {}       # {camera: set(mode)}
        self.on_board = {}             # {camera: rover}
        for fact in static:
            parts = fact[1:-1].split()
            if parts[0] == 'calibration_target':
                camera, obj = parts[1], parts[2]
                self.calibration_targets[camera] = obj
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.supports_modes.setdefault(camera, set()).add(mode)
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.on_board[camera] = rover

        # Visible_from for objectives
        self.visible_from = {}  # {objective: set(waypoint)}
        for fact in static:
            if fact.startswith('(visible_from '):
                parts = fact[1:-1].split()
                obj, wp = parts[1], parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)

    def __call__(self, node):
        state = node.state
        total = 0
        rover_positions = {}
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                rover_positions[parts[1]] = parts[2]

        # Process soil data goals
        for goal in self.goals:
            if not goal.startswith('(communicated_soil_data '):
                continue
            if goal in state:
                continue
            parts = goal[1:-1].split()
            wp = parts[1]
            sample_present = f'(at_soil_sample {wp})' in state
            min_steps = float('inf')
            if sample_present:
                for rover in self.rover_equipment:
                    if not self.rover_equipment[rover].get('soil', False):
                        continue
                    store = self.rover_stores.get(rover)
                    if not store:
                        continue
                    steps = 0
                    if f'(full {store})' in state:
                        steps += 1  # Drop action
                    current_pos = rover_positions.get(rover)
                    if current_pos != wp:
                        steps += 1  # Navigate to wp
                    steps += 1  # Sample
                    if current_pos not in self.visible_waypoints:
                        steps += 1  # Navigate to visible
                    steps += 1  # Communicate
                    min_steps = min(min_steps, steps)
            else:
                for rover in self.rover_equipment:
                    if f'(have_soil_analysis {rover} {wp})' not in state:
                        continue
                    current_pos = rover_positions.get(rover)
                    steps = 0
                    if current_pos not in self.visible_waypoints:
                        steps += 1  # Navigate
                    steps += 1  # Communicate
                    min_steps = min(min_steps, steps)
            if min_steps != float('inf'):
                total += min_steps

        # Process rock data goals
        for goal in self.goals:
            if not goal.startswith('(communicated_rock_data '):
                continue
            if goal in state:
                continue
            parts = goal[1:-1].split()
            wp = parts[1]
            sample_present = f'(at_rock_sample {wp})' in state
            min_steps = float('inf')
            if sample_present:
                for rover in self.rover_equipment:
                    if not self.rover_equipment[rover].get('rock', False):
                        continue
                    store = self.rover_stores.get(rover)
                    if not store:
                        continue
                    steps = 0
                    if f'(full {store})' in state:
                        steps += 1  # Drop action
                    current_pos = rover_positions.get(rover)
                    if current_pos != wp:
                        steps += 1  # Navigate to wp
                    steps += 1  # Sample
                    if current_pos not in self.visible_waypoints:
                        steps += 1  # Navigate to visible
                    steps += 1  # Communicate
                    min_steps = min(min_steps, steps)
            else:
                for rover in self.rover_equipment:
                    if f'(have_rock_analysis {rover} {wp})' not in state:
                        continue
                    current_pos = rover_positions.get(rover)
                    steps = 0
                    if current_pos not in self.visible_waypoints:
                        steps += 1  # Navigate
                    steps += 1  # Communicate
                    min_steps = min(min_steps, steps)
            if min_steps != float('inf'):
                total += min_steps

        # Process image data goals
        for goal in self.goals:
            if not goal.startswith('(communicated_image_data '):
                continue
            if goal in state:
                continue
            parts = goal[1:-1].split()
            obj, mode = parts[1], parts[2]
            min_steps = float('inf')
            # Check if any rover already has the image
            image_found = False
            for rover in self.rover_equipment:
                if f'(have_image {rover} {obj} {mode})' in state:
                    current_pos = rover_positions.get(rover)
                    steps = 0
                    if current_pos not in self.visible_waypoints:
                        steps += 1
                    steps += 1
                    total += steps
                    image_found = True
                    break
            if image_found:
                continue
            # Find suitable cameras
            for camera in self.calibration_targets:
                if self.calibration_targets[camera] != obj:
                    continue
                if mode not in self.supports_modes.get(camera, set()):
                    continue
                rover = self.on_board.get(camera)
                if not rover or not self.rover_equipment.get(rover, {}).get('imaging', False):
                    continue
                current_pos = rover_positions.get(rover)
                calibrated = f'(calibrated {camera} {rover})' in state
                steps = 0
                if calibrated:
                    steps += 1  # Take image
                    if current_pos not in self.visible_waypoints:
                        steps += 1  # Navigate
                    steps += 1  # Communicate
                else:
                    cal_wps = self.visible_from.get(obj, set())
                    if current_pos in cal_wps:
                        steps += 1  # Calibrate
                        steps += 1  # Take image
                        if current_pos not in self.visible_waypoints:
                            steps += 1  # Navigate
                        steps += 1  # Communicate
                    else:
                        steps += 1  # Navigate to cal wp
                        steps += 1  # Calibrate
                        steps += 1  # Take image
                        steps += 1  # Navigate to visible
                        steps += 1  # Communicate
                min_steps = min(min_steps, steps)
            if min_steps != float('inf'):
                total += min_steps

        return total
