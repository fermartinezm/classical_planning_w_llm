from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class rovers18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all communicated data goals. It considers the steps needed for each rover to collect samples, take images, and communicate the data.

    # Assumptions
    - Each rover can handle multiple tasks, but the heuristic assigns each goal to the most efficient rover.
    - Navigate actions between waypoints are assumed to take one step if directly traversable.
    - Communication requires the rover to be at a waypoint visible from the lander's location.

    # Heuristic Initialization
    - Extract lander positions, rover capabilities, camera details, and traversal information from static facts.
    - Precompute visibility and calibration targets for objectives.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unachieved goal, determine the type (soil, rock, image).
    2. Check if the data is already collected but not communicated.
    3. For soil/rock data:
        a. If collected, add navigate and communicate steps.
        b. If not, add steps to navigate, sample, navigate, and communicate.
    4. For image data:
        a. If image is taken, add navigate and communicate steps.
        b. If not, add steps to calibrate (if needed), take image, navigate, and communicate.
    5. Sum the minimal steps for all goals, considering rover capabilities and current state.
    """

    def __init__(self, task):
        self.static = task.static
        self.goals = task.goals

        # Extract lander positions
        self.lander_positions = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'at_lander':
                self.lander_positions[parts[1]] = parts[2]

        # Extract rover capabilities and stores
        self.rovers = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']:
                rover = parts[1]
                if rover not in self.rovers:
                    self.rovers[rover] = {'soil': False, 'rock': False, 'imaging': False, 'stores': [], 'cameras': []}
                if parts[0] == 'equipped_for_soil_analysis':
                    self.rovers[rover]['soil'] = True
                elif parts[0] == 'equipped_for_rock_analysis':
                    self.rovers[rover]['rock'] = True
                elif parts[0] == 'equipped_for_imaging':
                    self.rovers[rover]['imaging'] = True
            elif parts[0] == 'store_of':
                rover = parts[2]
                if rover not in self.rovers:
                    self.rovers[rover] = {'soil': False, 'rock': False, 'imaging': False, 'stores': [], 'cameras': []}
                self.rovers[rover]['stores'].append(parts[1])
            elif parts[0] == 'on_board':
                rover = parts[2]
                if rover not in self.rovers:
                    self.rovers[rover] = {'soil': False, 'rock': False, 'imaging': False, 'stores': [], 'cameras': []}
                self.rovers[rover]['cameras'].append(parts[1])

        # Extract camera calibration and modes
        self.calibration_targets = {}
        self.supported_modes = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'calibration_target':
                self.calibration_targets[parts[1]] = parts[2]
            elif parts[0] == 'supports':
                if parts[1] not in self.supported_modes:
                    self.supported_modes[parts[1]] = set()
                self.supported_modes[parts[1]].add(parts[2])

        # Extract visible_from and can_traverse
        self.visible_from = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'visible_from':
                obj = parts[1]
                if obj not in self.visible_from:
                    self.visible_from[obj] = set()
                self.visible_from[obj].add(parts[2])

        self.can_traverse = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'can_traverse':
                rover = parts[1]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = set()
                self.can_traverse[rover].add((parts[2], parts[3]))
                self.can_traverse[rover].add((parts[3], parts[2]))

        # Landers' visible waypoints
        self.lander_visible = {}
        for lander, wp in self.lander_positions.items():
            self.lander_visible[lander] = set()
            for fact in self.static:
                parts = fact[1:-1].split()
                if parts[0] == 'visible' and parts[2] == wp:
                    self.lander_visible[lander].add(parts[1])

    def __call__(self, node):
        state = node.state
        total = 0

        for goal in self.goals:
            if goal in state:
                continue
            parts = goal[1:-1].split()
            if parts[0] == 'communicated_soil_data':
                total += self._soil_rock_cost(parts[1], 'soil', state)
            elif parts[0] == 'communicated_rock_data':
                total += self._soil_rock_cost(parts[1], 'rock', state)
            elif parts[0] == 'communicated_image_data':
                total += self._image_cost(parts[1], parts[2], state)

        return total

    def _soil_rock_cost(self, wp, type_, state):
        # Check if already communicated
        if f'(communicated_{type_}_data {wp})' in state:
            return 0

        # Check if any rover has the analysis
        for rover in self.rovers:
            if f'(have_{type_}_analysis {rover} {wp})' in state:
                lander_wp = next(iter(self.lander_positions.values()))
                comm_wps = self.lander_visible.get(next(iter(self.lander_positions)), set())
                current_wp = self._get_rover_position(rover, state)
                if current_wp in comm_wps:
                    return 1  # Communicate
                else:
                    return 2  # Navigate + communicate

        # Find minimal cost to collect and communicate
        min_cost = float('inf')
        for rover, data in self.rovers.items():
            if not data[type_]:
                continue
            if not data['stores']:
                continue
            store = data['stores'][0]
            current_wp = self._get_rover_position(rover, state)
            if not current_wp:
                continue

            # Check store status
            store_status = 'full' if f'(full {store})' in state else 'empty'
            drop_cost = 1 if store_status == 'full' else 0

            # Check traversal to wp
            if (current_wp, wp) not in self.can_traverse.get(rover, set()):
                continue

            # Cost to collect and communicate
            collect_cost = drop_cost + 1 + 1  # navigate to wp, sample
            comm_wps = self.lander_visible.get(next(iter(self.lander_positions)), set())
            comm_cost = 1 if current_wp in comm_wps else 2  # navigate + communicate
            total = collect_cost + comm_cost
            if total < min_cost:
                min_cost = total

        return min_cost if min_cost != float('inf') else 0

    def _image_cost(self, obj, mode, state):
        if f'(communicated_image_data {obj} {mode})' in state:
            return 0

        # Check if any rover has the image
        for rover in self.rovers:
            if f'(have_image {rover} {obj} {mode})' in state:
                current_wp = self._get_rover_position(rover, state)
                comm_wps = self.lander_visible.get(next(iter(self.lander_positions)), set())
                if current_wp in comm_wps:
                    return 1  # Communicate
                else:
                    return 2  # Navigate + communicate

        # Find minimal cost to take image and communicate
        min_cost = float('inf')
        for rover, data in self.rovers.items():
            if not data['imaging']:
                continue
            for camera in data['cameras']:
                if mode not in self.supported_modes.get(camera, set()):
                    continue
                cal_obj = self.calibration_targets.get(camera)
                if not cal_obj:
                    continue

                # Calibration cost
                calibrated = f'(calibrated {camera} {rover})' in state
                cal_wps = self.visible_from.get(cal_obj, set())
                current_wp = self._get_rover_position(rover, state)
                if not current_wp:
                    continue

                cal_cost = 0
                if not calibrated:
                    if current_wp in cal_wps:
                        cal_cost = 1  # Calibrate
                    else:
                        cal_cost = 2  # Navigate + calibrate

                # Take image cost
                img_wps = self.visible_from.get(obj, set())
                if not img_wps:
                    continue
                img_cost = 2  # Assume navigate to img_wp + take image

                # Communicate cost
                comm_cost = 2  # navigate to comm + communicate

                total = cal_cost + img_cost + comm_cost
                if total < min_cost:
                    min_cost = total

        return min_cost if min_cost != float('inf') else 0

    def _get_rover_position(self, rover, state):
        for fact in state:
            if fact.startswith(f'(at {rover} '):
                return fact.split()[2][:-1]
        return None
