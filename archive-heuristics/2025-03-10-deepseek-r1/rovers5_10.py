from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern_parts))

class Rovers5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all communicated goals by considering the steps needed for each unachieved goal, including navigation, sampling, imaging, and communication.

    # Assumptions
    - Each rover can carry one sample at a time (soil or rock) and must empty its store before collecting a new sample.
    - Cameras need calibration once per use, and calibration requires being at a waypoint visible from the camera's target.
    - Communication requires the rover to be at a waypoint visible from the lander's location.

    # Heuristic Initialization
    - Extract static information such as lander location, camera calibration targets, supported modes, and waypoint visibility.
    - Precompute mappings for quick access during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unachieved goal (soil, rock, image data):
        a. If a rover already has the required sample/image:
            - Estimate the cost to navigate to a communication point and transmit.
        b. If the sample is available but not collected:
            - Estimate the cost for a rover to collect the sample and communicate.
        c. For images not yet taken:
            - Estimate calibration, navigation to imaging location, capturing, and communication.
    2. Sum the minimal costs for all goals, assuming optimal rover assignment.
    """

    def __init__(self, task):
        self.task = task
        self.static = task.static
        self.goals = task.goals

        # Extract lander's waypoint
        self.lander_location = None
        for fact in self.static:
            if match(fact, 'at_lander * *'):
                parts = get_parts(fact)
                self.lander_location = parts[2]
                break

        # Extract calibration targets and camera info
        self.calibration_targets = {}
        self.on_board = defaultdict(list)
        self.supports = defaultdict(set)
        for fact in self.static:
            if match(fact, 'calibration_target * *'):
                parts = get_parts(fact)
                self.calibration_targets[parts[1]] = parts[2]
            elif match(fact, 'on_board * *'):
                parts = get_parts(fact)
                self.on_board[parts[2]].append(parts[1])
            elif match(fact, 'supports * *'):
                parts = get_parts(fact)
                self.supports[parts[1]].add(parts[2])

        # Extract visible_from for objectives
        self.visible_from = defaultdict(set)
        for fact in self.static:
            if match(fact, 'visible_from * *'):
                parts = get_parts(fact)
                self.visible_from[parts[1]].add(parts[2])

        # Extract can_traverse and visible waypoints
        self.can_traverse = defaultdict(set)
        self.visible_waypoints = set()
        for fact in self.static:
            if match(fact, 'can_traverse * * *'):
                parts = get_parts(fact)
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                self.can_traverse[rover].add((from_wp, to_wp))
                self.can_traverse[rover].add((to_wp, from_wp))
            elif match(fact, 'visible * *'):
                parts = get_parts(fact)
                self.visible_waypoints.add((parts[1], parts[2]))
                self.visible_waypoints.add((parts[2], parts[1]))

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Check unachieved goals
        unachieved = [g for g in self.goals if g not in state]

        for goal in unachieved:
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                wp = parts[1]
                total_cost += self.handle_soil(wp, state)
            elif parts[0] == 'communicated_rock_data':
                wp = parts[1]
                total_cost += self.handle_rock(wp, state)
            elif parts[0] == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                total_cost += self.handle_image(obj, mode, state)

        return total_cost

    def handle_soil(self, wp, state):
        # Check if any rover has the analysis
        for fact in state:
            if match(fact, f'have_soil_analysis * {wp}'):
                rover = get_parts(fact)[1]
                cost = self.comm_cost(rover, state)
                return cost  # Assume first rover is sufficient

        # Check if sample is available
        if f'(at_soil_sample {wp})' in state:
            min_cost = float('inf')
            for rover in self.get_equipped_rovers('soil'):
                store = self.get_store(rover)
                if not store:
                    continue
                # Check store state
                steps = 0
                if f'(full {store})' in state:
                    steps += 1  # drop
                # Move to wp
                current_pos = self.get_rover_pos(rover, state)
                if current_pos != wp:
                    steps += 1
                steps += 1  # sample
                steps += self.comm_cost(rover, state, after_sample=True)
                min_cost = min(min_cost, steps)
            return min_cost if min_cost != float('inf') else 0
        else:
            # Sample taken but no rover has it (should not happen in solvable)
            return 0

    def handle_rock(self, wp, state):
        # Similar to handle_soil
        for fact in state:
            if match(fact, f'have_rock_analysis * {wp}'):
                rover = get_parts(fact)[1]
                return self.comm_cost(rover, state)

        if f'(at_rock_sample {wp})' in state:
            min_cost = float('inf')
            for rover in self.get_equipped_rovers('rock'):
                store = self.get_store(rover)
                if not store:
                    continue
                steps = 0
                if f'(full {store})' in state:
                    steps += 1
                current_pos = self.get_rover_pos(rover, state)
                if current_pos != wp:
                    steps += 1
                steps += 1
                steps += self.comm_cost(rover, state, after_sample=True)
                min_cost = min(min_cost, steps)
            return min_cost if min_cost != float('inf') else 0
        else:
            return 0

    def handle_image(self, obj, mode, state):
        # Check if any rover has the image
        for fact in state:
            if match(fact, f'have_image * {obj} {mode}'):
                rover = get_parts(fact)[1]
                return self.comm_cost(rover, state)

        # Find eligible rovers
        min_cost = float('inf')
        for rover in self.on_board:
            for camera in self.on_board[rover]:
                if mode not in self.supports[camera]:
                    continue
                # Check calibration
                calibrated = f'(calibrated {camera} {rover})' in state
                cal_target = self.calibration_targets.get(camera)
                if not calibrated and cal_target:
                    # Need to calibrate
                    cal_wps = self.visible_from.get(cal_target, set())
                    current_pos = self.get_rover_pos(rover, state)
                    if current_pos in cal_wps:
                        cal_cost = 1
                    else:
                        cal_cost = 2  # move + calibrate
                else:
                    cal_cost = 0

                # Move to imaging location
                img_wps = self.visible_from.get(obj, set())
                current_pos = self.get_rover_pos(rover, state)
                if current_pos in img_wps:
                    img_cost = 1  # take_image
                else:
                    img_cost = 2  # move + take_image

                # Communicate cost
                comm_cost = self.comm_cost(rover, state, after_sample=False)

                total = cal_cost + img_cost + comm_cost
                if total < min_cost:
                    min_cost = total
        return min_cost if min_cost != float('inf') else 0

    def comm_cost(self, rover, state, after_sample=False):
        # Cost to move to visible from lander and communicate
        current_pos = self.get_rover_pos(rover, state)
        if not current_pos:
            return float('inf')
        # Find x such that visible(x, lander_location)
        x_candidates = [wp for (wp, lander_wp) in self.visible_waypoints if lander_wp == self.lander_location]
        if current_pos in x_candidates:
            return 1  # communicate
        else:
            # Assume can move to x in 1 step
            return 2  # move + communicate

    def get_equipped_rovers(self, analysis_type):
        rovers = []
        for fact in self.static:
            if analysis_type == 'soil' and match(fact, 'equipped_for_soil_analysis *'):
                rovers.append(get_parts(fact)[1])
            elif analysis_type == 'rock' and match(fact, 'equipped_for_rock_analysis *'):
                rovers.append(get_parts(fact)[1])
        return list(set(rovers))

    def get_store(self, rover):
        for fact in self.static:
            if match(fact, 'store_of * *'):
                parts = get_parts(fact)
                if parts[2] == rover:
                    return parts[1]
        return None

    def get_rover_pos(self, rover, state):
        for fact in state:
            if match(fact, f'at {rover} *'):
                return get_parts(fact)[2]
        return None
