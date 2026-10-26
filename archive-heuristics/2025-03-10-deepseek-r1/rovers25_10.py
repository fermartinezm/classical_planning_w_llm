from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def parse_fact(fact):
    return fact.strip('()').split()

class rovers25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goals by considering the steps needed for each unachieved goal. It accounts for soil/rock sampling, image capturing, calibration, and communication actions, assuming optimal rover assignments and minimal navigation steps.

    # Assumptions
    - Each rover can be assigned optimally to minimize steps.
    - Navigation between waypoints requires one action per move.
    - Calibration and communication actions are performed at the earliest possible step.
    - If a rover's store is full, it must drop the sample before collecting a new one.

    # Heuristic Initialization
    - Extracts static information such as lander positions, calibration targets, camera supports, and rover capabilities.
    - Preprocesses static facts to build data structures for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Goal Check**: For each goal, check if it's already achieved in the current state.
    2. **Soil/Rock Data**:
        - If unachieved, check if any rover has the sample. If so, compute communication steps.
        - If no rover has the sample, compute steps for sampling (including store management and navigation).
    3. **Image Data**:
        - If unachieved, check if any rover has the image. If so, compute communication steps.
        - If no rover has the image, compute steps for calibration, image capture, and communication.
    4. **Sum Costs**: Accumulate the minimal steps for all unachieved goals to form the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Static data structures
        self.lander_positions = {}  # {lander: waypoint}
        self.calibration_targets = {}  # {camera: objective}
        self.on_board = {}  # {camera: rover}
        self.supports = {}  # {camera: set(mode)}
        self.visible_from = {}  # {objective: set(waypoint)}
        self.store_of = {}  # {store: rover}
        self.can_traverse = {}  # {rover: set((from, to))}
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()

        for fact in self.static:
            parts = parse_fact(fact)
            if parts[0] == 'at_lander':
                self.lander_positions[parts[1]] = parts[2]
            elif parts[0] == 'calibration_target':
                self.calibration_targets[parts[1]] = parts[2]
            elif parts[0] == 'on_board':
                self.on_board[parts[1]] = parts[2]
            elif parts[0] == 'supports':
                cam, mode = parts[1], parts[2]
                if cam not in self.supports:
                    self.supports[cam] = set()
                self.supports[cam].add(mode)
            elif parts[0] == 'visible_from':
                obj, wp = parts[1], parts[2]
                if obj not in self.visible_from:
                    self.visible_from[obj] = set()
                self.visible_from[obj].add(wp)
            elif parts[0] == 'store_of':
                self.store_of[parts[1]] = parts[2]
            elif parts[0] == 'can_traverse':
                rover, from_wp, to_wp = parts[1], parts[2], parts[3]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = set()
                self.can_traverse[rover].add((from_wp, to_wp))
            elif parts[0] == 'equipped_for_soil_analysis':
                self.equipped_for_soil.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_for_rock.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_for_imaging.add(parts[1])

        self.rovers = set(self.store_of.values())
        self.store_to_rover = {s: r for r, s in self.store_of.items()}

    def __call__(self, node):
        state = node.state
        total = 0
        for goal in self.goals:
            if goal in state:
                continue
            parts = parse_fact(goal)
            if parts[0] == 'communicated_soil_data':
                total += self._soil_cost(parts[1], state)
            elif parts[0] == 'communicated_rock_data':
                total += self._rock_cost(parts[1], state)
            elif parts[0] == 'communicated_image_data':
                total += self._image_cost(parts[1], parts[2], state)
        return total

    def _soil_cost(self, wp, state):
        if f'(communicated_soil_data {wp})' in state:
            return 0
        min_cost = float('inf')
        lander_wp = next(iter(self.lander_positions.values()), None)
        if not lander_wp:
            return 0

        # Check if any rover already has the sample
        for rover in self.rovers:
            if f'(have_soil_analysis {rover} {wp})' in state:
                pos = self._rover_position(rover, state)
                cost = 1 if self._is_visible(pos, lander_wp) else 2
                min_cost = min(min_cost, cost)
                break
        if min_cost != float('inf'):
            return min_cost

        # No rover has the sample; find best rover to collect it
        for rover in self.rovers:
            if rover not in self.equipped_for_soil:
                continue
            store = self.store_to_rover.get(rover)
            if not store:
                continue
            cost = 0
            if f'(full {store})' in state:
                cost += 1  # Drop action
            rover_pos = self._rover_position(rover, state)
            if rover_pos != wp:
                cost += 1  # Navigate to wp
            cost += 1  # Sample
            if self._is_visible(wp, lander_wp):
                cost += 1  # Communicate
            else:
                cost += 2  # Navigate + communicate
            min_cost = min(min_cost, cost)
        return min_cost if min_cost != float('inf') else 0

    def _rock_cost(self, wp, state):
        if f'(communicated_rock_data {wp})' in state:
            return 0
        min_cost = float('inf')
        lander_wp = next(iter(self.lander_positions.values()), None)
        if not lander_wp:
            return 0

        for rover in self.rovers:
            if f'(have_rock_analysis {rover} {wp})' in state:
                pos = self._rover_position(rover, state)
                cost = 1 if self._is_visible(pos, lander_wp) else 2
                min_cost = min(min_cost, cost)
                break
        if min_cost != float('inf'):
            return min_cost

        for rover in self.rovers:
            if rover not in self.equipped_for_rock:
                continue
            store = self.store_to_rover.get(rover)
            if not store:
                continue
            cost = 0
            if f'(full {store})' in state:
                cost += 1  # Drop
            rover_pos = self._rover_position(rover, state)
            if rover_pos != wp:
                cost += 1  # Navigate
            cost += 1  # Sample
            if self._is_visible(wp, lander_wp):
                cost += 1  # Communicate
            else:
                cost += 2
            min_cost = min(min_cost, cost)
        return min_cost if min_cost != float('inf') else 0

    def _image_cost(self, obj, mode, state):
        if f'(communicated_image_data {obj} {mode})' in state:
            return 0
        min_cost = float('inf')
        lander_wp = next(iter(self.lander_positions.values()), None)
        if not lander_wp:
            return 0

        # Check existing images
        for rover in self.rovers:
            if f'(have_image {rover} {obj} {mode})' in state:
                pos = self._rover_position(rover, state)
                cost = 1 if self._is_visible(pos, lander_wp) else 2
                min_cost = min(min_cost, cost)
                break
        if min_cost != float('inf'):
            return min_cost

        # Find suitable cameras
        for cam in self.supports:
            if mode not in self.supports.get(cam, set()):
                continue
            rover = self.on_board.get(cam)
            if not rover or rover not in self.equipped_for_imaging:
                continue
            cal_target = self.calibration_targets.get(cam)
            if not cal_target:
                continue
            cal_wps = self.visible_from.get(cal_target, set())
            if not cal_wps:
                continue
            cost = 0
            # Calibration needed?
            if f'(calibrated {cam} {rover})' not in state:
                rover_pos = self._rover_position(rover, state)
                if rover_pos not in cal_wps:
                    cost += 1  # Navigate to cal wp
                cost += 1  # Calibrate
            # Take image
            obj_wps = self.visible_from.get(obj, set())
            if not obj_wps:
                continue
            cost += 1  # Navigate to obj wp (assumed needed)
            cost += 1  # Take image
            # Communicate
            if any(self._is_visible(wp, lander_wp) for wp in obj_wps):
                cost += 1
            else:
                cost += 2
            min_cost = min(min_cost, cost)
        return min_cost if min_cost != float('inf') else 0

    def _rover_position(self, rover, state):
        for fact in state:
            parts = parse_fact(fact)
            if parts[0] == 'at' and parts[1] == rover:
                return parts[2]
        return None

    def _is_visible(self, from_wp, to_wp):
        for fact in self.static:
            parts = parse_fact(fact)
            if parts[0] == 'visible' and parts[1] == from_wp and parts[2] == to_wp:
                return True
        return False
