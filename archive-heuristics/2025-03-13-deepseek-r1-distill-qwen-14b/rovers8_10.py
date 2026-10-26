from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state, which involves communicating soil, rock, and image data. The heuristic considers the current state of data communication, sample collection, and rover positions to compute the minimum steps required.

    # Assumptions:
    - Each rover can perform a limited set of actions, including navigation, sampling, calibration, and communication.
    - Communication actions require the rover to be at a specific waypoint and have the necessary data collected.
    - Calibration is required before taking images, and it depends on the rover's position relative to the calibration target.

    # Heuristic Initialization
    - Extract goal conditions for each type of data (soil, rock, image).
    - Extract static facts including navigation capabilities, visibility between waypoints, and calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Communicated Data Check**: For each goal, if the data is already communicated, no action is needed. Otherwise, proceed to estimate the required steps.
    2. **Image Communication**: If an image hasn't been communicated, check if it has been taken and if the rover is at the correct location to communicate it. If not, estimate the steps to move the rover to the communication location and send the data.
    3. **Sample Communication**: For soil and rock samples, check if they have been analyzed and communicated. If not, estimate the steps to collect the sample, analyze it, and then communicate the data.
    4. **Image Taking**: If an image hasn't been taken, check if the camera is calibrated. If not, estimate the steps to calibrate the camera and then take the image.
    5. **Calibration**: If calibration is needed, estimate the steps to move the rover to the calibration target waypoint and perform the calibration.
    6. **Navigation**: If the rover isn't at the required waypoint, estimate the steps to navigate there, considering the shortest path based on static facts.

    The heuristic assumes that the shortest path between waypoints is used for navigation and that each action (navigation, calibration, communication) has a fixed cost.
    """

    def __init__(self, task):
        """Initialize the heuristic with task-specific information."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.can_traverse = set()
        self.visible = set()
        self.calibration_targets = dict()
        self.visible_from = dict()

        for fact in self.static:
            if fact.startswith('(can_traverse'):
                parts = fact[1:-1].split()
                rover, from_wpt, to_wpt = parts[1], parts[2], parts[3]
                self.can_traverse.add((rover, from_wpt, to_wpt))

            elif fact.startswith('(visible'):
                parts = fact[1:-1].split()
                if len(parts) == 3:
                    from_wpt, to_wpt = parts[1], parts[2]
                    self.visible.add((from_wpt, to_wpt))
                    self.visible.add((to_wpt, from_wpt))
                elif len(parts) == 4:
                    obj, from_wpt, to_wpt = parts[1], parts[2], parts[3]
                    if obj == 'objective':
                        self.visible_from.setdefault(obj, set()).add((from_wpt, to_wpt))

            elif fact.startswith('(calibration_target'):
                parts = fact[1:-1].split()
                camera, obj = parts[1], parts[2]
                self.calibration_targets[camera] = obj

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        static = node.task.static

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract current state information
        current_communicated = set()
        current_have_image = set()
        current_have_analysis = dict()
        current_at = dict()
        current_calibrated = set()
        current_samples = dict()

        for fact in state:
            if match(fact, 'communicated_rock_data', '*'):
                current_communicated.add('rock')
            elif match(fact, 'communicated_soil_data', '*'):
                current_communicated.add('soil')
            elif match(fact, 'communicated_image_data', '*', '*'):
                current_communicated.add('image')
            elif match(fact, 'have_image', '*', '*', '*'):
                rover, obj, mode = get_parts(fact)[1:]
                current_have_image[(rover, obj, mode)] = True
            elif match(fact, 'have_rock_analysis', '*', '*'):
                rover, wpt = get_parts(fact)[1:]
                current_samples.setdefault(rover, dict())['rock'] = wpt
            elif match(fact, 'have_soil_analysis', '*', '*'):
                rover, wpt = get_parts(fact)[1:]
                current_samples.setdefault(rover, dict())['soil'] = wpt
            elif match(fact, 'calibrated', '*', '*'):
                current_calibrated.add(fact)
            elif match(fact, 'at', '*', '*'):
                rover, wpt = get_parts(fact)[1:]
                current_at[rover] = wpt

        # Initialize heuristic cost
        total_cost = 0

        # Check each goal type
        for goal in self.goals:
            if match(goal, 'communicated_image_data', '*', '*'):
                obj, mode = get_parts(goal)[1:]
                key = (obj, mode)
                if key not in current_communicated:
                    # Need to communicate this image
                    total_cost += 1  # Communication action

                    # Check if image is already taken
                    if (f"have_image rover{obj} {obj} {mode}") not in state:
                        total_cost += 1  # Take image action

                        # Check if camera is calibrated
                        if not any(fnmatch(fact, f"calibrated * rover{obj} *") for fact in state):
                            # Need to calibrate
                            total_cost += 1  # Calibration action

                            # Find calibration target
                            for fact in static:
                                if match(fact, 'calibration_target', '*', obj):
                                    camera = get_parts(fact)[1]
                                    for cal_fact in static:
                                        if match(cal_fact, 'on_board', camera, f"rover{obj}"):
                                            # Rover is at current position, need to navigate to calibration target
                                            current_pos = next(wpt for rover, wpt in current_at.items() if rover == f"rover{obj}")
                                            target_wpt = self.calibration_targets.get(camera, None)
                                            if target_wpt and current_pos != target_wpt:
                                                # Navigation steps: assume shortest path
                                                total_cost += 2  # Move to target and calibrate
                                            break
                            else:
                                # No calibration target found (shouldn't happen)
                                total_cost += 100  # High cost penalty
            elif match(goal, 'communicated_soil_data', '*'):
                wpt = get_parts(goal)[1]
                if f"communicated_soil_data {wpt}" not in current_communicated:
                    # Need to communicate soil data
                    total_cost += 1

                    # Check if soil analysis is done
                    if not any(fnmatch(fact, f"have_soil_analysis rover* {wpt}") for fact in state):
                        # Need to analyze soil
                        total_cost += 1

                        # Check if soil sample is collected
                        has_sample = any(fnmatch(fact, f"have_soil_analysis rover* {wpt}") for fact in state)
                        if not has_sample:
                            # Need to collect soil sample
                            total_cost += 1

                            # Check if rover is at the waypoint
                            required_rover = f"rover{wpt[-1]}"  # Simplified assumption
                            if current_at.get(required_rover, None) != wpt:
                                # Navigation steps: assume shortest path
                                total_cost += 2  # Move to waypoint and sample
            elif match(goal, 'communicated_rock_data', '*'):
                wpt = get_parts(goal)[1]
                if f"communicated_rock_data {wpt}" not in current_communicated:
                    # Need to communicate rock data
                    total_cost += 1

                    # Check if rock analysis is done
                    if not any(fnmatch(fact, f"have_rock_analysis rover* {wpt}") for fact in state):
                        # Need to analyze rock
                        total_cost += 1

                        # Check if rock sample is collected
                        has_sample = any(fnmatch(fact, f"have_rock_analysis rover* {wpt}") for fact in state)
                        if not has_sample:
                            # Need to collect rock sample
                            total_cost += 1

                            # Check if rover is at the waypoint
                            required_rover = f"rover{wpt[-1]}"  # Simplified assumption
                            if current_at.get(required_rover, None) != wpt:
                                # Navigation steps: assume shortest path
                                total_cost += 2  # Move to waypoint and sample

        return total_cost
