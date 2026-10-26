from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to:
    1. Collect soil and rock samples.
    2. Take and communicate images.
    3. Communicate collected data to the lander.

    # Assumptions:
    - The rover starts at the initial waypoint and must navigate to collect samples.
    - Each sample requires specific equipment and actions to collect and communicate.
    - Images require calibration and specific modes to be taken and communicated.
    - Communication actions require visibility between the rover's location and the lander.

    # Heuristic Initialization
    - Extracts goal conditions to identify which data needs communication.
    - Parses static facts to identify calibration targets and other static information.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each type of data (soil, rock, image):
       a. Check if the data is already communicated.
       b. If not, estimate the actions needed to collect/analyze the data and communicate it.
    2. Sum the estimated actions for all required data points.
    3. Return the total as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with task-specific information."""
        self.goals = task.goals
        self.static = task.static

        # Extract calibration targets from static facts
        self.calibration_targets = {}
        for fact in self.static:
            if fnmatch(fact, '(calibration_target * * *)'):
                _, camera, target = fact[1:-1].split()
                self.calibration_targets[camera] = target

    def __call__(self, node):
        """Compute the heuristic value for the given node."""
        state = node.state
        current_actions = 0

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Check if all goal conditions are already met
        if all(goal in state for goal in self.goals):
            return 0

        # Track collected samples and communicated data
        collected_soil = set()
        collected_rock = set()
        communicated_soil = set()
        communicated_rock = set()
        communicated_images = set()

        # Track camera calibration status
        calibrated_cameras = set()

        # Track rover's current location
        current_location = None
        for fact in state:
            if match(fact, 'at', '*', 'waypoint'):
                _, rover, loc = get_parts(fact)
                current_location = loc
                break

        # Process each type of goal
        for goal in self.goals:
            if match(goal, 'communicated_soil_data', '*'):
                sample_waypoint = goal.split()[-1]
                if sample_waypoint in communicated_soil:
                    continue
                # Estimate actions to collect and communicate soil sample
                current_actions += 3  # navigate to sample, collect, communicate
            elif match(goal, 'communicated_rock_data', '*'):
                sample_waypoint = goal.split()[-1]
                if sample_waypoint in communicated_rock:
                    continue
                # Estimate actions to collect and communicate rock sample
                current_actions += 3  # navigate to sample, collect, communicate
            elif match(goal, 'communicated_image_data', '*', '*', '*'):
                obj, mode = goal.split()[-2], goal.split()[-1]
                if f"{obj}_{mode}" in communicated_images:
                    continue
                # Check if camera is calibrated
                camera = None
                for fact in state:
                    if match(fact, 'calibrated', '*', '*'):
                        _, cam, _ = get_parts(fact)
                        camera = cam
                        calibrated_cameras.add(cam)
                        break
                if not camera:
                    # Need to calibrate camera
                    current_actions += 2  # calibrate, then take image
                # Take image and communicate
                current_actions += 2  # take image, communicate

        return current_actions
