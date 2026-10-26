from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed for each rover to achieve the goal conditions, which include communicating soil, rock, and image data from specific waypoints to the lander.

    # Assumptions:
    - Each rover can independently handle different data collection and communication tasks.
    - The rover may need to navigate to the required waypoint, collect the sample or image, and then communicate the data.
    - Calibration of the camera is required before taking images.
    - Communication actions require the rover to be at a waypoint visible to the lander.

    # Heuristic Initialization
    - Extract the goal conditions for each waypoint data (soil, rock, image).
    - Store static facts about communication visibility and traversal capabilities.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each type of data (soil, rock, image):
       a. Check if the data has already been communicated.
       b. If not, determine the required actions:
          - Navigation to the waypoint if not already there.
          - Sampling or imaging if not done.
          - Navigation to the lander's visible waypoint if needed.
          - Communication action.
    2. Sum the estimated actions for all required data points.
    3. Return the total estimated actions as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract communication visibility facts
        self.visible_to_lander = set()
        for fact in self.static:
            if fnmatch(fact, '(visible_from * _)'):
                parts = fact[1:-1].split()
                if parts[1] == '_':
                    self.visible_to_lander.add(parts[2])

        # Extract can_traverse relations
        self.can_traverse = {}
        for fact in self.static:
            if fnmatch(fact, '(can_traverse * * *)'):
                rover, from_wpt, to_wpt = fact[1:-1].split()
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                self.can_traverse[rover][from_wpt] = to_wpt

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state
        total_actions = 0

        def get_parts(fact):
            return fact[1:-1].split()

        def match(fact, pattern):
            return fnmatch(fact, pattern)

        # Helper function to check if a fact is in the state
        def has_fact(pattern):
            return any(match(f, pattern) for f in state)

        # Check if all goals are already achieved
        if all(has_fact(f) for f in self.goals):
            return 0

        # Process each type of data communication
        # Soil data
        for fact in self.goals:
            if fnmatch(fact, '(communicated_soil_data _)'):
                wpt = fact[1:-1].split()[1]
                if not has_fact(f'(communicated_soil_data {wpt})'):
                    # Find which rover can handle soil analysis
                    soil_rover = None
                    for obj in state:
                        if fnmatch(obj, '(equipped_for_soil_analysis *)'):
                            soil_rover = obj[1:-1].split()[1]
                            break
                    if soil_rover:
                        # Check if rover is at the waypoint
                        at_fact = f'(at {soil_rover} {wpt})'
                        if not has_fact(at_fact):
                            # Navigation needed
                            total_actions += 1
                        # Check if sample is taken
                        has_sample = f'(have_soil_analysis {soil_rover} {wpt})'
                        if not has_fact(has_sample):
                            total_actions += 1
                        # Check if at lander's visible waypoint
                        at_lander_visible = any(f'({wpt} == {v})' in f for f in self.visible_to_lander)
                        if not at_lander_visible:
                            total_actions += 1
                        # Communication action
                        total_actions += 1

        # Rock data
        for fact in self.goals:
            if fnmatch(fact, '(communicated_rock_data _)'):
                wpt = fact[1:-1].split()[1]
                if not has_fact(f'(communicated_rock_data {wpt})'):
                    # Find which rover can handle rock analysis
                    rock_rover = None
                    for obj in state:
                        if fnmatch(obj, '(equipped_for_rock_analysis *)'):
                            rock_rover = obj[1:-1].split()[1]
                            break
                    if rock_rover:
                        # Check if rover is at the waypoint
                        at_fact = f'(at {rock_rover} {wpt})'
                        if not has_fact(at_fact):
                            total_actions += 1
                        # Check if sample is taken
                        has_sample = f'(have_rock_analysis {rock_rover} {wpt})'
                        if not has_fact(has_sample):
                            total_actions += 1
                        # Check if at lander's visible waypoint
                        at_lander_visible = any(f'({wpt} == {v})' in f for v in self.visible_to_lander)
                        if not at_lander_visible:
                            total_actions += 1
                        # Communication action
                        total_actions += 1

        # Image data
        for fact in self.goals:
            if fnmatch(fact, '(communicated_image_data * * *)'):
                obj, mode = fact[1:-1].split()[1:3]
                if not has_fact(f'(communicated_image_data {obj} {mode})'):
                    # Find which rover has the camera equipped
                    img_rover = None
                    for obj in state:
                        if fnmatch(obj, '(equipped_for_imaging *)'):
                            img_rover = obj[1:-1].split()[1]
                            break
                    if img_rover:
                        # Check if camera is calibrated
                        calibrated = f'(calibrated * {img_rover})'
                        if not has_fact(calibrated):
                            total_actions += 1
                        # Take image if not done
                        has_image = f'(have_image {img_rover} {obj} {mode})'
                        if not has_fact(has_image):
                            total_actions += 1
                        # Check if at lander's visible waypoint
                        at_wpt = None
                        for wpt in self.visible_to_lander:
                            if f'(visible_from {obj} {wpt})' in state:
                                at_wpt = wpt
                                break
                        if at_wpt:
                            at_fact = f'(at {img_rover} {at_wpt})'
                            if not has_fact(at_fact):
                                total_actions += 1
                        else:
                            # Find a waypoint visible from the objective
                            visible_wpt = None
                            for wpt in self.visible_to_lander:
                                if f'(visible_from {obj} {wpt})' in state:
                                    visible_wpt = wpt
                                    break
                            if visible_wpt:
                                # Navigation needed
                                total_actions += 1
                        # Communication action
                        total_actions += 1

        return total_actions
