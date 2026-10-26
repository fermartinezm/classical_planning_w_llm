from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state, which involves communicating soil, rock, and image data. The heuristic considers the current state of each rover, the locations of samples and objectives, and the required actions to complete the communication of data.

    # Assumptions:
    - Each rover can perform soil and rock sampling, image capture, and communication.
    - Communication requires visibility between the rover's location and the lander.
    - Image capture requires calibration and may involve multiple modes.
    - The goal is achieved when all required data (soil, rock, images) is communicated.

    # Heuristic Initialization
    - Extract goal conditions (communicated data points).
    - Extract static facts (navigation capabilities, visibility, calibration targets).

    # Step-by-Step Thinking for Computing Heuristic
    1. Check if all goal conditions are already met. If so, return 0.
    2. For each type of goal (soil, rock, image data):
       a. Determine if the data has been collected and communicated.
       b. If not communicated, estimate the actions needed to communicate it.
    3. For each rover, calculate the actions needed to reach the required waypoints for sampling or imaging.
    4. Sum the estimated actions for all incomplete goal conditions.
    """

    def __init__(self, task):
        """Initialize the heuristic with task-specific information."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information into useful data structures
        self.can_traverse = set()
        self.visible = set()
        self calibration_targets = set()
        self.visible_from = set()
        self.communication_cost = {}

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
                    self.visible_from.add((obj, from_wpt, to_wpt))

            elif fact.startswith('(calibration_target'):
                parts = fact[1:-1].split()
                camera, obj = parts[1], parts[2]
                self.calibration_targets.add((camera, obj))

    def __call__(self, node):
        """Estimate the minimal number of actions to achieve the goal state."""
        state = node.state
        goal_achieved = True
        total_cost = 0

        # Check if all goals are already achieved
        for goal in self.goals:
            if goal not in state:
                goal_achieved = False
                break
        if goal_achieved:
            return 0

        # Extract relevant information from the state
        rover_locations = {}
        sample_taken = {'soil': set(), 'rock': set()}
        image_taken = set()
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()

        for fact in state:
            if fact.startswith('(at'):
                parts = fact[1:-1].split()
                rover, wpt = parts[1], parts[2]
                rover_locations[rover] = wpt

            elif fact.startswith('(have_soil_analysis'):
                parts = fact[1:-1].split()
                rover, wpt = parts[1], parts[2]
                sample_taken['soil'].add(wpt)

            elif fact.startswith('(have_rock_analysis'):
                parts = fact[1:-1].split()
                rover, wpt = parts[1], parts[2]
                sample_taken['rock'].add(wpt)

            elif fact.startswith('(have_image'):
                parts = fact[1:-1].split()
                rover, obj, mode = parts[1], parts[2], parts[3]
                image_taken.add((rover, obj, mode))

            elif fact.startswith('(communicated_soil_data'):
                parts = fact[1:-1].split()
                wpt = parts[1]
                communicated_soil.add(wpt)

            elif fact.startswith('(communicated_rock_data'):
                parts = fact[1:-1].split()
                wpt = parts[1]
                communicated_rock.add(wpt)

            elif fact.startswith('(communicated_image_data'):
                parts = fact[1:-1].split()
                obj, mode = parts[1], parts[2]
                communicated_image.add((obj, mode))

        # Calculate required actions for each type of goal
        # Soil and Rock Communication
        for wpt in sample_taken['soil']:
            if wpt not in communicated_soil:
                # Need to communicate soil data
                total_cost += 2  # Navigate to lander's waypoint and communicate

        for wpt in sample_taken['rock']:
            if wpt not in communicated_rock:
                # Need to communicate rock data
                total_cost += 2  # Navigate to lander's waypoint and communicate

        # Image Communication
        for (rover, obj, mode) in image_taken:
            if (obj, mode) not in communicated_image:
                # Need to communicate image data
                total_cost += 2  # Navigate to lander's waypoint and communicate

        return total_cost
