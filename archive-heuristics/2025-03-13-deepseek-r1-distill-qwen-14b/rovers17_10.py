from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goal state by considering:
    - Communicating soil, rock, and image data.
    - Navigating to required waypoints.
    - Collecting samples and taking images.
    - Calibrating cameras when necessary.

    # Assumptions:
    - The rover must collect samples before communicating soil or rock data.
    - The rover must calibrate its camera before taking images.
    - Navigation between waypoints is required if the rover is not already at the needed location.

    # Heuristic Initialization
    - Extracts goal conditions and static facts from the task.
    - Maps waypoints to their locations and identifies calibration targets.

    # Step-by-Step Thinking for Computing Heuristic
    1. Extract the number of each type of data (soil, rock, image) that needs to be communicated.
    2. For each soil or rock data:
       a. Check if the sample is already collected and communicated.
       b. If not, add actions for navigating to the waypoint, collecting the sample, and communicating the data.
    3. For each image data:
       a. Check if the image is already taken and communicated.
       b. If not, add actions for navigating to the waypoint, calibrating the camera, taking the image, and communicating the data.
    4. Sum all the required actions to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each data type
        self.goal_communications = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == "communicated_soil_data":
                waypoint = parts[1]
                self.goal_communications[(waypoint, "soil")] = True
            elif parts[0] == "communicated_rock_data":
                waypoint = parts[1]
                self.goal_communications[(waypoint, "rock")] = True
            elif parts[0] == "communicated_image_data":
                obj, mode = parts[1], parts[2]
                self.goal_communications[(obj, mode, "image")] = True

        # Extract static facts about waypoints and calibration targets
        self.waypoint_locations = {}
        self.calibration_targets = {}
        for fact in self.static:
            if fact.startswith("(at_soil_sample ") or fact.startswith("(at_rock_sample "):
                waypoint = fact.split()[1]
                self.waypoint_locations[waypoint] = waypoint
            elif fact.startswith("(calibration_target "):
                camera, obj = fact.split()[1], fact.split()[2]
                self.calibration_targets[obj] = camera

    def __call__(self, node):
        """Compute the estimated number of actions needed to reach the goal."""
        state = node.state
        current_facts = set(state)

        def get_parts(fact):
            return fact[1:-1].split()

        def match(fact, *args):
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        heuristic = 0

        # Check for each type of goal communication
        for (waypoint, data_type), _ in self.goal_communications.items():
            if match(f"(communicated_{data_type}_data {waypoint})", f"*"):
                continue  # Already communicated

            if data_type in ["soil", "rock"]:
                # Check if the sample is already collected
                has_sample = False
                for fact in current_facts:
                    if (fact.startswith("(have_soil_analysis rover1 ") and waypoint in fact) or \
                       (fact.startswith("(have_rock_analysis rover1 ") and waypoint in fact):
                        has_sample = True
                        break
                if not has_sample:
                    # Need to navigate to waypoint, collect sample, then communicate
                    heuristic += 2  # Navigate and collect
                # Communicate the data
                heuristic += 1
            else:
                # Image data
                obj, mode = waypoint.split("-") if "-" in waypoint else (waypoint, None)
                # Check if image is already taken
                has_image = False
                for fact in current_facts:
                    if match(f"(have_image rover1 {obj} {mode})", "*"):
                        has_image = True
                        break
                if not has_image:
                    # Need to navigate to waypoint, calibrate, take image, then communicate
                    heuristic += 3  # Navigate, calibrate, take image
                else:
                    heuristic += 1  # Only communicate

        return heuristic
