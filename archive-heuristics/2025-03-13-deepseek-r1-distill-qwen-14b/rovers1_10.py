from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all required communications (soil, rock, and image data) for each objective and waypoint.

    # Assumptions:
    - Each communication requires specific steps: data collection, storage, and transmission.
    - Soil and rock data require sampling and communication via the lander.
    - Image data requires calibration, image capture, and communication.
    - The rover may need to navigate to specific waypoints to collect data.

    # Heuristic Initialization
    - Extract goal conditions for each objective and waypoint.
    - Store static facts about visible waypoints, calibration targets, and camera support modes.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify which communication goals (soil, rock, image) are not yet achieved.
    2. For each unachieved goal:
       a. Determine if the required data has been collected and communicated.
       b. If not, calculate the steps needed to collect the data (navigation, sampling, storing).
       c. Estimate the actions required to communicate the data (navigation, communication action).
    3. Sum the estimated actions for all unachieved goals to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each objective and waypoint
        self.goal_communications = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == "communicated_soil_data":
                waypoint = parts[1]
                self.goal_communications[(waypoint, "soil")] = False
            elif parts[0] == "communicated_rock_data":
                waypoint = parts[1]
                self.goal_communications[(waypoint, "rock")] = False
            elif parts[0] == "communicated_image_data":
                objective = parts[1]
                mode = parts[2]
                self.goal_communications[(objective, mode, "image")] = False

        # Extract static facts
        self.visible_from = {}
        self.calibration_targets = {}
        self.camera_supports = {}
        for fact in self.static:
            if fact.startswith("(visible_from "):
                obj, waypoint = fact[1:-1].split(" ", 2)[1:3]
                if obj not in self.visible_from:
                    self.visible_from[obj] = []
                self.visible_from[obj].append(waypoint)
            elif fact.startswith("(calibration_target "):
                camera, obj = fact[1:-1].split(" ", 2)[1:3]
                if camera not in self.calibration_targets:
                    self.calibration_targets[camera] = []
                self.calibration_targets[camera].append(obj)
            elif fact.startswith("(supports "):
                camera, mode = fact[1:-1].split(" ", 2)[1:3]
                if camera not in self.camera_supports:
                    self.camera_supports[camera] = []
                self.camera_supports[camera].append(mode)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        current_communicated = set()

        # Check which goals are already achieved
        for goal in self.goal_communications:
            if f"({goal[0]} {goal[1]} {goal[2]})" in state:
                self.goal_communications[goal] = True

        total_actions = 0

        # For each communication goal, calculate required actions if not achieved
        for (identifier, data_type, _) in self.goal_communications:
            if not self.goal_communications[(identifier, data_type, "communication")]:
                if data_type in ["soil", "rock"]:
                    # Soil/rock data requires sampling and communication
                    # Assume each requires 2 actions: sample and communicate
                    total_actions += 2
                else:
                    # Image data requires calibration, capture, and communication
                    # Assume each requires 3 actions: calibrate, capture, communicate
                    total_actions += 3

        return total_actions
