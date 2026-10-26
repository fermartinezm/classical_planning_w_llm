from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to communicate all required data points, including soil samples, rock samples, and images in different modes.

    # Assumptions:
    - Each soil sample requires 4 actions: navigate to the sample, collect it, navigate to the lander, and communicate the data.
    - Each rock sample requires 4 actions: similar to soil samples.
    - Each image requires 6 actions: calibrate (if needed), navigate to the image waypoint, take the image, navigate back to the lander, and communicate the data.

    # Heuristic Initialization
    - Extract the required data points from the goal conditions.
    - Store static information about waypoints, rover capabilities, and camera calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each required soil sample, check if it has been communicated. If not, add 4 actions.
    2. For each required rock sample, check if it has been communicated. If not, add 4 actions.
    3. For each required image, check if it has been communicated. If not, add 6 actions.
    4. Sum all the actions to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        static_facts = task.static

        # Parse the goals to collect required data points
        self.required_soil = set()
        self.required_rock = set()
        self.required_images = set()

        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'communicated_soil_data':
                self.required_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                self.required_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data':
                self.required_images.add((parts[1], parts[2]))

    def __call__(self, node):
        """Estimate the minimum cost to achieve all required data communications."""
        state = node.state
        total_actions = 0

        # Check each required soil data point
        for waypoint in self.required_soil:
            fact = f'(communicated_soil_data {waypoint})'
            if fact not in state:
                total_actions += 4  # 4 actions per soil data point

        # Check each required rock data point
        for waypoint in self.required_rock:
            fact = f'(communicated_rock_data {waypoint})'
            if fact not in state:
                total_actions += 4  # 4 actions per rock data point

        # Check each required image data point
        for (objective, mode) in self.required_images:
            fact = f'(communicated_image_data {objective} {mode})'
            if fact not in state:
                total_actions += 6  # 6 actions per image data point

        return total_actions
