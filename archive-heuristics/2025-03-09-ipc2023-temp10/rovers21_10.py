from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class rovers21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goals in the Rovers domain.
    It considers the number of objectives that need to be imaged, soil samples to be collected,
    rock samples to be collected, and the communication of data back to the lander.

    # Assumptions
    - Each objective requires a calibrate and take_image action for each mode.
    - Each sample (soil or rock) requires a sample action.
    - Each data point (image, soil, or rock) requires a communicate action.
    - Rovers need to navigate to locations where samples can be taken or objectives are visible.
    - The heuristic ignores the capacity of the rover's store.
    - The heuristic assumes that all objectives, soil samples and rock samples can be reached from the current rover locations.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Identify the objectives, modes, soil samples, and rock samples from the goal conditions.
    - Store static information about the problem, such as which objectives are visible from which waypoints,
      which waypoints have soil/rock samples, and which rovers are equipped for certain tasks.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Count the number of uncommunicated image data goals and estimate the cost:
       - For each uncommunicated image, estimate the cost as 2 (calibrate + take_image) + 1 (communicate).
    3. Count the number of uncommunicated soil data goals and estimate the cost:
       - For each uncommunicated soil sample, estimate the cost as 1 (sample) + 1 (communicate).
    4. Count the number of uncommunicated rock data goals and estimate the cost:
       - For each uncommunicated rock sample, estimate the cost as 1 (sample) + 1 (communicate).
    5. If the state satisfies all goals, return 0. Otherwise, return the estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.objectives = set()
        self.modes = set()
        self.soil_samples = set()
        self.rock_samples = set()

        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'communicated_image_data':
                self.objectives.add(parts[1])
                self.modes.add(parts[2])
            elif parts[0] == 'communicated_soil_data':
                self.soil_samples.add(parts[1])
            elif parts[0] == 'communicated_rock_data':
                self.rock_samples.add(parts[1])

        self.visible_from = {}
        self.at_soil_sample = set()
        self.at_rock_sample = set()
        self.equipped_for_imaging = set()
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.calibration_targets = {}
        self.on_board = {}

        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'visible_from':
                objective = parts[1]
                waypoint = parts[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            elif parts[0] == 'at_soil_sample':
                self.at_soil_sample.add(parts[1])
            elif parts[0] == 'at_rock_sample':
                self.at_rock_sample.add(parts[1])
            elif parts[0] == 'equipped_for_imaging':
                self.equipped_for_imaging.add(parts[1])
            elif parts[0] == 'equipped_for_soil_analysis':
                self.equipped_for_soil_analysis.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis':
                self.equipped_for_rock_analysis.add(parts[1])
            elif parts[0] == 'calibration_target':
                camera = parts[1]
                objective = parts[2]
                self.calibration_targets[camera] = objective
            elif parts[0] == 'on_board':
                camera = parts[1]
                rover = parts[2]
                self.on_board[camera] = rover

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state

        if self.goals <= state:
            return 0

        cost = 0

        # Uncommunicated image data
        uncommunicated_images = set()
        for objective in self.objectives:
            for mode in self.modes:
                if f'(communicated_image_data {objective} {mode})' not in state:
                    uncommunicated_images.add((objective, mode))

        cost += len(uncommunicated_images) * 3  # calibrate + take_image + communicate

        # Uncommunicated soil data
        uncommunicated_soil = set()
        for sample in self.soil_samples:
            if f'(communicated_soil_data {sample})' not in state:
                uncommunicated_soil.add(sample)

        cost += len(uncommunicated_soil) * 2  # sample + communicate

        # Uncommunicated rock data
        uncommunicated_rock = set()
        for sample in self.rock_samples:
            if f'(communicated_rock_data {sample})' not in state:
                uncommunicated_rock.add(sample)

        cost += len(uncommunicated_rock) * 2  # sample + communicate

        return cost
