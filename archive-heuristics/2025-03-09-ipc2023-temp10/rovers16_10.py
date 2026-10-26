from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class rovers16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of images to be taken and communicated, the number of soil and rock samples to be collected and communicated,
    and the number of navigations and calibrations required.

    # Assumptions
    - Each objective requires imaging in all specified modes.
    - Each sample (soil or rock) needs to be analyzed and communicated.
    - Rovers need to be at the correct location to perform actions.
    - Calibration is needed before taking images.
    - Rovers have sufficient capacity to store samples.

    # Heuristic Initialization
    - Extract the goal conditions.
    - Identify the locations of soil and rock samples.
    - Identify the objectives and their visibility from waypoints.
    - Identify the lander location.
    - Identify camera calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize the heuristic value to 0.
    2.  Count the number of uncommunicated image data goals. For each uncommunicated image, estimate the cost as follows:
        a. Check if the rover has the image. If not, estimate the cost of taking the image:
            i.  Find a waypoint from which the objective is visible.
            ii. Estimate the cost of navigating to that waypoint.
            iii.Estimate the cost of calibrating the camera if it is not already calibrated.
            iv. Add the cost of taking the image.
        b. Find the lander location.
        c. Find the rover's current location.
        d. Estimate the cost of navigating to a waypoint visible from the lander.
        e. Add the cost of communicating the image data.
    3.  Count the number of uncommunicated soil data goals. For each uncommunicated soil data, estimate the cost as follows:
        a. Check if the rover has the soil analysis. If not, estimate the cost of sampling the soil:
            i.  Find the location of the soil sample.
            ii. Estimate the cost of navigating to that location.
            iii.Add the cost of sampling the soil.
        b. Find the lander location.
        c. Find the rover's current location.
        d. Estimate the cost of navigating to a waypoint visible from the lander.
        e. Add the cost of communicating the soil data.
    4.  Count the number of uncommunicated rock data goals. For each uncommunicated rock data, estimate the cost as follows:
        a. Check if the rover has the rock analysis. If not, estimate the cost of sampling the rock:
            i.  Find the location of the rock sample.
            ii. Estimate the cost of navigating to that location.
            iii.Add the cost of sampling the rock.
        b. Find the lander location.
        c. Find the rover's current location.
        d. Estimate the cost of navigating to a waypoint visible from the lander.
        e. Add the cost of communicating the rock data.
    5.  If all goals are achieved, return 0. Otherwise, return the estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.soil_samples = set()
        self.rock_samples = set()
        self.objectives = set()
        self.visible_from = {}
        self.calibration_targets = {}
        self.lander_location = None

        for fact in self.static:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] == 'at_soil_sample':
                self.soil_samples.add(fact_parts[1])
            elif fact_parts[0] == 'at_rock_sample':
                self.rock_samples.add(fact_parts[1])
            elif fact_parts[0] == 'visible_from':
                objective = fact_parts[1]
                waypoint = fact_parts[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            elif fact_parts[0] == 'calibration_target':
                camera = fact_parts[1]
                objective = fact_parts[2]
                self.calibration_targets[camera] = objective
            elif fact_parts[0] == 'at_lander':
                self.lander_location = fact_parts[2]

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        if self.goals <= state:
            return 0

        h = 0

        # Uncommunicated image data
        uncommunicated_images = set()
        for goal in self.goals:
            fact_parts = self._extract_objects_from_fact(goal)
            if fact_parts[0] == 'communicated_image_data':
                uncommunicated_images.add(goal)

        for image_goal in uncommunicated_images:
            fact_parts = self._extract_objects_from_fact(image_goal)
            objective = fact_parts[1]
            mode = fact_parts[2]
            have_image = f'(have_image * {objective} {mode})'
            if not any(fnmatch(fact, have_image) for fact in state):
                # Need to take the image
                best_waypoint = None
                min_distance = float('inf')
                if objective in self.visible_from:
                    for waypoint in self.visible_from[objective]:
                        for fact in state:
                            rover_parts = self._extract_objects_from_fact(fact)
                            if rover_parts[0] == 'at':
                                rover_location = rover_parts[2]
                                if self._can_traverse(rover_location, waypoint):
                                    distance = self._shortest_path(rover_location, waypoint)
                                    if distance < min_distance:
                                        min_distance = distance
                                        best_waypoint = waypoint
                    if best_waypoint:
                        h += min_distance
                        camera_calibrated = '(calibrated * *)'
                        if not any(fnmatch(fact, camera_calibrated) for fact in state):
                            h += 1  # calibrate
                        h += 1  # take_image
            # Communicate the image
            rover_location = None
            for fact in state:
                rover_parts = self._extract_objects_from_fact(fact)
                if rover_parts[0] == 'at':
                    rover_location = rover_parts[2]
                    break
            if rover_location:
                if self._can_traverse(rover_location, self.lander_location):
                    distance = self._shortest_path(rover_location, self.lander_location)
                    h += distance
                else:
                    h += 5 # arbitrary number if no path to lander
                h += 1  # communicate_image_data

        # Uncommunicated soil data
        uncommunicated_soil = set()
        for goal in self.goals:
            fact_parts = self._extract_objects_from_fact(goal)
            if fact_parts[0] == 'communicated_soil_data':
                uncommunicated_soil.add(goal)

        for soil_goal in uncommunicated_soil:
            fact_parts = self._extract_objects_from_fact(soil_goal)
            waypoint = fact_parts[1]
            have_soil = f'(have_soil_analysis * {waypoint})'
            if not any(fnmatch(fact, have_soil) for fact in state):
                # Need to sample soil
                rover_location = None
                for fact in state:
                    rover_parts = self._extract_objects_from_fact(fact)
                    if rover_parts[0] == 'at':
                        rover_location = rover_parts[2]
                        break
                if rover_location:
                    if self._can_traverse(rover_location, waypoint):
                        distance = self._shortest_path(rover_location, waypoint)
                        h += distance
                    else:
                        h += 5 # arbitrary number if no path to soil sample
                    h += 1  # sample_soil
            # Communicate soil data
            rover_location = None
            for fact in state:
                rover_parts = self._extract_objects_from_fact(fact)
                if rover_parts[0] == 'at':
                    rover_location = rover_parts[2]
                    break
            if rover_location:
                if self._can_traverse(rover_location, self.lander_location):
                    distance = self._shortest_path(rover_location, self.lander_location)
                    h += distance
                else:
                    h += 5 # arbitrary number if no path to lander
                h += 1  # communicate_soil_data

        # Uncommunicated rock data
        uncommunicated_rock = set()
        for goal in self.goals:
            fact_parts = self._extract_objects_from_fact(goal)
            if fact_parts[0] == 'communicated_rock_data':
                uncommunicated_rock.add(goal)

        for rock_goal in uncommunicated_rock:
            fact_parts = self._extract_objects_from_fact(rock_goal)
            waypoint = fact_parts[1]
            have_rock = f'(have_rock_analysis * {waypoint})'
            if not any(fnmatch(fact, have_rock) for fact in state):
                # Need to sample rock
                rover_location = None
                for fact in state:
                    rover_parts = self._extract_objects_from_fact(fact)
                    if rover_parts[0] == 'at':
                        rover_location = rover_parts[2]
                        break
                if rover_location:
                    if self._can_traverse(rover_location, waypoint):
                        distance = self._shortest_path(rover_location, waypoint)
                        h += distance
                    else:
                        h += 5 # arbitrary number if no path to rock sample
                    h += 1  # sample_rock
            # Communicate rock data
            rover_location = None
            for fact in state:
                rover_parts = self._extract_objects_from_fact(fact)
                if rover_parts[0] == 'at':
                    rover_location = rover_parts[2]
                    break
            if rover_location:
                if self._can_traverse(rover_location, self.lander_location):
                    distance = self._shortest_path(rover_location, self.lander_location)
                    h += distance
                else:
                    h += 5 # arbitrary number if no path to lander
                h += 1  # communicate_rock_data

        return h

    def _extract_objects_from_fact(self, fact):
        """Extract object names from a PDDL fact string."""
        return fact[1:-1].split()

    def _can_traverse(self, start, end):
        """Check if the rover can traverse between two waypoints."""
        can_traverse_fact = f'(can_traverse * {start} {end})'
        return any(fnmatch(fact, can_traverse_fact) for fact in self.static)

    def _shortest_path(self, start, end):
        """Estimate the cost of navigating between two waypoints (simplified)."""
        if start == end:
            return 0
        else:
            return 1  # Assuming each traverse action costs 1

