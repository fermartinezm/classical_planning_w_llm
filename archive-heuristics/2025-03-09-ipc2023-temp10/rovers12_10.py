from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class rovers12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the costs of navigating to locations with samples or objectives, sampling, calibrating cameras,
    taking images, and communicating data back to the lander. The heuristic prioritizes achieving the goals
    related to communicated data, assuming that the other actions are performed to enable these goals.

    # Assumptions
    - Rovers can perform multiple tasks concurrently.
    - Communicating data requires the rover to be in a visible location from the lander.
    - Objectives and samples are independent of each other.
    - The heuristic assumes that the rover needs to perform all actions to achieve the goals.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Extract static information about the environment, such as waypoint connections, visibility,
      sample locations, objective locations, camera capabilities, and lander location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state satisfies the goal. If so, return 0.
    2. Initialize the heuristic value to 0.
    3. For each goal related to communicating data (soil, rock, image):
       a. Check if the data has already been communicated. If so, skip this goal.
       b. Identify the rover that needs to communicate the data.
       c. Identify the waypoint where the rover has the required data.
       d. Identify the lander location.
       e. Estimate the cost of navigating the rover to a waypoint visible from the lander.
       f. Add the cost of communicating the data (1 action).
    4. For each goal related to having soil analysis:
       a. Check if the soil analysis has already been performed. If so, skip this goal.
       b. Identify the rover that needs to perform the soil analysis.
       c. Identify the waypoint where the soil sample is located.
       d. Estimate the cost of navigating the rover to the soil sample location.
       e. Add the cost of sampling the soil (1 action).
    5. For each goal related to having rock analysis:
       a. Check if the rock analysis has already been performed. If so, skip this goal.
       b. Identify the rover that needs to perform the rock analysis.
       c. Identify the waypoint where the rock sample is located.
       d. Estimate the cost of navigating the rover to the rock sample location.
       e. Add the cost of sampling the rock (1 action).
    6. For each goal related to having images:
       a. Check if the image has already been taken. If so, skip this goal.
       b. Identify the rover that needs to take the image.
       c. Identify the objective and mode for the image.
       d. Identify a waypoint visible from the objective.
       e. Estimate the cost of navigating the rover to the waypoint.
       f. Add the cost of calibrating the camera (1 action).
       g. Add the cost of taking the image (1 action).
    7. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal information and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_location = next(
            (get_parts(fact)[2] for fact in self.static if match(fact, "at_lander", "*", "*")), None
        )
        self.waypoint_connections = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static
            if match(fact, "can_traverse", "*", "*", "*")
        }
        self.visible_waypoints = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in self.static
            if match(fact, "visible", "*", "*")
        }
        self.soil_sample_locations = {
            get_parts(fact)[1] for fact in self.static if match(fact, "at_soil_sample", "*")
        }
        self.rock_sample_locations = {
            get_parts(fact)[1] for fact in self.static if match(fact, "at_rock_sample", "*")
        }
        self.objective_visible_from = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.objective_visible_from:
                    self.objective_visible_from[objective] = []
                self.objective_visible_from[objective].append(waypoint)

        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_targets[camera] = objective

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state

        if self.goal_reached(state):
            return 0

        total_cost = 0

        # Communicate soil data goals
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                waypoint = get_parts(goal)[1]
                if goal not in state:
                    # Find a rover with soil analysis at the waypoint
                    rover = next(
                        (
                            get_parts(fact)[1]
                            for fact in state
                            if match(fact, "have_soil_analysis", "*", waypoint)
                        ),
                        None,
                    )
                    if rover:
                        # Find the rover's current location
                        rover_location = next(
                            (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                        )
                        if rover_location:
                            # Find a waypoint visible from the lander
                            visible_waypoint = next(
                                (
                                    wp
                                    for wp in self.visible_waypoints
                                    if self.visible_waypoints[wp] == self.lander_location
                                    and wp == rover_location
                                ),
                                None,
                            )
                            if not visible_waypoint:
                                total_cost += 1  # Cost to navigate to a visible waypoint
                            total_cost += 1  # Cost to communicate soil data

        # Communicate rock data goals
        for goal in self.goals:
            if match(goal, "communicated_rock_data", "*"):
                waypoint = get_parts(goal)[1]
                if goal not in state:
                    # Find a rover with rock analysis at the waypoint
                    rover = next(
                        (
                            get_parts(fact)[1]
                            for fact in state
                            if match(fact, "have_rock_analysis", "*", waypoint)
                        ),
                        None,
                    )
                    if rover:
                        # Find the rover's current location
                        rover_location = next(
                            (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                        )
                        if rover_location:
                            # Find a waypoint visible from the lander
                            visible_waypoint = next(
                                (
                                    wp
                                    for wp in self.visible_waypoints
                                    if self.visible_waypoints[wp] == self.lander_location
                                    and wp == rover_location
                                ),
                                None,
                            )
                            if not visible_waypoint:
                                total_cost += 1  # Cost to navigate to a visible waypoint
                            total_cost += 1  # Cost to communicate rock data

        # Communicate image data goals
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                if goal not in state:
                    # Find a rover with the image
                    rover = next(
                        (
                            get_parts(fact)[1]
                            for fact in state
                            if match(fact, "have_image", "*", objective, mode)
                        ),
                        None,
                    )
                    if rover:
                        # Find the rover's current location
                        rover_location = next(
                            (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                        )
                        if rover_location:
                            # Find a waypoint visible from the lander
                            visible_waypoint = next(
                                (
                                    wp
                                    for wp in self.visible_waypoints
                                    if self.visible_waypoints[wp] == self.lander_location
                                    and wp == rover_location
                                ),
                                None,
                            )
                            if not visible_waypoint:
                                total_cost += 1  # Cost to navigate to a visible waypoint
                            total_cost += 1  # Cost to communicate image data

        # Sample soil goals
        for waypoint in self.soil_sample_locations:
            goal = f"(communicated_soil_data {waypoint})"
            if goal in self.goals and goal not in state:
                # Find a rover equipped for soil analysis
                rover = next(
                    (
                        get_parts(fact)[1]
                        for fact in state
                        if match(fact, "equipped_for_soil_analysis", "*")
                    ),
                    None,
                )
                if rover:
                    # Find the rover's current location
                    rover_location = next(
                        (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                    )
                    if rover_location:
                        if rover_location != waypoint:
                            total_cost += 1  # Cost to navigate to the soil sample location
                        total_cost += 1  # Cost to sample soil

        # Sample rock goals
        for waypoint in self.rock_sample_locations:
            goal = f"(communicated_rock_data {waypoint})"
            if goal in self.goals and goal not in state:
                # Find a rover equipped for rock analysis
                rover = next(
                    (
                        get_parts(fact)[1]
                        for fact in state
                        if match(fact, "equipped_for_rock_analysis", "*")
                    ),
                    None,
                )
                if rover:
                    # Find the rover's current location
                    rover_location = next(
                        (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                    )
                    if rover_location:
                        if rover_location != waypoint:
                            total_cost += 1  # Cost to navigate to the rock sample location
                        total_cost += 1  # Cost to sample rock

        # Take image goals
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                if goal not in state:
                    # Find a rover equipped for imaging
                    rover = next(
                        (
                            get_parts(fact)[1]
                            for fact in state
                            if match(fact, "equipped_for_imaging", "*")
                        ),
                        None,
                    )
                    if rover:
                        # Find the rover's current location
                        rover_location = next(
                            (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                        )
                        if rover_location:
                            # Find a waypoint visible from the objective
                            if objective in self.objective_visible_from:
                                visible_waypoint = next(
                                    (
                                        wp
                                        for wp in self.objective_visible_from[objective]
                                        if wp == rover_location
                                    ),
                                    None,
                                )
                                if not visible_waypoint:
                                    total_cost += 1  # Cost to navigate to a visible waypoint
                                # Find the camera on board the rover
                                camera = next(
                                    (
                                        cam
                                        for cam in self.calibration_targets
                                        if self.calibration_targets[cam] == objective
                                    ),
                                    None,
                                )
                                if camera:
                                    calibrated = next(
                                        (
                                            True
                                            for fact in state
                                            if match(fact, "calibrated", camera, rover)
                                        ),
                                        False,
                                    )
                                    if not calibrated:
                                        total_cost += 1  # Cost to calibrate the camera
                                total_cost += 1  # Cost to take the image

        return total_cost

    def goal_reached(self, state):
        """Check if the current state satisfies all goal conditions."""
        return self.goals <= state
