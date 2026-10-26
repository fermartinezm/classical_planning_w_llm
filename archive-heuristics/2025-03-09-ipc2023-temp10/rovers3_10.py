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


class rovers3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of soil samples, rock samples, and images that need to be collected and communicated.
    It also accounts for the need to calibrate cameras before taking images.

    # Assumptions
    - Rovers can perform tasks independently.
    - Communicating data requires the rover to be at a waypoint visible from the lander.
    - Calibrating a camera requires the rover to be at a waypoint visible from the calibration target.
    - Rovers have sufficient capacity to store all samples and images.

    # Heuristic Initialization
    - Extract the goal conditions.
    - Identify the locations of soil and rock samples.
    - Identify the objectives and their required image modes.
    - Identify the calibration targets for each camera.
    - Store static information about visibility and traversability.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check if the current state satisfies the goal. If so, return 0.
    3. For each goal related to communicated soil data:
       - If the soil data has not been communicated, estimate the cost to:
         - Sample the soil (if not already sampled).
         - Navigate to a waypoint visible from the lander.
         - Communicate the soil data.
    4. For each goal related to communicated rock data:
       - If the rock data has not been communicated, estimate the cost to:
         - Sample the rock (if not already sampled).
         - Navigate to a waypoint visible from the lander.
         - Communicate the rock data.
    5. For each goal related to communicated image data:
       - If the image data has not been communicated, estimate the cost to:
         - Calibrate the camera (if not already calibrated).
         - Navigate to a waypoint visible from the objective.
         - Take the image.
         - Navigate to a waypoint visible from the lander.
         - Communicate the image data.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract information about soil and rock samples
        self.soil_samples = {
            get_parts(fact)[1] for fact in self.static if match(fact, "at_soil_sample", "*")
        }
        self.rock_samples = {
            get_parts(fact)[1] for fact in self.static if match(fact, "at_rock_sample", "*")
        }

        # Extract information about objectives and their visibility
        self.objectives = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.objectives:
                    self.objectives[objective] = set()
                self.objectives[objective].add(waypoint)

        # Extract information about calibration targets
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_targets[camera] = objective

        # Extract information about lander location
        self.lander_location = next(
            (get_parts(fact)[2] for fact in self.static if match(fact, "at_lander", "*", "*")),
            None,
        )

        # Extract visibility information
        self.visibility = {}
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.visibility:
                    self.visibility[waypoint1] = set()
                self.visibility[waypoint1].add(waypoint2)

    def __call__(self, node):
        """Estimate the cost to reach the goal from the given state."""
        state = node.state
        if self.goals <= state:
            return 0

        cost = 0

        # Soil data goals
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                waypoint = get_parts(goal)[1]
                if goal not in state:
                    # Sample soil if not already sampled
                    if f"(have_soil_analysis * {waypoint})" not in state:
                        cost += 1  # sample_soil action

                    # Navigate to lander and communicate
                    cost += 2  # navigate + communicate_soil_data

            elif match(goal, "communicated_rock_data", "*"):
                waypoint = get_parts(goal)[1]
                if goal not in state:
                    # Sample rock if not already sampled
                    if f"(have_rock_analysis * {waypoint})" not in state:
                        cost += 1  # sample_rock action

                    # Navigate to lander and communicate
                    cost += 2  # navigate + communicate_rock_data

            elif match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                if goal not in state:
                    # Find a rover that can take the image
                    camera = next(
                        (
                            cam
                            for cam in self.calibration_targets
                            if self.calibration_targets[cam] == objective
                        ),
                        None,
                    )
                    if not camera:
                        continue

                    # Calibrate the camera if not already calibrated
                    rover = next(
                        (
                            get_parts(fact)[2]
                            for fact in self.static
                            if match(fact, "on_board", camera, "*")
                        ),
                        None,
                    )

                    if rover:
                        if f"(calibrated {camera} {rover})" not in state:
                            cost += 1  # calibrate action

                        # Take image
                        cost += 1  # take_image action

                        # Navigate to lander and communicate
                        cost += 2  # navigate + communicate_image_data

        return cost
