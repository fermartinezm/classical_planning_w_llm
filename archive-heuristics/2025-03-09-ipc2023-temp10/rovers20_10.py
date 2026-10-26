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


class rovers20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the costs of navigating to locations with samples or objectives, sampling, calibrating,
    taking images, and communicating data back to the lander. The heuristic prioritizes collecting
    samples and images and then communicating them.

    # Assumptions
    - Rovers can perform multiple tasks concurrently (e.g., collect soil and rock samples).
    - Rovers have limited storage and must drop samples before collecting more.
    - Rovers must calibrate cameras before taking images.
    - Rovers must be within communication range of the lander to communicate data.

    # Heuristic Initialization
    - Extract the locations of soil and rock samples.
    - Extract the visibility relationships between waypoints.
    - Extract the objectives and their visible waypoints.
    - Identify the rover's capabilities (equipped for soil/rock analysis, imaging).
    - Identify the lander's location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check if the current state satisfies the goal. If so, return 0.
    3. Identify the rover locations.
    4. Estimate the cost of collecting soil samples:
       - For each waypoint with a soil sample that hasn't been communicated, estimate the cost of navigating to that waypoint, sampling the soil, and communicating the data.
    5. Estimate the cost of collecting rock samples:
       - For each waypoint with a rock sample that hasn't been communicated, estimate the cost of navigating to that waypoint, sampling the rock, and communicating the data.
    6. Estimate the cost of taking images:
       - For each objective for which images in all required modes haven't been communicated, estimate the cost of navigating to a waypoint visible from the objective, calibrating the camera, taking the image, and communicating the data.
    7. The heuristic value is the sum of the estimated costs for all tasks.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Extract sample locations
        self.soil_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")
        }
        self.rock_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")
        }

        # Extract visibility relationships
        self.visible = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1, waypoint2 = get_parts(fact)[1], get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = set()
                self.visible[waypoint1].add(waypoint2)

        # Extract objective visibility
        self.objective_visibility = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1], get_parts(fact)[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)

        # Extract rover capabilities
        self.equipped_for_soil = any(
            match(fact, "equipped_for_soil_analysis", "*") for fact in static_facts
        )
        self.equipped_for_rock = any(
            match(fact, "equipped_for_rock_analysis", "*") for fact in static_facts
        )
        self.equipped_for_imaging = any(
            match(fact, "equipped_for_imaging", "*") for fact in static_facts
        )

        # Extract lander location
        self.lander_location = next(
            (get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None
        )

        # Extract calibration targets
        self.calibration_targets = {}
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
                camera, target = get_parts(fact)[1], get_parts(fact)[2]
                self.calibration_targets[camera] = target

        # Extract supported camera modes
        self.supported_modes = {}
        for fact in static_facts:
            if match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1], get_parts(fact)[2]
                if camera not in self.supported_modes:
                    self.supported_modes[camera] = set()
                self.supported_modes[camera].add(mode)

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state

        if self.goals <= state:
            return 0

        heuristic_value = 0

        # Get rover locations
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*") and get_parts(fact)[1] not in ["general"]:
                rover, location = get_parts(fact)[1], get_parts(fact)[2]
                rover_locations[rover] = location

        # Soil samples
        for sample_location in self.soil_samples:
            if not any(match(fact, "communicated_soil_data", sample_location) for fact in state):
                rover = next(
                    (rover for rover, location in rover_locations.items() if location == sample_location),
                    None,
                )
                if not rover:
                    heuristic_value += 1  # Cost to navigate
                heuristic_value += 1  # Cost to sample
                heuristic_value += 1  # Cost to communicate

        # Rock samples
        for sample_location in self.rock_samples:
            if not any(match(fact, "communicated_rock_data", sample_location) for fact in state):
                rover = next(
                    (rover for rover, location in rover_locations.items() if location == sample_location),
                    None,
                )
                if not rover:
                    heuristic_value += 1  # Cost to navigate
                heuristic_value += 1  # Cost to sample
                heuristic_value += 1  # Cost to communicate

        # Images
        objectives_to_image = set()
        for goal in self.goals:
            if "communicated_image_data" in goal:
                objectives_to_image.add(get_parts(goal)[1])

        for objective in objectives_to_image:
            for mode in [get_parts(goal)[2] for goal in self.goals if "communicated_image_data" in goal and get_parts(goal)[1] == objective]:
                if not any(match(fact, "communicated_image_data", objective, mode) for fact in state):
                    # Find a rover with a camera that supports the mode
                    rover = None
                    camera = None
                    for fact in state:
                        if match(fact, "calibrated", "*", "*"):
                            cam, rov = get_parts(fact)[1], get_parts(fact)[2]
                            if cam in self.supported_modes and mode in self.supported_modes[cam]:
                                rover = rov
                                camera = cam
                                break
                    if not rover:
                        heuristic_value += 2  # calibrate + take image
                    heuristic_value += 1  # Cost to take image
                    heuristic_value += 1  # Cost to communicate

        return heuristic_value
