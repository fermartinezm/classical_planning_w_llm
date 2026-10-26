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


class rovers4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy the goal conditions in the Rovers domain.
    It considers the number of soil samples, rock samples, and images that need to be collected and communicated,
    as well as the need to calibrate cameras and navigate between waypoints.

    # Assumptions
    - Each objective requires a single image in a single mode.
    - Each soil/rock sample needs to be taken only once.
    - Communicating data requires the rover to be at a waypoint visible from the lander.
    - Calibration is needed before taking images.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Identify the objectives, modes, soil samples, and rock samples that need to be communicated.
    - Store static information about visibility between waypoints and objectives, rover equipment, and camera capabilities.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check if the current state satisfies the goal conditions. If so, return 0.
    3. Count the number of uncommunicated soil samples, rock samples, and images.
    4. For each uncommunicated soil sample, estimate the cost:
       - Navigate to the soil sample location.
       - Sample the soil.
       - Navigate to a waypoint visible from the lander.
       - Communicate the soil data.
    5. For each uncommunicated rock sample, estimate the cost:
       - Navigate to the rock sample location.
       - Sample the rock.
       - Navigate to a waypoint visible from the lander.
       - Communicate the rock data.
    6. For each uncommunicated image, estimate the cost:
       - Navigate to a waypoint visible from the objective.
       - If the camera is not calibrated, calibrate it.
       - Take the image.
       - Navigate to a waypoint visible from the lander.
       - Communicate the image data.
    7. Sum the estimated costs for all uncommunicated data to obtain the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.soil_samples_needed = set()
        self.rock_samples_needed = set()
        self.images_needed = set()

        self.visible = {}
        self.visible_from = {}
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()
        self.calibration_targets = {}
        self.on_board = {}
        self.supports = {}
        self.lander_location = None

        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = set()
                self.visible[waypoint1].add(waypoint2)
            elif match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil_analysis.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock_analysis.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(get_parts(fact)[1])
            elif match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_targets[camera] = objective
            elif match(fact, "on_board", "*", "*"):
                camera = get_parts(fact)[1]
                rover = get_parts(fact)[2]
                self.on_board[camera] = rover
            elif match(fact, "supports", "*", "*"):
                camera = get_parts(fact)[1]
                mode = get_parts(fact)[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)
            elif match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]

        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                self.soil_samples_needed.add(get_parts(goal)[1])
            elif match(goal, "communicated_rock_data", "*"):
                self.rock_samples_needed.add(get_parts(goal)[1])
            elif match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                self.images_needed.add((objective, mode))

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state

        if self.goals <= state:
            return 0

        cost = 0

        # Soil samples
        for sample in self.soil_samples_needed:
            if f"(communicated_soil_data {sample})" not in state:
                cost += 4  # Navigate, sample, navigate to lander, communicate

        # Rock samples
        for sample in self.rock_samples_needed:
            if f"(communicated_rock_data {sample})" not in state:
                cost += 4  # Navigate, sample, navigate to lander, communicate

        # Images
        for objective, mode in self.images_needed:
            if f"(communicated_image_data {objective} {mode})" not in state:
                cost += 3  # Navigate, take image, navigate to lander, communicate
                
                # Check if calibration is needed
                for camera, rover in self.on_board.items():
                    if self.calibration_targets.get(camera) == objective and f"(calibrated {camera} {rover})" not in state:
                        cost += 1 # calibrate

        return cost
