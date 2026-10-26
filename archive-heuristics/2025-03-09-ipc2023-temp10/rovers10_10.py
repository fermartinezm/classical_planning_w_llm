from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class rovers10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of soil samples, rock samples, and images that need to be collected and communicated.
    It also takes into account the need to navigate to waypoints to perform these actions and communicate with the lander.

    # Assumptions
    - Each rover can perform one action at a time.
    - Communicating data requires the rover to be within visible range of the lander.
    - Taking images requires the camera to be calibrated and the objective to be visible from the rover's location.
    - Sampling soil and rock requires the rover to be at the sample location and have an empty store.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Identify the locations of soil and rock samples from the static facts.
    - Identify the visibility relationships between waypoints and objectives.
    - Identify the lander location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Count the number of uncommunicated soil samples. For each uncommunicated soil sample:
       - Estimate the cost to sample the soil (navigate to the soil sample location, sample the soil, navigate to the lander, communicate the data).
    3. Count the number of uncommunicated rock samples. For each uncommunicated rock sample:
       - Estimate the cost to sample the rock (navigate to the rock sample location, sample the rock, navigate to the lander, communicate the data).
    4. Count the number of uncommunicated images. For each uncommunicated image:
       - Estimate the cost to take the image (navigate to a waypoint visible from the objective, calibrate the camera, take the image, navigate to the lander, communicate the data).
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract soil and rock sample locations
        self.soil_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")
        }
        self.rock_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")
        }

        # Extract visibility relationships between waypoints and objectives
        self.visible_from = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)

        # Extract lander location
        self.lander_location = None
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]
                break

        # Extract rover locations
        self.rovers = set()
        for fact in static_facts:
            if match(fact, "store_of", "*", "*"):
                self.rovers.add(get_parts(fact)[2])

        self.equipped_for_soil_analysis = set()
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil_analysis.add(get_parts(fact)[1])

        self.equipped_for_rock_analysis = set()
        for fact in static_facts:
            if match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock_analysis.add(get_parts(fact)[1])

        self.equipped_for_imaging = set()
        for fact in static_facts:
            if match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(get_parts(fact)[1])

        self.calibration_targets = {}
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_targets[camera] = objective

        self.on_board = {}
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                camera = get_parts(fact)[1]
                rover = get_parts(fact)[2]
                self.on_board[camera] = rover

        self.supports = {}
        for fact in static_facts:
            if match(fact, "supports", "*", "*"):
                camera = get_parts(fact)[1]
                mode = get_parts(fact)[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        if task.goal_reached(state):
            return 0

        cost = 0

        # Uncommunicated soil samples
        uncommunicated_soil = set()
        for sample in self.soil_samples:
            if f"(communicated_soil_data {sample})" not in state:
                uncommunicated_soil.add(sample)

        for sample in uncommunicated_soil:
            # Find a rover that can sample soil and has an empty store
            sampling_rover = None
            for rover in self.rovers:
                if (f"at {rover} {sample})" in state and
                        rover in self.equipped_for_soil_analysis):
                    store = None
                    for fact in state:
                        if match(fact, "store_of", "*", rover):
                            store = get_parts(fact)[1]
                            break
                    if store and f"empty {store})" in state:
                        sampling_rover = rover
                        break

            if sampling_rover:
                # Cost: sample soil + navigate to lander + communicate data
                cost += 3

        # Uncommunicated rock samples
        uncommunicated_rock = set()
        for sample in self.rock_samples:
            if f"(communicated_rock_data {sample})" not in state:
                uncommunicated_rock.add(sample)

        for sample in uncommunicated_rock:
            # Find a rover that can sample rock and has an empty store
            sampling_rover = None
            for rover in self.rovers:
                if (f"at {rover} {sample})" in state and
                        rover in self.equipped_for_rock_analysis):
                    store = None
                    for fact in state:
                        if match(fact, "store_of", "*", rover):
                            store = get_parts(fact)[1]
                            break
                    if store and f"empty {store})" in state:
                        sampling_rover = rover
                        break

            if sampling_rover:
                # Cost: sample rock + navigate to lander + communicate data
                cost += 3

        # Uncommunicated images
        uncommunicated_images = []
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                if f"(communicated_image_data {objective} {mode})" not in state:
                    uncommunicated_images.append((objective, mode))

        for objective, mode in uncommunicated_images:
            # Find a rover that can take the image
            imaging_rover = None
            for rover in self.rovers:
                if rover in self.equipped_for_imaging:
                    for camera, cam_rover in self.on_board.items():
                        if cam_rover == rover and mode in self.supports.get(camera, set()):
                            if camera in self.calibration_targets and self.calibration_targets[camera] == objective:
                                if objective in self.visible_from:
                                    for waypoint in self.visible_from[objective]:
                                        if f"at {rover} {waypoint})" in state:
                                            imaging_rover = rover
                                            break
                                    if imaging_rover:
                                        break
                    if imaging_rover:
                        break

            if imaging_rover:
                # Cost: calibrate + take image + navigate to lander + communicate data
                cost += 4

        return cost
