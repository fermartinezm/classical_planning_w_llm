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


class rovers5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the rovers domain.
    It considers the number of images, soil samples, and rock samples that need to be communicated,
    as well as the need to calibrate cameras and navigate to waypoints.

    # Assumptions
    - Each rover can perform actions independently.
    - Communicating data requires the rover to be at a waypoint visible from the lander.
    - Taking an image requires the rover to be at a waypoint visible from the objective and the camera to be calibrated.
    - Sampling soil/rock requires the rover to be at a waypoint with a soil/rock sample and an empty store.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Identify the locations of soil and rock samples.
    - Identify the visibility relationships between waypoints and objectives.
    - Identify the lander location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check if the current state satisfies the goal. If so, return 0.
    3. Count the number of uncommunicated soil samples. For each uncommunicated soil sample:
       - Estimate the cost of navigating to the soil sample location.
       - Estimate the cost of sampling the soil.
       - Estimate the cost of navigating to a waypoint visible from the lander.
       - Estimate the cost of communicating the soil data.
    4. Count the number of uncommunicated rock samples. For each uncommunicated rock sample:
       - Estimate the cost of navigating to the rock sample location.
       - Estimate the cost of sampling the rock.
       - Estimate the cost of navigating to a waypoint visible from the lander.
       - Estimate the cost of communicating the rock data.
    5. Count the number of uncommunicated images. For each uncommunicated image:
       - Estimate the cost of navigating to a waypoint visible from the objective.
       - Estimate the cost of calibrating the camera (if not already calibrated).
       - Estimate the cost of taking the image.
       - Estimate the cost of navigating to a waypoint visible from the lander.
       - Estimate the cost of communicating the image data.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.soil_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")
        }
        self.rock_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")
        }
        self.visible_from = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)

        self.lander_location = next(
            (get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None
        )

        self.can_traverse = {}
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover = get_parts(fact)[1]
                waypoint1 = get_parts(fact)[2]
                waypoint2 = get_parts(fact)[3]
                if (rover, waypoint1) not in self.can_traverse:
                    self.can_traverse[(rover, waypoint1)] = set()
                self.can_traverse[(rover, waypoint1)].add(waypoint2)

        self.calibration_targets = {}
        self.on_board = {}
        self.supports = {}
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
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

        self.equipped_for_imaging = set()
        for fact in static_facts:
            if match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(get_parts(fact)[1])

        self.equipped_for_soil_analysis = set()
        for fact in static_facts:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil_analysis.add(get_parts(fact)[1])

        self.equipped_for_rock_analysis = set()
        for fact in static_facts:
            if match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock_analysis.add(get_parts(fact)[1])

        self.store_of = {}
        for fact in static_facts:
            if match(fact, "store_of", "*", "*"):
                store = get_parts(fact)[1]
                rover = get_parts(fact)[2]
                self.store_of[rover] = store

        self.visible = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = set()
                self.visible[waypoint1].add(waypoint2)

    def __call__(self, node):
        """Estimate the minimum cost to achieve the goals from the current state."""
        state = node.state
        if task.goal_reached(state):
            return 0

        total_cost = 0

        # Uncommunicated soil samples
        uncommunicated_soil = set()
        for sample in self.soil_samples:
            if f"(communicated_soil_data {sample})" not in state:
                uncommunicated_soil.add(sample)

        for sample in uncommunicated_soil:
            rover = next((get_parts(fact)[1] for fact in state if match(fact, "at", "*", sample)), None)
            if rover is None:
                rover = next((r for r in self.equipped_for_soil_analysis), None)
                if rover is None:
                    continue

            current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
            if current_location is None:
                continue

            if f"(have_soil_analysis {rover} {sample})" not in state:
                total_cost += 1  # Navigate to soil sample
                total_cost += 1  # Sample soil
            
            total_cost += 1 # Navigate to lander
            total_cost += 1  # Communicate soil data

        # Uncommunicated rock samples
        uncommunicated_rock = set()
        for sample in self.rock_samples:
            if f"(communicated_rock_data {sample})" not in state:
                uncommunicated_rock.add(sample)

        for sample in uncommunicated_rock:
            rover = next((get_parts(fact)[1] for fact in state if match(fact, "at", "*", sample)), None)
            if rover is None:
                rover = next((r for r in self.equipped_for_rock_analysis), None)
                if rover is None:
                    continue

            current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
            if current_location is None:
                continue

            if f"(have_rock_analysis {rover} {sample})" not in state:
                total_cost += 1  # Navigate to rock sample
                total_cost += 1  # Sample rock

            total_cost += 1 # Navigate to lander
            total_cost += 1  # Communicate rock data

        # Uncommunicated images
        uncommunicated_images = []
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                if f"(communicated_image_data {objective} {mode})" not in state:
                    uncommunicated_images.append((objective, mode))

        for objective, mode in uncommunicated_images:
            rover = next((r for r in self.equipped_for_imaging), None)
            if rover is None:
                continue

            camera = next((c for c, r in self.on_board.items() if r == rover and mode in self.supports.get(c, set())), None)
            if camera is None:
                continue

            current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
            if current_location is None:
                continue

            if f"(have_image {rover} {objective} {mode})" not in state:
                total_cost += 1  # Navigate to objective

                if f"(calibrated {camera} {rover})" not in state:
                    total_cost += 1  # Calibrate camera

                total_cost += 1  # Take image

            total_cost += 1 # Navigate to lander
            total_cost += 1  # Communicate image data

        return total_cost
