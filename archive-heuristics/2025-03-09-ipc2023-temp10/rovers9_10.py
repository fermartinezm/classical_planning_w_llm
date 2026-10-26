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


class rovers9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goals in the rovers domain.
    It considers the number of communicated data objectives (soil, rock, and image) that still need to be achieved,
    and estimates the cost based on the need to navigate, sample, calibrate, take images, and communicate data.

    # Assumptions
    - Each objective requires a sequence of actions: navigate, sample/calibrate/take_image, and communicate.
    - The heuristic assumes that each rover can perform all necessary actions if equipped.
    - It does not explicitly consider store capacity or the need to drop samples.
    - It assumes that rovers are already equipped and calibrated if needed.

    # Heuristic Initialization
    - Extract the goal conditions (communicated data objectives) from the task.
    - Identify the locations of landers from the static facts.
    - Identify the visible waypoints from objectives.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Count the number of uncommunicated soil, rock, and image data objectives.
    3. For each uncommunicated soil data objective:
       - Add a cost for sampling soil (if not already sampled).
       - Add a cost for navigating to the soil sample location.
       - Add a cost for communicating the soil data.
    4. For each uncommunicated rock data objective:
       - Add a cost for sampling rock (if not already sampled).
       - Add a cost for navigating to the rock sample location.
       - Add a cost for communicating the rock data.
    5. For each uncommunicated image data objective:
       - Add a cost for calibrating the camera (if not already calibrated).
       - Add a cost for taking the image.
       - Add a cost for navigating to a waypoint visible from the objective.
       - Add a cost for communicating the image data.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract lander locations
        self.lander_locations = {
            get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")
        }

        # Extract visible waypoints from objectives
        self.visible_waypoints_from_objective = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_waypoints_from_objective:
                    self.visible_waypoints_from_objective[objective] = set()
                self.visible_waypoints_from_objective[objective].add(waypoint)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Count uncommunicated data objectives
        uncommunicated_soil_data = set()
        uncommunicated_rock_data = set()
        uncommunicated_image_data = set()

        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                waypoint = get_parts(goal)[1]
                if goal not in state:
                    uncommunicated_soil_data.add(waypoint)
            elif match(goal, "communicated_rock_data", "*"):
                waypoint = get_parts(goal)[1]
                if goal not in state:
                    uncommunicated_rock_data.add(waypoint)
            elif match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                if goal not in state:
                    uncommunicated_image_data.add((objective, mode))

        # Estimate cost for uncommunicated soil data
        for waypoint in uncommunicated_soil_data:
            if not any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state):
                heuristic_value += 1  # Sample soil
            if not any(match(fact, "at", "*", waypoint) for fact in state):
                heuristic_value += 1  # Navigate to soil sample
            heuristic_value += 1  # Communicate soil data

        # Estimate cost for uncommunicated rock data
        for waypoint in uncommunicated_rock_data:
            if not any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state):
                heuristic_value += 1  # Sample rock
            if not any(match(fact, "at", "*", waypoint) for fact in state):
                heuristic_value += 1  # Navigate to rock sample
            heuristic_value += 1  # Communicate rock data

        # Estimate cost for uncommunicated image data
        for objective, mode in uncommunicated_image_data:
            # Check if calibrated
            # if not any(match(fact, "calibrated", "*", "*") for fact in state):
            #    heuristic_value += 1  # Calibrate camera
            heuristic_value += 1  # Take image
            if objective in self.visible_waypoints_from_objective:
                visible_waypoints = self.visible_waypoints_from_objective[objective]
                if not any(match(fact, "at", "*", waypoint) for fact in state for waypoint in visible_waypoints):
                    heuristic_value += 1  # Navigate to waypoint visible from objective
            heuristic_value += 1  # Communicate image data

        return heuristic_value
