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


class rovers22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the following aspects:
    - Communicating data (soil, rock, image) for each waypoint/objective.
    - Sampling soil and rock samples.
    - Taking images of objectives in required modes.
    - Navigating to waypoints to perform actions.
    - Calibrating cameras.

    # Assumptions
    - Each rover has the necessary equipment to perform its tasks.
    - Rovers can move between any two visible waypoints.
    - Communicating data requires the rover to be at a waypoint visible from the lander.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Identify the locations of soil and rock samples.
    - Identify the objectives and the required modes for imaging.
    - Store the visibility relationships between waypoints and objectives.
    - Store the rover's capabilities (equipped for soil/rock analysis, imaging).
    - Store camera calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. For each goal predicate:
       a. If the goal is to communicate soil data:
          - Find a rover that has soil analysis data for the waypoint.
          - Estimate the cost to navigate the rover to a waypoint visible from the lander.
          - Add 1 (for communication action) to the heuristic value.
       b. If the goal is to communicate rock data:
          - Find a rover that has rock analysis data for the waypoint.
          - Estimate the cost to navigate the rover to a waypoint visible from the lander.
          - Add 1 (for communication action) to the heuristic value.
       c. If the goal is to communicate image data:
          - Find a rover that has the image for the objective and mode.
          - Estimate the cost to navigate the rover to a waypoint visible from the lander.
          - Add 1 (for communication action) to the heuristic value.
       d. If the goal is to have soil analysis data:
          - Find a rover at the waypoint.
          - If the rover doesn't have the soil analysis data, add 1 (for sampling action) to the heuristic value.
       e. If the goal is to have rock analysis data:
          - Find a rover at the waypoint.
          - If the rover doesn't have the rock analysis data, add 1 (for sampling action) to the heuristic value.
       f. If the goal is to have an image:
          - Find a rover at a waypoint visible from the objective.
          - If the rover doesn't have the image, estimate the cost to calibrate the camera (if needed) and take the image.
          - Add 1 (for imaging action) + calibration cost to the heuristic value.
    3. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal information and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.soil_samples = {get_parts(fact)[1] for fact in self.static if match(fact, "at_soil_sample", "*")}
        self.rock_samples = {get_parts(fact)[1] for fact in self.static if match(fact, "at_rock_sample", "*")}
        self.visible_from = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)
        self.equipped_for_soil_analysis = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_soil_analysis", "*")}
        self.equipped_for_rock_analysis = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_rock_analysis", "*")}
        self.equipped_for_imaging = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_imaging", "*")}
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_targets[camera] = objective
        self.on_board = {}
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                camera = get_parts(fact)[1]
                rover = get_parts(fact)[2]
                self.on_board[camera] = rover
        self.lander_location = next(get_parts(fact)[2] for fact in self.static if match(fact, "at_lander", "*", "*"))
        self.visible = {}
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = set()
                self.visible[waypoint1].add(waypoint2)

    def __call__(self, node):
        """Estimate the cost to reach the goal from the current state."""
        state = node.state
        heuristic_value = 0

        # Helper function to check if a rover has a certain image
        def has_image(rover, objective, mode):
            return f"(have_image {rover} {objective} {mode})" in state

        # Helper function to check if a rover has soil analysis data
        def has_soil_analysis(rover, waypoint):
            return f"(have_soil_analysis {rover} {waypoint})" in state

        # Helper function to check if a rover has rock analysis data
        def has_rock_analysis(rover, waypoint):
            return f"(have_rock_analysis {rover} {waypoint})" in state

        # Helper function to find a rover at a waypoint
        def find_rover_at_waypoint(waypoint):
            for fact in state:
                if match(fact, "at", "*", waypoint):
                    return get_parts(fact)[1]
            return None

        # Helper function to find a waypoint visible from the lander
        def find_waypoint_visible_from_lander(rover_location):
            if rover_location in self.visible:
                for waypoint in self.visible[rover_location]:
                    if waypoint == self.lander_location or (self.lander_location in self.visible and waypoint in self.visible[self.lander_location]):
                        return waypoint
            return None

        # Iterate through the goal predicates
        for goal in self.goals:
            predicate, *args = get_parts(goal)

            if predicate == "communicated_soil_data":
                waypoint = args[0]
                # Find a rover that has soil analysis data for the waypoint
                rover_with_data = None
                for fact in state:
                    if match(fact, "have_soil_analysis", "*", waypoint):
                        rover_with_data = get_parts(fact)[1]
                        break

                if rover_with_data:
                    # Estimate the cost to navigate the rover to a waypoint visible from the lander
                    rover_location = next(get_parts(fact)[2] for fact in state if match(fact, "at", rover_with_data, "*"))
                    waypoint_visible_from_lander = find_waypoint_visible_from_lander(rover_location)
                    if waypoint_visible_from_lander is None:
                        heuristic_value += 1  # Assume one move to a waypoint visible from lander is needed
                    # Add 1 (for communication action) to the heuristic value
                    heuristic_value += 1

            elif predicate == "communicated_rock_data":
                waypoint = args[0]
                # Find a rover that has rock analysis data for the waypoint
                rover_with_data = None
                for fact in state:
                    if match(fact, "have_rock_analysis", "*", waypoint):
                        rover_with_data = get_parts(fact)[1]
                        break

                if rover_with_data:
                    # Estimate the cost to navigate the rover to a waypoint visible from the lander
                    rover_location = next(get_parts(fact)[2] for fact in state if match(fact, "at", rover_with_data, "*"))
                    waypoint_visible_from_lander = find_waypoint_visible_from_lander(rover_location)
                    if waypoint_visible_from_lander is None:
                        heuristic_value += 1  # Assume one move to a waypoint visible from lander is needed
                    # Add 1 (for communication action) to the heuristic value
                    heuristic_value += 1

            elif predicate == "communicated_image_data":
                objective = args[0]
                mode = args[1]
                # Find a rover that has the image for the objective and mode
                rover_with_image = None
                for fact in state:
                    if match(fact, "have_image", "*", objective, mode):
                        rover_with_image = get_parts(fact)[1]
                        break

                if rover_with_image:
                    # Estimate the cost to navigate the rover to a waypoint visible from the lander
                    rover_location = next(get_parts(fact)[2] for fact in state if match(fact, "at", rover_with_image, "*"))
                    waypoint_visible_from_lander = find_waypoint_visible_from_lander(rover_location)
                    if waypoint_visible_from_lander is None:
                        heuristic_value += 1  # Assume one move to a waypoint visible from lander is needed
                    # Add 1 (for communication action) to the heuristic value
                    heuristic_value += 1

            elif predicate == "have_soil_analysis":
                rover = args[0]
                waypoint = args[1]
                if not has_soil_analysis(rover, waypoint):
                    heuristic_value += 1  # Add 1 (for sampling action) to the heuristic value

            elif predicate == "have_rock_analysis":
                rover = args[0]
                waypoint = args[1]
                if not has_rock_analysis(rover, waypoint):
                    heuristic_value += 1  # Add 1 (for sampling action) to the heuristic value

            elif predicate == "have_image":
                rover = args[0]
                objective = args[1]
                mode = args[2]
                if not has_image(rover, objective, mode):
                    # Estimate the cost to calibrate the camera (if needed) and take the image
                    camera = next((cam for cam in self.on_board if self.on_board[cam] == rover), None)
                    if camera and f"(calibrated {camera} {rover})" not in state:
                        heuristic_value += 1  # Calibration cost
                    heuristic_value += 1  # Add 1 (for imaging action) to the heuristic value

        return heuristic_value
