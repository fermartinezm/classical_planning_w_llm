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


class rovers24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the following goals: communicating soil data, rock data, and image data.
    The heuristic estimates the cost of achieving each goal and returns the maximum of these costs.

    # Assumptions
    - Rovers need to navigate to waypoints where samples are available or objectives are visible.
    - Rovers need to be equipped with the necessary equipment (soil analysis, rock analysis, imaging).
    - Rovers need to calibrate cameras before taking images.
    - Rovers need to communicate data to the lander, which requires visibility and proximity.
    - The heuristic assumes that each rover can perform only one task at a time.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Extract static information about the environment, such as waypoint locations, visibility,
      equipment capabilities, and calibration targets.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Calculate the cost of achieving each type of goal (soil data, rock data, image data).
    3. For soil data goals:
       - Identify waypoints for which soil data needs to be communicated.
       - Estimate the cost of navigating to the waypoint, sampling soil, and communicating the data.
    4. For rock data goals:
       - Identify waypoints for which rock data needs to be communicated.
       - Estimate the cost of navigating to the waypoint, sampling rock, and communicating the data.
    5. For image data goals:
       - Identify objectives and modes for which image data needs to be communicated.
       - Estimate the cost of navigating to a waypoint visible from the objective, calibrating the camera,
         taking the image, and communicating the data.
    6. The heuristic value is the maximum of the costs calculated for each type of goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.lander_location = next(
            (get_parts(fact)[2] for fact in self.static if match(fact, "at_lander", "*", "*")), None
        )
        self.rovers = {get_parts(fact)[1] for fact in self.static if match(fact, "store_of", "*", "*")}
        self.stores = {get_parts(fact)[1] for fact in self.static if match(fact, "store_of", "*", "*")}
        self.equipped_for_soil_analysis = {
            get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_soil_analysis", "*")
        }
        self.equipped_for_rock_analysis = {
            get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_rock_analysis", "*")
        }
        self.equipped_for_imaging = {
            get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_imaging", "*")
        }
        self.can_traverse = {
            (get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3])
            for fact in self.static
            if match(fact, "can_traverse", "*", "*", "*")
        }
        self.visible = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in self.static
            if match(fact, "visible", "*", "*")
        }
        self.calibration_targets = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in self.static
            if match(fact, "calibration_target", "*", "*")
        }
        self.on_board = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in self.static
            if match(fact, "on_board", "*", "*")
        }
        self.supports = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in self.static
            if match(fact, "supports", "*", "*")
        }
        self.visible_from = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in self.static
            if match(fact, "visible_from", "*", "*")
        }

    def __call__(self, node):
        """Estimate the minimum cost to achieve the goals from the current state."""
        state = node.state

        if task.goal_reached(self, state):
            return 0

        # Extract current state information
        rover_locations = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")
        }
        have_soil_analysis = {get_parts(fact)[2] for fact in state if match(fact, "have_soil_analysis", "*", "*")}
        have_rock_analysis = {get_parts(fact)[2] for fact in state if match(fact, "have_rock_analysis", "*", "*")}
        have_image = {
            (get_parts(fact)[2], get_parts(fact)[3]) for fact in state if match(fact, "have_image", "*", "*", "*")
        }
        calibrated = {get_parts(fact)[1] for fact in state if match(fact, "calibrated", "*", "*")}
        empty_stores = {get_parts(fact)[1] for fact in state if match(fact, "empty", "*")}

        # Goal predicates
        communicated_soil_data_goals = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "communicated_soil_data", "*")
        }
        communicated_rock_data_goals = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "communicated_rock_data", "*")
        }
        communicated_image_data_goals = {
            (get_parts(goal)[1], get_parts(goal)[2])
            for goal in self.goals
            if match(goal, "communicated_image_data", "*", "*")
        }

        # Calculate cost for soil data goals
        soil_data_cost = 0
        uncommunicated_soil_data = communicated_soil_data_goals - {
            get_parts(fact)[1] for fact in state if match(fact, "communicated_soil_data", "*")
        }
        for waypoint in uncommunicated_soil_data:
            rover_with_soil_analysis = next(
                (
                    rover
                    for rover in self.rovers
                    if rover in self.equipped_for_soil_analysis
                    and rover_locations[rover] == waypoint
                ),
                None,
            )
            if rover_with_soil_analysis is None:
                soil_data_cost += 2  # Move and sample soil
            
            rover_at_waypoint = next(
                (
                    rover
                    for rover in self.rovers
                    if rover in self.equipped_for_soil_analysis
                    and rover_locations[rover] == waypoint
                    and waypoint in have_soil_analysis
                ),
                None,
            )
            if rover_at_waypoint is not None:
                rover_location = rover_locations[rover_at_waypoint]
                lander_visible = any(
                    (rover_location, self.lander_location) in self.visible
                    for rover in self.rovers
                    if rover_locations.get(rover) == rover_location
                )
                if not lander_visible:
                    soil_data_cost += 1  # Move to a visible waypoint
                soil_data_cost += 1  # Communicate soil data

        # Calculate cost for rock data goals
        rock_data_cost = 0
        uncommunicated_rock_data = communicated_rock_data_goals - {
            get_parts(fact)[1] for fact in state if match(fact, "communicated_rock_data", "*")
        }
        for waypoint in uncommunicated_rock_data:
            rover_with_rock_analysis = next(
                (
                    rover
                    for rover in self.rovers
                    if rover in self.equipped_for_rock_analysis
                    and rover_locations[rover] == waypoint
                ),
                None,
            )
            if rover_with_rock_analysis is None:
                rock_data_cost += 2  # Move and sample rock
            
            rover_at_waypoint = next(
                (
                    rover
                    for rover in self.rovers
                    if rover in self.equipped_for_rock_analysis
                    and rover_locations[rover] == waypoint
                    and waypoint in have_rock_analysis
                ),
                None,
            )
            if rover_at_waypoint is not None:
                rover_location = rover_locations[rover_at_waypoint]
                lander_visible = any(
                    (rover_location, self.lander_location) in self.visible
                    for rover in self.rovers
                    if rover_locations.get(rover) == rover_location
                )
                if not lander_visible:
                    rock_data_cost += 1  # Move to a visible waypoint
                rock_data_cost += 1  # Communicate rock data

        # Calculate cost for image data goals
        image_data_cost = 0
        uncommunicated_image_data = communicated_image_data_goals - {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in state
            if match(fact, "communicated_image_data", "*", "*")
        }
        for objective, mode in uncommunicated_image_data:
            # Find a rover that can take the image
            rover_capable = next(
                (
                    rover
                    for rover in self.rovers
                    if rover in self.equipped_for_imaging
                    and any(
                        (camera, rover) in self.on_board and (camera, mode) in self.supports
                        for camera in {cam[0] for cam in self.on_board if cam[1] == rover}
                    )
                ),
                None,
            )
            if rover_capable is None:
                image_data_cost += 5  # Impossible to achieve
                continue

            # Find a waypoint visible from the objective
            waypoint_visible = next(
                (
                    waypoint
                    for waypoint in {wp[1] for wp in self.visible_from if wp[0] == objective}
                    if rover_locations[rover_capable] == waypoint
                ),
                None,
            )
            if waypoint_visible is None:
                image_data_cost += 1  # Move to a visible waypoint

            # Check if the camera is calibrated
            camera = next(
                (
                    cam[0]
                    for cam in self.on_board
                    if cam[1] == rover_capable and (cam[0], objective) in self.calibration_targets
                ),
                None,
            )
            if camera is None:
                image_data_cost += 5  # Impossible to achieve
                continue

            if camera not in calibrated:
                image_data_cost += 1  # Calibrate camera

            image_data_cost += 1  # Take image

            rover_location = rover_locations[rover_capable]
            lander_visible = any(
                (rover_location, self.lander_location) in self.visible
                for rover in self.rovers
                if rover_locations.get(rover) == rover_location
            )
            if not lander_visible:
                image_data_cost += 1  # Move to a visible waypoint
            image_data_cost += 1  # Communicate image data

        # Return the maximum cost
        return max(soil_data_cost, rock_data_cost, image_data_cost)
