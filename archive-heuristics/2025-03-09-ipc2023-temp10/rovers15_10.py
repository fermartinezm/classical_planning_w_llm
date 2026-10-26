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


class rovers15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the following aspects:
    - Navigating rovers to waypoints where samples need to be taken or images need to be captured.
    - Sampling soil and rock samples.
    - Calibrating cameras.
    - Taking images of objectives.
    - Communicating data back to the lander.

    # Assumptions
    - Each goal requires at least one action to achieve.
    - Navigating to a waypoint requires one action.
    - Sampling requires one action.
    - Calibrating requires one action.
    - Taking an image requires one action.
    - Communicating data requires one action.
    - The heuristic is not admissible.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Extract static information about the environment, such as visibility between waypoints,
      which rover is equipped for which task, and which camera supports which mode.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal in the goal state.
    3. For each goal, determine the type of action required to achieve it.
    4. If the goal is to have communicated data (soil, rock, or image), check if the data has already been communicated.
       If not, estimate the number of actions required to:
          a. Navigate the rover to the location where the sample/image was taken.
          b. If the sample/image hasn't been taken, take the sample/image.
          c. Navigate the rover to a waypoint visible from the lander.
          d. Communicate the data.
    5. If the goal is to have an image, check if the image has already been taken.
       If not, estimate the number of actions required to:
          a. Navigate the rover to a waypoint visible from the objective.
          b. If the camera isn't calibrated, calibrate it.
          c. Take the image.
    6. If the goal is to have soil/rock analysis, check if the analysis has already been done.
       If not, estimate the number of actions required to:
          a. Navigate the rover to the soil/rock sample location.
          b. Take the sample.
    7. Return the total estimated number of actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.lander_location = next(
            (get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None
        )
        self.equipped_for_soil_analysis = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")
        }
        self.equipped_for_rock_analysis = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*")
        }
        self.equipped_for_imaging = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")
        }
        self.store_of = {
            get_parts(fact)[2]: get_parts(fact)[1] for fact in static_facts if match(fact, "store_of", "*", "*")
        }
        self.visible = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "visible", "*", "*")
        }
        self.can_traverse = {
            (get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]) for fact in static_facts if match(fact, "can_traverse", "*", "*", "*")
        }
        self.calibration_target = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "calibration_target", "*", "*")
        }
        self.on_board = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "on_board", "*", "*")
        }
        self.supports = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "supports", "*", "*")
        }
        self.visible_from = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "visible_from", "*", "*")
        }

    def __call__(self, node):
        """
        Estimate the number of actions required to achieve the goals from the current state.
        """
        state = node.state
        heuristic_value = 0

        # Helper functions to check state conditions
        def have_soil_analysis(rover, waypoint):
            return f"(have_soil_analysis {rover} {waypoint})" in state

        def have_rock_analysis(rover, waypoint):
            return f"(have_rock_analysis {rover} {waypoint})" in state

        def have_image(rover, objective, mode):
            return f"(have_image {rover} {objective} {mode})" in state

        def communicated_soil_data(waypoint):
            return f"(communicated_soil_data {waypoint})" in state

        def communicated_rock_data(waypoint):
            return f"(communicated_rock_data {waypoint})" in state

        def communicated_image_data(objective, mode):
            return f"(communicated_image_data {objective} {mode})" in state

        def at_rover(rover, waypoint):
            return f"(at {rover} {waypoint})" in state

        def calibrated(camera, rover):
            return f"(calibrated {camera} {rover})" in state

        def at_soil_sample(waypoint):
            return f"(at_soil_sample {waypoint})" in state

        def at_rock_sample(waypoint):
            return f"(at_rock_sample {waypoint})" in state

        # Iterate through each goal and estimate the cost
        for goal in self.goals:
            predicate, *args = get_parts(goal)

            if predicate == "communicated_soil_data":
                waypoint = args[0]
                if not communicated_soil_data(waypoint):
                    # Find a rover that has soil analysis data for this waypoint
                    rover = next((r for r in self.store_of if have_soil_analysis(r, waypoint)), None)
                    if rover:
                        # Find the rover's current location
                        rover_location = next((wp for wp in self.can_traverse if at_rover(rover, wp)), None)
                        if rover_location:
                            # Check if the rover is at a location visible from the lander
                            if not any((rover_location, self.lander_location) in self.visible):
                                heuristic_value += 1  # Navigate to a waypoint visible from the lander
                            heuristic_value += 1  # Communicate soil data
                        else:
                            heuristic_value += 2 # rover not at any waypoint, add 2 to heuristic value
                    else:
                        # Find a rover that can perform soil analysis
                        rover = next((r for r in self.equipped_for_soil_analysis), None)
                        if rover:
                            # Find the rover's current location
                            rover_location = next((wp for wp in self.can_traverse if at_rover(rover, wp)), None)
                            if rover_location:
                                # Find the location of the soil sample
                                if not at_soil_sample(waypoint):
                                    heuristic_value += 1 # Soil sample already taken
                                else:
                                    heuristic_value += 1 # Take soil sample
                                # Check if the rover is at a location visible from the lander
                                if not any((rover_location, self.lander_location) in self.visible):
                                    heuristic_value += 1  # Navigate to a waypoint visible from the lander
                                heuristic_value += 1  # Communicate soil data
                            else:
                                heuristic_value += 2 # rover not at any waypoint, add 2 to heuristic value
                        else:
                            heuristic_value += 3 # no rover can perform soil analysis, add 3 to heuristic value

            elif predicate == "communicated_rock_data":
                waypoint = args[0]
                if not communicated_rock_data(waypoint):
                    # Find a rover that has rock analysis data for this waypoint
                    rover = next((r for r in self.store_of if have_rock_analysis(r, waypoint)), None)
                    if rover:
                        # Find the rover's current location
                        rover_location = next((wp for wp in self.can_traverse if at_rover(rover, wp)), None)
                        if rover_location:
                            # Check if the rover is at a location visible from the lander
                            if not any((rover_location, self.lander_location) in self.visible):
                                heuristic_value += 1  # Navigate to a waypoint visible from the lander
                            heuristic_value += 1  # Communicate rock data
                        else:
                            heuristic_value += 2 # rover not at any waypoint, add 2 to heuristic value
                    else:
                        # Find a rover that can perform rock analysis
                        rover = next((r for r in self.equipped_for_rock_analysis), None)
                        if rover:
                            # Find the rover's current location
                            rover_location = next((wp for wp in self.can_traverse if at_rover(rover, wp)), None)
                            if rover_location:
                                # Find the location of the rock sample
                                if not at_rock_sample(waypoint):
                                    heuristic_value += 1 # Rock sample already taken
                                else:
                                    heuristic_value += 1 # Take rock sample
                                # Check if the rover is at a location visible from the lander
                                if not any((rover_location, self.lander_location) in self.visible):
                                    heuristic_value += 1  # Navigate to a waypoint visible from the lander
                                heuristic_value += 1  # Communicate rock data
                            else:
                                heuristic_value += 2 # rover not at any waypoint, add 2 to heuristic value
                        else:
                            heuristic_value += 3 # no rover can perform rock analysis, add 3 to heuristic value

            elif predicate == "communicated_image_data":
                objective, mode = args
                if not communicated_image_data(objective, mode):
                    # Find a rover that has the image
                    rover = next((r for r in self.equipped_for_imaging if have_image(r, objective, mode)), None)
                    if rover:
                        # Find the rover's current location
                        rover_location = next((wp for wp in self.can_traverse if at_rover(rover, wp)), None)
                        if rover_location:
                            # Check if the rover is at a location visible from the lander
                            if not any((rover_location, self.lander_location) in self.visible):
                                heuristic_value += 1  # Navigate to a waypoint visible from the lander
                            heuristic_value += 1  # Communicate image data
                        else:
                            heuristic_value += 2 # rover not at any waypoint, add 2 to heuristic value
                    else:
                        # Find a rover that can take images
                        rover = next((r for r in self.equipped_for_imaging), None)
                        if rover:
                            # Find the rover's current location
                            rover_location = next((wp for wp in self.can_traverse if at_rover(rover, wp)), None)
                            if rover_location:
                                # Find a camera on board the rover that supports the mode
                                camera = next((c for c, ro in self.on_board if ro == rover and (c, mode) in self.supports), None)
                                if camera:
                                    # Check if the camera is calibrated
                                    if not calibrated(camera, rover):
                                        heuristic_value += 1  # Calibrate the camera
                                    # Check if the rover is at a location visible from the objective
                                    if not any((objective, rover_location) in self.visible_from):
                                        heuristic_value += 1  # Navigate to a waypoint visible from the objective
                                    heuristic_value += 1  # Take image
                                    # Check if the rover is at a location visible from the lander
                                    if not any((rover_location, self.lander_location) in self.visible):
                                        heuristic_value += 1  # Navigate to a waypoint visible from the lander
                                    heuristic_value += 1  # Communicate image data
                                else:
                                    heuristic_value += 3 # no camera supports the mode, add 3 to heuristic value
                            else:
                                heuristic_value += 2 # rover not at any waypoint, add 2 to heuristic value
                        else:
                            heuristic_value += 4 # no rover can take images, add 4 to heuristic value

        return heuristic_value
