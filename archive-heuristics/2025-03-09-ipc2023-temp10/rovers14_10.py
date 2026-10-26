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


class rovers14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of images to be taken and communicated, the number of soil and rock samples to be taken and communicated,
    and the number of navigations required to reach the waypoints where these actions can be performed.

    # Assumptions
    - Each objective requires a certain number of images in different modes.
    - Each waypoint may require soil and/or rock samples.
    - Rovers need to be at specific waypoints to perform sampling and imaging.
    - Rovers need to be at a waypoint visible from the lander to communicate data.
    - Rovers have limited storage, so they may need to drop samples before taking more.

    # Heuristic Initialization
    - Extract the goal conditions to determine the objectives, modes, waypoints, and data types that need to be communicated.
    - Extract static facts about the environment, such as waypoint locations, visibility, traversal capabilities, and equipment.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check if the current state satisfies the goal. If so, return 0.
    3. Count the number of uncommunicated image data goals. For each uncommunicated image:
       - Find a rover that can take the image.
       - Estimate the cost to navigate to a waypoint visible from the objective.
       - Add the cost of calibrating the camera (if not already calibrated).
       - Add the cost of taking the image.
       - Add the cost of navigating to a waypoint visible from the lander.
       - Add the cost of communicating the image data.
    4. Count the number of uncommunicated soil data goals. For each uncommunicated soil sample:
       - Find a rover that can sample soil.
       - Estimate the cost to navigate to the soil sample location.
       - Add the cost of sampling the soil.
       - Add the cost of navigating to a waypoint visible from the lander.
       - Add the cost of communicating the soil data.
    5. Count the number of uncommunicated rock data goals. For each uncommunicated rock sample:
       - Find a rover that can sample rock.
       - Estimate the cost to navigate to the rock sample location.
       - Add the cost of sampling the rock.
       - Add the cost of navigating to a waypoint visible from the lander.
       - Add the cost of communicating the rock data.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract relevant information from static facts
        self.at_lander = next(
            (get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None
        )
        self.rovers = {get_parts(fact)[1] for fact in static_facts if match(fact, "*", "* - rover")}
        self.waypoints = {get_parts(fact)[1] for fact in static_facts if match(fact, "*", "* - waypoint")}
        self.objectives = {get_parts(fact)[1] for fact in static_facts if match(fact, "*", "* - objective")}
        self.cameras = {get_parts(fact)[1] for fact in static_facts if match(fact, "*", "* - camera")}
        self.stores = {get_parts(fact)[1] for fact in static_facts if match(fact, "*", "* - store")}
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
            (get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3])
            for fact in static_facts
            if match(fact, "can_traverse", "*", "*", "*")
        }
        self.calibration_target = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in static_facts
            if match(fact, "calibration_target", "*", "*")
        }
        self.on_board = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "on_board", "*", "*")
        }
        self.supports = {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static_facts if match(fact, "supports", "*", "*")
        }
        self.visible_from = {
            (get_parts(fact)[1], get_parts(fact)[2])
            for fact in static_facts
            if match(fact, "visible_from", "*", "*")
        }
        self.at_soil_sample = {get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")}
        self.at_rock_sample = {get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")}

    def __call__(self, node):
        """
        Estimate the minimum cost to achieve the goals from the current state.
        """
        state = node.state

        # If the goal is reached, the heuristic value is 0
        if self.goals <= state:
            return 0

        total_cost = 0

        # 1. Image Data Goals
        uncommunicated_images = []
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                if goal not in state:
                    uncommunicated_images.append(goal)

        for image_goal in uncommunicated_images:
            _, objective, mode = get_parts(image_goal)
            best_rover = None
            min_image_cost = float('inf')

            for rover in self.rovers:
                if rover not in [get_parts(fact)[1] for fact in state if match(fact, "equipped_for_imaging", "*")]:
                    continue

                for camera in self.cameras:
                    if (camera, rover) not in self.on_board:
                        continue
                    if (camera, mode) not in self.supports:
                        continue

                    # Find a waypoint visible from the objective
                    visible_waypoints = [
                        waypoint for waypoint in self.waypoints if (objective, waypoint) in self.visible_from
                    ]
                    if not visible_waypoints:
                        continue

                    # Find the rover's current location
                    rover_location = next(
                        (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                    )
                    if not rover_location:
                        continue

                    # Calculate navigation cost to the closest visible waypoint
                    navigation_cost = float('inf')
                    for visible_waypoint in visible_waypoints:
                        if (rover, rover_location, visible_waypoint) in self.can_traverse:
                            navigation_cost = 1  # Simplified navigation cost
                            break
                    if navigation_cost == float('inf'):
                        continue

                    # Check if the camera is calibrated
                    calibrated = any(
                        match(fact, "calibrated", camera, rover) for fact in state
                    )

                    # Calculate the total cost for this rover
                    image_cost = navigation_cost + 1  # Navigate + Take Image
                    if not calibrated:
                        image_cost += 1  # Calibrate

                    # Find a waypoint visible from the lander
                    lander_visible_waypoints = [
                        waypoint for waypoint in self.waypoints if (rover_location, self.at_lander) in self.visible
                    ]
                    if not lander_visible_waypoints:
                        image_cost += 1 #Navigate to lander
                    image_cost += 1  # Communicate

                    if image_cost < min_image_cost:
                        min_image_cost = image_cost
                        best_rover = rover

            if best_rover:
                total_cost += min_image_cost

        # 2. Soil Data Goals
        uncommunicated_soil = []
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                if goal not in state:
                    uncommunicated_soil.append(goal)

        for soil_goal in uncommunicated_soil:
            _, waypoint = get_parts(soil_goal)
            best_rover = None
            min_soil_cost = float('inf')

            for rover in self.rovers:
                if rover not in [get_parts(fact)[1] for fact in state if match(fact, "equipped_for_soil_analysis", "*")]:
                    continue

                # Find the rover's current location
                rover_location = next(
                    (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                )
                if not rover_location:
                    continue

                # Calculate navigation cost to the soil sample location
                navigation_cost = float('inf')
                if (rover, rover_location, waypoint) in self.can_traverse:
                    navigation_cost = 1  # Simplified navigation cost
                if navigation_cost == float('inf'):
                    continue

                soil_cost = navigation_cost + 1  # Navigate + Sample Soil

                # Find a waypoint visible from the lander
                lander_visible_waypoints = [
                    waypoint for waypoint in self.waypoints if (rover_location, self.at_lander) in self.visible
                ]
                if not lander_visible_waypoints:
                    soil_cost += 1 #Navigate to lander
                soil_cost += 1  # Communicate

                if soil_cost < min_soil_cost:
                    min_soil_cost = soil_cost
                    best_rover = rover

            if best_rover:
                total_cost += min_soil_cost

        # 3. Rock Data Goals
        uncommunicated_rock = []
        for goal in self.goals:
            if match(goal, "communicated_rock_data", "*"):
                if goal not in state:
                    uncommunicated_rock.append(goal)

        for rock_goal in uncommunicated_rock:
            _, waypoint = get_parts(rock_goal)
            best_rover = None
            min_rock_cost = float('inf')

            for rover in self.rovers:
                if rover not in [get_parts(fact)[1] for fact in state if match(fact, "equipped_for_rock_analysis", "*")]:
                    continue

                # Find the rover's current location
                rover_location = next(
                    (get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None
                )
                if not rover_location:
                    continue

                # Calculate navigation cost to the rock sample location
                navigation_cost = float('inf')
                if (rover, rover_location, waypoint) in self.can_traverse:
                    navigation_cost = 1  # Simplified navigation cost
                if navigation_cost == float('inf'):
                    continue

                rock_cost = navigation_cost + 1  # Navigate + Sample Rock

                # Find a waypoint visible from the lander
                lander_visible_waypoints = [
                    waypoint for waypoint in self.waypoints if (rover_location, self.at_lander) in self.visible
                ]
                if not lander_visible_waypoints:
                    rock_cost += 1 #Navigate to lander
                rock_cost += 1  # Communicate

                if rock_cost < min_rock_cost:
                    min_rock_cost = rock_cost
                    best_rover = rover

            if best_rover:
                total_cost += min_rock_cost

        return total_cost
