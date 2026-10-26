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


class rovers2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve the goals in the rovers domain.
    It considers the number of images, soil samples, and rock samples that need to be communicated,
    as well as the need to navigate to locations to perform these actions.

    # Assumptions
    - Each objective requires a certain number of images in different modes.
    - Each waypoint may require soil and/or rock samples.
    - The rover needs to be at a waypoint to sample or communicate data.
    - The rover needs to be at a waypoint visible from an objective to take an image.
    - The rover needs to be at a waypoint visible from the lander to communicate data.
    - Calibrating a camera is required before taking images.

    # Heuristic Initialization
    - Extract the goal conditions related to communicated data (images, soil, rock).
    - Identify the objectives and modes for image goals.
    - Identify the waypoints for soil and rock sample goals.
    - Store static information about visibility between waypoints and objectives/lander.
    - Store static information about rover capabilities (equipped for soil/rock analysis, imaging).
    - Store static information about camera calibration targets.
    - Store static information about cameras on board rovers and their supported modes.
    - Store static information about the location of the lander.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check if the current state satisfies the goal. If so, return 0.
    3. Count the number of uncommunicated image data goals.
       - For each uncommunicated image, check if the rover has the image.
       - If the rover doesn't have the image, estimate the cost to take the image:
         - Find a waypoint visible from the objective.
         - Estimate the cost to navigate to that waypoint.
         - Add 1 for the take_image action.
         - If the camera is not calibrated, add 1 for the calibrate action.
       - Estimate the cost to communicate the image data:
         - Find a waypoint visible from the lander.
         - Estimate the cost to navigate to that waypoint.
         - Add 1 for the communicate_image_data action.
    4. Count the number of uncommunicated soil data goals.
       - For each uncommunicated soil sample, check if the rover has the sample.
       - If the rover doesn't have the sample, estimate the cost to sample:
         - Estimate the cost to navigate to the waypoint with the soil sample.
         - Add 1 for the sample_soil action.
       - Estimate the cost to communicate the soil data:
         - Find a waypoint visible from the lander.
         - Estimate the cost to navigate to that waypoint.
         - Add 1 for the communicate_soil_data action.
    5. Count the number of uncommunicated rock data goals.
       - This step is similar to the soil data goals.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal information
        self.image_goals = set()
        self.soil_goals = set()
        self.rock_goals = set()
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                self.image_goals.add(goal)
            elif match(goal, "communicated_soil_data", "*"):
                self.soil_goals.add(goal)
            elif match(goal, "communicated_rock_data", "*"):
                self.rock_goals.add(goal)

        # Extract static information
        self.visible = {}
        self.can_traverse = {}
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()
        self.calibration_targets = {}
        self.on_board = {}
        self.supports = {}
        self.lander_location = None
        self.visible_from = {}

        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = set()
                self.visible[waypoint1].add(waypoint2)
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover = get_parts(fact)[1]
                waypoint1 = get_parts(fact)[2]
                waypoint2 = get_parts(fact)[3]
                if (rover, waypoint1) not in self.can_traverse:
                    self.can_traverse[(rover, waypoint1)] = set()
                self.can_traverse[(rover, waypoint1)].add(waypoint2)
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
            elif match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state
        if self.goals <= state:
            return 0

        total_cost = 0

        # Track rover locations
        rover_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                rover = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                rover_locations[rover] = waypoint

        # Helper function to estimate navigation cost (simplified)
        def estimate_navigation_cost(rover, start, end):
            if (rover, start) in self.can_traverse and end in self.can_traverse[(rover, start)]:
                return 1
            else:
                return 2  # Assume at least 2 moves if not directly traversable

        # Image data goals
        for goal in self.image_goals:
            objective = get_parts(goal)[1]
            mode = get_parts(goal)[2]
            if goal not in state:
                rover = None
                camera = None
                for cam, rov in self.on_board.items():
                    if rov in rover_locations and objective in self.calibration_targets and self.calibration_targets[cam] == objective and mode in self.supports.get(cam, set()):
                        rover = rov
                        camera = cam
                        break

                if rover is None:
                    total_cost += 10 # Big penalty if no rover can take the image
                    continue

                if f"(have_image {rover} {objective} {mode})" not in state:
                    # Find a waypoint visible from the objective
                    best_waypoint = None
                    if objective in self.visible_from:
                        for waypoint in self.visible_from[objective]:
                            best_waypoint = waypoint
                            break
                    if best_waypoint is None:
                        total_cost += 5 # Big penalty if no waypoint is visible from the objective
                        continue

                    # Estimate cost to navigate to the waypoint
                    total_cost += estimate_navigation_cost(rover, rover_locations[rover], best_waypoint)

                    # Add cost for taking the image
                    total_cost += 1

                    # Add cost for calibrating the camera if needed
                    if f"(calibrated {camera} {rover})" not in state:
                        total_cost += 1

                # Estimate cost to communicate the image data
                best_waypoint = None
                if self.lander_location in self.visible:
                    for waypoint in self.visible[self.lander_location]:
                        best_waypoint = waypoint
                        break
                if best_waypoint is None:
                    total_cost += 5 # Big penalty if no waypoint is visible from the lander
                    continue

                total_cost += estimate_navigation_cost(rover, rover_locations[rover], self.lander_location)

                # Add cost for communicating the image data
                total_cost += 1

        # Soil data goals
        for goal in self.soil_goals:
            waypoint = get_parts(goal)[1]
            if goal not in state:
                rover = None
                for rov in rover_locations:
                    if rov in self.equipped_for_soil_analysis:
                        rover = rov
                        break
                if rover is None:
                    total_cost += 10 # Big penalty if no rover can sample soil
                    continue

                if f"(have_soil_analysis {rover} {waypoint})" not in state:
                    # Estimate cost to navigate to the waypoint
                    total_cost += estimate_navigation_cost(rover, rover_locations[rover], waypoint)

                    # Add cost for sampling the soil
                    total_cost += 1

                # Estimate cost to communicate the soil data
                total_cost += estimate_navigation_cost(rover, rover_locations[rover], self.lander_location)

                # Add cost for communicating the soil data
                total_cost += 1

        # Rock data goals (similar to soil data)
        for goal in self.rock_goals:
            waypoint = get_parts(goal)[1]
            if goal not in state:
                rover = None
                for rov in rover_locations:
                    if rov in self.equipped_for_rock_analysis:
                        rover = rov
                        break
                if rover is None:
                    total_cost += 10 # Big penalty if no rover can sample rock
                    continue

                if f"(have_rock_analysis {rover} {waypoint})" not in state:
                    # Estimate cost to navigate to the waypoint
                    total_cost += estimate_navigation_cost(rover, rover_locations[rover], waypoint)

                    # Add cost for sampling the rock
                    total_cost += 1

                # Estimate cost to communicate the rock data
                total_cost += estimate_navigation_cost(rover, rover_locations[rover], self.lander_location)

                # Add cost for communicating the rock data
                total_cost += 1

        return total_cost
