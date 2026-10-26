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


class rovers8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of soil samples, rock samples, and images that need to be collected and communicated,
    as well as the navigation actions required to reach the necessary waypoints. The heuristic prioritizes
    achieving the goals related to communicated data.

    # Assumptions
    - Rovers can perform actions concurrently.
    - The cost of each action is assumed to be 1.
    - The heuristic does not account for resource limitations (e.g., store capacity).
    - It assumes that all objectives are reachable from some waypoint.

    # Heuristic Initialization
    - Extract the goal conditions, including communicated soil data, rock data, and image data.
    - Identify the locations of soil samples, rock samples, and objectives.
    - Store the visibility relationships between waypoints and objectives.
    - Store the traversability relationships between waypoints.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Identify the uncommunicated soil samples. For each uncommunicated soil sample:
       - Find a rover that can sample soil and is at a waypoint with a soil sample.
       - Estimate the cost to navigate to a waypoint visible from the lander and communicate the data.
       - Add the estimated cost to the heuristic value.
    3. Identify the uncommunicated rock samples. For each uncommunicated rock sample:
       - Find a rover that can sample rock and is at a waypoint with a rock sample.
       - Estimate the cost to navigate to a waypoint visible from the lander and communicate the data.
       - Add the estimated cost to the heuristic value.
    4. Identify the uncommunicated images. For each uncommunicated image:
       - Find a rover that can take images and is at a waypoint visible from the objective.
       - Estimate the cost to calibrate the camera (if needed), take the image, navigate to a waypoint visible from the lander, and communicate the data.
       - Add the estimated cost to the heuristic value.
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract locations of soil and rock samples
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

        # Extract traversability relationships between waypoints
        self.can_traverse = {}
        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover = get_parts(fact)[1]
                waypoint1 = get_parts(fact)[2]
                waypoint2 = get_parts(fact)[3]
                if (rover, waypoint1) not in self.can_traverse:
                    self.can_traverse[(rover, waypoint1)] = set()
                self.can_traverse[(rover, waypoint1)].add(waypoint2)

        # Extract which rovers are equipped for soil analysis
        self.equipped_for_soil_analysis = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")
        }

        # Extract which rovers are equipped for rock analysis
        self.equipped_for_rock_analysis = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*")
        }

        # Extract which rovers are equipped for imaging
        self.equipped_for_imaging = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")
        }

        # Extract calibration targets
        self.calibration_targets = {}
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_targets[camera] = objective

        # Extract on_board cameras
        self.on_board = {}
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                camera = get_parts(fact)[1]
                rover = get_parts(fact)[2]
                if rover not in self.on_board:
                    self.on_board[rover] = set()
                self.on_board[rover].add(camera)

        # Extract supports relationships
        self.supports = {}
        for fact in static_facts:
            if match(fact, "supports", "*", "*"):
                camera = get_parts(fact)[1]
                mode = get_parts(fact)[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)

        # Extract lander location
        self.lander_location = next(get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*"))

        # Extract visibility between waypoints
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
        heuristic = 0

        # Helper function to check if a goal is achieved
        def is_goal_achieved(goal):
            return goal in state

        # Helper function to find a rover at a given location
        def find_rover_at(location):
            for fact in state:
                if match(fact, "at", "*", location):
                    return get_parts(fact)[1]
            return None

        # Helper function to find the location of a rover
        def find_rover_location(rover):
            for fact in state:
                if match(fact, "at", rover, "*"):
                    return get_parts(fact)[3]
            return None

        # Helper function to check if a camera is calibrated
        def is_calibrated(camera, rover):
            return f"(calibrated {camera} {rover})" in state

        # Helper function to find a path between two waypoints
        def find_path(rover, start, end):
            if (rover, start) in self.can_traverse and end in self.can_traverse[(rover, start)]:
                return 1
            else:
                return float('inf')

        # 1. Communicate Soil Data
        uncommunicated_soil_data = set()
        for sample in self.soil_samples:
            if not is_goal_achieved(f"(communicated_soil_data {sample})"):
                uncommunicated_soil_data.add(sample)

        for sample in uncommunicated_soil_data:
            rover_location = next((find_rover_location(rover) for rover in self.equipped_for_soil_analysis if find_rover_location(rover) and f"(at_soil_sample {sample})" in state and find_rover_location(rover) == sample), None)
            if rover_location:
                rover = find_rover_at(rover_location)
                # Find a path to a waypoint visible from the lander
                min_path_cost = float('inf')
                for waypoint in self.visible.get(rover_location, []):
                    if waypoint in self.visible.get(self.lander_location, []):
                        path_cost = find_path(rover, rover_location, waypoint)
                        min_path_cost = min(min_path_cost, path_cost)
                if min_path_cost != float('inf'):
                    heuristic += 1 + min_path_cost # 1 for communicate action

        # 2. Communicate Rock Data
        uncommunicated_rock_data = set()
        for sample in self.rock_samples:
            if not is_goal_achieved(f"(communicated_rock_data {sample})"):
                uncommunicated_rock_data.add(sample)

        for sample in uncommunicated_rock_data:
            rover_location = next((find_rover_location(rover) for rover in self.equipped_for_rock_analysis if find_rover_location(rover) and f"(at_rock_sample {sample})" in state and find_rover_location(rover) == sample), None)
            if rover_location:
                rover = find_rover_at(rover_location)
                # Find a path to a waypoint visible from the lander
                min_path_cost = float('inf')
                for waypoint in self.visible.get(rover_location, []):
                    if waypoint in self.visible.get(self.lander_location, []):
                        path_cost = find_path(rover, rover_location, waypoint)
                        min_path_cost = min(min_path_cost, path_cost)
                if min_path_cost != float('inf'):
                    heuristic += 1 + min_path_cost # 1 for communicate action

        # 3. Communicate Image Data
        uncommunicated_images = set()
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                if not is_goal_achieved(goal):
                    uncommunicated_images.add((objective, mode))

        for objective, mode in uncommunicated_images:
            # Find a rover that can take images of the objective
            possible_rovers = set()
            for rover in self.equipped_for_imaging:
                if rover in self.on_board:
                    for camera in self.on_board[rover]:
                        if camera in self.supports and mode in self.supports[camera] and objective in self.calibration_targets and self.calibration_targets[camera] == objective:
                            possible_rovers.add(rover)

            # Find a waypoint visible from the objective
            if objective in self.visible_from:
                waypoint_visible_from_objective = self.visible_from[objective]
                for rover in possible_rovers:
                    rover_location = find_rover_location(rover)
                    if rover_location in waypoint_visible_from_objective:
                        camera = next(camera for camera in self.on_board[rover] if camera in self.supports and mode in self.supports[camera] and objective in self.calibration_targets and self.calibration_targets[camera] == objective)
                        if not is_calibrated(camera, rover):
                            heuristic += 1 # Calibrate action
                        heuristic += 1 # Take image action

                        # Find a path to a waypoint visible from the lander
                        min_path_cost = float('inf')
                        for waypoint in self.visible.get(rover_location, []):
                            if waypoint in self.visible.get(self.lander_location, []):
                                path_cost = find_path(rover, rover_location, waypoint)
                                min_path_cost = min(min_path_cost, path_cost)
                        if min_path_cost != float('inf'):
                            heuristic += 1 + min_path_cost # 1 for communicate action

        return heuristic
