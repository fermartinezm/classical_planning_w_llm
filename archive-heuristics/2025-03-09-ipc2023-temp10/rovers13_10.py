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


class rovers13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of images to be taken and communicated, soil and rock samples to be collected and communicated,
    and the number of navigations required to reach the waypoints where these actions can be performed.

    # Assumptions
    - Each objective requires a certain number of images in different modes.
    - Each waypoint may require soil or rock samples.
    - Rovers need to be at specific waypoints to perform actions like sampling and imaging.
    - Rovers need to be at a waypoint visible to the lander to communicate data.
    - The heuristic assumes that each task (image, sample) requires a separate visit to the waypoint.

    # Heuristic Initialization
    - Extract the goal conditions.
    - Identify the objectives, modes, waypoints, and rovers involved in the goals.
    - Store static information about visibility, sample locations, and rover capabilities.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. For each goal condition:
       a. If the goal is to communicate image data:
          - Find a rover capable of imaging.
          - Find a waypoint visible from the objective and reachable by the rover.
          - Estimate the cost as the sum of:
            - Cost to navigate to the waypoint.
            - Cost to calibrate the camera (if not already calibrated).
            - Cost to take the image.
            - Cost to navigate to a waypoint visible from the lander.
            - Cost to communicate the image data.
       b. If the goal is to communicate soil data:
          - Find a rover equipped for soil analysis.
          - Find the waypoint with the soil sample.
          - Estimate the cost as the sum of:
            - Cost to navigate to the waypoint.
            - Cost to sample the soil.
            - Cost to navigate to a waypoint visible from the lander.
            - Cost to communicate the soil data.
       c. If the goal is to communicate rock data:
          - Find a rover equipped for rock analysis.
          - Find the waypoint with the rock sample.
          - Estimate the cost as the sum of:
            - Cost to navigate to the waypoint.
            - Cost to sample the rock.
            - Cost to navigate to a waypoint visible from the lander.
            - Cost to communicate the rock data.
    3. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.objectives = set()
        self.modes = set()
        self.soil_waypoints = set()
        self.rock_waypoints = set()
        self.rovers = set()
        self.cameras = set()

        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                self.objectives.add(get_parts(fact)[1])
            if match(fact, "supports", "*", "*"):
                self.modes.add(get_parts(fact)[2])
            if match(fact, "at_soil_sample", "*"):
                self.soil_waypoints.add(get_parts(fact)[1])
            if match(fact, "at_rock_sample", "*"):
                self.rock_waypoints.add(get_parts(fact)[1])
            if match(fact, "equipped_for_imaging", "*"):
                self.rovers.add(get_parts(fact)[1])
            if match(fact, "on_board", "*", "*"):
                self.cameras.add(get_parts(fact)[1])

        self.visible_from = {}
        for fact in self.static:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)

        self.visible = {}
        for fact in self.static:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = set()
                self.visible[waypoint1].add(waypoint2)

        self.equipped_for_imaging = set()
        for fact in self.static:
            if match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging.add(get_parts(fact)[1])

        self.equipped_for_soil_analysis = set()
        for fact in self.static:
            if match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil_analysis.add(get_parts(fact)[1])

        self.equipped_for_rock_analysis = set()
        for fact in self.static:
            if match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock_analysis.add(get_parts(fact)[1])

        self.calibration_target = {}
        for fact in self.static:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_target[camera] = objective

        self.on_board = {}
        for fact in self.static:
            if match(fact, "on_board", "*", "*"):
                camera = get_parts(fact)[1]
                rover = get_parts(fact)[2]
                self.on_board[camera] = rover

        self.supports = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                camera = get_parts(fact)[1]
                mode = get_parts(fact)[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)

        self.can_traverse = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover = get_parts(fact)[1]
                waypoint1 = get_parts(fact)[2]
                waypoint2 = get_parts(fact)[3]
                if (rover, waypoint1) not in self.can_traverse:
                    self.can_traverse[(rover, waypoint1)] = set()
                self.can_traverse[(rover, waypoint1)].add(waypoint2)

    def __call__(self, node):
        """Estimate the minimum cost to achieve the goal state."""
        state = node.state
        if self.is_goal(state):
            return 0

        total_cost = 0

        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]

                # Find a rover capable of imaging
                for rover in self.rovers:
                    if rover in self.equipped_for_imaging:
                        # Find a camera on board the rover
                        for camera, r in self.on_board.items():
                            if r == rover and camera in self.supports and mode in self.supports[camera]:
                                # Find a waypoint visible from the objective
                                if objective in self.visible_from:
                                    for waypoint in self.visible_from[objective]:
                                        # Check if the rover is at the waypoint
                                        rover_at_waypoint = False
                                        for fact in state:
                                            if match(fact, "at", rover, waypoint):
                                                rover_at_waypoint = True
                                                break

                                        if rover_at_waypoint:
                                            # Check if the camera is calibrated
                                            camera_calibrated = False
                                            for fact in state:
                                                if match(fact, "calibrated", camera, rover):
                                                    camera_calibrated = True
                                                    break

                                            # Find a waypoint visible from the lander
                                            lander_waypoint = None
                                            rover_waypoint = None
                                            for fact in state:
                                                if match(fact, "at", rover, "*"):
                                                    rover_waypoint = get_parts(fact)[2]
                                                    break

                                            if rover_waypoint in self.visible:
                                                for lander_waypoint in self.visible[rover_waypoint]:
                                                    break

                                            if lander_waypoint:
                                                cost = 1  # Communicate image data
                                                if not camera_calibrated:
                                                    cost += 1  # Calibrate
                                                cost += 1  # Take image
                                                total_cost += cost
                                                break
                                    else:
                                        total_cost += 2 # Navigate + other actions
                                        break
                                break
                        break

            elif match(goal, "communicated_soil_data", "*"):
                waypoint = get_parts(goal)[1]

                # Find a rover equipped for soil analysis
                for rover in self.rovers:
                    if rover in self.equipped_for_soil_analysis:
                        # Check if the rover is at the waypoint
                        rover_at_waypoint = False
                        for fact in state:
                            if match(fact, "at", rover, waypoint):
                                rover_at_waypoint = True
                                break

                        if rover_at_waypoint:
                            # Find a waypoint visible from the lander
                            lander_waypoint = None
                            rover_waypoint = None
                            for fact in state:
                                if match(fact, "at", rover, "*"):
                                    rover_waypoint = get_parts(fact)[2]
                                    break

                            if rover_waypoint in self.visible:
                                for lander_waypoint in self.visible[rover_waypoint]:
                                    break

                            if lander_waypoint:
                                total_cost += 2  # Sample soil + communicate
                                break
                    else:
                        total_cost += 2 # Navigate + other actions
                        break

            elif match(goal, "communicated_rock_data", "*"):
                waypoint = get_parts(goal)[1]

                # Find a rover equipped for rock analysis
                for rover in self.rovers:
                    if rover in self.equipped_for_rock_analysis:
                        # Check if the rover is at the waypoint
                        rover_at_waypoint = False
                        for fact in state:
                            if match(fact, "at", rover, waypoint):
                                rover_at_waypoint = True
                                break

                        if rover_at_waypoint:
                            # Find a waypoint visible from the lander
                            lander_waypoint = None
                            rover_waypoint = None
                            for fact in state:
                                if match(fact, "at", rover, "*"):
                                    rover_waypoint = get_parts(fact)[2]
                                    break

                            if rover_waypoint in self.visible:
                                for lander_waypoint in self.visible[rover_waypoint]:
                                    break

                            if lander_waypoint:
                                total_cost += 2  # Sample rock + communicate
                                break
                    else:
                        total_cost += 2 # Navigate + other actions
                        break

        return total_cost

    def is_goal(self, state):
        """Check if the given state is a goal state."""
        return self.goals <= state
