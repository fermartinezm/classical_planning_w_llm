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


class rovers6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the following goals: communicating soil data, rock data, and image data.
    The heuristic estimates the cost of achieving each goal and sums them up.

    # Assumptions
    - Rovers need to navigate to waypoints with samples or objectives.
    - Rovers need to be at a waypoint visible from the lander to communicate data.
    - Rovers need to calibrate cameras before taking images.
    - Rovers need to sample soil/rock before communicating the data.

    # Heuristic Initialization
    - Extract the goal conditions from the task.
    - Extract static information such as waypoint adjacencies (can_traverse), visibility between waypoints,
      locations of soil and rock samples, and visibility from objectives.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Goal Decomposition:** Decompose the overall goal into individual sub-goals related to communicating
       soil data, rock data, and image data.

    2. **Soil Data Communication Cost:** For each waypoint with soil data to be communicated:
       - Find the rover that can perform soil analysis.
       - Estimate the cost for the rover to navigate to the soil sample location (if not already there).
       - Estimate the cost for the rover to sample the soil (if not already sampled).
       - Estimate the cost for the rover to navigate to a waypoint visible from the lander.
       - Add 1 for the communicate_soil_data action.

    3. **Rock Data Communication Cost:** For each waypoint with rock data to be communicated:
       - Find the rover that can perform rock analysis.
       - Estimate the cost for the rover to navigate to the rock sample location (if not already there).
       - Estimate the cost for the rover to sample the rock (if not already sampled).
       - Estimate the cost for the rover to navigate to a waypoint visible from the lander.
       - Add 1 for the communicate_rock_data action.

    4. **Image Data Communication Cost:** For each objective and mode combination:
       - Find the rover equipped for imaging.
       - Estimate the cost for the rover to navigate to a waypoint visible from the objective.
       - Estimate the cost to calibrate the camera (if not already calibrated).
       - Add 1 for the take_image action.
       - Estimate the cost for the rover to navigate to a waypoint visible from the lander.
       - Add 1 for the communicate_image_data action.

    5. **Summation:** Sum up the estimated costs for all sub-goals to obtain the overall heuristic value.
       If the state is a goal state, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.can_traverse = {}
        self.visible = {}
        self.at_soil_sample = set()
        self.at_rock_sample = set()
        self.visible_from = {}
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()
        self.calibration_targets = {}
        self.on_board = {}
        self.supports = {}
        self.lander_location = None

        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                rover, waypoint1, waypoint2 = get_parts(fact)[1:]
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = []
                self.can_traverse[rover].append((waypoint1, waypoint2))
            elif match(fact, "visible", "*", "*"):
                waypoint1, waypoint2 = get_parts(fact)[1:]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = []
                self.visible[waypoint1].append(waypoint2)
            elif match(fact, "at_soil_sample", "*"):
                waypoint = get_parts(fact)[1]
                self.at_soil_sample.add(waypoint)
            elif match(fact, "at_rock_sample", "*"):
                waypoint = get_parts(fact)[1]
                self.at_rock_sample.add(waypoint)
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1:]
                if objective not in self.visible_from:
                    self.visible_from[objective] = []
                self.visible_from[objective].append(waypoint)
            elif match(fact, "equipped_for_soil_analysis", "*"):
                rover = get_parts(fact)[1]
                self.equipped_for_soil_analysis.add(rover)
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = get_parts(fact)[1]
                self.equipped_for_rock_analysis.add(rover)
            elif match(fact, "equipped_for_imaging", "*"):
                rover = get_parts(fact)[1]
                self.equipped_for_imaging.add(rover)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_parts(fact)[1:]
                self.calibration_targets[camera] = objective
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1:]
                if rover not in self.on_board:
                    self.on_board[rover] = []
                self.on_board[rover].append(camera)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1:]
                if camera not in self.supports:
                    self.supports[camera] = []
                self.supports[camera].append(mode)
            elif match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state

        if self.goals <= state:
            return 0

        total_cost = 0

        # Soil data communication goals
        soil_goals = [get_parts(goal)[1] for goal in self.goals if match(goal, "communicated_soil_data", "*")]
        for waypoint in soil_goals:
            if f"(communicated_soil_data {waypoint})" not in state:
                rover = next((r for r in self.equipped_for_soil_analysis if f"(at {r} *)" in state), None)
                if not rover:
                    continue

                rover_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
                if not rover_location:
                    continue

                cost = 0

                # Navigate to soil sample location
                if f"(at_soil_sample {waypoint})" in self.static and f"(have_soil_analysis {rover} {waypoint})" not in state:
                    if rover_location != waypoint:
                        cost += 1  # Assuming 1 action to navigate

                    # Sample soil
                    if f"(have_soil_analysis {rover} {waypoint})" not in state:
                        cost += 1

                    # Navigate to lander
                    lander_visible = False
                    for visible_waypoint in self.visible.get(rover_location, []):
                        if visible_waypoint == self.lander_location:
                            lander_visible = True
                            break
                    if not lander_visible:
                        cost += 1  # Assuming 1 action to navigate

                    cost += 1  # Communicate soil data
                    total_cost += cost

        # Rock data communication goals
        rock_goals = [get_parts(goal)[1] for goal in self.goals if match(goal, "communicated_rock_data", "*")]
        for waypoint in rock_goals:
            if f"(communicated_rock_data {waypoint})" not in state:
                rover = next((r for r in self.equipped_for_rock_analysis if f"(at {r} *)" in state), None)
                if not rover:
                    continue

                rover_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
                if not rover_location:
                    continue

                cost = 0

                # Navigate to rock sample location
                if f"(at_rock_sample {waypoint})" in self.static and f"(have_rock_analysis {rover} {waypoint})" not in state:
                    if rover_location != waypoint:
                        cost += 1  # Assuming 1 action to navigate

                    # Sample rock
                    if f"(have_rock_analysis {rover} {waypoint})" not in state:
                        cost += 1

                    # Navigate to lander
                    lander_visible = False
                    for visible_waypoint in self.visible.get(rover_location, []):
                        if visible_waypoint == self.lander_location:
                            lander_visible = True
                            break
                    if not lander_visible:
                        cost += 1  # Assuming 1 action to navigate

                    cost += 1  # Communicate rock data
                    total_cost += cost

        # Image data communication goals
        image_goals = [(get_parts(goal)[1], get_parts(goal)[2]) for goal in self.goals if match(goal, "communicated_image_data", "*", "*")]
        for objective, mode in image_goals:
            if f"(communicated_image_data {objective} {mode})" not in state:
                rover = next((r for r in self.equipped_for_imaging if f"(at {r} *)" in state), None)
                if not rover:
                    continue

                rover_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
                if not rover_location:
                    continue

                cost = 0

                # Navigate to waypoint visible from objective
                if objective in self.visible_from:
                    objective_visible = False
                    for visible_waypoint in self.visible_from[objective]:
                        if rover_location == visible_waypoint:
                            objective_visible = True
                            break
                    if not objective_visible:
                        cost += 1  # Assuming 1 action to navigate

                    # Calibrate camera
                    camera = next((c for c in self.on_board.get(rover, []) if self.calibration_targets.get(c) == objective), None)
                    if camera and f"(calibrated {camera} {rover})" not in state:
                        cost += 1

                    # Take image
                    cost += 1

                    # Navigate to lander
                    lander_visible = False
                    for visible_waypoint in self.visible.get(rover_location, []):
                        if visible_waypoint == self.lander_location:
                            lander_visible = True
                            break
                    if not lander_visible:
                        cost += 1  # Assuming 1 action to navigate

                    # Communicate image data
                    cost += 1
                    total_cost += cost

        return total_cost
