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


class rovers19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of communicated data goals (soil, rock, and image) that are not yet achieved,
    and estimates the cost based on the actions needed to sample, calibrate, take images, navigate, and communicate.

    # Assumptions
    - Each rover has the necessary equipment to perform its tasks.
    - Rovers can navigate between waypoints.
    - Communicating data requires the rover to be at a waypoint visible from the lander.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Identify the locations of soil and rock samples.
    - Identify the objectives and their required modes.
    - Store static information about visibility between waypoints and objectives.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize the heuristic value to 0.
    2.  Count the number of uncommunicated soil data goals. For each uncommunicated soil data goal:
        -   Estimate the cost to sample soil at the waypoint (if not already sampled).
        -   Estimate the cost to navigate to a waypoint visible from the lander.
        -   Add the cost to communicate the soil data.
    3.  Count the number of uncommunicated rock data goals. For each uncommunicated rock data goal:
        -   Estimate the cost to sample rock at the waypoint (if not already sampled).
        -   Estimate the cost to navigate to a waypoint visible from the lander.
        -   Add the cost to communicate the rock data.
    4.  Count the number of uncommunicated image data goals. For each uncommunicated image data goal:
        -   Estimate the cost to calibrate the camera (if not already calibrated).
        -   Estimate the cost to navigate to a waypoint visible from the objective.
        -   Estimate the cost to take the image.
        -   Estimate the cost to navigate to a waypoint visible from the lander.
        -   Add the cost to communicate the image data.
    5.  Return the total estimated cost.
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

        self.objective_waypoints = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.objective_waypoints:
                    self.objective_waypoints[objective] = set()
                self.objective_waypoints[objective].add(waypoint)

        self.waypoint_visibilities = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.waypoint_visibilities:
                    self.waypoint_visibilities[waypoint1] = set()
                self.waypoint_visibilities[waypoint1].add(waypoint2)

        self.lander_location = next(
            (get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None
        )

        self.calibration_targets = {}
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.calibration_targets[camera] = objective

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        if node.task.goal_reached(state):
            return 0

        cost = 0

        # Soil data communication goals
        uncommunicated_soil_data = set()
        for goal in self.goals:
            if match(goal, "communicated_soil_data", "*"):
                waypoint = get_parts(goal)[1]
                if goal not in state:
                    uncommunicated_soil_data.add(waypoint)

        for waypoint in uncommunicated_soil_data:
            if "(have_soil_analysis rover1 {})".format(waypoint) not in state:
                cost += 1  # Sample soil
            
            rover_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", "rover1", "*")), None)
            if rover_location:
                if self.lander_location:
                    if rover_location not in self.waypoint_visibilities or self.lander_location not in self.waypoint_visibilities[rover_location]:
                        cost += 1 # Navigate to waypoint visible from lander
            cost += 1  # Communicate soil data

        # Rock data communication goals
        uncommunicated_rock_data = set()
        for goal in self.goals:
            if match(goal, "communicated_rock_data", "*"):
                waypoint = get_parts(goal)[1]
                if goal not in state:
                    uncommunicated_rock_data.add(waypoint)

        for waypoint in uncommunicated_rock_data:
            if "(have_rock_analysis rover1 {})".format(waypoint) not in state:
                cost += 1  # Sample rock
            
            rover_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", "rover1", "*")), None)
            if rover_location:
                if self.lander_location:
                    if rover_location not in self.waypoint_visibilities or self.lander_location not in self.waypoint_visibilities[rover_location]:
                        cost += 1 # Navigate to waypoint visible from lander
            cost += 1  # Communicate rock data

        # Image data communication goals
        uncommunicated_image_data = set()
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective = get_parts(goal)[1]
                mode = get_parts(goal)[2]
                if goal not in state:
                    uncommunicated_image_data.add((objective, mode))

        for objective, mode in uncommunicated_image_data:
            camera = next((cam for cam, obj in self.calibration_targets.items() if obj == objective), None)
            if camera:
                if "(calibrated {} rover1)".format(camera) not in state:
                    cost += 1  # Calibrate camera

            if objective in self.objective_waypoints:
                
                rover_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", "rover1", "*")), None)
                if rover_location:
                    if rover_location not in self.objective_waypoints[objective]:
                        cost += 1  # Navigate to waypoint visible from objective
                cost += 1  # Take image
                
                rover_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", "rover1", "*")), None)
                if rover_location:
                    if self.lander_location:
                        if rover_location not in self.waypoint_visibilities or self.lander_location not in self.waypoint_visibilities[rover_location]:
                            cost += 1 # Navigate to waypoint visible from lander
                cost += 1  # Communicate image data

        return cost
