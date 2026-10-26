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


class rovers11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the following goals: communicated soil data, communicated rock data, and communicated image data.
    The heuristic estimates the cost based on the number of samples/images that still need to be communicated,
    the need to navigate to waypoints to take samples/images, and the need to navigate to the lander to communicate data.

    # Assumptions
    - Rovers have limited storage, but this heuristic does not explicitly model store capacity.
    - Rovers need to be at a waypoint visible from the lander to communicate data.
    - Rovers need to be at a waypoint where a sample/objective is located to sample/image it.
    - Calibration is always possible when needed.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Identify the locations of soil samples, rock samples, and objectives from the initial state.
    - Store the visibility information between waypoints and from objectives to waypoints.
    - Store the rover capabilities (equipped for soil, rock, imaging).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Check for uncommunicated soil data goals:
       - For each uncommunicated soil data goal, find the closest rover that has the soil analysis.
       - Estimate the cost as the sum of:
         - Cost to navigate to the waypoint with the soil sample.
         - Cost to navigate to a waypoint visible from the lander.
         - Cost to communicate the soil data.
    3. Check for uncommunicated rock data goals:
       - For each uncommunicated rock data goal, find the closest rover that has the rock analysis.
       - Estimate the cost as the sum of:
         - Cost to navigate to the waypoint with the rock sample.
         - Cost to navigate to a waypoint visible from the lander.
         - Cost to communicate the rock data.
    4. Check for uncommunicated image data goals:
       - For each uncommunicated image data goal, find the closest rover that has the image.
       - Estimate the cost as the sum of:
         - Cost to navigate to a waypoint visible from the objective.
         - Cost to take the image.
         - Cost to navigate to a waypoint visible from the lander.
         - Cost to communicate the image data.
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Extract soil and rock sample locations
        self.soil_samples = {get_parts(fact)[1] for fact in initial_state if match(fact, "at_soil_sample", "*")}
        self.rock_samples = {get_parts(fact)[1] for fact in initial_state if match(fact, "at_rock_sample", "*")}

        # Extract rover capabilities
        self.rovers_equipped_for_soil = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")}
        self.rovers_equipped_for_rock = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*")}
        self.rovers_equipped_for_imaging = {get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")}

        # Extract visibility information
        self.visible = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1, waypoint2 = get_parts(fact)[1], get_parts(fact)[2]
                if waypoint1 not in self.visible:
                    self.visible[waypoint1] = set()
                self.visible[waypoint1].add(waypoint2)

        self.visible_from = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1], get_parts(fact)[2]
                if objective not in self.visible_from:
                    self.visible_from[objective] = set()
                self.visible_from[objective].add(waypoint)

        # Extract lander location
        self.lander_location = next(get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*"))

    def __call__(self, node):
        """Estimate the number of actions required to achieve the goals."""
        state = node.state
        if task.goal_reached(state):
            return 0

        cost = 0

        # Uncommunicated soil data
        uncommunicated_soil = set()
        for fact in self.goals:
            if match(fact, "communicated_soil_data", "*") and fact not in state:
                uncommunicated_soil.add(get_parts(fact)[1])

        for soil_waypoint in uncommunicated_soil:
            # Find a rover that has the soil analysis
            available_rovers = set()
            for fact in state:
                if match(fact, "have_soil_analysis", "*", soil_waypoint):
                    rover = get_parts(fact)[1]
                    if rover in self.rovers_equipped_for_soil:
                        available_rovers.add(rover)

            if not available_rovers:
                # Find a rover that is equipped for soil analysis and is at the soil sample location
                for fact in state:
                    if match(fact, "at", "*", soil_waypoint):
                        rover = get_parts(fact)[1]
                        if rover in self.rovers_equipped_for_soil:
                            available_rovers.add(rover)
                if not available_rovers:
                    cost += 100 # Big penalty if no rover can sample soil at this waypoint
                    continue
                else:
                    cost += 1 # sampling action

            # Find a rover that can communicate the soil data
            rover_location = None
            for fact in state:
                for rover in available_rovers:
                    if match(fact, "at", rover, "*"):
                        rover_location = get_parts(fact)[2]
                        break
                if rover_location:
                    break

            if not rover_location:
                cost += 100 # Big penalty if no rover is available
                continue

            # Check if the rover is at a waypoint visible from the lander
            can_communicate = False
            if rover_location in self.visible and self.lander_location in self.visible[rover_location]:
                can_communicate = True

            if not can_communicate:
                cost += 2 # Moving to a waypoint visible from the lander

            cost += 1 # Communicate action

        # Uncommunicated rock data
        uncommunicated_rock = set()
        for fact in self.goals:
            if match(fact, "communicated_rock_data", "*") and fact not in state:
                uncommunicated_rock.add(get_parts(fact)[1])

        for rock_waypoint in uncommunicated_rock:
            # Find a rover that has the rock analysis
            available_rovers = set()
            for fact in state:
                if match(fact, "have_rock_analysis", "*", rock_waypoint):
                    rover = get_parts(fact)[1]
                    if rover in self.rovers_equipped_for_rock:
                        available_rovers.add(rover)

            if not available_rovers:
                # Find a rover that is equipped for rock analysis and is at the rock sample location
                for fact in state:
                    if match(fact, "at", "*", rock_waypoint):
                        rover = get_parts(fact)[1]
                        if rover in self.rovers_equipped_for_rock:
                            available_rovers.add(rover)
                if not available_rovers:
                    cost += 100 # Big penalty if no rover can sample rock at this waypoint
                    continue
                else:
                    cost += 1 # sampling action

            # Find a rover that can communicate the rock data
            rover_location = None
            for fact in state:
                for rover in available_rovers:
                    if match(fact, "at", rover, "*"):
                        rover_location = get_parts(fact)[2]
                        break
                if rover_location:
                    break

            if not rover_location:
                cost += 100 # Big penalty if no rover is available
                continue

            # Check if the rover is at a waypoint visible from the lander
            can_communicate = False
            if rover_location in self.visible and self.lander_location in self.visible[rover_location]:
                can_communicate = True

            if not can_communicate:
                cost += 2 # Moving to a waypoint visible from the lander

            cost += 1 # Communicate action

        # Uncommunicated image data
        uncommunicated_images = set()
        for fact in self.goals:
            if match(fact, "communicated_image_data", "*", "*") and fact not in state:
                objective = get_parts(fact)[1]
                uncommunicated_images.add(objective)

        for objective in uncommunicated_images:
            # Find a rover that has the image
            available_rovers = set()
            for fact in state:
                if match(fact, "have_image", "*", objective, "*"):
                    rover = get_parts(fact)[1]
                    if rover in self.rovers_equipped_for_imaging:
                        available_rovers.add(rover)

            if not available_rovers:
                # Find a rover that is equipped for imaging and is at a waypoint visible from the objective
                visible_waypoints = self.visible_from.get(objective, set())
                for fact in state:
                    if match(fact, "at", "*", "*"):
                        rover = get_parts(fact)[1]
                        waypoint = get_parts(fact)[2]
                        if rover in self.rovers_equipped_for_imaging and waypoint in visible_waypoints:
                            available_rovers.add(rover)
                if not available_rovers:
                    cost += 100 # Big penalty if no rover can take image of this objective
                    continue
                else:
                    cost += 2 # Moving and taking image

            # Find a rover that can communicate the image data
            rover_location = None
            for fact in state:
                for rover in available_rovers:
                    if match(fact, "at", rover, "*"):
                        rover_location = get_parts(fact)[2]
                        break
                if rover_location:
                    break

            if not rover_location:
                cost += 100 # Big penalty if no rover is available
                continue

            # Check if the rover is at a waypoint visible from the lander
            can_communicate = False
            if rover_location in self.visible and self.lander_location in self.visible[rover_location]:
                can_communicate = True

            if not can_communicate:
                cost += 2 # Moving to a waypoint visible from the lander

            cost += 1 # Communicate action

        return cost
