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


class RoversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions needed to achieve all goals by:
    1) Counting unsatisfied communication goals (soil, rock, image data)
    2) Estimating the steps required to:
       - Move to sample locations
       - Take samples (soil/rock)
       - Calibrate and take images
       - Communicate data to the lander

    # Assumptions:
    - Each unsatisfied goal requires at least one action (communication)
    - Moving between waypoints takes 1 action per step
    - Sampling, calibrating, and imaging each take 1 action
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal conditions (what needs to be communicated)
    - Extract static information about:
      - Waypoint connectivity (can_traverse)
      - Visible waypoints
      - Lander location
      - Camera capabilities
      - Sample locations

    # Step-By-Step Thinking for Computing Heuristic
    1) Identify unsatisfied communication goals:
       - For each uncommunicated soil/rock sample or image, add 1 to the heuristic
    2) For each unsatisfied goal:
       a) If it's soil/rock data:
          - Check if rover has the sample (if yes, just need to communicate)
          - If not, check if rover can collect it (needs to be at sample location with empty store)
          - Estimate movement cost to sample location if needed
       b) If it's image data:
          - Check if rover has the image (if yes, just need to communicate)
          - If not, check if camera is calibrated and can take the image
          - Estimate movement cost to visible waypoint if needed
    3) For communication:
       - Estimate movement cost to a waypoint visible to the lander
    4) Sum all estimated actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract lander location
        self.lander_location = None
        for fact in self.static:
            if match(fact, "at_lander", "*", "*"):
                _, lander, location = get_parts(fact)
                self.lander_location = location
                break

        # Build waypoint connectivity graph
        self.waypoint_graph = {}
        for fact in self.static:
            if match(fact, "can_traverse", "*", "*", "*"):
                _, rover, wp1, wp2 = get_parts(fact)
                if wp1 not in self.waypoint_graph:
                    self.waypoint_graph[wp1] = set()
                if wp2 not in self.waypoint_graph:
                    self.waypoint_graph[wp2] = set()
                self.waypoint_graph[wp1].add(wp2)
                self.waypoint_graph[wp2].add(wp1)

        # Extract sample locations
        self.rock_samples = set()
        self.soil_samples = set()
        for fact in self.static:
            if match(fact, "at_rock_sample", "*"):
                self.rock_samples.add(get_parts(fact)[1])
            elif match(fact, "at_soil_sample", "*"):
                self.soil_samples.add(get_parts(fact)[1])

        # Extract camera information
        self.camera_modes = {}
        self.calibration_targets = {}
        for fact in self.static:
            if match(fact, "supports", "*", "*"):
                _, camera, mode = get_parts(fact)
                if camera not in self.camera_modes:
                    self.camera_modes[camera] = set()
                self.camera_modes[camera].add(mode)
            elif match(fact, "calibration_target", "*", "*"):
                _, camera, objective = get_parts(fact)
                self.calibration_targets[camera] = objective

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the current state."""
        state = node.state
        heuristic_value = 0

        # Check if we've already reached the goal
        if self.goals <= state:
            return 0

        # Count unsatisfied communication goals
        uncommunicated_soil = set()
        uncommunicated_rock = set()
        uncommunicated_images = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "communicated_soil_data" and goal not in state:
                uncommunicated_soil.add(parts[1])
            elif parts[0] == "communicated_rock_data" and goal not in state:
                uncommunicated_rock.add(parts[1])
            elif parts[0] == "communicated_image_data" and goal not in state:
                uncommunicated_images.add((parts[1], parts[2]))

        # For each unsatisfied goal, estimate the required actions
        for wp in uncommunicated_soil:
            # Check if any rover has the sample
            has_sample = any(
                match(fact, "have_soil_analysis", "*", wp) for fact in state
            )
            if has_sample:
                heuristic_value += 1  # Just need to communicate
            else:
                # Need to sample and communicate (2 actions) plus movement
                heuristic_value += 2

        for wp in uncommunicated_rock:
            # Check if any rover has the sample
            has_sample = any(
                match(fact, "have_rock_analysis", "*", wp) for fact in state
            )
            if has_sample:
                heuristic_value += 1  # Just need to communicate
            else:
                # Need to sample and communicate (2 actions) plus movement
                heuristic_value += 2

        for obj, mode in uncommunicated_images:
            # Check if any rover has the image
            has_image = any(
                match(fact, "have_image", "*", obj, mode) for fact in state
            )
            if has_image:
                heuristic_value += 1  # Just need to communicate
            else:
                # Need to calibrate, take image, and communicate (3 actions) plus movement
                heuristic_value += 3

        # Add movement cost (estimate 1 per unsatisfied goal)
        heuristic_value += len(uncommunicated_soil) + len(uncommunicated_rock) + len(uncommunicated_images)

        return heuristic_value
