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


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions in the rovers domain.
    It considers the necessary steps for each goal type: communicating soil data, rock data, and image data.
    The heuristic sums up the estimated costs for each unsatisfied goal, providing a simple and efficient estimate.

    # Assumptions:
    - Rovers can always navigate to any visible waypoint if needed. Navigation cost is assumed to be 1 action per waypoint move.
    - Communication is always possible if a rover is at a waypoint and the lander is at another waypoint, and they are visible to each other. Communication cost is assumed to be 1 action.
    - Sampling (soil or rock) and imaging require the rover to be at the correct waypoint and have the necessary equipment. These actions are assumed to cost 1 action each.
    - Calibration is required before taking an image and is assumed to cost 1 action, plus navigation to a suitable waypoint if needed (navigation cost is included in other steps).
    - Store capacity and other resource constraints are not explicitly considered in this simplified heuristic.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal conditions from the task.
    - It also extracts static information about equipment capabilities, store assignments,
      camera configurations, visibility relations, and traversal capabilities to be used in heuristic calculations.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition, the heuristic estimates the minimum number of actions needed to satisfy it if it's not already satisfied in the current state.
    The total heuristic value is the sum of the estimated costs for all unsatisfied goal conditions.

    1. For each 'communicated_soil_data' goal (communicated_soil_data ?waypoint):
       - If the goal is not achieved:
         - Check if 'have_soil_analysis' is already done at the waypoint. If not, estimate actions for:
           - Navigate to a waypoint with a soil sample (estimated as 1 action, assuming reachability).
           - Sample soil (1 action).
         - Estimate actions for communication:
           - Navigate to a waypoint visible from the lander's waypoint (estimated as 1 action, assuming reachability).
           - Communicate soil data (1 action).

    2. For each 'communicated_rock_data' goal (communicated_rock_data ?waypoint):
       - Similar to 'communicated_soil_data', but for rock samples and rock analysis equipment.

    3. For each 'communicated_image_data' goal (communicated_image_data ?objective ?mode):
       - If the goal is not achieved:
         - Check if 'have_image' is already taken for the objective and mode. If not, estimate actions for:
           - Calibration (if not already calibrated for a suitable camera):
             - Calibrate camera (1 action).
           - Navigate to a waypoint visible from the objective (estimated as 1 action, assuming reachability).
           - Take image (1 action).
         - Estimate actions for communication:
           - Navigate to a waypoint visible from the lander's waypoint (estimated as 1 action, assuming reachability).
           - Communicate image data (1 action).

    4. Sum up the estimated action counts for all unsatisfied goals to get the final heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the rovers heuristic. Extracts goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.soil_samples = set()
        self.rock_samples = set()
        self.imaging_rovers = set()
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.rover_stores = {}
        self.cameras_on_rovers = {}
        self.camera_supports_modes = {}
        self.calibration_targets = {}
        self.lander_location = None

        for fact in static_facts:
            if match(fact, "at_soil_sample", "*"):
                self.soil_samples.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.rock_samples.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.imaging_rovers.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.soil_rovers.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rock_rovers.add(get_parts(fact)[1])
            elif match(fact, "store_of", "*", "*"):
                self.rover_stores[get_parts(fact)[2]] = get_parts(fact)[1]
            elif match(fact, "on_board", "*", "*"):
                rover = get_parts(fact)[2]
                camera = get_parts(fact)[1]
                if rover not in self.cameras_on_rovers:
                    self.cameras_on_rovers[rover] = []
                self.cameras_on_rovers[rover].append(camera)
            elif match(fact, "supports", "*", "*"):
                camera = get_parts(fact)[1]
                mode = get_parts(fact)[2]
                if camera not in self.camera_supports_modes:
                    self.camera_supports_modes[camera] = []
                self.camera_supports_modes[camera].append(mode)
            elif match(fact, "calibration_target", "*", "*"):
                self.calibration_targets[get_parts(fact)[1]] = get_parts(fact)[2]
            elif match(fact, "at_lander", "*", "*"):
                self.lander_location = get_parts(fact)[2]


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        goal_achieved = True
        for goal in self.goals:
            if goal not in state:
                goal_achieved = False
                break
        if goal_achieved:
            return 0

        for goal in self.goals:
            if goal not in state:
                if match(goal, "communicated_soil_data", "*"):
                    waypoint = get_parts(goal)[1]
                    if not any(match(fact, "have_soil_analysis", "*", waypoint) for fact in state):
                        heuristic_value += 2  # sample_soil + navigate to sample
                    heuristic_value += 2      # communicate_soil_data + navigate to communication point

                elif match(goal, "communicated_rock_data", "*"):
                    waypoint = get_parts(goal)[1]
                    if not any(match(fact, "have_rock_analysis", "*", waypoint) for fact in state):
                        heuristic_value += 2  # sample_rock + navigate to sample
                    heuristic_value += 2      # communicate_rock_data + navigate to communication point

                elif match(goal, "communicated_image_data", "*", "*"):
                    objective = get_parts(goal)[1]
                    mode = get_parts(goal)[2]
                    if not any(match(fact, "have_image", "*", objective, mode) for fact in state):
                        heuristic_value += 1  # calibrate (assuming needed)
                        heuristic_value += 2  # take_image + navigate to imaging point
                    heuristic_value += 2      # communicate_image_data + navigate to communication point
        return heuristic_value
