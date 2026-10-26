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
    This heuristic estimates the number of actions required to achieve all goal predicates in the Rovers domain.
    It calculates the cost based on the remaining goal predicates and the necessary actions to achieve them,
    considering the current state and available capabilities.

    # Assumptions:
    - The heuristic assumes that for each goal predicate, the rover will need to perform a sequence of actions:
        navigation (if needed), data acquisition (sampling or imaging), and communication.
    - It simplifies the problem by not explicitly planning navigation paths or considering resource constraints like store capacity.
    - It assumes that each goal predicate is independent and the total cost is the sum of the costs for each goal.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - It also extracts static information about equipped rovers, calibration targets, and supported camera modes to efficiently check preconditions later.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate that is not satisfied in the current state:
    1. If the goal is `communicated_soil_data(?waypoint)`:
        - Check if `have_soil_analysis` for the waypoint exists in the state.
        - If not, estimate 2 actions: `sample_soil` and `communicate_soil_data`.
        - If yes, estimate 1 action: `communicate_soil_data`.
    2. If the goal is `communicated_rock_data(?waypoint)`:
        - Check if `have_rock_analysis` for the waypoint exists in the state.
        - If not, estimate 2 actions: `sample_rock` and `communicate_rock_data`.
        - If yes, estimate 1 action: `communicate_rock_data`.
    3. If the goal is `communicated_image_data(?objective, ?mode)`:
        - Check if `have_image` for the objective and mode exists in the state.
        - If not, check if `calibrated` for any camera on any rover exists in the state.
            - If not calibrated, estimate 3 actions: `calibrate`, `take_image`, and `communicate_image_data`.
            - If calibrated, estimate 2 actions: `take_image` and `communicate_image_data`.
        - If yes (have_image exists), estimate 1 action: `communicate_image_data`.
    4. Sum up the estimated action counts for all unsatisfied goal predicates to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the rovers heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.equipped_for_soil_analysis_rovers = set()
        self.equipped_for_rock_analysis_rovers = set()
        self.equipped_for_imaging_rovers = set()
        self.calibration_targets = {}  # camera -> objective
        self.camera_supports_modes = {} # camera -> set of modes

        for fact in static_facts:
            if match(fact, 'equipped_for_soil_analysis', '*'):
                self.equipped_for_soil_analysis_rovers.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                self.equipped_for_rock_analysis_rovers.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_imaging', '*'):
                self.equipped_for_imaging_rovers.add(get_parts(fact)[1])
            elif match(fact, 'calibration_target', '*', '*'):
                camera, objective = get_parts(fact)[1], get_parts(fact)[2]
                self.calibration_targets[camera] = objective
            elif match(fact, 'supports', '*', '*'):
                camera, mode = get_parts(fact)[1], get_parts(fact)[2]
                if camera not in self.camera_supports_modes:
                    self.camera_supports_modes[camera] = set()
                self.camera_supports_modes[camera].add(mode)


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        for goal in self.goals:
            if goal not in state:
                parts = get_parts(goal)
                goal_predicate = parts[0]

                if goal_predicate == 'communicated_soil_data':
                    waypoint = parts[1]
                    have_soil = False
                    for fact in state:
                        if match(fact, 'have_soil_analysis', '*', waypoint):
                            have_soil = True
                            break
                    if not have_soil:
                        heuristic_value += 2  # sample_soil + communicate_soil_data
                    else:
                        heuristic_value += 1  # communicate_soil_data

                elif goal_predicate == 'communicated_rock_data':
                    waypoint = parts[1]
                    have_rock = False
                    for fact in state:
                        if match(fact, 'have_rock_analysis', '*', waypoint):
                            have_rock = True
                            break
                    if not have_rock:
                        heuristic_value += 2  # sample_rock + communicate_rock_data
                    else:
                        heuristic_value += 1  # communicate_rock_data

                elif goal_predicate == 'communicated_image_data':
                    objective = parts[1]
                    mode = parts[2]
                    have_image = False
                    for fact in state:
                        if match(fact, 'have_image', '*', objective, mode):
                            have_image = True
                            break
                    if not have_image:
                        is_calibrated = False
                        for fact in state:
                            if match(fact, 'calibrated', '*', '*'):
                                is_calibrated = True
                                break
                        if not is_calibrated:
                            heuristic_value += 3 # calibrate + take_image + communicate_image_data
                        else:
                            heuristic_value += 2 # take_image + communicate_image_data
                    else:
                        heuristic_value += 1  # communicate_image_data

        return heuristic_value
