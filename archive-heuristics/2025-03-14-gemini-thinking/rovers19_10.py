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
    It focuses on the necessary steps to communicate data (soil, rock, image) for each objective and waypoint specified in the goal.
    The heuristic considers the actions needed to sample, calibrate, take images, and communicate, but simplifies navigation costs.

    # Assumptions:
    - Each goal fact (communicated data) is considered independently.
    - Navigation costs are roughly estimated and not based on shortest paths.
    - Rovers are assumed to have the necessary equipment if the problem instance allows for a solution.
    - Stores are always available and empty when needed for sampling.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal facts from the task definition.
    - It also extracts static information about sample locations (soil and rock), objectives visible from waypoints,
      calibration targets, and camera support for different modes.
    - This pre-processed information helps in quickly assessing the state and estimating costs.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact in the goal state:
    1. Check if the goal fact is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If not achieved, estimate the minimum actions required to achieve it based on the type of goal:
       - For `communicated_soil_data(waypoint)`:
         - If `communicated_soil_data(waypoint)` is not achieved, add a base cost of 1 (for the communicate_soil_data action).
         - Check if any rover `r` has `have_soil_analysis(r, waypoint)`. If not, add a cost of 1 (for the sample_soil action).
           Implicitly assumes a rover can reach a waypoint with a soil sample and has the equipment.
       - For `communicated_rock_data(waypoint)`:
         - Similar to soil data, add 1 for communicate_rock_data and 1 for sample_rock if needed.
       - For `communicated_image_data(objective, mode)`:
         - If `communicated_image_data(objective, mode)` is not achieved, add a base cost of 1 (for the communicate_image_data action).
         - Check if any rover `r` has `have_image(r, objective, mode)`. If not, add a cost of 1 (for the take_image action).
         - Check if any camera `c` on a rover `r` is `calibrated(c, r)`. If not, add a cost of 1 (for the calibrate action).
           Implicitly assumes a rover can reach a waypoint visible from the objective and has the necessary equipment and camera.

    3. Sum up the estimated costs for all goal facts. This sum represents the heuristic estimate for the given state.
    This heuristic is admissible under the simplified assumptions and provides a quick estimate of the remaining actions.
    It prioritizes states that have already achieved more goal conditions or are closer to achieving them in terms of necessary actions like sampling, calibrating, and taking images.
    """

    def __init__(self, task):
        """
        Initialize the rovers heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.soil_sample_locations = set()
        self.rock_sample_locations = set()
        self.objective_visible_from = {}
        self.calibration_targets = {}
        self.camera_supports = {}
        self.rovers_equipped_imaging = set()
        self.rovers_equipped_soil = set()
        self.rovers_equipped_rock = set()
        self.cameras_on_board = {} # {camera: rover}

        for fact in static_facts:
            if match(fact, "at_soil_sample", "*"):
                self.soil_sample_locations.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.rock_sample_locations.add(get_parts(fact)[1])
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1], get_parts(fact)[2]
                if objective not in self.objective_visible_from:
                    self.objective_visible_from[objective] = set()
                self.objective_visible_from[objective].add(waypoint)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_parts(fact)[1], get_parts(fact)[2]
                if camera not in self.calibration_targets:
                    self.calibration_targets[camera] = set()
                self.calibration_targets[camera].add(objective)
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1], get_parts(fact)[2]
                if camera not in self.camera_supports:
                    self.camera_supports[camera] = set()
                self.camera_supports[camera].add(mode)
            elif match(fact, "equipped_for_imaging", "*"):
                self.rovers_equipped_imaging.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.rovers_equipped_soil.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.rovers_equipped_rock.add(get_parts(fact)[1])
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_parts(fact)[1], get_parts(fact)[2]
                self.cameras_on_board[camera] = rover


    def __call__(self, node):
        """
        Estimate the number of actions to reach the goal state from the current state.
        """
        state = node.state
        heuristic_value = 0

        for goal_fact in self.goals:
            if goal_fact in state:
                continue

            if match(goal_fact, "communicated_soil_data", "*"):
                waypoint = get_parts(goal_fact)[1]
                heuristic_value += 1 # communicate_soil_data action
                soil_analysis_achieved = False
                for fact in state:
                    if match(fact, "have_soil_analysis", "*", waypoint):
                        soil_analysis_achieved = True
                        break
                if not soil_analysis_achieved:
                    heuristic_value += 1 # sample_soil action

            elif match(goal_fact, "communicated_rock_data", "*"):
                waypoint = get_parts(goal_fact)[1]
                heuristic_value += 1 # communicate_rock_data action
                rock_analysis_achieved = False
                for fact in state:
                    if match(fact, "have_rock_analysis", "*", waypoint):
                        rock_analysis_achieved = True
                        break
                if not rock_analysis_achieved:
                    heuristic_value += 1 # sample_rock action

            elif match(goal_fact, "communicated_image_data", "*", "*"):
                objective = get_parts(goal_fact)[1]
                mode = get_parts(goal_fact)[2]
                heuristic_value += 1 # communicate_image_data action
                image_taken = False
                for fact in state:
                    if match(fact, "have_image", "*", objective, mode):
                        image_taken = True
                        break
                if not image_taken:
                    heuristic_value += 1 # take_image action
                    calibration_needed = True
                    for fact in state:
                        if match(fact, "calibrated", "*", "*"):
                            camera_rover = get_parts(fact)[1:]
                            for cam, rov in self.cameras_on_board.items():
                                if cam == camera_rover[0] and rov == camera_rover[1]:
                                    if mode in self.camera_supports.get(cam, set()) and objective in self.calibration_targets.get(cam, set()):
                                        calibration_needed = False
                                        break
                        if not calibration_needed:
                            break
                    if calibration_needed:
                        heuristic_value += 1 # calibrate action

        return heuristic_value
