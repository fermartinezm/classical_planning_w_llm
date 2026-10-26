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
    It considers the necessary steps for data collection (imaging, soil, rock analysis) and communication,
    including navigation, calibration, and sampling actions.

    # Assumptions:
    - Each goal fact (communicated data) is treated independently.
    - Navigation cost is roughly estimated as one action when needed to reach a required waypoint.
    - Rovers are capable of performing all necessary actions if equipped.
    - Store management (dropping samples) is not explicitly considered in detail, assuming stores are managed efficiently.

    # Heuristic Initialization
    - Extracts the goal conditions from the task definition.
    - Extracts static facts like visibility, traversal capabilities, equipment, and camera support to efficiently check preconditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact in the goal state:
    1. Check if the goal fact is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If not achieved, estimate the cost based on the type of goal:
        a. For `communicated_image_data(objective, mode)`:
           - Cost for communication: 1 (communicate_image_data action).
           - Check if `have_image(rover, objective, mode)` is true. If not:
             - Cost for taking image: 1 (take_image action).
             - Check if `calibrated(camera, rover)` is true. If not:
               - Cost for calibration: 1 (calibrate action).
               - Cost for navigation to a waypoint visible from the objective for calibration (if needed): 1 (navigate action - simplified).
             - Cost for navigation to a waypoint visible from the objective for taking image (if needed): 1 (navigate action - simplified).
           - Cost for navigation to a waypoint visible from the lander for communication (if needed): 1 (navigate action - simplified).
        b. For `communicated_soil_data(waypoint)`:
           - Cost for communication: 1 (communicate_soil_data action).
           - Check if `have_soil_analysis(rover, waypoint)` is true. If not:
             - Cost for sampling soil: 1 (sample_soil action).
             - Cost for navigation to the waypoint with soil sample (if needed): 1 (navigate action - simplified).
           - Cost for navigation to a waypoint visible from the lander for communication (if needed): 1 (navigate action - simplified).
        c. For `communicated_rock_data(waypoint)`:
           - Cost for communication: 1 (communicate_rock_data action).
           - Check if `have_rock_analysis(rover, waypoint)` is true. If not:
             - Cost for sampling rock: 1 (sample_rock action).
             - Cost for navigation to the waypoint with rock sample (if needed): 1 (navigate action - simplified).
           - Cost for navigation to a waypoint visible from the lander for communication (if needed): 1 (navigate action - simplified).
    3. Sum up the estimated costs for all goal facts to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        self.calibration_targets = {}
        self.camera_supports = {}
        self.on_board_cameras = {}
        self.equipped_for_imaging_rovers = set()
        self.equipped_for_soil_rovers = set()
        self.equipped_for_rock_rovers = set()
        self.store_of_rovers = {}
        self.visible_from_objectives = {}
        self.visible_waypoints = {}
        self.at_lander_locations = set()

        for fact in self.static_facts:
            if match(fact, 'calibration_target', '*', '*'):
                parts = get_parts(fact)
                camera = parts[1]
                objective = parts[2]
                if camera not in self.calibration_targets:
                    self.calibration_targets[camera] = []
                self.calibration_targets[camera].append(objective)
            elif match(fact, 'supports', '*', '*'):
                parts = get_parts(fact)
                camera = parts[1]
                mode = parts[2]
                if camera not in self.camera_supports:
                    self.camera_supports[camera] = []
                self.camera_supports[camera].append(mode)
            elif match(fact, 'on_board', '*', '*'):
                parts = get_parts(fact)
                camera = parts[1]
                rover = parts[2]
                if rover not in self.on_board_cameras:
                    self.on_board_cameras[rover] = []
                self.on_board_cameras[rover].append(camera)
            elif match(fact, 'equipped_for_imaging', '*'):
                self.equipped_for_imaging_rovers.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_soil_analysis', '*'):
                self.equipped_for_soil_rovers.add(get_parts(fact)[1])
            elif match(fact, 'equipped_for_rock_analysis', '*'):
                self.equipped_for_rock_rovers.add(get_parts(fact)[1])
            elif match(fact, 'store_of', '*', '*'):
                parts = get_parts(fact)
                store = parts[1]
                rover = parts[2]
                self.store_of_rovers[rover] = store
            elif match(fact, 'visible_from', '*', '*'):
                parts = get_parts(fact)
                objective = parts[1]
                waypoint = parts[2]
                if objective not in self.visible_from_objectives:
                    self.visible_from_objectives[objective] = []
                self.visible_from_objectives[objective].append(waypoint)
            elif match(fact, 'visible', '*', '*'):
                parts = get_parts(fact)
                waypoint1 = parts[1]
                waypoint2 = parts[2]
                if waypoint1 not in self.visible_waypoints:
                    self.visible_waypoints[waypoint1] = []
                self.visible_waypoints[waypoint1].append(waypoint2)
            elif match(fact, 'at_lander', '*', '*'):
                self.at_lander_locations.add(get_parts(fact)[2])


    def __call__(self, node):
        """
        Estimate the number of actions to reach the goal state from the current state.
        """
        state = node.state
        heuristic_value = 0

        for goal_fact in self.goals:
            if goal_fact in state:
                continue

            if match(goal_fact, 'communicated_image_data', '*', '*'):
                objective = get_parts(goal_fact)[1]
                mode = get_parts(goal_fact)[2]
                heuristic_value += 1  # communicate_image_data action

                found_image = False
                for fact in state:
                    if match(fact, 'have_image', '*', objective, mode):
                        found_image = True
                        break
                if not found_image:
                    heuristic_value += 1  # take_image action
                    found_calibration = False
                    rover_for_image = None
                    camera_for_image = None
                    for rover in self.equipped_for_imaging_rovers:
                        if rover in self.on_board_cameras:
                            for camera in self.on_board_cameras[rover]:
                                if camera in self.calibration_targets and objective in self.calibration_targets[camera] and camera in self.camera_supports and mode in self.camera_supports[camera]:
                                    rover_for_image = rover
                                    camera_for_image = camera
                                    break
                            if rover_for_image:
                                break

                    if rover_for_image and camera_for_image:
                        for fact in state:
                            if match(fact, 'calibrated', camera_for_image, rover_for_image):
                                found_calibration = True
                                break
                        if not found_calibration:
                            heuristic_value += 1 # calibrate action
                            # heuristic_value += 1 # navigate to calibration waypoint (simplified)
                    # heuristic_value += 1 # navigate to waypoint visible from objective (simplified)
                # heuristic_value += 1 # navigate to waypoint visible from lander (simplified)


            elif match(goal_fact, 'communicated_soil_data', '*'):
                waypoint = get_parts(goal_fact)[1]
                heuristic_value += 1  # communicate_soil_data action
                found_soil_analysis = False
                for fact in state:
                    if match(fact, 'have_soil_analysis', '*', waypoint):
                        found_soil_analysis = True
                        break
                if not found_soil_analysis:
                    heuristic_value += 1  # sample_soil action
                    # heuristic_value += 1 # navigate to sample waypoint (simplified)
                # heuristic_value += 1 # navigate to waypoint visible from lander (simplified)


            elif match(goal_fact, 'communicated_rock_data', '*'):
                waypoint = get_parts(goal_fact)[1]
                heuristic_value += 1  # communicate_rock_data action
                found_rock_analysis = False
                for fact in state:
                    if match(fact, 'have_rock_analysis', '*', waypoint):
                        found_rock_analysis = True
                        break
                if not found_rock_analysis:
                    heuristic_value += 1  # sample_rock action
                    # heuristic_value += 1 # navigate to sample waypoint (simplified)
                # heuristic_value += 1 # navigate to waypoint visible from lander (simplified)

        return heuristic_value
