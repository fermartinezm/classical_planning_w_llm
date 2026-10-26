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
    It focuses on achieving each goal predicate independently and sums up the estimated costs.
    The heuristic considers actions like navigation, sampling (soil/rock), calibration, imaging, and communication.

    # Assumptions:
    - Each action (navigate, sample_soil, sample_rock, drop, calibrate, take_image, communicate_soil_data,
      communicate_rock_data, communicate_image_data) has a uniform cost of 1.
    - The heuristic does not explicitly consider store capacity or optimize action sequences.
    - Navigation cost is simplified: it assumes 1 action if the rover is not at the required waypoint, and 0 otherwise.
    - For image goals, calibration is always assumed to be needed before taking an image if not already calibrated.

    # Heuristic Initialization
    - Extracts static facts from the task to build data structures for efficient lookup:
        - `can_traverse_map`:  Set of tuples (rover, waypoint1, waypoint2) representing traversable paths.
        - `visible_map`: Set of tuples (waypoint1, waypoint2) representing visibility between waypoints.
        - `visible_from_objective_map`: Dictionary mapping objectives to sets of waypoints from which they are visible.
        - `equipped_for_soil_analysis_rovers`, `equipped_for_rock_analysis_rovers`, `equipped_for_imaging_rovers`: Sets of rovers with specific equipment.
        - `calibration_target_map`: Dictionary mapping cameras to their calibration objectives.
        - `supports_mode_map`: Dictionary mapping cameras to sets of supported modes.
        - `store_of_rover_map`: Dictionary mapping stores to their respective rovers.
        - `at_soil_sample_waypoints`, `at_rock_sample_waypoints`: Sets of waypoints with soil and rock samples.
        - `lander_location`: Location of the lander.
    - Stores the goal predicates for later evaluation.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate in the goal state:
    1. Check if the goal predicate is already satisfied in the current state. If yes, the cost for this goal is 0.
    2. If not satisfied, determine the type of goal (communicate_soil_data, communicate_rock_data, communicate_image_data).
    3. Estimate the minimum actions needed to achieve this goal:
        - **For communicate_soil_data(waypoint):**
            - Check if communicated_soil_data(waypoint) is already true.
            - If not, check if have_soil_analysis(rover, waypoint) is true for any rover.
                - If not, estimate cost to sample soil:
                    - 1 (sample_soil) + 1 (navigate to waypoint if not there) + 1 (if rover not equipped for soil analysis)
                - If have_soil_analysis(rover, waypoint) is true, estimate cost to communicate:
                    - 1 (communicate_soil_data) + 1 (navigate to a waypoint visible from lander if not there).
        - **For communicate_rock_data(waypoint):**
            - Similar to communicate_soil_data, but for rock samples and rock analysis equipment.
        - **For communicate_image_data(objective, mode):**
            - Check if communicated_image_data(objective, mode) is already true.
            - If not, check if have_image(rover, objective, mode) is true for any rover.
                - If not, estimate cost to take image:
                    - 1 (take_image) + 1 (calibrate if not calibrated) + 1 (navigate to waypoint visible from objective if not there) + 1 (if rover not equipped for imaging) + 1 (if camera does not support mode)
                - If have_image(rover, objective, mode) is true, estimate cost to communicate:
                    - 1 (communicate_image_data) + 1 (navigate to a waypoint visible from lander if not there).
    4. Sum up the estimated costs for all unsatisfied goal predicates to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the rovers heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.can_traverse_map = set()
        self.visible_map = set()
        self.visible_from_objective_map = {}
        self.equipped_for_soil_analysis_rovers = set()
        self.equipped_for_rock_analysis_rovers = set()
        self.equipped_for_imaging_rovers = set()
        self.calibration_target_map = {}
        self.supports_mode_map = {}
        self.store_of_rover_map = {}
        self.at_soil_sample_waypoints = set()
        self.at_rock_sample_waypoints = set()
        self.lander_location = None

        for fact in static_facts:
            if match(fact, "can_traverse", "*", "*", "*"):
                parts = get_parts(fact)
                self.can_traverse_map.add((parts[1], parts[2], parts[3]))
            elif match(fact, "visible", "*", "*"):
                parts = get_parts(fact)
                self.visible_map.add((parts[1], parts[2]))
            elif match(fact, "visible_from", "*", "*"):
                parts = get_parts(fact)
                objective = parts[1]
                waypoint = parts[2]
                if objective not in self.visible_from_objective_map:
                    self.visible_from_objective_map[objective] = set()
                self.visible_from_objective_map[objective].add(waypoint)
            elif match(fact, "equipped_for_soil_analysis", "*"):
                self.equipped_for_soil_analysis_rovers.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_rock_analysis", "*"):
                self.equipped_for_rock_analysis_rovers.add(get_parts(fact)[1])
            elif match(fact, "equipped_for_imaging", "*"):
                self.equipped_for_imaging_rovers.add(get_parts(fact)[1])
            elif match(fact, "calibration_target", "*", "*"):
                parts = get_parts(fact)
                self.calibration_target_map[parts[1]] = parts[2]
            elif match(fact, "supports", "*", "*"):
                parts = get_parts(fact)
                camera = parts[1]
                mode = parts[2]
                if camera not in self.supports_mode_map:
                    self.supports_mode_map[camera] = set()
                self.supports_mode_map[camera].add(mode)
            elif match(fact, "store_of", "*", "*"):
                parts = get_parts(fact)
                self.store_of_rover_map[parts[1]] = parts[2]
            elif match(fact, "at_soil_sample", "*"):
                self.at_soil_sample_waypoints.add(get_parts(fact)[1])
            elif match(fact, "at_rock_sample", "*"):
                self.at_rock_sample_waypoints.add(get_parts(fact)[1])
            elif match(fact, "at_lander", "*", "*"):
                self.lander_location = (get_parts(fact)[1], get_parts(fact)[2])


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        current_rover_locations = {}
        current_camera_calibrated = {}
        current_have_soil_analysis = {}
        current_have_rock_analysis = {}
        current_have_image = {}
        current_communicated_soil_data = set()
        current_communicated_rock_data = set()
        current_communicated_image_data = set()

        for fact in state:
            if match(fact, "at", "?r", "?w"):
                parts = get_parts(fact)
                current_rover_locations[parts[1]] = parts[2]
            elif match(fact, "calibrated", "?c", "?r"):
                parts = get_parts(fact)
                current_camera_calibrated[(parts[1], parts[2])] = True
            elif match(fact, "have_soil_analysis", "?r", "?w"):
                parts = get_parts(fact)
                current_have_soil_analysis[(parts[1], parts[2])] = True
            elif match(fact, "have_rock_analysis", "?r", "?w"):
                parts = get_parts(fact)
                current_have_rock_analysis[(parts[1], parts[2])] = True
            elif match(fact, "have_image", "?r", "?o", "?m"):
                parts = get_parts(fact)
                current_have_image[(parts[1], parts[2], parts[3])] = True
            elif match(fact, "communicated_soil_data", "?w"):
                current_communicated_soil_data.add(get_parts(fact)[1])
            elif match(fact, "communicated_rock_data", "?w"):
                current_communicated_rock_data.add(get_parts(fact)[1])
            elif match(fact, "communicated_image_data", "?o", "?m"):
                current_communicated_image_data.add(tuple(get_parts(fact)[1:]))


        for goal in self.goals:
            if goal in state:
                continue

            parts = get_parts(goal)
            goal_predicate = parts[0]

            if goal_predicate == "communicated_soil_data":
                waypoint = parts[1]
                if waypoint not in current_communicated_soil_data:
                    sample_cost = 0
                    if not any(current_have_soil_analysis.get((rover, waypoint), False) for rover in self.equipped_for_soil_analysis_rovers):
                        sample_cost += 1 # sample_soil action
                        sample_cost += 1 # navigate to sample waypoint if not there (simplified)
                        sample_cost += 0 # assume rover is equipped, or add cost if checking equipment is needed.

                    communicate_cost = 1 # communicate_soil_data action
                    communicate_cost += 1 # navigate to waypoint visible from lander (simplified)

                    heuristic_value += min(sample_cost + communicate_cost, 2) # Simplified cost, at least communicate and maybe sample

            elif goal_predicate == "communicated_rock_data":
                waypoint = parts[1]
                if waypoint not in current_communicated_rock_data:
                    sample_cost = 0
                    if not any(current_have_rock_analysis.get((rover, waypoint), False) for rover in self.equipped_for_rock_analysis_rovers):
                        sample_cost += 1 # sample_rock action
                        sample_cost += 1 # navigate to sample waypoint if not there (simplified)
                        sample_cost += 0 # assume rover is equipped

                    communicate_cost = 1 # communicate_rock_data action
                    communicate_cost += 1 # navigate to waypoint visible from lander (simplified)
                    heuristic_value += min(sample_cost + communicate_cost, 2) # Simplified cost, at least communicate and maybe sample

            elif goal_predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                if tuple([objective, mode]) not in current_communicated_image_data:
                    image_cost = 0
                    if not any(current_have_image.get((rover, objective, mode), False) for rover in self.equipped_for_imaging_rovers):
                        image_cost += 1 # take_image action
                        image_cost += 1 # calibrate action (simplified, always assume needed if not have_image)
                        image_cost += 1 # navigate to waypoint visible from objective (simplified)
                        image_cost += 0 # assume rover is equipped and supports mode

                    communicate_cost = 1 # communicate_image_data action
                    communicate_cost += 1 # navigate to waypoint visible from lander (simplified)
                    heuristic_value += min(image_cost + communicate_cost, 3) # Simplified cost, at least communicate and maybe image and calibrate

        return heuristic_value
