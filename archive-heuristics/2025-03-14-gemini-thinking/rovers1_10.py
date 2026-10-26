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
    It considers the necessary actions for each goal type (communicating image, soil, and rock data) and sums up the estimated costs.
    Navigation costs are implicitly considered by assuming a cost if the rover is not already at the required location for an action.

    # Assumptions:
    - Each goal predicate needs to be achieved independently.
    - Navigation between waypoints is always possible if `can_traverse` and `visible` predicates are defined.
    - Rovers are equipped as specified in the initial state and equipment doesn't change.
    - Stores are initially empty and can hold only one sample at a time.

    # Heuristic Initialization
    - Extracts the goal predicates from the task definition.
    - Extracts static facts such as `at_lander`, `equipped_for_*`, `store_of`, `calibration_target`, `on_board`, `supports`, `visible_from`, `at_soil_sample`, `at_rock_sample`.
    - Stores relevant static information in dictionaries for efficient access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate that is not satisfied in the current state:
    1. Identify the type of goal: `communicated_image_data`, `communicated_soil_data`, or `communicated_rock_data`.
    2. Estimate the minimum number of actions required to achieve this goal, considering the preconditions of each action.
    3. For `communicated_image_data(objective, mode)`:
        - If `communicated_image_data(objective, mode)` is not in the state, add a cost of 1 (for `communicate_image_data`).
        - If `have_image(rover, objective, mode)` is not in the state, add a cost of 1 (for `take_image`). Also add a navigation cost of 1 if the rover is not at a waypoint visible from the objective.
        - If `calibrated(camera, rover)` is not in the state, add a cost of 1 (for `calibrate`). Also add a navigation cost of 1 if the rover is not at a waypoint visible from the objective.
    4. For `communicated_soil_data(waypoint)`:
        - If `communicated_soil_data(waypoint)` is not in the state, add a cost of 1 (for `communicate_soil_data`).
        - If `have_soil_analysis(rover, waypoint)` is not in the state, add a cost of 1 (for `sample_soil`). Also add a navigation cost of 1 if the rover is not at the waypoint.
    5. For `communicated_rock_data(waypoint)`:
        - If `communicated_rock_data(waypoint)` is not in the state, add a cost of 1 (for `communicate_rock_data`).
        - If `have_rock_analysis(rover, waypoint)` is not in the state, add a cost of 1 (for `sample_rock`). Also add a navigation cost of 1 if the rover is not at the waypoint.
    6. Sum up the estimated costs for all unsatisfied goal predicates.
    7. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.lander_location = next((get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None)
        self.equipped_for_soil_analysis = set(get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*"))
        self.equipped_for_rock_analysis = set(get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*"))
        self.equipped_for_imaging = set(get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*"))
        self.store_of = {get_parts(fact)[2]: get_parts(fact)[1] for fact in static_facts if match(fact, "store_of", "*", "*")}
        self.calibration_targets = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "calibration_target", "*", "*")}
        self.on_board_cameras = {get_parts(fact)[2]: get_parts(fact)[1] for fact in static_facts if match(fact, "on_board", "*", "*")}
        self.camera_supports = {}
        for fact in static_facts:
            if match(fact, "supports", "*", "*"):
                camera, mode = get_parts(fact)[1], get_parts(fact)[2]
                if camera not in self.camera_supports:
                    self.camera_supports[camera] = set()
                self.camera_supports[camera].add(mode)
        self.visible_from_objective = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_parts(fact)[1], get_parts(fact)[2]
                if objective not in self.visible_from_objective:
                    self.visible_from_objective[objective] = set()
                self.visible_from_objective[objective].add(waypoint)
        self.at_soil_samples = set(get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*"))
        self.at_rock_samples = set(get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*"))
        self.visible_waypoints = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1, waypoint2 = get_parts(fact)[1], get_parts(fact)[2]
                if waypoint1 not in self.visible_waypoints:
                    self.visible_waypoints[waypoint1] = set()
                self.visible_waypoints[waypoint1].add(waypoint2)


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        unsatisfied_goals = self.goals - state
        heuristic_value = 0

        for goal in unsatisfied_goals:
            if match(goal, "communicated_image_data", "*", "*"):
                objective, mode = get_parts(goal)[1], get_parts(goal)[2]
                rover = next((rover for rover in self.equipped_for_imaging if self.on_board_cameras.get(rover) is not None and self.calibration_targets.get(self.on_board_cameras[rover]) == objective and mode in self.camera_supports.get(self.on_board_cameras[rover], set())), None)
                camera = self.on_board_cameras.get(rover) if rover else None

                if not match(state, "have_image", rover, objective, mode):
                    heuristic_value += 1 # take_image action
                    if not match(state, "calibrated", camera, rover):
                        heuristic_value += 1 # calibrate action
                        heuristic_value += 1 # navigate for calibrate (simplified cost)
                    heuristic_value += 1 # navigate for take_image (simplified cost)
                heuristic_value += 1 # communicate_image_data action
                heuristic_value += 1 # navigate for communicate (simplified cost)


            elif match(goal, "communicated_soil_data", "*"):
                waypoint = get_parts(goal)[1]
                rover = next((rover for rover in self.equipped_for_soil_analysis if self.store_of.get(rover) is not None), None)
                store = self.store_of.get(rover) if rover else None

                if not match(state, "have_soil_analysis", rover, waypoint):
                    heuristic_value += 1 # sample_soil action
                    heuristic_value += 1 # navigate for sample_soil (simplified cost)
                heuristic_value += 1 # communicate_soil_data action
                heuristic_value += 1 # navigate for communicate (simplified cost)


            elif match(goal, "communicated_rock_data", "*"):
                waypoint = get_parts(goal)[1]
                rover = next((rover for rover in self.equipped_for_rock_analysis if self.store_of.get(rover) is not None), None)
                store = self.store_of.get(rover) if rover else None

                if not match(state, "have_rock_analysis", rover, waypoint):
                    heuristic_value += 1 # sample_rock action
                    heuristic_value += 1 # navigate for sample_rock (simplified cost)
                heuristic_value += 1 # communicate_rock_data action
                heuristic_value += 1 # navigate for communicate (simplified cost)

        return heuristic_value
