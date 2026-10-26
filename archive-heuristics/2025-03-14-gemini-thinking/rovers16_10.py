from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_string):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at rover1 waypoint2)' it returns ['rover1', 'waypoint2'].
    Ignores the predicate name.
    """
    parts = fact_string[1:-1].split()
    return parts[1:]

def match(fact, *args):
    """
    Utility function to check if a PDDL fact matches a given pattern.
    - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
    - `args`: The pattern to match (e.g., "at", "*", "rooma").
    - Returns `True` if the fact matches the pattern, `False` otherwise.
    """
    parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal state in the rovers domain.
    It focuses on the key tasks: capturing images, analyzing soil and rocks, and communicating data.
    The heuristic sums up the estimated costs for each goal fact that is not yet achieved in the current state.

    # Assumptions:
    - Each goal fact requires at least one action to achieve if it's not already true.
    - Navigation to a required waypoint, calibration, sampling, imaging, and communication each contribute a fixed cost to the heuristic.
    - Store management (dropping samples) is considered if necessary before sampling.

    # Heuristic Initialization
    - Extracts static information from the task, such as:
        - Waypoints visible from each other.
        - Waypoints visible from objectives.
        - Calibration targets for cameras.
        - Camera support for different modes.
        - Equipment of rovers (imaging, soil analysis, rock analysis).
        - Stores associated with rovers.
        - Lander location.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal fact in the goal description:
    1. Check if the goal fact is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If not achieved, estimate the minimum actions needed to achieve it based on the goal type:
        - For `(communicated_image_data ?o ?m)`:
            - Assume 1 action for communication.
            - Check if `(have_image ?r ?o ?m)` exists for any rover ?r. If not, assume 1 action for taking image.
                - Check if any camera is calibrated on a rover that can take images. If not, assume 1 action for calibration.
                - Assume 1 action for navigation to a waypoint visible from the objective (simplified navigation cost).
        - For `(communicated_soil_data ?w)` or `(communicated_rock_data ?w)`:
            - Assume 1 action for communication.
            - Check if `(have_soil_analysis ?r ?w)` or `(have_rock_analysis ?r ?w)` exists for any rover ?r. If not, assume 1 action for sampling.
                - Check if a rover's store is empty. If not, assume 1 action for dropping the current content of the store.
                - Assume 1 action for navigation to the sample waypoint (simplified navigation cost).
    3. Sum up the estimated costs for all unachieved goal facts.
    """

    def __init__(self, task):
        """Initialize the rovers heuristic by pre-processing static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.visible_waypoints = {}
        self.can_traverse_relations = {}
        self.visible_from_objectives = {}
        self.calibration_targets = {}
        self.supports_modes = {}
        self.store_of_rover = {}
        self.equipped_for_soil_analysis = set()
        self.equipped_for_rock_analysis = set()
        self.equipped_for_imaging = set()
        self.on_board_cameras = {}
        self.lander_location = None

        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1, waypoint2 = get_objects_from_fact(fact)
                self.visible_waypoints.setdefault(waypoint1, set()).add(waypoint2)
            elif match(fact, "can_traverse", "*", "*", "*"):
                rover, wp1, wp2 = get_objects_from_fact(fact)
                self.can_traverse_relations.setdefault(rover, {}).setdefault(wp1, set()).add(wp2)
            elif match(fact, "visible_from", "*", "*"):
                objective, waypoint = get_objects_from_fact(fact)
                self.visible_from_objectives.setdefault(objective, set()).add(waypoint)
            elif match(fact, "calibration_target", "*", "*"):
                camera, objective = get_objects_from_fact(fact)
                self.calibration_targets[camera] = objective
            elif match(fact, "supports", "*", "*"):
                camera, mode = get_objects_from_fact(fact)
                self.supports_modes.setdefault(camera, set()).add(mode)
            elif match(fact, "store_of", "*", "*"):
                store, rover = get_objects_from_fact(fact)
                self.store_of_rover[store] = rover
            elif match(fact, "equipped_for_soil_analysis", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_soil_analysis.add(rover)
            elif match(fact, "equipped_for_rock_analysis", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_rock_analysis.add(rover)
            elif match(fact, "equipped_for_imaging", "*"):
                rover = get_objects_from_fact(fact)[0]
                self.equipped_for_imaging.add(rover)
            elif match(fact, "on_board", "*", "*"):
                camera, rover = get_objects_from_fact(fact)
                self.on_board_cameras.setdefault(rover, set()).add(camera)
            elif match(fact, "at_lander", "*", "*"):
                lander, waypoint = get_objects_from_fact(fact)
                self.lander_location = waypoint


    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        for goal_fact in self.goals:
            if goal_fact not in state:
                predicate = goal_fact[1:-1].split()[0]
                objects = get_objects_from_fact(goal_fact)

                if predicate == 'communicated_image_data':
                    objective, mode = objects
                    heuristic_value += 1 # For communicate action
                    have_image_goal = False
                    for r in self.equipped_for_imaging:
                        if f'(have_image {r} {objective} {mode})' in state:
                            have_image_goal = True
                            break
                    if not have_image_goal:
                        heuristic_value += 1 # For take_image action
                        calibrated_goal = False
                        for r in self.equipped_for_imaging:
                            for camera in self.on_board_cameras.get(r, []):
                                if camera in self.supports_modes and mode in self.supports_modes[camera]:
                                    if f'(calibrated {camera} {r})' in state:
                                        calibrated_goal = True
                                        break
                            if calibrated_goal:
                                break
                        if not calibrated_goal:
                            heuristic_value += 1 # For calibrate action
                        heuristic_value += 1 # For navigate to visible waypoint (simplified)

                elif predicate == 'communicated_soil_data':
                    waypoint = objects[0]
                    heuristic_value += 1 # For communicate_soil_data
                    have_soil_analysis_goal = False
                    for r in self.equipped_for_soil_analysis:
                        if f'(have_soil_analysis {r} {waypoint})' in state:
                            have_soil_analysis_goal = True
                            break
                    if not have_soil_analysis_goal:
                        heuristic_value += 1 # For sample_soil action
                        for r in self.equipped_for_soil_analysis:
                            store = None
                            for s, rover in self.store_of_rover.items():
                                if rover == r:
                                    store = s
                                    break
                            if store and f'(full {store})' in state:
                                heuristic_value += 1 # For drop action (if store full)
                                break # Assuming one drop is enough if any store is full
                        heuristic_value += 1 # For navigate to sample waypoint (simplified)

                elif predicate == 'communicated_rock_data':
                    waypoint = objects[0]
                    heuristic_value += 1 # For communicate_rock_data
                    have_rock_analysis_goal = False
                    for r in self.equipped_for_rock_analysis:
                        if f'(have_rock_analysis {r} {waypoint})' in state:
                            have_rock_analysis_goal = True
                            break
                    if not have_rock_analysis_goal:
                        heuristic_value += 1 # For sample_rock action
                        for r in self.equipped_for_rock_analysis:
                            store = None
                            for s, rover in self.store_of_rover.items():
                                if rover == r:
                                    store = s
                                    break
                            if store and f'(full {store})' in state:
                                heuristic_value += 1 # For drop action (if store full)
                                break # Assuming one drop is enough if any store is full
                        heuristic_value += 1 # For navigate to sample waypoint (simplified)
        return heuristic_value
