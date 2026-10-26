from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_objects_from_fact(fact_string):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it returns ['rover1', 'waypoint1'].
    Ignores the predicate name.
    """
    parts = fact_string[1:-1].split()
    return parts[1:]

def get_predicate_name(fact_string):
    """
    Extracts the predicate name from a PDDL fact string.
    For example, from '(at rover1 waypoint1)' it returns 'at'.
    """
    parts = fact_string[1:-1].split()
    return parts[0]


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
    This heuristic estimates the number of actions required to achieve all goal conditions in the rovers domain.
    It considers the necessary steps for navigation, sampling (soil and rock), image taking, calibration, and communication for each goal.

    # Assumptions:
    - Each goal condition is treated independently and the costs are summed up.
    - Navigation cost is estimated as 1 action if the rover is not at the required waypoint.
    - Sampling, calibration, image taking, and communication are estimated as 1 action each if their preconditions are met (or assumed to be met after navigation/previous actions).
    - It does not consider optimal paths or resource contention (e.g., store capacity).
    - It assumes that all necessary equipment is available on some rover if the goal requires it.

    # Heuristic Initialization
    - Extracts goal predicates from the task definition.
    - Extracts static facts like visibility, traversal capabilities, equipment, calibration targets, etc., to be used in heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition in the goal state:
    1. Check if the goal is already achieved in the current state. If yes, the cost for this goal is 0.
    2. If not achieved, estimate the minimum actions needed:
        a. For `(communicated_soil_data ?waypoint)`:
           - Cost += 1 (communicate_soil_data action)
           - If `(have_soil_analysis ?rover ?waypoint)` is not in the state:
             - Cost += 1 (sample_soil action)
             - Find a rover equipped for soil analysis and a store for it.
             - If the rover is not at `?waypoint`:
               - Cost += 1 (navigate action to ?waypoint)
        b. For `(communicated_rock_data ?waypoint)`:
           - Cost += 1 (communicate_rock_data action)
           - If `(have_rock_analysis ?rover ?waypoint)` is not in the state:
             - Cost += 1 (sample_rock action)
             - Find a rover equipped for rock analysis and a store for it.
             - If the rover is not at `?waypoint`:
               - Cost += 1 (navigate action to ?waypoint)
        c. For `(communicated_image_data ?objective ?mode)`:
           - Cost += 1 (communicate_image_data action)
           - If `(have_image ?rover ?objective ?mode)` is not in the state:
             - Cost += 1 (take_image action)
             - Find a rover equipped for imaging, a camera on board, and supporting the mode.
             - If `(calibrated ?camera ?rover)` is not in the state:
               - Cost += 1 (calibrate action)
               - Find a waypoint `?w` from which `?objective` is visible.
               - If the rover is not at `?w`:
                 - Cost += 1 (navigate action to ?w)
             - Else (if already calibrated):
               - Find a waypoint `?p` from which `?objective` is visible.
               - If the rover is not at `?p`:
                 - Cost += 1 (navigate action to ?p)
        d. For communication actions, we also need to ensure the rover is at a waypoint visible from the lander's waypoint. This cost is added for each communication goal if needed.

    3. Sum up the costs for all goal conditions to get the total heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        self.lander_location = next(get_objects_from_fact(fact)[1] for fact in self.static_facts if get_predicate_name(fact) == 'at_lander')
        self.rovers_equipped_for_soil = set(get_objects_from_fact(fact)[0] for fact in self.static_facts if get_predicate_name(fact) == 'equipped_for_soil_analysis')
        self.rovers_equipped_for_rock = set(get_objects_from_fact(fact)[0] for fact in self.static_facts if get_predicate_name(fact) == 'equipped_for_rock_analysis')
        self.rovers_equipped_for_imaging = set(get_objects_from_fact(fact)[0] for fact in self.static_facts if get_predicate_name(fact) == 'equipped_for_imaging')
        self.stores_of_rovers = {get_objects_from_fact(fact)[1]: get_objects_from_fact(fact)[0] for fact in self.static_facts if get_predicate_name(fact) == 'store_of'}
        self.cameras_on_rovers = {get_objects_from_fact(fact)[1]: get_objects_from_fact(fact)[0] for fact in self.static_facts if get_predicate_name(fact) == 'on_board'}
        self.camera_supports_modes = {}
        for fact in self.static_facts:
            if get_predicate_name(fact) == 'supports':
                camera, mode = get_objects_from_fact(fact)
                if camera not in self.camera_supports_modes:
                    self.camera_supports_modes[camera] = set()
                self.camera_supports_modes[camera].add(mode)
        self.calibration_targets = {get_objects_from_fact(fact)[0]: get_objects_from_fact(fact)[1] for fact in self.static_facts if get_predicate_name(fact) == 'calibration_target'}
        self.visible_from_objective = {}
        for fact in self.static_facts:
            if get_predicate_name(fact) == 'visible_from':
                objective, waypoint = get_objects_from_fact(fact)
                if objective not in self.visible_from_objective:
                    self.visible_from_objective[objective] = set()
                self.visible_from_objective[objective].add(waypoint)
        self.visible_waypoints = {}
        for fact in self.static_facts:
            if get_predicate_name(fact) == 'visible':
                waypoint1, waypoint2 = get_objects_from_fact(fact)
                if waypoint1 not in self.visible_waypoints:
                    self.visible_waypoints[waypoint1] = set()
                self.visible_waypoints[waypoint1].add(waypoint2)
                if waypoint2 not in self.visible_waypoints:
                    self.visible_waypoints[waypoint2] = set()
                self.visible_waypoints[waypoint2].add(waypoint1)
        self.can_traverse = {}
        for fact in self.static_facts:
            if get_predicate_name(fact) == 'can_traverse':
                rover, wp1, wp2 = get_objects_from_fact(fact)
                if rover not in self.can_traverse:
                    self.can_traverse[rover] = {}
                if wp1 not in self.can_traverse[rover]:
                    self.can_traverse[rover][wp1] = set()
                self.can_traverse[rover][wp1].add(wp2)
                if wp2 not in self.can_traverse[rover]:
                    self.can_traverse[rover][wp2] = set()
                self.can_traverse[rover][wp2].add(wp1)


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        current_rover_locations = {}
        for fact in state:
            if get_predicate_name(fact) == 'at' and fact.startswith('('+'at rover'):
                rover, location = get_objects_from_fact(fact)
                current_rover_locations[rover] = location

        for goal in self.goals:
            if goal in state:
                continue

            predicate = get_predicate_name(goal)
            objects = get_objects_from_fact(goal)

            if predicate == 'communicated_soil_data':
                waypoint = objects[0]
                heuristic_value += 1 # communicate_soil_data
                if not any(match(fact, 'have_soil_analysis', '*', waypoint) for fact in state):
                    heuristic_value += 1 # sample_soil
                    rover = next((r for r in self.rovers_equipped_for_soil if r in current_rover_locations), None) # Assuming at least one rover is equipped
                    rover_location = current_rover_locations.get(rover)
                    if rover_location != waypoint:
                        heuristic_value += 1 # navigate

            elif predicate == 'communicated_rock_data':
                waypoint = objects[0]
                heuristic_value += 1 # communicate_rock_data
                if not any(match(fact, 'have_rock_analysis', '*', waypoint) for fact in state):
                    heuristic_value += 1 # sample_rock
                    rover = next((r for r in self.rovers_equipped_for_rock if r in current_rover_locations), None) # Assuming at least one rover is equipped
                    rover_location = current_rover_locations.get(rover)
                    if rover_location != waypoint:
                        heuristic_value += 1 # navigate

            elif predicate == 'communicated_image_data':
                objective, mode = objects
                heuristic_value += 1 # communicate_image_data
                if not any(match(fact, 'have_image', '*', objective, mode) for fact in state):
                    heuristic_value += 1 # take_image
                    rover_imaging = next((r for r in self.rovers_equipped_for_imaging if r in current_rover_locations), None) # Assuming at least one rover is equipped
                    camera = self.cameras_on_rovers.get(rover_imaging)
                    if not any(match(fact, 'calibrated', camera, rover_imaging) for fact in state):
                        heuristic_value += 1 # calibrate
                        visible_waypoints_obj = self.visible_from_objective.get(objective, [])
                        if visible_waypoints_obj:
                            waypoint_for_calibration = next(iter(visible_waypoints_obj)) # Take any visible waypoint
                            rover_location = current_rover_locations.get(rover_imaging)
                            if rover_location != waypoint_for_calibration:
                                heuristic_value += 1 # navigate for calibration
                    else: # already calibrated, just need to be at a visible waypoint for taking image
                        visible_waypoints_obj = self.visible_from_objective.get(objective, [])
                        if visible_waypoints_obj:
                            waypoint_for_image = next(iter(visible_waypoints_obj)) # Take any visible waypoint
                            rover_location = current_rover_locations.get(rover_imaging)
                            if rover_location != waypoint_for_image:
                                heuristic_value += 1 # navigate for taking image

            # Common communication navigation cost: move to a waypoint visible from lander
            if predicate.startswith('communicated_'):
                rover_comm = next(iter(current_rover_locations)) # Assuming at least one rover exists
                rover_location = current_rover_locations.get(rover_comm)
                visible_to_lander_waypoints = self.visible_waypoints.get(self.lander_location, [])
                if visible_to_lander_waypoints:
                    waypoint_for_comm = next(iter(visible_to_lander_waypoints)) # Take any visible waypoint
                    if rover_location != waypoint_for_comm:
                        heuristic_value += 1 # navigate for communication


        return heuristic_value
